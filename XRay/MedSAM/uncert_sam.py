import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.ndimage import label, gaussian_filter
from skimage.transform import resize, rotate

from segment_anything import sam_model_registry

# --- PHYSICS-BASED X-RAY DOSE TTA ---
def apply_xray_dose_tta(image, dose_factor):
    if dose_factor is None:
        return image
        
    image_norm = image.astype(np.float32)
    if image_norm.max() > 1.0:
        image_norm /= 255.0

    img_dose_scaled = image_norm * dose_factor
    img_noisy = np.random.poisson(img_dose_scaled) / dose_factor
    
    image_aug = np.clip(img_noisy * 255.0, 0, 255).astype(np.float32)
    return image_aug

def plot_intensity_distributions(original_image, output_path):
    dose_factors = [None, 500.0, 100.0, 10.0]
    labels = ['Original', 'Dose: 500 (Low Noise)', 'Dose: 100 (High Noise)', 'Dose: 10 (Severe)']
    
    plt.figure(figsize=(10, 6))
    for dose, label_name in zip(dose_factors, labels):
        aug_img = apply_xray_dose_tta(original_image, dose)
        pixels = aug_img.flatten()
        pixels = pixels[pixels > 5] 
        sns.kdeplot(pixels, label=label_name, alpha=0.6, warn_singular=False)

    plt.title('Pixel Intensity Distributions at Different X-Ray Dose Levels')
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Density')
    plt.xlim(0, 255)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# --- DROPOUT INJECTION ---
def enable_dropout(model):
    """Enable dropout layers strictly during inference for MC Dropout"""
    dropout_found = False
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            dropout_found = True
    return dropout_found

def inject_dropout_into_decoder(module, p=0.1):
    """Recursively search for Linear layers and wrap them in Sequential with Dropout."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, nn.Sequential(child, nn.Dropout(p=p)))
        else:
            inject_dropout_into_decoder(child, p)

def get_dual_bboxes_from_mask(mask, bbox_shift=10):
    labeled_mask, num_features = label(mask > 0)
    if num_features == 0:
        H, W = mask.shape
        return np.array([[0, 0, W, H]])
        
    bboxes = []
    areas = [(i, np.sum(labeled_mask == i)) for i in range(1, num_features + 1)]
    areas.sort(key=lambda x: x[1], reverse=True)
    
    top_labels = [x[0] for x in areas[:2]]
    
    H, W = mask.shape
    for lbl in top_labels:
        y_indices, x_indices = np.where(labeled_mask == lbl)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        x_min = max(0, x_min - bbox_shift)
        x_max = min(W, x_max + bbox_shift)
        y_min = max(0, y_min - bbox_shift)
        y_max = min(H, y_max + bbox_shift)
        
        bboxes.append([x_min, y_min, x_max, y_max])
        
    return np.array(bboxes)

# --- EXACT MODEL ARCHITECTURE FROM TRAINING ---
class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box):
        B_in, N_boxes, _ = box.shape
        with torch.no_grad():
            image_embedding = self.image_encoder(image) 
            image_embedding = image_embedding.unsqueeze(1).repeat(1, N_boxes, 1, 1, 1)
            image_embedding = image_embedding.view(-1, 256, image_embedding.shape[3], image_embedding.shape[4])

            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            box_torch = box_torch.view(-1, 1, 4)  
            
            sparse_embeddings, dense_embeddings = self.prompt_encoder( points=None, boxes=box_torch, masks=None )
            
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  
            image_pe=self.prompt_encoder.get_dense_pe(),  
            sparse_prompt_embeddings=sparse_embeddings,  
            dense_prompt_embeddings=dense_embeddings,  
            multimask_output=False,
        )
        
        low_res_masks = low_res_masks.view(B_in, N_boxes, 1, low_res_masks.shape[-2], low_res_masks.shape[-1])
        low_res_masks, _ = torch.max(low_res_masks, dim=1) 
            
        ori_res_masks = F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]), 
            mode="bilinear", align_corners=False,
        )
        return ori_res_masks

def calculate_dice(pred, gt):
    pred_mask = (pred > 0).astype(np.float32)
    gt_mask = (gt > 0).astype(np.float32)
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    if union == 0: return 1.0
    return 2.0 * intersection / union

def save_visualization(img, gt, pred, bboxes_native, p_map, var_map, ent_map, output_path, dice_score):
    if img.max() > img.min():
        img_p1, img_p99 = np.percentile(img[img > 0], [1, 99]) if np.max(img)>0 else (0, 1)
        img_norm = np.clip((img - img_p1) / (img_p99 - img_p1 + 1e-10), 0, 1)
    else:
        img_norm = img

    gt_mask = (gt > 0).astype(np.uint8)
    pred_mask = (pred > 0).astype(np.uint8)

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    axes[0, 0].imshow(img_norm, cmap='gray')
    for bbox in bboxes_native:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                                 linewidth=2, edgecolor='yellow', facecolor='none')
        axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('Input Image + GT BBox Prompts', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_mask, cmap='tab10', vmin=0, vmax=10, interpolation='nearest')
    axes[0, 1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=10, interpolation='nearest')
    axes[0, 2].set_title(f'MedSAM Prediction (Dice: {dice_score:.3f})', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(img_norm, cmap='gray', alpha=0.7)
    overlay_gt = np.zeros((*gt_mask.shape, 3))
    overlay_gt[gt_mask == 1] = [0, 1, 0] 
    axes[1, 0].imshow(overlay_gt, alpha=0.5)
    axes[1, 0].set_title('Ground Truth Overlay', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img_norm, cmap='gray', alpha=0.7)
    overlay_pred = np.zeros((*pred_mask.shape, 3))
    overlay_pred[pred_mask == 1] = [1, 0, 0] 
    axes[1, 1].imshow(overlay_pred, alpha=0.5)
    axes[1, 1].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off') 

    axes[1, 2].imshow(img_norm, cmap='gray', alpha=0.7)
    if gt_mask.max() > 0: axes[1, 2].contour(gt_mask, colors='green', linewidths=2, alpha=0.8, linestyles='solid')
    if pred_mask.max() > 0: axes[1, 2].contour(pred_mask, colors='red', linewidths=2, alpha=0.8, linestyles='dashed')
    axes[1, 2].set_title('Contours: GT (Solid), Pred (Dashed)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    axes[2, 0].imshow(img_norm, cmap='gray', alpha=0.5)
    im1 = axes[2, 0].imshow(p_map, cmap='inferno', alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(im1, ax=axes[2, 0], fraction=0.046, pad=0.04)
    axes[2, 0].set_title('Mean Predict Probability', fontsize=14, fontweight='bold')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(img_norm, cmap='gray', alpha=0.5)
    vmax_var = max(np.max(var_map), 1e-4) 
    im2 = axes[2, 1].imshow(var_map, cmap='inferno', alpha=0.7, vmin=0, vmax=vmax_var)
    plt.colorbar(im2, ax=axes[2, 1], fraction=0.046, pad=0.04)
    axes[2, 1].set_title('Segmentation Disagreement (Variance)', fontsize=14, fontweight='bold')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(img_norm, cmap='gray', alpha=0.5)
    vmax_ent = max(np.max(ent_map), 1e-4) 
    im3 = axes[2, 2].imshow(ent_map, cmap='inferno', alpha=0.7, vmin=0, vmax=vmax_ent)
    plt.colorbar(im3, ax=axes[2, 2], fraction=0.046, pad=0.04)
    axes[2, 2].set_title('Total Predictive Entropy', fontsize=14, fontweight='bold')
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def save_summary_grid(summary_data, output_path):
    """
    Creates a large summary plot where each row is an augmentation method, 
    and columns are [Input View, GT Mask, Pred Mask, Overlay, Entropy Map].
    """
    methods = list(summary_data.keys())
    n_rows = len(methods)
    n_cols = 5
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
    col_titles = ['Augmented Input', 'GT Mask', 'Prediction', 'GT & Pred Overlap', 'Total Entropy']

    for r, method in enumerate(methods):
        data = summary_data[method]
        img = data['image']
        gt  = data['gt']
        pred = data['pred']
        ent = data['ent']
        dice = data['dice']
        
        if img.max() > img.min():
            img_p1, img_p99 = np.percentile(img[img > 0], [1, 99]) if np.max(img)>0 else (0, 1)
            img_norm = np.clip((img - img_p1) / (img_p99 - img_p1 + 1e-10), 0, 1)
        else:
            img_norm = img

        gt_mask = (gt > 0).astype(np.uint8)
        pred_mask = (pred > 0).astype(np.uint8)

        # Col 0: Input Form
        ax = axes[r, 0]
        ax.imshow(img_norm, cmap='gray')
        ax.set_title(f"{method.upper()}\nDice = {dice:.4f}", fontsize=15, fontweight='bold')
        ax.axis('off')

        # Col 1: GT Mask
        ax = axes[r, 1]
        ax.imshow(gt_mask, cmap='tab10', vmin=0, vmax=10, interpolation='nearest')
        if r == 0: ax.set_title(col_titles[1], fontsize=16, fontweight='bold')
        ax.axis('off')

        # Col 2: Prediction
        ax = axes[r, 2]
        ax.imshow(pred_mask, cmap='tab10', vmin=0, vmax=10, interpolation='nearest')
        if r == 0: ax.set_title(col_titles[2], fontsize=16, fontweight='bold')
        ax.axis('off')

        # Col 3: Contours Overlap
        ax = axes[r, 3]
        ax.imshow(img_norm, cmap='gray', alpha=0.7)
        if gt_mask.max() > 0:
            ax.contour(gt_mask, colors='green', linewidths=2, alpha=0.8, linestyles='solid')
        if pred_mask.max() > 0:
            ax.contour(pred_mask, colors='red', linewidths=2, alpha=0.8, linestyles='dashed')
        if r == 0: ax.set_title(col_titles[3], fontsize=16, fontweight='bold')
        ax.axis('off')

        # Col 4: Predictive Entropy
        ax = axes[r, 4]
        ax.imshow(img_norm, cmap='gray', alpha=0.5)
        vmax_ent = max(np.max(ent), 1e-4)
        im = ax.imshow(ent, cmap='inferno', alpha=0.7, vmin=0, vmax=vmax_ent)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if r == 0: ax.set_title(col_titles[4], fontsize=16, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MedSAM Inference Compares MC Dropout vs Multiple TTA Methods")
    parser.add_argument('-i', '--image_dir', required=True, help="Directory with input images")
    parser.add_argument('-g', '--mask_dir', required=True, help="Directory with ground truth masks")
    parser.add_argument('-o', '--output_dir', required=True, help="Output directory")
    
    parser.add_argument('-b', '--base_checkpoint', default='work_dir/MedSAM/medsam_vit_b.pth')
    parser.add_argument('-m', '--model_path', required=True, help="Finetuned checkpoint (medsam_model_best.pth)")
    
    parser.add_argument('--device', default='cuda:0', help="Device")
    parser.add_argument('-n', '--n_mc_samples', type=int, default=20, help="Number of MC samples")
    parser.add_argument('--dose_factor', type=float, default=200.0, help="Base strict dose limit for Physics TTA")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Base Architectures...")
    sam_model = sam_model_registry['vit_b'](checkpoint=args.base_checkpoint)
    medsam_model = MedSAM(sam_model.image_encoder, sam_model.mask_decoder, sam_model.prompt_encoder)
    
    print("Aligning Dropout architecture into strictly the Mask Decoder...")
    inject_dropout_into_decoder(medsam_model.mask_decoder, p=0.2)

    print(f"Loading finetuned weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    medsam_model.load_state_dict(state_dict, strict=True)
    medsam_model = medsam_model.to(args.device)

    img_files = sorted([f for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))])
    print(f"Found {len(img_files)} images to process.\n{'='*60}")

    # --- ADDED 'baseline' AS THE FIRST METHOD ---
    methods = ['baseline', 'mc_dropout', 'physics_tta', 'gaussian_noise', 'blur', 'jitter', 'rotation']

    i = 0
    for idx, filename in enumerate(img_files):
        if i >= 10:
            print(f"Limiting to 10 cases for quick testing. Remove this condition to process all cases.")
            break
        print(f"\n[{idx+1}/{len(img_files)}] Processing {filename}...")
        
        img_name_no_ext = filename.rsplit('.', 1)[0]
        img_path = os.path.join(args.image_dir, filename)
        mask_path = os.path.join(args.mask_dir, filename)
        if not os.path.exists(mask_path): continue
            
        try:
            img_sitk = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img_sitk)
            if len(img_np.shape) == 3 and img_np.shape[0] == 1: img_np = img_np[0] 
                
            mask_sitk = sitk.ReadImage(mask_path)
            mask_np = sitk.GetArrayFromImage(mask_sitk)
            if len(mask_np.shape) == 3 and mask_np.shape[0] == 1: mask_np = mask_np[0]

            img_base = np.clip(img_np, np.percentile(img_np, 1), np.percentile(img_np, 99))
            img_base = (img_base - img_base.min()) / (img_base.max() - img_base.min() + 1e-8)
            img_base = (img_base * 255).astype(np.float32)

            mask_1024_base = resize(mask_np, (1024, 1024), order=0, preserve_range=True, anti_aliasing=False)
            bboxes_1024_base = get_dual_bboxes_from_mask((mask_1024_base > 0).astype(np.uint8), bbox_shift=0)
            boxes_tensor_base = torch.tensor(bboxes_1024_base).float().unsqueeze(0).to(args.device) 
            
            # Save Base Distributions once per image
            base_img_dir = os.path.join(args.output_dir, img_name_no_ext)
            os.makedirs(base_img_dir, exist_ok=True)
            plot_intensity_distributions(img_base, os.path.join(base_img_dir, f"physics_intensities_dist.png"))

            # Initialize dict to store results for mass-plotting
            summary_data = {}

            for method in methods:
                print(f"  -> Evaluating via: {method.upper()}")
                
                # Setup Folders Named According to Image -> Method
                method_dir = os.path.join(base_img_dir, method)
                viz_dir = os.path.join(method_dir, 'visualizations')
                pred_dir = os.path.join(method_dir, 'predictions')
                uncert_dir = os.path.join(method_dir, 'uncertainty')
                
                os.makedirs(viz_dir, exist_ok=True)
                os.makedirs(pred_dir, exist_ok=True)
                os.makedirs(uncert_dir, exist_ok=True)

                all_mc_preds = []
                
                if method == 'mc_dropout':
                    medsam_model.eval() 
                    enable_dropout(medsam_model) # Turn on nodes randomly
                else:
                    medsam_model.eval() # Strictly evaluative for TTA math and baseline
                
                with torch.no_grad():
                    for mc_iter in range(args.n_mc_samples):
                        noisy_img = img_base.copy()
                        angle = 0
                        boxes_tensor = boxes_tensor_base
                        
                        # --- APPLY SELECTED AUGMENTATION ---
                        if method == 'baseline':
                            pass # Keep the pure unmodified image
                        elif method == 'physics_tta':
                            noisy_img = apply_xray_dose_tta(img_base, args.dose_factor)
                        elif method == 'gaussian_noise':
                            noisy_img = np.clip(img_base + np.random.normal(0, 15, img_base.shape), 0, 255)
                        elif method == 'blur':
                            noisy_img = gaussian_filter(img_base, sigma=np.random.uniform(1.0, 5.0))
                        elif method == 'jitter':
                            alpha = np.random.uniform(1.5, 2.0)
                            beta = np.random.uniform(-20, 20)
                            noisy_img = np.clip(img_base * alpha + beta, 0, 255)
                        elif method == 'rotation':
                            angle = np.random.uniform(-90, 90)
                            noisy_img = rotate(img_base, angle, preserve_range=True, mode='reflect')
                            # Rotate mask to recalculate accurate testing bounding box!
                            mask_rot = rotate(mask_np, angle, order=0, preserve_range=True)
                            mask_1024_rot = resize(mask_rot, (1024, 1024), order=0, preserve_range=True, anti_aliasing=False)
                            bboxes_1024_rot = get_dual_bboxes_from_mask((mask_1024_rot > 0).astype(np.uint8), bbox_shift=0)
                            boxes_tensor = torch.tensor(bboxes_1024_rot).float().unsqueeze(0).to(args.device)

                        # Standard Tensor Conversions (User: keep un-normalised!)
                        if len(noisy_img.shape) == 2: img_3c = np.repeat(noisy_img[:, :, None], 3, axis=-1)
                        else: img_3c = noisy_img

                        img_1024 = resize(img_3c, (1024, 1024), order=1, preserve_range=True, anti_aliasing=True)
                        img_1024_tensor = np.transpose(img_1024, (2, 0, 1))
                        image_tensor = torch.tensor(img_1024_tensor).float().unsqueeze(0).to(args.device) 
                        
                        # Generate Prediction
                        logits = medsam_model(image_tensor, boxes_tensor)
                        prob_1024 = torch.sigmoid(logits).squeeze().cpu().numpy()
                        prob_native = resize(prob_1024, mask_np.shape, order=1, preserve_range=True, anti_aliasing=False)
                        
                        # Reverse Rotation Spatial Realignment so variances perfectly overlap
                        if method == 'rotation':
                            prob_native = rotate(prob_native, -angle, preserve_range=True, mode='reflect')
                            
                        all_mc_preds.append(prob_native)
                
                # --- Compile Variances ---
                all_mc_preds = np.stack(all_mc_preds, axis=0) # [N_samples, H, W]
                
                p_map = np.mean(all_mc_preds, axis=0) 
                var_map = np.var(all_mc_preds, axis=0)
                
                p_safe = np.clip(p_map, 1e-5, 1.0 - 1e-5)
                ent_map = - (p_safe * np.log(p_safe) + (1.0 - p_safe) * np.log(1.0 - p_safe))
                
                final_medsam_pred = (p_map > 0.5).astype(np.uint8)
                bboxes_native = get_dual_bboxes_from_mask((mask_np > 0).astype(np.uint8), bbox_shift=10)
                dice = calculate_dice(final_medsam_pred, mask_np)
                print(f"     -> Box(es): {len(bboxes_native)} | Dice: {dice:.4f}")
                
                # --- Saving Individual Files ---
                pred_sitk = sitk.GetImageFromArray(final_medsam_pred)
                pred_sitk.CopyInformation(img_sitk)
                sitk.WriteImage(pred_sitk, os.path.join(pred_dir, f"{img_name_no_ext}_{method}.nii.gz"))
                
                np.save(os.path.join(uncert_dir, f"{img_name_no_ext}_{method}_var.npy"), var_map)
                np.save(os.path.join(uncert_dir, f"{img_name_no_ext}_{method}_ent.npy"), ent_map)
                
                viz_name = f"{img_name_no_ext}_{method}.png"

                # Define the image representation for plotting purposes
                viz_target_img = img_base if method in ['mc_dropout', 'rotation', 'baseline'] else noisy_img
                
                # Cache results for massive plot
                summary_data[method] = {
                    'image': viz_target_img.copy(),
                    'gt': mask_np.copy(),
                    'pred': final_medsam_pred.copy(),
                    'ent': ent_map.copy(),
                    'dice': dice
                }

                save_visualization(viz_target_img, mask_np, final_medsam_pred, bboxes_native, p_map, var_map, ent_map, os.path.join(viz_dir, viz_name), dice)
            
            # --- Generate Mass Figure Output ---
            print("  -> Generating Method Comparison Summary Plot...")
            save_summary_grid(summary_data, os.path.join(args.output_dir, f"{img_name_no_ext}_FULL_SUMMARY.png"))

            i += 1
        except Exception as e:
            print(f"  [!] Failed to process {filename}: {e}")
        
if __name__ == "__main__":
    main()