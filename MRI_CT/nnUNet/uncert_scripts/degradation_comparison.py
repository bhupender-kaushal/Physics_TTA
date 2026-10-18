import torch
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from scipy.ndimage import shift, gaussian_filter, rotate, affine_transform, map_coordinates
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# --- 1. EVALUATION & METRICS LOGIC ---
def calculate_dice(pred, gt, class_idx=1):
    """Calculates Dice score for a specific class."""
    p = (pred == class_idx).astype(np.float32)
    g = (gt == class_idx).astype(np.float32)
    intersection = np.sum(p * g)
    union = np.sum(p) + np.sum(g)
    if union == 0: return 1.0 if np.sum(p) == 0 else 0.0
    return 2. * intersection / union

# --- 2. FORWARD SPATIAL TRANSFORM LOGIC ---
def apply_spatial_mask_3d(mask, params):
    """
    Applies the exact same spatial transform to the discrete Ground Truth mask.
    Uses strict nearest neighbor (order=0) to prevent fractional class labels.
    """
    aug_type = params['aug_type']
    perturbed_mask = np.copy(mask).astype(np.float32)

    # Pure intensity changes and zero-mean breathing ghosting do not move the primary anatomy
    if aug_type in ['noise', 'breathing', 'tta_brightness', 'tta_contrast', 'tta_gamma']:
        return mask.astype(np.uint8)

    # K-Space linear phase ramps dictate a rigid spatial shift
    elif aug_type in ['motion', 'motion_and_breathing']:
        shift_vals = (params['dy'], params['dx'])
        for z in range(mask.shape[2]):
            perturbed_mask[:, :, z] = shift(mask[:, :, z], shift=shift_vals, order=0, mode='constant', cval=0.0)

    elif aug_type == 'tta_flip':
        axis = params['flip_axis']
        for z in range(mask.shape[2]):
            perturbed_mask[:, :, z] = np.flip(mask[:, :, z], axis=axis)

    elif aug_type == 'tta_rotate':
        ang = params['angle']
        for z in range(mask.shape[2]):
            perturbed_mask[:, :, z] = rotate(mask[:, :, z], ang, reshape=False, order=0, mode='constant', cval=0.0)

    elif aug_type == 'tta_scale':
        factor = params['scale_factor']
        H, W = mask.shape[0], mask.shape[1]
        center = np.array([H, W]) / 2.0
        transform = np.array([[1.0/factor, 0], [0, 1.0/factor]])
        offset = center - np.dot(transform, center)
        for z in range(mask.shape[2]):
            perturbed_mask[:, :, z] = affine_transform(mask[:, :, z], transform, offset=offset, order=0, mode='constant', cval=0.0)

    elif aug_type == 'tta_translation':
        shift_vals = (params['trans_dy'], params['trans_dx'])
        for z in range(mask.shape[2]):
            perturbed_mask[:, :, z] = shift(mask[:, :, z], shift_vals, order=0, mode='constant', cval=0.0)

    elif aug_type == 'tta_elastic':
        H, W = mask.shape[0], mask.shape[1]
        for z in range(mask.shape[2]):
            if z < len(params['elastic_fields']):
                dx, dy = params['elastic_fields'][z]
                x, y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                indices = np.array([x + dx, y + dy])
                perturbed_mask[:, :, z] = map_coordinates(mask[:, :, z], indices, order=0, mode='constant', cval=0.0)

    return perturbed_mask.astype(np.uint8)

def apply_synchronized_augmentation_3d(image_array, params=None):
    if params is None:
        aug_type = np.random.choice(['motion', 'noise', 'breathing', 'motion_and_breathing'])
        params = {'aug_type': aug_type}

    aug_type = params['aug_type']
    perturbed_vol = np.zeros_like(image_array, dtype=np.float32)

    if aug_type in ['motion', 'noise', 'breathing', 'motion_and_breathing']:
        dy, dx = params['dy'], params['dx']
        noise_rel_std = params['noise_std']
        
        for z in range(image_array.shape[2]):
            slice_2d = image_array[:, :, z]
            H, W = slice_2d.shape
            
            k_space = np.fft.fftshift(np.fft.fft2(slice_2d))
            u = np.linspace(-0.5, 0.5, H, endpoint=False).reshape(-1, 1)
            v = np.linspace(-0.5, 0.5, W, endpoint=False).reshape(1, -1)
                
            if aug_type == 'motion': phase_ramp = -2 * np.pi * (u * dy + v * dx)
            elif aug_type == 'noise': phase_ramp = -2 * np.pi * (u * v)
            elif aug_type == 'breathing': phase_ramp = -2 * np.pi * (np.sin(u * dy + v * dx))
            elif aug_type == 'motion_and_breathing': phase_ramp = -2 * np.pi * (u * dy + v * dx + np.sin(u * dy + v * dx))
            else: phase_ramp = 0
                
            k_space_shifted = k_space * np.exp(1j * phase_ramp)
            slice_std = np.std(slice_2d) if np.std(slice_2d) > 0 else 1.0
            actual_noise_std = noise_rel_std * slice_std * np.sqrt(H * W)
            
            complex_noise = (np.random.normal(0, actual_noise_std / np.sqrt(2), (H, W)) + 
                             1j * np.random.normal(0, actual_noise_std / np.sqrt(2), (H, W)))
            
            aug_slice = np.abs(np.fft.ifft2(np.fft.ifftshift(k_space_shifted + complex_noise)))
            perturbed_vol[:, :, z] = aug_slice.astype(np.float32)

    elif aug_type == 'tta_flip':
        axis = params['flip_axis']
        for z in range(image_array.shape[2]):
            perturbed_vol[:, :, z] = np.flip(image_array[:, :, z], axis=axis)
            
    elif aug_type == 'tta_rotate':
        ang = params['angle']
        for z in range(image_array.shape[2]):
            perturbed_vol[:, :, z] = rotate(image_array[:, :, z], ang, reshape=False, order=1, mode='nearest')
            
    elif aug_type == 'tta_scale':
        factor = params['scale_factor']
        H, W = image_array.shape[0], image_array.shape[1]
        center = np.array([H, W]) / 2.0
        transform = np.array([[1.0/factor, 0], [0, 1.0/factor]])
        offset = center - np.dot(transform, center)
        for z in range(image_array.shape[2]):
            perturbed_vol[:, :, z] = affine_transform(image_array[:, :, z], transform, offset=offset, order=1, mode='nearest')

    elif aug_type == 'tta_translation':
        for z in range(image_array.shape[2]):
            perturbed_vol[:, :, z] = shift(image_array[:, :, z], (params['trans_dy'], params['trans_dx']), order=1, mode='nearest')

    elif aug_type == 'tta_brightness':
        for z in range(image_array.shape[2]):
            perturbed_vol[:, :, z] = image_array[:, :, z] + params['bright_add']

    elif aug_type == 'tta_contrast':
        for z in range(image_array.shape[2]):
            perturbed_vol[:, :, z] = image_array[:, :, z] * params['cont_mul']

    elif aug_type == 'tta_gamma':
        val = params['gamma_pow']
        img_min, img_max = image_array.min(), image_array.max()
        if img_max > img_min:
            norm_vol = (image_array - img_min) / (img_max - img_min)
            perturbed_vol = (norm_vol ** val) * (img_max - img_min) + img_min
        else:
            perturbed_vol = image_array.copy()

    elif aug_type == 'tta_elastic':
        alpha, sigma = params['elastic_alpha'], params['elastic_sigma']
        H, W = image_array.shape[0], image_array.shape[1]
        if 'elastic_fields' not in params:
            params['elastic_fields'] = []
            for _ in range(image_array.shape[2]):
                dx = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma) * alpha
                dy = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma) * alpha
                params['elastic_fields'].append((dx, dy))
        
        for z in range(image_array.shape[2]):
            dx, dy = params['elastic_fields'][z]
            x, y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
            perturbed_vol[:, :, z] = map_coordinates(image_array[:, :, z], indices, order=1, mode='nearest').reshape((H, W))

    return perturbed_vol, params

# --- 3. VISUALIZATION LOGIC ---
def plot_supergraph(t2_vol, gt_vol, base_pred, base_ent, pert_vol, pert_gt_vol, pert_base_pred, pert_base_ent, 
                    pert_cons_pred, pert_cons_ent, pert_all_masks, best_slices, case_id, output_path, aug_type, 
                    base_all_masks=None, pert_params=None):
    n_rows = len(best_slices)
    fig, axes = plt.subplots(n_rows, 11, figsize=(55, 5 * n_rows))
    if n_rows == 1: axes = [axes]
    
    fig.suptitle(f"[{case_id}] | Perturbation: {aug_type.replace('_', ' ').title()}", fontsize=28, weight='bold', y=1.02)
    contour_colors = ['red', 'cyan', 'lime', 'magenta', 'yellow', 'orange', 'white', 'pink', 'blue', 'green']

    for i, z in enumerate(best_slices):
        img_slice = np.rot90(t2_vol[:, :, z])
        gt_slice = np.rot90(gt_vol[:, :, z])
        b_pred_slice = np.rot90(base_pred[:, :, z])
        b_ent_slice = np.rot90(base_ent[:, :, z])
        
        pert_img_slice = np.rot90(pert_vol[:, :, z])
        pert_gt_slice = np.rot90(pert_gt_vol[:, :, z]) if pert_gt_vol is not None else None
        
        p_base_pred_slice = np.rot90(pert_base_pred[:, :, z])
        p_base_ent_slice = np.rot90(pert_base_ent[:, :, z])
        p_cons_pred_slice = np.rot90(pert_cons_pred[:, :, z])
        p_cons_ent_slice = np.rot90(pert_cons_ent[:, :, z])

        # Cols 0-2 (Unperturbed Reference)
        axes[i][0].imshow(img_slice, cmap='gray')
        axes[i][0].imshow(np.ma.masked_where(gt_slice == 0, gt_slice), cmap='tab10', alpha=0.5)
        axes[i][0].set_title(f"Orig GT (Slice {z})", fontsize=16)
        axes[i][0].axis('off')
        
        axes[i][1].imshow(img_slice, cmap='gray')
        axes[i][1].imshow(np.ma.masked_where(b_pred_slice == 0, b_pred_slice), cmap='tab10', alpha=0.5)
        axes[i][1].set_title(f"Orig Base Pred", fontsize=16)
        axes[i][1].axis('off')

        axes[i][2].imshow(img_slice, cmap='gray', alpha=0.5)
        im = axes[i][2].imshow(b_ent_slice, cmap='inferno', alpha=0.8, vmin=0)
        axes[i][2].set_title("Orig Base Entropy", fontsize=16)
        axes[i][2].axis('off')

        # Cols 3-8 (Perturbed Space Evaluation)
        axes[i][3].imshow(pert_img_slice, cmap='gray')
        if pert_gt_slice is not None:
            axes[i][3].imshow(np.ma.masked_where(pert_gt_slice == 0, pert_gt_slice), cmap='tab10', alpha=0.5)
        axes[i][3].set_title(f"Perturbed GT", fontsize=16)
        axes[i][3].axis('off')
        
        axes[i][4].imshow(pert_img_slice, cmap='gray')
        axes[i][4].imshow(np.ma.masked_where(p_base_pred_slice == 0, p_base_pred_slice), cmap='tab10', alpha=0.5)
        axes[i][4].set_title(f"Pert Pass 1 Pred", fontsize=16)
        axes[i][4].axis('off')
        
        axes[i][5].imshow(pert_img_slice, cmap='gray', alpha=0.5)
        im_pb = axes[i][5].imshow(p_base_ent_slice, cmap='inferno', alpha=0.8, vmin=0)
        axes[i][5].set_title("Pert Pass 1 Entropy", fontsize=16)
        axes[i][5].axis('off')

        axes[i][6].imshow(pert_img_slice, cmap='gray')
        for pass_idx in range(pert_all_masks.shape[0]):
            pass_mask = np.rot90(pert_all_masks[pass_idx, :, :, z])
            if np.any(pass_mask == 1):
                lw = 2.5 if pass_idx == 0 else 1.0
                alpha = 0.9 if pass_idx == 0 else 0.5
                c_idx = pass_idx % len(contour_colors)
                axes[i][6].contour(pass_mask == 1, levels=[0.5], colors=[contour_colors[c_idx]], alpha=alpha, linewidths=lw)
        axes[i][6].set_title("Multi-Pass Disagreement", fontsize=16)
        axes[i][6].axis('off')

        axes[i][7].imshow(pert_img_slice, cmap='gray')
        axes[i][7].imshow(np.ma.masked_where(p_cons_pred_slice == 0, p_cons_pred_slice), cmap='tab10', alpha=0.5)
        axes[i][7].set_title(f"Pert Cons Pred", fontsize=16)
        axes[i][7].axis('off')

        axes[i][8].imshow(pert_img_slice, cmap='gray', alpha=0.5)
        im2 = axes[i][8].imshow(p_cons_ent_slice, cmap='inferno', alpha=0.8, vmin=0)
        axes[i][8].set_title("Pert Cons Entropy", fontsize=16)
        axes[i][8].axis('off')

        if base_all_masks is not None and pert_all_masks is not None and pert_gt_slice is not None:
            base_slice_dices = [calculate_dice(m[:, :, z], np.rot90(gt_slice, -1), class_idx=1) for m in base_all_masks]
            # Calculating Perturbed Run variances natively in the augmented geometry
            pert_slice_dices = [calculate_dice(m[:, :, z], np.rot90(pert_gt_slice, -1), class_idx=1) for m in pert_all_masks]
            
            box = axes[i][9].boxplot([base_slice_dices, pert_slice_dices], labels=['Base N', 'Pert N'], patch_artist=True)
            axes[i][9].scatter([1]*len(base_slice_dices), base_slice_dices, color='darkblue', alpha=0.6, zorder=3)
            axes[i][9].scatter([2]*len(pert_slice_dices[1:]), pert_slice_dices[1:], color='darkred', alpha=0.4, zorder=3)
            axes[i][9].scatter(2, pert_slice_dices[0], color='gold', edgecolor='black', s=100, zorder=4, label="Pert Pass 1")
            
            for patch, color in zip(box['boxes'], ['lightblue', 'lightcoral']): patch.set_facecolor(color)
            axes[i][9].set_title(f"Slice {z} Run Variances", fontsize=16)
            axes[i][9].set_ylim(.40, 1.0)
            axes[i][9].grid(axis='y', linestyle='--', alpha=0.7)

            if pert_params is not None:
                if aug_type in ['motion', 'motion_and_breathing']: x_vals, x_label = [np.sqrt(p['dy']**2 + p['dx']**2) for p in pert_params], 'Dist (dx+dy)'
                elif aug_type == 'noise': x_vals, x_label = [p['noise_std']*100 for p in pert_params], 'Noise %'
                elif aug_type == 'tta_rotate': x_vals, x_label = [p['angle'] for p in pert_params], 'Angle'
                elif aug_type == 'tta_translation': x_vals, x_label = [np.sqrt(p['trans_dy']**2 + p['trans_dx']**2) for p in pert_params], 'Dist Shift'
                else: x_vals, x_label = [1 for _ in pert_params], 'Fixed'

                num_p = len(pert_slice_dices)
                cmap = plt.cm.viridis
                for p_idx in range(num_p):
                    c = cmap(p_idx / max(1, num_p - 1))
                    s = 150 if p_idx == 0 else 70
                    ec = 'black' if p_idx == 0 else 'none'
                    lw = 2 if p_idx == 0 else 0
                    axes[i][10].scatter(x_vals[p_idx], pert_slice_dices[p_idx], color=c, marker='o', s=s, edgecolor=ec, linewidth=lw, alpha=0.8)
                
                axes[i][10].set_title("Perturbation vs Dice", fontsize=16)
                axes[i][10].set_ylabel("Dice Score")
                axes[i][10].grid(linestyle='--', alpha=0.5)
        else:
            axes[i][9].axis('off'); axes[i][10].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# --- 4. MAIN EXECUTION ---
def main():
    # Initialize Predictors for BOTH models
    predictor_base = nnUNetPredictor(tile_step_size=1.0, use_mirroring=False, device=torch.device('cuda'), verbose=False)
    predictor_pre = nnUNetPredictor(tile_step_size=1.0, use_mirroring=False, device=torch.device('cuda'), verbose=False)
    
    # Adjust paths accordingly
    img_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/prostate/images'
    gt_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/prostate/labels'

    output_base = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/uncert_analysis/mc_results/degradation_comparison'
    
    model_folder_baseline = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/nnUNet_results/Dataset005_Prostate/nnUNetTrainer__nnUNetPlans__2d'
    model_folder_pretrained = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/nnUNet_results/Dataset005_Prostate/nnUNetTrainerDropoutPhysics__nnUNetPlans__2d'

    print("Loading Baseline Model...")
    predictor_base.initialize_from_trained_model_folder(model_folder_baseline, use_folds=(0,), checkpoint_name='checkpoint_final.pth')
    
    print("Loading Pretrained Model...")
    predictor_pre.initialize_from_trained_model_folder(model_folder_pretrained, use_folds=(0,), checkpoint_name='checkpoint_final.pth')

    all_files = [f for f in os.listdir(img_ts_folder) if f.endswith('.nii.gz')]
    case_ids = sorted(list(set([f.rsplit('_', 1)[0] for f in all_files])))
    
    # Target configurations
    target_augs = ['breathing', 'motion_and_breathing']
    intensity_levels = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0]  # The values to scale dx/dy by

    for case_id in case_ids:
        print(f"\n[{case_id}] Loading Data...")
        
        case_output_dir = os.path.join(output_base, case_id)
        perturbed_dir = os.path.join(case_output_dir, 'perturbed_images')
        os.makedirs(perturbed_dir, exist_ok=True)
        
        modality_files = sorted([os.path.join(img_ts_folder, f) for f in all_files if f.startswith(f"{case_id}_")])
        base_arrays = []
        affine, header = None, None
        for mod in modality_files:
            nii = nib.load(mod)
            base_arrays.append(nii.get_fdata())
            if affine is None: affine, header = nii.affine, nii.header

        gt_path = os.path.join(gt_ts_folder, f"{case_id}.nii.gz")
        has_gt = os.path.exists(gt_path)
        gt_array = nib.load(gt_path).get_fdata() if has_gt else None

        if not has_gt:
            print(f"  -> Skipping {case_id}, no GT found.")
            continue

        degradation_log_path = os.path.join(case_output_dir, f"{case_id}_degradation_log.csv")
        
        with open(degradation_log_path, "w") as log_f:
            log_f.write("Case,Augmentation,Intensity,Baseline_Dice,Pretrained_Dice\n")
            
            for aug_type in target_augs:
                print(f"\n  === Testing Degradation: {aug_type.upper()} ===")
                
                for intensity in intensity_levels:
                    # Dynamically set parameters based on the current intensity
                    sync_params = {
                        'aug_type': aug_type, 
                        'dy': intensity, 
                        'dx': intensity, 
                        'noise_std': 0.0  # Not testing noise here
                    }

                    temp_nii_paths = []
                    for c_idx, channel_array in enumerate(base_arrays):
                        perturbed_array, sync_params = apply_synchronized_augmentation_3d(channel_array, sync_params)
                        temp_path = os.path.join(perturbed_dir, f"{case_id}_{aug_type}_int{intensity}_000{c_idx}.nii.gz")
                        nib.save(nib.Nifti1Image(perturbed_array.astype(np.float32), affine, header), temp_path)
                        temp_nii_paths.append(temp_path)
                        
                    # 1. Forward transform GT natively
                    perturbed_gt = apply_spatial_mask_3d(gt_array, sync_params)

                    # 2. Evaluate Baseline Model
                    probs_base = predictor_base.predict_from_files([temp_nii_paths], None, save_probabilities=True, overwrite=True,
                                                                   num_processes_preprocessing=1, num_processes_segmentation_export=1)
                    raw_prob_base = probs_base[0][1].transpose(0, 3, 2, 1)
                    mask_base = np.argmax(raw_prob_base, axis=0)
                    
                    # 3. Evaluate Pretrained Model
                    probs_pre = predictor_pre.predict_from_files([temp_nii_paths], None, save_probabilities=True, overwrite=True,
                                                                 num_processes_preprocessing=1, num_processes_segmentation_export=1)
                    raw_prob_pre = probs_pre[0][1].transpose(0, 3, 2, 1)
                    mask_pre = np.argmax(raw_prob_pre, axis=0)

                    # 4. Calculate native Dice scores on the overlapping slices
                    slice_dices_base = [calculate_dice(mask_base[:, :, z], perturbed_gt[:, :, z], 1) for z in range(perturbed_gt.shape[2]) if np.sum(perturbed_gt[:, :, z] == 1) > 0]
                    slice_dices_pre = [calculate_dice(mask_pre[:, :, z], perturbed_gt[:, :, z], 1) for z in range(perturbed_gt.shape[2]) if np.sum(perturbed_gt[:, :, z] == 1) > 0]
                    
                    mean_dice_base = np.mean(slice_dices_base) if slice_dices_base else 0.0
                    mean_dice_pre = np.mean(slice_dices_pre) if slice_dices_pre else 0.0

                    print(f"    -> Intensity {intensity:5.1f} | Base Dice: {mean_dice_base:.4f} | Pretrained Dice: {mean_dice_pre:.4f}")
                    log_f.write(f"{case_id},{aug_type},{intensity},{mean_dice_base:.4f},{mean_dice_pre:.4f}\n")

        break
if __name__ == '__main__':
    main()