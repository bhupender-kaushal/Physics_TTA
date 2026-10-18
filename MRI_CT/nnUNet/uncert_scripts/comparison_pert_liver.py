import torch
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from scipy.ndimage import shift, gaussian_filter, rotate, affine_transform, map_coordinates
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from skimage.transform import radon, iradon  


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

    # # K-Space linear phase ramps dictate a rigid spatial shift
    # elif aug_type in ['motion', 'motion_and_breathing']:
    #     shift_vals = (params['dy'], params['dx'])
    #     for z in range(mask.shape[2]):
    #         perturbed_mask[:, :, z] = shift(mask[:, :, z], shift=shift_vals, order=0, mode='constant', cval=0.0)

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

    # --- NEW CT PHYSICS LOGIC ---
    if aug_type in ['motion', 'breathing', 'motion_and_breathing']:
        dy_pixels = params['dy']
        dx_pixels = params['dx']
        
        for z in range(image_array.shape[2]):
            img_slice = image_array[:, :, z]
            H, W = img_slice.shape
            
            # 1. Forward Physics: Radon Transform to Sinogram
            theta = np.linspace(0., 180., max(H, W), endpoint=False)
            sinogram = radon(img_slice, theta=theta, circle=False)
            R_dim, Theta_dim = sinogram.shape
            sinogram_aug = np.zeros_like(sinogram)
            
            # 2. Apply Physical Shifts to Sinogram Angles
            if aug_type == 'motion':
                shifts = np.zeros(Theta_dim, dtype=int)
                shifts[Theta_dim//2:] = int(dx_pixels) 
            elif aug_type == 'breathing':
                shifts = (np.sin(np.linspace(0, 4 * np.pi, Theta_dim)) * dy_pixels).astype(int)
            elif aug_type == 'motion_and_breathing':
                breathing = np.sin(np.linspace(0, 4 * np.pi, Theta_dim)) * dy_pixels
                shifts = np.zeros(Theta_dim, dtype=int)
                shifts[Theta_dim//2:] = int(dx_pixels)
                shifts = (shifts + breathing).astype(int)

            for i in range(Theta_dim):
                sinogram_aug[:, i] = np.roll(sinogram[:, i], shifts[i])

            # 3. Inverse Physics: Filtered Back Projection
            aug_slice = iradon(sinogram_aug, theta=theta, circle=False).astype(np.float32)
            
            # Re-crop if iradon padding expanded the frame
            if aug_slice.shape != (H, W):
                out_H, out_W = aug_slice.shape
                crop_top = (out_H - H) // 2
                crop_left = (out_W - W) // 2
                aug_slice = aug_slice[crop_top:crop_top+H, crop_left:crop_left+W]
                
            perturbed_vol[:, :, z] = aug_slice

    elif aug_type == 'noise':
        noise_rel_std = params['noise_std']
        for z in range(image_array.shape[2]):
            slice_2d = image_array[:, :, z]
            slice_std = np.std(slice_2d) if np.std(slice_2d) > 0 else 1.0
            actual_noise_std = noise_rel_std * slice_std
            noise = np.random.normal(0, actual_noise_std, slice_2d.shape)
            perturbed_vol[:, :, z] = (slice_2d + noise).astype(np.float32)

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
def plot_supergraph(image_vol, gt_vol, base_pred, base_ent, pert_vol, pert_gt_vol, pert_base_pred, pert_base_ent, 
                    pert_cons_pred, pert_cons_ent, pert_all_masks, best_slices, case_id, output_path, aug_type, 
                    base_all_masks=None, pert_params=None):
    n_rows = len(best_slices)
    fig, axes = plt.subplots(n_rows, 11, figsize=(55, 5 * n_rows))
    if n_rows == 1: axes = [axes]
    
    fig.suptitle(f"[{case_id}] | Perturbation: {aug_type.replace('_', ' ').title()}", fontsize=28, weight='bold', y=1.02)
    contour_colors = ['red', 'cyan', 'lime', 'magenta', 'yellow', 'orange', 'white', 'pink', 'blue', 'green']

    for i, z in enumerate(best_slices):
        img_slice = np.rot90(image_vol[:, :, z])
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
    predictor = nnUNetPredictor(tile_step_size=1.0, use_mirroring=False, device=torch.device('cuda'), verbose=True)
    
    # Paths adjusted based on your environment

    img_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/liver/images'
    gt_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/liver/labels'

# for dropout predictions
    # output_base = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/uncert_analysis/mc_results/dropout/prostate_pert_0.2'
    # model_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/nnUNet_results/Dataset005_Prostate/nnUNetTrainerDropoutPhysics__nnUNetPlans__2d'
# for base predictions without dropout

    output_base = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/uncert_analysis/mc_results/normal/trained_physics/liver'
    model_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/nnUNet_results/Dataset003_Liver/nnUNetTrainerPhysics__nnUNetPlans__2d'


    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,), checkpoint_name='checkpoint_best.pth')

    original_eval = predictor.network.eval
    def custom_mcd_eval():
        original_eval() 
        for m in predictor.network.modules():
            if type(m) in [torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d]: m.train() 
        return predictor.network
    # predictor.network.eval = custom_mcd_eval 

    all_files = [f for f in os.listdir(img_ts_folder) if f.endswith('.nii.gz')]
    case_ids = sorted(list(set([f.rsplit('_', 1)[0] for f in all_files])))
    
    tta_types = [
        'motion', 'noise', 'breathing', 'motion_and_breathing', 
        'tta_flip', 'tta_rotate', 'tta_scale', 'tta_translation', 
        'tta_brightness', 'tta_contrast', 'tta_gamma', 'tta_elastic'
    ]

    for case_id in case_ids:
        print(f"\n[{case_id}] Loading Data...")
        
        case_output_dir = os.path.join(output_base, case_id)
        perturbed_dir = os.path.join(case_output_dir, 'perturbed_images')
        uncert_dir = os.path.join(case_output_dir, 'uncertainties')
        viz_dir = os.path.join(case_output_dir, 'visualizations')
        os.makedirs(perturbed_dir, exist_ok=True); os.makedirs(uncert_dir, exist_ok=True); os.makedirs(viz_dir, exist_ok=True)
        
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

        num_passes = 1
        param_log_path = os.path.join(uncert_dir, f"{case_id}_perturbation_params.txt")
        dice_log_path = os.path.join(uncert_dir, f"{case_id}_perturbation_dices.txt")

        # --- STAGE 1: COMPUTE BASELINE ---
        print(f"  === Computing Unperturbed Baseline Consensus (N={num_passes}) ===")
        base_orig_probs = []; base_dices = []
        
        with open(param_log_path, "w") as pf, open(dice_log_path, "w") as df:
            pf.write("--- BASE UNPERTURBED ---\n"); df.write("--- BASE UNPERTURBED VOL DICES ---\n")
            
            for p_idx in range(num_passes):
                probs = predictor.predict_from_files([modality_files], None, save_probabilities=True, overwrite=True,
                                                     num_processes_preprocessing=1, num_processes_segmentation_export=1)
                pred_prob = probs[0][1].transpose(0, 3, 2, 1)
                base_orig_probs.append(pred_prob)
                pf.write(f"Pass {p_idx}: {{'aug_type': 'base'}}\n")

                if has_gt:
                    pass_mask = np.argmax(pred_prob, axis=0)
                    slice_dices = [calculate_dice(pass_mask[:, :, z], gt_array[:, :, z], 1) for z in range(gt_array.shape[2]) if np.sum(gt_array[:, :, z] == 1) > 0]
                    mean_v_dice = np.mean(slice_dices) if slice_dices else 0.0
                    base_dices.append(mean_v_dice)
                    df.write(f"Pass {p_idx}: Mean Slice Dice = {mean_v_dice:.4f}\n")

        base_orig_probs = np.array(base_orig_probs)
        base_mean_prob = np.mean(base_orig_probs, axis=0)
        b_safe = np.clip(base_mean_prob, 1e-7, 1.0)
        base_entropy_map = -np.sum(b_safe * np.log(b_safe), axis=0)
        base_final_mask = np.argmax(base_mean_prob, axis=0)
        base_all_masks = np.argmax(base_orig_probs, axis=1) if has_gt else None

        # --- STAGE 2: ALL COMPARED PERTURBATIONS LOOP ---
        for aug_type in tta_types:
            print(f"\n  === Testing Modality: {aug_type.upper()} ===")
            all_pass_masks = []
            raw_pass_probs = []
            
            with open(param_log_path, "a") as pf, open(dice_log_path, "a") as df:
                pf.write(f"\n--- TTA: {aug_type.upper()} ---\n"); df.write(f"\n--- TTA: {aug_type.upper()} VOL DICES ---\n")
                shared_sync_params = {
                    'aug_type': aug_type, 'dy': 15.0, 'dx': 15.0, 'noise_std': 0.7,    
                    'flip_axis': 1, 'angle': 30.0, 'scale_factor': 1.2,         
                    'trans_dy': 5.0, 'trans_dx': 5.0, 'bright_add': 5, 
                    'cont_mul': 3.5, 'gamma_pow': 0.5, 'elastic_alpha': 50.0, 'elastic_sigma': 5.0                 
                }

                temp_nii_paths = []
                sync_params = shared_sync_params.copy()
                example_pert_vol = None                
                for c_idx, channel_array in enumerate(base_arrays):
                    perturbed_array, sync_params = apply_synchronized_augmentation_3d(channel_array, sync_params)
                    if c_idx == 0: example_pert_vol = perturbed_array.copy()

                    temp_path = os.path.join(perturbed_dir, f"{case_id}_{aug_type}_000{c_idx}.nii.gz")
                    nib.save(nib.Nifti1Image(perturbed_array.astype(np.float32), affine, header), temp_path)
                    temp_nii_paths.append(temp_path)
                    
                log_params = {k:v for k,v in sync_params.items() if k != 'elastic_fields'}
                all_pass_params = [sync_params.copy()] * num_passes 

                # 1. Transform GT natively into augmented space
                perturbed_gt = apply_spatial_mask_3d(gt_array, sync_params) if has_gt else None

                # 2. Evaluate Network natively in augmented space
                for p_idx in range(num_passes):
                    print(f"    -> Pass {p_idx+1}/{num_passes}...")
                    pf.write(f"Pass {p_idx}: {log_params}\n")

                    probs = predictor.predict_from_files([temp_nii_paths], None, save_probabilities=True, overwrite=True,
                                                         num_processes_preprocessing=1, num_processes_segmentation_export=1)
                    
                    pred_prob = probs[0][1].transpose(0, 3, 2, 1)
                    raw_pass_probs.append(pred_prob)
                    
                    raw_mask = np.argmax(pred_prob, axis=0)
                    all_pass_masks.append(raw_mask)

                    if has_gt:
                        slice_dices = [calculate_dice(raw_mask[:, :, z], perturbed_gt[:, :, z], 1) for z in range(perturbed_gt.shape[2]) if np.sum(perturbed_gt[:, :, z] == 1) > 0]
                        mean_v_dice = np.mean(slice_dices) if slice_dices else 0.0
                        df.write(f"Pass {p_idx}: Mean Slice Dice = {mean_v_dice:.4f}\n")

            all_pass_masks = np.array(all_pass_masks)
            raw_pass_probs = np.array(raw_pass_probs)

            pert_base_mask = all_pass_masks[0]
            pert_cons_mask = (np.mean(all_pass_masks == 1, axis=0) >= 0.5).astype(np.uint8)

            vis_base_prob = raw_pass_probs[0]
            vb_safe = np.clip(vis_base_prob, 1e-7, 1.0)
            vis_base_entropy = -np.sum(vb_safe * np.log(vb_safe), axis=0)

            vis_cons_prob = np.mean(raw_pass_probs, axis=0)
            vc_safe = np.clip(vis_cons_prob, 1e-7, 1.0)
            vis_cons_entropy = -np.sum(vc_safe * np.log(vc_safe), axis=0)

            nib.save(nib.Nifti1Image(pert_cons_mask.astype(np.uint8), affine), os.path.join(uncert_dir, f"{case_id}_{aug_type}_pred.nii.gz"))
            nib.save(nib.Nifti1Image(vis_cons_entropy.astype(np.float32), affine), os.path.join(uncert_dir, f"{case_id}_{aug_type}_entropy.nii.gz"))
            
            if has_gt:
                b_slice_dices = [calculate_dice(pert_base_mask[:, :, z], perturbed_gt[:, :, z], 1) for z in range(perturbed_gt.shape[2]) if np.sum(perturbed_gt[:, :, z] == 1) > 0]
                base_mean_dice = np.mean(b_slice_dices) if b_slice_dices else 0.0
                
                c_slice_dices = [calculate_dice(pert_cons_mask[:, :, z], perturbed_gt[:, :, z], 1) for z in range(perturbed_gt.shape[2]) if np.sum(perturbed_gt[:, :, z] == 1) > 0]
                cons_mean_dice = np.mean(c_slice_dices) if c_slice_dices else 0.0

                print(f"    -> {aug_type.upper()} | Pass 1 Mean Dice: {base_mean_dice:.4f} | Consensus Mean Dice: {cons_mean_dice:.4f}")

                # --- NEW: Write detailed per-pass per-slice Dices to visualizations dir ---
                detailed_dice_path = os.path.join(viz_dir, f"{case_id}_{aug_type}_slice_dices_per_pass.txt")
                with open(detailed_dice_path, "w") as f_det:
                    f_det.write(f"Detailed Dice scores for {case_id} under {aug_type} perturbation\n")
                    f_det.write("="*65 + "\n")
                    for z in range(perturbed_gt.shape[2]):
                        if np.sum(perturbed_gt[:, :, z] == 1) > 0:
                            f_det.write(f"Slice {z}:\n")
                            for p_idx in range(all_pass_masks.shape[0]):
                                p_dice = calculate_dice(all_pass_masks[p_idx, :, :, z], perturbed_gt[:, :, z], 1)
                                f_det.write(f"  Pass {p_idx+1} Dice: {p_dice:.4f}\n")
                            cons_dice = calculate_dice(pert_cons_mask[:, :, z], perturbed_gt[:, :, z], 1)
                            f_det.write(f"  Consensus Dice: {cons_dice:.4f}\n\n")
                # --------------------------------------------------------------------------

                slice_scores = [(z, calculate_dice(pert_cons_mask[:, :, z], perturbed_gt[:, :, z], 1)) for z in range(perturbed_gt.shape[2]) if np.sum(perturbed_gt[:, :, z] == 1) > 0]
                slice_scores.sort(key=lambda x: x[1], reverse=True)
                top_slices = [s[0] for s in slice_scores[:3]]

                t2_vol = base_arrays[0] 
                viz_path = os.path.join(viz_dir, f"{case_id}_{aug_type}_summary.png")

                
                # --- NEW: Save the constituent images for ALL slices ---
                images_dir = os.path.join(case_output_dir, 'images', aug_type)
                os.makedirs(images_dir, exist_ok=True)
                
                for z in range(example_pert_vol.shape[2]):
                    # Create a specific directory for this slice
                    slice_dir = os.path.join(images_dir, f"slice_{z}")
                    os.makedirs(slice_dir, exist_ok=True)

                    # Extract and rotate to match the supergraph orientation
                    img_slice = np.rot90(example_pert_vol[:, :, z])
                    pred_slice = (np.rot90(pert_cons_mask[:, :, z]) == 1).astype(np.uint8)
                    
                    plt.imsave(os.path.join(slice_dir, f"{case_id}_slice{z}_img.png"), img_slice, cmap='gray')
                    plt.imsave(os.path.join(slice_dir, f"{case_id}_slice{z}_pred.png"), pred_slice, cmap='gray')
                    
                    if has_gt:
                        gt_slice = (np.rot90(perturbed_gt[:, :, z]) == 1).astype(np.uint8)
                        plt.imsave(os.path.join(slice_dir, f"{case_id}_slice{z}_gt.png"), gt_slice, cmap='gray')

                        # Calculate and save the individual slice dice score
                        slice_dice = calculate_dice(pert_cons_mask[:, :, z], perturbed_gt[:, :, z], 1)
                        dice_txt_path = os.path.join(slice_dir, f"{case_id}_slice{z}_dice.txt")
                        with open(dice_txt_path, "w") as f:
                            f.write(f"Slice: {z}\n")
                            f.write(f"Augmentation: {aug_type}\n")
                            f.write(f"Dice Score (Class 1): {slice_dice:.4f}\n")
                # -------------------------------------------------------
                    
                if top_slices:
                    # Passed perturbed_gt to the plotting function
                    plot_supergraph(t2_vol, gt_array, base_final_mask, base_entropy_map, example_pert_vol, perturbed_gt,
                                    pert_base_mask, vis_base_entropy, pert_cons_mask, vis_cons_entropy, 
                                    all_pass_masks, top_slices, case_id, viz_path, aug_type, 
                                    base_all_masks=base_all_masks, pert_params=all_pass_params)
        break
if __name__ == '__main__':
    main()