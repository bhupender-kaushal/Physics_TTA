# import torch
# import numpy as np
# import nibabel as nib
# import os
# from scipy.ndimage import rotate
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# # --- 1. SPATIAL & INTENSITY PERTURBATION LOGIC ---
# def apply_synchronized_augmentation(image_array, params=None):
#     """
#     Applies custom augmentations. If params is None, it generates new random parameters.
#     By passing the generated params to the next channel, we ensure T2 and ADC match perfectly.
#     """
#     if params is None:
#         # Generate random perturbation parameters for this specific pass
#         angle = np.random.uniform(-15, 15)
#         noise_std = np.random.uniform(0.01, 0.05)
#         params = {'angle': angle, 'noise_std': noise_std}

#     perturbed = image_array.copy()

#     # 1. Intensity Augmentation (Gaussian Noise)
#     # Applied dynamically based on the channel's standard deviation
#     noise = np.random.normal(0, np.std(perturbed) * params['noise_std'], perturbed.shape)
#     perturbed += noise
    
#     # 2. Spatial Augmentation (Rotation)
#     # Assumes shape [X, Y, Z], rotating cleanly along the X/Y plane (axes 0 and 1)
#     perturbed = rotate(perturbed, params['angle'], axes=(0, 1), reshape=False, order=3, mode='nearest')
    
#     return perturbed, params

# def invert_spatial_augmentation(prob_map, params):
#     """
#     Reverts the spatial shift on the predictions so they strictly align with the ground truth.
#     prob_map shape: [Classes, X, Y, Z]
#     """
#     aligned_prob = np.zeros_like(prob_map)
#     inv_angle = -params['angle'] # Negative angle perfectly reverses the rotation
    
#     # Rotate each class probability map independently back to the original space
#     for c in range(prob_map.shape[0]):
#         aligned_prob[c] = rotate(prob_map[c], inv_angle, axes=(0, 1), reshape=False, order=1, mode='nearest')
        
#     return np.clip(aligned_prob, 0.0, 1.0)


# # --- 2. MAIN EXECUTION ---
# def main():
#     # Setup the Predictor
#     predictor = nnUNetPredictor(
#         tile_step_size=0.9, # Fast inference
#         use_mirroring=False, 
#         device=torch.device('cuda'),
#         verbose=False
#     )

#     # Note: You can use the standard nnUNetTrainer or the Dropout one depending on your goals.
#     model_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/nnUNet_results/Dataset005_Prostate/nnUNetTrainerDropout__nnUNetPlans__2d'
#     img_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/Prostate/Images'
#     output_base = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/uncert_analysis/mc_results/prostate_pert'
    
#     perturbed_dir = os.path.join(output_base, 'perturbed_images')
#     uncert_dir = os.path.join(output_base, 'uncertainties')
#     os.makedirs(perturbed_dir, exist_ok=True)
#     os.makedirs(uncert_dir, exist_ok=True)

#     predictor.initialize_from_trained_model_folder(
#         model_folder, use_folds=(0,), checkpoint_name='checkpoint_final.pth'
#     )

#     # Force Dropout to stay active (Remove if you ONLY want physics perturbations, not MC Dropout)
#     original_eval = predictor.network.eval
#     def custom_mcd_eval():
#         original_eval() 
#         for m in predictor.network.modules():
#             if type(m) in [torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d]:
#                 m.train() 
#         return predictor.network
#     predictor.network.eval = custom_mcd_eval 

#     # Find Unique Patient Cases
#     all_files = [f for f in os.listdir(img_ts_folder) if f.endswith('.nii.gz')]
#     case_ids = sorted(list(set([f.rsplit('_', 1)[0] for f in all_files])))

#     for case_id in case_ids:
#         print(f"\nProcessing Multi-Channel TTA for Case: {case_id}")
        
#         # Collect modalities cleanly
#         modality_files = sorted([f for f in all_files if f.startswith(f"{case_id}_")])
#         print(f"  -> Loading {len(modality_files)} channels...")

#         # Pre-load arrays into memory to apply augmentations quickly
#         base_arrays = []
#         affine, header = None, None
#         for mod in modality_files:
#             nii = nib.load(os.path.join(img_ts_folder, mod))
#             base_arrays.append(nii.get_fdata())
#             if affine is None:
#                 affine = nii.affine
#                 header = nii.header

#         num_passes = 5  
#         all_pass_probs = []
        
#         for p_idx in range(num_passes):
#             print(f"  -> Running Perturbed Pass {p_idx+1}/{num_passes}...")
            
#             temp_nii_paths = []
#             sync_params = None
            
#             # 1. Apply synchronized augmentations to ALL channels for this pass
#             for c_idx, channel_array in enumerate(base_arrays):
#                 # Generates random params on channel 0, re-uses them on channel 1, 2...
#                 perturbed_array, sync_params = apply_synchronized_augmentation(channel_array, sync_params)
                
#                 # Save the explicitly perturbed temporary file
#                 temp_path = os.path.join(perturbed_dir, f"{case_id}_pass_{p_idx}_000{c_idx}.nii.gz")
#                 nib.save(nib.Nifti1Image(perturbed_array.astype(np.float32), affine, header), temp_path)
#                 temp_nii_paths.append(temp_path)
            
#             # 2. Predict using the newly generated perturbed channels
#             probs = predictor.predict_from_files(
#                 [temp_nii_paths], # nnU-Net requires a list of file lists
#                 None, 
#                 save_probabilities=True, overwrite=True,
#                 num_processes_preprocessing=1, num_processes_segmentation_export=1
#             )
            
#             pred_mask, pred_prob = probs[0]
            
#             # 3. CRITICAL: Invert the spatial rotation back to the patient's native orientation
#             aligned_prob = invert_spatial_augmentation(pred_prob, sync_params)
#             all_pass_probs.append(aligned_prob)

#         # 4. Calculate Final Alignment Metrics
#         print("  -> Calculating Uncertainty Maps...")
#         all_pass_probs = np.array(all_pass_probs) # Shape: [N_mc, num_classes, X, Y, Z]
        
#         mean_prob = np.mean(all_pass_probs, axis=0)
#         final_mask = np.argmax(mean_prob, axis=0)
        
#         uncertainty_map = np.var(all_pass_probs, axis=0).sum(axis=0)

#         # 5. Save Final Consensus and Uncertainty
#         print("  -> Saving final output files...")
#         nib.save(nib.Nifti1Image(final_mask.astype(np.uint8), affine), 
#                  os.path.join(uncert_dir, f"{case_id}_pred.nii.gz"))
        
#         nib.save(nib.Nifti1Image(uncertainty_map.astype(np.float32), affine), 
#                  os.path.join(uncert_dir, f"{case_id}_uncertainty.nii.gz"))

#         break # Remove this break to process all patients instead of just the first

#     print("\nTTA & MC Dropout Inference Complete!")

# if __name__ == '__main__':
#     main()

import torch
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from scipy.ndimage import shift
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

# --- 2. NOVEL PHYSICS-BASED PERTURBATION LOGIC ---
def apply_synchronized_physics_3d(image_array, params=None):
    if params is None:
        aug_type = np.random.choice(['motion', 'noise', 'breathing', 'motion_and_breathing'])
        dy = np.random.uniform(-4.0, 4.0)
        dx = np.random.uniform(-4.0, 4.0)
        noise_std = np.random.uniform(0.01, 0.05)
        params = {'aug_type': aug_type, 'dy': dy, 'dx': dx, 'noise_std': noise_std}

    aug_type = params['aug_type']
    dy, dx = params['dy'], params['dx']
    noise_rel_std = params['noise_std']
    
    perturbed_vol = np.zeros_like(image_array, dtype=np.float32)

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
        
    return perturbed_vol, params

def invert_physics_spatial_3d(prob_map, params):
    aligned_prob = np.copy(prob_map)
    aug_type = params['aug_type']
    
    if aug_type in ['motion', 'motion_and_breathing']:
        inv_shift = (-params['dy'], -params['dx'], 0) 
        for c in range(prob_map.shape[0]):
            aligned_prob[c] = shift(prob_map[c], shift=inv_shift, order=1, mode='nearest')
            
    return np.clip(aligned_prob, 0.0, 1.0)

# --- 3. VISUALIZATION LOGIC ---
def plot_supergraph(t2_vol, gt_vol, base_pred, base_ent, pert_vol, pert_base_pred, pert_base_ent, 
                    pert_cons_pred, pert_cons_ent, pert_all_masks, best_slices, case_id, output_path, aug_type, 
                    base_all_masks=None, pert_params=None):
    n_rows = len(best_slices)
    fig, axes = plt.subplots(n_rows, 11, figsize=(55, 5 * n_rows))
    if n_rows == 1: axes = [axes]
    
    fig.suptitle(f"[{case_id}] | Perturbation: {aug_type.upper()}", fontsize=28, weight='bold', y=1.02)
    contour_colors = ['red', 'cyan', 'lime', 'magenta', 'yellow', 'orange', 'white', 'pink', 'blue', 'green']

    for i, z in enumerate(best_slices):
        img_slice = np.rot90(t2_vol[:, :, z])
        gt_slice = np.rot90(gt_vol[:, :, z])
        b_pred_slice = np.rot90(base_pred[:, :, z])
        b_ent_slice = np.rot90(base_ent[:, :, z])
        pert_img_slice = np.rot90(pert_vol[:, :, z])
        p_base_pred_slice = np.rot90(pert_base_pred[:, :, z])
        p_base_ent_slice = np.rot90(pert_base_ent[:, :, z])
        p_cons_pred_slice = np.rot90(pert_cons_pred[:, :, z])
        p_cons_ent_slice = np.rot90(pert_cons_ent[:, :, z])
        
        # Cols 0-8: (Original Code Layout)
        axes[i][0].imshow(img_slice, cmap='gray')
        axes[i][0].imshow(np.ma.masked_where(gt_slice == 0, gt_slice), cmap='tab10', alpha=0.5)
        axes[i][0].set_title(f"Orig GT (Slice {z})", fontsize=16)
        axes[i][0].axis('off')
        
        axes[i][1].imshow(img_slice, cmap='gray')
        axes[i][1].imshow(np.ma.masked_where(b_pred_slice == 0, b_pred_slice), cmap='tab10', alpha=0.5)
        axes[i][1].set_title(f"Orig Base Pred (Dice: {calculate_dice(b_pred_slice, gt_slice):.3f})", fontsize=16)
        axes[i][1].axis('off')

        axes[i][2].imshow(img_slice, cmap='gray', alpha=0.5)
        im = axes[i][2].imshow(b_ent_slice, cmap='inferno', alpha=0.8, vmin=0)
        fig.colorbar(im, ax=axes[i][2], fraction=0.046, pad=0.04)
        axes[i][2].set_title("Orig Base Entropy", fontsize=16)
        axes[i][2].axis('off')

        axes[i][3].imshow(pert_img_slice, cmap='gray')
        axes[i][3].set_title(f"Perturbed Image Context", fontsize=16)
        axes[i][3].axis('off')
        
        axes[i][4].imshow(pert_img_slice, cmap='gray')
        axes[i][4].imshow(np.ma.masked_where(p_base_pred_slice == 0, p_base_pred_slice), cmap='tab10', alpha=0.5)
        axes[i][4].set_title(f"Pert Pass 1 Pred (Dice: {calculate_dice(p_base_pred_slice, gt_slice):.3f})", fontsize=16)
        axes[i][4].axis('off')
        
        axes[i][5].imshow(img_slice, cmap='gray', alpha=0.5)
        im_pb = axes[i][5].imshow(p_base_ent_slice, cmap='inferno', alpha=0.8, vmin=0)
        fig.colorbar(im_pb, ax=axes[i][5], fraction=0.046, pad=0.04)
        axes[i][5].set_title("Pert Pass 1 Entropy", fontsize=16)
        axes[i][5].axis('off')

        axes[i][6].imshow(img_slice, cmap='gray')
        for pass_idx in range(pert_all_masks.shape[0]):
            pass_mask = np.rot90(pert_all_masks[pass_idx, :, :, z])
            if np.any(pass_mask == 1):
                lw = 2.5 if pass_idx == 0 else 1.0
                alpha = 0.9 if pass_idx == 0 else 0.5
                c_idx = pass_idx % len(contour_colors)
                axes[i][6].contour(pass_mask == 1, levels=[0.5], colors=[contour_colors[c_idx]], alpha=alpha, linewidths=lw)
        axes[i][6].set_title("Multi-Pass Disagreement", fontsize=16)
        axes[i][6].axis('off')

        axes[i][7].imshow(img_slice, cmap='gray')
        axes[i][7].imshow(np.ma.masked_where(p_cons_pred_slice == 0, p_cons_pred_slice), cmap='tab10', alpha=0.5)
        axes[i][7].set_title(f"Pert Cons Pred (Dice: {calculate_dice(p_cons_pred_slice, gt_slice):.3f})", fontsize=16)
        axes[i][7].axis('off')

        axes[i][8].imshow(img_slice, cmap='gray', alpha=0.5)
        im2 = axes[i][8].imshow(p_cons_ent_slice, cmap='inferno', alpha=0.8, vmin=0)
        fig.colorbar(im2, ax=axes[i][8], fraction=0.046, pad=0.04)
        axes[i][8].set_title("Pert Cons Entropy", fontsize=16)
        axes[i][8].axis('off')

        if base_all_masks is not None and pert_all_masks is not None:
            base_slice_dices = [calculate_dice(m[:, :, z], np.rot90(gt_slice, -1), class_idx=1) for m in base_all_masks]
            pert_slice_dices = [calculate_dice(m[:, :, z], np.rot90(gt_slice, -1), class_idx=1) for m in pert_all_masks]
            
            # Col 9: Box Plot
            box = axes[i][9].boxplot([base_slice_dices, pert_slice_dices], labels=['Base Img N', 'Pert Img N'], patch_artist=True)
            axes[i][9].scatter([1]*len(base_slice_dices), base_slice_dices, color='darkblue', alpha=0.6, zorder=3)
            axes[i][9].scatter([2]*len(pert_slice_dices[1:]), pert_slice_dices[1:], color='darkred', alpha=0.4, zorder=3)
            axes[i][9].scatter(2, pert_slice_dices[0], color='gold', edgecolor='black', s=100, zorder=4, label="Pert Pass 1")
            
            for patch, color in zip(box['boxes'], ['lightblue', 'lightcoral']): patch.set_facecolor(color)
            axes[i][9].set_title(f"Slice {z} Run Variances", fontsize=16)
            axes[i][9].set_ylim(-0.05, 1.05)
            axes[i][9].grid(axis='y', linestyle='--', alpha=0.7)
            axes[i][9].legend(loc="lower right", fontsize=8)

            # Col 10: Disagreement Correlations
            if pert_params is not None:
                motion_mags = [np.sqrt(p['dy']**2 + p['dx']**2) for p in pert_params]
                noises_scaled = [p['noise_std'] * 100 for p in pert_params] # Scale to 1-5 range for visible overlapping

                num_p = len(pert_slice_dices)
                cmap = plt.cm.viridis # Use viridis colormap to dynamically map pass indices to color
                
                for p_idx in range(num_p):
                    # Intensity of the color regulates exactly which pass it is
                    c = cmap(p_idx / max(1, num_p - 1))
                    
                    # Highlight pass 1 with bold borders and larger size
                    s = 150 if p_idx == 0 else 70
                    ec = 'black' if p_idx == 0 else 'none'
                    lw = 2 if p_idx == 0 else 0
                    zord = 5 if p_idx == 0 else 3
                    
                    # Square shape for Motion
                    axes[i][10].scatter(motion_mags[p_idx], pert_slice_dices[p_idx], color=c, marker='s', s=s, edgecolor=ec, linewidth=lw, zorder=zord, alpha=0.8)
                    # Round shape for Noise
                    axes[i][10].scatter(noises_scaled[p_idx], pert_slice_dices[p_idx], color=c, marker='o', s=s, edgecolor=ec, linewidth=lw, zorder=zord, alpha=0.8)
                
                # Invisible Scatters purely to generate an accurate Shape Legend
                axes[i][10].scatter([], [], color='gray', marker='s', label='Motion Mag (dx+dy)')
                axes[i][10].scatter([], [], color='gray', marker='o', label='Noise (x100)')
                
                axes[i][10].set_title("Perturbation vs Dice", fontsize=16)
                axes[i][10].set_ylabel("Dice Score")
                axes[i][10].set_xlabel("Parameter Intensity")
                axes[i][10].grid(linestyle='--', alpha=0.5)
                axes[i][10].legend(loc='lower left', fontsize=10)
        else:
            axes[i][9].axis('off')
            axes[i][10].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# --- 4. MAIN EXECUTION ---
def main():
    predictor = nnUNetPredictor(
        tile_step_size=0.9, use_mirroring=False, device=torch.device('cuda'), verbose=False
    )

    # model_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/nnUNet_results/Dataset005_Prostate/nnUNetTrainer__nnUNetPlans__2d'
    model_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/nnUNet_results/Dataset005_Prostate/nnUNetTrainerDropout__nnUNetPlans__2d'
    img_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/prostate/images'
    gt_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/prostate/labels'
    output_base = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/uncert_analysis/mc_results/prostate_pert'
    
    perturbed_dir = os.path.join(output_base, 'perturbed_images')
    uncert_dir = os.path.join(output_base, 'uncertainties')
    viz_dir = os.path.join(output_base, 'visualizations')
    os.makedirs(perturbed_dir, exist_ok=True); os.makedirs(uncert_dir, exist_ok=True); os.makedirs(viz_dir, exist_ok=True)

    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,), checkpoint_name='checkpoint_best.pth')

    original_eval = predictor.network.eval
    def custom_mcd_eval():
        original_eval() 
        for m in predictor.network.modules():
            if type(m) in [torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d]: m.train() 
        return predictor.network
    predictor.network.eval = custom_mcd_eval 

    all_files = [f for f in os.listdir(img_ts_folder) if f.endswith('.nii.gz')]
    case_ids = sorted(list(set([f.rsplit('_', 1)[0] for f in all_files])))
    physics_types = ['motion', 'noise', 'breathing', 'motion_and_breathing']

    for case_id in case_ids:
        print(f"\n[{case_id}] Loading Data...")
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

        num_passes = 20

        # Output Log Files Initialization for this specific case
        param_log_path = os.path.join(uncert_dir, f"{case_id}_perturbation_params.txt")
        dice_log_path = os.path.join(uncert_dir, f"{case_id}_perturbation_dices.txt")

        # ----------------------------------------------------
        # STAGE 1: COMPUTE BASELINE (NO PERTURBATION) GT CONSENSUS
        # ----------------------------------------------------
        print(f"  === Computing Unperturbed Baseline Consensus (N={num_passes}) ===")
        base_orig_probs = []
        base_dices = []
        
        with open(param_log_path, "w") as pf, open(dice_log_path, "w") as df:
            pf.write("--- BASE UNPERTURBED ---\n")
            df.write("--- BASE UNPERTURBED VOL DICES ---\n")
            
            for p_idx in range(num_passes):
                probs = predictor.predict_from_files([modality_files], None, save_probabilities=True, overwrite=True,
                                                     num_processes_preprocessing=1, num_processes_segmentation_export=1)
                pred_prob = probs[0][1].transpose(0, 3, 2, 1)
                base_orig_probs.append(pred_prob)

                # Log "Base" Parameter
                base_param = {'aug_type': 'base', 'dy': 0.0, 'dx': 0.0, 'noise_std': 0.0}
                pf.write(f"Pass {p_idx}: {base_param}\n")

                if has_gt:
                    pass_mask = np.argmax(pred_prob, axis=0)
                    v_dice = calculate_dice(pass_mask, gt_array, class_idx=1)
                    base_dices.append(v_dice)
                    df.write(f"Pass {p_idx}: Dice = {v_dice:.4f}\n")

        base_orig_probs = np.array(base_orig_probs)
        base_mean_prob = np.mean(base_orig_probs, axis=0)
        b_safe = np.clip(base_mean_prob, 1e-7, 1.0)
        base_entropy_map = -np.sum(b_safe * np.log(b_safe), axis=0)
        base_final_mask = np.argmax(base_mean_prob, axis=0)
        base_all_masks = np.argmax(base_orig_probs, axis=1) if has_gt else None

        # ----------------------------------------------------
        # STAGE 2: COMPUTE PERTURBED MC DROPOUT LOOP
        # ----------------------------------------------------
        for aug_type in physics_types:
            print(f"\n  === Testing Physics Modality: {aug_type.upper()} ===")
            all_pass_probs = []
            all_pass_params = []
            example_pert_vol = None 
            
            with open(param_log_path, "a") as pf, open(dice_log_path, "a") as df:
                pf.write(f"\n--- PERTURBED: {aug_type.upper()} ---\n")
                df.write(f"\n--- PERTURBED: {aug_type.upper()} VOL DICES ---\n")

                for p_idx in range(num_passes):
                    print(f"    -> Pass {p_idx+1}/{num_passes}...")
                    temp_nii_paths = []
                    
                    # sync_params = {
                    #     'aug_type': aug_type,
                    #     'dy': np.random.uniform(-4.0, 4.0),
                    #     'dx': np.random.uniform(-4.0, 4.0),
                    #     'noise_std': np.random.uniform(0.01, 0.05)
                    # }

                                        # Fix parameters identical across passes as requested
                    sync_params = {
                        'aug_type': aug_type,
                        'dy': 5.0,
                        'dx': 3.0,
                        'noise_std': 0.1
                    } # making the parameters fixed across passes for demonstration; replace with random generation if desired
                    all_pass_params.append(sync_params)
                    pf.write(f"Pass {p_idx}: {sync_params}\n")
                    
                    for c_idx, channel_array in enumerate(base_arrays):
                        perturbed_array, sync_params = apply_synchronized_physics_3d(channel_array, sync_params)
                        if p_idx == 0 and c_idx == 0: example_pert_vol = perturbed_array.copy()

                        temp_path = os.path.join(perturbed_dir, f"{case_id}_{aug_type}_pass_{p_idx}_000{c_idx}.nii.gz")
                        nib.save(nib.Nifti1Image(perturbed_array.astype(np.float32), affine, header), temp_path)
                        temp_nii_paths.append(temp_path)
                        
                    probs = predictor.predict_from_files([temp_nii_paths], None, save_probabilities=True, overwrite=True,
                                                         num_processes_preprocessing=1, num_processes_segmentation_export=1)
                    
                    pred_prob = probs[0][1].transpose(0, 3, 2, 1)
                    aligned_prob = invert_physics_spatial_3d(pred_prob, sync_params)
                    all_pass_probs.append(aligned_prob)

                    if has_gt:
                        pass_mask = np.argmax(aligned_prob, axis=0)
                        v_dice = calculate_dice(pass_mask, gt_array, class_idx=1)
                        df.write(f"Pass {p_idx}: Dice = {v_dice:.4f}\n")

            all_pass_probs = np.array(all_pass_probs)
            pert_all_masks = np.argmax(all_pass_probs, axis=1) 

            # Extract Consensuses
            pert_base_prob = all_pass_probs[0]
            pert_base_mask = np.argmax(pert_base_prob, axis=0)
            pb_safe = np.clip(pert_base_prob, 1e-7, 1.0)
            pert_base_entropy = -np.sum(pb_safe * np.log(pb_safe), axis=0)

            pert_cons_prob = np.mean(all_pass_probs, axis=0)
            pert_cons_mask = np.argmax(pert_cons_prob, axis=0)
            pc_safe = np.clip(pert_cons_prob, 1e-7, 1.0)
            pert_cons_entropy = -np.sum(pc_safe * np.log(pc_safe), axis=0) 
            
            nib.save(nib.Nifti1Image(pert_cons_mask.astype(np.uint8), affine), os.path.join(uncert_dir, f"{case_id}_{aug_type}_pred.nii.gz"))
            nib.save(nib.Nifti1Image(pert_cons_entropy.astype(np.float32), affine), os.path.join(uncert_dir, f"{case_id}_{aug_type}_entropy.nii.gz"))

            if has_gt:
                base_vol_dice = calculate_dice(pert_base_mask, gt_array, class_idx=1)
                cons_vol_dice = calculate_dice(pert_cons_mask, gt_array, class_idx=1)
                print(f"    -> {aug_type} | Pert Pass 1 Dice: {base_vol_dice:.4f} | Pert Consensus Dice: {cons_vol_dice:.4f}")

                slice_scores = []
                for z in range(gt_array.shape[2]):
                    if np.sum(gt_array[:, :, z] == 1) > 0:
                        score = calculate_dice(pert_cons_mask[:, :, z], gt_array[:, :, z], class_idx=1)
                        slice_scores.append((z, score))
                
                slice_scores.sort(key=lambda x: x[1], reverse=True)
                top_slices = [s[0] for s in slice_scores[:3]]

                if len(top_slices) > 0:
                    print(f"    -> Saving visualization with extended columns...")
                    t2_vol = base_arrays[0] 
                    viz_path = os.path.join(viz_dir, f"{case_id}_{aug_type}_summary.png")
                    
                    plot_supergraph(t2_vol, gt_array, base_final_mask, base_entropy_map, example_pert_vol, 
                                    pert_base_mask, pert_base_entropy, pert_cons_mask, pert_cons_entropy, 
                                    pert_all_masks, top_slices, case_id, viz_path, aug_type, 
                                    base_all_masks=base_all_masks, pert_params=all_pass_params)

        break # Remove break to process all patients

    print("\nTTA Physics & MC Dropout Inference Complete!")

if __name__ == '__main__':
    main()