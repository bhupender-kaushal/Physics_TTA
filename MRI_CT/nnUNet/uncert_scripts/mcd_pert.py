import torch
import numpy as np
import nibabel as nib
import os
from scipy.ndimage import rotate
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# --- 1. DEFINE YOUR CUSTOM AUGMENTATIONS & INVERSES ---
def apply_novel_augmentation(image_array, pass_idx):
    """
    Applies your custom augmentations. Returns the perturbed image 
    and any parameters needed to reverse the spatial effect later.
    """
    perturbed = image_array.copy()
    
    # Example 1: Additive Noise (No spatial shift needed)
    noise = np.random.normal(0, np.std(perturbed) * 0.1, perturbed.shape)
    perturbed += noise
    
    # Example 2: Spatial Shift (Rotation)
    angle = np.random.uniform(-15, 15)
    
    # Assuming shape is [X, Y, Z], rotate along X/Y axes
    perturbed = rotate(perturbed, angle, axes=(0, 1), reshape=False, order=3, mode='nearest')
    
    # return the perturbed image AND the inverse parameters (negative angle)
    inverse_params = {'angle': -angle}
    return perturbed, inverse_params

def invert_novel_augmentation(prob_map, inverse_params):
    """
    Applies the inverse spatial shift to the probability maps so they align perfectly 
    for variance/uncertainty calculation.
    """
    # prob_map shape is [C, X, Y, Z], so we must iterate over channels or specify axes carefully
    aligned_prob = np.zeros_like(prob_map)
    angle = inverse_params['angle']
    
    for c in range(prob_map.shape[0]):
        aligned_prob[c] = rotate(prob_map[c], angle, axes=(0, 1), reshape=False, order=1, mode='nearest')
        
    # Re-normalize just in case interpolation caused values outside [0, 1]
    return np.clip(aligned_prob, 0.0, 1.0)

# --- 2. MAIN SCRIPT ---
def main():
    predictor = nnUNetPredictor(
        tile_step_size=0.9, use_mirroring=False, 
        device=torch.device('cuda'), verbose=False
    )

    model_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/nnUNet_results/Dataset005_Prostate/nnUNetTrainerDropout__nnUNetPlans__2d'
    # model_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/nnUNet_results/Dataset003_Liver/nnUNetTrainerDropout__nnUNetPlans__2d'

    # img_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/Liver/Images'
    img_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/Prostate/Images'
    output_base = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/uncert_analysis/mc_results/prostate'
    
    # Setup directories to save the perturbed inputs and the final uncertainties
    perturbed_dir = os.path.join(output_base, 'perturbed_images')
    uncert_dir = os.path.join(output_base, 'uncertainties')
    os.makedirs(perturbed_dir, exist_ok=True)
    os.makedirs(uncert_dir, exist_ok=True)

    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,), checkpoint_name='checkpoint_final.pth')

    # Force Model into Eval but keep Dropout active (if you want MCD + Custom TTA simultaneously)
    original_eval = predictor.network.eval
    def custom_mcd_eval():
        original_eval()
        for m in predictor.network.modules():
            if type(m) in [torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d]:
                m.train()
        return predictor.network
    predictor.network.eval = custom_mcd_eval 

    for file_name in os.listdir(img_ts_folder):
        if not file_name.endswith('.nii.gz'): continue
            
        img_path = os.path.join(img_ts_folder, file_name)
        case_id = file_name.replace('_0000.nii.gz', '')
        print(f"\nProcessing: {case_id}")
        
        # Load Original NIfTI
        nii_img = nib.load(img_path)
        base_array = nii_img.get_fdata()
        affine = nii_img.affine
        header = nii_img.header

        num_passes = 5
        all_pass_probs = []
        
        for i in range(num_passes):
            print(f"  -> Pass {i+1}/{num_passes}...")
            
            # 1. Perturb the specific pass
            perturbed_array, inv_params = apply_novel_augmentation(base_array, i)
            
            # 2. Save the Perturbed Image (as requested)
            temp_nii_path = os.path.join(perturbed_dir, f"{case_id}_pass_{i}_0000.nii.gz")
            nib.save(nib.Nifti1Image(perturbed_array.astype(np.float32), affine, header), temp_nii_path)
            
            # 3. Predict on this newly generated perturbed file
            probs = predictor.predict_from_files(
                [[temp_nii_path]], None, save_probabilities=True, overwrite=True,
                num_processes_preprocessing=1, num_processes_segmentation_export=1
            )
            
            # 4. Extract Probability Map [C, X, Y, Z]
            pred_mask, pred_prob = probs[0]
            
            # 5. INVERT the spatial augmentations so it aligns with the GT/Original image
            aligned_prob = invert_novel_augmentation(pred_prob, inv_params)
            
            all_pass_probs.append(aligned_prob)

        # 6. Calculate Final Statistics natively on aligned predictions
        all_pass_probs = np.array(all_pass_probs) # [N_mc, C, X, Y, Z]
        
        mean_prob = np.mean(all_pass_probs, axis=0)
        final_mask = np.argmax(mean_prob, axis=0) # Consensus mask
        uncertainty_map = np.var(all_pass_probs, axis=0).sum(axis=0) # Variance map
        
        # 7. Save outputs
        nib.save(nib.Nifti1Image(final_mask.astype(np.uint8), affine), 
                 os.path.join(uncert_dir, f"{case_id}_pred.nii.gz"))
        nib.save(nib.Nifti1Image(uncertainty_map.astype(np.float32), affine), 
                 os.path.join(uncert_dir, f"{case_id}_uncertainty.nii.gz"))
                 
        # Optional: Clean up temporary files if you don't want them filling up your hard drive
        # os.remove(temp_nii_path)

if __name__ == '__main__':
    main()