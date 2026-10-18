import torch
import numpy as np
import nibabel as nib
import os
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    # 1. Setup the Predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.9, # 0.9 = Fast, 0.5 = High quality
        use_mirroring=False, 
        device=torch.device('cuda'),
        verbose=False
    )

    model_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/nnUNet_results/Dataset005_Prostate/nnUNetTrainerDropout__nnUNetPlans__2d'
    img_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/Prostate/Images'
    output_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/uncert_analysis/mc_results/prostate'
    
    os.makedirs(output_folder, exist_ok=True)

    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )

    # --- THE MC DROPOUT FIX ---
    original_eval = predictor.network.eval
    def custom_mcd_eval():
        original_eval() 
        for m in predictor.network.modules():
            if type(m) in [torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d]:
                m.train() 
        return predictor.network
    
    predictor.network.eval = custom_mcd_eval 

    # --- MULTI-CHANNEL PATIENT GROUPING ---
    all_files = [f for f in os.listdir(img_ts_folder) if f.endswith('.nii.gz')]
    # Extract unique case IDs by stripping the '_000x.nii.gz' suffix
    case_ids = sorted(list(set([f.rsplit('_', 1)[0] for f in all_files])))

    # 4. Iterate through your Case IDs (Not individual files)
    for case_id in case_ids:
        print(f"\nProcessing MC Dropout for Case: {case_id}")
        
        # Collect all modality channels for this strict case (e.g. _0000.nii.gz and _0001.nii.gz)
        modality_files = sorted([os.path.join(img_ts_folder, f) for f in all_files if f.startswith(f"{case_id}_")])
        print(f"  -> Found {len(modality_files)} channels: {[os.path.basename(m) for m in modality_files]}")

        num_mc_iterations = 5  
        all_pass_probs = []
        
        for i in range(num_mc_iterations):
            print(f"  -> Running MC Pass {i+1}/{num_mc_iterations}...")
            
            # Predict returns a list of tuples: [(mask, probability_map)]
            # We pass a single list of all modality paths: [[chan_0, chan_1, ...]]
            probs = predictor.predict_from_files(
                [modality_files], 
                None, 
                save_probabilities=True, 
                overwrite=True,
                num_processes_preprocessing=1, 
                num_processes_segmentation_export=1
            )
            
            pred_mask, pred_prob = probs[0]
            all_pass_probs.append(pred_prob) 

        # 5. Calculate Metrics
        print("  -> Calculating Uncertainty Maps...")
        all_pass_probs = np.array(all_pass_probs) # Shape: [N_mc, num_classes, X, Y, Z]
        
        mean_prob = np.mean(all_pass_probs, axis=0)
        final_mask = np.argmax(mean_prob, axis=0)
        
        uncertainty_map = np.var(all_pass_probs, axis=0).sum(axis=0)

        # 6. Save Results using the affine from the first channel image
        print("  -> Saving NIfTI files...")
        affine = nib.load(modality_files[0]).affine
        
        nib.save(nib.Nifti1Image(final_mask.astype(np.uint8), affine), 
                 os.path.join(output_folder, f"{case_id}_pred.nii.gz"))
        
        nib.save(nib.Nifti1Image(uncertainty_map.astype(np.float32), affine), 
                 os.path.join(output_folder, f"{case_id}_uncertainty.nii.gz"))

        break # Remove break to process all cases in folder
            
    print("MC Dropout Inference Complete!")

if __name__ == '__main__':
    main()