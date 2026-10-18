# import torch
# import numpy as np
# import nibabel as nib
# import os
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# def main():
#     # 1. Setup the Predictor
#     predictor = nnUNetPredictor(
#         tile_step_size=1, # 1 = Fast, 0.5 = High quality
#         use_mirroring=False, 
#         device=torch.device('cuda'),
#         verbose=False
#     )

#     model_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/nnUNet_results/Dataset003_Liver/nnUNetTrainerDropout__nnUNetPlans__2d'
#     img_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/Liver/Images'
#     output_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/uncert_analysis/mc_results'
#     os.makedirs(output_folder, exist_ok=True)

#     predictor.initialize_from_trained_model_folder(
#         model_folder,
#         use_folds=(0,),
#         checkpoint_name='checkpoint_final.pth',
#     )

#     # --- THE MC DROPOUT FIX ---
#     # nnU-Net automatically calls .eval() when predict_from_files starts, which turns off Dropout.
#     # We override the network's eval method to ensure Dropout layers stay in train() mode permanently.
#     original_eval = predictor.network.eval
#     def custom_mcd_eval():
#         original_eval() # Let other layers (like BatchNorm) go into eval mode
#         for m in predictor.network.modules():
#             if type(m) in [torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d]:
#                 m.train() # Force Dropout to stay active
#         return predictor.network
    
#     predictor.network.eval = custom_mcd_eval 

#     # 4. Iterate through your images
#     for file_name in os.listdir(img_ts_folder):
#         if file_name.endswith('.nii.gz'):
#             img_path = os.path.join(img_ts_folder, file_name)
#             print(f"Processing MC Dropout for: {file_name}")
            
#             num_mc_iterations = 2  # Set to desired number of passes
            
#             # --- THE BATCH WORKAROUND ---
#             # Instead of calling inference in a Python loop, we build a list of the exact same image 
#             # path multiple times. nnU-Net's queue will chew through them efficiently as one big job.
#             mc_batch_paths = [[img_path] for _ in range(num_mc_iterations)]
            
#             print(f"  -> Seding {num_mc_iterations} isolated passes to nnU-Net queue...")
#             probs = predictor.predict_from_files(mc_batch_paths, None, save_probabilities=True, overwrite=True)
            
#             # Extract just the probability arrays from the (mask, probability) tuple nnU-Net returns
#             all_pass_probs = np.array([p[1] for p in probs])

#             # 5. Calculate Metrics
#             print("  -> Calculating Uncertainty Maps...")
#             # Mean Probability (Final Prediction)
#             mean_prob = np.mean(all_pass_probs, axis=0)
#             final_mask = np.argmax(mean_prob, axis=0)
            
#             # Variance (Uncertainty Map) - sum across classes
#             uncertainty_map = np.var(all_pass_probs, axis=0).sum(axis=0)

#             # 6. Save Results
#             affine = nib.load(img_path).affine
#             case_id = file_name.replace('_0000.nii.gz', '')
            
#             nib.save(nib.Nifti1Image(final_mask.astype(np.uint8), affine), 
#                      os.path.join(output_folder, f"{case_id}_pred.nii.gz"))
            
#             nib.save(nib.Nifti1Image(uncertainty_map.astype(np.float32), affine), 
#                      os.path.join(output_folder, f"{case_id}_uncertainty.nii.gz"))

#             break # Remove break to process all images
            
#     print("MC Dropout Inference Complete!")

# if __name__ == '__main__':
#     main()


## the previous was code for parallel MC Dropout inference using nnU-Net's queue system. Below is a simpler version that runs MC Dropout in a standard Python loop, which is easier to understand but may be slower for large datasets.

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
    # model_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/nnUNet_results/Dataset003_Liver/nnUNetTrainerDropout__nnUNetPlans__2d'

    # img_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/Liver/Images'
    img_ts_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/datasets/uncert_selected/Prostate/Images'
    output_folder = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/uncert_analysis/mc_results/prostate'
    
    os.makedirs(output_folder, exist_ok=True)

    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )

    # --- THE MC DROPOUT FIX ---
    # nnU-Net automatically calls .eval() when predict_from_files starts, which turns off Dropout.
    # We override the network's eval method to ensure Dropout layers stay in train() mode permanently.
    original_eval = predictor.network.eval
    def custom_mcd_eval():
        original_eval() # Let other layers (like BatchNorm/InstanceNorm) go into eval mode
        for m in predictor.network.modules():
            if type(m) in [torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d]:
                m.train() # Force Dropout to stay active!
        return predictor.network
    
    predictor.network.eval = custom_mcd_eval 

    # 4. Iterate through your images
    for file_name in os.listdir(img_ts_folder):
        if file_name.endswith('.nii.gz'):
            img_path = os.path.join(img_ts_folder, file_name)
            print(f"\nProcessing MC Dropout for: {file_name}")
            
            num_mc_iterations = 5  # Set to desired number of passes
            all_pass_probs = []
            
            for i in range(num_mc_iterations):
                print(f"  -> Running MC Pass {i+1}/{num_mc_iterations}...")
                # Predict returns a list of tuples: [(mask, probability_map)]
                # Passing a single list of lists: [[img_path]]
                probs = predictor.predict_from_files(
                    [[img_path]], 
                    None, 
                    save_probabilities=True, 
                    overwrite=True,
                    # Limit workers to prevent memory crashes on large images
                    num_processes_preprocessing=1, 
                    num_processes_segmentation_export=1
                )
                
                # Extract the probability map (Item 1 in the tuple)
                pred_mask, pred_prob = probs[0]
                all_pass_probs.append(pred_prob) 

            # 5. Calculate Metrics
            print("  -> Calculating Uncertainty Maps...")
            all_pass_probs = np.array(all_pass_probs) # Shape: [N_mc, num_classes, X, Y, Z]
            
            # Mean Probability (Final Prediction)
            mean_prob = np.mean(all_pass_probs, axis=0)
            final_mask = np.argmax(mean_prob, axis=0)
            
            # Variance (Uncertainty Map) - sum across classes
            uncertainty_map = np.var(all_pass_probs, axis=0).sum(axis=0)

            # 6. Save Results
            print("  -> Saving NIfTI files...")
            affine = nib.load(img_path).affine
            case_id = file_name.replace('_0000.nii.gz', '')
            
            nib.save(nib.Nifti1Image(final_mask.astype(np.uint8), affine), 
                     os.path.join(output_folder, f"{case_id}_pred.nii.gz"))
            
            nib.save(nib.Nifti1Image(uncertainty_map.astype(np.float32), affine), 
                     os.path.join(output_folder, f"{case_id}_uncertainty.nii.gz"))

            break # Remove break to process all images in folder
            
    print("MC Dropout Inference Complete!")

if __name__ == '__main__':
    main()