from typing import Union, List, Tuple
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.custom_transforms.custom_physics_transforms import MRIPerformanceTransform
import numpy as np

class nnUNetTrainerDropoutPhysics(nnUNetTrainer):
    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        conv_op_str = str(arch_init_kwargs.get('conv_op', ''))
        is_2d = '2d' in conv_op_str.lower()
            
        dropout_class = "Dropout2d" if is_2d else "Dropout3d"
        arch_init_kwargs['dropout_op'] = f"torch.nn.modules.dropout.{dropout_class}"
        arch_init_kwargs['dropout_op_kwargs'] = {'p': 0.05, 'inplace': True}
        

        print(f"initializing architecture with dropout_op: {arch_init_kwargs['dropout_op']} and kwargs: {arch_init_kwargs['dropout_op_kwargs']}")
        req_imports = list(arch_init_kwargs_req_import)
        if "dropout_op" not in req_imports:
            req_imports.append("dropout_op")
        
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            tuple(req_imports),
            num_input_channels,
            num_output_channels,
            enable_deep_supervision
        )

    def get_training_transforms(self,
                                patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: float,
                                deep_supervision_scales: Union[List, Tuple, None],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None):
        
        # 1. Generate the standard pipeline from the base class
        tr_transforms = super().get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label
        )

        if tr_transforms is not None:
            # 2. Determine patch dimensionality
            is_3d = len(patch_size) == 3
            
            # nnUNet 3D patches are (Z, Y, X). If acquisition is axial, in-plane is Y, X (indices 1, 2).
            # If 2D training, patch is (Y, X) (indices 0, 1).
            spatial_dims = (1, 2) if is_3d else (0, 1)
            
            physics_transform = MRIPerformanceTransform(
                p_per_sample=0.4, # Set your desired probability
                spatial_dims=spatial_dims
            )
            
            # 3. Inject at index 0. 
            # Note: If do_dummy_2d_data_aug is True, index 0 is Convert3DTo2DTransform.
            # Inserting at 0 means MRIPerformanceTransform operates on the 3D tensor before it is flattened to 2D.
            tr_transforms.transforms.insert(0, physics_transform)
        
        return tr_transforms