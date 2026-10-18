from typing import Union, List, Tuple
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerDropout(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # 1. Determine if 2D or 3D by checking the 'conv_op' string (e.g. 'torch.nn.modules.conv.Conv2d')
        conv_op_str = str(arch_init_kwargs.get('conv_op', ''))
        is_2d = '2d' in conv_op_str.lower()
            
        dropout_class = "Dropout2d" if is_2d else "Dropout3d"
        
        # 2. Inject the import path into the kwargs dictionary as a SINGLE STRING
        arch_init_kwargs['dropout_op'] = f"torch.nn.modules.dropout.{dropout_class}"
        
        # 3. Inject the dropout configurations (e.g., probability p=0.2)


        arch_init_kwargs['dropout_op_kwargs'] = {'p': 0.2, 'inplace': True}
        print(f"Injecting {dropout_class} with 0.8 into the architecture initialization kwargs.")
        
        # 4. Ensure the string key "dropout_op" is registered in the required imports list
        req_imports = list(arch_init_kwargs_req_import)
        if "dropout_op" not in req_imports:
            req_imports.append("dropout_op")
        
        # 5. Pass cleanly modified configurations back to nnU-Net's native builder
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            tuple(req_imports),
            num_input_channels,
            num_output_channels,
            enable_deep_supervision
        )



# from typing import Union, List, Tuple
# from torch import nn
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# from nnunetv2.training.data_augmentation.custom_physics_transforms import MRIPerformanceTransform

# class nnUNetTrainerDropoutPhysics(nnUNetTrainer):###with Physics transforms
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     @staticmethod
#     def build_network_architecture(architecture_class_name: str,
#                                    arch_init_kwargs: dict,
#                                    arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
#                                    num_input_channels: int,
#                                    num_output_channels: int,
#                                    enable_deep_supervision: bool = True) -> nn.Module:
        
#         # 1. Determine if 2D or 3D by checking the 'conv_op' string
#         conv_op_str = str(arch_init_kwargs.get('conv_op', ''))
#         is_2d = '2d' in conv_op_str.lower()
            
#         dropout_class = "Dropout2d" if is_2d else "Dropout3d"
        
#         # 2. Inject the import path into the kwargs dictionary
#         arch_init_kwargs['dropout_op'] = f"torch.nn.modules.dropout.{dropout_class}"
        
#         # 3. Inject the dropout configurations
#         arch_init_kwargs['dropout_op_kwargs'] = {'p': 0.2, 'inplace': True}
#         print(f"Injecting {dropout_class} with p=0.2 into the architecture initialization kwargs.")
        
#         # 4. Ensure the string key "dropout_op" is registered
#         req_imports = list(arch_init_kwargs_req_import)
#         if "dropout_op" not in req_imports:
#             req_imports.append("dropout_op")
        
#         # 5. Pass cleanly modified configurations back to nnU-Net's native builder
#         return nnUNetTrainer.build_network_architecture(
#             architecture_class_name,
#             arch_init_kwargs,
#             tuple(req_imports),
#             num_input_channels,
#             num_output_channels,
#             enable_deep_supervision
#         )

#     def get_training_transforms(self, patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2D_data_aug, order_resampling_data, order_resampling_seg, border_val_seg, use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label):
        
#         # 1. Get the standard default nnUNet augmentations
#         tr_transforms = super().get_training_transforms(
#             patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, 
#             do_dummy_2D_data_aug, order_resampling_data, order_resampling_seg, 
#             border_val_seg, use_mask_for_norm, is_cascaded, foreground_labels, 
#             regions, ignore_label
#         )
        
#         # 2. Append the Multi-Physics Transform + Mask synchronization
#         # It will be applied at a 30% probability per sample 
#         if tr_transforms is not None:
#             tr_transforms.transforms.append(MRIPerformanceTransform(p_per_sample=0.3))
        
#         return tr_transforms