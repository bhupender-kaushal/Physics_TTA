import torch
import numpy as np
from scipy.ndimage import shift

class MRIPerformanceTransform:
    def __init__(self, p_per_sample=0.3, spatial_dims=(0, 1)):
        """
        Args:
            p_per_sample: Probability of applying the transform per batch sample.
            spatial_dims: The spatial axes representing the in-plane MRI acquisition 
                          (e.g., (0, 1) for X,Y in a (X, Y, Z) 3D patch).
        """
        self.p_per_sample = p_per_sample
        # The axes within the spatial domain where the 2D FFT should be applied.
        self.spatial_dims = spatial_dims 
        self.physics_types = ['motion_and_breathing', 'breathing', 'motion_and_breathing']

    def __call__(self, **data_dict):
        data = data_dict.get('image', data_dict.get('data', None))
        if data is None:
            return data_dict
            
        img_key = 'image' if 'image' in data_dict else 'data'
        seg_key = 'segmentation' if 'segmentation' in data_dict else ('seg' if 'seg' in data_dict else None)
        seg = data_dict[seg_key] if seg_key else None

        # --- 1. SAFELY CONVERT TO NUMPY ---
        is_data_tensor = isinstance(data, torch.Tensor)
        device = data.device if is_data_tensor else None
        if is_data_tensor: data = data.cpu().numpy()
            
        is_seg_tensor = isinstance(seg, torch.Tensor)
        is_seg_list = isinstance(seg, (list, tuple))
         
        if is_seg_tensor: 
            seg = seg.cpu().numpy()
        elif is_seg_list: 
            seg = [s.cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s) for s in seg]

        # nnUNet typically provides (B, C, X, Y, [Z])
        # If no batch dim exists (e.g., during single validation inference), wrap it.
        has_batch = data.ndim in [4, 5] 
        B = data.shape[0] if has_batch else 1

        for b in range(B):
            if np.random.uniform() < self.p_per_sample:
                aug_type = np.random.choice(self.physics_types)
                
                # MATHEMATICAL FIX 1: Enforce strict integer shifts to prevent 
                # sub-pixel misalignment between continuous k-space and discrete masks.
                dy = float(np.random.randint(-10, 10))
                dx = float(np.random.randint(-10, 10))
                
                sample_img = data[b] if has_batch else data
                C = sample_img.shape[0]
                spatial_shape = sample_img.shape[1:]
                
                is_3d = len(spatial_shape) == 3
                
                # Define in-plane dimensions based on user input (default assumes first two spatial dims)
                dim_y, dim_x = self.spatial_dims
                H = spatial_shape[dim_y]
                W = spatial_shape[dim_x]

                # Setup k-space grid
                u = np.linspace(-0.5, 0.5, H, endpoint=False).reshape(-1, 1)
                v = np.linspace(-0.5, 0.5, W, endpoint=False).reshape(1, -1)
                
                if aug_type == 'motion': phase_ramp = -2 * np.pi * (u * dy + v * dx)
                elif aug_type == 'breathing': phase_ramp = -2 * np.pi * (np.sin(u * dy + v * dx))
                elif aug_type == 'motion_and_breathing': phase_ramp = -2 * np.pi * (u * dy + v * dx + np.sin(u * dy + v * dx))
                else: phase_ramp = 0

                phase_multiplier = np.exp(1j * phase_ramp)

                # --- UPDATE IMAGES ---
                for c in range(C):
                    if is_3d:
                        # Iterate through the axis NOT in spatial_dims (the slice direction)
                        slice_axis = [i for i in range(3) if i not in self.spatial_dims][0]
                        for z in range(spatial_shape[slice_axis]):
                            # Extract 2D plane correctly
                            indices = [slice(None)] * 3
                            indices[slice_axis] = z
                            img_slice = sample_img[c][tuple(indices)]
                            
                            k_space = np.fft.fftshift(np.fft.fft2(img_slice))
                            aug_slice = np.abs(np.fft.ifft2(np.fft.ifftshift(k_space * phase_multiplier)))
                            sample_img[c][tuple(indices)] = aug_slice.astype(np.float32)
                    else:
                        img_slice = sample_img[c]
                        k_space = np.fft.fftshift(np.fft.fft2(img_slice))
                        aug_slice = np.abs(np.fft.ifft2(np.fft.ifftshift(k_space * phase_multiplier)))
                        sample_img[c] = aug_slice.astype(np.float32)

                if has_batch: data[b] = sample_img
                else: data = sample_img

                # --- UPDATE GROUND TRUTH MASKS ---
                if seg is not None and aug_type in ['motion', 'motion_and_breathing']:
                    seg_items = seg if is_seg_list else [seg]
                    
                    # Map dy, dx to the correct axes in the scipy shift tuple
                    shift_tuple = [0.0] * len(spatial_shape)
                    shift_tuple[dim_y] = dy
                    shift_tuple[dim_x] = dx
                    
                    for s_idx in range(len(seg_items)):
                        sample_seg = seg_items[s_idx][b] if has_batch else seg_items[s_idx]
                        seg_C = sample_seg.shape[0]
                        
                        for c in range(seg_C):
                            # MATHEMATICAL FIX 2: mode='constant', cval=0.0 prevents infinite boundary bleed
                            sample_seg[c] = shift(sample_seg[c], shift=tuple(shift_tuple), order=0, mode='constant', cval=0.0)
                            
                        if has_batch: seg_items[s_idx][b] = sample_seg
                        else: seg_items[s_idx] = sample_seg
                    
                    if not is_seg_list: seg = seg_items[0]

        # --- 3. REVERT TO APPROPRIATE TYPES ---
        if is_data_tensor:
            data = torch.from_numpy(data).to(device)
            
        if seg is not None:
            if is_seg_tensor:
                seg = torch.from_numpy(seg).to(device)
            elif is_seg_list:
                seg = [torch.from_numpy(s).to(device) if isinstance(s, np.ndarray) else s for s in seg]

        data_dict[img_key] = data
        if seg_key:
            data_dict[seg_key] = seg
            
        return data_dict

class CTPerformanceTransform:
    def __init__(self, p_per_sample=0.3, spatial_dims=(0, 1)):
        """
        Args:
            p_per_sample: Probability of applying the transform per batch sample.
            spatial_dims: The spatial axes representing the in-plane CT acquisition 
                          (e.g., (0, 1) for X,Y in a (X, Y, Z) 3D patch).
        """
        self.p_per_sample = p_per_sample
        self.spatial_dims = spatial_dims 
        self.physics_types = ['motion', 'breathing', 'motion_and_breathing']

    def __call__(self, **data_dict):
        from skimage.transform import radon, iradon
        
        data = data_dict.get('image', data_dict.get('data', None))
        if data is None:
            return data_dict
            
        img_key = 'image' if 'image' in data_dict else 'data'
        seg_key = 'segmentation' if 'segmentation' in data_dict else ('seg' if 'seg' in data_dict else None)
        seg = data_dict[seg_key] if seg_key else None

        # --- 1. SAFELY CONVERT TO NUMPY ---
        is_data_tensor = isinstance(data, torch.Tensor)
        device = data.device if is_data_tensor else None
        if is_data_tensor: data = data.cpu().numpy()
            
        is_seg_tensor = isinstance(seg, torch.Tensor)
        is_seg_list = isinstance(seg, (list, tuple))
        
        if is_seg_tensor: 
            seg = seg.cpu().numpy()
        elif is_seg_list: 
            seg = [s.cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s) for s in seg]

        has_batch = data.ndim in [4, 5] 
        B = data.shape[0] if has_batch else 1

        for b in range(B):
            if np.random.uniform() < self.p_per_sample:
                aug_type = np.random.choice(self.physics_types)
                
                dy_pixels = float(np.random.randint(2, 10))
                dx_pixels = float(np.random.randint(2, 10))
                
                sample_img = data[b] if has_batch else data
                C = sample_img.shape[0]
                spatial_shape = sample_img.shape[1:]
                is_3d = len(spatial_shape) == 3
                
                dim_y, dim_x = self.spatial_dims
                H = spatial_shape[dim_y]
                W = spatial_shape[dim_x]

                # --- UPDATE IMAGES (CT PHYSICS) ---
                for c in range(C):
                    if is_3d:
                        slice_axis = [i for i in range(3) if i not in self.spatial_dims][0]
                        for z in range(spatial_shape[slice_axis]):
                            indices = [slice(None)] * 3
                            indices[slice_axis] = z
                            img_slice = sample_img[c][tuple(indices)]
                            
                            aug_slice = self._apply_ct_physics(img_slice, aug_type, dy_pixels, dx_pixels, radon, iradon)
                            sample_img[c][tuple(indices)] = aug_slice.astype(np.float32)
                    else:
                        img_slice = sample_img[c]
                        aug_slice = self._apply_ct_physics(img_slice, aug_type, dy_pixels, dx_pixels, radon, iradon)
                        sample_img[c] = aug_slice.astype(np.float32)

                if has_batch: data[b] = sample_img
                else: data = sample_img
                
                # NOTE: For CT, physical motion introduces artifacts (streaks/blur) but the underlying 
                # reconstructed anatomy doesn't rigidly translate like MRI K-space phase ramps.
                # Therefore, we DO NOT shift the Ground Truth (`seg`) discrete mask for CT.

        # --- 3. REVERT TO APPROPRIATE TYPES ---
        if is_data_tensor:
            data = torch.from_numpy(data).to(device)
            
        if seg is not None:
            if is_seg_tensor:
                seg = torch.from_numpy(seg).to(device)
            elif is_seg_list:
                seg = [torch.from_numpy(s).to(device) if isinstance(s, np.ndarray) else s for s in seg]

        data_dict[img_key] = data
        if seg_key:
            data_dict[seg_key] = seg
            
        return data_dict

    def _apply_ct_physics(self, image, aug_type, dy_pixels, dx_pixels, radon, iradon):
        H, W = image.shape
        image_norm = image.astype(np.float32)

        # 1. Forward Physics: Radon Transform to Sinogram
        theta = np.linspace(0., 180., max(H, W), endpoint=False)
        sinogram = radon(image_norm, theta=theta, circle=False)
        
        R_dim, Theta_dim = sinogram.shape
        sinogram_aug = np.zeros_like(sinogram)
        
        # 2. Base Physical Augmentations
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
        else:
            shifts = np.zeros(Theta_dim, dtype=int)

        for i in range(Theta_dim):
            sinogram_aug[:, i] = np.roll(sinogram[:, i], shifts[i])

        # 3. Inverse Physics: Filtered Back Projection on augmented Sinogram
        image_aug = iradon(sinogram_aug, theta=theta, circle=False).astype(np.float32)
        
        if image_aug.shape != (H, W):
            out_H, out_W = image_aug.shape
            crop_top = (out_H - H) // 2
            crop_left = (out_W - W) // 2
            image_aug = image_aug[crop_top:crop_top+H, crop_left:crop_left+W]
        
        return image_aug