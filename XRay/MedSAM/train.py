# -*- coding: utf-8 -*-
"""
Train the MedSAM mask decoder on X-Ray Data with MC Dropout Injection
Reads directly from Raw Images and Masks folders. Extracts Dual Bounding Boxes.
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage.transform import resize
from scipy.ndimage import label
import SimpleITK as sitk

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import monai
from segment_anything import sam_model_registry
import argparse
import random
from datetime import datetime
import shutil

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"  
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "6"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "6"  

class RawImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, bbox_shift=20, target_size=(1024, 1024)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bbox_shift = bbox_shift
        self.target_size = target_size
        
        self.files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(join(image_dir, f))])
        self.valid_files = [f for f in self.files if os.path.isfile(join(mask_dir, f))]
        print(f"Found {len(self.valid_files)} valid matching image-mask pairs.")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, index):
        filename = self.valid_files[index]
        
        # Read and prepare Image
        img_np = sitk.GetArrayFromImage(sitk.ReadImage(join(self.image_dir, filename)))
        if len(img_np.shape) == 3 and img_np.shape[0] == 1: 
            img_np = img_np[0]
            
        # Read and prepare Mask
        mask_np = sitk.GetArrayFromImage(sitk.ReadImage(join(self.mask_dir, filename)))
        if len(mask_np.shape) == 3 and mask_np.shape[0] == 1: 
            mask_np = mask_np[0]
            
        if len(img_np.shape) == 2: 
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else: 
            img_3c = img_np
            
        # Normalize directly to 0-255 RGB scaling
        img_p1, img_p99 = np.percentile(img_3c[img_3c > 0], [1, 99]) if np.max(img_3c)>0 else (0, 1)
        img_3c = np.clip(img_3c, img_p1, img_p99)
        img_3c = (img_3c - img_3c.min()) / (img_3c.max() - img_3c.min() + 1e-8)
        img_3c = (img_3c * 255).astype(np.uint8)
        
        # Resize to 1024x1024
        img_1024 = resize(img_3c, self.target_size, order=1, preserve_range=True, anti_aliasing=True)
        mask_1024 = resize(mask_np, self.target_size, order=0, preserve_range=True, anti_aliasing=False)
        
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        gt2D = np.uint8(mask_1024 > 0)
        
        # --- DUAL BOUNDING BOX LOGIC ---
        labeled_mask, num_features = label(gt2D > 0)
        bboxes = []
        
        if num_features > 0:
            # Find the two largest structures (lungs)
            areas = [(i, np.sum(labeled_mask == i)) for i in range(1, num_features + 1)]
            areas.sort(key=lambda x: x[1], reverse=True)
            top_labels = [x[0] for x in areas[:2]]
            
            H, W = gt2D.shape
            for lbl in top_labels:
                y_indices, x_indices = np.where(labeled_mask == lbl)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # Apply Random Jitter/augmentation to EACH box
                x_min = max(0, x_min - random.randint(0, self.bbox_shift))
                x_max = min(W, x_max + random.randint(0, self.bbox_shift))
                y_min = max(0, y_min - random.randint(0, self.bbox_shift))
                y_max = min(H, y_max + random.randint(0, self.bbox_shift))
                
                bboxes.append([x_min, y_min, x_max, y_max])
                
            # If the image only has 1 lung visible, duplicate the box so tensor shapes strictly match (2, 4)
            if len(bboxes) == 1:
                bboxes.append(bboxes[0])
        else:
            bboxes = [[0, 0, self.target_size[1], self.target_size[0]]] * 2
            
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(np.array(bboxes)).float(), # Shape will explicitly be (2, 4)
            filename,
        )

# --- DROPOUT INJECTION FUNCTION ---
def inject_dropout_into_decoder(module, p=0.1):
    """Recursively search for Linear layers and wrap them in Sequential with Dropout."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Replace the Linear layer with a Sequential block containing the Linear + Dropout
            setattr(module, name, nn.Sequential(child, nn.Dropout(p=p)))
        else:
            inject_dropout_into_decoder(child, p)
class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
        # Explicitly entirely freeze both encoders to massively drop VRAM and only train mask_decoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        # image: shape (B, 3, 1024, 1024)
        # box: shape (B, 2, 4) -> 2 boxes per image

        B_in, N_boxes, _ = box.shape

        with torch.no_grad():
            # 1. Get image embeddings normally
            image_embedding = self.image_encoder(image) # shape: (B, 256, 64, 64)
            
            # 2. Duplicate image embeddings to match the number of boxes
            # shape changes to: (B, N, 256, 64, 64) -> (B * N, 256, 64, 64)
            image_embedding = image_embedding.unsqueeze(1).repeat(1, N_boxes, 1, 1, 1)
            image_embedding = image_embedding.view(-1, 256, image_embedding.shape[3], image_embedding.shape[4])

            # 3. Reshape boxes to match the virtually expanded batch size
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            box_torch = box_torch.view(-1, 1, 4)  # shape: (B * N, 1, 4)
            
            # Predict boxes individually within the expanded batch
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None,
            )
            
        # 4. Run Decoder (Which now has Active Dropout!))
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  
            image_pe=self.prompt_encoder.get_dense_pe(),  
            sparse_prompt_embeddings=sparse_embeddings,  
            dense_prompt_embeddings=dense_embeddings,  
            multimask_output=False,
        ) # Returns: (B * N, 1, 256, 256)
        
        # 5. Bring back to original shape: (B, N, 1, 256, 256)
        low_res_masks = low_res_masks.view(B_in, N_boxes, 1, low_res_masks.shape[-2], low_res_masks.shape[-1])
        
        # --- COMBINE DUAL PREDICTIONS ---
        # Take the max across the N_boxes dimension (dim=1) to fuse them into a single segmentation
        low_res_masks, _ = torch.max(low_res_masks, dim=1) # Result: (B, 1, 256, 256)
            
        ori_res_masks = F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]), 
            mode="bilinear", align_corners=False,
        )
        return ori_res_masks

    # --- ADD THIS SIMPLE BATCH DICE FUNCTION ---
def compute_batch_dice(pred_logits, gt_mask):
    """Calculate hard dice score for a batch of logits vs ground truth"""
    pred_binary = (torch.sigmoid(pred_logits) > 0.5).float()
    gt_binary = gt_mask.float()
    
    intersection = torch.sum(pred_binary * gt_binary)
    union = torch.sum(pred_binary) + torch.sum(gt_binary)
    
    if union == 0:
        return 1.0
    return (2.0 * intersection / union).item()
# -------------------------------------------



def main():
    # %% set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir", type=str, default='/bhupender/Physics_Uncert/XRay/MedSAM/Data/Images', help="path to training images directory")
    parser.add_argument("-g", "--mask_dir", type=str, default='/bhupender/Physics_Uncert/XRay/MedSAM/Data/Masks', help="path to training Ground Truth masks directory")
    parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B-Drop")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    parser.add_argument("-num_epochs", type=int, default=200)
    parser.add_argument("-batch_size", type=int, default=2)
    parser.add_argument("-num_workers", type=int, default=4)
    parser.add_argument("-weight_decay", type=float, default=0.01)
    parser.add_argument("-lr", type=float, default=0.0001, metavar="LR")
    parser.add_argument("-use_amp", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("-dropout_rate", type=float, default=0.05, help="Dropout probability for Mask Decoder")

    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    device = torch.device(args.device)

    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, join(model_save_path, run_id + "_" + os.path.basename(__file__)))

    print("Building Model...")
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    # --- INJECT DROPOUT BEFORE TRAINING ---
    print(f"Injecting Dropout (p={args.dropout_rate}) into strictly the Mask Decoder...")
    inject_dropout_into_decoder(medsam_model.mask_decoder, p=args.dropout_rate)
    
    medsam_model.train()

    print("Number of total parameters: ", sum(p.numel() for p in medsam_model.parameters()))
    print("Number of trainable parameters: ", sum(p.numel() for p in medsam_model.parameters() if p.requires_grad))

    # ONLY optimize the mask_decoder's weights
    optimizer = torch.optim.AdamW(medsam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    # %% train setup
    train_dataset = RawImageDataset(args.image_dir, args.mask_dir)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
            
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    losses = []
    dices = []
    best_dice = 0.0  
    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0
        epoch_dice = 0
        
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            
            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            
            # <--- Calculate and add Batch Dice Score --->
            with torch.no_grad():
                batch_dice_score = compute_batch_dice(medsam_pred, gt2D)
                epoch_dice += batch_dice_score

        epoch_loss /= (step + 1)
        epoch_dice /= (step + 1)
        
        losses.append(epoch_loss)
        dices.append(epoch_dice)
            
        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}')
        
        checkpoint_data = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        # Save latest
        torch.save(checkpoint_data, join(model_save_path, "medsam_model_latest.pth"))
        
        # Save best based on max DICE score
        if epoch_dice > best_dice:
            best_dice = epoch_dice
            torch.save(checkpoint_data, join(model_save_path, "medsam_model_best.pth"))
            print(f"  -> Saved new best model with Dice: {best_dice:.4f}")

        # Plot Loss & Dice Curve
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        ax1.plot(losses, 'g-')
        ax2.plot(dices, 'b-')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='g')
        ax2.set_ylabel('Dice Score', color='b')
        
        plt.title("Training Loss vs Dice Score")
        plt.savefig(join(model_save_path, "train_metrics.png"))
        plt.close()

if __name__ == "__main__":
    main()