import os
import glob
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import h5py
import cv2
from itertools import cycle
import argparse

from Architecture import model_generator
from utils import BCEDiceLoss, set_random_seed 

parser = argparse.ArgumentParser(description='Train Stage 2')
parser.add_argument('--method', type=str, default='star_net_l', help='Model name to train')
parser.add_argument('--seed', type=int, default=42, help='Random seed for experiment') args = parser.parse_args()

MODEL_NAME = args.method
SEED = args.seed

set_random_seed(SEED)

BATCH_SIZE = 1
BATCH_SIZE_REPLAY_STAR = 4
BATCH_SIZE_REPLAY_BASELINE = 1
NUM_WORKERS = 4
EPOCHS = 300

LR_CORN = 1e-3
LR_ROUTER = 5e-4
LR_BASELINE = 1e-5

CORN_MASK_DIR = r"Corn\few_shot_train"
CORN_MAT_DIR = r"Corn\aligned_mat"
GRAPE_MASK_DIR = r"Grape\few_shot_train"
GRAPE_MAT_DIR = r"Grape\aligned_mat"

BASE_EXP_DIR = r'exp'

STAGE1_WEIGHT_PATH = os.path.join(BASE_EXP_DIR, MODEL_NAME, f"stage1_seed_{SEED}", "stage1_routing_base.pth")

SAVE_DIR = os.path.join(BASE_EXP_DIR, MODEL_NAME, f"stage2_seed_{SEED}")
WEIGHT_NAME = "stage2_corn_routing.pth"
CHECKPOINT_NAME = "stage2_checkpoint.pth"

BEST_WEIGHT_PATH = os.path.join(SAVE_DIR, WEIGHT_NAME)
CHECKPOINT_PATH = os.path.join(SAVE_DIR, CHECKPOINT_NAME)

RESUME_TRAINING = False

class PureCornDataset(Dataset):
    def __init__(self, mask_dir, mat_dir, crop_size=256):
        self.crop_size = crop_size
        self.images, self.masks = [], []
        for mask_path in glob.glob(os.path.join(mask_dir, "*_mask.png")):
            base_name = os.path.basename(mask_path).replace('_mask.png', '')
            mat_path = os.path.join(mat_dir, f"{base_name}_aligned.mat")
            with h5py.File(mat_path, 'r') as f:
                img = np.array(f['cube']).transpose(0, 2, 1).astype(np.float32)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
            self.images.append(torch.from_numpy(img))
            self.masks.append(torch.from_numpy(np.expand_dims(mask, axis=0)))
    def __len__(self): return len(self.images) * 10
    def __getitem__(self, idx):
        img_t, mask_t = self.images[idx % len(self.images)], self.masks[idx % len(self.masks)]
        _, H, W = img_t.shape
        if H > self.crop_size and W > self.crop_size:
            top, left = np.random.randint(0, H - self.crop_size), np.random.randint(0, W - self.crop_size)
            img_t, mask_t = img_t[:, top:top+self.crop_size, left:left+self.crop_size], mask_t[:, top:top+self.crop_size, left:left+self.crop_size]
        else:
            img_t = F.pad(img_t, (0, max(0, self.crop_size - W), 0, max(0, self.crop_size - H)), mode='reflect')
            mask_t = F.pad(mask_t, (0, max(0, self.crop_size - W), 0, max(0, self.crop_size - H)), mode='constant', value=0)
        return img_t, mask_t, torch.tensor(1, dtype=torch.long)

class PureGrapeReplayDataset(Dataset):
    def __init__(self, mask_dir, mat_dir, crop_size=256):
        self.crop_size, self.images = crop_size, []
        for mask_path in glob.glob(os.path.join(mask_dir, "*_mask.png")):
            base_name = os.path.basename(mask_path).replace('_mask.png', '')
            mat_path_aligned = os.path.join(mat_dir, f"{base_name}_aligned.mat")
            actual_mat_path = mat_path_aligned if os.path.exists(mat_path_aligned) else os.path.join(mat_dir, f"{base_name}.mat")
            try:
                with h5py.File(actual_mat_path, 'r') as f:
                    self.images.append(torch.from_numpy(np.array(f['cube']).transpose(0, 2, 1).astype(np.float32)))
            except Exception: pass
    def __len__(self): return len(self.images) * 10
    def __getitem__(self, idx):
        img_t = self.images[idx % len(self.images)]
        _, H, W = img_t.shape
        if H > self.crop_size and W > self.crop_size:
            top, left = np.random.randint(0, H - self.crop_size), np.random.randint(0, W - self.crop_size)
            img_t = img_t[:, top:top+self.crop_size, left:left+self.crop_size]
        else: img_t = F.pad(img_t, (0, max(0, self.crop_size - W), 0, max(0, self.crop_size - H)), mode='reflect')
        return img_t, torch.tensor(0, dtype=torch.long)

class BaselineGrapeReplayDataset(Dataset):
    def __init__(self, mask_dir, mat_dir, crop_size=256):
        self.crop_size, self.images, self.masks = crop_size, [], []
        for mask_path in glob.glob(os.path.join(mask_dir, "*_mask.png")):
            base_name = os.path.basename(mask_path).replace('_mask.png', '')
            mat_path_aligned = os.path.join(mat_dir, f"{base_name}_aligned.mat")
            actual_mat_path = mat_path_aligned if os.path.exists(mat_path_aligned) else os.path.join(mat_dir, f"{base_name}.mat")
            try:
                with h5py.File(actual_mat_path, 'r') as f:
                    self.images.append(torch.from_numpy(np.array(f['cube']).transpose(0, 2, 1).astype(np.float32)))
                mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
                self.masks.append(torch.from_numpy(np.expand_dims(mask, axis=0)))
            except Exception: pass
    def __len__(self): return len(self.images) * 10
    def __getitem__(self, idx):
        img_t, mask_t = self.images[idx % len(self.images)], self.masks[idx % len(self.masks)]
        _, H, W = img_t.shape
        if H > self.crop_size and W > self.crop_size:
            top, left = np.random.randint(0, H - self.crop_size), np.random.randint(0, W - self.crop_size)
            img_t, mask_t = img_t[:, top:top+self.crop_size, left:left+self.crop_size], mask_t[:, top:top+self.crop_size, left:left+self.crop_size]
        else:
            img_t = F.pad(img_t, (0, max(0, self.crop_size - W), 0, max(0, self.crop_size - H)), mode='reflect')
            mask_t = F.pad(mask_t, (0, max(0, self.crop_size - W), 0, max(0, self.crop_size - H)), mode='constant', value=0)
        return img_t, mask_t

# ==========================================

def train_stage2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"Stage 2: Wheat | Model: [{MODEL_NAME.upper()}] | Seed: {SEED}")
    print("=" * 65)

    corn_loader = DataLoader(PureCornDataset(CORN_MASK_DIR, CORN_MAT_DIR), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    is_auto_routing = ('star' in MODEL_NAME)

    if is_auto_routing:
        grape_loader = DataLoader(PureGrapeReplayDataset(GRAPE_MASK_DIR, GRAPE_MAT_DIR), batch_size=BATCH_SIZE_REPLAY_STAR, shuffle=True, pin_memory=True)
    else:
        grape_loader = DataLoader(BaselineGrapeReplayDataset(GRAPE_MASK_DIR, GRAPE_MAT_DIR), batch_size=BATCH_SIZE_REPLAY_BASELINE, shuffle=True, pin_memory=True)

    grape_iter = cycle(grape_loader)
    model = model_generator(MODEL_NAME).to(device)

    if os.path.exists(STAGE1_WEIGHT_PATH):
        state_dict = torch.load(STAGE1_WEIGHT_PATH, map_location=device)
        if is_auto_routing and 'router.2.weight' in state_dict and 'router.3.weight' not in state_dict:
            state_dict['router.3.weight'] = state_dict.pop('router.2.weight')
            state_dict['router.3.bias'] = state_dict.pop('router.2.bias')
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Cannot find Stage 1 pth！")

    if is_auto_routing:
        for param in model.encoder.parameters(): param.requires_grad = False
        for param in model.decoders['grape'].parameters(): param.requires_grad = False
        for param in model.decoders['tomato'].parameters(): param.requires_grad = False
        optimizer = optim.AdamW([
            {'params': model.decoders['corn'].parameters(), 'lr': LR_CORN},
            {'params': model.router.parameters(), 'lr': LR_ROUTER}
        ], weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR_BASELINE, weight_decay=1e-4)

    seg_criterion = BCEDiceLoss(dice_weight=0.5)
    cls_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.0]).to(device))
    start_epoch, best_loss = 0, float('inf')

    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch, best_loss = checkpoint['epoch'] + 1, checkpoint.get('best_loss', float('inf'))

    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(start_epoch, EPOCHS):
        if is_auto_routing:
            model.train()
            model.encoder.eval()
            model.decoders['grape'].eval()
            model.decoders['tomato'].eval()
        else:
            model.train()
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Dropout):
                    module.eval()

        epoch_seg, epoch_cls = 0.0, 0.0

        for (c_imgs, c_masks, c_ids) in corn_loader:
            c_imgs, c_masks, c_ids = c_imgs.to(device), c_masks.to(device), c_ids.to(device)
            optimizer.zero_grad()

            if is_auto_routing:
                g_imgs, g_ids = next(grape_iter)
                g_imgs, g_ids = g_imgs.to(device), g_ids.to(device)

                with torch.no_grad(): cx = model.encoder(c_imgs)
                corn_mask_logits = model.decoders['corn'](*cx)
                corn_task_logits = model.router(cx[-1])
                loss_corn_seg = seg_criterion(corn_mask_logits, c_masks)
                loss_corn_cls = cls_criterion(corn_task_logits, c_ids)

                with torch.no_grad(): gx = model.encoder(g_imgs)
                grape_task_logits = model.router(gx[-1])
                loss_grape_cls = cls_criterion(grape_task_logits, g_ids)

                loss_total = loss_corn_seg + 0.1 * (loss_corn_cls + loss_grape_cls)
                loss_total.backward()
                epoch_cls += (loss_corn_cls.item() + loss_grape_cls.item()) / 2
                epoch_seg += loss_corn_seg.item()
            else:
                g_imgs, g_masks = next(grape_iter)
                g_imgs, g_masks = g_imgs.to(device), g_masks.to(device)
                logits_new = model(c_imgs)
                loss_new = seg_criterion(logits_new, c_masks)
                logits_old = model(g_imgs)
                loss_old = seg_criterion(logits_old, g_masks)
                loss_total = loss_new + loss_old
                loss_total.backward()
                epoch_seg += (loss_new.item() + loss_old.item()) / 2.0

            optimizer.step()

        avg_loss = epoch_seg / len(corn_loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            if is_auto_routing: print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] | Seg(Corn): {avg_loss:.4f} | Cls: {epoch_cls/len(corn_loader):.4f}")
            else: print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] | Baseline Seg(Mix): {avg_loss:.4f}")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_loss}, CHECKPOINT_PATH)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BEST_WEIGHT_PATH)

if __name__ == "__main__":
    train_stage2()