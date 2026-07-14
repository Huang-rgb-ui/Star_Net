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

parser = argparse.ArgumentParser(description='Train Stage 3')
parser.add_argument('--method', type=str, default='star_net_l', help='Model name to train')
parser.add_argument('--seed', type=int, default=42, help='Random seed for experiment') args = parser.parse_args()

MODEL_NAME = args.method
SEED = args.seed

set_random_seed(SEED)

BATCH_SIZE_TOMATO = 4
BATCH_SIZE_REPLAY_STAR = 4
BATCH_SIZE_REPLAY_BASELINE = 8
NUM_WORKERS = 0
EPOCHS = 100

LR_TOMATO = 1e-4
LR_ROUTER = 5e-4
LR_BASELINE = 1e-5

TOMATO_MASK_DIR = r"tomato\masks"
TOMATO_MAT_DIR = r"tomato\aligned_mat"
CORN_MASK_DIR = r"Corn\few_shot_train"
CORN_MAT_DIR = r"Corn\aligned_mat"
GRAPE_MASK_DIR = r"Grape\few_shot_train"
GRAPE_MAT_DIR = r"Grape\aligned_mat"

BASE_EXP_DIR = r'exp'

STAGE2_WEIGHT_PATH = os.path.join(BASE_EXP_DIR, MODEL_NAME, f"stage2_seed_{SEED}", "stage2_corn_routing.pth")

SAVE_DIR = os.path.join(BASE_EXP_DIR, MODEL_NAME, f"stage3_seed_{SEED}")
WEIGHT_NAME = "stage3_ultimate_routing.pth"
CHECKPOINT_NAME = "stage3_checkpoint.pth"

BEST_WEIGHT_PATH = os.path.join(SAVE_DIR, WEIGHT_NAME)
CHECKPOINT_PATH = os.path.join(SAVE_DIR, CHECKPOINT_NAME)

RESUME_TRAINING = True

# ==========================================
# (保留原有的 Dataset 类: PureTomatoDataset, MixedReplayDataset, BaselineSegmentationReplayDataset)
# ==========================================
class PureTomatoDataset(Dataset):
    def __init__(self, mask_dir, mat_dir, crop_size=256):
        self.crop_size, self.images, self.masks = crop_size, [], []
        for mask_path in glob.glob(os.path.join(mask_dir, "*_mask.png")):
            base_name = os.path.basename(mask_path).replace('_mask.png', '')
            mat_path = os.path.join(mat_dir, f"{base_name}.mat")
            if not os.path.exists(mat_path): continue
            with h5py.File(mat_path, 'r') as f:
                self.images.append(torch.from_numpy(np.array(f['cube']).transpose(0, 2, 1).astype(np.float32)))
            mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
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
        return img_t, mask_t, torch.tensor(2, dtype=torch.long)

class MixedReplayDataset(Dataset):
    def __init__(self, grape_mask_dir, grape_mat_dir, corn_mask_dir, corn_mat_dir, crop_size=256):
        self.crop_size, self.images, self.task_ids = crop_size, [], []
        def load_data(mask_dir, mat_dir, task_id):
            for m_path in glob.glob(os.path.join(mask_dir, "*_mask.png")):
                base_name = os.path.basename(m_path).replace('_mask.png', '')
                mat_path_aligned = os.path.join(mat_dir, f"{base_name}_aligned.mat")
                actual = mat_path_aligned if os.path.exists(mat_path_aligned) else os.path.join(mat_dir, f"{base_name}.mat")
                try:
                    with h5py.File(actual, 'r') as f:
                        self.images.append(torch.from_numpy(np.array(f['cube']).transpose(0, 2, 1).astype(np.float32)))
                        self.task_ids.append(task_id)
                except Exception: pass
        load_data(grape_mask_dir, grape_mat_dir, 0)
        load_data(corn_mask_dir, corn_mat_dir, 1)
    def __len__(self): return len(self.images) * 10
    def __getitem__(self, idx):
        img_t, t_id = self.images[idx % len(self.images)], self.task_ids[idx % len(self.images)]
        _, H, W = img_t.shape
        if H > self.crop_size and W > self.crop_size:
            top, left = np.random.randint(0, H - self.crop_size), np.random.randint(0, W - self.crop_size)
            img_t = img_t[:, top:top+self.crop_size, left:left+self.crop_size]
        else: img_t = F.pad(img_t, (0, max(0, self.crop_size - W), 0, max(0, self.crop_size - H)), mode='reflect')
        return img_t, torch.tensor(t_id, dtype=torch.long)

class BaselineSegmentationReplayDataset(Dataset):
    def __init__(self, grape_mask_dir, grape_mat_dir, corn_mask_dir, corn_mat_dir, crop_size=256):
        self.crop_size, self.images, self.masks = crop_size, [], []
        def load_data(mask_dir, mat_dir):
            for m_path in glob.glob(os.path.join(mask_dir, "*_mask.png")):
                base_name = os.path.basename(m_path).replace('_mask.png', '')
                mat_path_aligned = os.path.join(mat_dir, f"{base_name}_aligned.mat")
                actual = mat_path_aligned if os.path.exists(mat_path_aligned) else os.path.join(mat_dir, f"{base_name}.mat")
                try:
                    with h5py.File(actual, 'r') as f:
                        self.images.append(torch.from_numpy(np.array(f['cube']).transpose(0, 2, 1).astype(np.float32)))
                    mask = (cv2.imread(m_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
                    self.masks.append(torch.from_numpy(np.expand_dims(mask, axis=0)))
                except Exception: pass
        load_data(grape_mask_dir, grape_mat_dir)
        load_data(corn_mask_dir, corn_mat_dir)
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

def train_stage3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"Stage 3: Tomato started | Model: [{MODEL_NAME.upper()}] | Seed: {SEED}")
    print("=" * 65)

    tomato_loader = DataLoader(PureTomatoDataset(TOMATO_MASK_DIR, TOMATO_MAT_DIR), batch_size=BATCH_SIZE_TOMATO, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    is_auto_routing = ('star' in MODEL_NAME)

    if is_auto_routing:
        replay_loader = DataLoader(MixedReplayDataset(GRAPE_MASK_DIR, GRAPE_MAT_DIR, CORN_MASK_DIR, CORN_MAT_DIR), batch_size=BATCH_SIZE_REPLAY_STAR, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    else:
        replay_loader = DataLoader(BaselineSegmentationReplayDataset(GRAPE_MASK_DIR, GRAPE_MAT_DIR, CORN_MASK_DIR, CORN_MAT_DIR), batch_size=BATCH_SIZE_REPLAY_BASELINE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)

    replay_iter = cycle(replay_loader)
    model = model_generator(MODEL_NAME).to(device)

    if os.path.exists(STAGE2_WEIGHT_PATH):
        state_dict = torch.load(STAGE2_WEIGHT_PATH, map_location=device)
        if is_auto_routing and 'router.2.weight' in state_dict and 'router.3.weight' not in state_dict:
            state_dict['router.3.weight'] = state_dict.pop('router.2.weight')
            state_dict['router.3.bias'] = state_dict.pop('router.2.bias')
        model.load_state_dict(state_dict, strict=False)

    if is_auto_routing:
        for param in model.encoder.parameters(): param.requires_grad = False
        for param in model.decoders['grape'].parameters(): param.requires_grad = False
        for param in model.decoders['corn'].parameters(): param.requires_grad = False
        gate_params, other_params = [], []
        for name, param in model.decoders['tomato'].named_parameters():
            if 'res_gate' in name: gate_params.append(param)
            else: other_params.append(param)
        optimizer = optim.AdamW([
            {'params': other_params, 'lr': LR_TOMATO},
            {'params': gate_params, 'lr': LR_TOMATO * 10},
            {'params': model.router.parameters(), 'lr': LR_ROUTER}
        ], weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR_BASELINE, weight_decay=1e-4)

    seg_criterion = BCEDiceLoss(dice_weight=0.5)
    cls_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 2.0]).to(device))
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
            model.decoders['corn'].eval()
        else:
            model.train()
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Dropout):
                    module.eval()

        epoch_seg, epoch_cls = 0.0, 0.0

        for (t_imgs, t_masks, t_ids) in tomato_loader:
            t_imgs, t_masks = t_imgs.to(device), t_masks.to(device)
            optimizer.zero_grad()

            if is_auto_routing:
                t_ids = t_ids.to(device)
                r_imgs, r_ids = next(replay_iter)
                r_imgs, r_ids = r_imgs.to(device), r_ids.to(device)

                with torch.no_grad():
                    tx = model.encoder(t_imgs)
                    rx = model.encoder(r_imgs)

                tomato_mask_logits = model.decoders['tomato'](*tx)
                loss_seg = seg_criterion(tomato_mask_logits, t_masks)

                combined_deep_features = torch.cat([tx[-1], rx[-1]], dim=0)
                combined_labels = torch.cat([t_ids, r_ids], dim=0)
                combined_task_logits = model.router(combined_deep_features)
                loss_cls = cls_criterion(combined_task_logits, combined_labels)

                loss_total = loss_seg + loss_cls
                loss_total.backward()
                epoch_cls += loss_cls.item()
                epoch_seg += loss_seg.item()
            else:
                r_imgs, r_masks = next(replay_iter)
                r_imgs, r_masks = r_imgs.to(device), r_masks.to(device)
                logits_new = model(t_imgs)
                loss_new = seg_criterion(logits_new, t_masks)
                logits_old = model(r_imgs)
                loss_old = seg_criterion(logits_old, r_masks)
                loss_total = loss_new + loss_old
                loss_total.backward()
                epoch_seg += (loss_new.item() + loss_old.item()) / 2.0

            optimizer.step()

        avg_loss = epoch_seg / len(tomato_loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            if is_auto_routing: print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] | Seg(Tomato): {avg_loss:.4f} | Router Cls: {epoch_cls/len(tomato_loader):.4f}")
            else: print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] | Baseline Seg(Mix): {avg_loss:.4f}")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_loss}, CHECKPOINT_PATH)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BEST_WEIGHT_PATH)

if __name__ == "__main__":
    train_stage3()