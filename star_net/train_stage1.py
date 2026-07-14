import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from hsi_dataset import GrapeSourceDataset
import argparse

from Architecture import model_generator
from utils import JointRoutingLoss, BCEDiceLoss, set_random_seed 

parser = argparse.ArgumentParser(description='Train Stage 1')
parser.add_argument('--method', type=str, default='star_net_l', help='Model name to train')
parser.add_argument('--batch', type=int, default=10, help='Batch size')
parser.add_argument('--seed', type=int, default=42, help='Random seed for experiment')  args = parser.parse_args()

MODEL_NAME = args.method
BATCH_SIZE = args.batch
SEED = args.seed

set_random_seed(SEED)

NUM_WORKERS = 4
EPOCHS = 100
LR = 3e-4

DATA_ROOT = r"datasets\hsi"


BASE_EXP_DIR = r'exp'
SAVE_DIR = os.path.join(BASE_EXP_DIR, MODEL_NAME, f"stage1_seed_{SEED}")

WEIGHT_NAME = "stage1_routing_base.pth"
CHECKPOINT_NAME = "stage1_checkpoint.pth"

BEST_WEIGHT_PATH = os.path.join(SAVE_DIR, WEIGHT_NAME)
CHECKPOINT_PATH = os.path.join(SAVE_DIR, CHECKPOINT_NAME)

RESUME_TRAINING = True


# ==========================================

def train_stage1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"Stage 1 | Model: [{MODEL_NAME.upper()}] | Seed: {SEED}")
    print("=" * 65)

    full_dataset = GrapeSourceDataset(DATA_ROOT, crop_size=256)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    print(f"data_split: train: {train_size}  | val:{val_size} ")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = model_generator(MODEL_NAME).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    is_auto_routing = ('star' in MODEL_NAME)
    if is_auto_routing:
        criterion = JointRoutingLoss(lambda_cls=0.1)
    else:
        criterion = BCEDiceLoss(dice_weight=0.5)

    start_epoch = 0
    best_val_loss = float('inf')

    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resume successful！From {start_epoch + 1}  Epoch Resume")

    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss, train_seg, train_cls = 0.0, 0.0, 0.0

        for imgs, masks, task_ids in train_loader:
            imgs, masks, task_ids = imgs.to(device), masks.to(device), task_ids.to(device)
            optimizer.zero_grad()

            if is_auto_routing:
                mask_logits, task_logits = model(imgs, is_training=True, target_task_name='grape')
                loss, l_seg, l_cls = criterion(mask_logits, masks, task_logits, task_ids)
                train_cls += l_cls.item()
            else:
                mask_logits = model(imgs)
                loss = criterion(mask_logits, masks)
                l_seg = loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_seg += l_seg.item()

        avg_t_loss = train_loss / len(train_loader)

        model.eval()
        val_loss, val_seg, val_cls = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, masks, task_ids in val_loader:
                imgs, masks, task_ids = imgs.to(device), masks.to(device), task_ids.to(device)
                if is_auto_routing:
                    mask_logits, task_logits = model(imgs, is_training=True, target_task_name='grape')
                    loss, l_seg, l_cls = criterion(mask_logits, masks, task_logits, task_ids)
                    val_cls += l_cls.item()
                else:
                    mask_logits = model(imgs)
                    loss = criterion(mask_logits, masks)
                    l_seg = loss
                val_loss += loss.item()
                val_seg += l_seg.item()

        avg_v_loss = val_loss / len(val_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            if is_auto_routing:
                print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] | Train Loss: {avg_t_loss:.4f} | Val Loss: {avg_v_loss:.4f}")
            else:
                print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] | Train Loss: {avg_t_loss:.4f} | Val Loss: {avg_v_loss:.4f}")

        torch.save(
            {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
             'best_val_loss': best_val_loss}, CHECKPOINT_PATH)
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            torch.save(model.state_dict(), BEST_WEIGHT_PATH)
            print(f"Loss refresh({best_val_loss:.4f})，saved。")

    print(f"Stage 1 complete！")


if __name__ == "__main__":
    train_stage1()