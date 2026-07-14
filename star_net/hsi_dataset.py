import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2
import h5py



class GrapeSourceDataset(Dataset):

    def __init__(self, root_dir, crop_size=256):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.samples = []

        self.classes = [
            "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)",
            "Grape___healthy",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
        ]

        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_path): continue

            npy_files = glob.glob(os.path.join(cls_path, "*.npy"))
            for npy_path in npy_files:
                base_name = os.path.basename(npy_path).replace('.npy', '')
                mask_path = os.path.join(cls_path, "GT_Masks", f"{base_name}_mask.png")

                is_healthy = "healthy" in cls_name
                self.samples.append({
                    "npy": npy_path,
                    "mask": mask_path if not is_healthy else None,
                    "is_healthy": is_healthy
                })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = np.load(sample["npy"]).astype(np.float32)
        if img.shape[0] > img.shape[2]:  
            img = img.transpose(2, 0, 1)

        C, H, W = img.shape

        if sample["is_healthy"] or sample["mask"] is None or not os.path.exists(sample["mask"]):
            mask = np.zeros((1, H, W), dtype=np.float32)
        else:
            mask = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
            mask = np.expand_dims(mask, axis=0)

        img_t = torch.from_numpy(img)
        mask_t = torch.from_numpy(mask)

        if H > self.crop_size and W > self.crop_size:
            top = np.random.randint(0, H - self.crop_size)
            left = np.random.randint(0, W - self.crop_size)
            img_t = img_t[:, top: top + self.crop_size, left: left + self.crop_size]
            mask_t = mask_t[:, top: top + self.crop_size, left: left + self.crop_size]

        return img_t, mask_t, torch.tensor(0, dtype=torch.long)


class IncrementalReplayDataset(Dataset):

    def __init__(self, new_task_masks, new_task_mat_dir, new_task_id,
                 replay_npys, replay_masks, replay_task_id, crop_size=256):
        self.crop_size = crop_size
        self.images_cache = []
        self.masks_cache = []
        self.task_ids = []

        for mask_path in new_task_masks:
            base_name = os.path.basename(mask_path).replace('_mask.png', '')
            mat_path = os.path.join(new_task_mat_dir, f"{base_name}_aligned.mat")
            with h5py.File(mat_path, 'r') as f:
                img = np.array(f['cube']).transpose(0, 2, 1).astype(np.float32)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
            self.images_cache.append(torch.from_numpy(img))
            self.masks_cache.append(torch.from_numpy(np.expand_dims(mask, axis=0)))
            self.task_ids.append(new_task_id)

        for npy_path, mask_path in zip(replay_npys, replay_masks):
            img = np.load(npy_path).astype(np.float32)
            if img.shape[0] > img.shape[2]: img = img.transpose(2, 0, 1)

            if mask_path is None: 
                mask = np.zeros((1, img.shape[1], img.shape[2]), dtype=np.float32)
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.float32)
                mask = np.expand_dims(mask, axis=0)

            self.images_cache.append(torch.from_numpy(img))
            self.masks_cache.append(torch.from_numpy(mask))
            self.task_ids.append(replay_task_id)


    def __len__(self):
        return len(self.images_cache) * 10  # 虚拟扩充，增加 Epoch 迭代次数

    def __getitem__(self, idx):
        real_idx = idx % len(self.images_cache)
        img_t = self.images_cache[real_idx]
        mask_t = self.masks_cache[real_idx]
        t_id = self.task_ids[real_idx]
        _, H, W = img_t.shape

        if H > self.crop_size and W > self.crop_size:
            top = np.random.randint(0, H - self.crop_size)
            left = np.random.randint(0, W - self.crop_size)
            img_t = img_t[:, top: top + self.crop_size, left: left + self.crop_size]
            mask_t = mask_t[:, top: top + self.crop_size, left: left + self.crop_size]
        else:
            img_t = F.pad(img_t, (0, max(0, self.crop_size - W), 0, max(0, self.crop_size - H)), mode='reflect')
            mask_t = F.pad(mask_t, (0, max(0, self.crop_size - W), 0, max(0, self.crop_size - H)), mode='constant',
                           value=0)

        if np.random.random() > 0.5:
            img_t, mask_t = torch.flip(img_t, dims=[2]), torch.flip(mask_t, dims=[2])
        if np.random.random() > 0.5:
            img_t, mask_t = torch.flip(img_t, dims=[1]), torch.flip(mask_t, dims=[1])

        return img_t, mask_t, torch.tensor(t_id, dtype=torch.long)