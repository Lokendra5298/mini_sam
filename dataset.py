import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class MiniSAMDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        image_size=256,
        augment=False,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        self.image_dir = self.root_dir / split / "images"
        self.mask_dir = self.root_dir / split / "masks"

        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")))
        self.mask_paths = sorted(list(self.mask_dir.glob("*.png")))

        assert len(self.image_paths) > 0, f"No images found in {self.image_dir}"
        assert len(self.image_paths) == len(self.mask_paths), "Image/mask count mismatch"

    def __len__(self):
        return len(self.image_paths)

    def load_image_mask(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        return image, mask

    def random_augment(self, image, mask):
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask

    def mask_to_box(self, mask_np):
        ys, xs = np.where(mask_np > 0)

        if len(xs) == 0 or len(ys) == 0:
            return np.array([0, 0, self.image_size - 1, self.image_size - 1], dtype=np.float32)

        x1 = xs.min()
        y1 = ys.min()
        x2 = xs.max()
        y2 = ys.max()

        # Add small random box noise during training
        if self.augment:
            noise = random.randint(0, 10)
            x1 = max(0, x1 - noise)
            y1 = max(0, y1 - noise)
            x2 = min(self.image_size - 1, x2 + noise)
            y2 = min(self.image_size - 1, y2 + noise)

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def mask_to_positive_point(self, mask_np):
        ys, xs = np.where(mask_np > 0)

        if len(xs) == 0 or len(ys) == 0:
            x = self.image_size // 2
            y = self.image_size // 2
        else:
            idx = random.randint(0, len(xs) - 1)
            x = xs[idx]
            y = ys[idx]

        return np.array([[x, y]], dtype=np.float32)

    def __getitem__(self, idx):
        image, mask = self.load_image_mask(idx)
        image, mask = self.random_augment(image, mask)

        image_np = np.array(image).astype(np.float32) / 255.0
        mask_np = np.array(mask)

        mask_bin = (mask_np > 127).astype(np.float32)

        box = self.mask_to_box(mask_bin)
        point = self.mask_to_positive_point(mask_bin)
        point_label = np.array([1], dtype=np.int64)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0).float()

        box_tensor = torch.from_numpy(box).float()
        point_tensor = torch.from_numpy(point).float()
        point_label_tensor = torch.from_numpy(point_label).long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "box": box_tensor,
            "point": point_tensor,
            "point_label": point_label_tensor,
            "image_path": str(self.image_paths[idx]),
            "mask_path": str(self.mask_paths[idx]),
        }
