import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import OxfordIIITPet


IMAGE_SIZE = 256


def resize_image(img):
    return img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)


def resize_mask(mask):
    return mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)


def process_split(root, split_name, out_split):
    dataset = OxfordIIITPet(
        root=root,
        split=split_name,
        target_types="segmentation",
        download=True,
    )

    out_img_dir = Path(root) / "processed" / out_split / "images"
    out_mask_dir = Path(root) / "processed" / out_split / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {split_name} -> {out_split}, total = {len(dataset)}")

    for idx in tqdm(range(len(dataset))):
        img, trimap = dataset[idx]

        img = resize_image(img.convert("RGB"))
        trimap = resize_mask(trimap)

        trimap_np = np.array(trimap)

        # Oxford trimap values:
        # 1 = foreground animal
        # 2 = border
        # 3 = background
        binary_mask = (trimap_np != 3).astype(np.uint8) * 255

        img.save(out_img_dir / f"{idx:06d}.jpg")
        Image.fromarray(binary_mask).save(out_mask_dir / f"{idx:06d}.png")

    print(f"Saved to: {Path(root) / 'processed' / out_split}")


def main():
    root = "./data/oxford_pet"

    process_split(root, "trainval", "train")
    process_split(root, "test", "val")

    print("Done.")


if __name__ == "__main__":
    main()