import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import DataLoader

from dataset import MiniSAMDataset


def main():
    dataset = MiniSAMDataset(
        root_dir="data/oxford_pet/processed",
        split="train",
        image_size=256,
        augment=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
    )

    batch = next(iter(loader))

    print("Dataset size:", len(dataset))
    print("Image shape :", batch["image"].shape)
    print("Mask shape  :", batch["mask"].shape)
    print("Box shape   :", batch["box"].shape)
    print("Point shape :", batch["point"].shape)
    print("Label shape :", batch["point_label"].shape)

    print("Box example  :", batch["box"][0])
    print("Point example:", batch["point"][0])
    print("Mask min/max :", batch["mask"].min().item(), batch["mask"].max().item())

    assert batch["image"].shape == (4, 3, 256, 256)
    assert batch["mask"].shape == (4, 1, 256, 256)
    assert batch["box"].shape == (4, 4)
    assert batch["point"].shape == (4, 1, 2)
    assert batch["point_label"].shape == (4, 1)

    print("Dataset test passed.")


if __name__ == "__main__":
    main()
