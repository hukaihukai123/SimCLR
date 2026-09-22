import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from src.augmentations import build_simclr_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_ssl_loader(
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Build the two-view CIFAR-10 loader from the fixed SSL split."""
    split_path = PROJECT_ROOT / "splits" / f"cifar10_seed{seed}.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. "
            "Run `python -m src.datasets.create_splits` first."
        )

    with split_path.open("r", encoding="utf-8") as file:
        split = json.load(file)

    dataset = CIFAR10(
        root=str(PROJECT_ROOT / "data" / "raw"),
        train=True,
        download=False,
        transform=build_simclr_transform(),
    )
    subset = Subset(dataset, split["ssl_train"])

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
