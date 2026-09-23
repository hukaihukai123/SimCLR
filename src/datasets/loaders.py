import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from src.augmentations import build_evaluation_transform, build_simclr_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_ssl_loader(batch_size: int = 128, num_workers: int = 0,
                     split_seed: int = 42, run_seed: int = 42) -> DataLoader:
    """Build a reproducible two-view CIFAR-10 loader from a fixed split."""
    split_path = PROJECT_ROOT / "splits" / f"cifar10_seed{split_seed}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}. Run `python -m src.datasets.create_splits` first.")
    with split_path.open("r", encoding="utf-8") as file:
        split = json.load(file)
    dataset = CIFAR10(root=str(PROJECT_ROOT / "data" / "raw"), train=True,
                      download=False, transform=build_simclr_transform())
    generator = torch.Generator().manual_seed(run_seed)
    return DataLoader(
        Subset(dataset, split["ssl_train"]), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=num_workers > 0, worker_init_fn=_seed_worker, generator=generator,
    )


def build_feature_loader(split_name: str, batch_size: int = 512,
                         num_workers: int = 0, split_seed: int = 42) -> DataLoader:
    """Build a deterministic loader for the SSL feature bank or validation set."""
    if split_name not in {"ssl_train", "validation"}:
        raise ValueError("split_name must be 'ssl_train' or 'validation'")
    split_path = PROJECT_ROOT / "splits" / f"cifar10_seed{split_seed}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as file:
        split = json.load(file)
    dataset = CIFAR10(
        root=str(PROJECT_ROOT / "data" / "raw"), train=True, download=False,
        transform=build_evaluation_transform(),
    )
    return DataLoader(
        Subset(dataset, split[split_name]), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False,
        persistent_workers=num_workers > 0,
    )
