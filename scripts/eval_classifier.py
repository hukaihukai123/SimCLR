"""Evaluate a trained CIFAR-10 classifier checkpoint on the official test set."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentations import build_evaluation_transform
from src.models import EncoderClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a classifier checkpoint on CIFAR-10 test data"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_test_loader(batch_size: int, num_workers: int) -> DataLoader:
    dataset = CIFAR10(
        root=str(PROJECT_ROOT / "data" / "raw"),
        train=False,
        download=False,
        transform=build_evaluation_transform(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[EncoderClassifier, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a classifier checkpoint containing 'model'")

    model = EncoderClassifier().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def evaluate(model: EncoderClassifier, loader: DataLoader, device: torch.device) -> dict:
    confusion_matrix = torch.zeros(10, 10, dtype=torch.long)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        encoded = labels * 10 + predictions
        confusion_matrix += torch.bincount(
            encoded.cpu(), minlength=100
        ).reshape(10, 10)

    true_positive = confusion_matrix.diag().float()
    precision = true_positive / confusion_matrix.sum(dim=0).float().clamp_min(1)
    recall = true_positive / confusion_matrix.sum(dim=1).float().clamp_min(1)
    class_f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    total = confusion_matrix.sum().item()
    correct = true_positive.sum().item()
    return {
        "test_samples": total,
        "test_accuracy": correct / total,
        "test_macro_f1": class_f1.mean().item(),
        "per_class_accuracy": (
            true_positive / confusion_matrix.sum(dim=1).float().clamp_min(1)
        ).tolist(),
        "confusion_matrix": confusion_matrix.tolist(),
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    loader = build_test_loader(args.batch_size, args.num_workers)
    metrics = evaluate(model, loader, device)

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "mode": checkpoint.get("mode"),
        "config": checkpoint.get("config"),
        **metrics,
    }

    output = args.output or args.checkpoint.parents[1] / "test_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Test samples: {metrics['test_samples']}")
    print(f"Test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"Test Macro-F1: {metrics['test_macro_f1']:.2%}")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
