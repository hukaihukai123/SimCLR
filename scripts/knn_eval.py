"""Evaluate a SimCLR encoder with weighted k-nearest neighbours."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import build_feature_loader
from src.models import SimCLRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weighted k-NN evaluation for SimCLR")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compare-random", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.k <= 0 or args.feature_batch_size <= 0 or args.query_batch_size <= 0:
        parser.error("k and batch sizes must be positive")
    if args.temperature <= 0:
        parser.error("temperature must be positive")
    return args


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(model: SimCLRModel, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.encoder.load_state_dict(checkpoint)


@torch.inference_mode()
def extract_features(model: SimCLRModel, loader, device: torch.device):
    model.eval()
    feature_parts = []
    label_parts = []
    amp_enabled = device.type == "cuda"
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            features = model.encode(images)
        feature_parts.append(F.normalize(features.float(), dim=1).cpu())
        label_parts.append(labels.cpu())
    return torch.cat(feature_parts), torch.cat(label_parts)


@torch.inference_mode()
def weighted_knn_accuracy(train_features, train_labels, query_features, query_labels,
                          device: torch.device, k: int, temperature: float,
                          query_batch_size: int, num_classes: int = 10) -> float:
    if k > len(train_features):
        raise ValueError(f"k={k} exceeds feature-bank size {len(train_features)}")
    feature_bank = train_features.to(device)
    feature_labels = train_labels.to(device)
    correct = 0
    for start in range(0, len(query_features), query_batch_size):
        query = query_features[start:start + query_batch_size].to(device)
        similarities = query @ feature_bank.T
        values, indices = similarities.topk(k=k, dim=1)
        neighbour_labels = feature_labels[indices]
        weights = torch.exp(values / temperature)
        scores = torch.zeros(len(query), num_classes, device=device)
        scores.scatter_add_(1, neighbour_labels, weights)
        predictions = scores.argmax(dim=1).cpu()
        correct += (predictions == query_labels[start:start + query_batch_size]).sum().item()
    return correct / len(query_labels)

