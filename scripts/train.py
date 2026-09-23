import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pretrain import train_one_epoch
from src.datasets.loaders import build_ssl_loader
from src.models import NTXentLoss, SimCLRModel


LOG_FIELDS = ["epoch", "loss", "positive_similarity", "negative_similarity",
              "embedding_std", "learning_rate", "epoch_time_seconds", "completed_steps"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimCLR on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pretrain/resnet18_seed42"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "log_interval", "save_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or args.temperature <= 0:
        parser.error("lr and temperature must be positive; weight decay cannot be negative")
    return args


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and PyTorch CUDA build, or pass --device cpu.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def append_metrics(path: Path, metrics: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: metrics[key] for key in LOG_FIELDS})


def train(args: argparse.Namespace) -> None:
    seed_everything(args.run_seed)
    device = select_device(args.device)
    amp_enabled = device.type == "cuda"
    output_directory = args.output_dir.resolve()
    checkpoint_directory = output_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config["resume"] = str(args.resume) if args.resume else None
    config["device_resolved"] = str(device)
    with (output_directory / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"Using device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Output directory: {output_directory}")
    loader = build_ssl_loader(args.batch_size, args.num_workers, args.split_seed, args.run_seed)
    model = SimCLRModel(512, 128).to(device)
    criterion = NTXentLoss(args.temperature, "mean")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 1

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint.get("rng_state"))
        if checkpoint.get("loader_generator_state") is not None:
            loader.generator.set_state(checkpoint["loader_generator_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if start_epoch > args.epochs:
        raise ValueError(f"Checkpoint reached epoch {start_epoch - 1}; --epochs must be at least {start_epoch}.")

    for epoch in range(start_epoch, args.epochs + 1):
        metrics = train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                                  amp_enabled, args.max_steps, args.log_interval)
        if args.max_steps is None:
            scheduler.step()
        metrics = {"epoch": epoch, **metrics}
        append_metrics(output_directory / "train_log.csv", metrics)
        print(f"Epoch {epoch} completed: loss={metrics['loss']:.4f}, "
              f"pos={metrics['positive_similarity']:.4f}, neg={metrics['negative_similarity']:.4f}, "
              f"std={metrics['embedding_std']:.4f}")
        checkpoint = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "metrics": metrics,
            "config": config, "rng_state": capture_rng_state(),
            "loader_generator_state": loader.generator.get_state(),
        }
        torch.save(checkpoint, checkpoint_directory / "latest.pth")
        torch.save(model.encoder.state_dict(), checkpoint_directory / "encoder.pth")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(checkpoint, checkpoint_directory / f"epoch_{epoch:03d}.pth")


if __name__ == "__main__":
    train(parse_args())
