"""Train and validate CIFAR-10 scratch, linear-probe, or fine-tune baselines."""

import argparse
import csv
import json
import pickle
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

from src.datasets import build_labeled_loader
from src.models import EncoderClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-10 classifier"
    )

    parser.add_argument(
        "--mode",
        choices=("scratch", "linear_probe", "fine_tune"),
        required=True,
    )
    parser.add_argument(
        "--pretrained-encoder",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--label-percentage",
        choices=("1_percent", "10_percent", "100_percent"),
        default="10_percent",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.mode in {"linear_probe", "fine_tune"}:
        if args.pretrained_encoder is None:
            parser.error(
                "--pretrained-encoder is required "
                f"when --mode={args.mode}"
            )
    if args.lr is None:
        args.lr = 1e-3 if args.mode == "linear_probe" else 3e-4

    positive_arguments = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "val-batch-size": args.val_batch_size,
        "lr": args.lr,
        "encoder-lr": args.encoder_lr,
        "classifier-lr": args.classifier_lr,
    }
    for name, value in positive_arguments.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")

    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_encoder(
    model: EncoderClassifier,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except pickle.UnpicklingError:
        # Full checkpoints created by scripts/train.py also contain NumPy RNG
        # state, which PyTorch's restricted weights-only loader rejects. These
        # are local training artifacts, so retry with the regular loader.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint).__name__}"
        )

    # Accept the standalone encoder.pth written by scripts/train.py, as well
    # as a complete SimCLR checkpoint containing the model state dictionary.
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")

    if any(key.startswith("encoder.") for key in state):
        state = {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        }

    if not state:
        raise ValueError(f"No encoder weights found in {checkpoint_path}")

    model.encoder.load_state_dict(state, strict=True)


def configure_training_mode(
    model: EncoderClassifier,
    mode: str,
) -> None:
    if mode in {"scratch", "fine_tune"}:
        for parameter in model.parameters():
            parameter.requires_grad = True

    elif mode == "linear_probe":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True


def train_one_epoch(
    model: EncoderClassifier,
    loader,
    criterion,
    optimizer,
    scaler,
    device: torch.device,
    mode: str,
) -> dict[str, float]:
    model.train()

    # 严格Linear Probe：
    # encoder的BatchNorm也不能更新running statistics。
    if mode == "linear_probe":
        model.encoder.eval()

    amp_enabled = device.type == "cuda"

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite classification loss: {loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.inference_mode()
def evaluate(
    model: EncoderClassifier,
    loader,
    criterion,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    confusion_matrix = torch.zeros(
        10,
        10,
        dtype=torch.long,
    )

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = images.shape[0]

        total_loss += loss.item() * batch_size
        total_correct += (
            predictions == labels
        ).sum().item()
        total_samples += batch_size

        encoded = labels * 10 + predictions

        confusion_matrix += torch.bincount(
            encoded.cpu(),
            minlength=100,
        ).reshape(10, 10)

    true_positive = confusion_matrix.diag().float()
    false_positive = (
        confusion_matrix.sum(dim=0).float()
        - true_positive
    )
    false_negative = (
        confusion_matrix.sum(dim=1).float()
        - true_positive
    )

    precision = true_positive / (
        true_positive + false_positive
    ).clamp_min(1)

    recall = true_positive / (
        true_positive + false_negative
    ).clamp_min(1)

    class_f1 = (
        2 * precision * recall
        / (precision + recall).clamp_min(1e-12)
    )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_f1": class_f1.mean().item(),
        "confusion_matrix": confusion_matrix.tolist(),
    }


def append_log(
    path: Path,
    row: dict,
) -> None:
    fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
    ]

    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        if not exists:
            writer.writeheader()

        writer.writerow({
            field: row[field]
            for field in fields
        })


def main() -> None:
    args = parse_args()

    seed_everything(args.run_seed)
    device = select_device(args.device)

    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config["pretrained_encoder"] = (
        str(args.pretrained_encoder)
        if args.pretrained_encoder
        else None
    )
    config["device_resolved"] = str(device)

    with (
        output_dir / "config.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Labels: {args.label_percentage}")
    print(f"Output: {output_dir}")

    train_loader = build_labeled_loader(
        split_name=args.label_percentage,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
        run_seed=args.run_seed,
        training=True,
    )

    val_loader = build_labeled_loader(
        split_name="validation",
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
        run_seed=args.run_seed,
        training=False,
    )

    model = EncoderClassifier().to(device)

    if args.mode in {"linear_probe", "fine_tune"}:
        load_encoder(
            model,
            args.pretrained_encoder,
            device,
        )

    configure_training_mode(
        model,
        args.mode,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in trainable_parameters
        ),
    )

    criterion = torch.nn.CrossEntropyLoss()

    if args.mode == "fine_tune":
        optimizer = AdamW(
            [
                {
                    "params": model.encoder.parameters(),
                    "lr": args.encoder_lr,
                    "name": "encoder",
                },
                {
                    "params": model.classifier.parameters(),
                    "lr": args.classifier_lr,
                    "name": "classifier",
                },
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = AdamW(
            trainable_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    best_accuracy = -1.0
    best_epoch = 0
    log_path = output_dir / "train_log.csv"

    # This script does not implement resume. Starting a new run in an existing
    # directory must therefore replace the old log instead of appending epochs
    # from two unrelated runs.
    if log_path.exists():
        log_path.unlink()

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mode=args.mode,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        append_log(
            log_path,
            row,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.2%} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.2%} "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "mode": args.mode,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": config,
        }

        torch.save(
            checkpoint,
            checkpoint_dir / "latest.pth",
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch

            torch.save(
                checkpoint,
                checkpoint_dir / "best.pth",
            )

            with (
                output_dir / "best_metrics.json"
            ).open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "val_accuracy": best_accuracy,
                        "val_macro_f1": val_metrics[
                            "macro_f1"
                        ],
                        "confusion_matrix": val_metrics[
                            "confusion_matrix"
                        ],
                    },
                    file,
                    indent=2,
                )

        scheduler.step()

    print(
        f"Training completed. "
        f"Best epoch={best_epoch}, "
        f"best validation accuracy={best_accuracy:.2%}"
    )


if __name__ == "__main__":
    main()
