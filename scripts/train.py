import sys
from pathlib import Path

# Support both `python scripts/train.py` and `python -m scripts.train`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.loaders import build_ssl_loader
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scripts.pretrain import train_one_epoch
from src.models.NT_Xent_loss import NTXentLoss
from src.models.simclr import SimCLRModel


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    amp_enabled = device.type == "cuda"

    print(f"Using device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    ssl_loader = build_ssl_loader(
        batch_size=128,
        num_workers=0,
    )

    model = SimCLRModel(
        projection_hidden_dim=512,
        projection_output_dim=128,
    ).to(device)

    criterion = NTXentLoss(
        temperature=0.2,
        reduction="mean",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    epochs = 1

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model=model,
            loader=ssl_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            amp_enabled=amp_enabled,
            max_steps=100,  # smoke test
            log_interval=10,
        )

        scheduler.step()

        print(
            f"Epoch {epoch} completed: "
            f"loss={metrics['loss']:.4f}, "
            f"positive_similarity="
            f"{metrics['positive_similarity']:.4f}"
        )
        output_directory = Path("outputs/smoke/checkpoints")
        output_directory.mkdir(parents=True,exist_ok=True,)

        checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "metrics": metrics,
        }

        torch.save(
        checkpoint,
        output_directory / "latest.pth",
        
        )
        torch.save(
            model.encoder.state_dict(),
            output_directory / "encoder.pth",   )
if __name__ == "__main__":
    main()
