import torch
import torch.nn.functional as F


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch: int,
    amp_enabled: bool,
    max_steps: int | None = None,
    log_interval: int = 20,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_positive_similarity = 0.0
    completed_steps = 0

    for step, ((view_1, view_2), _) in enumerate(loader, start=1):

        view_1 = view_1.to(
            device,
            non_blocking=True,
        )

        view_2 = view_2.to(
            device,
            non_blocking=True,
        )

        # [B, C, H, W] + [B, C, H, W]
        #              ↓
        #          [2B, C, H, W]
        images = torch.cat(
            [view_1, view_2],
            dim=0,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):

            representations, projections = model(images)

            z_1, z_2 = projections.chunk(2, dim=0)

            loss = criterion(z_1, z_2)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected at epoch={epoch}, "
                f"step={step}: {loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            positive_similarity = F.cosine_similarity(
                z_1.float(),
                z_2.float(),
                dim=1,
            ).mean()

        total_loss += loss.item()
        total_positive_similarity += positive_similarity.item()
        completed_steps += 1

        if step % log_interval == 0:
            average_loss = total_loss / completed_steps
            average_positive_similarity = (
                total_positive_similarity / completed_steps
            )

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch}] "
                f"Step [{step}/{len(loader)}] "
                f"Loss: {average_loss:.4f} "
                f"PosSim: {average_positive_similarity:.4f} "
                f"LR: {current_lr:.6f}"
            )

        if max_steps is not None and step >= max_steps:
            break

    if completed_steps == 0:
        raise RuntimeError("The training loader produced no batches.")

    return {
        "loss": total_loss / completed_steps,
        "positive_similarity": (
            total_positive_similarity / completed_steps
        ),
        "learning_rate": optimizer.param_groups[0]["lr"],
    }






    
