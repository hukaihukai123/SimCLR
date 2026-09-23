import time

import torch
import torch.nn.functional as F


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch: int,
                    amp_enabled: bool, max_steps: int | None = None,
                    log_interval: int = 20) -> dict[str, float]:
    model.train()
    started_at = time.perf_counter()
    totals = {"loss": 0.0, "positive": 0.0, "negative": 0.0, "std": 0.0}
    completed_steps = 0

    for step, ((view_1, view_2), _) in enumerate(loader, start=1):
        view_1 = view_1.to(device, non_blocking=True)
        view_2 = view_2.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            _, projections = model(torch.cat([view_1, view_2], dim=0))
            z_1, z_2 = projections.chunk(2, dim=0)
            loss = criterion(z_1, z_2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            normalized_1 = F.normalize(z_1.float(), dim=1)
            normalized_2 = F.normalize(z_2.float(), dim=1)
            positive = (normalized_1 * normalized_2).sum(dim=1).mean()
            similarities = normalized_1 @ normalized_2.T
            negative_mask = ~torch.eye(len(similarities), dtype=torch.bool, device=similarities.device)
            negative = similarities[negative_mask].mean()
            embedding_std = torch.cat([z_1.float(), z_2.float()]).std(dim=0).mean()

        totals["loss"] += loss.item()
        totals["positive"] += positive.item()
        totals["negative"] += negative.item()
        totals["std"] += embedding_std.item()
        completed_steps += 1
        if step % log_interval == 0:
            print(f"Epoch [{epoch}] Step [{step}/{len(loader)}] Loss: {totals['loss']/completed_steps:.4f} "
                  f"PosSim: {totals['positive']/completed_steps:.4f} "
                  f"NegSim: {totals['negative']/completed_steps:.4f} "
                  f"Std: {totals['std']/completed_steps:.4f} LR: {optimizer.param_groups[0]['lr']:.6f}")
        if max_steps is not None and step >= max_steps:
            break

    if completed_steps == 0:
        raise RuntimeError("The training loader produced no batches.")
    return {
        "loss": totals["loss"] / completed_steps,
        "positive_similarity": totals["positive"] / completed_steps,
        "negative_similarity": totals["negative"] / completed_steps,
        "embedding_std": totals["std"] / completed_steps,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "epoch_time_seconds": time.perf_counter() - started_at,
        "completed_steps": completed_steps,
    }
