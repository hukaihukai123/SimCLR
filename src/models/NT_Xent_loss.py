import torch
import torch.nn.functional as F
from torch import nn
class NTXentLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.2,
        reduction: str = "mean",
    ) -> None:

        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                "reduction must be 'mean', 'sum', or 'none'."
            )

        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        z_1: torch.Tensor,
        z_2: torch.Tensor,
    ) -> torch.Tensor:
        if z_1.ndim != 2 or z_2.ndim != 2:
            raise ValueError("Input tensors must be 2-dimensional")
        if z_1.shape != z_2.shape:
            raise ValueError("Input tensors must have the same shape")
        if z_1.shape[0]<2:
            raise ValueError("Batch size must be at least 2")

        batch_size = z_1.shape[0]
        z_1 = F.normalize(z_1, dim=1)
        z_2 = F.normalize(z_2, dim=1)

        z=torch.cat([z_1, z_2], dim=0)

        logits=z @ z.transpose(0, 1)/ self.temperature

        self_mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=z.device,
        )
        logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min,)

        positive_indices = (torch.arange(2 * batch_size, device=z.device)+ batch_size) %(2 * batch_size)

        loss=F.cross_entropy(logits, positive_indices, reduction=self.reduction)
        return loss
