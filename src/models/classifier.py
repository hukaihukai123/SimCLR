"""Linear classifier attached to a CIFAR-10 image encoder."""

import torch
from torch import nn

from src.models.simclr import CIFARResNet18


class EncoderClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module | None = None,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else CIFARResNet18()
        self.classifier = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.classifier(features)
