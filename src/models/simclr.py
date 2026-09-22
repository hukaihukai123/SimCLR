import torch
from torch import nn
from torchvision.models import resnet18

class CIFARResNet18(nn.Module):

    output_dim: int = 512

    def __init__(self):
        super().__init__()

        backbone=resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        """防止第一层尺度缩小太小"""
        nn.init.kaiming_normal_(
            backbone.conv1.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        backbone.maxpool = nn.Identity()
        """去除分类头"""
        backbone.fc = nn.Identity()

        self.backbone = backbone
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class ProjectionHead(nn.Module):
    def __init__(self,
    input_dim:int=512,
    hidden_dim:int=512,
    output_dim:int=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim,bias=True),
        )
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.network(x)


class SimCLRModel(nn.Module):
    def __init__(self,
    projection_hidden_dim:int = 512,
    projection_output_dim:int = 128,):
        super().__init__()
        self.encoder = CIFARResNet18()

        self.projector = ProjectionHead(
            input_dim=self.encoder.output_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
        )

    def forward(self,images:torch.Tensor)->torch.Tensor:
        representations = self.encoder(images)
        projections = self.projector(representations)
        return representations, projections

        
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)
