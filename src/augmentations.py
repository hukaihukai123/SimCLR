"""Data preprocessing and augmentation."""

from collections.abc import Callable

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

class TwoViewTransform:
    def __init__(self, base_transform:Callable):
        self.base_transform = base_transform


    def __call__(self, x:Image.Image)-> tuple[torch.Tensor, torch.Tensor]:
        view1 = self.base_transform(x)
        view2 = self.base_transform(x)
        return view1, view2

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  base_transform={self.base_transform}\n"
            f")"
        )

def build_simclr_transform(
    image_size:int=32,
    crop_scale:tuple[float, float]=(0.6,1.0),
    color_jitter_strength:float=0.5,
    color_jitter_probability:float=0.8,
    grayscale_probability:float=0.2,
    blur_probability:float=0.0
)->TwoViewTransform:
    """ color_jitter_strength:
            Global strength coefficient. Following the SimCLR convention:

                brightness = 0.8 * strength
                contrast   = 0.8 * strength
                saturation = 0.8 * strength
                hue        = 0.2 * strength"""
    for name, probability in (
        ("color_jitter_probability", color_jitter_probability),
        ("grayscale_probability", grayscale_probability),
        ("blur_probability", blur_probability),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

    brightness = 0.8 * color_jitter_strength
    contrast = 0.8 * color_jitter_strength
    saturation = 0.8 * color_jitter_strength
    hue = 0.2 * color_jitter_strength


    augmentation_steps: list[Callable] = [
        transforms.RandomResizedCrop(
            size=image_size,
            scale=crop_scale,
            ratio=(3 / 4, 4 / 3),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            ],
            p=color_jitter_probability,
        ),
        transforms.RandomGrayscale(p=grayscale_probability),
    ]

    if blur_probability > 0:
        augmentation_steps.append(
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(0.1, 2.0),
                    )

                ],
                p=blur_probability,
            )
        )

    augmentation_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )
    base_transform = transforms.Compose(augmentation_steps)

    return TwoViewTransform(base_transform)



def build_evaluation_transform() -> transforms.Compose:
    """Build the deterministic transform for validation and testing."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )


def denormalize_cifar10(tensor: torch.Tensor) -> torch.Tensor:
   
    if tensor.ndim not in (3, 4):
        raise ValueError(
            "Expected tensor with shape [C, H, W] or [B, C, H, W], "
            f"but received shape {tuple(tensor.shape)}."
        )

    mean = torch.as_tensor(
        CIFAR10_MEAN,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    std = torch.as_tensor(
        CIFAR10_STD,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    if tensor.ndim == 3:
        mean = mean[:, None, None]
        std = std[:, None, None]
    else:
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]

    image = tensor * std + mean

    return image.clamp(0.0, 1.0)