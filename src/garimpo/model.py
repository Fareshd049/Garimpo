"""U-Net model factory."""

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(encoder_name: str = "resnet18", encoder_weights: str | None = "imagenet") -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )
