"""Joint image+mask augmentation for training.

Spatial transforms (flips, rotations) are applied identically to image and mask so
they stay pixel-aligned; photometric jitter (brightness/contrast) is image-only.
"""

import random

import torch


class JointAugmentation:
    """Callable transform: (image, mask) -> (image, mask), matching the signature
    GarimpoDataset already accepts.
    """

    def __init__(self, seed: int, brightness: float = 0.2, contrast: float = 0.2) -> None:
        self.rng = random.Random(seed)
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rng.random() < 0.5:
            image = image.flip(-1)
            mask = mask.flip(-1)
        if self.rng.random() < 0.5:
            image = image.flip(-2)
            mask = mask.flip(-2)

        k = self.rng.randint(0, 3)
        if k:
            image = torch.rot90(image, k, dims=(-2, -1))
            mask = torch.rot90(mask, k, dims=(-2, -1))

        brightness_factor = 1.0 + self.rng.uniform(-self.brightness, self.brightness)
        image = (image * brightness_factor).clamp(0.0, 1.0)

        contrast_factor = 1.0 + self.rng.uniform(-self.contrast, self.contrast)
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = ((image - mean) * contrast_factor + mean).clamp(0.0, 1.0)

        return image, mask
