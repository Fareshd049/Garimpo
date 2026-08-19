"""Weighted BCE + Dice loss for binary segmentation, with data-derived pos_weight."""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def compute_pos_weight(filenames: Sequence[str], masks_dir: Path, max_pos_weight: float) -> float:
    """neg_pixels / pos_pixels over the given masks, capped at max_pos_weight for
    training stability (the raw ratio can be extreme with sparse targets)."""
    pos_pixels = 0
    total_pixels = 0
    for filename in filenames:
        mask = np.array(Image.open(masks_dir / filename))
        pos_pixels += int((mask > 0).sum())
        total_pixels += mask.size

    if pos_pixels == 0:
        raise ValueError("No positive pixels found in the given masks; cannot derive pos_weight")

    neg_pixels = total_pixels - pos_pixels
    return min(neg_pixels / pos_pixels, max_pos_weight)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss (1 - Dice), averaged over the batch."""
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.dim()))
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class WeightedBCEDiceLoss(nn.Module):
    """combined = bce_weight * BCE(pos_weight) + dice_weight * DiceLoss.

    Returns (combined, bce, dice) so all three can be logged separately.
    """

    def __init__(self, pos_weight: float, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        dice = dice_loss(logits, target)
        combined = self.bce_weight * bce + self.dice_weight * dice
        return combined, bce, dice
