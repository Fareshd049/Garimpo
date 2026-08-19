"""PyTorch Dataset pairing tiles and masks by filename."""

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Transform = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class GarimpoDataset(Dataset):
    """Pairs tile images with their rasterized masks for the given filenames.

    Takes an explicit filename list rather than scanning a directory, so train/val/test
    splitting is decided by the caller and stays independent of this class.
    """

    def __init__(
        self,
        filenames: Sequence[str],
        tiles_dir: Path,
        masks_dir: Path,
        transform: Transform | None = None,
    ) -> None:
        self.filenames = list(filenames)
        self.tiles_dir = Path(tiles_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]

        with Image.open(self.tiles_dir / filename) as tile:
            image = np.array(tile.convert("RGB"), dtype=np.uint8)
        with Image.open(self.masks_dir / filename) as mask_img:
            mask = np.array(mask_img, dtype=np.uint8)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        # Joint transform so spatial augmentations (flips, crops, ...) stay in sync.
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
