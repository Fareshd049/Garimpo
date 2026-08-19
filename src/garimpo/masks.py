"""Rasterize bounding boxes into per-tile binary masks for segmentation training."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def rasterize_boxes(size: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Build a binary (0/255) mask of the given (width, height) with each box filled in."""
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for xmin, ymin, xmax, ymax in boxes:
        x0, x1 = sorted((max(0, min(xmin, width)), max(0, min(xmax, width))))
        y0, y1 = sorted((max(0, min(ymin, height)), max(0, min(ymax, height))))
        mask[y0:y1, x0:x1] = 255
    return mask


def load_boxes_by_filename(bboxes_csv: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    df = pd.read_csv(bboxes_csv)
    boxes_by_filename: dict[str, list[tuple[int, int, int, int]]] = {}
    for filename, group in df.groupby("filename"):
        boxes_by_filename[filename] = list(
            group[["xmin", "ymin", "xmax", "ymax"]].itertuples(index=False, name=None)
        )
    return boxes_by_filename


def generate_masks(tiles_dir: Path, bboxes_csv: Path, labels_csv: Path, masks_dir: Path) -> None:
    labels = pd.read_csv(labels_csv)
    boxes_by_filename = load_boxes_by_filename(bboxes_csv)

    targets = labels[labels["label"] != "uncertain"]
    n_uncertain = len(labels) - len(targets)

    masks_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_with_foreground = 0
    n_skipped = 0

    for filename in tqdm(targets["filename"], desc="Rasterizing masks"):
        tile_path = tiles_dir / filename
        if not tile_path.exists():
            logger.warning("Tile not found, skipping: %s", tile_path)
            n_skipped += 1
            continue

        with Image.open(tile_path) as tile:
            size = tile.size

        boxes = boxes_by_filename.get(filename, [])
        mask = rasterize_boxes(size, boxes)

        Image.fromarray(mask, mode="L").save(masks_dir / filename)
        n_written += 1
        if mask.any():
            n_with_foreground += 1

    logger.info(
        "Masks written: %d | with foreground: %d | uncertain excluded: %d | skipped (missing tile): %d",
        n_written,
        n_with_foreground,
        n_uncertain,
        n_skipped,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-dir", type=Path, default=REPO_ROOT / "data" / "tiles")
    parser.add_argument("--bboxes-csv", type=Path, default=REPO_ROOT / "data" / "bboxes.csv")
    parser.add_argument("--labels-csv", type=Path, default=REPO_ROOT / "data" / "labels.csv")
    parser.add_argument("--masks-dir", type=Path, default=REPO_ROOT / "data" / "masks")
    args = parser.parse_args()

    generate_masks(args.tiles_dir, args.bboxes_csv, args.labels_csv, args.masks_dir)


if __name__ == "__main__":
    main()
