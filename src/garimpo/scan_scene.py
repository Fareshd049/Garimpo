"""Scan a stitched probability map (from infer_scene.py) for high-confidence blobs and
check each blob's proximity to nodata/black artifact regions.

Reusable as a library (scan_scene / scan_for_blobs) or as a CLI for one scene at a time.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def near_black_mask(image: np.ndarray, threshold: int) -> np.ndarray:
    return (image[..., 0] < threshold) & (image[..., 1] < threshold) & (image[..., 2] < threshold)


def scan_for_blobs(
    prob: np.ndarray,
    image: np.ndarray,
    prob_threshold: float,
    min_blob_size: int,
    black_threshold: int,
) -> list[dict]:
    """One dict per connected blob of prob > prob_threshold (>= min_blob_size px),
    sorted by size descending. near_black_frac is the fraction of the blob's bounding
    box that is near-black in the source image (see audit_artifacts.py)."""
    hot = prob > prob_threshold
    labeled, n = ndimage.label(hot)
    if n == 0:
        return []

    sizes = ndimage.sum(hot, labeled, index=np.arange(1, n + 1))
    near_black = near_black_mask(image, black_threshold)

    blobs = []
    for label_id in range(1, n + 1):
        size = int(sizes[label_id - 1])
        if size < min_blob_size:
            continue
        ys, xs = np.where(labeled == label_id)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        blobs.append({
            "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1,
            "size_px": size,
            "max_prob": float(prob[y0:y1 + 1, x0:x1 + 1].max()),
            "near_black_frac": float(near_black[y0:y1 + 1, x0:x1 + 1].mean()),
        })

    blobs.sort(key=lambda b: b["size_px"], reverse=True)
    return blobs


def scan_scene(
    prob_image_path: Path,
    scene_image_path: Path,
    prob_threshold: float = 0.5,
    min_blob_size: int = 1,
    black_threshold: int = 10,
) -> list[dict]:
    prob = np.array(Image.open(prob_image_path)).astype(np.float32) / 255.0
    image = np.array(Image.open(scene_image_path).convert("RGB"))
    if prob.shape != image.shape[:2]:
        raise ValueError(f"prob map shape {prob.shape} doesn't match scene shape {image.shape[:2]}")
    return scan_for_blobs(prob, image, prob_threshold, min_blob_size, black_threshold)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob-image", type=Path, required=True, help="*_prob.png from infer_scene.py")
    parser.add_argument("--scene-image", type=Path, required=True, help="Original full-res scene PNG")
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--min-blob-size", type=int, default=1)
    parser.add_argument("--black-threshold", type=int, default=10)
    args = parser.parse_args()

    blobs = scan_scene(
        args.prob_image, args.scene_image, args.prob_threshold, args.min_blob_size, args.black_threshold
    )

    logger.info("Scene: %s", args.scene_image.name)
    logger.info("%d blob(s) with prob > %.2f (min size %dpx)", len(blobs), args.prob_threshold, args.min_blob_size)
    for b in blobs:
        logger.info(
            "  bbox=(x:%d-%d, y:%d-%d) size=%dpx max_prob=%.4f near_black_frac=%.3f",
            b["xmin"], b["xmax"], b["ymin"], b["ymax"], b["size_px"], b["max_prob"], b["near_black_frac"],
        )


if __name__ == "__main__":
    main()
