"""Diagnostic: are positive tiles/boxes disproportionately near black/nodata artifacts?

Part 1 checks whether positive tiles simply contain more near-black pixels than negative
tiles overall. Part 2 is the more direct test: for each labeled box, how close is it to
the nearest large near-black blob in its own tile? A box sitting right next to a nodata
seam is a candidate for "the model learned to find scan artifacts, not mining sites."

Diagnostic only — does not exclude or relabel anything.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from garimpo.masks import load_boxes_by_filename

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def near_black_mask(image: np.ndarray, threshold: int) -> np.ndarray:
    return (image[..., 0] < threshold) & (image[..., 1] < threshold) & (image[..., 2] < threshold)


def significant_blob_mask(mask: np.ndarray, min_blob_size: int) -> np.ndarray:
    """mask with only connected components larger than min_blob_size kept."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    keep_labels = np.where(sizes > min_blob_size)[0] + 1
    return np.isin(labeled, keep_labels)


def distance_box_to_mask(box: tuple[int, int, int, int], significant_mask: np.ndarray) -> float:
    """Min distance from any pixel in box to the nearest True pixel in significant_mask.
    0 if the box overlaps a significant blob; inf if the tile has no significant blob."""
    if not significant_mask.any():
        return float("inf")
    dist_map = ndimage.distance_transform_edt(~significant_mask)
    xmin, ymin, xmax, ymax = box
    return float(dist_map[ymin:ymax, xmin:xmax].min())


def run_part1(tiles_dir: Path, labels_csv: Path, black_threshold: int) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    labels = labels[labels["label"] != "uncertain"].copy()

    fractions = []
    for filename in tqdm(labels["filename"], desc="Part 1: black fraction per tile"):
        image = np.array(Image.open(tiles_dir / filename).convert("RGB"))
        fractions.append(near_black_mask(image, black_threshold).mean())
    labels["black_fraction"] = fractions
    return labels


def run_part2(tiles_dir: Path, bboxes_csv: Path, black_threshold: int, min_blob_size: int) -> pd.DataFrame:
    boxes_by_filename = load_boxes_by_filename(bboxes_csv)

    rows = []
    for filename, boxes in tqdm(boxes_by_filename.items(), desc="Part 2: box-to-artifact distance"):
        image = np.array(Image.open(tiles_dir / filename).convert("RGB"))
        mask = near_black_mask(image, black_threshold)
        sig_mask = significant_blob_mask(mask, min_blob_size)

        for xmin, ymin, xmax, ymax in boxes:
            distance = distance_box_to_mask((xmin, ymin, xmax, ymax), sig_mask)
            rows.append({
                "filename": filename,
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                "distance_to_artifact": distance,
            })

    return pd.DataFrame(rows).sort_values("distance_to_artifact", ascending=True, ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-dir", type=Path, default=REPO_ROOT / "data" / "tiles")
    parser.add_argument("--labels-csv", type=Path, default=REPO_ROOT / "data" / "labels.csv")
    parser.add_argument("--bboxes-csv", type=Path, default=REPO_ROOT / "data" / "bboxes.csv")
    parser.add_argument("--black-threshold", type=int, default=10, help="R<T and G<T and B<T counts as near-black")
    parser.add_argument("--min-blob-size", type=int, default=500, help="px; smaller near-black blobs are ignored")
    parser.add_argument("--suspicious-distance", type=float, default=100.0, help="px; flag boxes closer than this")
    args = parser.parse_args()

    # --- Part 1 ---
    part1 = run_part1(args.tiles_dir, args.labels_csv, args.black_threshold)
    positive_frac = part1.loc[part1["label"] == "positive", "black_fraction"]
    negative_frac = part1.loc[part1["label"] == "negative", "black_fraction"]

    stat, pvalue = mannwhitneyu(positive_frac, negative_frac, alternative="greater")

    logger.info("=" * 70)
    logger.info("PART 1: tile-level near-black pixel fraction (threshold R,G,B < %d)", args.black_threshold)
    logger.info("=" * 70)
    logger.info(
        "Positive (n=%d): mean=%.4f  median=%.4f  max=%.4f",
        len(positive_frac), positive_frac.mean(), positive_frac.median(), positive_frac.max(),
    )
    logger.info(
        "Negative (n=%d): mean=%.4f  median=%.4f  max=%.4f",
        len(negative_frac), negative_frac.mean(), negative_frac.median(), negative_frac.max(),
    )
    logger.info(
        "Mann-Whitney U (H1: positive > negative): U=%.1f, p=%.6f", stat, pvalue,
    )

    # --- Part 2 ---
    part2 = run_part2(args.tiles_dir, args.bboxes_csv, args.black_threshold, args.min_blob_size)
    part2["suspicious"] = part2["distance_to_artifact"] < args.suspicious_distance

    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 2: box distance to nearest near-black blob (min size %dpx)", args.min_blob_size)
    logger.info("=" * 70)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        logger.info("\n%s", part2.to_string(index=False))

    # --- Summary ---
    n_boxes = len(part2)
    n_tiles = part2["filename"].nunique()
    n_suspicious = int(part2["suspicious"].sum())
    n_no_blob = int((part2["distance_to_artifact"] == float("inf")).sum())

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "Part 2: %d/%d positive boxes (across %d positive tiles) are within %.0fpx of a near-black blob >= %dpx "
        "(flagged suspicious). %d boxes have no qualifying blob in their tile at all.",
        n_suspicious, n_boxes, n_tiles, args.suspicious_distance, args.min_blob_size, n_no_blob,
    )
    logger.info(
        "Part 1: positive tiles have %s more near-black pixel content than negative tiles "
        "(Mann-Whitney one-sided p=%.6f, %s at alpha=0.05).",
        "significantly" if pvalue < 0.05 else "NOT significantly",
        pvalue,
        "reject H0" if pvalue < 0.05 else "fail to reject H0",
    )
    logger.info("No tiles/boxes were excluded or modified - diagnostic only.")


if __name__ == "__main__":
    main()
