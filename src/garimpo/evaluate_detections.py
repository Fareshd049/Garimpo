"""Match detected blobs (scan_scene.py) against ground-truth boxes (bboxes.csv) per scene,
producing TP/FP/FN counts and precision/recall -- the detection results table.

A blob and a ground-truth box are matched if their bounding rectangles overlap or are
within --match-distance pixels of each other (greedy nearest-first, one-to-one).
Unmatched boxes are false negatives; unmatched blobs are false positives.
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from garimpo.masks import load_boxes_by_filename
from garimpo.scan_scene import scan_scene
from garimpo.splits import make_scene_split, scene_id_from_filename

REPO_ROOT = Path(__file__).resolve().parents[2]
TILE_OFFSET_RE = re.compile(r"_x(\d+)_y(\d+)\.png$")

logger = logging.getLogger(__name__)


def tile_offset_from_filename(filename: str) -> tuple[int, int]:
    match = TILE_OFFSET_RE.search(filename)
    if not match:
        raise ValueError(f"Could not parse tile x/y offset from filename: {filename}")
    return int(match.group(1)), int(match.group(2))


def ground_truth_boxes_by_scene(bboxes_csv: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """Converts each tile-local box in bboxes.csv to absolute scene coordinates."""
    boxes_by_filename = load_boxes_by_filename(bboxes_csv)
    boxes_by_scene: dict[str, list[tuple[int, int, int, int]]] = {}
    for filename, local_boxes in boxes_by_filename.items():
        scene_id = scene_id_from_filename(filename)
        x_off, y_off = tile_offset_from_filename(filename)
        for xmin, ymin, xmax, ymax in local_boxes:
            boxes_by_scene.setdefault(scene_id, []).append(
                (x_off + xmin, y_off + ymin, x_off + xmax, y_off + ymax)
            )
    return boxes_by_scene


def rect_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """0 if the rectangles overlap or touch, else the Euclidean gap between them."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(ax0 - bx1, bx0 - ax1, 0)
    dy = max(ay0 - by1, by0 - ay1, 0)
    return float(np.hypot(dx, dy))


def merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]], merge_threshold: float = 0.0
) -> list[tuple[int, int, int, int]]:
    """Merges boxes within merge_threshold px of each other into one (their union's bounding
    box). Ground-truth boxes were drawn per-tile, so a real site sitting near a shared
    corner of several adjacent tiles gets an independent box in each tile -- these are
    duplicate labels of one site, not distinct sites, and inflate FN under 1:1 matching if
    left unmerged. Nearest-neighbor box distances are cleanly bimodal (0px or 59px+), so the
    default threshold of 0 (literal overlap/touch only) merges duplicates without risking
    collapsing genuinely distinct nearby sites."""
    n = len(boxes)
    if n <= 1:
        return list(boxes)

    adjacency = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if rect_distance(boxes[i], boxes[j]) <= merge_threshold:
                adjacency[i, j] = adjacency[j, i] = True

    n_components, labels = connected_components(csr_matrix(adjacency), directed=False)
    merged = []
    for label in range(n_components):
        members = [boxes[i] for i in range(n) if labels[i] == label]
        merged.append((
            min(b[0] for b in members),
            min(b[1] for b in members),
            max(b[2] for b in members),
            max(b[3] for b in members),
        ))
    return merged


def match_boxes_to_blobs(
    boxes: list[tuple[int, int, int, int]], blobs: list[dict], distance_threshold: float
) -> tuple[int, int, int]:
    """Greedy nearest-first one-to-one matching. Returns (tp, fp, fn)."""
    candidates = []
    for bi, box in enumerate(boxes):
        for bj, blob in enumerate(blobs):
            blob_rect = (blob["xmin"], blob["ymin"], blob["xmax"], blob["ymax"])
            d = rect_distance(box, blob_rect)
            if d <= distance_threshold:
                candidates.append((d, bi, bj))
    candidates.sort(key=lambda c: c[0])

    matched_boxes: set[int] = set()
    matched_blobs: set[int] = set()
    for _, bi, bj in candidates:
        if bi in matched_boxes or bj in matched_blobs:
            continue
        matched_boxes.add(bi)
        matched_blobs.add(bj)

    tp = len(matched_boxes)
    fn = len(boxes) - tp
    fp = len(blobs) - len(matched_blobs)
    return tp, fp, fn


def precision_recall(sub: pd.DataFrame) -> tuple[float, float]:
    tp, fp, fn = int(sub["tp"].sum()), int(sub["fp"].sum()), int(sub["fn"].sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return precision, recall


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-inference-dir", type=Path, default=REPO_ROOT / "runs" / "runs_v3" / "scene_inference")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--bboxes-csv", type=Path, default=REPO_ROOT / "data" / "bboxes.csv")
    parser.add_argument("--labels-csv", type=Path, default=REPO_ROOT / "data" / "labels.csv")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--min-blob-size", type=int, default=1)
    parser.add_argument("--black-threshold", type=int, default=10)
    parser.add_argument("--match-distance", type=float, default=50.0, help="px; max blob<->box gap to count as a match")
    parser.add_argument("--merge-duplicate-boxes", dest="merge_duplicate_boxes", action="store_true", default=True)
    parser.add_argument("--no-merge-duplicate-boxes", dest="merge_duplicate_boxes", action="store_false")
    parser.add_argument("--merge-threshold", type=float, default=0.0, help="px; ground-truth boxes this close are merged into one site")
    args = parser.parse_args()

    boxes_by_scene = ground_truth_boxes_by_scene(args.bboxes_csv)
    n_raw_boxes = sum(len(b) for b in boxes_by_scene.values())
    if args.merge_duplicate_boxes:
        boxes_by_scene = {
            scene_id: merge_overlapping_boxes(boxes, args.merge_threshold)
            for scene_id, boxes in boxes_by_scene.items()
        }
        n_merged_boxes = sum(len(b) for b in boxes_by_scene.values())
        logger.info(
            "Merged duplicate ground-truth boxes (threshold %.0fpx): %d raw boxes -> %d unique sites",
            args.merge_threshold, n_raw_boxes, n_merged_boxes,
        )

    _, val_filenames = make_scene_split(args.labels_csv, args.val_fraction)
    val_scenes = {scene_id_from_filename(f) for f in val_filenames}

    prob_paths = sorted(args.scene_inference_dir.glob("*_prob.png"))
    logger.info("Found %d scene probability maps in %s", len(prob_paths), args.scene_inference_dir)

    rows = []
    for prob_path in prob_paths:
        scene_id = prob_path.stem.removesuffix("_prob")
        raw_candidates = sorted(args.raw_dir.glob(f"*{scene_id}*.png"))
        if not raw_candidates:
            logger.warning("No raw scene image found for %s, skipping", scene_id)
            continue
        if len(raw_candidates) > 1:
            logger.warning("Multiple raw images matched %s, using %s", scene_id, raw_candidates[0].name)

        blobs = scan_scene(prob_path, raw_candidates[0], args.prob_threshold, args.min_blob_size, args.black_threshold)
        boxes = boxes_by_scene.get(scene_id, [])

        tp, fp, fn = match_boxes_to_blobs(boxes, blobs, args.match_distance)
        rows.append({
            "scene": scene_id,
            "split": "val" if scene_id in val_scenes else "train",
            "n_boxes": len(boxes),
            "n_blobs": len(blobs),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    df = pd.DataFrame(rows).sort_values("scene", ignore_index=True)

    logger.info("\n%s", df.to_string(index=False))
    logger.info("")

    for label, sub in [
        ("ALL 13 scenes", df),
        ("dev-train scenes only", df[df["split"] == "train"]),
        ("dev-val scenes only", df[df["split"] == "val"]),
    ]:
        tp, fp, fn = int(sub["tp"].sum()), int(sub["fp"].sum()), int(sub["fn"].sum())
        precision, recall = precision_recall(sub)
        logger.info(
            "%-22s TP=%-4d FP=%-4d FN=%-4d precision=%.3f recall=%.3f",
            label, tp, fp, fn, precision, recall,
        )


if __name__ == "__main__":
    main()
