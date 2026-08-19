"""Scene-grouped development train/validation split.

Tiles from the same source scene are spatially correlated, so splitting at the tile
level would leak information between train and val. Instead we hold out whole scenes.
With only ~12 scenes and 39 positive tiles spread thinly (1-5 per scene), a random
group split can easily starve val of positives, so we exhaustively search the (small)
space of scene subsets for the one whose tile-count and positive-count fractions are
both closest to val_fraction.

This is a DEVELOPMENT split, not a held-out test set: the val set is used for model
selection and reported metrics during development, so those metrics should not be
described as an unbiased estimate of final generalization performance. There is no
separate test split yet.

Known limitation: the whole dataset covers only 2 distinct geographic footprints
(path/row 226_120 and 226_121), each re-imaged at ~6 different dates. Any split that
targets a ~20% val fraction necessarily puts some scenes from both footprints in both
train and val (there are too few scenes per footprint to do otherwise), so val is not
fully spatially independent of train — a persistent mining site could in principle
appear in both, at different dates.
"""

import argparse
import itertools
import logging
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_RE = re.compile(r"CBERS_4A_WPM_(\d{8}_\d{3}_\d{3})")

logger = logging.getLogger(__name__)


def scene_id_from_filename(filename: str) -> str:
    match = SCENE_RE.match(filename)
    if not match:
        raise ValueError(f"Could not parse scene id from filename: {filename}")
    return match.group(1)


def make_scene_split(labels_csv: Path, val_fraction: float = 0.2) -> tuple[list[str], list[str]]:
    """Return (train_filenames, dev_val_filenames), split by holding out whole scenes.

    dev_val_filenames is a development validation set for model selection/reporting,
    not a final held-out test set.
    """
    labels = pd.read_csv(labels_csv)
    labels = labels[labels["label"] != "uncertain"].copy()
    labels["scene"] = labels["filename"].map(scene_id_from_filename)

    filenames_by_scene = labels.groupby("scene")["filename"].apply(list)
    n_positive_by_scene = labels.groupby("scene")["label"].apply(lambda s: int((s == "positive").sum()))

    scenes = list(filenames_by_scene.index)
    total_tiles = sum(len(v) for v in filenames_by_scene)
    total_positive = int(n_positive_by_scene.sum())

    best_subset: tuple[str, ...] | None = None
    best_cost = float("inf")
    for r in range(1, len(scenes)):
        for subset in itertools.combinations(scenes, r):
            val_pos = sum(n_positive_by_scene[s] for s in subset)
            if val_pos == 0:
                continue  # keep at least some positives in val
            val_tiles = sum(len(filenames_by_scene[s]) for s in subset)
            tile_frac = val_tiles / total_tiles
            pos_frac = val_pos / total_positive
            cost = abs(tile_frac - val_fraction) + abs(pos_frac - val_fraction)
            if cost < best_cost:
                best_cost = cost
                best_subset = subset

    if best_subset is None:
        raise ValueError("No scene subset has any positive tiles; cannot build a val split")

    val_scenes = set(best_subset)
    train_filenames = [f for s in scenes if s not in val_scenes for f in filenames_by_scene[s]]
    val_filenames = [f for s in val_scenes for f in filenames_by_scene[s]]
    return train_filenames, val_filenames


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, default=REPO_ROOT / "data" / "labels.csv")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels_csv)
    labels = labels[labels["label"] != "uncertain"].copy()
    labels["scene"] = labels["filename"].map(scene_id_from_filename)
    n_positive_by_scene = labels.groupby("scene")["label"].apply(lambda s: int((s == "positive").sum()))

    train_filenames, val_filenames = make_scene_split(args.labels_csv, args.val_fraction)

    val_scenes = sorted({scene_id_from_filename(f) for f in val_filenames})
    train_scenes = sorted({scene_id_from_filename(f) for f in train_filenames})
    val_pos = sum(n_positive_by_scene[s] for s in val_scenes)
    train_pos = sum(n_positive_by_scene[s] for s in train_scenes)

    logger.info("Dev-train: %d tiles (%d positive) from %d scenes: %s", len(train_filenames), train_pos, len(train_scenes), train_scenes)
    logger.info("Dev-val:   %d tiles (%d positive) from %d scenes: %s", len(val_filenames), val_pos, len(val_scenes), val_scenes)
    logger.info(
        "Note: dev-val is for model selection/reporting during development, not an unbiased "
        "final test estimate (no held-out test split exists yet)."
    )


if __name__ == "__main__":
    main()
