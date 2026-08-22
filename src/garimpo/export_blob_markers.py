"""Export blob detections at a grid of probability thresholds for the web viewer's
per-detection markers and threshold slider.

Connected-component blob detection (scan_scene.py) needs scipy and full prob arrays --
not practical to run live in a browser -- so this precomputes blobs at a threshold grid
fine enough to feel continuous when the viewer's slider snaps to the nearest level.
Centroids are exported in raw image-pixel coordinates; the viewer converts them to
lat/lng using the same footprint bounds already defined in index.html, so the
georeferencing bounds aren't duplicated here.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from garimpo.scan_scene import scan_for_blobs

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-inference-dir", type=Path, default=REPO_ROOT / "runs" / "runs_v3" / "scene_inference")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--output-path", type=Path, default=REPO_ROOT / "web_viewer" / "blobs.js")
    parser.add_argument("--min-threshold", type=float, default=0.05)
    parser.add_argument("--max-threshold", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--min-blob-size", type=int, default=1)
    parser.add_argument("--black-threshold", type=int, default=10)
    args = parser.parse_args()

    thresholds = np.round(
        np.arange(args.min_threshold, args.max_threshold + 1e-9, args.threshold_step), 2
    )

    prob_paths = sorted(args.scene_inference_dir.glob("*_prob.png"))
    logger.info("Found %d scene probability maps; %d threshold levels each", len(prob_paths), len(thresholds))

    data: dict[str, dict] = {}
    for prob_path in prob_paths:
        scene_id = prob_path.stem.removesuffix("_prob")
        raw_candidates = sorted(args.raw_dir.glob(f"*{scene_id}*.png"))
        if not raw_candidates:
            logger.warning("No raw scene image found for %s, skipping", scene_id)
            continue

        prob = np.array(Image.open(prob_path)).astype(np.float32) / 255.0
        image = np.array(Image.open(raw_candidates[0]).convert("RGB"))
        height, width = prob.shape

        by_threshold = {}
        total_blobs = 0
        for t in thresholds:
            blobs = scan_for_blobs(prob, image, float(t), args.min_blob_size, args.black_threshold)
            by_threshold[f"{t:.2f}"] = [
                {
                    "x": round(b["cx"], 1),
                    "y": round(b["cy"], 1),
                    "max_prob": round(b["max_prob"], 4),
                    "mean_prob": round(b["mean_prob"], 4),
                    "near_black_frac": round(b["near_black_frac"], 3),
                }
                for b in blobs
            ]
            total_blobs += len(blobs)

        data[scene_id] = {"width": width, "height": height, "by_threshold": by_threshold}
        logger.info("  %s: %d total blob-instances across %d thresholds", scene_id, total_blobs, len(thresholds))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("const BLOBS = " + json.dumps(data, indent=2) + ";\n")
    logger.info("Wrote %s", args.output_path)


if __name__ == "__main__":
    main()
