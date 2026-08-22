"""Convert saved probability maps (infer_scene.py's *_prob.png) into transparent RGBA
heatmap overlays for the web viewer, and generate web_viewer/manifest.(json|js).

Unlike visualize.py's overlay (which blends the heat color into the original tile),
this produces just the heat with alpha scaling by probability, so it can sit on top of
a real satellite basemap without a second, misaligned copy of the imagery underneath.
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image

from garimpo.visualize import hot_colormap

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_ID_RE = re.compile(r"^(\d{8})_(\d{3}_\d{3})$")

logger = logging.getLogger(__name__)


def colorize(prob: np.ndarray, low_threshold: float) -> np.ndarray:
    """prob: (H,W) float in [0,1]. Returns (H,W,4) uint8 RGBA: hot colormap for RGB,
    alpha 0 below low_threshold ramping linearly to 255 at prob=1."""
    rgb = hot_colormap(prob)
    alpha = np.clip((prob - low_threshold) / (1 - low_threshold), 0.0, 1.0)
    rgba = np.dstack([rgb, alpha])
    return (rgba * 255).astype(np.uint8)


def scene_id_parts(scene_id: str) -> tuple[str, str]:
    match = SCENE_ID_RE.match(scene_id)
    if not match:
        raise ValueError(f"Unrecognized scene id: {scene_id}")
    date_raw, path_row = match.groups()
    date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
    return date, path_row


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-inference-dir", type=Path, default=REPO_ROOT / "runs" / "runs_v3" / "scene_inference")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "web_viewer" / "heatmaps")
    parser.add_argument("--manifest-dir", type=Path, default=REPO_ROOT / "web_viewer")
    parser.add_argument("--low-threshold", type=float, default=0.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prob_paths = sorted(args.scene_inference_dir.glob("*_prob.png"))
    logger.info("Found %d probability maps in %s", len(prob_paths), args.scene_inference_dir)

    manifest = []
    for prob_path in prob_paths:
        scene_id = prob_path.stem.removesuffix("_prob")
        date, path_row = scene_id_parts(scene_id)

        prob = np.array(Image.open(prob_path)).astype(np.float32) / 255.0
        rgba = colorize(prob, args.low_threshold)

        heat_filename = f"{scene_id}_heat.png"
        Image.fromarray(rgba, mode="RGBA").save(args.output_dir / heat_filename)
        logger.info("  %s -> heatmaps/%s", scene_id, heat_filename)

        manifest.append({
            "id": scene_id,
            "date": date,
            "path_row": path_row,
            "heatmap_file": f"heatmaps/{heat_filename}",
        })

    manifest.sort(key=lambda m: (m["path_row"], m["date"]))

    manifest_json_path = args.manifest_dir / "manifest.json"
    manifest_json_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %s", manifest_json_path)

    # Browsers block fetch()/XHR of local files opened via file://, but a <script src>
    # load is unaffected -- so the page includes this instead of fetching manifest.json.
    manifest_js_path = args.manifest_dir / "manifest.js"
    manifest_js_path.write_text("const MANIFEST = " + json.dumps(manifest, indent=2) + ";\n")
    logger.info("Wrote %s", manifest_js_path)


if __name__ == "__main__":
    main()
