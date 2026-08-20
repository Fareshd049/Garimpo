"""Run a checkpoint on val tiles and save predicted-probability heatmaps overlaid on the tile."""

import argparse
import logging
import os
import pathlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from garimpo.masks import load_boxes_by_filename
from garimpo.model import build_unet
from garimpo.splits import make_scene_split

GROUND_TRUTH_BOX_COLOR = (0, 255, 255)  # cyan — distinct from the black/red/yellow/white heatmap

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

# Checkpoints saved on Colab (Linux) embed PosixPath objects in their metadata; unpickling
# those on Windows fails outright unless PosixPath is made constructible here.
if os.name == "nt":
    pathlib.PosixPath = pathlib.PurePosixPath


def hot_colormap(prob: np.ndarray) -> np.ndarray:
    """Map [0,1] probabilities to an RGB "hot" heatmap (black -> red -> yellow -> white)."""
    r = np.clip(prob * 3.0, 0.0, 1.0)
    g = np.clip(prob * 3.0 - 1.0, 0.0, 1.0)
    b = np.clip(prob * 3.0 - 2.0, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(image: np.ndarray, prob: np.ndarray, max_alpha: float = 0.7) -> np.ndarray:
    """image: (H,W,3) float in [0,1]. prob: (H,W) float in [0,1]. Blend strength scales with
    probability so low-confidence areas stay close to the original image."""
    heat = hot_colormap(prob)
    alpha = (prob * max_alpha)[..., None]
    return image * (1 - alpha) + heat * alpha


def draw_ground_truth_boxes(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    """boxes use the same [xmin,ymin,xmax,ymax) convention as masks.rasterize_boxes."""
    draw = ImageDraw.Draw(image)
    for xmin, ymin, xmax, ymax in boxes:
        draw.rectangle([xmin, ymin, xmax - 1, ymax - 1], outline=GROUND_TRUTH_BOX_COLOR, width=2)
    return image


@torch.no_grad()
def predict_and_save(
    model,
    device,
    filenames,
    tiles_dir: Path,
    output_dir: Path,
    boxes_by_filename: dict[str, list[tuple[int, int, int, int]]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for filename in tqdm(filenames, desc="Predicting"):
        with Image.open(tiles_dir / filename) as tile:
            image = np.array(tile.convert("RGB"), dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)
        logits = model(image_tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        overlay = overlay_heatmap(image, prob)
        overlay_image = Image.fromarray((overlay * 255).astype(np.uint8))

        if boxes_by_filename is not None and filename in boxes_by_filename:
            overlay_image = draw_ground_truth_boxes(overlay_image, boxes_by_filename[filename])

        overlay_image.save(output_dir / filename)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tiles-dir", type=Path, default=REPO_ROOT / "data" / "tiles")
    parser.add_argument("--labels-csv", type=Path, default=REPO_ROOT / "data" / "labels.csv")
    parser.add_argument("--bboxes-csv", type=Path, default=None, help="If given, overlay ground-truth box outlines")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.checkpoint.parent.parent / "visualizations" / args.checkpoint.stem)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    _, val_filenames = make_scene_split(args.labels_csv, args.val_fraction)
    logger.info("Val tiles: %d", len(val_filenames))

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    logger.info(
        "Loaded checkpoint: epoch=%s val_combined=%s",
        checkpoint.get("epoch"), checkpoint.get("val_combined"),
    )

    model = build_unet(encoder_weights=None).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    boxes_by_filename = None
    if args.bboxes_csv is not None:
        boxes_by_filename = load_boxes_by_filename(args.bboxes_csv)
        logger.info("Loaded ground-truth boxes for %d tiles from %s", len(boxes_by_filename), args.bboxes_csv)

    predict_and_save(model, device, val_filenames, args.tiles_dir, output_dir, boxes_by_filename)
    logger.info("Wrote %d overlay images to %s", len(val_filenames), output_dir)


if __name__ == "__main__":
    main()
