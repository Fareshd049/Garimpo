"""Sliding-window inference on a full-size scene, producing one stitched heatmap overlay.

Full scenes are far larger than the 640x640 tiles the model was trained on, so this
slides a tile-sized window across the scene with overlap, runs the model on each window,
and blends overlapping predictions back into one continuous probability map. Blending is
weighted toward each window's center (a raised-cosine/Hann kernel) so window boundaries
don't show up as visible seams in the stitched output.
"""

import argparse
import logging
import os
import pathlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from garimpo.model import build_unet
from garimpo.visualize import overlay_heatmap

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

if os.name == "nt":
    pathlib.PosixPath = pathlib.PurePosixPath


def window_starts(size: int, tile_size: int, stride: int) -> list[int]:
    """Window start offsets covering [0, size), flush against the far edge even if
    (size - tile_size) isn't a multiple of stride."""
    if size <= tile_size:
        return [0]
    starts = list(range(0, size - tile_size + 1, stride))
    if starts[-1] != size - tile_size:
        starts.append(size - tile_size)
    return starts


def make_weight_kernel(tile_size: int, min_weight: float = 0.1) -> np.ndarray:
    """2D center-weighted blend kernel (outer product of two Hann windows), floored at
    min_weight so pixels at the scene's outer border - covered by only one window's edge -
    don't collapse to near-zero total weight."""
    hann_1d = np.hanning(tile_size)
    hann_1d = min_weight + (1 - min_weight) * hann_1d
    return np.outer(hann_1d, hann_1d).astype(np.float32)


@torch.no_grad()
def sliding_window_predict(
    model, device, image: np.ndarray, tile_size: int, overlap: float
) -> np.ndarray:
    """image: (H,W,3) uint8. Returns (H,W) float32 probability map, same size as input."""
    height, width = image.shape[:2]
    stride = max(1, int(round(tile_size * (1 - overlap))))

    pad_h = max(0, tile_size - height)
    pad_w = max(0, tile_size - width)
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0))) if (pad_h or pad_w) else image
    padded_h, padded_w = padded.shape[:2]

    y_starts = window_starts(padded_h, tile_size, stride)
    x_starts = window_starts(padded_w, tile_size, stride)
    weight_kernel = make_weight_kernel(tile_size)

    prob_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)

    model.eval()
    for y0 in tqdm(y_starts, desc="Sliding window (rows)"):
        for x0 in x_starts:
            window = padded[y0:y0 + tile_size, x0:x0 + tile_size]
            window_tensor = torch.from_numpy(window.astype(np.float32) / 255.0)
            window_tensor = window_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

            logits = model(window_tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            prob_sum[y0:y0 + tile_size, x0:x0 + tile_size] += prob * weight_kernel
            weight_sum[y0:y0 + tile_size, x0:x0 + tile_size] += weight_kernel

    stitched = prob_sum / np.clip(weight_sum, 1e-6, None)
    return stitched[:height, :width]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    logger.info(
        "Loaded checkpoint: epoch=%s val_combined=%s",
        checkpoint.get("epoch"), checkpoint.get("val_combined"),
    )

    model = build_unet(encoder_weights=None).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    with Image.open(args.input_image) as scene:
        image = np.array(scene.convert("RGB"))
    logger.info("Scene size: %dx%d", image.shape[1], image.shape[0])

    stride = max(1, int(round(args.tile_size * (1 - args.overlap))))
    n_rows = len(window_starts(max(image.shape[0], args.tile_size), args.tile_size, stride))
    n_cols = len(window_starts(max(image.shape[1], args.tile_size), args.tile_size, stride))
    logger.info(
        "Tile size %d, overlap %.0f%% (stride %d) -> %d x %d = %d windows",
        args.tile_size, args.overlap * 100, stride, n_cols, n_rows, n_cols * n_rows,
    )

    prob = sliding_window_predict(model, device, image, args.tile_size, args.overlap)

    overlay = overlay_heatmap(image.astype(np.float32) / 255.0, prob)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(args.output_path)
    logger.info("Wrote stitched heatmap overlay to %s", args.output_path)

    prob_path = args.output_path.with_name(args.output_path.stem + "_prob" + args.output_path.suffix)
    Image.fromarray((prob * 255).astype(np.uint8), mode="L").save(prob_path)
    logger.info("Wrote raw probability map to %s", prob_path)


if __name__ == "__main__":
    main()
