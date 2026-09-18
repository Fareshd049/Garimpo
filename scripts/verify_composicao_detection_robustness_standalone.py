"""
Standalone check: does the trained garimpo model still detect known mining sites on a
scene composited with the best-effort percentile/gamma combo found by
verify_composicao_standalone.py's grid search, compared to the original training-data
composite of the same scene?

This is the robustness question that follows on from the compositing-accuracy checks:
even if the newly-composited scene isn't a pixel-perfect match for the training
distribution, the model may still be robust enough to the tonal difference for practical
use. If detections hold up here, that's real evidence for shipping the live backend with
this best-effort compositing (clearly documented as an approximation). If a known site's
detection collapses, that's evidence the actual TOM_PADRAO calibration is needed, not
just a better percentile/gamma guess.

Self-contained on purpose -- paste this whole file into a single Kaggle notebook cell and
run top to bottom. No garimpo/repo imports; the model architecture, sliding-window
inference, and blob-scanning logic are copied inline from
src/garimpo/model.py / infer_scene.py / scan_scene.py so this has no dependency on the
rest of the repo -- only the trained checkpoint file itself.

Needs, in addition to numpy/rasterio/Pillow/scipy (Kaggle defaults):
  - torch (Kaggle default)
  - segmentation-models-pytorch -- NOT always preinstalled; if the import below fails,
    run `!pip install -q segmentation-models-pytorch` in a cell above this one first.

Steps:
  1. Edit the CONFIG block: the four band paths for 20220921_226_120, REFERENCE_PNG (the
     original training-data composite of that same scene), and CHECKPOINT_PATH (upload
     runs/runs_v3/best.pt -- or your preferred checkpoint -- as a small Kaggle Dataset;
     unlike the raw imagery, a checkpoint is small enough to upload directly).
  2. Run All / execute the cell.
  3. Read the per-site table and the VERDICT at the bottom.
"""

import os
import pathlib
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from PIL import Image
from scipy import ndimage

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise ImportError(
        "segmentation_models_pytorch is not installed. Run `!pip install -q "
        "segmentation-models-pytorch` in a cell above this one, then re-run."
    ) from exc

# =============================================================================
# CONFIG -- edit these before running
# =============================================================================
BAND1_TIF = Path("/kaggle/input/datasets/raraabdel/projet-ufam/cbers4a_L4_bandas/CBERS4A_WPM_20220921_226_120_L4/CBERS_4A_WPM_20220921_226_120_L4_BAND1.tif")
BAND2_TIF = Path("/kaggle/input/datasets/raraabdel/projet-ufam/cbers4a_L4_bandas/CBERS4A_WPM_20220921_226_120_L4/CBERS_4A_WPM_20220921_226_120_L4_BAND2.tif")
BAND3_TIF = Path("/kaggle/input/datasets/raraabdel/projet-ufam/cbers4a_L4_bandas/CBERS4A_WPM_20220921_226_120_L4/CBERS_4A_WPM_20220921_226_120_L4_BAND3.tif")
BAND4_TIF = Path("/kaggle/input/datasets/raraabdel/projet-ufam/cbers4a_L4_bandas/CBERS4A_WPM_20220921_226_120_L4/CBERS_4A_WPM_20220921_226_120_L4_BAND4.tif")

REFERENCE_PNG = Path("/kaggle/input/datasets/.../CBERS_4A_WPM_20220921_226_120_L4_RGB_TrueColor_PanSharpened_TOM_PADRAO_V4.png")

# Upload the checkpoint as a Kaggle Dataset (it's small, unlike the raw imagery).
CHECKPOINT_PATH = Path("/kaggle/input/.../best.pt")

OUTPUT_DIR = Path("/kaggle/working/verify_composicao_detection_robustness")

# Geometry/resolution already confirmed correct at this value in the earlier grid-search
# check -- keep it fixed here.
TAMANHO_PNG_MAX = 6000

# Best-effort compositing parameters found by verify_composicao_standalone.py's grid
# search. Not claimed to be pixel-exact -- that's the whole point of this check.
PERCENTIL_MIN = 0.5
PERCENTIL_MAX = 95
GAMMA = 1.8

# Sliding-window inference settings -- match src/garimpo/infer_scene.py's defaults.
TILE_SIZE = 640
OVERLAP = 0.25

# Blob-detection settings -- match the project's own convention (pipeline_report.pdf
# section 4: "blob detection at probability > 0.5, connected components, no minimum
# size").
PROB_THRESHOLD = 0.5
MIN_BLOB_SIZE = 1
BLACK_THRESHOLD = 10

# How close (px) a detected blob must be to a known site's coordinates to count as
# "found" -- matches the project's own blob<->box matching gap (pipeline_report.pdf
# section 4: "blob<->box gap <= 50px").
MATCH_DISTANCE_PX = 50

# Known site pixel coordinates for 20220921_226_120, derived from data/bboxes.csv (full
# composed-scene coordinates = tile offset from the filename's _x<N>_y<N> suffix + the
# box's local center). 4 of the raw labeled boxes in this scene sit within a few pixels
# of each other -- the "group of four at a single site" duplicate cluster documented in
# docs/dataset_limitations.md -- and are merged into one site here, matching
# evaluate_detections.py's default duplicate-merging. That gives 3 unique sites; adjust
# this list if your own count differs.
KNOWN_SITES = [
    {"label": "site_A (tile-corner cluster x4)", "x": 1088, "y": 4662},
    {"label": "site_B", "x": 1227, "y": 4616},
    {"label": "site_C", "x": 4273, "y": 2998},
]

# =============================================================================
# Compositing -- copied from composicao_cbers4a_wpm.py, single fixed combo (no grid).
# =============================================================================
def read_band_raw(caminho: Path, tamanho_max: int) -> np.ndarray:
    with rasterio.open(caminho) as src:
        escala = max(src.width, src.height) / tamanho_max
        if escala > 1:
            largura = int(src.width / escala)
            altura = int(src.height / escala)
        else:
            largura, altura = src.width, src.height
        return src.read(
            1, out_shape=(altura, largura), resampling=Resampling.average
        ).astype("float32")


def stretch_band(banda_raw: np.ndarray, p_min: float, p_max: float, gamma: float) -> np.ndarray:
    validos = banda_raw[banda_raw > 0]
    if validos.size == 0:
        return np.zeros(banda_raw.shape, dtype="uint8")
    v_min, v_max = np.percentile(validos, [p_min, p_max])
    banda = banda_raw.copy()
    banda -= v_min
    banda /= max(v_max - v_min, 1e-6)
    np.clip(banda, 0.0, 1.0, out=banda)
    if gamma and gamma != 1.0:
        np.power(banda, 1.0 / gamma, out=banda)
    banda *= 255.0
    return banda.astype("uint8")


def compose_true_color(raw_r: np.ndarray, raw_g: np.ndarray, raw_b: np.ndarray,
                        p_min: float, p_max: float, gamma: float) -> np.ndarray:
    return np.stack(
        [stretch_band(raw_r, p_min, p_max, gamma),
         stretch_band(raw_g, p_min, p_max, gamma),
         stretch_band(raw_b, p_min, p_max, gamma)],
        axis=-1,
    )


# =============================================================================
# Model + sliding-window inference -- copied from src/garimpo/model.py and
# src/garimpo/infer_scene.py.
# =============================================================================
def build_unet(encoder_name: str = "resnet18", encoder_weights: str | None = None) -> torch.nn.Module:
    return smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=1)


def window_starts(size: int, tile_size: int, stride: int) -> list[int]:
    if size <= tile_size:
        return [0]
    starts = list(range(0, size - tile_size + 1, stride))
    if starts[-1] != size - tile_size:
        starts.append(size - tile_size)
    return starts


def make_weight_kernel(tile_size: int, min_weight: float = 0.1) -> np.ndarray:
    hann_1d = np.hanning(tile_size)
    hann_1d = min_weight + (1 - min_weight) * hann_1d
    return np.outer(hann_1d, hann_1d).astype(np.float32)


@torch.no_grad()
def sliding_window_predict(model, device, image: np.ndarray, tile_size: int, overlap: float) -> np.ndarray:
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
    total = len(y_starts) * len(x_starts)
    done = 0
    for y0 in y_starts:
        for x0 in x_starts:
            window = padded[y0:y0 + tile_size, x0:x0 + tile_size]
            window_tensor = torch.from_numpy(window.astype(np.float32) / 255.0)
            window_tensor = window_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            logits = model(window_tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            prob_sum[y0:y0 + tile_size, x0:x0 + tile_size] += prob * weight_kernel
            weight_sum[y0:y0 + tile_size, x0:x0 + tile_size] += weight_kernel
            done += 1
            if done % 10 == 0 or done == total:
                print(f"    window {done}/{total}")

    stitched = prob_sum / np.clip(weight_sum, 1e-6, None)
    return stitched[:height, :width]


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    # Cross-platform pickle safety net: a checkpoint's saved args (argparse Namespace)
    # can embed a WindowsPath or PosixPath depending on where it was trained, which fails
    # to unpickle on the other OS. Patch both directions defensively.
    if os.name == "nt":
        pathlib.PosixPath = pathlib.PurePosixPath
    else:
        pathlib.WindowsPath = pathlib.PureWindowsPath

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: epoch={checkpoint.get('epoch')} val_combined={checkpoint.get('val_combined')}")
    model = build_unet(encoder_weights=None).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


# =============================================================================
# Blob scanning -- copied from src/garimpo/scan_scene.py.
# =============================================================================
def near_black_mask(image: np.ndarray, threshold: int) -> np.ndarray:
    return (image[..., 0] < threshold) & (image[..., 1] < threshold) & (image[..., 2] < threshold)


def scan_for_blobs(prob: np.ndarray, image: np.ndarray, prob_threshold: float,
                    min_blob_size: int, black_threshold: int) -> list[dict]:
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
        blob_probs = prob[ys, xs]
        blobs.append({
            "cx": float(xs.mean()), "cy": float(ys.mean()),
            "size_px": size,
            "max_prob": float(blob_probs.max()),
            "mean_prob": float(blob_probs.mean()),
            "near_black_frac": float(near_black[y0:y1 + 1, x0:x1 + 1].mean()),
        })
    blobs.sort(key=lambda b: b["size_px"], reverse=True)
    return blobs


def nearest_blob(x: float, y: float, blobs: list[dict], max_distance: float) -> dict | None:
    best, best_dist = None, max_distance
    for b in blobs:
        dist = float(np.hypot(b["cx"] - x, b["cy"] - y))
        if dist <= best_dist:
            best, best_dist = b, dist
    return best


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nReading raw bands and compositing the new scene ...")
    raw_r = read_band_raw(BAND3_TIF, TAMANHO_PNG_MAX)
    raw_g = read_band_raw(BAND2_TIF, TAMANHO_PNG_MAX)
    raw_b = read_band_raw(BAND1_TIF, TAMANHO_PNG_MAX)
    new_composed = compose_true_color(raw_r, raw_g, raw_b, PERCENTIL_MIN, PERCENTIL_MAX, GAMMA)
    print(f"New composite: {new_composed.shape[1]} x {new_composed.shape[0]} px")

    reference = np.array(Image.open(REFERENCE_PNG).convert("RGB"))
    print(f"Reference (training-data) composite: {reference.shape[1]} x {reference.shape[0]} px")

    if new_composed.shape != reference.shape:
        print("\n!! New composite and reference are different shapes -- geometry check from")
        print("!! the earlier grid-search script hasn't been repeated here. Aborting rather")
        print("!! than comparing misaligned images.")
        print(f"!!   new composite = {new_composed.shape}")
        print(f"!!   reference     = {reference.shape}")
        return

    model = load_model(CHECKPOINT_PATH, device)

    print(f"\nRunning sliding-window inference on the NEW composite ({TILE_SIZE}px tiles, {OVERLAP:.0%} overlap) ...")
    prob_new = sliding_window_predict(model, device, new_composed, TILE_SIZE, OVERLAP)

    print(f"\nRunning sliding-window inference on the REFERENCE composite ...")
    prob_ref = sliding_window_predict(model, device, reference, TILE_SIZE, OVERLAP)

    Image.fromarray((prob_new * 255).astype("uint8"), mode="L").save(OUTPUT_DIR / "prob_new.png")
    Image.fromarray((prob_ref * 255).astype("uint8"), mode="L").save(OUTPUT_DIR / "prob_reference.png")

    blobs_new = scan_for_blobs(prob_new, new_composed, PROB_THRESHOLD, MIN_BLOB_SIZE, BLACK_THRESHOLD)
    blobs_ref = scan_for_blobs(prob_ref, reference, PROB_THRESHOLD, MIN_BLOB_SIZE, BLACK_THRESHOLD)
    print(f"\nTotal blobs above prob>{PROB_THRESHOLD}: reference={len(blobs_ref)}, new composite={len(blobs_new)}")
    print("(Most of these are false positives on both scenes per pipeline_report.pdf -- this total")
    print(" is context, not the main signal. The per-site table below is what matters.)")

    print("\n" + "=" * 100)
    print(f"{'site':<32} {'ref: found':<11} {'ref prob':>9} {'new: found':<11} {'new prob':>9} {'delta':>8}  status")
    print("=" * 100)

    held, dropped, lost = 0, 0, 0
    for site in KNOWN_SITES:
        ref_blob = nearest_blob(site["x"], site["y"], blobs_ref, MATCH_DISTANCE_PX)
        new_blob = nearest_blob(site["x"], site["y"], blobs_new, MATCH_DISTANCE_PX)

        ref_found = ref_blob is not None
        new_found = new_blob is not None
        ref_prob = ref_blob["max_prob"] if ref_blob else float("nan")
        new_prob = new_blob["max_prob"] if new_blob else float("nan")
        delta = (new_prob - ref_prob) if (ref_found and new_found) else float("nan")

        if not ref_found:
            status = "!! not even detected in the reference composite -- check KNOWN_SITES coordinates"
        elif new_found:
            held += 1
            status = "HELD" if delta > -0.15 else "HELD (confidence dropped notably)"
        else:
            lost += 1
            status = "LOST -- no matching blob in new composite"

        print(f"{site['label']:<32} {str(ref_found):<11} {ref_prob:>9.3f} {str(new_found):<11} "
              f"{new_prob:>9.3f} {delta:>8.3f}  {status}")

    print("=" * 100)
    print(f"\n{held}/{len(KNOWN_SITES)} known sites held their detection in the new composite; {lost} lost.")

    print("\n" + "=" * 60)
    if lost == 0:
        print("VERDICT: all known sites still detected on the best-effort composite (even if")
        print("confidence shifted somewhat). This is evidence the model is robust enough to this")
        print("tonal difference for practical use -- reasonable to proceed with the live backend")
        print("using this best-effort compositing, clearly documented as an approximation rather")
        print("than an exact reproduction of the training distribution.")
    else:
        print(f"VERDICT: {lost} known site(s) lost their detection entirely on the best-effort")
        print("composite. This is real evidence the model is NOT robust to this compositing gap --")
        print("a better percentile/gamma guess is not enough. Next steps:")
        print("  - search INPE's documentation specifically for \"TOM_PADRAO\" / \"tom padrao\" to find")
        print("    the actual calibration/rendering method behind that product name, rather than")
        print("    approximating it further")
        print("  - reconsider the live-processing feature's scope until that calibration is found")
        print("    or reproduced, since shipping it as-is risks silently missing real sites")
    print("=" * 60)


if __name__ == "__main__":
    main()
