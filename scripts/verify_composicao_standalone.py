"""
Standalone verification: does composicao_cbers4a_wpm.py reproduce the RGB PNG(s) used to
train the garimpo model -- and if not, is a different percentile/gamma choice the fix?

Self-contained on purpose -- paste this whole file into a single Kaggle notebook cell and
run top to bottom. No garimpo/repo imports; only numpy, rasterio, and Pillow, all present
in Kaggle's default Python environment.

This is a grid-search version: raw band decoding (the expensive part, one rasterio read
per band) happens exactly once. Only the percentile-stretch + gamma step (cheap, pure
numpy) reruns per grid combination, so searching dozens of (p_min, p_max, gamma) triples
costs about the same as a handful of single runs.

Steps:
  1. Edit the CONFIG block below: the four band paths, REFERENCE_PNG, and the three grid
     lists if you want a different search range.
  2. Run All / execute the cell.
  3. Read the results table and the VERDICT at the bottom. diff.png / side_by_side.png
     for the single best combination are saved to OUTPUT_DIR.
"""

import itertools
from pathlib import Path
import re

import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image

try:
    from skimage.exposure import match_histograms as _sk_match_histograms
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False

# =============================================================================
# CONFIG -- edit these before running
# =============================================================================
BAND1_TIF = Path("/kaggle/input/datasets/raraabdel/projet-ufam/cbers4a_L4_bandas/CBERS4A_WPM_20201008_226_121_L4/CBERS_4A_WPM_20201008_226_121_L4_BAND1.tif")  # Blue
BAND2_TIF = Path("/kaggle/input/datasets/raraabdel/projet-ufam/cbers4a_L4_bandas/CBERS4A_WPM_20201008_226_121_L4/CBERS_4A_WPM_20201008_226_121_L4_BAND2.tif")  # Green
BAND3_TIF = Path("/kaggle/input/datasets/raraabdel/projet-ufam/cbers4a_L4_bandas/CBERS4A_WPM_20201008_226_121_L4/CBERS_4A_WPM_20201008_226_121_L4_BAND3.tif")  # Red
BAND4_TIF = Path("/kaggle/input/datasets/raraabdel/projet-ufam/cbers4a_L4_bandas/CBERS4A_WPM_20201008_226_121_L4/CBERS_4A_WPM_20201008_226_121_L4_BAND4.tif")  # NIR — unused

REFERENCE_PNG = Path("/kaggle/input/datasets/fareselhamdaoui/reference-png/CBERS_4A_WPM_20201008_226_121_L4_RGB_TrueColor_PanSharpened_TOM_PADRAO_V4.png")
OUTPUT_DIR = Path("/kaggle/working/verify_composicao")

# Geometry/resolution are already confirmed correct, so this stays fixed -- only the
# stretch (percentile/gamma) is being searched.
TAMANHO_PNG_MAX = 6000

# Grid search ranges. Edit freely; the Cartesian product of these three lists is what
# gets tried. Keep it "small" (per the ask) -- 4 x 5 x 4 = 80 combinations by default,
# which is cheap since raw bands are decoded only once total.
PERCENTIL_MIN_GRID = [0.5, 1, 2, 5]
PERCENTIL_MAX_GRID = [95, 98, 99, 99.5, 99.9]
GAMMA_GRID = [1.0, 1.2, 1.5, 1.8]

DIFF_THRESHOLD = 5  # 0-255 pixel-value gap counted as "differing" for the % stat
CLOSE_MATCH_MEAN_DIFF = 2.0
CLOSE_MATCH_PCT_DIFFERING = 1.0
ROUGH_MATCH_MEAN_DIFF = 8.0
ROUGH_MATCH_PCT_DIFFERING = 10.0

# Histogram matching: reference tiles whose per-channel color distribution the
# best-effort composite gets matched against, as an alternative to further
# percentile/gamma guessing. For a fair test these should be DIFFERENT tiles than
# REFERENCE_PNG -- matching against the exact same tile you're scoring against is
# circular and will look artificially perfect. If you only have one reference tile
# available, leave this as [REFERENCE_PNG] and the script will print a warning.
HISTOGRAM_MATCH_REFERENCE_TILES = [
    Path("/kaggle/input/datasets/fareselhamdaoui/tiles-test/CBERS_4A_WPM_20200707_226_120_L4_RGB_TrueColor_PanSharpened_TOM_PADRAO_V4_x1536_y4096.png"),
    Path("/kaggle/input/datasets/fareselhamdaoui/tiles-test/CBERS_4A_WPM_20200707_226_120_L4_RGB_TrueColor_PanSharpened_TOM_PADRAO_V4_x2048_y4096.png"),
    Path("/kaggle/input/datasets/fareselhamdaoui/tiles-test/CBERS_4A_WPM_20200707_226_120_L4_RGB_TrueColor_PanSharpened_TOM_PADRAO_V4_x2560_y3072.png"),
    Path("/kaggle/input/datasets/fareselhamdaoui/tiles-test/CBERS_4A_WPM_20200707_226_120_L4_RGB_TrueColor_PanSharpened_TOM_PADRAO_V4_x3072_y2560.png"),
    Path("/kaggle/input/datasets/fareselhamdaoui/tiles-test/CBERS_4A_WPM_20200707_226_120_L4_RGB_TrueColor_PanSharpened_TOM_PADRAO_V4_x3072_y3072.png"),
    Path("/kaggle/input/datasets/fareselhamdaoui/tiles-test/CBERS_4A_WPM_20200707_226_120_L4_RGB_TrueColor_PanSharpened_TOM_PADRAO_V4_x3584_y2560.png"),
    Path("/kaggle/input/datasets/fareselhamdaoui/tiles-test/CBERS_4A_WPM_20200707_226_121_L4_RGB_TrueColor_PanSharpened_TOM_PADRAO_V4_x1024_y0.png"),
    Path("/kaggle/input/datasets/fareselhamdaoui/tiles-test/CBERS_4A_WPM_20200707_226_121_L4_RGB_TrueColor_PanSharpened_TOM_PADRAO_V4_x1024_y512.png"),
    Path("/kaggle/input/datasets/fareselhamdaoui/tiles-test/CBERS_4A_WPM_20201008_226_121_L4_RGB_TrueColor_PanSharpened_TOM_PADRAO_V4_x1024_y512.png"),
]

# Pixels this dark (max channel value <= threshold) are treated as nodata/border and
# excluded from both the histogram-matching source/target pools and the permutation
# diagnostic -- same convention as garimpo.scan_scene's near_black_mask (threshold=10).
EXCLUDE_NEAR_BLACK_THRESHOLD = 10

# Histogram matching only counts as "the fix" if it beats the grid search's best mean
# absolute diff by at least this fraction (e.g. 0.2 = at least 20% lower).
HISTOGRAM_MATCH_IMPROVEMENT_FRACTION = 0.2

# =============================================================================
# Raw band decoding -- the expensive part. Runs exactly once per band, independent of
# the percentile/gamma grid.
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


# =============================================================================
# Percentile-stretch + gamma -- the cheap part, copied from composicao_cbers4a_wpm.py's
# per-band logic but taking p_min/p_max/gamma as arguments instead of module globals, and
# operating on a copy so the same raw array can be reused across grid combinations.
# =============================================================================
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
        [
            stretch_band(raw_r, p_min, p_max, gamma),
            stretch_band(raw_g, p_min, p_max, gamma),
            stretch_band(raw_b, p_min, p_max, gamma),
        ],
        axis=-1,
    )


# =============================================================================
# Alignment: figure out, once, how the composed scene lines up with the reference (full
# scene or a training tile), independent of the stretch parameters.
# =============================================================================
def resolve_alignment(composed_shape: tuple[int, int], reference: np.ndarray):
    """Returns a callable crop(composed_rgb) -> comparable region, or None if the
    reference can't be aligned to the composed scene at all (a geometry problem, not a
    stretch problem -- grid search can't fix this)."""
    tile_match = re.search(r"_x(\d+)_y(\d+)\.png$", REFERENCE_PNG.name)
    if tile_match:
        x0, y0 = int(tile_match.group(1)), int(tile_match.group(2))
        h, w = reference.shape[:2]
        print(f"Reference looks like a tile at offset (x={x0}, y={y0}); cropping composed scene to match.")
        if y0 + h > composed_shape[0] or x0 + w > composed_shape[1]:
            print("!! Crop falls outside the composed scene bounds -- composed scene is smaller than expected.")
            return None
        return lambda rgb: rgb[y0 : y0 + h, x0 : x0 + w]

    if composed_shape != reference.shape[:2]:
        print("!! Composed scene and reference full-scene PNG are different sizes:")
        print(f"!!   composed  = {composed_shape[1]} x {composed_shape[0]}")
        print(f"!!   reference = {reference.shape[1]} x {reference.shape[0]}")
        print("!! Resizing composed to reference size (nearest-neighbor) for comparison --")
        print("!! a size mismatch this large is itself evidence of a geometry problem, not a stretch problem.")
        size = (reference.shape[1], reference.shape[0])
        return lambda rgb: np.array(Image.fromarray(rgb).resize(size, Image.NEAREST))

    return lambda rgb: rgb


def _match_cdf_1d(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pure-numpy per-channel histogram matching (same cumulative-CDF algorithm
    scikit-image uses internally), for environments without scikit-image."""
    src_values, src_inverse, src_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    ref_values, ref_counts = np.unique(reference.ravel(), return_counts=True)

    src_cdf = np.cumsum(src_counts).astype(np.float64) / source.size
    ref_cdf = np.cumsum(ref_counts).astype(np.float64) / reference.size

    interp_ref_values = np.interp(src_cdf, ref_cdf, ref_values)
    return interp_ref_values[src_inverse].reshape(source.shape)


def load_reference_pixel_pool(paths: list[Path], exclude_near_black: int) -> list[np.ndarray]:
    """Reads each reference tile, drops near-black (nodata/border) pixels, and
    concatenates per-channel pixel values across all tiles. Returns [pool_r, pool_g,
    pool_b]."""
    pools: list[list[np.ndarray]] = [[], [], []]
    for path in paths:
        img = np.array(Image.open(path).convert("RGB"))
        mask = img.max(axis=-1) > exclude_near_black
        for c in range(3):
            pools[c].append(img[..., c][mask])
    return [
        np.concatenate(pool) if pool and sum(p.size for p in pool) > 0 else np.array([], dtype="uint8")
        for pool in pools
    ]


def match_histograms_rgb(
    source: np.ndarray, reference_pools: list[np.ndarray], exclude_near_black: int
) -> np.ndarray:
    """source: (H,W,3) uint8. reference_pools: [pool_r, pool_g, pool_b], each a 1D array
    of reference pixel values for that channel (see load_reference_pixel_pool). Matches
    each channel's histogram independently over non-near-black pixels only, leaving
    near-black (border) pixels untouched. Returns (H,W,3) uint8."""
    mask = source.max(axis=-1) > exclude_near_black
    matched = source.astype(np.float64).copy()

    for c in range(3):
        pool = reference_pools[c]
        src_channel = source[..., c]
        src_valid = src_channel[mask]
        if src_valid.size == 0 or pool.size == 0:
            continue

        if _HAVE_SKIMAGE:
            try:
                matched_valid = _sk_match_histograms(src_valid, pool)
            except Exception as exc:  # fall back rather than abort the whole run
                print(f"  [warn] skimage.exposure.match_histograms failed ({exc}); using numpy fallback")
                matched_valid = _match_cdf_1d(src_valid, pool)
        else:
            matched_valid = _match_cdf_1d(src_valid, pool)

        channel_out = matched[..., c]
        channel_out[mask] = matched_valid
        matched[..., c] = channel_out

    return np.clip(matched, 0, 255).astype("uint8")


def permutation_diagnostic(
    raw_r: np.ndarray, raw_g: np.ndarray, raw_b: np.ndarray,
    p_min: float, p_max: float, gamma: float,
    crop, reference: np.ndarray,
) -> list[tuple]:
    """Tries all 6 assignments of the 3 (already-decoded) raw bands to the R/G/B output
    channels, using one fixed stretch. A real band-assignment bug should show up as a
    large diff swing between permutations; if all 6 land in a similar range, the problem
    is elsewhere (not which band goes to which channel)."""
    stretched = {
        "BAND3": stretch_band(raw_r, p_min, p_max, gamma),
        "BAND2": stretch_band(raw_g, p_min, p_max, gamma),
        "BAND1": stretch_band(raw_b, p_min, p_max, gamma),
    }
    results = []
    for r_band, g_band, b_band in itertools.permutations(stretched.keys()):
        composed = np.stack([stretched[r_band], stretched[g_band], stretched[b_band]], axis=-1)
        candidate = crop(composed)
        if candidate.shape != reference.shape:
            continue
        mean_abs_diff, max_diff, pct_differing, _ = diff_stats(candidate, reference)
        results.append((mean_abs_diff, max_diff, pct_differing, r_band, g_band, b_band))
    results.sort(key=lambda r: r[0])
    return results


def diff_stats(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, int, float, np.ndarray]:
    diff = np.abs(candidate.astype("int16") - reference.astype("int16")).astype("uint8")
    diff_gray = diff.max(axis=-1)  # worst channel per pixel, so color-only fringing still shows up
    mean_abs_diff = float(diff_gray.mean())
    max_diff = int(diff_gray.max())
    pct_differing = float((diff_gray > DIFF_THRESHOLD).mean() * 100)
    return mean_abs_diff, max_diff, pct_differing, diff_gray


# =============================================================================
# Grid search
# =============================================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading raw bands (once) ...")
    raw_r = read_band_raw(BAND3_TIF, TAMANHO_PNG_MAX)  # R (BAND3)
    raw_g = read_band_raw(BAND2_TIF, TAMANHO_PNG_MAX)  # G (BAND2)
    raw_b = read_band_raw(BAND1_TIF, TAMANHO_PNG_MAX)  # B (BAND1)
    composed_shape = raw_r.shape
    print(f"Composed scene size (fixed for all grid combos): {composed_shape[1]} x {composed_shape[0]} px")

    reference = np.array(Image.open(REFERENCE_PNG).convert("RGB"))
    print(f"Reference image: {reference.shape[1]} x {reference.shape[0]} px  ({REFERENCE_PNG.name})")

    crop = resolve_alignment(composed_shape, reference)
    if crop is None:
        print("\nVERDICT: cannot compare -- this is a geometry/resolution mismatch, not a")
        print("percentile/gamma problem. Grid search would not help; re-check TAMANHO_PNG_MAX")
        print("and confirm the reference actually corresponds to these four band files.")
        return

    grid = list(itertools.product(PERCENTIL_MIN_GRID, PERCENTIL_MAX_GRID, GAMMA_GRID))
    print(f"\nRunning grid search over {len(grid)} (percentil_min, percentil_max, gamma) combinations ...")

    results = []
    for p_min, p_max, gamma in grid:
        composed = compose_true_color(raw_r, raw_g, raw_b, p_min, p_max, gamma)
        candidate = crop(composed)
        if candidate.shape != reference.shape:
            continue  # shouldn't happen once alignment is resolved, but guard anyway
        mean_abs_diff, max_diff, pct_differing, _ = diff_stats(candidate, reference)
        results.append((mean_abs_diff, max_diff, pct_differing, p_min, p_max, gamma))

    if not results:
        print("\nVERDICT: no grid combination produced a comparable shape. Cannot proceed.")
        return

    results.sort(key=lambda r: r[0])  # best (lowest mean abs diff) first

    print("\n" + "=" * 78)
    print(f"{'p_min':>7} {'p_max':>7} {'gamma':>7} {'mean|diff|':>11} {'max diff':>9} {'% >thresh':>10}")
    print("=" * 78)
    for mean_abs_diff, max_diff, pct_differing, p_min, p_max, gamma in results:
        print(f"{p_min:>7} {p_max:>7} {gamma:>7} {mean_abs_diff:>11.2f} {max_diff:>9d} {pct_differing:>9.2f}%")
    print("=" * 78)

    best_mean_abs_diff, best_max_diff, best_pct_differing, best_p_min, best_p_max, best_gamma = results[0]

    # Re-render and save the best combination's diff/side-by-side images.
    best_composed = compose_true_color(raw_r, raw_g, raw_b, best_p_min, best_p_max, best_gamma)
    best_candidate = crop(best_composed)
    _, _, _, best_diff_gray = diff_stats(best_candidate, reference)

    diff_path = OUTPUT_DIR / "diff.png"
    Image.fromarray(best_diff_gray, mode="L").save(diff_path)

    side_by_side = Image.new("RGB", (best_candidate.shape[1] * 2, best_candidate.shape[0]))
    side_by_side.paste(Image.fromarray(best_candidate), (0, 0))
    side_by_side.paste(Image.fromarray(reference), (best_candidate.shape[1], 0))
    side_by_side_path = OUTPUT_DIR / "side_by_side.png"
    side_by_side.save(side_by_side_path)

    print(f"\nBEST COMBINATION: percentil_min={best_p_min}, percentil_max={best_p_max}, gamma={best_gamma}")
    print(f"  Mean absolute difference (0-255 scale): {best_mean_abs_diff:.2f}")
    print(f"  Max difference (any pixel, any channel): {best_max_diff}")
    print(f"  % of pixels differing by more than {DIFF_THRESHOLD}/255: {best_pct_differing:.2f}%")
    print(f"  Diff image saved to:         {diff_path}")
    print(f"  Side-by-side image saved to: {side_by_side_path}")

    print("\n" + "=" * 60)
    if best_mean_abs_diff < CLOSE_MATCH_MEAN_DIFF and best_pct_differing < CLOSE_MATCH_PCT_DIFFERING:
        print("VERDICT: found a close match. Lock TAMANHO_PNG_MAX from this run plus")
        print(f"percentil_min={best_p_min}, percentil_max={best_p_max}, gamma={best_gamma}")
        print("into the real pipeline instead of the script's original defaults.")
    elif best_mean_abs_diff < ROUGH_MATCH_MEAN_DIFF and best_pct_differing < ROUGH_MATCH_PCT_DIFFERING:
        print("VERDICT: best grid combination is close but not exact. Inspect diff.png --")
        print("if the remaining error looks like uniform brightness/contrast noise, a finer")
        print("grid around this combination may close the gap. If it looks structured (edges,")
        print("color casts in specific regions), it's probably not a stretch-parameter issue.")
    else:
        print("VERDICT: even the best grid combination differs meaningfully from the reference.")
        print("This suggests the mismatch is NOT a percentile/gamma issue. Worth checking next:")
        print("  - band-to-channel assignment (confirm R=BAND3, G=BAND2, B=BAND1 is actually right")
        print("    for this product, not e.g. a different band numbering or order)")
        print("  - a per-scene DYNAMIC stretch (min/max computed some other way than fixed percentiles,")
        print("    e.g. per-channel histogram equalization, or stats computed over a different region)")
        print("  - whether the reference PNG really was produced from these exact four band files")
        print("    (wrong scene, wrong date, or a product variant mismatch)")
    print("=" * 60)

    # =========================================================================
    # Histogram matching: an alternative to further percentile/gamma guessing. Takes the
    # grid search's best-effort composite (doesn't need to be exact) and matches its
    # per-channel color histogram against known-good reference tile(s).
    # =========================================================================
    print("\n" + "#" * 78)
    print("HISTOGRAM MATCHING")
    print("#" * 78)
    print(f"scikit-image available: {_HAVE_SKIMAGE} "
          f"({'using skimage.exposure.match_histograms' if _HAVE_SKIMAGE else 'using numpy CDF-matching fallback'})")

    if REFERENCE_PNG in HISTOGRAM_MATCH_REFERENCE_TILES:
        print("[warn] HISTOGRAM_MATCH_REFERENCE_TILES includes REFERENCE_PNG itself -- matching")
        print("[warn] against the exact tile you're scoring against is circular and will look")
        print("[warn] artificially good. Set HISTOGRAM_MATCH_REFERENCE_TILES to OTHER known-good")
        print("[warn] tiles for a trustworthy result.")

    reference_pools = load_reference_pixel_pool(HISTOGRAM_MATCH_REFERENCE_TILES, EXCLUDE_NEAR_BLACK_THRESHOLD)
    pool_sizes = {ch: pool.size for ch, pool in zip("RGB", reference_pools)}
    print(f"Reference pixel pool sizes (non-near-black): {pool_sizes}")

    matched = match_histograms_rgb(best_candidate, reference_pools, EXCLUDE_NEAR_BLACK_THRESHOLD)
    matched_mean_abs_diff, matched_max_diff, matched_pct_differing, matched_diff_gray = diff_stats(
        matched, reference
    )

    matched_path = OUTPUT_DIR / "matched.png"
    Image.fromarray(matched).save(matched_path)
    matched_diff_path = OUTPUT_DIR / "matched_diff.png"
    Image.fromarray(matched_diff_gray, mode="L").save(matched_diff_path)
    matched_side_by_side = Image.new("RGB", (matched.shape[1] * 2, matched.shape[0]))
    matched_side_by_side.paste(Image.fromarray(matched), (0, 0))
    matched_side_by_side.paste(Image.fromarray(reference), (matched.shape[1], 0))
    matched_side_by_side_path = OUTPUT_DIR / "matched_side_by_side.png"
    matched_side_by_side.save(matched_side_by_side_path)

    print(f"\nHistogram-matched result (source = grid search's best combination, "
          f"percentil_min={best_p_min}, percentil_max={best_p_max}, gamma={best_gamma}):")
    print(f"  Mean absolute difference (0-255 scale): {matched_mean_abs_diff:.2f}")
    print(f"  Max difference (any pixel, any channel): {matched_max_diff}")
    print(f"  % of pixels differing by more than {DIFF_THRESHOLD}/255: {matched_pct_differing:.2f}%")
    print(f"  Matched image saved to:      {matched_path}")
    print(f"  Diff image saved to:         {matched_diff_path}")
    print(f"  Side-by-side image saved to: {matched_side_by_side_path}")

    print(f"\nDIRECT COMPARISON: grid search best mean|diff| = {best_mean_abs_diff:.2f}  "
          f"vs.  histogram-matched mean|diff| = {matched_mean_abs_diff:.2f}")

    improved = matched_mean_abs_diff < best_mean_abs_diff * (1 - HISTOGRAM_MATCH_IMPROVEMENT_FRACTION)

    print("\n" + "=" * 60)
    if improved:
        print("VERDICT: histogram matching is meaningfully closer than grid search alone --")
        print(f"a {(1 - matched_mean_abs_diff / best_mean_abs_diff) * 100:.0f}% reduction in mean|diff|.")
        print("This is the fix: build histogram matching into the live pipeline as a")
        print("post-processing step applied to every newly-composited scene, using a small")
        print("fixed set of reference tiles (like HISTOGRAM_MATCH_REFERENCE_TILES here) as the")
        print("matching target -- not the grid-search percentile/gamma tuning.")
        print("=" * 60)
    else:
        print("VERDICT: histogram matching did NOT get meaningfully closer than grid search.")
        print("This is a more serious finding than a bad percentile/gamma choice: it suggests")
        print("the mismatch isn't just color statistics -- the composited image's underlying")
        print("geometric/structural content may not match the reference at all. Running the")
        print("cheap diagnostic for this now: swapping which raw band feeds which output")
        print("channel. A real band-assignment bug should show a large diff swing between")
        print("permutations, unlike the small variation seen across the percentile/gamma grid.")
        print("=" * 60)

        perm_results = permutation_diagnostic(raw_r, raw_g, raw_b, best_p_min, best_p_max, best_gamma, crop, reference)

        print("\n" + "-" * 60)
        print("BAND-ASSIGNMENT PERMUTATION DIAGNOSTIC")
        print("-" * 60)
        print(f"{'R':>7} {'G':>7} {'B':>7} {'mean|diff|':>11} {'max diff':>9} {'% >thresh':>10}")
        for mean_abs_diff, max_diff, pct_differing, r_band, g_band, b_band in perm_results:
            print(f"{r_band:>7} {g_band:>7} {b_band:>7} {mean_abs_diff:>11.2f} {max_diff:>9d} {pct_differing:>9.2f}%")

        perm_best = perm_results[0]
        canonical = next(r for r in perm_results if r[3:] == ("BAND3", "BAND2", "BAND1"))
        swing = canonical[0] / perm_best[0] if perm_best[0] > 0 else float("inf")

        print("\n" + "=" * 60)
        if perm_best[3:] != ("BAND3", "BAND2", "BAND1") and swing > 1.5:
            print(f"VERDICT: a different band assignment (R={perm_best[3]}, G={perm_best[4]}, "
                  f"B={perm_best[5]}) scores {swing:.1f}x better than the canonical R=BAND3/G=BAND2/B=BAND1")
            print("assignment. This looks like a real band-assignment issue -- check the product's")
            print("band numbering/order before trusting composicao_cbers4a_wpm.py as-is.")
        else:
            print("VERDICT: all 6 band-assignment permutations land in a similar range (no dramatic")
            print("swing). This rules out a simple channel-order bug. The mismatch is more likely a")
            print("genuinely different source (wrong scene/date/product variant) or a processing")
            print("step this script doesn't replicate at all -- worth re-confirming that these four")
            print("band files and REFERENCE_PNG really correspond to the same scene.")
        print("=" * 60)


if __name__ == "__main__":
    main()
