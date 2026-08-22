# Dataset Limitation: Mosaic-Seam / Nodata Proximity in Positive Labels

## Background

CBERS-4A scene mosaics can leave dark stitching seams and nodata borders that visually
resemble water or shadow. Since garimpo detection boxes were manually drawn on RGB
mosaics, there's a risk that some labeled "positive" boxes actually sit on such an
artifact rather than on a genuine mining site.

## Methodology

`src/garimpo/audit_artifacts.py` runs two checks:

- **Part 1 (tile-level)**: compares the fraction of near-black pixels (R,G,B < 10) between
  positive and negative tiles, using a one-sided Mann-Whitney U test (H1: positives have
  more near-black content). Result: **not significant** (p = 0.47) — no tile-level
  evidence that positive tiles are systematically artifact-heavy.
- **Part 2 (box-level, more direct)**: for each labeled box, finds the nearest large
  (> 500 px) connected near-black blob within its own tile and measures the distance from
  the box to that blob. Boxes within 100 px are flagged, then split by whether the
  nearest blob touches the tile border — mosaic/nodata seams always originate at a scene
  edge, so a border-touching blob is consistent with a seam artifact, while a fully
  interior blob is more consistent with a cloud shadow.

Reproduce with: `python -m garimpo.audit_artifacts` (defaults: black_threshold=10,
min_blob_size=500px, suspicious_distance=100px).

## Findings

Of the original 39 positive tiles / 43 boxes, **7 boxes** were flagged as suspiciously
close (< 100px) to a large near-black blob.

### 1 excluded as a data-quality issue

Visually confirmed as sitting on a hard mosaic stitching line, not clean sensor data:

| Filename | Box (xmin,ymin,xmax,ymax) | Distance to seam |
|---|---|---|
| `CBERS_4A_WPM_20230728_226_120_L4_RGB_TrueColor_PanSharpened_TOM_PADRAO_V4_x512_y4096.png` | (270,477,292,512) | 1.41 px |

Action taken: label changed from `positive` to `negative` in `labels.csv` (the tile
itself is valid image content and is retained as a negative example — only the box was
invalid); the box row removed from `bboxes.csv`; mask regenerated as all-zero for this
tile.

### 6 kept, flagged as border-touching (not visually confirmed, retained pending further review)

Nearest large near-black blob touches the tile edge — consistent with a mosaic/nodata
seam, but not independently confirmed as a mislabel:

| Filename | Box (xmin,ymin,xmax,ymax) | Distance to seam |
|---|---|---|
| `..._20200707_226_120_..._x3072_y2560.png` | (497,506,512,525) | 33.12 px |
| `..._20220921_226_121_..._x1536_y512.png` | (283,413,302,435) | 42.01 px |
| `..._20210613_226_120_..._x3584_y2560.png` | (503,504,519,525) | 42.72 px |
| `..._20230728_226_120_..._x2560_y3072.png` | (64,222,81,238) | 50.12 px |
| `..._20200707_226_121_..._x1024_y0.png` | (130,381,152,412) | 80.06 px |
| `..._20230627_226_120_..._x5120_y1024.png` | (366,424,382,438) | 83.15 px |

### 3 kept, flagged as interior (lower concern)

Nearest large near-black blob is fully interior — more consistent with a cloud shadow
than scene nodata:

| Filename | Box (xmin,ymin,xmax,ymax) | Distance to blob |
|---|---|---|
| `..._20230627_226_120_..._x2048_y3072.png` | (403,306,422,330) | 55.57 px |
| `..._20210613_226_120_..._x1536_y4096.png` | (272,142,287,154) | 55.97 px |
| `..._20210613_226_120_..._x2048_y4096.png` | (359,379,374,394) | 88.29 px |

## Dataset status after this correction

38 positive / 1254 negative / 11 uncertain tiles (1303 total; uncertain tiles excluded
from training, see `src/garimpo/masks.py`). 42 boxes across the 38 positive tiles (4
tiles have 2 boxes each). The 9 remaining flagged boxes above were left as-is — this
audit is diagnostic, not a relabeling pass, so only the one visually-confirmed case was
corrected.

## Related limitation: duplicate box labels at tile-grid corners

Discovered while building full-scene evaluation (`src/garimpo/evaluate_detections.py`).
Boxes were labeled per-tile on the non-overlapping 512px tiling grid, so a real site
sitting near a corner shared by several adjacent tiles gets an independent box drawn in
each tile it touches — the same physical site, labeled multiple times.

Evidence: across all 12 scenes with ground-truth boxes, nearest-neighbor distance
between boxes in the same scene is cleanly bimodal — either exactly 0px (literal
bounding-box overlap) or 59px+, nothing in between. 14 of the 42 boxes fall into 6
overlapping clusters (five pairs, one group of four at a single site in
`20220921_226_120`), reducing 42 raw boxes to **34 unique sites** once merged.

This doesn't affect tile-level training (each tile's mask is rasterized independently,
so a duplicate-labeled site just means its mask is correctly filled in each of the
tiles it appears in — no double-counting there). It only matters for full-scene,
absolute-coordinate evaluation: matching detections against the 42 raw boxes under
strict one-to-one matching manufactures 8 false negatives that are actually the same
site being counted multiple times against a single correct detection.
`evaluate_detections.py` merges boxes within 0px of each other (i.e. literal overlap)
into one site by default (`--merge-duplicate-boxes`, on by default;
`--no-merge-duplicate-boxes` to see the raw/unmerged numbers).

Last updated: 2026-08-21.
