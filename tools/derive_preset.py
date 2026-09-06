"""Fit a preset to reference frames.

Point this at stills from the look you want to match -- e.g. frames from the
*Tetris* (2023) pixel sequences in ``references/tetris/`` -- and it measures the
style numerically instead of leaving you to guess:

* **cell size** -- the upscale factor, recovered from where colour changes
  cluster along each axis. This is the single most important number and the
  easiest to get wrong by eye.
* **palette** -- the actual colours used, and how many, clustered across all
  reference frames together so one frame's colour cast does not dominate.
* **shading bands** -- distinct lightness levels, which tells you how many steps
  the artist used per material.
* **outline weight** -- how much of the image is dark boundary pixels.
* **dither presence** -- detected from checkerboard-like alternation, which is
  what distinguishes an ordered dither from flat fill.

The output is a YAML overlay to drop into ``src/vid2_8bit/presets/``.

Usage:
    python tools/derive_preset.py references/tetris/*.png -o tetris-movie.yaml
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vid2_8bit.color.palette import fit_palette  # noqa: E402
from vid2_8bit.color.spaces import srgb_to_oklab, u8_to_float  # noqa: E402


def detect_cell_size(frame: np.ndarray, max_cell: int = 24) -> int:
    """Recover the upscale factor from the periodicity of colour changes.

    In upscaled pixel art, colour changes can only happen on cell boundaries.
    So the correct cell size is the period that captures nearly all changes
    while leaving the off-period columns almost empty.
    """
    gray = frame.mean(axis=2)
    changes_x = (np.abs(np.diff(gray, axis=1)) > 6).mean(axis=0)
    changes_y = (np.abs(np.diff(gray, axis=0)) > 6).mean(axis=1)

    def best_period(activity: np.ndarray) -> tuple[int, float]:
        best, best_score = 1, 0.0
        for cell in range(2, min(max_cell, len(activity) // 8)):
            on_mask = np.zeros(len(activity), dtype=bool)
            on_mask[cell - 1 :: cell] = True
            on = activity[on_mask].mean()
            off = activity[~on_mask].mean() if (~on_mask).any() else 0.0
            # Reward periods that concentrate change and leave gaps clean.
            score = on - 3.0 * off
            if score > best_score:
                best, best_score = cell, score
        return best, best_score

    cx, sx = best_period(changes_x)
    cy, sy = best_period(changes_y)
    return cx if sx >= sy else cy


def analyse(paths: list[Path], max_colors: int = 64) -> dict:
    frames, cells, samples = [], [], []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  skipped (unreadable): {p}", file=sys.stderr)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        cell = detect_cell_size(rgb)
        cells.append(cell)
        # Sample one pixel per cell so each logical pixel counts once, rather
        # than weighting by how many screen pixels it happens to occupy.
        logical = rgb[cell // 2 :: cell, cell // 2 :: cell]
        samples.append(u8_to_float(logical).reshape(-1, 3))
        print(f"  {p.name}: {rgb.shape[1]}x{rgb.shape[0]}, cell {cell}px")

    if not frames:
        raise SystemExit("no readable reference images")

    cell = int(np.median(cells))
    pixels = np.concatenate(samples)

    # Distinct colours actually present, at 8-bit precision.
    quantized = np.unique(np.rint(pixels * 255).astype(np.uint8), axis=0)
    n_distinct = len(quantized)
    palette_size = int(min(max_colors, max(4, n_distinct)))
    palette = fit_palette(pixels, palette_size)

    lab = srgb_to_oklab(pixels)
    # Shading bands: count clusters of lightness, which is the artist's ramp.
    hist, edges = np.histogram(lab[:, 0], bins=48, range=(0, 1))
    peaks = int(((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:])
                 & (hist[1:-1] > hist.max() * 0.05)).sum())

    # Outline weight: share of logical pixels in the darkest decile.
    dark_threshold = float(np.percentile(lab[:, 0], 10))
    outline_share = float((lab[:, 0] <= dark_threshold).mean())

    # Dither detection: in a dithered region, a pixel differs from both
    # horizontal neighbours but matches the one two steps away.
    dither_score = 0.0
    for rgb, c in zip(frames, cells):
        logical = rgb[c // 2 :: c, c // 2 :: c].astype(np.int16)
        if logical.shape[1] < 5:
            continue
        d1 = np.abs(logical[:, 1:-1] - logical[:, :-2]).sum(-1) > 12
        d2 = np.abs(logical[:, 2:] - logical[:, :-2]).sum(-1) < 12
        dither_score = max(dither_score, float((d1[:, : d2.shape[1]] & d2).mean()))

    mean_chroma = float(np.abs(lab[:, 1:]).mean())

    return {
        "cell": cell,
        "palette_size": palette_size,
        "palette": palette,
        "n_distinct": n_distinct,
        "bands": max(3, min(12, peaks)),
        "outline_share": outline_share,
        "dither_score": dither_score,
        "mean_chroma": mean_chroma,
        "frames": len(frames),
    }


def to_yaml(a: dict, name: str) -> str:
    dithered = a["dither_score"] > 0.06
    hexes = ["".join(f"{v:02X}" for v in np.rint(c * 255).astype(int))
             for c in a["palette"]]
    lines = [
        f"description: >",
        f"  Derived from {a['frames']} reference frame(s) by tools/derive_preset.py.",
        f"  {a['n_distinct']} distinct colours observed; dither score "
        f"{a['dither_score']:.3f}.",
        "extends: base",
        "sample:",
        f"  cell_size: {a['cell']}",
        "  method: outline_expand",
        "palette:",
        "  mode: auto",
        f"  size: {a['palette_size']}",
        "abstract:",
        "  texture_removal: 0.35",
        f"  luma_bands: {a['bands']}",
        f"  saturation: {min(1.4, max(0.9, a['mean_chroma'] / 0.055)):.2f}",
        "structure:",
        "  outlines: true",
        f"  outline_strength: {min(1.0, a['outline_share'] * 4.0):.2f}",
        "dither:",
        f"  mode: {'bayer4' if dithered else 'none'}",
        f"  amount: {0.4 if dithered else 0.0:.2f}",
        "",
        "# Observed palette, for reference or for use as a custom palette file:",
    ]
    lines += [f"#   #{h}" for h in hexes]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="reference frames (globs allowed)")
    ap.add_argument("-o", "--output", default=None, help="write YAML here")
    ap.add_argument("--name", default="derived")
    ap.add_argument("--max-colors", type=int, default=64)
    args = ap.parse_args()

    paths: list[Path] = []
    for pattern in args.images:
        matched = [Path(p) for p in glob.glob(pattern)]
        paths.extend(matched or ([Path(pattern)] if Path(pattern).exists() else []))
    if not paths:
        print("no images matched", file=sys.stderr)
        return 1

    print(f"analysing {len(paths)} reference image(s):")
    result = analyse(paths, args.max_colors)
    print(
        f"\nderived: cell {result['cell']}px, palette {result['palette_size']}, "
        f"{result['bands']} bands, outline share {result['outline_share']:.3f}, "
        f"dither {result['dither_score']:.3f}"
    )

    yaml_text = to_yaml(result, args.name)
    if args.output:
        Path(args.output).write_text(yaml_text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print("\n" + yaml_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
