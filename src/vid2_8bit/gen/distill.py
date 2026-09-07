"""Distil a generative look into deterministic render settings.

The problem with running a diffusion model on every frame is not quality, it is
that quality is unstable. Diffusion re-rolls the image each frame; two adjacent
frames of a static shot come back different. This project spent considerable
effort getting temporal churn down to 0.01% on a locked-off shot, and per-frame
generation throws all of it away. On a 10 GB RDNA2 card it also costs hours per
clip rather than minutes.

So the generative model is used as an *art director*, not a renderer. It sees a
handful of keyframes, and what comes back is measured rather than kept:

* the palette it chose,
* how much contrast and chroma it applied,
* how coarse it made the blocks.

Those measurements become a preset, and the deterministic pipeline renders the
whole clip to that spec -- stable, fast, and reproducible. It is the difference
between copying an artist's painting and learning which colours they mixed.

Per-frame generation still has a place for single images, and `gen render`
covers that. This module is for video.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..color.spaces import srgb_to_oklab, u8_to_float

log = logging.getLogger(__name__)


@dataclass
class StyleFit:
    """What was measured from the generated keyframes."""

    palette: np.ndarray                       # (N, 3) float sRGB
    cell: float                               # detected block size, source px
    target_width: int
    contrast: float
    target_chroma: float
    even_spacing: float
    frames_used: int = 0
    notes: list[str] = field(default_factory=list)

    def to_preset(self, name: str = "distilled") -> dict:
        return {
            "description": (
                f"Distilled from {self.frames_used} generated keyframe(s). "
                f"Palette and grade measured from the generator's output; the "
                f"render itself stays deterministic."
            ),
            "extends": "tetris-movie",
            "sample": {"cell_size": None, "target_width": int(self.target_width),
                       "min_cell": 4.0},
            "palette": {"mode": "auto", "size": int(len(self.palette)),
                        "even_spacing": round(float(self.even_spacing), 3),
                        "redistribute": 1.0, "anchor_ink": True},
            "tone": {"contrast": round(float(self.contrast), 3),
                     "target_chroma": round(float(self.target_chroma), 4),
                     "auto_levels": True},
        }


def detect_cell(img: np.ndarray, max_cell: int = 32) -> float:
    """Estimate block size from where colour changes cluster.

    Generated pixel art is rarely on an exact grid -- the model paints something
    that looks blocky rather than sampling one -- so this returns the dominant
    period and the caller should treat it as an estimate.
    """
    gray = img.mean(axis=2).astype(np.float32)
    ax = (np.abs(np.diff(gray, axis=1)) > 6).mean(axis=0)
    ay = (np.abs(np.diff(gray, axis=0)) > 6).mean(axis=1)

    def best(activity: np.ndarray) -> tuple[int, float]:
        top, score = 1, 0.0
        for cell in range(2, min(max_cell, max(3, len(activity) // 8))):
            on = np.zeros(len(activity), dtype=bool)
            on[cell - 1 :: cell] = True
            s = float(activity[on].mean() - 3.0 * activity[~on].mean())
            if s > score:
                top, score = cell, s
        return top, score

    cx, sx = best(ax)
    cy, sy = best(ay)
    return float(cx if sx >= sy else cy)


def measure(frames: list[np.ndarray], palette_size: int = 24) -> StyleFit:
    """Measure renderable style parameters from generated keyframes."""
    from ..color.palette import fit_palette

    if not frames:
        raise ValueError("no keyframes to measure")

    cells, samples = [], []
    for f in frames:
        # Denoise first: generated images carry soft edges and JPEG-ish fringing
        # that would otherwise dominate both the cell estimate and the palette.
        den = cv2.bilateralFilter(cv2.medianBlur(f, 3), 9, 40, 9)
        c = detect_cell(den)
        cells.append(c)
        step = max(1, int(round(c)))
        samples.append(u8_to_float(den[step // 2 :: step, step // 2 :: step]).reshape(-1, 3))

    cell = float(np.median(cells))
    pixels = np.concatenate(samples)
    palette = fit_palette(pixels, palette_size, chroma_weight=1.2)

    lab = srgb_to_oklab(pixels)
    L = lab[..., 0]
    chroma = float(np.hypot(lab[..., 1], lab[..., 2]).mean())

    # Contrast: how two-pole is the lightness distribution? A generator that
    # committed to ink shadows and bright highlights leaves few mid-tones.
    mid = float(((L > 0.35) & (L < 0.65)).mean())
    contrast = float(np.clip(1.0 - mid * 2.2, 0.0, 1.0))

    # Even spacing: if the generator's own palette is well separated, ask the
    # renderer for a well-separated one too.
    plab = srgb_to_oklab(palette)
    gaps = np.diff(np.sort(plab[:, 0]))
    even = float(np.clip(1.0 - (gaps.std() / (gaps.mean() + 1e-6)), 0.2, 0.9))

    width = int(np.clip(round(frames[0].shape[1] / max(cell, 1.0)), 60, 480))
    notes = [
        f"cell {cell:.1f}px -> logical width {width}",
        f"mid-tone share {mid * 100:.1f}% -> contrast {contrast:.2f}",
        f"mean chroma {chroma:.4f}",
    ]
    return StyleFit(palette=palette, cell=cell, target_width=width,
                    contrast=contrast, target_chroma=chroma,
                    even_spacing=even, frames_used=len(frames), notes=notes)


def keyframe_indices(n_frames: int, count: int) -> list[int]:
    """Evenly spaced keyframes, avoiding the very first and last frames."""
    count = max(1, min(count, max(1, n_frames)))
    if n_frames <= count:
        return list(range(n_frames))
    return list(np.linspace(n_frames * 0.08, n_frames * 0.92, count).astype(int))


def save_palette_hex(palette: np.ndarray, path: str | Path) -> None:
    """Write the fitted palette as a .hex file, usable as a custom palette."""
    lines = ["".join(f"{v:02X}" for v in np.rint(c * 255).astype(int))
             for c in palette]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
