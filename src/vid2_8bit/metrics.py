"""Quality metrics.

Two failure modes of this pipeline are hard to judge by eye on a scrubbing
timeline but trivial to measure, so they are measured:

* **Spatial noise** -- the salt-and-pepper speckle that naive pixelation
  produces on detailed subjects. Quantified as the fraction of pixels whose
  color matches none of their neighbours. Real pixel art is built from
  contiguous regions, so this number should be small; a noisy render pushes it
  up sharply.
* **Temporal churn** -- the fraction of pixels that change between consecutive
  frames. On a locked-off shot this should approach zero. Naive per-frame
  processing leaves it high even when nothing in the scene moves, which is the
  measurable signature of boiling.

Both run on rendered output, so they work as regression gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import decode


def color_index_map(frame: np.ndarray) -> tuple[np.ndarray, int]:
    """Map an image's exact colors to indices. Returns (index map, n_colors)."""
    flat = frame.reshape(-1, frame.shape[-1])
    packed = (
        flat[:, 0].astype(np.uint32) << 16
        | flat[:, 1].astype(np.uint32) << 8
        | flat[:, 2].astype(np.uint32)
    )
    _, inverse = np.unique(packed, return_inverse=True)
    return inverse.reshape(frame.shape[:2]).astype(np.int32), int(inverse.max()) + 1


def detect_cell_size(frame: np.ndarray, max_cell: int = 32) -> int:
    """Infer the upscale factor of an already-rendered pixel-art frame.

    Finds the period at which horizontal colour changes cluster. Needed because
    the metrics are meaningful at *logical* resolution -- measured on the
    upscaled image, every block interior trivially matches its neighbours and
    both numbers collapse toward zero regardless of quality.
    """
    gray = frame.mean(axis=2)
    changes = np.abs(np.diff(gray, axis=1)) > 0.5
    column_activity = changes.mean(axis=0)
    if column_activity.sum() == 0:
        return 1

    best, best_score = 1, 0.0
    for cell in range(1, min(max_cell, len(column_activity) // 4) + 1):
        on = column_activity[cell - 1 :: cell].mean() if cell > 1 else column_activity.mean()
        mask = np.ones(len(column_activity), dtype=bool)
        mask[cell - 1 :: cell] = False
        off = column_activity[mask].mean() if mask.any() else 0.0
        score = on - off * 2.0
        if score > best_score:
            best, best_score = cell, score
    return best


def isolated_pixel_ratio(idx: np.ndarray) -> float:
    """Fraction of pixels whose index differs from all four neighbours."""
    if idx.shape[0] < 3 or idx.shape[1] < 3:
        return 0.0
    core = idx[1:-1, 1:-1]
    same = (
        (core == idx[:-2, 1:-1])
        | (core == idx[2:, 1:-1])
        | (core == idx[1:-1, :-2])
        | (core == idx[1:-1, 2:])
    )
    return float((~same).mean())


def temporal_churn(prev_idx: np.ndarray, cur_idx: np.ndarray) -> float:
    """Fraction of pixels whose index changed between two frames."""
    if prev_idx.shape != cur_idx.shape:
        return float("nan")
    return float((prev_idx != cur_idx).mean())


@dataclass
class VideoMetrics:
    frames: int
    cell: int
    colors: int
    noise: float
    churn: float

    def __str__(self) -> str:
        return (
            f"frames analysed : {self.frames}\n"
            f"detected cell   : {self.cell}px\n"
            f"distinct colors : {self.colors}\n"
            f"spatial noise   : {self.noise * 100:.3f}%  "
            f"(isolated pixels; lower is cleaner)\n"
            f"temporal churn  : {self.churn * 100:.3f}%  "
            f"(pixels changed per frame; lower is stabler)"
        )


def report_video(path: str | Path, max_frames: int = 60) -> VideoMetrics:
    """Measure a rendered video."""
    noise_vals: list[float] = []
    churn_vals: list[float] = []
    prev: np.ndarray | None = None
    cell = 1
    colors = 0
    n = 0

    for i, frame in enumerate(decode.read_frames(path, count=max_frames)):
        if i == 0:
            cell = detect_cell_size(frame)
        # Sample one pixel per logical cell so metrics describe logical pixels.
        small = frame[cell // 2 :: cell, cell // 2 :: cell]
        idx, ncol = color_index_map(small)
        colors = max(colors, ncol)
        noise_vals.append(isolated_pixel_ratio(idx))
        if prev is not None:
            churn_vals.append(temporal_churn(prev, idx))
        prev = idx
        n += 1
        if n >= max_frames:
            break

    return VideoMetrics(
        frames=n,
        cell=cell,
        colors=colors,
        noise=float(np.mean(noise_vals)) if noise_vals else 0.0,
        churn=float(np.mean(churn_vals)) if churn_vals else 0.0,
    )


def compare_renders(paths: list[str | Path], max_frames: int = 40) -> str:
    """Tabulate metrics for several renders side by side."""
    rows = ["render                          cell  colors    noise%   churn%",
            "-" * 66]
    for p in paths:
        m = report_video(p, max_frames)
        rows.append(
            f"{Path(p).name[:30]:30s}  {m.cell:4d}  {m.colors:6d}  "
            f"{m.noise * 100:7.3f}  {m.churn * 100:7.3f}"
        )
    return "\n".join(rows)
