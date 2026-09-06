"""Shot boundary detection.

Shots are the unit of state in this pipeline. A palette fitted across a cut
averages two unrelated colour worlds and satisfies neither; temporal filtering
across a cut smears the last frame of one shot into the first of the next. So
everything -- palette fitting, flow, hysteresis, grid anchoring -- resets here.

Detection is a histogram-correlation cut detector rather than a dependency on
PySceneDetect: it runs on a heavily downscaled decode pass, needs only OpenCV,
and cut detection is the one job where a simple method is genuinely adequate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .decode import probe, read_frames


@dataclass
class Shot:
    index: int
    start: int
    end: int  # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"Shot({self.index}: {self.start}-{self.end}, {self.length}f)"


def _histogram(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    """Per-channel normalized histogram, concatenated."""
    out = []
    for c in range(3):
        h, _ = np.histogram(frame[..., c], bins=bins, range=(0, 256))
        out.append(h.astype(np.float32))
    v = np.concatenate(out)
    total = v.sum()
    return v / total if total > 0 else v


def detect_shots(
    path: str | Path,
    threshold: float = 0.35,
    min_length: int = 8,
    analysis_width: int = 320,
    sample_step: int = 2,
) -> list[Shot]:
    """Detect shot boundaries. Returns a list covering the whole clip.

    `threshold` is the histogram L1 distance above which a cut is declared;
    higher values detect fewer cuts. Analysis runs at `sample_step` frame
    intervals, so boundaries are accurate to within that many frames -- which is
    fine, since the cost of being a frame or two late is one slightly-wrong
    palette frame.
    """
    info = probe(path)
    prev: np.ndarray | None = None
    distances: list[tuple[int, float]] = []

    for i, frame in enumerate(
        read_frames(path, step=sample_step, scale_width=analysis_width)
    ):
        hist = _histogram(frame)
        if prev is not None:
            distances.append((i * sample_step, float(np.abs(hist - prev).sum())))
        prev = hist

    total = info.n_frames or (len(distances) + 1) * sample_step
    boundaries = [0]
    for frame_idx, dist in distances:
        if dist > threshold and frame_idx - boundaries[-1] >= min_length:
            boundaries.append(frame_idx)
    boundaries.append(total)

    shots = [
        Shot(i, s, e)
        for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:]))
        if e > s
    ]
    return shots or [Shot(0, 0, total)]


def single_shot(path: str | Path) -> list[Shot]:
    """Treat the whole clip as one shot (skips the analysis decode pass)."""
    info = probe(path)
    return [Shot(0, 0, info.n_frames or 1)]
