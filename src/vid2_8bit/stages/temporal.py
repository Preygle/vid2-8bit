"""Stage 5 -- temporal stabilization.

Everything upstream is a per-frame function. Run it independently on each frame
of a static shot and the result still shakes: pixels sitting near a palette
boundary flip between two colors, the dither pattern crawls, and edges boil.
That flicker is far more objectionable than any single-frame artifact, because
real pixel art is *stable* -- an unchanged region is byte-identical frame to
frame.

Four mechanisms, applied at logical resolution:

1. **Grid anchoring** -- the sampling grid follows camera translation in whole
   source pixels, so a pan moves the image *through* a scene-locked grid instead
   of sliding it under a screen-locked one.
2. **Flow-guided blending** -- the previous logical frame, warped by optical
   flow, is mixed into the current one wherever the warp is trustworthy.
3. **Occlusion masking** -- forward/backward flow consistency decides where it
   is not trustworthy, so newly revealed areas are not smeared.
4. **Index hysteresis** -- applied downstream in the quantizer, using the
   previous index map carried here.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray_u8(img_srgb: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img_srgb.mean(axis=2) * 255.0), 0, 255).astype(np.uint8)


class FlowEstimator:
    """Optical flow, tier-selected.

    DIS is used for the fast tier; it is roughly two orders of magnitude cheaper
    than a neural estimator and accurate enough for the small displacements
    between adjacent frames at logical resolution. RAFT, when a cached assist
    pass has provided it, is used for the quality tier.
    """

    def __init__(self, mode: str = "dis"):
        self.mode = mode
        self._dis = None
        if mode != "none":
            preset = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
            self._dis = cv2.DISOpticalFlow_create(preset)
            self._dis.setUseSpatialPropagation(True)

    def __call__(self, prev_srgb: np.ndarray, cur_srgb: np.ndarray) -> np.ndarray | None:
        if self._dis is None:
            return None
        return self._dis.calc(to_gray_u8(prev_srgb), to_gray_u8(cur_srgb), None)


def warp_by_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp `img` forward along `flow`."""
    h, w = img.shape[:2]
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
    map_xy = grid + flow
    return cv2.remap(
        img,
        map_xy[..., 0],
        map_xy[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, tol: float = 1.0) -> np.ndarray:
    """1 where the forward flow is self-consistent, 0 where it is not.

    A pixel whose forward flow followed by the backward flow does not return it
    near its origin has been occluded or revealed. Blending there drags stale
    color into newly visible areas -- the classic ghosting trail.
    """
    warped_bwd = warp_by_flow(flow_bwd, flow_fwd)
    residual = flow_fwd + warped_bwd
    err = np.sqrt((residual**2).sum(-1))
    return np.clip(1.0 - (err / max(tol, 1e-6)), 0.0, 1.0).astype(np.float32)


def estimate_global_motion(prev_srgb: np.ndarray, cur_srgb: np.ndarray) -> tuple[float, float]:
    """Dominant translation (dy, dx) in pixels, via phase correlation.

    Phase correlation rather than feature matching: it is a single FFT pair, has
    no failure mode on low-texture frames like skies, and camera translation is
    the only component grid anchoring can act on anyway.
    """
    prev = prev_srgb.mean(axis=2).astype(np.float32)
    cur = cur_srgb.mean(axis=2).astype(np.float32)
    if prev.shape != cur.shape:
        return 0.0, 0.0
    try:
        (dx, dy), response = cv2.phaseCorrelate(prev, cur)
    except cv2.error:
        return 0.0, 0.0
    # A weak peak means no coherent global motion; forcing a shift then would
    # make the grid jitter on its own.
    if response < 0.05:
        return 0.0, 0.0
    return float(dy), float(dx)


class TemporalState:
    """Per-shot temporal state. Reset at every cut."""

    def __init__(self, cfg, cell: float):
        self.cfg = cfg
        self.cell = max(1.0, float(cell))
        self.flow = FlowEstimator(cfg.temporal.flow if cfg.temporal.enabled else "none")
        self.prev_logical: np.ndarray | None = None
        self.prev_idx: np.ndarray | None = None
        self.prev_source_small: np.ndarray | None = None
        self._accum = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self.prev_logical = None
        self.prev_idx = None
        self.prev_source_small = None
        self._accum[:] = 0.0

    # -- grid anchoring ----------------------------------------------------

    def grid_offset(self, source_srgb: np.ndarray) -> tuple[int, int]:
        """Sampling-grid offset in whole source pixels, tracking the camera."""
        if not (self.cfg.temporal.enabled and self.cfg.temporal.grid_anchor):
            return 0, 0
        # Motion is estimated at reduced resolution; it is a global translation,
        # so full resolution buys nothing but time.
        small = cv2.resize(source_srgb, (0, 0), fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_AREA)
        if self.prev_source_small is not None:
            dy, dx = estimate_global_motion(self.prev_source_small, small)
            self._accum += np.array([dy * 4.0, dx * 4.0])
        self.prev_source_small = small
        cell = self.cell
        return (
            int(round(self._accum[0])) % max(1, int(round(cell))),
            int(round(self._accum[1])) % max(1, int(round(cell))),
        )

    # -- flow-guided blending ---------------------------------------------

    def stabilize(self, logical_srgb: np.ndarray) -> np.ndarray:
        """Blend in the flow-warped previous logical frame."""
        cfg = self.cfg.temporal
        if not cfg.enabled or cfg.blend <= 0.0 or self.prev_logical is None:
            self.prev_logical = logical_srgb.copy()
            return logical_srgb

        prev = self.prev_logical
        if prev.shape != logical_srgb.shape:
            self.prev_logical = logical_srgb.copy()
            return logical_srgb

        flow_fwd = self.flow(prev, logical_srgb)
        if flow_fwd is None:
            self.prev_logical = logical_srgb.copy()
            return logical_srgb

        flow_bwd = self.flow(logical_srgb, prev)
        warped = warp_by_flow(prev, flow_fwd)
        mask = occlusion_mask(flow_fwd, flow_bwd, tol=1.5)[..., None]

        alpha = float(np.clip(cfg.blend, 0.0, 0.95)) * mask
        out = np.clip(logical_srgb * (1.0 - alpha) + warped * alpha, 0.0, 1.0)
        out = out.astype(np.float32)
        self.prev_logical = out.copy()
        return out

    # -- index carry-over --------------------------------------------------

    def hysteresis_args(self) -> dict:
        cfg = self.cfg.temporal
        if not cfg.enabled or cfg.hysteresis <= 0.0:
            return {}
        return {"prev_idx": self.prev_idx, "hysteresis": float(cfg.hysteresis)}

    def note_indices(self, idx: np.ndarray) -> None:
        self.prev_idx = idx


def decimate_plan(n_frames: int, src_fps: float, target_fps: float) -> list[int]:
    """Map output frame -> source frame index for on-twos style animation.

    Rendering at 12-15fps and holding is period-accurate (8-bit games could not
    animate faster) and, usefully, hides whatever residual flicker survives the
    other three mechanisms.
    """
    if target_fps <= 0 or target_fps >= src_fps:
        return list(range(n_frames))
    hold = max(1, int(round(src_fps / target_fps)))
    return [min(n_frames - 1, (i // hold) * hold) for i in range(n_frames)]
