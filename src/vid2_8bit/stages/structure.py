"""Stage 2 -- structure extraction.

Abstraction (Stage 1) removes detail. This stage decides which detail gets
*re-authored* as explicit line work, because that is what a pixel artist
actually does: a building is not a sampled photograph of a building, it is a
flat facade plus a deliberate 1-pixel silhouette and a regular grid of window
marks.

Lines are extracted at source resolution, where they are still well-resolved,
and composited back at logical resolution, where they land as crisp single
pixels instead of being averaged into grey.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..color.spaces import srgb_to_oklab


def luminance(img_srgb: np.ndarray) -> np.ndarray:
    """Perceptual lightness (Oklab L) as a single channel."""
    return srgb_to_oklab(img_srgb)[..., 0]


def xdog(
    gray: np.ndarray,
    sigma: float = 0.8,
    k: float = 1.6,
    tau: float = 0.98,
    phi: float = 12.0,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Extended difference-of-Gaussians edge response.

    Returns line *strength* in [0, 1], where 1 is a strong line. XDoG is used
    rather than Canny because it yields a continuous, thickness-controllable
    response tuned for stylized art, whereas Canny gives a binary mask whose
    lines break up under downsampling.
    """
    g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma * k)
    diff = g1 - tau * g2
    # Soft threshold: flat above epsilon, rolling off sharply below it.
    response = np.where(
        diff >= epsilon, 1.0, 1.0 + np.tanh(phi * (diff - epsilon))
    ).astype(np.float32)
    return np.clip(1.0 - response, 0.0, 1.0)


def depth_gate(lines: np.ndarray, depth: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Suppress lines that are not at a depth discontinuity.

    This is what separates a *silhouette* from interior texture. A window frame
    and a brick course produce similar luminance edges; only the building's
    outer edge produces a depth step. Gating on depth gives the clean
    outer-outline look the reference style depends on.
    """
    if depth is None:
        return lines
    if depth.shape != lines.shape:
        depth = cv2.resize(depth, (lines.shape[1], lines.shape[0]), interpolation=cv2.INTER_LINEAR)
    d = depth.astype(np.float32)
    rng = float(d.max() - d.min())
    if rng < 1e-6:
        return lines
    d = (d - d.min()) / rng
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.max() > 1e-6:
        mag /= mag.max()
    gate = np.clip(mag * 4.0, 0.0, 1.0)
    # Blend rather than hard-mask, so interior lines fade instead of vanishing.
    return (lines * (1.0 - strength + strength * gate)).astype(np.float32)


def snap_axis_lines(mask: np.ndarray, min_run: int = 3) -> np.ndarray:
    """Straighten near-horizontal and near-vertical runs at logical resolution.

    A photographed building edge lands on the pixel grid as a ragged staircase.
    Hand-drawn pixel art does not look like that -- axis-aligned architectural
    edges are drawn as perfectly straight runs. Morphological opening along each
    axis keeps only pixels that belong to a genuine run, which removes the
    single-pixel ragged spurs while leaving real structure intact.
    """
    binary = (mask > 0.35).astype(np.uint8)
    h_kernel = np.ones((1, max(2, min_run)), np.uint8)
    v_kernel = np.ones((max(2, min_run), 1), np.uint8)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    straight = np.maximum(horizontal, vertical).astype(np.float32)
    # Keep original strength where a run was confirmed; drop isolated spurs to
    # half strength rather than removing them, so detail is thinned not lost.
    return np.maximum(straight * mask, mask * 0.5).astype(np.float32)


def extract_lines(
    img_srgb: np.ndarray,
    cfg,
    depth: np.ndarray | None = None,
) -> np.ndarray:
    """Full Stage 2 at source resolution. Returns line strength in [0, 1]."""
    scfg = cfg.structure
    if not scfg.outlines:
        return np.zeros(img_srgb.shape[:2], dtype=np.float32)

    gray = luminance(img_srgb)
    lines = xdog(
        gray,
        sigma=scfg.xdog_sigma,
        k=scfg.xdog_k,
        tau=scfg.xdog_tau,
        phi=scfg.xdog_phi,
        epsilon=scfg.xdog_epsilon,
    )
    if scfg.depth_gated and depth is not None:
        lines = depth_gate(lines, depth, strength=0.85)
    return lines


def composite_outlines(
    img_srgb: np.ndarray,
    lines: np.ndarray,
    strength: float,
    darken: float,
) -> np.ndarray:
    """Darken the image along extracted lines.

    Outlines are produced by darkening toward the local color rather than
    stamping pure black, so a line over a bright facade and a line over a dark
    car both read correctly instead of the second disappearing.
    """
    if strength <= 0.0 or lines.max() <= 0.0:
        return img_srgb
    weight = np.clip(lines * float(strength), 0.0, 1.0)[..., None]
    target = img_srgb * (1.0 - float(darken))
    return np.clip(img_srgb * (1.0 - weight) + target * weight, 0.0, 1.0).astype(
        np.float32
    )
