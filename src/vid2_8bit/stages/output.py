"""Stage 6 -- upscale and optional CRT post.

The upscale must be nearest-neighbour at an *integer* factor. Any interpolation
reintroduces intermediate colors that are not in the palette, and a fractional
factor makes some logical pixels one output pixel wider than their neighbours --
an irregularity the eye picks up immediately as a shimmer during motion.
"""

from __future__ import annotations

import cv2
import numpy as np


def integer_scale(logical_w: int, logical_h: int, target_w: int, target_h: int) -> int:
    """Largest integer factor fitting the logical image inside the target."""
    return max(1, min(target_w // max(1, logical_w), target_h // max(1, logical_h)))


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour upscale by an integer factor."""
    factor = max(1, int(factor))
    if factor == 1:
        return img
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def pad_to(img: np.ndarray, width: int, height: int, color: float = 0.0) -> np.ndarray:
    """Center the image in a canvas of the given size (letterbox)."""
    h, w = img.shape[:2]
    if (h, w) == (height, width):
        return img
    canvas = np.full((height, width, img.shape[2]), color, dtype=img.dtype)
    y0 = max(0, (height - h) // 2)
    x0 = max(0, (width - w) // 2)
    crop = img[: min(h, height), : min(w, width)]
    canvas[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
    return canvas


def apply_crt(img: np.ndarray, scale: int, scanline_strength: float = 0.25) -> np.ndarray:
    """Scanlines, aperture grille and a little bloom.

    Scaled to the upscale factor so the effect stays one scanline per *logical*
    pixel row regardless of output resolution.
    """
    if scale < 2 or scanline_strength <= 0.0:
        return img

    h, w = img.shape[:2]
    out = img.astype(np.float32)

    rows = (np.arange(h) % scale) / float(scale)
    scan = 1.0 - scanline_strength * (np.sin(rows * np.pi) ** 0.5)
    out *= scan[:, None, None]

    # Aperture grille: cycle a slight per-channel emphasis across columns.
    if scale >= 3:
        grille = np.ones((w, 3), dtype=np.float32)
        phase = (np.arange(w) // max(1, scale // 3)) % 3
        for c in range(3):
            grille[phase == c, c] *= 1.0 + scanline_strength * 0.4
            grille[phase != c, c] *= 1.0 - scanline_strength * 0.15
        out *= grille[None, :, :]

    # Bloom: bright areas bleed, which is what stops CRT emulation looking
    # merely dimmed.
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=scale * 0.6)
    out = np.clip(out + np.clip(blur - 0.55, 0.0, None) * 0.35, 0.0, 1.0)
    return out.astype(np.float32)


def finalize(
    logical_srgb: np.ndarray,
    cfg,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Upscale a logical frame to delivery resolution."""
    lh, lw = logical_srgb.shape[:2]
    factor = cfg.output.scale or integer_scale(lw, lh, target_w, target_h)
    out = upscale(logical_srgb, factor)
    if cfg.output.crt:
        out = apply_crt(out, factor, cfg.output.scanline_strength)
    return pad_to(out, target_w, target_h)
