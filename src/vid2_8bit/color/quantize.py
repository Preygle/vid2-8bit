"""Palette quantization.

Two things here are load-bearing for video quality:

* Nearest-color search runs in **Oklab**, so "closest color" means closest to
  the eye rather than closest in RGB coordinates.
* :meth:`Quantizer.quantize` accepts the previous frame's index map and applies
  **hysteresis**: a pixel keeps its old palette index unless a different entry
  wins by a margin. Without this, pixels sitting exactly between two palette
  entries flip back and forth every frame, which reads as sparkling noise even
  when the underlying footage is static.
"""

from __future__ import annotations

import numpy as np

from .spaces import oklab_to_srgb, srgb_to_oklab


class Quantizer:
    """Maps images to palette indices and back.

    A 3D lookup table over sRGB is built lazily for the fast tier. It costs one
    cube build per shot and turns per-pixel nearest-neighbour search into an
    array index, which is what makes interactive preview viable.
    """

    def __init__(self, palette_srgb: np.ndarray, lut_bits: int = 0):
        self.palette = np.ascontiguousarray(palette_srgb, dtype=np.float32)
        if self.palette.ndim != 2 or self.palette.shape[1] != 3:
            raise ValueError("palette must be (N, 3)")
        self.palette_lab = srgb_to_oklab(self.palette)
        self.lut_bits = int(lut_bits)
        self._lut: np.ndarray | None = None

    # -- properties --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.palette)

    # -- LUT ---------------------------------------------------------------

    def _build_lut(self) -> np.ndarray:
        n = 1 << self.lut_bits
        axis = np.arange(n, dtype=np.float32) / max(1, n - 1)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        flat = grid.reshape(-1, 3)
        idx = self._nearest(srgb_to_oklab(flat))
        return idx.reshape(n, n, n).astype(np.int32)

    # -- core --------------------------------------------------------------

    def _nearest(self, lab: np.ndarray) -> np.ndarray:
        """Nearest palette index for (N, 3) Oklab samples."""
        n = lab.shape[0]
        out = np.empty(n, dtype=np.int32)
        step = max(1, 4_000_000 // max(1, len(self.palette)))
        for s in range(0, n, step):
            chunk = lab[s : s + step]
            d = ((chunk[:, None, :] - self.palette_lab[None, :, :]) ** 2).sum(-1)
            out[s : s + step] = np.argmin(d, axis=1)
        return out

    def _nearest_with_distance(self, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = lab.shape[0]
        idx = np.empty(n, dtype=np.int32)
        dist = np.empty(n, dtype=np.float32)
        step = max(1, 4_000_000 // max(1, len(self.palette)))
        for s in range(0, n, step):
            chunk = lab[s : s + step]
            d = ((chunk[:, None, :] - self.palette_lab[None, :, :]) ** 2).sum(-1)
            idx[s : s + step] = np.argmin(d, axis=1)
            dist[s : s + step] = np.min(d, axis=1)
        return idx, dist

    def quantize(
        self,
        img_srgb: np.ndarray,
        prev_idx: np.ndarray | None = None,
        hysteresis: float = 0.0,
    ) -> np.ndarray:
        """Quantize an (H, W, 3) sRGB image to an (H, W) int32 index map."""
        h, w = img_srgb.shape[:2]
        flat = np.ascontiguousarray(img_srgb, dtype=np.float32).reshape(-1, 3)

        if self.lut_bits > 0:
            if self._lut is None:
                self._lut = self._build_lut()
            n = self._lut.shape[0]
            q = np.clip(np.rint(flat * (n - 1)), 0, n - 1).astype(np.int32)
            idx = self._lut[q[:, 0], q[:, 1], q[:, 2]]
        else:
            idx = self._nearest(srgb_to_oklab(flat))

        idx = idx.reshape(h, w)

        if prev_idx is not None and hysteresis > 0.0:
            idx = self._apply_hysteresis(flat.reshape(h, w, 3), idx, prev_idx, hysteresis)
        return idx.astype(np.int32)

    def _apply_hysteresis(
        self,
        img_srgb: np.ndarray,
        new_idx: np.ndarray,
        prev_idx: np.ndarray,
        margin: float,
    ) -> np.ndarray:
        """Keep the previous index unless the new one wins by `margin` in Oklab."""
        if prev_idx.shape != new_idx.shape:
            return new_idx
        lab = srgb_to_oklab(img_srgb)
        prev_clamped = np.clip(prev_idx, 0, len(self.palette) - 1)
        d_prev = ((lab - self.palette_lab[prev_clamped]) ** 2).sum(-1)
        d_new = ((lab - self.palette_lab[new_idx]) ** 2).sum(-1)
        # Compare in distance units, not squared units, so the margin is a
        # readable Oklab distance rather than an arbitrary scale.
        keep = np.sqrt(d_prev) - np.sqrt(d_new) < margin
        return np.where(keep, prev_clamped, new_idx).astype(np.int32)

    # -- output ------------------------------------------------------------

    def to_rgb(self, idx: np.ndarray) -> np.ndarray:
        """Index map -> (H, W, 3) float sRGB."""
        return self.palette[np.clip(idx, 0, len(self.palette) - 1)]

    def error_map(self, img_srgb: np.ndarray) -> np.ndarray:
        """Oklab distance from each pixel to its nearest palette entry.

        Drives selective dithering: dither only where the palette genuinely
        cannot represent the color, leaving flat regions untouched.
        """
        h, w = img_srgb.shape[:2]
        flat = np.ascontiguousarray(img_srgb, dtype=np.float32).reshape(-1, 3)
        _, dist = self._nearest_with_distance(srgb_to_oklab(flat))
        return np.sqrt(dist).reshape(h, w)


def snap_image_bit_depth(img_srgb: np.ndarray, bits) -> np.ndarray:
    """Quantize an image's channels to a reduced bit depth."""
    out = np.empty_like(img_srgb)
    for c, b in enumerate(bits):
        levels = (1 << int(b)) - 1
        out[..., c] = np.rint(img_srgb[..., c] * levels) / max(1, levels)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def posterize_luma(img_srgb: np.ndarray, bands: int) -> np.ndarray:
    """Quantize lightness into `bands` plateaus, preserving hue and chroma.

    This is what turns a smooth photographic gradient into the discrete shading
    steps that read as hand-drawn pixel art. Done in Oklab so only perceived
    lightness is stepped -- posterizing RGB directly shifts hues.
    """
    if bands < 2:
        return img_srgb
    lab = srgb_to_oklab(img_srgb)
    # Normalize on percentiles, not min/max. A single specular highlight or a
    # patch of sun sets the maximum far above the useful range, and dividing by
    # it compresses the whole scene into one or two bands.
    lo = float(np.percentile(lab[..., 0], 2.0))
    hi = float(np.percentile(lab[..., 0], 98.0))
    if hi - lo < 1e-6:
        return img_srgb
    norm = np.clip((lab[..., 0] - lo) / (hi - lo), 0.0, 1.0)
    stepped = np.rint(norm * (bands - 1)) / (bands - 1)
    lab[..., 0] = stepped * (hi - lo) + lo
    return oklab_to_srgb(lab)
