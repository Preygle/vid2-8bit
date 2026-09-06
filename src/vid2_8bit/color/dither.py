"""Dithering.

Ordered (Bayer / blue-noise) dithering perturbs a pixel by a fixed threshold
pattern before quantization. For video this is strongly preferable to error
diffusion: the pattern is a pure function of position, so a static region
produces an identical result every frame. Floyd-Steinberg is offered for stills
but propagates error serially, meaning a one-pixel change at the top of the
frame alters every pixel below it -- catastrophic for temporal stability.

Two refinements matter for the look this project targets:

* **Selective dithering** -- only dither where the palette genuinely cannot
  represent the color. Skies band and need help; a flat concrete facade does
  not, and dithering it is exactly the "noise" this project exists to avoid.
* **Grid anchoring** -- the threshold matrix is sampled at an offset that tracks
  camera motion, so the dither pattern sticks to the scene instead of crawling
  across it as the camera pans.
"""

from __future__ import annotations

import numpy as np


def bayer_matrix(n: int) -> np.ndarray:
    """Normalized Bayer threshold matrix of size n x n (n a power of two).

    Values are in [-0.5, 0.5) so the matrix is zero-mean and dithering does not
    shift overall brightness.
    """
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError("Bayer matrix size must be a power of two >= 2")
    m = np.array([[0, 2], [3, 1]], dtype=np.float32)
    size = 2
    while size < n:
        m = np.block(
            [
                [4 * m + 0, 4 * m + 2],
                [4 * m + 3, 4 * m + 1],
            ]
        ).astype(np.float32)
        size *= 2
    # +0.5 centers the matrix so it is exactly zero-mean; without it
    # dithering darkens the image by half a quantization step.
    return ((m + 0.5) / (n * n) - 0.5).astype(np.float32)


def blue_noise_matrix(n: int = 64, seed: int = 0) -> np.ndarray:
    """Approximate blue-noise threshold mask via iterative void-and-cluster.

    Cheaper and less exact than true void-and-cluster, but produces a mask with
    the essential property: energy concentrated at high spatial frequencies, so
    the pattern is far less visually structured than Bayer at the same density.
    Deterministic for a given seed.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    values = rng.random((n, n)).astype(np.float32)
    # Repeatedly swap the pixel in the densest cluster with the one in the
    # largest void, which pushes energy toward high frequencies.
    for _ in range(24):
        blurred = gaussian_filter(values, sigma=1.5, mode="wrap")
        order = np.argsort(blurred, axis=None)
        low, high = order[: n * n // 8], order[-(n * n // 8) :]
        flat = values.reshape(-1)
        flat[low], flat[high] = flat[high].copy(), flat[low].copy()
        values = flat.reshape(n, n)
    ranks = np.argsort(np.argsort(values, axis=None)).reshape(n, n)
    return ((ranks.astype(np.float32) + 0.5) / (n * n) - 0.5).astype(np.float32)


_MATRIX_CACHE: dict[str, np.ndarray] = {}


def threshold_matrix(mode: str) -> np.ndarray | None:
    """Resolve a dither mode name to its threshold matrix."""
    if mode in ("none", "floyd"):
        return None
    if mode in _MATRIX_CACHE:
        return _MATRIX_CACHE[mode]
    if mode.startswith("bayer"):
        matrix = bayer_matrix(int(mode[5:]))
    elif mode == "bluenoise":
        matrix = blue_noise_matrix(64)
    else:
        raise ValueError(f"unknown dither mode {mode!r}")
    _MATRIX_CACHE[mode] = matrix
    return matrix


def _tile_to(matrix: np.ndarray, shape: tuple[int, int], offset=(0, 0)) -> np.ndarray:
    """Tile a threshold matrix over `shape`, sampled at an integer offset."""
    h, w = shape
    mh, mw = matrix.shape
    ys = (np.arange(h) + int(offset[0])) % mh
    xs = (np.arange(w) + int(offset[1])) % mw
    return matrix[np.ix_(ys, xs)]


def palette_spacing(quantizer) -> float:
    """Median nearest-neighbour distance between palette entries, in Oklab.

    Dither amplitude is scaled by this: the perturbation must be roughly the
    size of one palette step to move a pixel to an adjacent color, and no more,
    or dithering degenerates into noise.
    """
    lab = quantizer.palette_lab
    if len(lab) < 2:
        return 0.0
    d = ((lab[:, None, :] - lab[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(np.sqrt(d.min(axis=1))))


def apply_ordered(
    img_srgb: np.ndarray,
    quantizer,
    mode: str,
    amount: float = 0.6,
    selective_threshold: float = 0.0,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Perturb an image by a threshold matrix ahead of quantization.

    Returns a modified sRGB image; the caller still quantizes it.
    """
    matrix = threshold_matrix(mode)
    if matrix is None or amount <= 0.0:
        return img_srgb

    h, w = img_srgb.shape[:2]
    thresh = _tile_to(matrix, (h, w), offset)[..., None]

    # Amplitude in sRGB units, derived from how far apart palette entries are.
    step = palette_spacing(quantizer)
    amplitude = float(amount) * max(step, 1e-4) * 1.6

    if selective_threshold > 0.0:
        err = quantizer.error_map(img_srgb)
        # Ramp in smoothly so there is no visible boundary between the dithered
        # and undithered parts of a gradient.
        gate = np.clip(
            (err - selective_threshold) / max(selective_threshold, 1e-6), 0.0, 1.0
        )
        amplitude = amplitude * gate[..., None]

    return np.clip(img_srgb + thresh * amplitude, 0.0, 1.0).astype(np.float32)


def apply_floyd_steinberg(img_srgb: np.ndarray, quantizer) -> np.ndarray:
    """Serial error diffusion. Returns an index map.

    Temporally unstable by construction -- use for stills only.
    """
    h, w = img_srgb.shape[:2]
    work = img_srgb.astype(np.float32).copy()
    idx = np.zeros((h, w), dtype=np.int32)
    pal = quantizer.palette

    for y in range(h):
        for x in range(w):
            old = work[y, x].copy()
            i = int(quantizer.quantize(old.reshape(1, 1, 3))[0, 0])
            idx[y, x] = i
            err = old - pal[i]
            if x + 1 < w:
                work[y, x + 1] += err * (7 / 16)
            if y + 1 < h:
                if x > 0:
                    work[y + 1, x - 1] += err * (3 / 16)
                work[y + 1, x] += err * (5 / 16)
                if x + 1 < w:
                    work[y + 1, x + 1] += err * (1 / 16)
    return idx
