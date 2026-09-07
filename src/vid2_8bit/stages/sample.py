"""Stage 3 -- source resolution to the logical pixel grid.

Plain area-averaging is the default everywhere and it is why thin dark features
vanish: a 1-pixel-wide window mullion averaged over a 6x6 cell contributes 1/36
of the result and disappears entirely. Meanwhile the *silhouette* of a building
is exactly such a thin dark feature, and it is the single most important line in
the image.

``outline_expand`` fixes this with contrast-aware outline expansion (the
technique PixelOE is built on): before downsampling, thicken whichever extreme
-- dark or bright -- locally dominates, so features that matter grow to at least
one cell and survive the average as crisp pixels.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..color.spaces import srgb_to_oklab


def grid_size(cfg, src_w: int, src_h: int) -> tuple[int, int]:
    """Logical (w, h) for a source resolution."""
    return cfg.logical_size(src_w, src_h)


def area_downsample(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Box-average downsample to (w, h). Runs in linear light."""
    from ..color.spaces import linear_to_srgb, srgb_to_linear

    lin = srgb_to_linear(img)
    small = cv2.resize(lin, size, interpolation=cv2.INTER_AREA)
    return linear_to_srgb(small)


def median_downsample(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Take the MEDIAN of each cell rather than the mean.

    The mean of a cell straddling a dark facade and a lit window returns a
    midtone that exists nowhere in the scene; do that everywhere and the image
    turns to sludge, which a blind critic named as one of the tells against the
    film reference. The median returns whichever value actually dominates the
    cell, so flat regions stay exactly flat and edges land on one side or the
    other instead of inventing a halo.
    """
    w_out, h_out = size
    h, w = img.shape[:2]
    ph, pw = (-h) % h_out, (-w) % w_out
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    bh, bw = img.shape[0] // h_out, img.shape[1] // w_out
    if bh < 1 or bw < 1:
        return area_downsample(img, size)
    blocks = img[: bh * h_out, : bw * w_out].reshape(h_out, bh, w_out, bw, 3)
    return np.median(blocks, axis=(1, 3)).astype(np.float32)


def outline_expansion(
    img_srgb: np.ndarray,
    cell: float,
    weight: float = 0.7,
) -> np.ndarray:
    """Contrast-aware outline expansion at source resolution.

    For each neighbourhood, decide whether the locally *dark* or locally
    *bright* extreme carries more information relative to the local median, then
    push the image toward that extreme. Dark outlines get thicker, specular
    highlights stay put, and flat regions -- where both extremes are equidistant
    -- are left alone.
    """
    k = max(3, int(round(cell)) | 1)  # odd kernel matching one output cell
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    gray = srgb_to_oklab(img_srgb)[..., 0]
    local_max = cv2.dilate(gray, kernel)
    local_min = cv2.erode(gray, kernel)
    median = cv2.medianBlur(gray, k if k <= 5 else 5)

    dist_bright = np.maximum(local_max - median, 0.0)
    dist_dark = np.maximum(median - local_min, 0.0)
    total = dist_bright + dist_dark + 1e-6
    # High where dark detail dominates -> expand the dark extreme there.
    w = dist_dark / total
    # Smooth the decision so the choice does not flip pixel to pixel, which
    # would itself create noise.
    w = cv2.GaussianBlur(w, (0, 0), sigmaX=max(1.0, cell * 0.4))[..., None]

    eroded = cv2.erode(img_srgb, kernel)
    dilated = cv2.dilate(img_srgb, kernel)
    expanded = eroded * w + dilated * (1.0 - w)

    blend = float(np.clip(weight, 0.0, 1.0))
    return np.clip(img_srgb * (1.0 - blend) + expanded * blend, 0.0, 1.0).astype(
        np.float32
    )


def slic_downsample(
    img_srgb: np.ndarray,
    size: tuple[int, int],
    iters: int = 6,
    compactness: float = 12.0,
) -> np.ndarray:
    """Grid-constrained SLIC sampling (Gerstner-style).

    Each output pixel owns one superpixel whose centroid is constrained to stay
    within its own cell, so the output stays a regular grid while each cell's
    color is drawn from perceptually coherent source pixels rather than a blind
    square average. Slower, and noticeably better on curved and diagonal edges,
    which a box average turns into staircases of blended midtones.
    """
    w_out, h_out = size
    h, w = img_srgb.shape[:2]
    lab = srgb_to_oklab(img_srgb)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cell_y, cell_x = h / h_out, w / w_out

    # Cluster centers start at cell centers.
    cy = (np.arange(h_out, dtype=np.float32) + 0.5) * cell_y
    cx = (np.arange(w_out, dtype=np.float32) + 0.5) * cell_x
    centers_pos = np.stack(np.meshgrid(cy, cx, indexing="ij"), axis=-1)
    centers_col = cv2.resize(lab, size, interpolation=cv2.INTER_AREA)

    # Spatial scale that puts position and color on comparable footing.
    inv_s = compactness / max(cell_x, cell_y)

    labels = np.zeros((h, w), dtype=np.int32)
    for _ in range(iters):
        # Each pixel considers only the 3x3 neighbourhood of cells around it,
        # which is what keeps this linear instead of quadratic.
        best = np.full((h, w), np.inf, dtype=np.float32)
        gy = np.clip((ys / cell_y).astype(np.int32), 0, h_out - 1)
        gx = np.clip((xs / cell_x).astype(np.int32), 0, w_out - 1)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny = np.clip(gy + dy, 0, h_out - 1)
                nx = np.clip(gx + dx, 0, w_out - 1)
                pos = centers_pos[ny, nx]
                col = centers_col[ny, nx]
                d_col = ((lab - col) ** 2).sum(-1)
                d_pos = ((ys - pos[..., 0]) ** 2 + (xs - pos[..., 1]) ** 2) * inv_s**2
                d = d_col + d_pos
                take = d < best
                best = np.where(take, d, best)
                labels = np.where(take, ny * w_out + nx, labels)

        # Update centers from their members.
        flat = labels.reshape(-1)
        n = h_out * w_out
        counts = np.bincount(flat, minlength=n).astype(np.float32)
        counts[counts == 0] = 1.0
        for c in range(3):
            sums = np.bincount(flat, weights=lab[..., c].reshape(-1), minlength=n)
            centers_col.reshape(-1, 3)[:, c] = (sums / counts).astype(np.float32)
        sy = np.bincount(flat, weights=ys.reshape(-1), minlength=n) / counts
        sx = np.bincount(flat, weights=xs.reshape(-1), minlength=n) / counts
        new_pos = np.stack([sy, sx], -1).reshape(h_out, w_out, 2).astype(np.float32)
        # Constrain each centroid to its own cell so the grid stays regular.
        lo = np.stack(np.meshgrid(cy - cell_y * 0.5, cx - cell_x * 0.5, indexing="ij"), -1)
        hi = np.stack(np.meshgrid(cy + cell_y * 0.5, cx + cell_x * 0.5, indexing="ij"), -1)
        centers_pos = np.clip(new_pos, lo, hi)

    from ..color.spaces import oklab_to_srgb

    return oklab_to_srgb(centers_col)


def sample_frame(
    img_srgb: np.ndarray,
    cfg,
    size: tuple[int, int],
    tier: str = "quality",
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Downsample one abstracted frame to the logical grid.

    `offset` shifts the sampling grid by whole source pixels. Stage 5 uses it to
    anchor the grid to camera motion, which is what stops pixels crawling during
    a pan.
    """
    h, w = img_srgb.shape[:2]
    cell = w / float(size[0])

    src = img_srgb
    if offset != (0, 0):
        src = np.roll(src, (-int(offset[0]), -int(offset[1])), axis=(0, 1))

    method = cfg.sample.method
    if method == "superpixel" and tier == "quality":
        return slic_downsample(src, size)

    if method in ("outline_expand", "superpixel"):
        src = outline_expansion(
            src,
            cell * cfg.sample.expand_radius,
            weight=cfg.sample.contrast_weight,
        )
        return median_downsample(src, size)
    if method == "median":
        return median_downsample(src, size)
    return area_downsample(src, size)
