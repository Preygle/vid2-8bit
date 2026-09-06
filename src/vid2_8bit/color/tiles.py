"""Hardware tile color constraints.

Real 8-bit hardware did not let an artist use the whole palette freely. The NES
PPU holds four background sub-palettes of three colors each plus one shared
backdrop, and every 16x16 attribute area picks exactly one of them. That
restriction is *why* NES art looks the way it does -- large regions committed to
a single small color family, with hue shifts happening at tile boundaries.

Reproducing it is therefore a strong authenticity lever, not a limitation.

The assignment problem (choose S sub-palettes of C colors, and assign every tile
to one, minimizing total error) is NP-hard, so this solves it with a Lloyd-style
alternation that converges in a handful of passes:

1. Fit a small palette per tile independently.
2. Cluster those tile palettes into S groups.
3. Refit each group's palette from all pixels in its tiles.
4. Reassign every tile to its best-fitting group.
5. Repeat 3-4.
"""

from __future__ import annotations

import numpy as np

from .palette import kmeans
from .spaces import oklab_to_srgb, srgb_to_oklab


def _tile_view(img: np.ndarray, tile: int) -> tuple[np.ndarray, int, int, int, int]:
    """Pad to a whole number of tiles and reshape to (ty, tx, tile, tile, C)."""
    h, w = img.shape[:2]
    ph = (-h) % tile
    pw = (-w) % tile
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    th, tw = img.shape[0] // tile, img.shape[1] // tile
    view = img.reshape(th, tile, tw, tile, img.shape[2]).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(view), th, tw, h, w


def solve_tile_palettes(
    img_srgb: np.ndarray,
    master_palette: np.ndarray,
    tile_size: int = 8,
    colors_per_tile: int = 4,
    subpalettes: int = 4,
    shared_backdrop: bool = True,
    iters: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit sub-palettes and assign tiles.

    Returns ``(subpalette_colors, tile_assignment)`` where ``subpalette_colors``
    is ``(S, C)`` of indices into ``master_palette`` and ``tile_assignment`` is
    ``(tiles_y, tiles_x)`` of sub-palette indices.
    """
    view, th, tw, _, _ = _tile_view(img_srgb, tile_size)
    n_tiles = th * tw
    tiles = view.reshape(n_tiles, tile_size * tile_size, 3)
    tiles_lab = srgb_to_oklab(tiles)

    master_lab = srgb_to_oklab(master_palette)
    subpalettes = int(min(subpalettes, n_tiles))

    # Step 1-2: characterize each tile by its mean and spread, then cluster
    # tiles into groups. Using summary statistics rather than a per-tile k-means
    # keeps this fast and is sufficient to separate materials.
    feat = np.concatenate(
        [tiles_lab.mean(axis=1), tiles_lab.std(axis=1)], axis=1
    ).astype(np.float32)
    centers = kmeans(feat, subpalettes, seed=seed)
    d = ((feat[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    assign = np.argmin(d, axis=1).astype(np.int32)

    backdrop_idx: int | None = None
    if shared_backdrop:
        # The darkest heavily-used color reads as the backdrop, matching how the
        # hardware's universal background color was normally spent.
        flat_lab = tiles_lab.reshape(-1, 3)
        dist = ((flat_lab[:, None, :] - master_lab[None, :, :]) ** 2).sum(-1)
        counts = np.bincount(np.argmin(dist, axis=1), minlength=len(master_palette))
        common = np.where(counts > counts.max() * 0.02)[0]
        if len(common):
            backdrop_idx = int(common[np.argmin(master_lab[common, 0])])

    free_slots = colors_per_tile - (1 if backdrop_idx is not None else 0)
    sub_indices = np.zeros((subpalettes, colors_per_tile), dtype=np.int32)

    for _ in range(max(1, iters)):
        # Step 3: refit each group's palette from every pixel it owns.
        for s in range(subpalettes):
            members = np.where(assign == s)[0]
            if len(members) == 0:
                sub_indices[s] = sub_indices[max(0, s - 1)]
                continue
            px = tiles_lab[members].reshape(-1, 3)
            if len(px) > 20000:
                px = px[:: max(1, len(px) // 20000)]
            k = max(1, free_slots)
            fitted = kmeans(px, k, seed=seed + s)
            # Snap each fitted center onto the nearest master-palette entry,
            # since hardware colors are not freely choosable.
            dd = ((fitted[:, None, :] - master_lab[None, :, :]) ** 2).sum(-1)
            chosen = np.argmin(dd, axis=1)
            row = list(dict.fromkeys(chosen.tolist()))
            if backdrop_idx is not None:
                row = [backdrop_idx] + [c for c in row if c != backdrop_idx]
            while len(row) < colors_per_tile:
                row.append(row[-1] if row else 0)
            sub_indices[s] = np.asarray(row[:colors_per_tile], dtype=np.int32)

        # Step 4: reassign tiles to whichever sub-palette represents them best.
        errs = np.empty((n_tiles, subpalettes), dtype=np.float32)
        for s in range(subpalettes):
            cand = master_lab[sub_indices[s]]
            dd = ((tiles_lab[:, :, None, :] - cand[None, None, :, :]) ** 2).sum(-1)
            errs[:, s] = dd.min(axis=2).mean(axis=1)
        new_assign = np.argmin(errs, axis=1).astype(np.int32)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign

    return sub_indices, assign.reshape(th, tw)


def apply_tile_constraints(
    img_srgb: np.ndarray,
    master_palette: np.ndarray,
    tile_size: int = 8,
    colors_per_tile: int = 4,
    subpalettes: int = 4,
    shared_backdrop: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Quantize an image under hardware tile constraints. Returns index map."""
    sub_indices, assign = solve_tile_palettes(
        img_srgb,
        master_palette,
        tile_size=tile_size,
        colors_per_tile=colors_per_tile,
        subpalettes=subpalettes,
        shared_backdrop=shared_backdrop,
        seed=seed,
    )

    view, th, tw, h, w = _tile_view(img_srgb, tile_size)
    master_lab = srgb_to_oklab(master_palette)
    out = np.zeros((th, tw, tile_size, tile_size), dtype=np.int32)

    for s in range(len(sub_indices)):
        ty, tx = np.where(assign == s)
        if len(ty) == 0:
            continue
        px = srgb_to_oklab(view[ty, tx])            # (n, ts, ts, 3)
        cand = master_lab[sub_indices[s]]            # (C, 3)
        d = ((px[..., None, :] - cand[None, None, None, :, :]) ** 2).sum(-1)
        best = np.argmin(d, axis=-1)
        out[ty, tx] = sub_indices[s][best]

    full = out.transpose(0, 2, 1, 3).reshape(th * tile_size, tw * tile_size)
    return full[:h, :w].astype(np.int32)


def tile_palette_preview(master_palette: np.ndarray, sub_indices: np.ndarray) -> np.ndarray:
    """Render the solved sub-palettes as a small swatch image, for debugging."""
    s, c = sub_indices.shape
    swatch = 16
    out = np.zeros((s * swatch, c * swatch, 3), dtype=np.float32)
    for i in range(s):
        for j in range(c):
            out[i * swatch : (i + 1) * swatch, j * swatch : (j + 1) * swatch] = (
                master_palette[sub_indices[i, j]]
            )
    return out


__all__ = [
    "solve_tile_palettes",
    "apply_tile_constraints",
    "tile_palette_preview",
]
