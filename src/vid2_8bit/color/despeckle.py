"""Isolated-cell removal.

Operates on the palette index map, after quantization.

The distinction that matters: the film reference is *full* of isolated single
cells -- they are lit windows, signage, street lamps -- so removing every
isolated cell would destroy the look. What separates those from noise is
contrast. A deliberate window is a bright cell on a dark facade, several palette
steps away from everything around it. Speckle is a cell that landed one step off
its neighbours because a blurred gradient wobbled across a quantization
threshold, and it reads as dirt.

So this removes isolated cells only when they are *close* to their surroundings,
and leaves high-contrast islands alone.
"""

from __future__ import annotations

import numpy as np


def despeckle_indices(
    idx: np.ndarray,
    palette_lab: np.ndarray,
    threshold: float = 0.12,
    min_neighbours: int = 8,
) -> np.ndarray:
    """Replace low-contrast isolated cells with their dominant neighbour.

    `threshold` is an Oklab distance: an island further than this from the
    neighbourhood's dominant colour is treated as deliberate and kept.
    `min_neighbours` is how many of the 8 neighbours must agree before a cell is
    considered isolated. It defaults to 8 -- a fully surrounded cell. Allowing 7
    also eats the *end* cells of short runs, since an end cell has exactly seven
    background neighbours, which quietly shortens every architectural line the
    rest of the pipeline works to preserve.
    """
    if idx.shape[0] < 3 or idx.shape[1] < 3 or threshold <= 0.0:
        return idx

    padded = np.pad(idx, 1, mode="edge")
    neighbours = np.stack(
        [
            padded[dy : dy + idx.shape[0], dx : dx + idx.shape[1]]
            for dy in (0, 1, 2)
            for dx in (0, 1, 2)
            if not (dy == 1 and dx == 1)
        ],
        axis=-1,
    )

    # Dominant neighbour index per cell, and how many neighbours agree with it.
    n_colors = int(max(idx.max(), palette_lab.shape[0] - 1)) + 1
    onehot = (neighbours[..., None] == np.arange(n_colors)).sum(axis=2)
    dominant = np.argmax(onehot, axis=-1)
    agreement = np.max(onehot, axis=-1)

    differs = idx != dominant
    isolated = differs & (agreement >= int(min_neighbours))

    # Keep islands that genuinely contrast -- those are windows, not noise.
    d = np.sqrt(
        ((palette_lab[np.clip(idx, 0, len(palette_lab) - 1)]
          - palette_lab[np.clip(dominant, 0, len(palette_lab) - 1)]) ** 2).sum(-1)
    )
    replace = isolated & (d < float(threshold))
    return np.where(replace, dominant, idx).astype(np.int32)
