"""Palette derivation.

Three sources of palettes:

* ``hardware`` -- fixed historical palettes (NES, Game Boy, C64, ...). These are
  what actually produce a period-authentic look, because the constraint is the
  style.
* ``auto`` -- k-means in Oklab fitted once per shot over frames sampled across
  the whole shot. Fitting per-frame would make the palette flicker, which is the
  single worst artifact in pixel-art video, so the API deliberately takes a pool
  of samples rather than one image.
* ``custom`` -- ``.hex`` or ``.gpl`` files (Lospec-compatible).

k-means is implemented here rather than pulled from scikit-learn for two
reasons: the core pipeline stays dependency-light, and we need *exact*
determinism -- the same shot must always yield byte-identical colors or cached
renders stop matching.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .spaces import linear_to_srgb, srgb_to_linear, srgb_to_oklab

# -- hardware palettes -----------------------------------------------------

# NES (2C02) master palette. The hardware exposes 64 entries, several of which
# are duplicate blacks; the 54 unique colors are what artists actually had.
_NES_HEX = """
545454 001E74 081090 300088 440064 5C0030 540400 3C1800
202A00 083A00 004000 003C00 00323C 000000
989698 084CC4 3032EC 5C1EE4 8814B0 A01464 982220 783C00
545A00 287200 087C00 007628 006678
ECEEEC 4C9AEC 787CEC B062EC E454EC EC58B4 EC6A64 D48820
A0AA00 74C400 4CD020 38CC6C 38B4CC 3C3C3C
A8CCEC BCBCEC D4B2EC ECAEEC ECAED4 ECB4B0 E4C490
CCD278 B4DE78 A8E290 98E2B4 A0D6E4 A0A2A0
"""

_GAMEBOY_HEX = "0F380F 306230 8BAC0F 9BBC0F"

_C64_HEX = """
000000 FFFFFF 880000 AAFFEE CC44CC 00CC55 0000AA EEEE77
DD8855 664400 FF7777 333333 777777 AAFF66 0088FF BBBBBB
"""

_PICO8_HEX = """
000000 1D2B53 7E2553 008751 AB5236 5F574F C2C3C7 FFF1E8
FF004D FFA300 FFEC27 00E436 29ADFF 83769C FF77A8 FFCCAA
"""

_CGA_HEX = """
000000 0000AA 00AA00 00AAAA AA0000 AA00AA AA5500 AAAAAA
555555 5555FF 55FF55 55FFFF FF5555 FF55FF FFFF55 FFFFFF
"""


def _from_hex(text: str) -> np.ndarray:
    """Parse whitespace-separated RRGGBB tokens into float sRGB in [0, 1]."""
    out = []
    for tok in text.split():
        tok = tok.strip().lstrip("#")
        if len(tok) != 6:
            continue
        out.append([int(tok[i : i + 2], 16) for i in (0, 2, 4)])
    return np.asarray(out, dtype=np.float32) / 255.0


def _uniform_depth_palette(bits: tuple[int, int, int]) -> np.ndarray:
    """Every color representable at the given per-channel bit depth."""
    axes = []
    for b in bits:
        levels = (1 << b) - 1
        axes.append(np.arange(1 << b, dtype=np.float32) / max(1, levels))
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3).astype(np.float32)


def hardware_palette(name: str) -> np.ndarray:
    """Return a named hardware palette as float sRGB, shape (N, 3)."""
    key = name.strip().lower().replace("_", "").replace("-", "")
    table = {
        "nes": lambda: _from_hex(_NES_HEX),
        "famicom": lambda: _from_hex(_NES_HEX),
        "gameboy": lambda: _from_hex(_GAMEBOY_HEX),
        "dmg": lambda: _from_hex(_GAMEBOY_HEX),
        "c64": lambda: _from_hex(_C64_HEX),
        "commodore64": lambda: _from_hex(_C64_HEX),
        "pico8": lambda: _from_hex(_PICO8_HEX),
        "cga": lambda: _from_hex(_CGA_HEX),
        "ega": lambda: _from_hex(_CGA_HEX),
        # Master System: 2 bits per channel in hardware.
        "mastersystem": lambda: _uniform_depth_palette((2, 2, 2)),
        "sms": lambda: _uniform_depth_palette((2, 2, 2)),
        # Amiga OCS: 4 bits per channel.
        "amiga": lambda: _uniform_depth_palette((4, 4, 4)),
    }
    if key not in table:
        raise ValueError(
            f"unknown hardware palette {name!r}; "
            f"available: {', '.join(sorted(set(table)))}"
        )
    return table[key]()


HARDWARE_PALETTES = (
    "nes", "gameboy", "c64", "pico8", "cga", "mastersystem", "amiga",
)


# -- palette files ---------------------------------------------------------


def load_palette_file(path: str | Path) -> np.ndarray:
    """Load a .hex (one RRGGBB per line) or .gpl (GIMP) palette."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".gpl":
        colors = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line[0].isalpha():
                continue
            parts = line.split()
            if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
                colors.append([int(p) for p in parts[:3]])
        if not colors:
            raise ValueError(f"no colors parsed from {path}")
        return np.asarray(colors, dtype=np.float32) / 255.0
    pal = _from_hex(text.replace(",", " "))
    if len(pal) == 0:
        raise ValueError(f"no colors parsed from {path}")
    return pal


# -- bit depth -------------------------------------------------------------


def snap_to_bit_depth(palette: np.ndarray, bits: tuple[int, int, int] | list[int]) -> np.ndarray:
    """Quantize each channel to the given bit depth, in linear-free sRGB space.

    Hardware stored these values as gamma-encoded channel codes, so snapping is
    done on sRGB values, not linear ones.
    """
    out = np.empty_like(palette)
    for c, b in enumerate(bits):
        levels = (1 << int(b)) - 1
        out[..., c] = np.rint(palette[..., c] * levels) / max(1, levels)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def dedupe(palette: np.ndarray) -> np.ndarray:
    """Remove duplicate colors while preserving order."""
    seen: set[tuple[int, int, int]] = set()
    keep = []
    for i, c in enumerate(np.rint(palette * 255).astype(int)):
        key = (int(c[0]), int(c[1]), int(c[2]))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return palette[keep]


# -- k-means ---------------------------------------------------------------


def _kmeans_plusplus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding. `x` is (N, D)."""
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = x[rng.integers(n)]
    closest = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:
            # All points already coincide with a center; pad with random picks.
            centers[i:] = x[rng.integers(0, n, size=k - i)]
            break
        probs = closest / total
        centers[i] = x[rng.choice(n, p=probs)]
        closest = np.minimum(closest, np.sum((x - centers[i]) ** 2, axis=1))
    return centers


def kmeans(
    x: np.ndarray,
    k: int,
    iters: int = 40,
    seed: int = 0,
    tol: float = 1e-6,
) -> np.ndarray:
    """Lloyd's algorithm with k-means++ seeding. Deterministic for a given seed."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.shape[0]
    if n == 0:
        raise ValueError("kmeans called with no samples")
    k = int(min(k, n))
    rng = np.random.default_rng(seed)
    centers = _kmeans_plusplus(x, k, rng)

    for _ in range(iters):
        # Chunked assignment keeps peak memory bounded on large sample pools.
        labels = np.empty(n, dtype=np.int32)
        step = max(1, 4_000_000 // max(1, k))
        for start in range(0, n, step):
            chunk = x[start : start + step]
            d = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            labels[start : start + step] = np.argmin(d, axis=1)

        new = np.zeros_like(centers)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        np.add.at(new, labels, x)
        empty = counts == 0
        new[~empty] /= counts[~empty, None]
        if empty.any():
            # Re-seed dead clusters at the points furthest from their center,
            # which is what keeps small vivid accents from being swallowed.
            d = ((x - centers[labels]) ** 2).sum(-1)
            far = np.argsort(-d)[: int(empty.sum())]
            new[empty] = x[far]

        shift = float(((new - centers) ** 2).sum())
        centers = new
        if shift < tol:
            break

    return centers


def fit_palette(
    samples_srgb: np.ndarray,
    size: int,
    chroma_weight: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Fit a palette to sampled sRGB pixels, shape (N, 3).

    Clustering happens in Oklab so distances are perceptual. `chroma_weight`
    scales the a/b axes during clustering only: values above 1 make the fit
    spend clusters on saturated accents rather than on subtle grey steps, which
    matters because pixel art reads by hue contrast.
    """
    samples_srgb = np.asarray(samples_srgb, dtype=np.float32).reshape(-1, 3)
    lab = srgb_to_oklab(samples_srgb)
    weighted = lab.copy()
    weighted[:, 1:] *= float(chroma_weight)

    centers_w = kmeans(weighted, size, seed=seed)
    centers = centers_w.copy()
    if chroma_weight != 0:
        centers[:, 1:] /= float(chroma_weight)

    from .spaces import oklab_to_srgb  # local import avoids a cycle at import time

    pal = oklab_to_srgb(centers)
    # Sort by lightness so palette indices are stable and human-readable.
    order = np.argsort(srgb_to_oklab(pal)[:, 0])
    return np.ascontiguousarray(pal[order], dtype=np.float32)


# -- top-level -------------------------------------------------------------


def redistribute_lightness(
    palette_srgb: np.ndarray,
    source_L: np.ndarray,
    strength: float = 1.0,
    anchor_ink: bool = False,
) -> np.ndarray:
    """Place palette lightness at the source's own luminance percentiles.

    Two failed approaches led here, both found by blind critique against the
    film reference.

    Fitting by k-means alone gave entries 0.02 apart in lightness -- nominally
    16 colours, visually one muddy tone. So the first fix forced a uniform
    minimum gap. That was worse: it spread the ramp evenly across the whole
    0..1 range while the actual content of a night exterior lives between
    roughly L 0.08 and 0.25. Half the palette then described tones the shot does
    not contain, and everything the shot *does* contain collapsed into a single
    entry covering 72% of the frame, with a fourteen-unit hole beneath it.

    Placing entry *i* at the source's own (i+0.5)/N percentile fixes both at
    once: every entry lands on tones that exist, each gets a roughly equal share
    of pixels, and the steps are automatically closer together where the image
    has detail. It is histogram equalization applied to the palette rather than
    to the image, so contrast comes from *spending* the ramp rather than from
    stretching pixels.
    """
    from .spaces import oklab_to_srgb

    n = len(palette_srgb)
    if n < 2 or strength <= 0.0:
        return palette_srgb

    lab = srgb_to_oklab(palette_srgb)
    order = np.argsort(lab[:, 0])
    lab = lab[order]

    flat = np.asarray(source_L, dtype=np.float32).reshape(-1)
    if flat.size > 400000:
        flat = flat[:: max(1, flat.size // 400000)]
    targets = np.percentile(flat, (np.arange(n) + 0.5) / n * 100.0)
    targets = np.maximum.accumulate(targets)

    if anchor_ink:
        # Keep a true ink and a specular so the frame still has poles, but do
        # not stretch everything between them -- that was the previous mistake.
        targets[0] = min(targets[0], float(np.percentile(flat, 0.5)) * 0.5)
        targets[-1] = max(targets[-1], float(np.percentile(flat, 99.5)))

    a = float(np.clip(strength, 0.0, 1.0))
    lab[:, 0] = lab[:, 0] * (1.0 - a) + targets.astype(np.float32) * a
    return np.ascontiguousarray(oklab_to_srgb(lab), dtype=np.float32)


def warm_highlights(palette_srgb: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Keep chroma in the bright end of the ramp.

    Critique against the film reference: our highlights desaturated to neutral
    grey (S=13) while the reference rotates warm as it brightens and holds
    saturation to the top (p90 S=215). A ramp that goes colourless at the top
    reads as a photograph averaged down, not as authored art.
    """
    from .spaces import oklab_to_srgb

    if amount <= 0.0:
        return palette_srgb
    lab = srgb_to_oklab(palette_srgb)
    L = lab[:, 0]
    lo, hi = float(L.min()), float(L.max())
    t = (L - lo) / max(hi - lo, 1e-6)          # 0 at ink, 1 at specular
    boost = 1.0 + amount * t
    lab[:, 1] *= boost                          # a: toward warm
    lab[:, 2] *= boost
    lab[:, 1] += amount * 0.02 * t              # a small deliberate warm bias
    lab[:, 2] += amount * 0.03 * t
    return np.ascontiguousarray(oklab_to_srgb(lab), dtype=np.float32)


def build_palette(cfg, samples_srgb: np.ndarray | None = None) -> np.ndarray:
    """Resolve a PaletteConfig into concrete colors, shape (N, 3) float sRGB."""
    pcfg = cfg.palette
    if pcfg.mode == "hardware":
        pal = hardware_palette(pcfg.hardware)
        if pcfg.size and pcfg.size < len(pal) and samples_srgb is not None:
            pal = select_subset(pal, samples_srgb, pcfg.size)
    elif pcfg.mode == "custom":
        pal = load_palette_file(pcfg.custom_path)
    elif pcfg.mode == "reference":
        from ..stages.tone import palette_from_reference

        pal = palette_from_reference(pcfg.reference, pcfg.size, pcfg.chroma_weight)
    else:
        if samples_srgb is None or len(samples_srgb) == 0:
            raise ValueError("auto palette requires sample pixels")
        pal = fit_palette(samples_srgb, pcfg.size, pcfg.chroma_weight)

    if (pcfg.redistribute > 0.0 or pcfg.anchor_ink) and samples_srgb is not None:
        pal = redistribute_lightness(
            pal, srgb_to_oklab(np.asarray(samples_srgb, np.float32))[..., 0],
            pcfg.redistribute, pcfg.anchor_ink,
        )
    if pcfg.warm_highlights > 0.0:
        pal = warm_highlights(pal, pcfg.warm_highlights)
    if pcfg.bits_per_channel:
        pal = snap_to_bit_depth(pal, pcfg.bits_per_channel)
    pal = dedupe(pal)
    return np.ascontiguousarray(pal, dtype=np.float32)


def select_subset(palette: np.ndarray, samples_srgb: np.ndarray, size: int) -> np.ndarray:
    """Pick the `size` hardware colors that best cover the actual footage.

    NES-style hardware exposes a master palette but only lets a limited number
    be on screen at once, so choosing *which* ones is part of the look. Greedy
    max-coverage: repeatedly add the color that most reduces total error.
    """
    pal_lab = srgb_to_oklab(palette)
    smp = np.asarray(samples_srgb, dtype=np.float32).reshape(-1, 3)
    if len(smp) > 20000:
        idx = np.linspace(0, len(smp) - 1, 20000).astype(int)
        smp = smp[idx]
    smp_lab = srgb_to_oklab(smp)

    chosen: list[int] = []
    best = np.full(len(smp_lab), np.inf, dtype=np.float32)
    for _ in range(min(size, len(palette))):
        # Error if each candidate were added next.
        d = ((smp_lab[:, None, :] - pal_lab[None, :, :]) ** 2).sum(-1)
        improved = np.minimum(best[:, None], d).sum(axis=0)
        improved[chosen] = np.inf
        pick = int(np.argmin(improved))
        chosen.append(pick)
        best = np.minimum(best, d[:, pick])

    sub = palette[chosen]
    order = np.argsort(srgb_to_oklab(sub)[:, 0])
    return np.ascontiguousarray(sub[order], dtype=np.float32)
