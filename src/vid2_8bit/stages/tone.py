"""Stage 0.5 -- tone mapping and colour grading.

This stage was missing from the original pipeline and it turned out to matter
more than almost anything downstream.

Cinematic footage is graded for a display, not for quantization. A night
exterior might occupy only lightness 0.13-0.58 of the available range, all of it
bunched in the shadows. Feed that to k-means and every cluster lands on a
slightly different dark blue; feed it to L0 and there are no strong edges to
snap to, so it invents smooth curved region boundaries. The result is a single
muddy blob, which is exactly what this pipeline produced before this stage
existed.

The reference frames make the point: the *Tetris* film's pixel sequences are not
colour-faithful to their source shots. A blue-teal night exterior becomes deep
black and warm maroon; an orange car against grey street becomes saturated
orange against desaturated blue. Somebody re-graded hard before stylizing.

So there are two operations here:

* **Auto levels** -- stretch lightness percentiles across the full range, which
  gives the downstream stages actual contrast to work with.
* **Reference transfer** -- match the colour statistics of a style reference
  (Reinhard et al., 2001), done in Oklab rather than the original paper's
  LAB-alpha-beta because the rest of this pipeline already lives in Oklab and it
  is the better-behaved space.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..color.spaces import oklab_to_srgb, srgb_to_oklab, u8_to_float


def auto_levels(
    img_srgb: np.ndarray,
    black_point: float = 1.0,
    white_point: float = 99.0,
    gamma: float = 1.0,
    strength: float = 1.0,
    target_range: float = 0.85,
) -> np.ndarray:
    """Stretch lightness so the footage uses the whole range.

    Percentiles rather than min/max: one specular highlight or a single crushed
    black pixel would otherwise define the whole mapping and nothing would move.

    The stretch is *adaptive*. A night exterior using a third of the range needs
    a large lift; a well-exposed daylight shot needs almost none, and forcing the
    same stretch on it blows highlights to flat white -- an orange car roof turns
    into a white blob. So the applied amount scales with how compressed the input
    actually is, capped by `strength`. `target_range` is the lightness spread
    considered "already fine".
    """
    lab = srgb_to_oklab(img_srgb)
    _levels_in_place(lab, black_point, white_point, gamma, strength, target_range)
    return oklab_to_srgb(lab)


def _levels_in_place(lab, black_point, white_point, gamma, strength, target_range):
    """Auto-levels applied directly to an Oklab array. See auto_levels()."""
    L = lab[..., 0]
    lo = float(np.percentile(L, black_point))
    hi = float(np.percentile(L, white_point))
    span = hi - lo
    if span < 1e-6 or strength <= 0.0:
        return

    # 0 when the image already spans target_range, rising as it gets flatter.
    need = float(np.clip((target_range - span) / max(target_range, 1e-6), 0.0, 1.0))
    amount = float(np.clip(strength, 0.0, 1.0)) * need
    if amount < 1e-3 and abs(gamma - 1.0) < 1e-3:
        return

    stretched = np.clip((L - lo) / span, 0.0, 1.0)
    blended = L * np.float32(1.0 - amount) + stretched * np.float32(amount)
    if abs(gamma - 1.0) > 1e-3:
        blended = np.clip(blended, 0.0, 1.0) ** np.float32(gamma)
    lab[..., 0] = blended


def two_pole_contrast(img_srgb: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Push lightness toward a black pole and a highlight pole.

    A blind critic comparing our output to the film frame identified the absence
    of this as the single biggest tell: our whole picture sat inside the bottom
    8% of the luminance scale with a median around 0.09, while the film anchors
    on a true ink and pushes highlights to near-white with very little in
    between. Night-city pixel art is built as two poles, not a ramp.

    A smoothstep S-curve about the midpoint does that: it drags shadows to ink
    and lifts highlights, while leaving the midtone crossing where it was so the
    image does not simply get darker or lighter overall.
    """
    if amount <= 0.0:
        return img_srgb
    lab = srgb_to_oklab(img_srgb)
    _contrast_in_place(lab, amount)
    return oklab_to_srgb(lab)


def _contrast_in_place(lab, amount: float) -> None:
    """Two-pole S-curve applied directly to an Oklab array."""
    if amount <= 0.0:
        return
    L = np.clip(lab[..., 0], 0.0, 1.0)
    s_curve = L * L * (np.float32(3.0) - np.float32(2.0) * L)   # smoothstep
    a = np.float32(np.clip(amount, 0.0, 1.0))
    lab[..., 0] = L * (np.float32(1.0) - a) + s_curve * a


def image_stats(img_srgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation in Oklab."""
    lab = srgb_to_oklab(img_srgb).reshape(-1, 3)
    return lab.mean(axis=0), lab.std(axis=0) + 1e-6


_REF_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def reference_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Oklab mean/std of a style reference image, cached by path."""
    key = str(path)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    bgr = cv2.imread(key, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read style reference: {path}")
    rgb = u8_to_float(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    stats = image_stats(rgb)
    _REF_CACHE[key] = stats
    return stats


def color_transfer(
    img_srgb: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Reinhard colour transfer in Oklab.

    Rescales each Oklab axis so the image's mean and spread match the
    reference's. This is what moves a teal night exterior onto the film's warm
    maroon palette -- the hue relationship is carried by the a/b means, and the
    contrast by the standard deviations.

    `strength` blends against the original, so a grade can be dialled in rather
    than applied wholesale.
    """
    if strength <= 0.0:
        return img_srgb
    lab = srgb_to_oklab(img_srgb)
    _transfer_in_place(lab, ref_mean, ref_std, strength)
    return oklab_to_srgb(lab)


def _transfer_in_place(lab, ref_mean, ref_std, strength: float) -> None:
    """Reinhard transfer applied directly to an Oklab array."""
    if strength <= 0.0:
        return
    flat = lab.reshape(-1, 3)
    src_mean = flat.mean(axis=0)
    src_std = flat.std(axis=0) + 1e-6
    s = np.float32(np.clip(strength, 0.0, 1.0))
    scale = (ref_std / src_std).astype(np.float32)
    shifted = (lab - src_mean) * scale + ref_mean
    lab *= np.float32(1.0) - s
    lab += shifted * s


def apply_tone(img_srgb: np.ndarray, cfg) -> np.ndarray:
    """Run the tone stage for one frame."""
    tcfg = getattr(cfg, "tone", None)
    if tcfg is None or not tcfg.enabled:
        return img_srgb

    needs = (
        tcfg.auto_levels
        or tcfg.contrast > 0.0
        or (tcfg.reference and tcfg.transfer_strength > 0.0)
        or abs(tcfg.saturation - 1.0) > 1e-3
    )
    if not needs:
        return img_srgb

    # ONE Oklab round trip for the whole stage. Each conversion costs roughly
    # 200ms on an HD frame, and doing levels, contrast, transfer and saturation
    # as separate sRGB-level calls meant six of them per frame -- profiling put
    # colour conversion at about 70% of total render time, most of it here.
    lab = srgb_to_oklab(img_srgb)

    if tcfg.auto_levels:
        _levels_in_place(lab, tcfg.black_point, tcfg.white_point, tcfg.gamma,
                         tcfg.levels_strength, tcfg.target_range)
    _contrast_in_place(lab, tcfg.contrast)
    if tcfg.reference and tcfg.transfer_strength > 0.0:
        mean, std = reference_stats(tcfg.reference)
        _transfer_in_place(lab, mean, std, tcfg.transfer_strength)
    if abs(tcfg.saturation - 1.0) > 1e-3:
        lab[..., 1:] *= np.float32(tcfg.saturation)

    return oklab_to_srgb(lab)


def palette_from_reference(path: str | Path, size: int, chroma_weight: float = 1.0):
    """Fit a palette to a style reference image.

    Lets a render adopt a reference's actual colours rather than its statistics.
    The reference is denoised first: these are usually video screenshots, where
    codec noise turns a 16-colour frame into tens of thousands of near-duplicates
    and would otherwise dominate the fit.
    """
    from ..color.palette import fit_palette

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read palette reference: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    denoised = cv2.bilateralFilter(cv2.medianBlur(rgb, 5), 9, 40, 9)
    flat = u8_to_float(denoised).reshape(-1, 3)
    if len(flat) > 200000:
        rng = np.random.default_rng(0)
        flat = flat[rng.choice(len(flat), size=200000, replace=False)]
    return fit_palette(flat, size, chroma_weight=chroma_weight)
