"""Contact sheet builder -- the primary tuning instrument.

Pixel art style is not a thing you can evaluate from a parameter table; you have
to see variants next to each other on the same frame. This renders one source
frame through many presets and tiles the results into a labelled grid, so a
change to abstraction strength or palette size can be judged in one look rather
than by rendering, watching, tweaking and re-rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .config import build_config
from .pipeline import Converter

log = logging.getLogger(__name__)


def _label(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    """Draw a caption bar under an image."""
    bar_h = 30 if not sub else 46
    h, w = img.shape[:2]
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:h] = img
    canvas[h:] = (18, 18, 22)
    cv2.putText(canvas, text, (8, h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (240, 240, 240), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(canvas, sub, (8, h + 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (150, 150, 160), 1, cv2.LINE_AA)
    return canvas


def _fit(img: np.ndarray, max_width: int) -> np.ndarray:
    """Scale down to a display width, preserving hard pixel edges."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (max_width, max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_NEAREST)


def render_variants(
    source: str | Path,
    frame_index: int,
    presets: list[str],
    max_width: int = 520,
) -> list[tuple[str, str, np.ndarray]]:
    """Render one frame through each preset. Returns (name, caption, image)."""
    from .io import decode

    src = Path(source)
    is_image = src.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    if is_image:
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read {src}")
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        frame = decode.read_frame(src, frame_index)

    results: list[tuple[str, str, np.ndarray]] = [
        ("source", f"{frame.shape[1]}x{frame.shape[0]}", _fit(frame, max_width))
    ]

    for name in presets:
        try:
            cfg = build_config(preset=name)
            conv = Converter(cfg)
            out = conv.convert_image(frame)
            caption = (
                f"cell {cfg.sample.cell_size}  "
                f"pal {cfg.palette.size}"
                f"{'/' + cfg.palette.hardware if cfg.palette.hardware else ''}  "
                f"dither {cfg.dither.mode}"
            )
            results.append((name, caption, _fit(out, max_width)))
        except Exception as exc:  # a bad preset must not kill the whole sheet
            log.warning("preset %s failed: %s", name, exc)
            placeholder = np.zeros((120, max_width, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"FAILED: {exc}"[:60], (8, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 240), 1, cv2.LINE_AA)
            results.append((name, "failed", placeholder))

    return results


def build_contact_sheet(
    source: str | Path,
    frame_index: int,
    presets: list[str],
    columns: int = 3,
    max_width: int = 520,
    gap: int = 8,
) -> np.ndarray:
    """Render and tile a comparison grid."""
    variants = render_variants(source, frame_index, presets, max_width)
    tiles = [_label(img, name, caption) for name, caption, img in variants]

    cell_w = max(t.shape[1] for t in tiles)
    cell_h = max(t.shape[0] for t in tiles)
    rows = (len(tiles) + columns - 1) // columns

    sheet = np.full(
        (rows * cell_h + (rows + 1) * gap, columns * cell_w + (columns + 1) * gap, 3),
        12, dtype=np.uint8,
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        y = gap + r * (cell_h + gap)
        x = gap + c * (cell_w + gap)
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return sheet
