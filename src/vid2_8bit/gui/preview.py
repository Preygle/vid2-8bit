"""Interactive preview.

Tuning this pipeline by editing YAML and re-rendering is unworkable -- the
parameters interact (abstraction strength changes what the palette sees, which
changes what dithering does), so you need to see the result move as you drag.

Built on OpenCV's highgui rather than moderngl/imgui: OpenCV is already a hard
dependency, so the preview works out of the box with nothing extra to install.
The GPU path stays available for later, when the fast tier is ported to shaders.

Interactivity comes from running the fast tier at reduced resolution. The
palette is fitted once and reused across slider changes, because refitting
k-means on every drag is both slow and visually confusing -- colours would shift
underneath the parameter you are actually adjusting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from ..config import Config, available_presets, build_config
from ..io import decode
from ..io.encode import write_png
from ..pipeline import Converter

WINDOW = "vid2-8bit preview"

# (label, config path, slider max, scale factor from slider to value)
SLIDERS = [
    ("cell size", ("sample", "cell_size"), 24, 1.0),
    ("palette size", ("palette", "size"), 64, 1.0),
    ("texture removal x100", ("abstract", "texture_removal"), 100, 0.01),
    ("luma bands", ("abstract", "luma_bands"), 12, 1.0),
    ("saturation x100", ("abstract", "saturation"), 200, 0.01),
    ("outline x100", ("structure", "outline_strength"), 100, 0.01),
    ("outline darken x100", ("structure", "outline_darken"), 100, 0.01),
    ("dither amount x100", ("dither", "amount"), 100, 0.01),
    ("contrast weight x100", ("sample", "contrast_weight"), 100, 0.01),
]

DITHER_MODES = ["none", "bayer2", "bayer4", "bayer8", "bluenoise"]


def _get(cfg: Config, path: tuple[str, str]):
    return getattr(getattr(cfg, path[0]), path[1])


def _set(cfg: Config, path: tuple[str, str], value) -> None:
    setattr(getattr(cfg, path[0]), path[1], value)


def run_preview(source: str | Path, cfg: Config, frame_index: int = 0) -> int:
    """Open an interactive preview window. Returns a process exit code."""
    src = Path(source)
    if src.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"could not read {src}", file=sys.stderr)
            return 1
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        frame = decode.read_frame(src, frame_index)

    # Work at reduced resolution; the look scales with cell size, so a half-size
    # preview at half the cell size shows the same composition far faster.
    max_w = 720
    if frame.shape[1] > max_w:
        scale = max_w / frame.shape[1]
        frame = cv2.resize(
            frame, (max_w, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA
        )

    cfg.tier = "fast"
    presets = [p for p in available_presets() if p != "base"]
    state = {"preset": 0, "dither": DITHER_MODES.index(cfg.dither.mode)
             if cfg.dither.mode in DITHER_MODES else 0, "dirty": True}

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, frame.shape[1], frame.shape[0] + 40)

    def mark_dirty(_):
        state["dirty"] = True

    for label, path, maximum, scale in SLIDERS:
        value = _get(cfg, path) or 0
        cv2.createTrackbar(label, WINDOW, int(round(value / scale)), maximum, mark_dirty)
    cv2.createTrackbar("dither mode", WINDOW, state["dither"],
                       len(DITHER_MODES) - 1, mark_dirty)
    cv2.createTrackbar("preset", WINDOW, 0, len(presets) - 1, mark_dirty)

    print(
        "preview controls:\n"
        "  drag sliders to adjust    's' save PNG    'y' dump YAML\n"
        "  'p' apply selected preset    'q' or ESC to quit"
    )

    last_preset = -1
    display = None

    while True:
        if state["dirty"]:
            for label, path, _maximum, scale in SLIDERS:
                raw = cv2.getTrackbarPos(label, WINDOW)
                value = raw * scale
                _set(cfg, path, int(value) if scale == 1.0 else float(value))
            cfg.sample.cell_size = max(1, cfg.sample.cell_size or 1)
            cfg.palette.size = max(2, cfg.palette.size)
            cfg.dither.mode = DITHER_MODES[cv2.getTrackbarPos("dither mode", WINDOW)]

            try:
                out = Converter(cfg, collect_metrics=False).convert_image(frame)
                display = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            except Exception as exc:  # keep the window alive on a bad combo
                display = np.zeros((240, 640, 3), dtype=np.uint8)
                cv2.putText(display, str(exc)[:70], (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 240), 1, cv2.LINE_AA)
            state["dirty"] = False

        if display is not None:
            cv2.imshow(WINDOW, display)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("p"):
            idx = cv2.getTrackbarPos("preset", WINDOW)
            if idx != last_preset:
                cfg = build_config(presets[idx])
                cfg.tier = "fast"
                for label, path, _m, scale in SLIDERS:
                    value = _get(cfg, path) or 0
                    cv2.setTrackbarPos(label, WINDOW, int(round(value / scale)))
                if cfg.dither.mode in DITHER_MODES:
                    cv2.setTrackbarPos("dither mode", WINDOW,
                                       DITHER_MODES.index(cfg.dither.mode))
                last_preset = idx
                state["dirty"] = True
                print(f"applied preset: {presets[idx]}")
        if key == ord("s") and display is not None:
            path = f"preview_{src.stem}_{frame_index}.png"
            write_png(path, cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            print(f"saved {path}")
        if key == ord("y"):
            print(_as_yaml(cfg))

    cv2.destroyAllWindows()
    return 0


def _as_yaml(cfg: Config) -> str:
    """Dump the tuned values as a preset overlay, ready to paste into a YAML."""
    import yaml

    overlay = {
        "sample": {
            "cell_size": cfg.sample.cell_size,
            "contrast_weight": round(cfg.sample.contrast_weight, 3),
        },
        "palette": {"size": cfg.palette.size},
        "abstract": {
            "texture_removal": round(cfg.abstract.texture_removal, 3),
            "luma_bands": cfg.abstract.luma_bands,
            "saturation": round(cfg.abstract.saturation, 3),
        },
        "structure": {
            "outline_strength": round(cfg.structure.outline_strength, 3),
            "outline_darken": round(cfg.structure.outline_darken, 3),
        },
        "dither": {"mode": cfg.dither.mode, "amount": round(cfg.dither.amount, 3)},
    }
    return "# --- tuned preset overlay ---\n" + yaml.safe_dump(overlay, sort_keys=False)
