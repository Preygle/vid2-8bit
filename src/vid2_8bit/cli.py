"""Command line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from .assist import AssistCache
from .config import available_presets, build_config
from .pipeline import Converter


def _build_overrides(args) -> dict:
    """Turn CLI flags into a nested config overlay."""
    over: dict = {}

    def put(section: str, key: str, value):
        if value is not None:
            over.setdefault(section, {})[key] = value

    if args.tier:
        over["tier"] = args.tier
    put("sample", "cell_size", args.cell)
    put("sample", "target_width", args.target_width)
    put("sample", "method", args.sample_method)
    put("palette", "size", args.palette_size)
    put("palette", "mode", args.palette_mode)
    put("palette", "hardware", args.hardware_palette)
    put("palette", "custom_path", args.palette_file)
    put("dither", "mode", args.dither)
    put("dither", "amount", args.dither_amount)
    put("abstract", "luma_bands", args.bands)
    put("abstract", "texture_removal", args.texture_removal)
    put("structure", "outlines", args.outlines)
    put("temporal", "enabled", args.temporal)
    put("temporal", "decimate_fps", args.decimate_fps)
    put("output", "scale", args.scale)
    put("output", "crt", args.crt)
    put("output", "pix_fmt", args.pix_fmt)

    if args.bits:
        parts = [int(b) for b in args.bits.replace("-", ",").split(",")]
        if len(parts) == 1:
            parts = parts * 3
        over.setdefault("palette", {})["bits_per_channel"] = parts
    if args.tiles:
        over["tiles"] = {"enabled": True}
    return over


def _config_from_args(args):
    cfg = build_config(
        preset=args.preset,
        user_yaml=args.config,
        overrides=_build_overrides(args),
    )
    if getattr(args, "assist_cache", None):
        cfg.assist_cache = args.assist_cache
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", "-p", default=None,
                   help=f"era preset: {', '.join(available_presets())}")
    p.add_argument("--config", default=None, help="YAML config file overlay")
    p.add_argument("--tier", choices=("fast", "quality"), default=None)
    p.add_argument("--cell", type=int, default=None, help="source pixels per output pixel")
    p.add_argument("--target-width", type=int, default=None,
                   help="logical width; overrides --cell")
    p.add_argument("--sample-method", choices=("area", "outline_expand", "superpixel"),
                   default=None)
    p.add_argument("--palette-size", type=int, default=None)
    p.add_argument("--palette-mode", choices=("auto", "hardware", "custom"), default=None)
    p.add_argument("--hardware-palette", default=None,
                   help="nes, gameboy, c64, pico8, cga, mastersystem, amiga")
    p.add_argument("--palette-file", default=None, help=".hex or .gpl palette")
    p.add_argument("--bits", default=None,
                   help="bits per channel, e.g. 5-5-5 or 3-3-2 (or a single number)")
    p.add_argument("--dither", choices=("none", "bayer2", "bayer4", "bayer8",
                                        "bluenoise", "floyd"), default=None)
    p.add_argument("--dither-amount", type=float, default=None)
    p.add_argument("--bands", type=int, default=None, help="luminance bands (0 = off)")
    p.add_argument("--texture-removal", type=float, default=None, help="0..1")
    p.add_argument("--tiles", action="store_true", default=None,
                   help="enforce hardware per-tile color limits")
    p.add_argument("--no-outlines", dest="outlines", action="store_false", default=None)
    p.add_argument("--no-temporal", dest="temporal", action="store_false", default=None)
    p.add_argument("--decimate-fps", type=float, default=None,
                   help="render on twos/threes, e.g. 12 or 15")
    p.add_argument("--scale", type=int, default=None, help="integer upscale factor")
    p.add_argument("--crt", action="store_true", default=None)
    p.add_argument("--pix-fmt", default=None)
    p.add_argument("--assist-cache", default=None, help="directory of assist sidecars")
    p.add_argument("-v", "--verbose", action="store_true")


def cmd_convert(args) -> int:
    cfg = _config_from_args(args)
    conv = Converter(cfg, AssistCache(cfg.assist_cache))

    width = 42

    def progress(done: int, total: int) -> None:
        if total:
            filled = int(width * done / total)
            bar = "#" * filled + "-" * (width - filled)
            sys.stderr.write(f"\r[{bar}] {done}/{total}")
        else:
            sys.stderr.write(f"\r{done} frames")
        sys.stderr.flush()

    stats = conv.convert(
        args.input,
        args.output,
        detect_cuts=not args.no_cuts,
        max_frames=args.max_frames,
        progress=None if args.quiet else progress,
        keep_audio=not args.no_audio,
    )
    if not args.quiet:
        sys.stderr.write("\n")
    print(stats.report())
    print(f"wrote {args.output}")
    return 0


def cmd_frame(args) -> int:
    from .io.encode import write_png

    cfg = _config_from_args(args)
    conv = Converter(cfg, AssistCache(cfg.assist_cache))
    src = Path(args.input)

    if src.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        import cv2

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"could not read {src}", file=sys.stderr)
            return 1
        out = conv.convert_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        out = conv.convert_frame(src, args.index)

    write_png(args.output, out)
    print(f"wrote {args.output} ({out.shape[1]}x{out.shape[0]})")
    return 0


def cmd_contact_sheet(args) -> int:
    from .io.encode import write_png
    from .tools_contact import build_contact_sheet

    presets = args.presets.split(",") if args.presets else available_presets()
    presets = [p.strip() for p in presets if p.strip() and p.strip() != "base"]
    sheet = build_contact_sheet(
        args.input, args.index, presets, columns=args.columns,
        max_width=args.max_width,
    )
    write_png(args.output, sheet)
    print(f"wrote {args.output} ({sheet.shape[1]}x{sheet.shape[0]}) "
          f"comparing {len(presets)} presets")
    return 0


def cmd_presets(args) -> int:
    from .config import load_preset_dict

    for name in available_presets():
        data = load_preset_dict(name)
        desc = str(data.get("description", "")).strip().replace("\n", " ")
        print(f"{name:14s} {desc}")
    return 0


def cmd_metrics(args) -> int:
    from .metrics import report_video

    print(report_video(args.input, max_frames=args.max_frames))
    return 0


def cmd_preview(args) -> int:
    try:
        from .gui.preview import run_preview
    except ImportError as exc:
        print(
            f"preview GUI unavailable: {exc}\n"
            f"install extras with:  pip install -e \".[gui]\"",
            file=sys.stderr,
        )
        return 1
    cfg = _config_from_args(args)
    return run_preview(args.input, cfg, args.index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vid2-8bit",
        description="Convert video to pixel art with configurable palette, "
                    "cell size and bit depth.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="convert a whole video")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-cuts", action="store_true", help="skip shot detection")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--quiet", "-q", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("frame", help="convert one frame or a still image")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--index", "-i", type=int, default=0)
    _add_common(p)
    p.set_defaults(func=cmd_frame)

    p = sub.add_parser("contact-sheet", help="A/B compare presets on one frame")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--index", "-i", type=int, default=0)
    p.add_argument("--presets", default=None, help="comma-separated preset names")
    p.add_argument("--columns", type=int, default=3)
    p.add_argument("--max-width", type=int, default=520)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_contact_sheet)

    p = sub.add_parser("presets", help="list available era presets")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("metrics", help="measure noise and temporal churn")
    p.add_argument("input")
    p.add_argument("--max-frames", type=int, default=60)
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("preview", help="interactive preview GUI")
    p.add_argument("input")
    p.add_argument("--index", "-i", type=int, default=0)
    _add_common(p)
    p.set_defaults(func=cmd_preview)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
