"""Local web UI for tweaking conversion parameters.

Built on the standard library's http.server rather than Flask or FastAPI. Those
are both installed on the dev machine today, but this is launched from a .bat
file by someone who just wants it to work; depending on a package that might get
uninstalled or shadowed by a different interpreter is a failure mode not worth
accepting for a tool this small.

Two caches make slider-dragging feel immediate:

* the decoded source frame, so moving a colour slider does not re-decode video;
* the fitted palette, keyed only on the parameters that actually affect it, so
  adjusting e.g. outline strength does not re-run k-means -- which would also be
  visually confusing, since the colours would shift under the control being
  dragged.

Binds to 127.0.0.1 only. This exposes a filesystem-path parameter and starts
renders, so it must not be reachable from the network.
"""

from __future__ import annotations

import io
import json
import mimetypes
import threading
import traceback
import webbrowser
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from ..config import available_presets, build_config, load_preset_dict
from ..io import decode
from ..io.encode import write_png
from ..pipeline import Converter

STATIC_DIR = Path(__file__).parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}

# Parameters that change the fitted palette. Anything else can be tweaked
# without paying for a refit.
PALETTE_KEYS = (
    "preset", "source", "frame", "palette_mode", "palette_size",
    "hardware_palette", "bits", "texture_removal", "luma_bands", "saturation",
    "redistribute", "anchor_ink", "style_reference",
    "contrast", "levels_strength", "tone_gamma", "transfer_strength",
)


@dataclass
class JobState:
    """Progress of a full-video render."""

    running: bool = False
    done: int = 0
    total: int = 0
    message: str = ""
    output: str = ""
    error: str = ""

    def snapshot(self) -> dict:
        return asdict(self)


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame_cache: dict[tuple, np.ndarray] = {}
        self.palette_cache: dict[tuple, np.ndarray] = {}
        self.job = JobState()
        self.job_thread: threading.Thread | None = None


STATE = AppState()


# -- parameter plumbing ----------------------------------------------------


def _f(params: dict, key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def _i(params: dict, key: str, default: int) -> int:
    try:
        return int(float(params.get(key, default)))
    except (TypeError, ValueError):
        return default


def _b(params: dict, key: str, default: bool = False) -> bool:
    v = params.get(key, default)
    return v in (True, "true", "True", 1, "1", "on")


def config_from_params(params: dict):
    """Build a Config from the UI's flat parameter dict."""
    preset = params.get("preset") or None
    if preset in ("", "none", "custom"):
        preset = None

    overrides: dict = {
        "tone": {
            "enabled": _b(params, "tone_enabled", True),
            "auto_levels": _b(params, "auto_levels", True),
            "levels_strength": _f(params, "levels_strength", 1.0),
            "contrast": _f(params, "contrast", 0.0),
            "gamma": _f(params, "tone_gamma", 1.0),
            "saturation": _f(params, "tone_saturation", 1.0),
            "transfer_strength": _f(params, "transfer_strength", 0.0),
        },
        "abstract": {
            "texture_removal": _f(params, "texture_removal", 0.35),
            "luma_bands": _i(params, "luma_bands", 0),
            "saturation": _f(params, "saturation", 1.15),
            "smooth_radius_cells": _f(params, "smooth_radius_cells", 0.0),
        },
        "structure": {
            "outlines": _b(params, "outlines", True),
            "outline_strength": _f(params, "outline_strength", 0.75),
            "outline_darken": _f(params, "outline_darken", 0.55),
            "snap_lines": _b(params, "snap_lines", False),
        },
        "sample": {
            "method": params.get("sample_method", "outline_expand"),
            "contrast_weight": _f(params, "contrast_weight", 0.7),
        },
        "palette": {
            "mode": params.get("palette_mode", "auto"),
            "size": max(2, _i(params, "palette_size", 24)),
            "redistribute": _f(params, "redistribute", 0.0),
            "anchor_ink": _b(params, "anchor_ink", False),
        },
        "dither": {
            "mode": params.get("dither_mode", "none"),
            "amount": _f(params, "dither_amount", 0.5),
            "selective_threshold": _f(params, "selective_threshold", 0.02),
        },
        "tiles": {"enabled": _b(params, "tiles", False)},
        "temporal": {
            "enabled": _b(params, "temporal", True),
            "decimate_fps": _f(params, "decimate_fps", 0.0),
        },
        "output": {"crt": _b(params, "crt", False),
                   "fps": _f(params, "output_fps", 0.0)},
        "performance": {
            "oversample": _f(params, "oversample", 6.0),
            "workers": _i(params, "workers", 0),
        },
    }

    # Grid: logical width wins when set, otherwise cell size. Sending both would
    # silently override a preset that defines its geometry as a target width.
    tw = _i(params, "target_width", 0)
    if tw > 0:
        overrides["sample"]["target_width"] = tw
        overrides["sample"]["cell_size"] = None
    else:
        overrides["sample"]["cell_size"] = max(1, _i(params, "cell_size", 6))
        overrides["sample"]["target_width"] = None

    if overrides["palette"]["mode"] == "hardware":
        overrides["palette"]["hardware"] = params.get("hardware_palette") or "nes"
    # A style reference drives both the grade and, optionally, the palette.
    ref = (params.get("style_reference") or "").strip()
    if ref:
        overrides["tone"]["reference"] = ref
        if overrides["palette"]["mode"] == "reference":
            overrides["palette"]["reference"] = ref
    elif overrides["palette"]["mode"] == "reference":
        overrides["palette"]["mode"] = "auto"

    bits = (params.get("bits") or "").strip()
    if bits and bits.lower() not in ("none", "off", "8-8-8"):
        parts = [int(b) for b in bits.replace("-", ",").split(",") if b.strip()]
        if len(parts) == 1:
            parts *= 3
        if len(parts) == 3:
            overrides["palette"]["bits_per_channel"] = parts

    cfg = build_config(preset=preset, overrides=overrides)
    cfg.tier = params.get("tier", "fast")
    return cfg


# -- file selection --------------------------------------------------------

# Runs in a throwaway subprocess. Tk must own a thread's event loop, and driving
# it from inside a served request tends to deadlock or crash on Windows; a
# separate process sidesteps the whole problem and cannot take the server down.
_PICKER_SCRIPT = r"""
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
path = filedialog.askopenfilename(
    title="Choose a video or image",
    filetypes=[
        ("Video and image files",
         "*.mp4 *.mkv *.mov *.webm *.avi *.m4v *.png *.jpg *.jpeg *.bmp *.webp"),
        ("Video files", "*.mp4 *.mkv *.mov *.webm *.avi *.m4v"),
        ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
        ("All files", "*.*"),
    ],
)
root.destroy()
sys.stdout.write(path or "")
"""


def native_file_dialog() -> dict:
    """Open the OS file picker on the machine running the server.

    Preferred over uploading because this is a localhost tool: the file is
    already on this disk, so picking it costs nothing, while uploading a
    multi-gigabyte video through the browser would copy it for no reason.
    """
    import subprocess
    import sys

    kwargs: dict = {}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PICKER_SCRIPT],
            capture_output=True, text=True, timeout=600, **kwargs
        )
    except FileNotFoundError:
        return {"ok": False, "error": "could not launch the file dialog"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "file dialog timed out"}

    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        return {
            "ok": False,
            "error": f"native file dialog unavailable ({detail[0]}). "
                     f"Use the Upload button instead.",
        }
    path = (proc.stdout or "").strip()
    if not path:
        return {"ok": False, "cancelled": True}
    return {"ok": True, "path": path}


def uploads_dir() -> Path:
    import tempfile

    d = Path(tempfile.gettempdir()) / "vid2-8bit-uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_upload(filename: str, data: bytes) -> Path:
    """Persist a browser-uploaded file and return its path.

    The browser posts raw bytes with the name in the query string rather than a
    multipart form: the stdlib's `cgi` module was removed in Python 3.13, and
    hand-rolling a multipart parser to receive a single file would be pure
    overhead when we control both ends.
    """
    safe = Path(filename or "upload").name
    safe = "".join(c for c in safe if c.isalnum() or c in "._- ()[]").strip() or "upload"
    if Path(safe).suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        raise ValueError(f"unsupported file type: {Path(safe).suffix or '(none)'}")
    target = uploads_dir() / safe
    target.write_bytes(data)
    return target


def _resolve_source(raw: str) -> Path:
    path = Path(raw.strip().strip('"').strip("'")).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"not found: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        raise ValueError(f"unsupported file type: {path.suffix}")
    return path


def load_source_frame(source: str, frame_index: int, preview_width: int) -> np.ndarray:
    key = (source, frame_index, preview_width)
    with STATE.lock:
        if key in STATE.frame_cache:
            return STATE.frame_cache[key]

    path = _resolve_source(source)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read {path}")
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        frame = decode.read_frame(path, max(0, frame_index))

    if preview_width and frame.shape[1] > preview_width:
        scale = preview_width / frame.shape[1]
        frame = cv2.resize(
            frame, (preview_width, max(2, int(round(frame.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    with STATE.lock:
        if len(STATE.frame_cache) > 16:
            STATE.frame_cache.clear()
        STATE.frame_cache[key] = frame
    return frame


def source_info(source: str) -> dict:
    path = _resolve_source(source)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read {path}")
        h, w = bgr.shape[:2]
        return {"path": str(path), "name": path.name, "kind": "image",
                "width": w, "height": h, "frames": 1, "fps": 0}
    info = decode.probe(path)
    return {
        # `name` is sent so the browser never has to split OS paths itself --
        # a JS regex that handles both separators is easy to get subtly wrong,
        # and the server already knows the answer.
        "path": str(path), "name": path.name, "kind": "video",
        "width": info.width, "height": info.height,
        "frames": max(1, info.n_frames), "fps": info.fps,
    }


def render_preview(params: dict) -> tuple[bytes, dict]:
    """Render one frame and return (PNG bytes, stats)."""
    preview_width = max(160, _i(params, "preview_width", 640))
    frame_index = _i(params, "frame", 0)
    source = params.get("source", "")
    if not source:
        raise ValueError("no source file set")

    full = source_info(source)
    frame = load_source_frame(source, frame_index, preview_width)
    cfg = config_from_params(params)

    # Keep the preview's LOGICAL resolution identical to the full render's.
    # The preview works from a downscaled source, so applying the full cell size
    # to it would double the chunkiness and show something the final render will
    # never produce. Scaling the cell by the same factor as the source keeps
    # src_w/cell constant, so the preview predicts the output faithfully.
    shrink = frame.shape[1] / float(full["width"])
    if shrink < 1.0:
        target_logical_w = max(1, int(full["width"]) // max(1, cfg.sample.cell_size))
        cfg.sample.cell_size = None
        cfg.sample.target_width = min(frame.shape[1], target_logical_w)

    conv = Converter(cfg, collect_metrics=True)
    pal_key = tuple(str(params.get(k, "")) for k in PALETTE_KEYS) + (preview_width,)
    with STATE.lock:
        cached = STATE.palette_cache.get(pal_key)

    if cached is None:
        from ..color import palette as palette_mod
        from ..color.spaces import u8_to_float
        from ..stages import abstract as abstract_mod

        cell = cfg.effective_cell(frame.shape[1], frame.shape[0])
        abstracted = abstract_mod.abstract_frame(
            u8_to_float(frame), cfg, cell, tier=cfg.tier
        )
        flat = abstracted.reshape(-1, 3)
        rng = np.random.default_rng(0)
        n = min(cfg.palette.sample_pixels, len(flat))
        sample = flat[rng.choice(len(flat), size=n, replace=False)]
        cached = palette_mod.build_palette(cfg, sample)
        with STATE.lock:
            if len(STATE.palette_cache) > 32:
                STATE.palette_cache.clear()
            STATE.palette_cache[pal_key] = cached

    cell = cfg.effective_cell(frame.shape[1], frame.shape[0])
    state = conv.new_shot_state(cached, cell)
    size = cfg.logical_size(frame.shape[1], frame.shape[0])
    from ..stages import output as output_mod

    scale = cfg.output.scale or output_mod.integer_scale(
        size[0], size[1], frame.shape[1], frame.shape[0]
    )
    out = conv.render_frame(
        frame, state, frame_index=frame_index,
        target_size=(size[0] * scale, size[1] * scale),
    )

    ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode preview PNG")

    stats = {
        "logical": f"{size[0]}x{size[1]}",
        "output": f"{out.shape[1]}x{out.shape[0]}",
        "palette": int(len(cached)),
        "cell": round(cell, 2),
        "noise": round(conv.stats.noise * 100, 2),
        "colors": [
            "#%02X%02X%02X" % tuple(np.rint(c * 255).astype(int)) for c in cached
        ],
    }
    return buf.tobytes(), stats


def preset_overlay_yaml(params: dict) -> str:
    import yaml

    cfg = config_from_params(params)
    overlay = {
        "description": "Tuned in the web UI.",
        "extends": "base",
        "sample": {
            "cell_size": cfg.sample.cell_size,
            "method": cfg.sample.method,
            "contrast_weight": round(cfg.sample.contrast_weight, 3),
        },
        "palette": {"mode": cfg.palette.mode, "size": cfg.palette.size},
        "abstract": {
            "texture_removal": round(cfg.abstract.texture_removal, 3),
            "luma_bands": cfg.abstract.luma_bands,
            "saturation": round(cfg.abstract.saturation, 3),
        },
        "structure": {
            "outlines": cfg.structure.outlines,
            "outline_strength": round(cfg.structure.outline_strength, 3),
            "outline_darken": round(cfg.structure.outline_darken, 3),
            "snap_lines": cfg.structure.snap_lines,
        },
        "dither": {
            "mode": cfg.dither.mode,
            "amount": round(cfg.dither.amount, 3),
            "selective_threshold": round(cfg.dither.selective_threshold, 4),
        },
    }
    if cfg.palette.hardware:
        overlay["palette"]["hardware"] = cfg.palette.hardware
    if cfg.palette.bits_per_channel:
        overlay["palette"]["bits_per_channel"] = cfg.palette.bits_per_channel
    if cfg.tiles.enabled:
        overlay["tiles"] = {"enabled": True}
    return yaml.safe_dump(overlay, sort_keys=False)


# -- full render job -------------------------------------------------------


def start_convert(params: dict) -> dict:
    with STATE.lock:
        if STATE.job.running:
            return {"ok": False, "error": "a render is already running"}
        STATE.job = JobState(running=True, message="starting")

    source = _resolve_source(params.get("source", ""))
    out_path = (params.get("output") or "").strip()
    if not out_path:
        out_path = str(source.with_name(source.stem + "_pixelart.mp4"))

    cfg = config_from_params(params)
    cfg.tier = params.get("render_tier", "quality")
    max_frames = _i(params, "max_frames", 0) or None

    def worker() -> None:
        try:
            conv = Converter(cfg)

            def progress(done: int, total: int) -> None:
                with STATE.lock:
                    STATE.job.done = done
                    STATE.job.total = total
                    STATE.job.message = f"rendering frame {done}/{total or '?'}"

            if source.suffix.lower() in IMAGE_SUFFIXES:
                bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                png = out_path if out_path.lower().endswith(".png") else out_path + ".png"
                write_png(png, conv.convert_image(rgb))
                final = png
                summary = f"wrote {png}"
            else:
                stats = conv.convert(
                    source, out_path, max_frames=max_frames,
                    progress=progress, keep_audio=_b(params, "keep_audio", True),
                )
                final = out_path
                summary = stats.report()

            with STATE.lock:
                STATE.job.running = False
                STATE.job.output = final
                STATE.job.message = summary
        except Exception as exc:  # surface the reason in the UI, never hang
            with STATE.lock:
                STATE.job.running = False
                STATE.job.error = f"{type(exc).__name__}: {exc}"
                STATE.job.message = "failed"
            traceback.print_exc()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    STATE.job_thread = thread
    return {"ok": True, "output": out_path}


# -- HTTP ------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "vid2-8bit"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/render" not in (self.path or ""):
            super().log_message(fmt, *args)

    # -- helpers -----------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    # -- routes ------------------------------------------------------------

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        try:
            if route in ("/", "/index.html"):
                html = (STATIC_DIR / "index.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")

            if route == "/api/presets":
                out = []
                for name in available_presets():
                    if name == "base":
                        continue
                    data = load_preset_dict(name)
                    out.append({
                        "name": name,
                        "description": str(data.get("description", "")).strip(),
                    })
                return self._json({"presets": out})

            if route == "/api/preset":
                cfg = build_config(query.get("name") or "base")
                return self._json({"ok": True, "config": {
                    "cell_size": cfg.sample.cell_size or 6,
                    "target_width": cfg.sample.target_width or 0,
                    "sample_method": cfg.sample.method,
                    "contrast_weight": cfg.sample.contrast_weight,
                    "palette_mode": cfg.palette.mode,
                    "palette_size": cfg.palette.size,
                    "hardware_palette": cfg.palette.hardware or "nes",
                    "bits": "-".join(map(str, cfg.palette.bits_per_channel))
                            if cfg.palette.bits_per_channel else "",
                    "texture_removal": cfg.abstract.texture_removal,
                    "luma_bands": cfg.abstract.luma_bands,
                    "saturation": cfg.abstract.saturation,
                    "smooth_radius_cells": cfg.abstract.smooth_radius_cells,
                    "outlines": cfg.structure.outlines,
                    "outline_strength": cfg.structure.outline_strength,
                    "outline_darken": cfg.structure.outline_darken,
                    "snap_lines": cfg.structure.snap_lines,
                    "dither_mode": cfg.dither.mode,
                    "dither_amount": cfg.dither.amount,
                    "selective_threshold": cfg.dither.selective_threshold,
                    "redistribute": cfg.palette.redistribute,
                    "anchor_ink": cfg.palette.anchor_ink,
                    "contrast": cfg.tone.contrast,
                    "levels_strength": cfg.tone.levels_strength,
                    "tone_gamma": cfg.tone.gamma,
                    "tone_saturation": cfg.tone.saturation,
                    "transfer_strength": cfg.tone.transfer_strength,
                    "auto_levels": cfg.tone.auto_levels,
                    "decimate_fps": cfg.temporal.decimate_fps,
                    "output_fps": cfg.output.fps,
                    "oversample": cfg.performance.oversample,
                    "tiles": cfg.tiles.enabled,
                    "crt": cfg.output.crt,
                }})

            if route == "/api/source":
                return self._json({"ok": True, **source_info(query.get("path", ""))})

            if route == "/api/original":
                frame = load_source_frame(
                    query.get("source", ""), int(query.get("frame", 0)),
                    int(query.get("preview_width", 640)),
                )
                ok, buf = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                return self._send(200, buf.tobytes(), "image/png")

            if route == "/api/progress":
                with STATE.lock:
                    return self._json(STATE.job.snapshot())

            return self._json({"ok": False, "error": "not found"}, 404)

        except Exception as exc:
            traceback.print_exc()
            return self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)

    def do_POST(self):
        parsed = urlparse(self.path)
        route = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            # Upload carries raw file bytes, not JSON, so it is handled first.
            if route == "/api/upload":
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    return self._json({"ok": False, "error": "empty upload"}, 400)
                if length > 4 * 1024 * 1024 * 1024:
                    return self._json(
                        {"ok": False,
                         "error": "file larger than 4 GB; use Browse instead, "
                                  "which needs no copy"}, 400)
                # Read in chunks so a large video does not spike memory as badly
                # and a truncated stream fails cleanly.
                buf = io.BytesIO()
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    buf.write(chunk)
                    remaining -= len(chunk)
                if remaining > 0:
                    return self._json({"ok": False, "error": "upload truncated"}, 400)
                path = save_upload(query.get("name", "upload"), buf.getvalue())
                return self._json({"ok": True, **source_info(str(path))})

            if route == "/api/browse":
                return self._json(native_file_dialog())

            params = self._read_json()

            if route == "/api/render":
                png, stats = render_preview(params)
                return self._send(200, png, "image/png",
                                  {"X-Stats": json.dumps(stats)})

            if route == "/api/yaml":
                return self._json({"ok": True, "yaml": preset_overlay_yaml(params)})

            if route == "/api/convert":
                return self._json(start_convert(params))

            return self._json({"ok": False, "error": "not found"}, 404)

        except Exception as exc:
            traceback.print_exc()
            return self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)


def serve(host: str = "127.0.0.1", port: int = 8750, open_browser: bool = True) -> int:
    mimetypes.add_type("image/png", ".png")
    # Port may be taken by a previous run; walk forward rather than dying.
    for attempt in range(20):
        try:
            httpd = ThreadingHTTPServer((host, port + attempt), Handler)
            break
        except OSError:
            continue
    else:
        print(f"could not bind a port in {port}..{port + 19}")
        return 1

    url = f"http://{host}:{httpd.server_address[1]}/"
    print("=" * 62)
    print("  vid2-8bit web UI")
    print(f"  {url}")
    print("  Ctrl+C to stop")
    print("=" * 62)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0
