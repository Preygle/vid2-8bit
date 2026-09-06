"""Video decoding via an ffmpeg raw pipe.

Frames are streamed as raw ``rgb24`` over stdout rather than written to disk as
PNGs. For a 5000-frame clip that difference is minutes of I/O and several GB of
scratch space, and it keeps the pipeline able to run as a generator so memory
stays flat regardless of clip length.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


class FFmpegMissing(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FFmpegMissing(
                f"{tool} not found on PATH. Install ffmpeg and ensure both "
                f"ffmpeg and ffprobe are available."
            )


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


def probe(path: str | Path) -> VideoInfo:
    """Read stream metadata with ffprobe."""
    require_ffmpeg()
    path = str(path)
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    if not data.get("streams"):
        raise ValueError(f"no video stream found in {path}")
    st = data["streams"][0]

    num, _, den = st.get("r_frame_rate", "25/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 25.0

    duration = 0.0
    for candidate in (st.get("duration"), data.get("format", {}).get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue

    try:
        n_frames = int(st.get("nb_frames"))
    except (TypeError, ValueError):
        # Containers such as MKV frequently omit nb_frames; estimate from
        # duration rather than forcing a full decode pass to count.
        n_frames = int(round(duration * fps)) if duration else 0

    return VideoInfo(
        path=path,
        width=int(st["width"]),
        height=int(st["height"]),
        fps=fps,
        n_frames=n_frames,
        duration=duration,
    )


def read_frames(
    path: str | Path,
    start: int = 0,
    count: int | None = None,
    step: int = 1,
    scale_width: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield frames as uint8 RGB arrays of shape (H, W, 3).

    `step` decodes every Nth frame, which is how palette fitting samples a shot
    cheaply. `scale_width` lets analysis passes work at reduced resolution.
    """
    require_ffmpeg()
    info = probe(path)
    w, h = info.width, info.height
    if scale_width and scale_width < w:
        h = max(2, int(round(h * scale_width / w)) // 2 * 2)
        w = int(scale_width)

    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start / info.fps:.6f}"]
    cmd += ["-i", str(path)]
    if count is not None:
        cmd += ["-frames:v", str(count * step)]

    filters = []
    if scale_width:
        filters.append(f"scale={w}:{h}:flags=lanczos")
    if step > 1:
        filters.append(f"select=not(mod(n\\,{step}))")
    if filters:
        cmd += ["-vf", ",".join(filters), "-vsync", "0"]

    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    frame_bytes = w * h * 3
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 2
    )
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait(timeout=10)


def read_frame(path: str | Path, index: int, scale_width: int | None = None) -> np.ndarray:
    """Decode a single frame by index."""
    for frame in read_frames(path, start=index, count=1, scale_width=scale_width):
        return frame.copy()
    raise IndexError(f"frame {index} not available in {path}")


def sample_frames(
    path: str | Path,
    n: int,
    start: int = 0,
    end: int | None = None,
    scale_width: int | None = 640,
) -> list[np.ndarray]:
    """Evenly sample `n` frames from a range.

    Used for palette fitting, which must see the whole shot -- fitting to a
    single frame produces a palette that fails as soon as the content changes.
    """
    info = probe(path)
    end = min(end if end is not None else info.n_frames, info.n_frames) or info.n_frames
    end = max(end, start + 1)
    if info.n_frames <= 0:
        indices = list(range(start, start + n))
    else:
        indices = np.unique(
            np.linspace(start, max(start, end - 1), num=max(1, n)).astype(int)
        ).tolist()

    step = max(1, (end - start) // max(1, len(indices)))
    frames: list[np.ndarray] = []
    for frame in read_frames(
        path, start=start, count=len(indices), step=step, scale_width=scale_width
    ):
        frames.append(frame.copy())
        if len(frames) >= len(indices):
            break
    if not frames:
        frames = [read_frame(path, start, scale_width=scale_width)]
    return frames
