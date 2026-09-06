"""Video encoding via an ffmpeg raw pipe.

The default pixel format is ``yuv444p``, not the usual ``yuv420p``. This is not
a detail: 4:2:0 chroma subsampling averages color over 2x2 blocks, and pixel art
is *made* of hard 1-pixel color boundaries. Encoding this output at 4:2:0 visibly
smears every edge between two colors of similar luma, undoing the work of the
entire pipeline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .decode import require_ffmpeg


class VideoWriter:
    """Streaming writer. Use as a context manager."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        pix_fmt: str = "yuv444p",
        codec: str = "libx264",
        crf: int = 12,
        preset: str = "slow",
        audio_from: str | Path | None = None,
    ):
        require_ffmpeg()
        self.path = str(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._proc: subprocess.Popen | None = None
        self._frames = 0

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-v", "error", "-y", "-nostdin",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps}",
            "-i", "-",
        ]
        if audio_from:
            # Carry the original audio through untouched.
            cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", "aac", "-shortest"]

        cmd += ["-c:v", codec, "-pix_fmt", pix_fmt]
        if codec in ("libx264", "libx265"):
            cmd += ["-crf", str(int(crf)), "-preset", preset]
            # Nearest-neighbour upscaled art is all large flat areas and hard
            # edges, and the in-loop deblocking filter softens exactly those.
            # Turned fully negative rather than merely off. Note the value takes
            # a comma between alpha and beta; a colon separates distinct params.
            # `-tune animation` is deliberately NOT used -- it *raises* deblock
            # strength, which is the wrong direction here.
            if codec == "libx264":
                cmd += ["-x264-params", "deblock=-3,-3"]
        cmd.append(self.path)
        self._cmd = cmd

    def __enter__(self) -> "VideoWriter":
        self._proc = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return self

    def write(self, frame: np.ndarray) -> None:
        """Write one uint8 RGB frame of shape (H, W, 3)."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("VideoWriter used outside of its context manager")
        if frame.dtype != np.uint8:
            frame = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame is {frame.shape[:2]}, writer expects "
                f"{(self.height, self.width)}"
            )
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            # ffmpeg exited early. Its stderr explains why; without surfacing it
            # the caller only ever sees an opaque broken pipe.
            err = b""
            if self._proc.stderr:
                err = self._proc.stderr.read() or b""
            raise RuntimeError(
                f"ffmpeg exited after {self._frames} frames.\n"
                f"command: {' '.join(self._cmd)}\n"
                f"stderr: {err.decode('utf-8', 'replace')[-2000:]}"
            ) from exc
        self._frames += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except BrokenPipeError:
            pass
        _, err = self._proc.communicate(timeout=120)
        if self._proc.returncode != 0 and exc_type is None:
            raise RuntimeError(
                f"ffmpeg failed ({self._proc.returncode}):\n"
                f"{err.decode('utf-8', 'replace')[-2000:]}"
            )

    @property
    def frames_written(self) -> int:
        return self._frames


def write_png(path: str | Path, img: np.ndarray) -> None:
    """Write a single image. Accepts float [0,1] or uint8."""
    import cv2

    if img.dtype != np.uint8:
        img = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"failed to write {path}")
