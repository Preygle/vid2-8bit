"""Generate a synthetic test clip that exercises the pipeline's hard cases.

Real footage is the ultimate test, but a synthetic scene is reproducible and can
be built to contain exactly the things that break naive pixelation:

* **Fine facade detail** -- window grids at 4-8px, right at the aliasing
  threshold. This is what turns to salt-and-pepper noise without Stage 1.
* **Long straight architectural edges** -- these expose staircase artifacts and
  are what line snapping exists to fix.
* **A smooth sky gradient** -- exposes banding, and is the one place dithering
  genuinely helps.
* **A camera pan** -- exposes pixel crawl, which grid anchoring fixes.
* **An independently moving object** (a car) -- exposes ghosting from the
  temporal filter when occlusion masking is wrong.

Usage:
    python tools/make_testclip.py -o test_clip.mp4 --frames 48
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 1280, 720
WORLD_W = 2400  # wider than the frame so the camera can pan across it


def build_world(seed: int = 7) -> np.ndarray:
    """Paint a static wide scene once; the camera crops a window from it."""
    rng = np.random.default_rng(seed)
    world = np.zeros((HEIGHT, WORLD_W, 3), dtype=np.float32)

    # Sky: smooth vertical gradient, the banding/dither test.
    top = np.array([0.30, 0.48, 0.78], dtype=np.float32)
    bottom = np.array([0.86, 0.74, 0.58], dtype=np.float32)
    t = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None]
    world[:] = (top * (1 - t) + bottom * t)[:, None, :]

    # A sun, to give the sky a bright local feature.
    yy, xx = np.mgrid[0:HEIGHT, 0:WORLD_W].astype(np.float32)
    glow = np.exp(-(((xx - 420) ** 2 + (yy - 130) ** 2) / (2 * 95.0**2)))
    world += glow[..., None] * np.array([0.45, 0.38, 0.20], dtype=np.float32)

    # Buildings, back layer to front so nearer ones occlude.
    x = 40
    layer = 0
    while x < WORLD_W - 80:
        w = int(rng.integers(150, 290))
        h = int(rng.integers(240, 470))
        y0 = HEIGHT - 120 - h
        base = np.array(
            [rng.uniform(0.30, 0.62), rng.uniform(0.28, 0.58), rng.uniform(0.26, 0.55)],
            dtype=np.float32,
        )
        # Faces: a lit front and a shaded side, so there is real form to band.
        world[y0:HEIGHT - 120, x : x + w] = base
        side = min(28, w // 6)
        world[y0:HEIGHT - 120, x + w - side : x + w] = base * 0.68

        # Window grid -- the aliasing trap. Spacing lands near the cell size.
        wx, wy = 14, 20
        pad = 16
        lit = rng.random((h // wy + 2, w // wx + 2)) < 0.35
        for j, gy in enumerate(range(y0 + pad, HEIGHT - 132 - wy, wy)):
            for i, gx in enumerate(range(x + pad, x + w - side - wx, wx)):
                colour = (
                    np.array([0.95, 0.88, 0.55], np.float32)
                    if lit[j % lit.shape[0], i % lit.shape[1]]
                    else base * 0.45
                )
                world[gy : gy + wy - 12, gx : gx + wx - 7] = colour

        # Roof slab, a strong horizontal edge.
        world[y0 - 6 : y0, x - 4 : x + w + 4] = base * 1.25
        x += w + int(rng.integers(10, 60))
        layer += 1

    # Ground.
    world[HEIGHT - 120 :] = np.array([0.22, 0.21, 0.24], dtype=np.float32)
    world[HEIGHT - 120 : HEIGHT - 112] = np.array([0.34, 0.33, 0.36], dtype=np.float32)
    # Road markings, thin bright features on a dark ground.
    for mx in range(0, WORLD_W, 90):
        world[HEIGHT - 60 : HEIGHT - 54, mx : mx + 46] = 0.75

    return np.clip(world, 0, 1)


def draw_car(frame: np.ndarray, cx: int, cy: int) -> None:
    """A simple car silhouette -- curved edges and a dark outline."""
    body = (30, 60, 190)
    cv2.rectangle(frame, (cx - 70, cy - 18), (cx + 70, cy + 16), body, -1)
    cv2.ellipse(frame, (cx, cy - 18), (44, 26), 0, 180, 360, body, -1)
    # Windows.
    cv2.ellipse(frame, (cx, cy - 18), (36, 19), 0, 180, 360, (170, 205, 225), -1)
    cv2.line(frame, (cx, cy - 40), (cx, cy - 18), (20, 30, 60), 2)
    # Wheels.
    for wx in (cx - 42, cx + 42):
        cv2.circle(frame, (wx, cy + 16), 15, (25, 25, 28), -1)
        cv2.circle(frame, (wx, cy + 16), 6, (140, 140, 145), -1)
    # Outline, the thing that must survive downsampling.
    cv2.rectangle(frame, (cx - 70, cy - 18), (cx + 70, cy + 16), (15, 20, 45), 2)


def make_clip(out_path: str, frames: int, fps: float, add_cut: bool) -> None:
    world = build_world()
    world_u8 = np.clip(world * 255, 0, 255).astype(np.uint8)

    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "10",
            "-preset", "medium", out_path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    max_pan = WORLD_W - WIDTH
    try:
        for i in range(frames):
            phase = i / max(1, frames - 1)
            if add_cut and phase > 0.5:
                # Second half: a hard cut to a different part of the scene,
                # so shot detection and per-shot palettes get exercised.
                pan = int(max_pan * (0.85 - (phase - 0.5) * 0.4))
            else:
                pan = int(max_pan * phase * 0.9)

            frame = world_u8[:, pan : pan + WIDTH].copy()
            # Car crosses the frame independently of the camera.
            car_x = int(-120 + (WIDTH + 240) * ((i * 1.7 / frames) % 1.0))
            draw_car(frame, car_x, HEIGHT - 150)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        _, err = proc.communicate(timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(err.decode("utf-8", "replace")[-1500:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default="test_clip.mp4")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--no-cut", action="store_true", help="single continuous shot")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    make_clip(args.output, args.frames, args.fps, add_cut=not args.no_cut)
    print(f"wrote {args.output}  {WIDTH}x{HEIGHT}  {args.frames} frames @ {args.fps}fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
