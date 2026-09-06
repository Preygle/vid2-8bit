"""Optional neural assist passes.

These provide *structure* (depth, segmentation, flow) -- never pixels. The core
pipeline renders every pixel deterministically; assists only inform decisions
like "is this edge a silhouette or interior texture".

The contract is deliberately file-based rather than in-process:

    python -m vid2_8bit.assist.depth --input clip.mp4 --out cache/depth/

writes ``%08d.npy`` plus a ``meta.json``, and this module reads them back. That
means an assist pass can be produced on a machine with a GPU and consumed on one
without -- which is the actual deployment situation for this project (see the
compute topology section of the README).

Nothing here imports torch. A missing cache is not an error; it disables the
features that depend on it and the render proceeds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PASSES = ("depth", "segment", "flow")


class AssistCache:
    """Read-only view over sidecar caches produced elsewhere."""

    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        self._meta: dict[str, dict] = {}
        self._warned: set[str] = set()
        if self.root and not self.root.is_dir():
            log.warning("assist cache %s does not exist; assists disabled", self.root)
            self.root = None

    # -- availability ------------------------------------------------------

    def available(self, pass_name: str) -> bool:
        return self.root is not None and (self.root / pass_name).is_dir()

    def meta(self, pass_name: str) -> dict:
        if pass_name in self._meta:
            return self._meta[pass_name]
        if not self.available(pass_name):
            return {}
        path = self.root / pass_name / "meta.json"
        data: dict = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("could not read %s: %s", path, exc)
        self._meta[pass_name] = data
        return data

    # -- access ------------------------------------------------------------

    def get(self, pass_name: str, frame_index: int) -> np.ndarray | None:
        """Load one frame's array, or None if unavailable."""
        if not self.available(pass_name):
            return None
        path = self.root / pass_name / f"{frame_index:08d}.npy"
        if not path.is_file():
            if pass_name not in self._warned:
                self._warned.add(pass_name)
                log.warning(
                    "assist '%s' cache has no entry for frame %d (%s); "
                    "continuing without it",
                    pass_name, frame_index, path,
                )
            return None
        try:
            return np.load(path)
        except (OSError, ValueError) as exc:
            log.warning("failed to load %s: %s", path, exc)
            return None

    def depth(self, frame_index: int) -> np.ndarray | None:
        return self.get("depth", frame_index)

    def segments(self, frame_index: int) -> np.ndarray | None:
        return self.get("segment", frame_index)

    def flow(self, frame_index: int) -> np.ndarray | None:
        return self.get("flow", frame_index)

    def summary(self) -> str:
        if self.root is None:
            return "assists: none"
        have = [p for p in PASSES if self.available(p)]
        return f"assists: {', '.join(have) if have else 'none found'} ({self.root})"


NULL_CACHE = AssistCache(None)
