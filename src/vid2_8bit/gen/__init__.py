"""Optional generative-AI assist.

Nothing here is imported by the core pipeline. It needs a running ComfyUI and
several gigabytes of weights, and a plain deterministic render must never
depend on it.

Two modes, for two different jobs:

* ``distill`` -- generate a handful of keyframes, measure the palette and grade
  the model chose, and emit a preset the deterministic renderer can execute
  across a whole clip. Keeps temporal stability; this is the mode for video.
* ``render``  -- run img2img on frames directly. Better-looking stills, but it
  re-rolls every frame, so video made this way boils. For single images.
"""

from .comfy import ComfyClient, ComfyUnavailable, img2img_graph
from .distill import StyleFit, keyframe_indices, measure, save_palette_hex

__all__ = [
    "ComfyClient", "ComfyUnavailable", "img2img_graph",
    "StyleFit", "measure", "keyframe_indices", "save_palette_hex",
]
