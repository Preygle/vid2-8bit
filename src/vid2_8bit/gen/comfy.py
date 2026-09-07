"""Minimal ComfyUI API client.

Talks to a locally running ComfyUI over its HTTP + websocket API: POST a
workflow graph to /prompt, wait for the run to finish, pull the images back
from /view.

Deliberately not a general ComfyUI wrapper. It builds the two graphs this
project needs and nothing else, because a general wrapper would need to track
ComfyUI's node schema across versions and that is a maintenance burden with no
payoff here.

Nothing in the core pipeline imports this. Generative work is optional, needs a
running server and several gigabytes of weights, and must never be on the path
of a plain deterministic render.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_HOST = "127.0.0.1:8188"


class ComfyUnavailable(RuntimeError):
    """ComfyUI is not reachable. Callers should degrade, not crash."""


@dataclass
class ComfyClient:
    host: str = DEFAULT_HOST
    timeout: float = 600.0

    # -- plumbing ----------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"http://{self.host}{path}"

    def _get_json(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(self._url(path), timeout=15) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ComfyUnavailable(
                f"ComfyUI not reachable at {self.host}: {exc}. "
                f"Start it with comfyui-rocm.bat."
            ) from exc

    def available(self) -> bool:
        try:
            self._get_json("/system_stats")
            return True
        except ComfyUnavailable:
            return False

    def object_info(self) -> dict:
        return self._get_json("/object_info")

    def models(self, kind: str) -> list[str]:
        """List models ComfyUI can see, e.g. kind='checkpoints'."""
        info = self.object_info()
        node = {"checkpoints": "CheckpointLoaderSimple",
                "loras": "LoraLoader",
                "controlnet": "ControlNetLoader"}.get(kind)
        if not node or node not in info:
            return []
        inputs = info[node]["input"]["required"]
        for key in ("ckpt_name", "lora_name", "control_net_name"):
            if key in inputs and isinstance(inputs[key][0], list):
                return list(inputs[key][0])
        return []

    # -- running -----------------------------------------------------------

    def run(self, graph: dict, poll: float = 1.0) -> list[np.ndarray]:
        """Queue a workflow and return its output images as RGB uint8 arrays."""
        client_id = str(uuid.uuid4())
        payload = json.dumps({"prompt": graph, "client_id": client_id}).encode()
        req = urllib.request.Request(
            self._url("/prompt"), data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                prompt_id = json.loads(r.read())["prompt_id"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:600]
            raise ComfyUnavailable(f"ComfyUI rejected the workflow: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ComfyUnavailable(f"could not queue workflow: {exc}") from exc

        # Poll history rather than holding a websocket: simpler, and a long
        # generation on a 10 GB card can outlast an idle socket.
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            hist = self._get_json(f"/history/{prompt_id}")
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error" or not status.get("completed", True):
                    msgs = status.get("messages", [])
                    raise ComfyUnavailable(f"workflow failed: {msgs}")
                return self._collect(entry)
            time.sleep(poll)
        raise ComfyUnavailable(f"workflow timed out after {self.timeout:.0f}s")

    def _collect(self, entry: dict) -> list[np.ndarray]:
        import cv2

        out: list[np.ndarray] = []
        for node_out in entry.get("outputs", {}).values():
            for img in node_out.get("images", []):
                q = urllib.parse.urlencode({
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                })
                with urllib.request.urlopen(self._url(f"/view?{q}"), timeout=60) as r:
                    buf = np.frombuffer(r.read(), dtype=np.uint8)
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if bgr is not None:
                    out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return out

    def upload_image(self, img_rgb: np.ndarray, name: str | None = None) -> str:
        """Upload an image for img2img. Returns the server-side filename."""
        import cv2

        name = name or f"vid2_8bit_{uuid.uuid4().hex[:12]}.png"
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("failed to encode image for upload")

        boundary = uuid.uuid4().hex
        body = b"".join([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'.encode(),
            b"Content-Type: image/png\r\n\r\n",
            buf.tobytes(),
            f"\r\n--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
            f"--{boundary}--\r\n".encode(),
        ])
        req = urllib.request.Request(
            self._url("/upload/image"), data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["name"]
        except (urllib.error.URLError, OSError) as exc:
            raise ComfyUnavailable(f"image upload failed: {exc}") from exc


# -- workflow graphs -------------------------------------------------------


def img2img_graph(
    image_name: str,
    checkpoint: str,
    positive: str,
    negative: str = "blurry, photographic, smooth gradients, jpeg artifacts, text",
    lora: str | None = None,
    lora_strength: float = 1.0,
    denoise: float = 0.55,
    steps: int = 22,
    cfg: float = 6.0,
    sampler: str = "dpmpp_2m",
    scheduler: str = "karras",
    seed: int = 0,
) -> dict:
    """Build a plain img2img graph in ComfyUI's API format.

    Denoise is the important dial. Above roughly 0.7 the model reinvents the
    scene and the result no longer matches the shot it came from, which matters
    here because these frames have to sit in a sequence.
    """
    g: dict = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
    }
    model_src, clip_src = ["1", 0], ["1", 1]
    if lora:
        g["4"] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": lora, "strength_model": lora_strength,
            "strength_clip": lora_strength, "model": ["1", 0], "clip": ["1", 1]}}
        model_src, clip_src = ["4", 0], ["4", 1]

    g["5"] = {"class_type": "CLIPTextEncode",
              "inputs": {"text": positive, "clip": clip_src}}
    g["6"] = {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": clip_src}}
    g["7"] = {"class_type": "KSampler", "inputs": {
        "seed": int(seed), "steps": int(steps), "cfg": float(cfg),
        "sampler_name": sampler, "scheduler": scheduler,
        "denoise": float(denoise), "model": model_src,
        "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["3", 0]}}
    g["8"] = {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["1", 2]}}
    g["9"] = {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "vid2_8bit", "images": ["8", 0]}}
    return g
