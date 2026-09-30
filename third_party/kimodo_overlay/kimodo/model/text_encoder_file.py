# dropbear-wbc local addition (2026-09-26), not part of upstream Kimodo.
"""Text encoder client over a shared folder (WSL side of dropbear-wbc's ``tools/kimodo_text_embed_server.py``).

Same interface as :class:`kimodo.model.text_encoder_api.TextEncoderAPI`: ``__call__(texts)`` returns
``(padded (N, L, D) tensor, lengths)``. Selected with ``TEXT_ENCODER_MODE=file``; the folder is
``KIMODO_EMBED_BRIDGE`` (default ``/mnt/h/hf_cache/embed_bridge``).
"""

import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch


class TextEncoderFile:
    def __init__(self, bridge_dir: str | None = None, timeout_s: float = 900.0):
        self.bridge = Path(bridge_dir or os.environ.get("KIMODO_EMBED_BRIDGE", "/mnt/h/hf_cache/embed_bridge"))
        self.timeout_s = float(timeout_s)
        self.device = "cpu"
        self.dtype = torch.float
        alive = self.bridge / "alive.json"
        if not alive.is_file() or time.time() - alive.stat().st_mtime > 30:
            raise RuntimeError(f"text-embed server not running (no fresh {alive}); start "
                               "tools/kimodo_text_embed_server.py on Windows")

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def _encode_one(self, text: str) -> np.ndarray:
        rid = uuid.uuid4().hex
        req, resp = self.bridge / "requests", self.bridge / "responses" / f"{rid}.npy"
        tmp = req / f"{rid}.txt.part"
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, req / f"{rid}.txt")
        t0 = time.time()
        while not resp.is_file():
            if time.time() - t0 > self.timeout_s:
                raise TimeoutError(f"no embedding for {text!r} after {self.timeout_s:.0f} s")
            time.sleep(0.05)
        arr = np.load(resp)
        resp.unlink(missing_ok=True)
        return arr

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        tensors = [self._encode_one(t) for t in texts]
        lengths = [t.shape[0] for t in tensors]
        padded = np.zeros((len(lengths), max(lengths), tensors[0].shape[-1]), dtype=np.float32)
        for i, (t, n) in enumerate(zip(tensors, lengths)):
            padded[i, :n] = t
        return torch.from_numpy(padded).to(device=self.device, dtype=self.dtype), lengths
