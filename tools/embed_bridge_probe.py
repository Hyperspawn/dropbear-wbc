"""Embed a prompt list through the running text-embed server (tools/kimodo_text_embed_server.py) and save the vectors.

Used to compare encoder variants (e.g. bf16 vs --int8) on the same prompts:
  python tools/embed_bridge_probe.py data/motions_gen/prompts/everyday_v1.json logs/live/embed_ref_bf16.npz
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
BRIDGE = Path(os.environ.get("KIMODO_EMBED_BRIDGE_WIN") or _paths.hf_cache() / "embed_bridge")


def embed(text: str, timeout_s: float = 600.0) -> np.ndarray:
    rid = uuid.uuid4().hex
    req, resp = BRIDGE / "requests", BRIDGE / "responses" / f"{rid}.npy"
    (req / f"{rid}.txt.part").write_text(text, encoding="utf-8")
    os.replace(req / f"{rid}.txt.part", req / f"{rid}.txt")
    t0 = time.time()
    while not resp.is_file():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(text)
        time.sleep(0.05)
    time.sleep(0.05)
    a = np.load(resp)
    resp.unlink(missing_ok=True)
    return a


def main() -> int:
    items = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    texts = [it["prompt"] for it in items]
    vecs, secs = [], []
    for t in texts:
        t0 = time.time()
        vecs.append(embed(t).reshape(-1))
        secs.append(time.time() - t0)
        print(f"{len(vecs)}/{len(texts)} {secs[-1]:.2f} s  {t[:50]}", flush=True)
    np.savez(sys.argv[2], texts=np.asarray(texts), vecs=np.stack(vecs), seconds=np.asarray(secs))
    print(json.dumps({"n": len(texts), "median_s": float(np.median(secs))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
