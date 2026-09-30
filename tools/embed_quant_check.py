"""Offline fidelity check of the Kimodo text encoder variants (Windows CPU, .venv-embed), one model copy at a time.

Stages on the same prompts: bf16 as served -> PEFT merged -> weight-only int8 (per-output-channel, bf16 compute).
Prints cosine similarity of each stage to the bf16 reference and the committed memory after each step.

  .venv-embed/Scripts/python.exe tools/embed_quant_check.py data/motions_gen/prompts/everyday_v1.json --n 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
PEFT = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402


def commit_gb() -> float:
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    c = PMC()
    c.cb = ctypes.sizeof(c)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PagefileUsage / 2**30


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("prompts", type=Path)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", type=Path, default=REPO / "logs/live/embed_quant_check.json")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_CACHE", str(_paths.hf_cache() / "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(REPO / "third_party" / "kimodo" / "kimodo" / "model"))
    sys.path.insert(0, str(REPO / "tools"))
    import numpy as np
    import torch

    from kimodo_text_embed_server import to_weight_int8
    from llm2vec.llm2vec_wrapper import LLM2VecEncoder

    texts = [it["prompt"] for it in json.loads(args.prompts.read_text(encoding="utf-8"))][: args.n]
    t0 = time.perf_counter()
    enc = LLM2VecEncoder(base_model_name_or_path=BASE, peft_model_name_or_path=PEFT, dtype="bfloat16", llm_dim=4096,
                         device="cpu")
    rep = {"load_s": round(time.perf_counter() - t0, 1), "stages": {}}

    def run(stage: str) -> np.ndarray:
        t = time.perf_counter()
        v = np.stack([np.asarray(enc(x)[0].float().cpu().numpy()).reshape(-1) for x in texts])
        rep["stages"][stage] = {"s_per_prompt": round((time.perf_counter() - t) / len(texts), 2),
                                "commit_gb": round(commit_gb(), 2)}
        print(stage, rep["stages"][stage], flush=True)
        return v

    ref = run("bf16")
    lm = enc.model
    lm.model = lm.model.merge_and_unload()
    merged = run("bf16_merged")
    to_weight_int8(lm.model, torch)
    w8 = run("w8_merged")

    def cos(a, b):
        return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)

    for name, v in (("bf16_merged", merged), ("w8_merged", w8)):
        c = cos(ref.astype(np.float64), v.astype(np.float64))
        rep["stages"][name]["cos_to_bf16"] = {"min": round(float(c.min()), 5), "median": round(float(np.median(c)), 5)}
    rn = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    iu = np.triu_indices(len(ref), 1)
    rep["closest_distinct_prompt_distance"] = round(float((1 - rn @ rn.T)[iu].min()), 5)
    args.out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
