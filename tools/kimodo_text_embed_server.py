"""Kimodo text-encoder service over a shared folder (Windows side of the WSL <-> Windows bridge).

Why: Kimodo's LLM2Vec text encoder (Llama-3-8B, bf16) needs ~16 GB. The laptop GPU has 12 GB (on Windows/WSL the
driver then spills into shared memory and stalls), WSL has 15 GB of RAM, and transformers 5.1 cannot load the PEFT
adapters onto a bitsandbytes-quantized model. Windows has 31.6 GB, so the encoder runs HERE on the CPU and Kimodo
(the motion diffusion model, small) runs in WSL on the GPU with ``TEXT_ENCODER_MODE=file``
(``third_party/kimodo/kimodo/model/text_encoder_file.py``). No networking, no firewall rule, no system config change.

Protocol (``--bridge``, default ``$DROPBEAR_HF_CACHE/embed_bridge``): a client writes ``requests/<id>.txt`` (the prompt,
atomically); this server writes ``responses/<id>.npy`` (float32, shape (1, 4096), atomically) and deletes the request.
``alive.json`` is rewritten every loop (pid, time, prompts served), so a client can tell the service is up.

usage (Windows):  .venv-embed/Scripts/python.exe tools/kimodo_text_embed_server.py
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


_W8 = None


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402


def _w8_linear_class(torch):
    """``nn.Linear`` replacement: int8 weights with one scale per output row; bf16 compute, activations not quantized.

    Dynamic activation quantization (torch ``quantize_dynamic``) drifted the embeddings to cosine 0.82 of bf16 on the
    60 library prompts (Llama activation outliers); this weight-only form stays at 0.9996 (2026-09-26)."""
    global _W8
    if _W8 is None:
        import torch.nn.functional as F

        class W8Linear(torch.nn.Module):
            def __init__(self, in_features: int, out_features: int, dtype=torch.bfloat16, device=None):
                super().__init__()
                self.in_features, self.out_features = in_features, out_features
                self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
                self.register_buffer("scale", torch.empty(out_features, 1, dtype=dtype, device=device))
                self.bias = None

            @classmethod
            def from_linear(cls, lin):
                m = cls(lin.in_features, lin.out_features, dtype=lin.weight.dtype)
                w = lin.weight.detach().float()
                scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
                m.qweight = torch.round(w / scale).to(torch.int8)
                m.scale = scale.to(lin.weight.dtype)
                m.bias = lin.bias
                return m

            def forward(self, x):  # x @ (diag(s) Q)^T = (x @ Q^T) * s: the integer weights are exact in bf16
                y = F.linear(x, self.qweight.to(x.dtype)) * self.scale.view(-1).to(x.dtype)
                return y if self.bias is None else y + self.bias

        _W8 = W8Linear
    return _W8


def to_weight_int8(base, torch, meta: bool = False) -> int:
    """In place: every ``nn.Linear`` under ``base`` -> ``W8Linear`` (``meta``: empty shells for a checkpoint load)."""
    import gc

    W8 = _w8_linear_class(torch)
    n = 0
    for parent in list(base.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                setattr(parent, name, W8(child.in_features, child.out_features, device="meta") if meta
                        else W8.from_linear(child))
                n += 1
    gc.collect()
    return n


def build_w8(out: Path, torch) -> None:
    """One-time: bf16 encoder -> merged LoRA -> int8 weights, saved with everything needed to rebuild it."""
    from llm2vec.llm2vec_wrapper import LLM2VecEncoder

    enc = LLM2VecEncoder(base_model_name_or_path=BASE, peft_model_name_or_path=PEFT, dtype="bfloat16", llm_dim=4096,
                         device="cpu")
    lm = enc.model
    lm.model = lm.model.merge_and_unload()
    n = to_weight_int8(lm.model, torch)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": lm.model.state_dict(), "name_or_path": lm.model.config._name_or_path,
                "llm2vec": {k: getattr(lm, k) for k in ("pooling_mode", "max_length", "doc_max_length",
                                                        "skip_instruction")},
                "base": BASE, "peft": PEFT, "linear_layers": n, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "note": "LLM2Vec Llama-3-8B supervised, LoRA merged, weight-only int8 (per-row scales), bf16 rest; "
                        "tools/kimodo_text_embed_server.py --w8"}, str(out))
    print(f"[embed] wrote {out} ({out.stat().st_size / 2**30:.2f} GB, {n} int8 layers)", flush=True)


def load_w8(path: Path, torch):
    """Encoder from a ``build_w8`` / ``tools/build_llm2vec_w8.py`` checkpoint without ever materialising the bf16
    weights. A ``.pt`` file loads into RAM (~9.9 GB committed); a ``--to_dir`` directory is memory-mapped read-only
    (file-backed: no pagefile commit for the weights)."""
    from llm2vec.llm2vec import LLM2Vec
    from llm2vec.llm2vec_wrapper import LLM2VecEncoder
    from transformers import AutoConfig, AutoTokenizer

    if Path(path).is_dir():  # read-only memory-mapped .npy per tensor (tools/build_llm2vec_w8.py --to_dir)
        import warnings

        import numpy as np

        meta = json.loads((Path(path) / "index.json").read_text(encoding="utf-8"))
        sd = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # "non-writable NumPy array": intended, never written
            for k, info in meta.pop("tensors").items():
                t = torch.from_numpy(np.load(Path(path) / f"{k}.npy", mmap_mode="r"))
                sd[k] = t.view(torch.bfloat16) if info["dtype"] == "bfloat16" else t
        ck = {**meta, "state_dict": sd}
    else:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    config = AutoConfig.from_pretrained(ck["base"])
    config._name_or_path = ck["name_or_path"]  # selects LLM2Vec's Llama-3 prompt template
    model_class = LLM2Vec._get_model_class(config.__class__.__name__, enable_bidirectional=True)
    with torch.device("meta"):
        model = model_class(config)
    to_weight_int8(model, torch, meta=True)
    model.load_state_dict(ck["state_dict"], strict=True, assign=True)
    model.config._name_or_path = ck["name_or_path"]
    model.rotary_emb = model.rotary_emb.__class__(config=model.config, device="cpu")  # non-persistent buffers
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"w8 load left meta tensors: {left[:5]}")
    tokenizer = AutoTokenizer.from_pretrained(ck["base"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    enc = LLM2VecEncoder.__new__(LLM2VecEncoder)
    enc.llm_dim, enc._device, enc._quantized = 4096, "cpu", False
    enc.model = LLM2Vec(model=model.eval(), tokenizer=tokenizer, **ck["llm2vec"])
    for p in enc.model.parameters():
        p.requires_grad = False
    return enc


def serve(enc, req: Path, resp: Path, bridge: Path, np) -> int:
    """Answer embedding requests forever (see the module doc for the protocol)."""
    served = 0
    cache: dict[str, object] = {}  # text -> embedding: a repeated prompt (a UI chip, "wave again") costs nothing
    while True:
        for f in sorted(req.glob("*.txt")):
            try:
                text = f.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            t1 = time.perf_counter()
            arr = cache.get(text)
            if arr is None:
                vec, _ = enc(text)  # (1, 4096) for a single string
                arr = np.asarray(vec.float().cpu().numpy(), dtype=np.float32).reshape(1, -1)
                if len(cache) < 4096:
                    cache[text] = arr
            tmp = resp / (f.stem + ".npy.part")
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, resp / (f.stem + ".npy"))
            f.unlink(missing_ok=True)
            served += 1
            print(f"[embed] {f.stem}: {text[:60]!r} in {1e3 * (time.perf_counter() - t1):.0f} ms", flush=True)
        (bridge / "alive.json").write_text(json.dumps({"pid": os.getpid(), "time": time.time(), "served": served}))
        time.sleep(0.1)



def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--bridge", type=Path, default=_paths.hf_cache() / "embed_bridge")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--low_priority", action="store_true",
                    help="BELOW_NORMAL process priority (Windows): a prompt's encode should not slow the real-time "
                    "physics process (tools/live_session.sh)")
    ap.add_argument("--w8", type=Path, default=None,
                    help="serve from a weight-only int8 checkpoint (tools/build_llm2vec_w8.py): a .pt (~9.9 GB "
                    "committed) or its --to_dir directory (memory-mapped read-only; recommended with Isaac + Kimodo)")
    ap.add_argument("--build_w8", type=Path, default=None, help="write the int8 checkpoint here and exit (needs ~24 GB)")
    ap.add_argument("--int8", action="store_true",
                    help="merge the supervised LoRA and store every Linear as int8 weights with per-row scales "
                    "(bf16 compute, activations not quantized) to cut the committed memory. Check the embedding "
                    "drift with tools/embed_quant_check.py / tools/embed_bridge_probe.py before relying on it")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_CACHE", str(_paths.hf_cache() / "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # import the self-contained llm2vec subpackage only (the kimodo package root needs its C++ extension)
    sys.path.insert(0, str(REPO / "third_party" / "kimodo" / "kimodo" / "model"))
    import numpy as np
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.low_priority and sys.platform == "win32":
        import ctypes

        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass(ctypes.c_void_p(k32.GetCurrentProcess()), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
    from llm2vec.llm2vec_wrapper import LLM2VecEncoder

    req, resp = args.bridge / "requests", args.bridge / "responses"
    req.mkdir(parents=True, exist_ok=True)
    resp.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if args.build_w8 is not None:
        build_w8(args.build_w8, torch)
        return 0
    if args.w8 is not None:
        enc = load_w8(args.w8, torch)
        print(f"[embed] encoder loaded on CPU in {time.perf_counter() - t0:.1f} s (int8 checkpoint {args.w8}); "
              f"bridge {args.bridge}", flush=True)
        return serve(enc, req, resp, args.bridge, np)
    enc = LLM2VecEncoder(base_model_name_or_path=BASE, peft_model_name_or_path=PEFT, dtype="bfloat16", llm_dim=4096,
                         device="cpu")
    if args.int8:
        _to_int8(enc, torch)
    print(f"[embed] encoder loaded on CPU in {time.perf_counter() - t0:.1f} s ({'int8' if args.int8 else 'bf16'}); "
          f"bridge {args.bridge}", flush=True)
    return serve(enc, req, resp, args.bridge, np)


if __name__ == "__main__":
    raise SystemExit(main())
