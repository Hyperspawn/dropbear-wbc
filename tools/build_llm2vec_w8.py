"""Build the weight-only int8 Kimodo text encoder checkpoint by streaming the safetensors (peak ~9 GB, not ~24 GB).

Llama-3-8B-Instruct (bf16) + LLM2Vec MNTP LoRA + LLM2Vec supervised LoRA, merged in fp32
(``W + scale * B @ A`` per adapter, ``scale = lora_alpha / r``), then every projection is stored as int8 with one
scale per output row; embeddings and norms stay bf16. The result loads with
``tools/kimodo_text_embed_server.py --w8 <file>`` (``load_w8``), which never materialises the bf16 model.

Why not ``--build_w8`` (transformers + PEFT merge in process): on Windows transformers maps the safetensors
copy-on-write, which charges the full 16 GB against the commit limit for as long as the model lives; with the int8
copy on top that was 24 GB, and the laptop's pagefile sits on a nearly full C: (2026-09-26).

  .venv-embed/Scripts/python.exe tools/build_llm2vec_w8.py $DROPBEAR_HF_CACHE/llm2vec_llama3_8b_sup_w8.pt
  .venv-embed/Scripts/python.exe tools/build_llm2vec_w8.py --to_dir $DROPBEAR_HF_CACHE/llm2vec_llama3_8b_sup_w8 \
      $DROPBEAR_HF_CACHE/llm2vec_llama3_8b_sup_w8.pt   # per-tensor .npy for read-only memory mapping

The ``--to_dir`` form is what the server should use (``--w8 <dir>``): read-only file mappings are backed by the files
on H:, so Windows charges no pagefile commit for the 7.5 GB of weights (the .pt, loaded into RAM, commits ~9.9 GB).
bf16 tensors are stored as uint16 and viewed back as bfloat16.
"""
from __future__ import annotations

import sys

import argparse
import glob
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
HUB = _paths.hf_cache() / "hub"
BASE = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
PEFT = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
LLAMA = "meta-llama/Meta-Llama-3-8B-Instruct"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def snapshot(repo: str) -> Path:
    return Path(glob.glob(str(HUB / f"models--{repo.replace('/', '--')}" / "snapshots" / "*"))[0])


def load_adapter(repo: str, torch):
    from safetensors.torch import load_file

    d = snapshot(repo)
    cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("use_rslora") or cfg.get("fan_in_fan_out") or cfg.get("bias", "none") != "none":
        raise NotImplementedError(f"{repo}: unsupported adapter options")
    sd = load_file(str(d / "adapter_model.safetensors"))
    pairs = {}
    for k, v in sd.items():  # base_model.model.layers.0.mlp.down_proj.lora_A.weight
        mod, ab = k.removeprefix("base_model.model.").rsplit(".lora_", 1)
        pairs.setdefault(mod, {})[ab.split(".")[0]] = v.float()
    return float(cfg["lora_alpha"]) / float(cfg["r"]), pairs


def to_dir(ckpt: Path, out: Path, torch) -> int:
    import numpy as np

    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for k, t in ck.pop("state_dict").items():
        if t.dtype == torch.bfloat16:
            arr, dt = t.contiguous().view(torch.uint16).numpy(), "bfloat16"
        else:
            arr, dt = t.contiguous().numpy(), str(t.dtype).removeprefix("torch.")
        np.save(out / f"{k}.npy", arr)
        index[k] = {"dtype": dt, "shape": list(t.shape)}
    (out / "index.json").write_text(json.dumps({**{k: v for k, v in ck.items()}, "tensors": index}, indent=1),
                                    encoding="utf-8")
    print(f"[w8] {len(index)} tensors -> {out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("out", type=Path, help="checkpoint to write (or, with --to_dir, the checkpoint to convert)")
    ap.add_argument("--to_dir", type=Path, default=None, help="convert an existing checkpoint to a .npy directory")
    ap.add_argument("--name_or_path", default=LLAMA,
                    help="config._name_or_path recorded for LLM2Vec's prompt template (Llama-3 chat template)")
    args = ap.parse_args()
    import torch
    from safetensors import safe_open

    if args.to_dir is not None:
        return to_dir(args.out, args.to_dir, torch)
    t0 = time.perf_counter()
    adapters = [load_adapter(BASE, torch), load_adapter(PEFT, torch)]
    out, merged, n_q = {}, 0, 0
    for shard in sorted(snapshot(LLAMA).glob("model-*.safetensors")):
        with safe_open(str(shard), "pt") as fh:
            for k in fh.keys():
                if k == "lm_head.weight":
                    continue
                name = k.removeprefix("model.")
                w = fh.get_tensor(k)
                mod = name.removesuffix(".weight")
                if mod.rsplit(".", 1)[-1] in PROJ:
                    wf = w.float()
                    for scale, pairs in adapters:
                        if mod in pairs:
                            wf += scale * (pairs[mod]["B"] @ pairs[mod]["A"])
                            merged += 1
                    s = (wf.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
                    out[f"{mod}.qweight"] = torch.round(wf / s).to(torch.int8)
                    out[f"{mod}.scale"] = s.to(torch.bfloat16)
                    n_q += 1
                    del wf
                else:
                    out[name] = w.clone()
                del w
        print(f"[w8] {shard.name}: {n_q} layers quantized, {merged} LoRA merges ({time.perf_counter() - t0:.0f} s)",
              flush=True)
    expect = 32 * len(PROJ)
    if n_q != expect or merged != 2 * expect:
        raise RuntimeError(f"quantized {n_q} (expected {expect}), merged {merged} (expected {2 * expect})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out, "name_or_path": args.name_or_path,
                "llm2vec": {"pooling_mode": "mean", "max_length": 512, "doc_max_length": 400,
                            "skip_instruction": True},
                "base": BASE, "peft": PEFT, "linear_layers": n_q, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "note": "streamed build (tools/build_llm2vec_w8.py): Llama-3-8B-Instruct + MNTP + supervised LoRA "
                        "merged in fp32, weight-only int8 per-row, bf16 embeddings/norms"}, str(args.out))
    print(f"[w8] wrote {args.out} ({args.out.stat().st_size / 2**30:.2f} GB) in {time.perf_counter() - t0:.0f} s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
