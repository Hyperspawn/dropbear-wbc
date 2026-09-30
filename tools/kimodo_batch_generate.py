"""Generate many G1 motions from text with Kimodo, loading the model once (runs in the WSL venv ``.venv-kimodo-wsl``).

``kimodo_gen`` reloads the model for every call (~130 s here). This keeps one model in memory and writes one G1 qpos
CSV per prompt (the ``kimodo_g1`` source format of ``tools/retarget_g1.py``), plus ``index.json`` (name, prompt,
duration, seed, seconds taken). The text encoder is whatever ``TEXT_ENCODER_MODE`` selects; on this laptop ``file``
(``tools/kimodo_text_embed_server.py`` on Windows).

prompts JSON: ``[{"name": "walk_slow", "prompt": "a person walks forward slowly", "duration": 5.0}, ...]``

usage (WSL):
  HF_HOME=$DROPBEAR_HF_CACHE HF_HUB_OFFLINE=1 TEXT_ENCODER_MODE=file \\
    .venv-kimodo-wsl/bin/python tools/kimodo_batch_generate.py prompts.json out_dir
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("prompts", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--model", default="Kimodo-G1-RP-v1")
    ap.add_argument("--steps", type=int, default=100, help="diffusion steps")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.model.load_model import load_model
    from kimodo.tools import seed_everything

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    model, resolved = load_model(args.model, device=device, default_family="Kimodo", return_resolved_name=True)
    load_s = time.perf_counter() - t0
    print(f"[kimodo] {resolved} loaded in {load_s:.1f} s on {device}", flush=True)
    conv = MujocoQposConverter(model.skeleton)
    items = json.loads(args.prompts.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = {"model": resolved, "load_s": round(load_s, 1), "steps": args.steps, "clips": []}
    for k, it in enumerate(items):
        seed = int(it.get("seed", args.seed + k))
        seed_everything(seed)
        t1 = time.perf_counter()
        out = model([it["prompt"]], [int(float(it.get("duration", 5.0)) * model.fps)], constraint_lst=[],
                    num_denoising_steps=args.steps, num_samples=1, multi_prompt=True, num_transition_frames=5,
                    post_processing=False, return_numpy=True)
        qpos = conv.dict_to_qpos(out, device)
        csv = args.out_dir / f"{it['name']}.csv"
        conv.save_csv(qpos, str(csv))
        dt = time.perf_counter() - t1
        index["clips"].append({"name": it["name"], "prompt": it["prompt"], "duration": it.get("duration", 5.0),
                               "seed": seed, "csv": csv.name, "seconds": round(dt, 2)})
        print(f"[kimodo] {k + 1}/{len(items)} {it['name']}: {dt:.1f} s", flush=True)
        (args.out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
