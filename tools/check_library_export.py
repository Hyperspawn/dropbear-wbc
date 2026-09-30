"""Offline checks of a REAL motion-library export (``scripts/play.py --export`` on a library task; CONTRACTS 5.3).

CPU only (system Python: torch + onnx + onnxruntime). Checks:
 1. layout: the sidecar parses (``deploy.config.load_sidecar``), ``motion.source == "runtime"``, ``policy.onnx`` has no
    ``time_step`` input, no ``policy_motion.onnx`` next to it, ``obs_dim`` == sum of the observation terms;
 2. graph parity on ``parity_samples.pt`` (64 random observations): ONNX and TorchScript vs the live policy;
 3. real-rollout parity on ``parity_rollout.pt`` (env 0 observations/actions of the play rollout): ONNX vs live;
 4. command parity Isaac -> deploy: the command slice (44 or 44 x (1 + K) values) of those REAL Isaac observations
    equals what the deploy runner builds from the clip NPZ fed at runtime (``tools/policy_runner.build --motion``) at
    the same frame -- the reference path a hardware/sim2sim runner uses (play env 0 = clip 0 from frame 0).

    python tools/check_library_export.py <run>/exported --motion data/motions/synthetic/stand.npz \
        --out logs/multiclip/check_library_export.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "source", ROOT / "third_party" / "pydeps", ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--motion", type=Path, default=None,
                    help="clip NPZ that play env 0 tracked (default: clip 0 of the export's library report)")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import onnxruntime as ort
    import torch

    import policy_runner
    from dropbear_wbc.deploy.config import load_sidecar
    from dropbear_wbc.deploy.observations import ObsContext, TERMS

    d = a.export_dir.resolve()
    side = json.loads((d / "policy.json").read_text(encoding="utf-8"))
    cfg = load_sidecar(d / "policy.json")
    rep: dict = {"export_dir": str(d), "checks": {}}
    chk = rep["checks"]
    sess = ort.InferenceSession(str(d / "policy.onnx"), providers=["CPUExecutionProvider"])
    inputs = [i.name for i in sess.get_inputs()]
    chk["layout"] = {"source": cfg.motion.source if cfg.motion else None, "onnx_inputs": inputs,
                     "no_policy_motion_onnx": not (d / "policy_motion.onnx").exists(),
                     "obs_dim": cfg.obs_dim, "sum_terms": sum(t.dim for t in cfg.observations),
                     "export_type": side.get("dropbear_tracking", {}).get("export_type"),
                     "future_steps": (cfg.observations[0].params or {}).get("future_steps", [])}
    lay_ok = (chk["layout"]["source"] == "runtime" and inputs == ["obs"] and chk["layout"]["no_policy_motion_onnx"]
              and cfg.obs_dim == chk["layout"]["sum_terms"])
    jit = torch.jit.load(str(d / "policy.pt"), map_location="cpu")

    def onnx_act(x: np.ndarray) -> np.ndarray:
        return np.concatenate([sess.run(["actions"], {"obs": x[i:i + 1].astype(np.float32)})[0] for i in range(len(x))])

    ps = torch.load(d / "parity_samples.pt", map_location="cpu")
    x, live = ps["obs"].numpy(), ps["actions"].numpy()
    with torch.inference_mode():
        jit_a = jit(torch.as_tensor(x)).numpy()
    scale = max(1.0, float(np.abs(live).max()))
    chk["parity_random"] = {"onnx_vs_live": float(np.abs(onnx_act(x) - live).max()),
                            "torchscript_vs_live": float(np.abs(jit_a - live).max()), "max_abs_action": float(np.abs(live).max())}
    par_ok = max(chk["parity_random"]["onnx_vs_live"], chk["parity_random"]["torchscript_vs_live"]) <= a.tol * scale
    roll_ok = True
    cmd_ok = True
    rp = d / "parity_rollout.pt"
    if rp.is_file():
        r = torch.load(rp, map_location="cpu")
        ro, ra = r["obs"].numpy(), r["actions"].numpy()
        chk["parity_rollout"] = {"samples": int(len(ro)), "onnx_vs_live": float(np.abs(onnx_act(ro) - ra).max())}
        roll_ok = chk["parity_rollout"]["onnx_vs_live"] <= a.tol * max(1.0, float(np.abs(ra).max()))
        motion = a.motion
        if motion is None:
            clips = side.get("dropbear_tracking", {}).get("library", {}).get("clips") or []
            motion = Path(clips[0]["npz"]) if clips else None
        if motion is not None:
            args = policy_runner.parse_args(["--mode", "policy", "--sidecar", str(d / "policy.json"),
                                             "--allow-privileged", "--motion", str(motion)])
            cfg2, ctrl = policy_runner.build(args)
            term = cfg2.observations[0]
            n = min(a.frames, len(ro))

            def cmd_at(frame: int) -> np.ndarray:
                ctx = ObsContext(state=None, cfg=cfg2, last_action=np.zeros(cfg2.num_actions))
                ctx.reference = ctrl.motion.sample(frame)
                ctx.reference_ahead = lambda j: ctrl.motion.sample(frame + j)
                return TERMS[term.func](ctx, term.params)

            # rollout step k is frame k, unless env 0 fell and was reset (the play config restarts at frame 0)
            errs, frames, frame = [], [], -1
            for k in range(n):
                # after a mid-rollout reset Isaac writes frame 0 and the same step's command update advances it
                # to frame 1 (ManagerBasedRLEnv.step: _reset_idx, then command_manager.compute), so both can follow
                cand = {frame + 1: None, 0: None, 1: None}
                for f in cand:
                    v = cmd_at(f)
                    cand[f] = float(np.abs(v - ro[k, :v.size]).max())
                frame = min(cand, key=cand.get)
                errs.append(cand[frame])
                frames.append(frame)
            chk["command_isaac_vs_deploy"] = {"motion": str(motion), "frames_checked": n, "max_abs_err": max(errs),
                                              "resets_seen": sum(1 for i, f in enumerate(frames) if i > 0 and f in (0, 1) and f != frames[i - 1] + 1),
                                              "per_step_max": [round(e, 7) for e in errs[:10]],
                                              "runtime_motion": cfg2.meta.get("runtime_motion", {}).get("checks")}
            cmd_ok = max(errs) <= 1e-4
    else:
        chk["parity_rollout"] = "missing (play.py writes it with --export)"
    rep["ok"] = bool(lay_ok and par_ok and roll_ok and cmd_ok)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
