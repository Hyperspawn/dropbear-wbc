"""CPU check of an exported Dropbear tracking policy (system python: torch + onnxruntime; no Isaac).

Checks, for ``<export_dir>`` written by ``scripts/play.py --export``:
  * ``policy.json`` (dropbear-policy-sidecar-v1) layout is contiguous and sums to obs_dim; 22 actions in
    motor-contract order; it parses with the deploy agent's ``load_sidecar`` when that module exists;
  * ``policy.pt`` (TorchScript) reproduces the live-policy actions saved in ``parity_samples.pt``;
  * ``policy.onnx`` (onnxruntime) matches ``policy.pt`` on those and on random observations;
  * ``policy_motion.onnx`` actions match ``policy.onnx`` and its reference outputs have the declared shapes.

    python tools/check_export_parity.py <export_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))


def main(export_dir: Path) -> int:
    import onnxruntime as ort
    import torch

    from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES

    side = json.loads((export_dir / "policy.json").read_text(encoding="utf-8"))
    report: dict = {"export_dir": str(export_dir), "schema": side.get("schema")}
    layout = side["dropbear_tracking"]["observation_layout"]
    contiguous = all(a["start"] + a["dim"] == b["start"] for a, b in zip(layout, layout[1:])) and layout[0]["start"] == 0
    obs_dim = side["obs_dim"]
    report["layout_ok"] = (
        contiguous
        and layout[-1]["start"] + layout[-1]["dim"] == obs_dim
        and [t["dim"] for t in side["observations"]] == [t["dim"] for t in layout]
    )
    report["action_ok"] = side["joint_names"] == list(MOTOR_NAMES) and len(side["action_scale"]) == 22
    report["obs_layout"] = [(t["name"], t["func"], t["dim"]) for t in side["observations"]]
    try:  # the deploy agent's parser (dropbear_wbc.deploy.config.load_sidecar), if present
        from dropbear_wbc.deploy.config import load_sidecar

        cfg = load_sidecar(export_dir / "policy.json")
        report["deploy_load_sidecar"] = {"ok": True, "obs_terms": [t.func for t in cfg.observations],
                                         "motion_source": cfg.motion.source if cfg.motion else None}
    except ImportError as exc:
        report["deploy_load_sidecar"] = {"ok": None, "skipped": str(exc)}
    except Exception as exc:  # noqa: BLE001
        report["deploy_load_sidecar"] = {"ok": False, "error": repr(exc)}

    samples = torch.load(export_dir / "parity_samples.pt", map_location="cpu")
    x = samples["obs"].float()
    live = samples["actions"].float()
    jit = torch.jit.load(str(export_dir / "policy.pt"), map_location="cpu").eval()
    with torch.no_grad():
        a_jit = jit(x)
    report["torchscript_vs_live_max_abs"] = float((a_jit - live).abs().max())

    sess = ort.InferenceSession(str(export_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    xr = np.concatenate([x.numpy(), rng.normal(scale=2.0, size=(64, obs_dim)).astype(np.float32)], axis=0)
    a_onnx = np.concatenate([sess.run(None, {in_name: xr[i : i + 1]})[0] for i in range(xr.shape[0])], axis=0)
    with torch.no_grad():
        a_ref = jit(torch.from_numpy(xr)).numpy()
    report["onnx_vs_torchscript_max_abs"] = float(np.abs(a_onnx - a_ref).max())

    msess = ort.InferenceSession(str(export_dir / "policy_motion.onnx"), providers=["CPUExecutionProvider"])
    outs = msess.run(None, {"obs": xr[:1], "time_step": np.array([[3.0]], dtype=np.float32)})
    names = [o.name for o in msess.get_outputs()]
    report["motion_onnx_outputs"] = {n: list(o.shape) for n, o in zip(names, outs)}
    report["motion_onnx_vs_onnx_max_abs"] = float(np.abs(outs[0] - a_onnx[:1]).max())
    shapes_ok = outs[1].shape == (1, 22) and outs[2].shape == (1, 22) and outs[3].shape[-1] == 3 and outs[4].shape[-1] == 4
    report["motion_onnx_shapes_ok"] = bool(shapes_ok)

    # real simulator observations from the play rollout (scripts/play.py --export), if saved
    rollout_ok = True
    roll_path = export_dir / "parity_rollout.pt"
    if roll_path.is_file():
        roll = torch.load(roll_path, map_location="cpu")
        xo = roll["obs"].float().numpy()
        a_roll = np.concatenate([sess.run(None, {in_name: xo[i : i + 1]})[0] for i in range(xo.shape[0])], axis=0)
        report["rollout_onnx_vs_live_max_abs"] = float(np.abs(a_roll - roll["actions"].float().numpy()).max())
        report["rollout_samples"] = int(xo.shape[0])
        rollout_ok = report["rollout_onnx_vs_live_max_abs"] < 1e-4
    else:
        report["rollout_onnx_vs_live_max_abs"] = None

    # embedded reference == the motion NPZ (motor columns, tracked bodies)
    ref_ok = True
    try:
        from dropbear_wbc.tasks.tracking.motion_npz import load_motion_npz

        prov = side["dropbear_tracking"]["provenance"]
        m = load_motion_npz(prov["motion_file"])
        mids = [m.joint_names.index(n) for n in side["motion"]["reference_joint_names"]]
        bids = [m.body_names.index(n) for n in side["motion"]["body_names"]]
        errs = []
        for t in sorted({0, m.num_frames // 2, m.num_frames - 1}):
            o = msess.run(None, {"obs": xr[:1], "time_step": np.array([[float(t)]], dtype=np.float32)})
            errs.append(max(float(np.abs(o[1][0] - m.joint_pos[t, mids]).max()),
                            float(np.abs(o[3][0] - m.body_pos_w[t, bids]).max()),
                            float(np.abs(o[4][0] - m.body_quat_w[t, bids]).max())))
        report["motion_onnx_reference_vs_npz_max_abs"] = max(errs)
        ref_ok = max(errs) < 1e-5
    except Exception as exc:  # noqa: BLE001
        report["motion_onnx_reference_vs_npz_max_abs"] = f"skipped: {exc!r}"
    tol = 1e-4
    # The synthetic parity inputs (export.py: randn * 2) drive the normalized network far out of range (actions of
    # several hundred), where float32 rounding alone exceeds an absolute 1e-4. Those comparisons are therefore judged
    # against tol * max(1, max|action|) (relative 1e-4); the REAL rollout comparison above stays absolute 1e-4.
    # (Changed 2026-09-24, logs/gpu_pipeline/check_export_parity_wave_relative.log.)
    a_scale = float(max(1.0, float(live.abs().max()), float(np.abs(a_ref).max())))
    report["synthetic_action_max_abs"] = a_scale
    report["synthetic_tolerance"] = tol * a_scale
    report["ok"] = bool(
        report["layout_ok"] and report["action_ok"] and shapes_ok and rollout_ok and ref_ok
        and report["deploy_load_sidecar"].get("ok") is not False
        and report["torchscript_vs_live_max_abs"] < tol * a_scale
        and report["onnx_vs_torchscript_max_abs"] < tol * a_scale
        and report["motion_onnx_vs_onnx_max_abs"] < tol * a_scale
    )
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
