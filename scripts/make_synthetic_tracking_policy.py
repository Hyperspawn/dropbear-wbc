"""Write a SYNTHETIC BeyondMimic-layout ONNX + ``dropbear-policy-sidecar-v1`` for runner plumbing tests.

This is not a trained policy. ``actions = obs @ W`` with tiny random weights, and
the embedded reference is a smooth +/-0.05 rad sinusoid around the legacy default
pose. The ONNX has BeyondMimic's I/O signature (inputs ``obs`` [1,125] and
``time_step`` [1,1]; outputs ``actions`` plus the reference ``joint_pos``,
``joint_vel``, ``body_pos_w``, ``body_quat_w``). The observation layout is the full
BeyondMimic policy group, including the sim-only ``motion_anchor_pos_b`` and
``base_lin_vel``. Use it to exercise the policy path of ``tools/policy_runner.py``
end to end against the Newton bridge. It says nothing about tracking quality.

Example::

    python scripts/make_synthetic_tracking_policy.py --out logs/sdk_bridge/synthetic_policy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "source")]

import deploy_fixtures as fx  # noqa: E402
from dropbear_wbc.sdk import motors  # noqa: E402


def smooth_reference(seconds: float, fps: float = 50.0) -> dict[str, np.ndarray]:
    t = np.arange(int(seconds * fps)) / fps
    default = {n: motors.DEFAULT_POS[i] for i, n in enumerate(motors.MOTOR_NAMES)}
    jp = np.zeros((len(t), len(fx.REF_JOINTS)))
    jv = np.zeros_like(jp)
    for c, name in enumerate(fx.REF_JOINTS):
        if name in default:
            phase = 0.3 * c
            jp[:, c] = default[name] + 0.05 * np.sin(2 * np.pi * 0.5 * t + phase)
            jv[:, c] = 0.05 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t + phase)
    bp = np.zeros((len(t), len(fx.BODIES), 3))
    bp[:, 0] = [0.0, 0.0, 0.17]           # root 'world'
    bp[:, 1] = [0.0, 0.0, 0.17 + 1.1]     # synthetic 'anchor_body' rigidly above the root
    bq = np.zeros((len(t), len(fx.BODIES), 4))
    bq[..., 0] = 1.0
    return {"joint_pos": jp, "joint_vel": jv, "body_pos_w": bp, "body_quat_w": bq}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "sdk_bridge" / "synthetic_policy")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--weight-scale", type=float, default=0.002)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tables = smooth_reference(args.seconds)
    obs_dim = sum(t["dim"] for t in fx.OBS_TERMS_FULL)
    fx.write_beyondmimic_onnx(args.out / "policy.onnx", obs_dim=obs_dim, weight_scale=args.weight_scale,
                              tables=tables)
    fx.write_sidecar(args.out / "policy.json", "policy.onnx", terms=fx.OBS_TERMS_FULL,
                     num_frames=tables["joint_pos"].shape[0], align="yaw_xy")
    print(f"wrote {args.out / 'policy.onnx'} (obs_dim={obs_dim}, frames={tables['joint_pos'].shape[0]}) "
          f"and {args.out / 'policy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
