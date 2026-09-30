"""Reference-dynamics quality gate for settled motions (numpy only; review fix 2026-09-24).

A retarget can flip a serial3 (hip/shoulder) solution between frames; the settle then faithfully reproduces a motor
jump of up to pi in one 20 ms frame, i.e. tens of rad/s in ``joint_vel``. Those values enter the tracking command
observation and the RSI joint_vel writes, and no motor can follow them (the USD motor cap is 10 rad/s,
``physxJoint:maxJointVelocity`` 572.96 deg/s, docs/CONTRACTS.md 0.1). Used by ``tools/settle_motion.py`` (meta.status)
and ``tools/validate_motion_npz.py`` (verdict).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MAX_MOTOR_VEL_RAD_S: float = 10.0
"""USD motor ``maxJointVelocity`` (572.96 deg/s). The legacy config's looser 20 rad/s arm limit is not used."""
MAX_MOTOR_STEP_RAD: float = 0.3
"""Largest plausible motor change between two 50 Hz reference frames [rad]."""


def motor_dynamics(joint_pos: np.ndarray, joint_vel: np.ndarray, joint_names: Sequence[str],
                   motor_names: Sequence[str], max_vel: float = MAX_MOTOR_VEL_RAD_S,
                   max_step: float = MAX_MOTOR_STEP_RAD) -> dict:
    """Motor-column speed / frame-step statistics and the gate problems (empty list = pass)."""
    names = list(joint_names)
    idx = [names.index(n) for n in motor_names]
    q = np.asarray(joint_pos, dtype=np.float64)[:, idx]
    v = np.abs(np.asarray(joint_vel, dtype=np.float64)[:, idx])
    step = np.abs(np.diff(q, axis=0))
    per_v = v.max(axis=0)
    per_s = step.max(axis=0) if step.size else np.zeros(len(idx))
    out = {
        "max_abs_vel_rad_s": float(per_v.max()), "max_abs_vel_joint": motor_names[int(per_v.argmax())],
        "frames_over_vel_limit": int((v > max_vel).any(axis=1).sum()),
        "max_step_rad": float(per_s.max()), "max_step_joint": motor_names[int(per_s.argmax())],
        "max_step_frame": int(step.max(axis=1).argmax()) if step.size else 0,
        "steps_over_limit": int((step > max_step).any(axis=1).sum()) if step.size else 0,
        "worst_vel": [{"joint": motor_names[int(i)], "max_abs_vel_rad_s": round(float(per_v[i]), 2)}
                      for i in np.argsort(-per_v)[:3]],
        "limits": {"max_motor_vel_rad_s": max_vel, "max_motor_step_rad": max_step},
    }
    problems = []
    if out["max_abs_vel_rad_s"] > max_vel:
        problems.append(f"motor reference speed up to {out['max_abs_vel_rad_s']:.1f} rad/s ({out['max_abs_vel_joint']}) "
                        f"> {max_vel:.0f} rad/s motor cap on {out['frames_over_vel_limit']} frames")
    if out["max_step_rad"] > max_step:
        problems.append(f"motor reference jumps {out['max_step_rad']:.2f} rad in one frame ({out['max_step_joint']}, "
                        f"frame {out['max_step_frame']}) > {max_step} rad on {out['steps_over_limit']} frames "
                        "(retarget branch flip?)")
    out["problems"] = problems
    return out
