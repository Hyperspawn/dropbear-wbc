"""CPU tests for the reference-dynamics gate (dropbear_wbc.settle.quality; review fix 2026-09-24)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES  # noqa: E402
from dropbear_wbc.settle.quality import motor_dynamics  # noqa: E402


def _clip(t=100, flip_at=None, flip=3.0):
    names = list(MOTOR_NAMES) + ["LL_Revolute115:1"]
    q = np.zeros((t, len(names)))
    q[:, 0] = 0.2 * np.sin(np.linspace(0, 2 * np.pi, t))
    q[:, -1] = np.linspace(0, 40, t)  # a passive idle rod spin must NOT count
    if flip_at is not None:
        q[flip_at:, MOTOR_NAMES.index("RH_yaw")] += flip
    v = np.gradient(q, 0.02, axis=0)
    return q, v, names


def test_smooth_clip_passes():
    q, v, names = _clip()
    out = motor_dynamics(q, v, names, list(MOTOR_NAMES))
    assert out["problems"] == [] and out["max_abs_vel_rad_s"] < 10


def test_branch_flip_rejected():
    q, v, names = _clip(flip_at=50)
    out = motor_dynamics(q, v, names, list(MOTOR_NAMES))
    assert out["max_step_joint"] == "RH_yaw" and out["max_step_frame"] == 49
    assert any("jumps" in p for p in out["problems"]) and any("rad/s" in p for p in out["problems"])


@pytest.mark.parametrize("clip,ok", [("data/motions/synthetic/wave_right.npz", True),
                                     ("data/motions/kimodo_g1/output_wave.npz", False),
                                     ("data/motions/unitree_rl_lab_mimic/G1_Take_102.npz", False)])
def test_real_clips(clip, ok):
    from dropbear_wbc.tasks.tracking.motion_npz import load_motion_npz

    p = REPO / clip
    if not p.is_file():
        pytest.skip("clip not present")
    m = load_motion_npz(p)
    out = motor_dynamics(m.joint_pos, m.joint_vel, m.joint_names, m.motor_names)
    assert (out["problems"] == []) == ok
