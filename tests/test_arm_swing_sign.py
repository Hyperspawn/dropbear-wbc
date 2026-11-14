"""Arm-swing reward sign gate (``mdp.arm_swing_counter_phase``). CPU, Isaac-free.

The reward works in motor space with hand-written signs (``sh_r_motor - k*S*hip_l``, ``-sh_l_motor - k*S*hip_r``).
Whether that is counter-phase (left leg forward -> right arm forward) depends on the per-motor signs in the semantic
calibration: left shoulder pitch motor ``LH_yaw`` is mirrored (scale -1), right ``RH_yaw`` is not (+1), both hip pitch
motors are +0.984. A re-measured calibration or a re-authored USD that flips any of them silently turns the reward into
an in-phase (marching-soldier) or one-sided rule. These tests tie the reward's constants and its function body to the
calibration JSON, and the semantic convention (+pitch = limb backward for hips and shoulders) to the serial model.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
REWARDS = ROOT / "source/dropbear_wbc/tasks/locomotion/mdp/rewards.py"
CALIB = ROOT / "data/calibration/dropbear_semantic_calibration.json"
SERIAL_XML = ROOT / "data/robot/dropbear_serial_hw_v1.xml"

SEM = {"LL_hip_joint": "left_hip_pitch", "RL_hip_joint": "right_hip_pitch",
       "LH_yaw": "left_shoulder_pitch", "RH_yaw": "right_shoulder_pitch"}


def _module_consts() -> dict:
    tree = ast.parse(REWARDS.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in ("ARM_SWING_JOINTS", "HIP_PITCH_SCALE"):
                out[node.target.id] = ast.literal_eval(node.value)
    return out


def _reward_fn():
    """``arm_swing_counter_phase`` compiled from its own source with Isaac-free stand-ins."""
    torch = pytest.importorskip("torch")
    tree = ast.parse(REWARDS.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "arm_swing_counter_phase")
    fn.args.args[0].annotation = fn.args.args[2].annotation = None
    fn.returns = None
    mod = ast.Module(body=[fn], type_ignores=[])
    consts = _module_consts()
    ns = {"torch": torch, "as_tensor": lambda x: x, "_moving_command": lambda env, name: torch.ones(env.num_envs),
          **consts}
    exec(compile(ast.fix_missing_locations(mod), str(REWARDS), "exec"), ns)  # noqa: S102 (own repo source)
    return ns["arm_swing_counter_phase"], torch


def _scales() -> dict[str, float]:
    dofs = json.loads(CALIB.read_text())["dofs"]
    out = {}
    for motor, sem in SEM.items():
        d = dofs[sem]
        assert d["motors"] == [motor], (sem, d["motors"])
        out[motor] = float(d["scale"])
    return out


def test_constants_match_calibration():
    c = _module_consts()
    assert c["ARM_SWING_JOINTS"] == ("LL_hip_joint", "RL_hip_joint", "LH_yaw", "RH_yaw")
    s = _scales()
    assert math.isclose(c["HIP_PITCH_SCALE"], s["LL_hip_joint"], abs_tol=2e-3)
    assert math.isclose(c["HIP_PITCH_SCALE"], s["RL_hip_joint"], abs_tol=2e-3)
    # the reward's hard-coded mirror: left shoulder motor negated, right not
    assert s["LH_yaw"] == pytest.approx(-1.0, abs=1e-3) and s["RH_yaw"] == pytest.approx(1.0, abs=1e-3)


class _Env:
    def __init__(self, dq_motor):
        self.num_envs = dq_motor.shape[0]
        data = type("D", (), {})()
        data.joint_pos, data.default_joint_pos = dq_motor, dq_motor * 0.0
        self.scene = {"robot": type("A", (), {"data": data})()}


def _gait(phase_sign: float, k: float, amp: float = 0.35, n: int = 64):
    """Motor-space joint deltas for a semantic gait: hips in anti-phase, each shoulder = phase_sign*k*opposite hip."""
    torch = pytest.importorskip("torch")
    s = _scales()
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    hip_l, hip_r = amp * np.sin(t), -amp * np.sin(t)
    sh_l, sh_r = phase_sign * k * hip_r, phase_sign * k * hip_l
    sem = {"LL_hip_joint": hip_l, "RL_hip_joint": hip_r, "LH_yaw": sh_l, "RH_yaw": sh_r}
    motor = np.stack([sem[m] / s[m] for m in ("LL_hip_joint", "RL_hip_joint", "LH_yaw", "RH_yaw")], axis=1)
    return torch.tensor(motor, dtype=torch.float32)


def test_reward_peaks_on_counter_phase_semantic_gait():
    f, _ = _reward_fn()
    cfg = type("C", (), {"name": "robot", "joint_ids": [0, 1, 2, 3]})()
    for k in (0.40, 0.45, 0.55):
        good = f(_Env(_gait(+1.0, k)), "base_velocity", cfg, k=k, std=0.12)
        bad = f(_Env(_gait(-1.0, k)), "base_velocity", cfg, k=k, std=0.12)
        frozen = f(_Env(_gait(0.0, k)), "base_velocity", cfg, k=k, std=0.12)
        assert float(good.min()) > 0.999, k
        assert float(bad.mean()) < float(frozen.mean()) < 0.5 < float(good.mean()), k


def test_semantic_plus_pitch_is_backward_for_hips_and_shoulders():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(str(SERIAL_XML))
    d = mujoco.MjData(m)
    tips = {"left_hip_pitch": "left_ankle_roll_link", "right_hip_pitch": "right_ankle_roll_link",
            "left_shoulder_pitch": "left_wrist_roll_link", "right_shoulder_pitch": "right_wrist_roll_link"}
    for joint, tip in tips.items():
        a = m.jnt_qposadr[m.joint(joint).id]
        mujoco.mj_forward(m, d)
        x0 = float(d.body(tip).xpos[0])
        d.qpos[a] += 0.1
        mujoco.mj_forward(m, d)
        dx = float(d.body(tip).xpos[0]) - x0
        d.qpos[a] -= 0.1
        assert dx < -0.02, (joint, dx)  # +pitch about +y swings the limb backward (-x, pelvis x forward)
