"""CPU-only checks that the Isaac (torch) and Newton (numpy) sensor-age defaults agree and fit the history depth.

notes/39 (item 39): ages are per-boot phase constants; IMU 7-34 ms, joint 0-10 ms, action 0-15 ms at 50 Hz / 5 ms.
sensor_dr.py imports torch + isaaclab, so it is read with ast instead of imported.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import sdk_test_paths  # noqa: F401
from dropbear_wbc.newton_sim.sensor_model import PROFILES, FracDelay, SensorErrorCfg

ROOT = Path(__file__).resolve().parents[1] / "source" / "dropbear_wbc"


def _sensor_dr_ast():
    return ast.parse((ROOT / "tasks/tracking/mdp/sensor_dr.py").read_text())


def _defaults(fn: ast.FunctionDef) -> dict:
    a = fn.args
    pos = a.args[len(a.args) - len(a.defaults):]
    return {p.arg: ast.literal_eval(d) for p, d in zip(pos, a.defaults) if isinstance(d, (ast.Tuple, ast.Constant))}


def test_isaac_and_newton_age_defaults_match():
    tree = _sensor_dr_ast()
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "reset_imu_errors")
    d = _defaults(fn)
    c = SensorErrorCfg()
    assert tuple(d["imu_age_steps"]) == tuple(c.imu_age_ticks)
    assert tuple(d["joint_age_steps"]) == tuple(c.joint_age_ticks)


def test_age_defaults_fit_history_and_cover_chain_model():
    tree = _sensor_dr_ast()
    max_age = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", "") == "MAX_AGE_STEPS" for t in n.targets))
    c = SensorErrorCfg()
    for lo, hi in (c.imu_age_ticks, c.joint_age_ticks):
        assert 0.0 <= lo <= hi <= max_age
    step_ms = 20.0
    assert c.imu_age_ticks[0] * step_ms <= 8.0 and c.imu_age_ticks[1] * step_ms >= 33.0   # notes/39 A/B min-max
    assert c.joint_age_ticks[1] * step_ms >= 8.1                                         # notes/39 A max


def test_frac_delay_beyond_one_step():
    d = FracDelay(1.7)
    out = [d.push(np.array([float(i)]), 1.7)[0] for i in range(6)]
    assert np.isclose(out[-1], 5 - 1.7)


def test_torch_fracdelay_raises_instead_of_clamping():
    src = (ROOT / "tasks/tracking/mdp/sensor_dr.py").read_text()
    i = src.index("class _FracDelay")
    body = src[i:src.index("class measured_anchor_ori_b")]
    assert "raise ValueError" in body and body.index("raise ValueError") < body.index("lag.clamp")


def test_legacy_profile_keeps_old_ranges():
    assert PROFILES["deployed_054"].imu_age_ticks == (0.0, 0.75)
    assert PROFILES["center_0010"].gyro_lsb > PROFILES["motion_mode"].gyro_lsb
