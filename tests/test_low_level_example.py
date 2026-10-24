"""scripts/examples/dropbear_low_level_example.py safety deltas vs the Unitree original (item 266)."""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

from dropbear_wbc.robots.hw_motor_specs import MOTION_MODE_LIMITS
from dropbear_wbc.sdk.types import LowState, MotorStateBlock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ll_example", ROOT / "scripts/examples/dropbear_low_level_example.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


class _Pub:
    def __init__(self):
        self.sent = []

    def Write(self, cmd):
        self.sent.append((cmd.motor.q.astype(float).copy(), np.asarray(cmd.motor.kp, float).copy(),
                          np.asarray(cmd.motor.kd, float).copy()))


def _state(q):
    st = LowState(tick=7, stamp_ns=0)
    st.motor = MotorStateBlock.zeros(ex.NUM_MOTOR)
    st.motor.q[:] = np.asarray(q, np.float32)
    return st


def _custom(**kw):
    c = ex.Custom(1.0, **kw)
    c.lowcmd_publisher_ = _Pub()
    return c


def test_gains_fit_motion_mode_field():
    assert max(ex.Kd) <= MOTION_MODE_LIMITS["kd_max"]
    assert max(ex.Kp) <= MOTION_MODE_LIMITS["kp_max"]
    assert np.all(ex.PASSIVE_KD <= MOTION_MODE_LIMITS["kd_max"])


def test_no_state_writes_nothing():
    c = _custom(connect_timeout_s=0.05)
    with pytest.raises(ex.StateLost, match="no command was written"):
        c.Start()
    assert c.lowcmd_publisher_.sent == []
    assert c.Damp() is False and c.lowcmd_publisher_.sent == []


def test_stale_state_aborts_then_damps():
    c = _custom(state_timeout_s=0.05)
    q = np.full(ex.NUM_MOTOR, 0.4)
    c.LowStateHandler(_state(q))
    c.last_rx = time.perf_counter() - 1.0
    with pytest.raises(ex.StateLost, match="old"):
        c.Start()
    assert c.lowcmd_publisher_.sent == []
    assert c.Damp() is True
    q_cmd, kp, kd = c.lowcmd_publisher_.sent[-1]
    assert np.all(kp == 0.0) and np.allclose(kd, ex.PASSIVE_KD) and np.allclose(q_cmd, q, atol=1e-6)


def test_first_command_starts_at_measured_pose():
    c = _custom()
    q = np.asarray(ex.motors.DEFAULT_POS, float).copy()
    q[3] = 1.2
    c.LowStateHandler(_state(q))
    c.LowCmdWrite()
    q_cmd, kp, _ = c.lowcmd_publisher_.sent[0]
    assert abs(q_cmd[3] - 1.2) < 0.01 and kp.max() > 0
