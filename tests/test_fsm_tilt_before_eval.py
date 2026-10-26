"""Item 275: the torso tilt check (``fsm.bad_orientation_rad``) must act before the policy is evaluated.

Before, ``request(POLICY)`` did not check tilt, and ``step()`` evaluated the policy first and tripped to Passive only
afterwards, so the trip step still sent the policy target at policy kp (upside-down entry and every mid-run fall).
Now the entry is refused (``hold->policy refused(bad_orientation)``) and a fall in POLICY sends Passive damping on the
same step, without evaluating the policy. CPU only.
"""
from __future__ import annotations

import numpy as np
import pytest

import deploy_fixtures as fx
import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy.config import load_sidecar
from dropbear_wbc.deploy.fsm import DeployController, FsmState
from dropbear_wbc.sdk import motors

DT = 0.02


def _quat_x(deg):
    a = np.radians(deg)
    return (float(np.cos(a / 2)), float(np.sin(a / 2)), 0.0, 0.0)


def _st(tick, deg=0.0):
    s = fx.low_state(tick, np.asarray(motors.DEFAULT_POS), quat_wxyz=_quat_x(deg))
    s.motor.mode[:] = np.uint8(1)
    return s


@pytest.fixture
def ctrl_factory(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from dropbear_wbc.deploy.motion import OnnxMotion
    from dropbear_wbc.deploy.policy import OnnxPolicy

    fx.write_beyondmimic_onnx(tmp_path / "policy.onnx")
    fx.write_sidecar(tmp_path / "policy.json", "policy.onnx")
    policy = OnnxPolicy(tmp_path / "policy.onnx")

    def make(bad_orientation_rad=1.0):
        cfg = load_sidecar(tmp_path / "policy.json", policy.metadata)
        cfg.fsm.bad_orientation_rad = bad_orientation_rad
        motion = OnnxMotion(cfg.motion, cfg.joint_names, cfg.step_dt, policy.reference, list(fx.REF_JOINTS),
                            list(fx.BODIES), fx.T_FRAMES)
        ctrl = DeployController(cfg, policy, motion)
        ctrl.state = FsmState.HOLD
        return ctrl, cfg

    return make


@pytest.mark.parametrize("deg", [60.0, 90.0, 180.0])
def test_policy_entry_refused_when_tilted(ctrl_factory, deg):
    ctrl, cfg = ctrl_factory()
    st = _st(5, deg)
    assert not ctrl.request(FsmState.POLICY, st, 0.1)
    cmd, info = ctrl.step(st, 0.1)
    assert ctrl.state == FsmState.HOLD and info.transitions == ["hold->policy refused(bad_orientation)"]
    assert info.action is None and ctrl._policy_step == 0  # policy never evaluated
    np.testing.assert_allclose(cmd.motor.kp, cfg.fsm.hold_kp.astype(np.float32))
    # Retried every step (runner behaviour): the refusal is reported once per distinct fault.
    assert not ctrl.request(FsmState.POLICY, _st(6, deg), 0.12)
    _, info = ctrl.step(_st(6, deg), 0.12)
    assert info.transitions == []


def test_policy_entry_allowed_below_limit_and_when_disabled(ctrl_factory):
    ctrl, _ = ctrl_factory()
    assert ctrl.request(FsmState.POLICY, _st(5, 50.0), 0.1)  # 0.87 rad < 1.0
    ctrl, _ = ctrl_factory(bad_orientation_rad=None)
    assert ctrl.request(FsmState.POLICY, _st(5, 180.0), 0.1)


def test_fall_in_policy_sends_passive_on_trip_step(ctrl_factory):
    ctrl, cfg = ctrl_factory()
    assert ctrl.request(FsmState.POLICY, _st(5), 0.1)
    for k in range(3):
        _, info = ctrl.step(_st(6 + k), 0.1 + DT * k)
        assert info.state == FsmState.POLICY
    evals = ctrl._policy_step
    cmd, info = ctrl.step(_st(9, 90.0), 0.16)
    assert info.transitions == ["policy->passive(bad_orientation)"] and info.state == FsmState.PASSIVE
    assert ctrl._policy_step == evals and info.action is None  # not evaluated on the tilted sample
    assert info.extra["tilt_rad"] == pytest.approx(np.pi / 2, abs=1e-6)
    np.testing.assert_array_equal(cmd.motor.kp, np.zeros(motors.NUM_MOTORS, np.float32))
    np.testing.assert_allclose(cmd.motor.q, np.asarray(motors.DEFAULT_POS, np.float32), atol=1e-6)  # q_meas
