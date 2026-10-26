"""Item 260: transitions raised in the same controller step (request() + a trip in step()) must all be reported.

Before, ``StepInfo.transition`` was one string and each assignment overwrote the previous one, so e.g. the
``hold->policy`` entry vanished when the first policy step tripped to Passive. CPU only.
"""
from __future__ import annotations

import numpy as np
import pytest

import deploy_fixtures as fx
import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy.config import hold_config, load_sidecar
from dropbear_wbc.deploy.fsm import DeployController, FsmState
from dropbear_wbc.sdk import motors

N = motors.NUM_MOTORS
DT = 0.02


def _hold_ctrl(**fsm):
    cfg = hold_config()
    cfg.fsm.move_to_default_s = 0.1
    for k, v in fsm.items():
        setattr(cfg.fsm, k, v)
    return DeployController(cfg)


def _st(tick, mode=1, **kw):
    st = fx.low_state(tick, np.asarray(motors.DEFAULT_POS), **kw)
    st.motor.mode[:] = np.uint8(mode)
    return st


def test_request_and_motor_mode_trip_in_same_step_keep_both():
    c = _hold_ctrl(motor_mode_grace_steps=0)
    c.step(_st(0), 0.0)                       # Passive, all motors armed (mode 1)
    st = _st(1, mode=0)                       # every motor drops out of mode 1 on the request step
    assert c.request(FsmState.MOVE_TO_DEFAULT, st, DT)
    _, info = c.step(st, DT)
    assert c.state == FsmState.PASSIVE
    assert info.transitions[0] == "passive->move_to_default"
    assert info.transitions[1].startswith("move_to_default->passive(motor_mode_dropped")
    assert info.transition == info.transitions[-1]


def test_latched_refusal_survives_with_no_trip():
    c = _hold_ctrl(motor_mode_grace_steps=0)
    c.step(_st(0), 0.0)
    c.request(FsmState.MOVE_TO_DEFAULT, _st(1), DT)
    c.step(_st(1, mode=0), DT)                # trips + latches
    assert c.latched is not None
    assert not c.request(FsmState.MOVE_TO_DEFAULT, _st(2), 2 * DT)
    _, info = c.step(_st(2), 2 * DT)
    assert info.transitions == ["passive->move_to_default(refused:motor_mode_latched)"]


def test_single_transition_and_quiet_steps_unchanged():
    c = _hold_ctrl()
    c.request(FsmState.MOVE_TO_DEFAULT, _st(0), 0.0)
    _, info = c.step(_st(0), 0.0)
    assert info.transitions == ["passive->move_to_default"] and info.transition == "passive->move_to_default"
    _, info = c.step(_st(1), DT)
    assert info.transitions == [] and info.transition is None


def test_policy_entry_kept_when_first_policy_step_trips(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from dropbear_wbc.deploy.motion import OnnxMotion
    from dropbear_wbc.deploy.policy import OnnxPolicy

    fx.write_beyondmimic_onnx(tmp_path / "policy.onnx")
    fx.write_sidecar(tmp_path / "policy.json", "policy.onnx")
    policy = OnnxPolicy(tmp_path / "policy.onnx")
    cfg = load_sidecar(tmp_path / "policy.json", policy.metadata)
    motion = OnnxMotion(cfg.motion, cfg.joint_names, cfg.step_dt, policy.reference, list(fx.REF_JOINTS),
                        list(fx.BODIES), fx.T_FRAMES)
    ctrl = DeployController(cfg, policy, motion)
    ctrl.state = FsmState.HOLD
    up = fx.low_state(5, np.asarray(motors.DEFAULT_POS))
    up.motor.mode[:] = np.uint8(1)
    st = fx.low_state(6, np.asarray(motors.DEFAULT_POS), quat_wxyz=(0, 1, 0, 0))  # tips over before the first step
    st.motor.mode[:] = np.uint8(1)
    assert ctrl.request(FsmState.POLICY, up, 0.1)
    _, info = ctrl.step(st, 0.12)
    assert ctrl.state == FsmState.PASSIVE
    assert info.transitions == ["hold->policy", "policy->passive(bad_orientation)"]
