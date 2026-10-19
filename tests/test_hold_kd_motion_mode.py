"""Item 156: FSM hold/passive kd must fit the motion-mode field; runner fails if mode never ENABLE."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy.config import (
    FsmCfg,
    clamp_deploy_motion_mode_gains,
    hold_config,
    load_deploy_yaml,
    load_sidecar,
    motion_mode_field_kd,
    motion_mode_field_kp,
)
from dropbear_wbc.deploy.fsm import DeployController, FsmState
from dropbear_wbc.robots.hw_motor_specs import MOTION_MODE_LIMITS
from dropbear_wbc.sdk import motors

import deploy_fixtures as fx


def test_legacy_default_kd_knees_exceed_field():
    """Reproduce notes/46 F1: DEFAULT_KD knees = 12 > MOTION_MODE kd_max 5."""
    assert motors.DEFAULT_KD[3] == 12.0
    assert motors.DEFAULT_KD[9] == 12.0
    assert MOTION_MODE_LIMITS["kd_max"] == 5.0
    assert motors.DEFAULT_KD[3] > MOTION_MODE_LIMITS["kd_max"]


def test_motion_mode_field_kd_clamps_knees_only():
    kd = motion_mode_field_kd()
    assert kd.shape == (22,)
    assert kd[3] == 5.0 and kd[9] == 5.0
    for i, name in enumerate(motors.MOTOR_NAMES):
        if i not in (3, 9):
            assert kd[i] == motors.DEFAULT_KD[i], name
    assert np.all(kd <= MOTION_MODE_LIMITS["kd_max"])
    assert np.all(motion_mode_field_kp() <= MOTION_MODE_LIMITS["kp_max"])


def test_hold_config_defaults_are_field_legal():
    cfg = hold_config()
    assert np.all(cfg.fsm.hold_kd <= 5.0)
    assert np.all(cfg.fsm.passive_kd <= 5.0)
    assert cfg.fsm.hold_kd[3] == 5.0
    assert cfg.kd[3] == 5.0
    # no clamp log when defaults already safe
    assert not cfg.meta.get("motion_mode_gains_clamped")


def test_clamp_logs_legacy_knee_overshoot():
    cfg = hold_config()
    cfg.fsm.hold_kd = np.asarray(motors.DEFAULT_KD, float)  # restore illegal
    cfg.fsm.passive_kd = np.asarray(motors.DEFAULT_KD, float)
    cfg.kd = np.asarray(motors.DEFAULT_KD, float)
    with pytest.raises(ValueError, match="motion-mode"):
        cfg.validate()
    log = clamp_deploy_motion_mode_gains(cfg)
    names = {(e["name"], e["field"]) for e in log}
    assert ("LL_knee_actuator_joint", "fsm.hold_kd") in names
    assert ("RL_knee_actuator_joint", "fsm.hold_kd") in names
    assert ("LL_knee_actuator_joint", "kd") in names
    cfg.validate()  # now legal
    assert cfg.fsm.hold_kd[3] == 5.0


def test_fsm_passive_and_hold_emit_field_kd():
    cfg = hold_config()
    ctrl = DeployController(cfg)
    st = fx.low_state(0, np.zeros(22))
    cmd, info = ctrl.step(st, 0.0)
    assert info.state == FsmState.PASSIVE
    assert float(cmd.motor.kd.max()) <= 5.0 + 1e-6
    assert float(cmd.motor.kd[3]) == 5.0
    ctrl.request(FsmState.MOVE_TO_DEFAULT, st, 0.0)
    cmd, _ = ctrl.step(st, 2.0)
    assert float(cmd.motor.kd[3]) == 5.0
    assert float(cmd.motor.kd[9]) == 5.0


def test_sidecar_with_legacy_damping_is_clamped(tmp_path):
    side = {
        "schema": "dropbear-policy-sidecar-v1",
        "joint_names": list(motors.MOTOR_NAMES),
        "joint_stiffness": list(motors.DEFAULT_KP),
        "joint_damping": list(motors.DEFAULT_KD),  # knees 12
        "default_joint_pos": list(motors.DEFAULT_POS),
        "action_scale": 0.25,
        "observations": [],
        "fsm": {"hold_kd": list(motors.DEFAULT_KD), "passive_kd": list(motors.DEFAULT_KD)},
    }
    p = tmp_path / "side.json"
    p.write_text(json.dumps(side), encoding="utf-8")
    cfg = load_sidecar(p)
    assert cfg.kd[3] == 5.0 and cfg.fsm.hold_kd[3] == 5.0
    clamped = cfg.meta["motion_mode_gains_clamped"]
    assert any(e["before"] == 12.0 and e["after"] == 5.0 for e in clamped)


def test_deploy_yaml_damping_clamped(tmp_path):
    y = tmp_path / "deploy.yaml"
    y.write_text(textwrap.dedent(f"""\
        joint_names: {list(motors.MOTOR_NAMES)}
        step_dt: 0.02
        stiffness: {list(motors.DEFAULT_KP)}
        damping: {list(motors.DEFAULT_KD)}
        default_joint_pos: {list(motors.DEFAULT_POS)}
        fsm:
          hold_kd: {list(motors.DEFAULT_KD)}
          passive_kd: {list(motors.DEFAULT_KD)}
        actions:
          JointPositionAction:
            scale: 0.25
        observations: {{}}
    """), encoding="utf-8")
    cfg = load_deploy_yaml(y)
    assert cfg.fsm.hold_kd[3] == 5.0
    assert cfg.kd[3] == 5.0


def test_fsm_cfg_default_factory_is_safe():
    fsm = FsmCfg()
    assert fsm.hold_kd[3] == 5.0
    assert fsm.passive_kd[3] == 5.0
    assert float(np.max(fsm.hold_kd)) <= 5.0
