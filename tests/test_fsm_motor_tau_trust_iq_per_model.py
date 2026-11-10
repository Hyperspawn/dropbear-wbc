"""Item 328: per-slot / per-model iq*Kt trust for the motor_tau gate.

E2 (bench Kt per model) may confirm one model before the others; ``fsm.motor_tau_trust_iq`` therefore accepts a
bool[22] array, or motor / motor-model names resolved through ``JOINT_MOTOR_MAPS``. CPU only.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy.config import _fsm_from, trust_iq_from, trusted_iq_slots
from dropbear_wbc.deploy.fsm import FsmState
from dropbear_wbc.robots import hw_motor_specs as hw
from dropbear_wbc.sdk import motors
from test_fsm_motor_tau_trust_iq import KNEE, _ctrl, _run, _st, _to_hold, _to_policy

ROOT = sdk_test_paths.ROOT
RKNEE = motors.MOTOR_NAMES.index("RL_knee_actuator_joint")
HIP = motors.MOTOR_NAMES.index("LL_hip_joint")


def test_model_names_expand_through_map():
    v = trust_iq_from(["EPS-CEM-60"])
    assert trusted_iq_slots(v) == ["LL_knee_actuator_joint", "RL_knee_actuator_joint"]
    jm = hw.JOINT_MOTOR_MAPS[hw.BASE_ARM_MAP]
    for model in set(jm.values()):
        got = trusted_iq_slots(trust_iq_from([model]))
        assert got == [m for m in motors.MOTOR_NAMES if jm.get(m) == model] and got


def test_motor_names_csv_bools_and_errors():
    assert trusted_iq_slots(trust_iq_from("LL_hip_joint, RL_knee_actuator_joint")) == [
        "LL_hip_joint", "RL_knee_actuator_joint"]
    assert trust_iq_from([]) is False and trust_iq_from(True) is True and trust_iq_from(None) is False
    b = [False] * 22
    b[3] = True
    assert trusted_iq_slots(trust_iq_from(b)) == [motors.MOTOR_NAMES[3]]
    with pytest.raises(ValueError):
        trust_iq_from(["EPS-CEM-61"])
    with pytest.raises(ValueError):
        trust_iq_from([True] * 21)
    with pytest.raises(ValueError):
        trust_iq_from(["EPS-CEM-60"], "no_such_map")
    assert trusted_iq_slots(_fsm_from({"motor_tau_trust_iq": ["EPS-CEM-60"]}).motor_tau_trust_iq) == [
        "LL_knee_actuator_joint", "RL_knee_actuator_joint"]


def test_trusted_model_passes_untrusted_model_trips():
    trust = trust_iq_from(["EPS-CEM-60"])
    ctrl = _ctrl(motor_tau_trust_iq=trust)
    t, tick = _to_policy(ctrl)
    trip, infos = _run(ctrl, [(KNEE, RKNEE)] * 20, t, tick)
    assert trip is None and infos[-1].extra["motor_tau_n_iq_untrusted"] == 0
    trip, infos = _run(ctrl, [(KNEE, HIP)] * 8, t + 20 * 0.02, tick + 20)
    assert trip == 3 and infos[-1].transition == "policy->passive(motor_tau_invalid)"
    assert infos[0].extra["motor_tau_n_iq_untrusted"] == 1


@pytest.mark.parametrize("slots,ok", [((KNEE,), True), ((HIP,), False)])
def test_entry_per_model(slots, ok):
    ctrl = _ctrl(motor_tau_trust_iq=trust_iq_from(["EPS-CEM-60"]))
    _to_hold(ctrl)
    assert bool(ctrl.request(FsmState.POLICY, _st(1, slots), 2.0)) is ok


def test_runner_flag_and_meta():
    spec = importlib.util.spec_from_file_location("policy_runner", ROOT / "tools" / "policy_runner.py")
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    cfg, _ = pr.build(pr.parse_args(["--motor-tau-gate", "--motor-tau-trust-iq-models", "EPS-CEM-60"]))
    assert pr._trust_meta(cfg.fsm.motor_tau_trust_iq) == ["LL_knee_actuator_joint", "RL_knee_actuator_joint"]
    assert pr._trust_meta(False) is False
    with pytest.raises(SystemExit):
        pr.build(pr.parse_args(["--motor-tau-trust-iq", "--motor-tau-trust-iq-models", "EPS-CEM-60"]))
    with pytest.raises(SystemExit):
        pr.build(pr.parse_args(["--motor-tau-trust-iq-models", "bogus"]))
