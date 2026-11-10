"""Item 327: iq*Kt-substituted torque (LowState motor.tau_src=1) is untrusted by the motor_tau gate by default.

Kt is not bench-confirmed (notes/324), so a gated slot with ``tau_src=1`` counts as railed (debounced like a
0x500 rail, refused at POLICY entry) unless ``fsm.motor_tau_trust_iq``. CPU only.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import deploy_fixtures as fx
import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy.config import ObsTermCfg, _fsm_from, hold_config
from dropbear_wbc.deploy.fsm import DeployController, FsmState
from dropbear_wbc.sdk import motors
from dropbear_wbc.sdk.types import TAU_SRC_IQ_KT

ROOT = sdk_test_paths.ROOT
DEF = np.asarray(motors.DEFAULT_POS)
KNEE = motors.MOTOR_NAMES.index("LL_knee_actuator_joint")
DT = 0.02


class Pol:
    has_reference = False

    def act(self, obs, time_step=None):
        return np.zeros(motors.NUM_MOTORS, np.float32)


def _ctrl(**fsm):
    cfg = hold_config()
    cfg.observations = [ObsTermCfg("g", "projected_gravity")]
    fsm.setdefault("motor_tau_gate", True)
    for k, v in fsm.items():
        setattr(cfg.fsm, k, v)
    return DeployController(cfg, Pol())


def _st(tick, iq=(), railed=()):
    st = fx.low_state(tick, DEF, motor_tau_valid=np.ones(22, np.uint8), motor_age_us=0)
    for i in railed:
        st.motor.tau_valid[i] = 0
    for i in iq:
        st.motor.tau_src[i] = TAU_SRC_IQ_KT
    return st


def _to_hold(ctrl):
    st = _st(0)
    ctrl.request(FsmState.MOVE_TO_DEFAULT, st, 0.0)
    ctrl.step(st, 2.0)
    assert ctrl.state == FsmState.HOLD


def _to_policy(ctrl):
    _to_hold(ctrl)
    st = _st(1)
    assert ctrl.request(FsmState.POLICY, st, 2.0)
    ctrl.step(st, 2.0)
    assert ctrl.state == FsmState.POLICY
    return 2.0, 2


def _run(ctrl, seq, t, tick):
    infos = []
    for k, iq in enumerate(seq):
        t += DT
        _, info = ctrl.step(_st(tick, iq), t)
        tick += 1
        infos.append(info)
        if ctrl.state != FsmState.POLICY:
            return k, infos
    return None, infos


def test_default_yaml_and_runner_flag():
    assert _fsm_from(None).motor_tau_trust_iq is False
    assert _fsm_from({"motor_tau_trust_iq": True}).motor_tau_trust_iq is True
    spec = importlib.util.spec_from_file_location("policy_runner", ROOT / "tools" / "policy_runner.py")
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    cfg, _ = pr.build(pr.parse_args(["--motor-tau-gate"]))
    assert cfg.fsm.motor_tau_trust_iq is False
    cfg, _ = pr.build(pr.parse_args(["--motor-tau-gate", "--motor-tau-trust-iq"]))
    assert cfg.fsm.motor_tau_trust_iq is True


def test_sustained_iq_trips_after_debounce_by_default():
    ctrl = _ctrl()  # N 4
    t, tick = _to_policy(ctrl)
    trip, infos = _run(ctrl, [(KNEE,)] * 8, t, tick)
    assert trip == 3
    assert infos[-1].transition == "policy->passive(motor_tau_invalid)"
    assert infos[0].extra["motor_tau_n_iq_untrusted"] == 1


def test_short_iq_burst_debounced():
    ctrl = _ctrl()
    t, tick = _to_policy(ctrl)
    trip, _ = _run(ctrl, ([(KNEE,)] * 3 + [()]) * 4, t, tick)
    assert trip is None and ctrl.state == FsmState.POLICY


def test_trusted_iq_never_trips():
    ctrl = _ctrl(motor_tau_trust_iq=True)
    t, tick = _to_policy(ctrl)
    trip, infos = _run(ctrl, [(KNEE,)] * 20, t, tick)
    assert trip is None
    assert infos[-1].extra["motor_tau_n_iq_untrusted"] == 0 and infos[-1].extra["motor_tau_n_railed"] == 0


@pytest.mark.parametrize("trust,ok", [(False, False), (True, True)])
def test_entry_refused_on_untrusted_iq(trust, ok):
    ctrl = _ctrl(motor_tau_trust_iq=trust)
    _to_hold(ctrl)
    assert bool(ctrl.request(FsmState.POLICY, _st(1, (KNEE,)), 2.0)) is ok


def test_unmasked_iq_slot_ignored():
    mask = np.ones(22, bool)
    mask[KNEE] = False
    ctrl = _ctrl(motor_tau_mask=mask)
    t, tick = _to_policy(ctrl)
    trip, infos = _run(ctrl, [(KNEE,)] * 10, t, tick)
    assert trip is None
    assert infos[-1].extra["motor_tau_n_iq_untrusted"] == 0


def test_gate_off_ignores_iq():
    ctrl = _ctrl(motor_tau_gate=False)
    t, tick = _to_policy(ctrl)
    trip, _ = _run(ctrl, [(KNEE,)] * 10, t, tick)
    assert trip is None
