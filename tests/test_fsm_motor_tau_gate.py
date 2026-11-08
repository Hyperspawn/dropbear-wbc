"""Item 314: optional deploy FSM POLICY gate on motor.tau_valid / age_us (item 85 follow-up; CPU only)."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import deploy_fixtures as fx
import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy.config import ObsTermCfg, _fsm_from, hold_config
from dropbear_wbc.deploy.fsm import DeployController, FsmState
from dropbear_wbc.sdk import motors
from dropbear_wbc.sdk.types import MOTOR_AGE_UNKNOWN

ROOT = sdk_test_paths.ROOT
DEF = np.asarray(motors.DEFAULT_POS)


class CountingPolicy:
    has_reference = False

    def __init__(self):
        self.calls = 0

    def act(self, obs, time_step=None):
        self.calls += 1
        return np.zeros(motors.NUM_MOTORS, np.float32)


def _ctrl(**fsm):
    cfg = hold_config()
    cfg.observations = [ObsTermCfg("g", "projected_gravity")]
    for k, v in fsm.items():
        setattr(cfg.fsm, k, v)
    pol = CountingPolicy()
    return DeployController(cfg, pol), pol


def _to_hold(ctrl, t0=0.0, tick0=0):
    st = fx.low_state(tick0, DEF, motor_tau_valid=1, motor_age_us=0)
    ctrl.request(FsmState.MOVE_TO_DEFAULT, st, t0)
    ctrl.step(st, t0 + 2.0)
    assert ctrl.state == FsmState.HOLD
    return t0 + 2.0


def _healthy(tick, **kw):
    return fx.low_state(tick, DEF, motor_tau_valid=1, motor_age_us=0, **kw)


def _railed(tick, slots=(3,), age_us=0):
    tv = np.ones(22, np.uint8)
    for i in slots:
        tv[i] = 0
    return fx.low_state(tick, DEF, motor_tau_valid=tv, motor_age_us=age_us)


# ----------------------------------------------------------------------------- defaults / YAML / runner

def test_gate_default_off_ignores_railed_and_unknown_age():
    """Default keeps legacy / Newton sims working (tau_valid fail-open, age UNKNOWN on zeros())."""
    ctrl, pol = _ctrl()  # motor_tau_gate default False
    assert ctrl.cfg.fsm.motor_tau_gate is False
    t = _to_hold(ctrl)
    # zeros()-style unknown age + one railed slot must NOT refuse POLICY when gate is off
    bad = _railed(10, age_us=MOTOR_AGE_UNKNOWN)
    assert ctrl.request(FsmState.POLICY, bad, t) is True
    ctrl.step(bad, t)
    assert ctrl.state == FsmState.POLICY and pol.calls == 1


def test_fsm_yaml_and_runner_flags():
    f = _fsm_from({"motor_tau_gate": True, "motor_tau_max_age_ms": 40, "motor_tau_max_bad": 2,
                   "motor_tau_mask": ["LL_knee_actuator_joint", "RL_knee_actuator_joint"]})
    assert f.motor_tau_gate and f.motor_tau_max_age_s == pytest.approx(0.040) and f.motor_tau_max_bad == 2
    assert int(f.motor_tau_mask.sum()) == 2
    assert _fsm_from({"motor_tau_max_age_ms": None}).motor_tau_max_age_s is None
    d = _fsm_from(None)
    assert d.motor_tau_gate is False and d.motor_tau_max_age_s == pytest.approx(0.060) and d.motor_tau_max_bad == 0
    with pytest.raises(ValueError):
        _fsm_from({"motor_tau_max_bad": -1})

    spec = importlib.util.spec_from_file_location("policy_runner", ROOT / "tools" / "policy_runner.py")
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    cfg, _ = pr.build(pr.parse_args(["--motor-tau-gate", "--motor-tau-max-age-ms", "35", "--motor-tau-max-bad", "1"]))
    assert cfg.fsm.motor_tau_gate and cfg.fsm.motor_tau_max_age_s == pytest.approx(0.035) and cfg.fsm.motor_tau_max_bad == 1
    cfg, _ = pr.build(pr.parse_args(["--motor-tau-gate", "--motor-tau-max-age-ms", "0"]))
    assert cfg.fsm.motor_tau_gate and cfg.fsm.motor_tau_max_age_s is None


# ----------------------------------------------------------------------------- gate behaviour

def test_railed_slot_refuses_and_drops_policy():
    ctrl, pol = _ctrl(motor_tau_gate=True, motor_tau_invalid_steps=1)  # 0066 behaviour; debounce: item 321 tests
    t = _to_hold(ctrl)
    assert ctrl.request(FsmState.POLICY, _railed(10), t) is False
    _, info = ctrl.step(_railed(10), t)
    assert ctrl.state == FsmState.HOLD and info.transition == "hold->policy refused(motor_tau_invalid)"
    assert pol.calls == 0
    # enter on healthy, then rail mid-run: leave before policy eval
    assert ctrl.request(FsmState.POLICY, _healthy(20), t + 0.02)
    ctrl.step(_healthy(20), t + 0.02)
    assert pol.calls == 1 and ctrl.state == FsmState.POLICY
    cmd, info = ctrl.step(_railed(30), t + 0.04)
    assert info.transition == "policy->passive(motor_tau_invalid)" and ctrl.state == FsmState.PASSIVE
    assert pol.calls == 1  # not evaluated on the railed sample
    np.testing.assert_allclose(cmd.motor.kp, 0.0)
    assert info.extra["motor_tau_n_bad"] == 1


def test_stale_age_refuses_and_drops():
    ctrl, pol = _ctrl(motor_tau_gate=True)
    t = _to_hold(ctrl)
    stale = fx.low_state(10, DEF, motor_tau_valid=1, motor_age_us=61_000)
    assert ctrl.request(FsmState.POLICY, stale, t) is False
    # boundary: age == max passes
    assert ctrl.request(FsmState.POLICY, fx.low_state(20, DEF, motor_tau_valid=1, motor_age_us=60_000), t)
    ctrl.step(fx.low_state(20, DEF, motor_tau_valid=1, motor_age_us=60_000), t)
    _, info = ctrl.step(fx.low_state(30, DEF, motor_tau_valid=1, motor_age_us=80_000), t + 0.02)
    assert info.transition == "policy->passive(motor_tau_stale)" and ctrl.state == FsmState.PASSIVE
    assert pol.calls == 1


def test_unknown_age_is_stale_when_gate_on():
    ctrl, _ = _ctrl(motor_tau_gate=True)
    t = _to_hold(ctrl)
    unknown = fx.low_state(10, DEF, motor_tau_valid=1, motor_age_us=MOTOR_AGE_UNKNOWN)
    assert ctrl.request(FsmState.POLICY, unknown, t) is False


def test_reused_tick_ages_motor_slots():
    ctrl, pol = _ctrl(motor_tau_gate=True, state_gate=False, imu_gate=False)
    t = _to_hold(ctrl)
    st = fx.low_state(10, DEF, motor_tau_valid=1, motor_age_us=10_000)
    assert ctrl.request(FsmState.POLICY, st, t)
    for k in range(3):  # 10 + 0/20/40 ms <= 60
        ctrl.step(st, t + 0.02 * k)
        assert ctrl.state == FsmState.POLICY
    _, info = ctrl.step(st, t + 0.06)  # 10 + 60 = 70 ms
    assert info.transition == "policy->passive(motor_tau_stale)"
    assert info.extra["motor_tau_age_s"] == pytest.approx(0.07)


def test_max_bad_threshold_and_mask():
    # tolerate one railed slot
    ctrl, pol = _ctrl(motor_tau_gate=True, motor_tau_max_bad=1, motor_tau_invalid_steps=1)
    t = _to_hold(ctrl)
    one = _railed(10, slots=(3,))
    assert ctrl.request(FsmState.POLICY, one, t)  # n_bad=1 not > 1
    ctrl.step(one, t)
    two = _railed(20, slots=(3, 9))
    _, info = ctrl.step(two, t + 0.02)
    assert info.transition == "policy->passive(motor_tau_invalid)" and info.extra["motor_tau_n_bad"] == 2

    # mask: only LL knee gated; railing RL knee alone is ignored
    mask = np.zeros(22, bool)
    mask[motors.MOTOR_NAMES.index("LL_knee_actuator_joint")] = True
    ctrl2, _ = _ctrl(motor_tau_gate=True, motor_tau_mask=mask)
    t = _to_hold(ctrl2)
    rk = motors.MOTOR_NAMES.index("RL_knee_actuator_joint")
    assert ctrl2.request(FsmState.POLICY, _railed(10, slots=(rk,)), t)
    lk = motors.MOTOR_NAMES.index("LL_knee_actuator_joint")
    assert not ctrl2.request(FsmState.POLICY, _railed(20, slots=(lk,)), t + 0.02)


def test_age_check_disable_keeps_valid_only():
    ctrl, _ = _ctrl(motor_tau_gate=True, motor_tau_max_age_s=None)
    t = _to_hold(ctrl)
    # unknown age OK when age check off; rail still trips
    assert ctrl.request(FsmState.POLICY, fx.low_state(10, DEF, motor_tau_valid=1, motor_age_us=MOTOR_AGE_UNKNOWN), t)
    assert not ctrl.request(FsmState.POLICY, _railed(20), t + 0.02)


# ----------------------------------------------------------------------------- mutants (2/2 must fail under broken observe)

def _suite_rail_and_stale(observe_fn) -> str:
    """Minimal oracle: PASS only if railed refuses and stale refuses under motor_tau_gate=True."""
    ctrl, _ = _ctrl(motor_tau_gate=True)
    # monkey-patch instance method
    ctrl.observe_motor_tau = observe_fn.__get__(ctrl, DeployController)  # type: ignore[method-assign]
    t = _to_hold(ctrl)
    if ctrl.request(FsmState.POLICY, _railed(10), t) is not False:
        return "FAIL(missed_rail_refuse)"
    if ctrl.request(FsmState.POLICY, fx.low_state(20, DEF, motor_tau_valid=1, motor_age_us=80_000), t) is not False:
        return "FAIL(missed_stale_refuse)"
    if ctrl.request(FsmState.POLICY, _healthy(30), t) is not True:
        return "FAIL(false_trip_healthy)"
    return "PASS"


def test_mutants_killed():
    """Mutant A ignores tau_valid; mutant B ignores age. Both must fail the oracle (real observe PASSes)."""
    real = DeployController.observe_motor_tau

    def ignore_valid(self, low_state, t):
        # age-only: force tau_valid all-1 before delegating to a local copy of age logic
        st = low_state
        tv = np.asarray(st.motor.tau_valid).copy()
        st.motor.tau_valid[:] = 1
        try:
            fault, n_bad, worst = real(self, st, t)
        finally:
            st.motor.tau_valid[:] = tv
        # If the only badness was rail, real returns None now -> mutant misses rail
        return fault, n_bad, worst

    def ignore_age(self, low_state, t):
        st = low_state
        au = np.asarray(st.motor.age_us).copy()
        st.motor.age_us[:] = 0
        try:
            fault, n_bad, worst = real(self, st, t)
        finally:
            st.motor.age_us[:] = au
        return fault, n_bad, worst

    assert _suite_rail_and_stale(real) == "PASS"
    assert _suite_rail_and_stale(ignore_valid) == "FAIL(missed_rail_refuse)"
    assert _suite_rail_and_stale(ignore_age) == "FAIL(missed_stale_refuse)"
