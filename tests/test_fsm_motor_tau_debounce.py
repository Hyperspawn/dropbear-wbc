"""Item 321: rail debounce for the optional motor_tau_gate (CPU only).

A 5-10 ms 0x500 rail (notes/313 knee step) must not drop POLICY -> PASSIVE when ``motor_tau_max_bad=0``; N
consecutive railed LowState ticks must; a stale age still trips on the first sample.
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
from dropbear_wbc.sdk.types import MOTOR_AGE_UNKNOWN

ROOT = sdk_test_paths.ROOT
DEF = np.asarray(motors.DEFAULT_POS)
KNEE = motors.MOTOR_NAMES.index("LL_knee_actuator_joint")
DT = 0.02


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
    fsm.setdefault("motor_tau_gate", True)
    for k, v in fsm.items():
        setattr(cfg.fsm, k, v)
    pol = CountingPolicy()
    return DeployController(cfg, pol), pol


def _st(tick, railed=(), age_us=0):
    tv = np.ones(22, np.uint8)
    for i in railed:
        tv[i] = 0
    return fx.low_state(tick, DEF, motor_tau_valid=tv, motor_age_us=age_us)


def _to_policy(ctrl):
    st = _st(0)
    ctrl.request(FsmState.MOVE_TO_DEFAULT, st, 0.0)
    ctrl.step(st, 2.0)
    assert ctrl.state == FsmState.HOLD
    st = _st(1)
    assert ctrl.request(FsmState.POLICY, st, 2.0)
    ctrl.step(st, 2.0)
    assert ctrl.state == FsmState.POLICY
    return 2.0, 2


def _run(ctrl, seq, t, tick):
    """seq: list of railed-slot tuples, one new LowState tick per FSM step. Returns (trip step or None, infos)."""
    infos = []
    for k, railed in enumerate(seq):
        t += DT
        _, info = ctrl.step(_st(tick, railed), t)
        tick += 1
        infos.append(info)
        if ctrl.state != FsmState.POLICY:
            return k, infos
    return None, infos


def test_default_and_yaml_and_runner_flag():
    assert _fsm_from(None).motor_tau_invalid_steps == 4
    assert _fsm_from({"motor_tau_invalid_steps": 2}).motor_tau_invalid_steps == 2
    with pytest.raises(ValueError):
        _fsm_from({"motor_tau_invalid_steps": 0})
    spec = importlib.util.spec_from_file_location("policy_runner", ROOT / "tools" / "policy_runner.py")
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    cfg, _ = pr.build(pr.parse_args(["--motor-tau-gate", "--motor-tau-invalid-steps", "3"]))
    assert cfg.fsm.motor_tau_invalid_steps == 3
    cfg, _ = pr.build(pr.parse_args(["--motor-tau-gate"]))
    assert cfg.fsm.motor_tau_invalid_steps == 4
    with pytest.raises(SystemExit):
        pr.build(pr.parse_args(["--motor-tau-invalid-steps", "0"]))


def test_single_rail_does_not_trip_at_default_n():
    ctrl, pol = _ctrl()  # max_bad 0, N 4
    t, tick = _to_policy(ctrl)
    trip, infos = _run(ctrl, [(KNEE,)] + [()] * 10, t, tick)
    assert trip is None and ctrl.state == FsmState.POLICY
    assert infos[0].extra["motor_tau_n_railed"] == 1 and infos[0].extra["motor_tau_n_bad"] == 0
    assert infos[0].extra["motor_tau_fault"] is None


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_n_consecutive_rails_trip_on_nth(n):
    ctrl, pol = _ctrl(motor_tau_invalid_steps=n)
    t, tick = _to_policy(ctrl)
    calls0 = pol.calls
    trip, infos = _run(ctrl, [(KNEE,)] * (n + 3), t, tick)
    assert trip == n - 1
    assert infos[-1].transition == "policy->passive(motor_tau_invalid)"
    assert pol.calls == calls0 + n - 1  # not evaluated on the tripping sample
    assert infos[-1].extra["motor_tau_n_bad"] == 1


def test_interrupted_rails_reset_count():
    ctrl, _ = _ctrl(motor_tau_invalid_steps=4)
    t, tick = _to_policy(ctrl)
    seq = ([(KNEE,)] * 3 + [()]) * 5
    trip, _ = _run(ctrl, seq, t, tick)
    assert trip is None


def test_reused_tick_does_not_advance_debounce():
    ctrl, _ = _ctrl(motor_tau_invalid_steps=2, motor_tau_max_age_s=None, state_gate=False, imu_gate=False)
    t, tick = _to_policy(ctrl)
    st = _st(tick, (KNEE,))
    for k in range(5):  # same railed sample re-used 5 steps: still one sample
        ctrl.step(st, t + DT * (k + 1))
        assert ctrl.state == FsmState.POLICY
    _, info = ctrl.step(_st(tick + 1, (KNEE,)), t + DT * 6)
    assert info.transition == "policy->passive(motor_tau_invalid)"


def test_stale_still_immediate_with_debounce():
    ctrl, pol = _ctrl(motor_tau_invalid_steps=10)
    t, tick = _to_policy(ctrl)
    _, info = ctrl.step(_st(tick, age_us=61_000), t + DT)
    assert info.transition == "policy->passive(motor_tau_stale)"
    # railed + unknown age (HAL before first reply / lost reply) is stale now, not debounced as a rail
    ctrl2, _ = _ctrl(motor_tau_invalid_steps=10)
    t, tick = _to_policy(ctrl2)
    age = np.zeros(22, np.uint32)
    age[KNEE] = MOTOR_AGE_UNKNOWN  # only the railed slot is stale
    _, info = ctrl2.step(_st(tick, (KNEE,), age_us=age), t + DT)
    assert info.transition == "policy->passive(motor_tau_stale)" and info.extra["motor_tau_n_bad"] == 1


def test_entry_still_refuses_on_any_rail():
    ctrl, _ = _ctrl(motor_tau_invalid_steps=10)
    st = _st(0)
    ctrl.request(FsmState.MOVE_TO_DEFAULT, st, 0.0)
    ctrl.step(st, 2.0)
    assert ctrl.request(FsmState.POLICY, _st(1, (KNEE,)), 2.0) is False
    assert ctrl.request(FsmState.POLICY, _st(2), 2.02) is True


def test_mask_excluding_knee_ignores_sustained_knee_rail():
    """notes/31: stance knees are railed on every frame; a mask without the knees keeps POLICY."""
    mask = np.ones(22, bool)
    mask[[KNEE, motors.MOTOR_NAMES.index("RL_knee_actuator_joint")]] = False
    ctrl, _ = _ctrl(motor_tau_mask=mask, motor_tau_invalid_steps=1)
    t, tick = _to_policy(ctrl)
    trip, _ = _run(ctrl, [(KNEE,)] * 50, t, tick)
    assert trip is None


# ----------------------------------------------------------------------------- mutants

def _oracle(observe_fn) -> str:
    """PASS only if: single rail tolerated at N=4, 4 consecutive rails trip, single stale sample trips."""
    def mk():
        c, _ = _ctrl(motor_tau_invalid_steps=4)
        c.observe_motor_tau = observe_fn.__get__(c, DeployController)  # type: ignore[method-assign]
        return c
    c = mk(); t, tick = _to_policy(c)
    if _run(c, [(KNEE,)] + [()] * 5, t, tick)[0] is not None:
        return "FAIL(single_rail_tripped)"
    c = mk(); t, tick = _to_policy(c)
    if _run(c, [(KNEE,)] * 6, t, tick)[0] != 3:
        return "FAIL(n_rails_not_tripped_on_nth)"
    c = mk(); t, tick = _to_policy(c)
    _, info = c.step(_st(tick, age_us=61_000), t + DT)
    if info.transition != "policy->passive(motor_tau_stale)":
        return "FAIL(stale_not_immediate)"
    return "PASS"


def _with_n(n):
    real = DeployController.observe_motor_tau

    def f(self, low_state, t):
        old = self.cfg.fsm.motor_tau_invalid_steps
        self.cfg.fsm.motor_tau_invalid_steps = n
        try:
            return real(self, low_state, t)
        finally:
            self.cfg.fsm.motor_tau_invalid_steps = old
    return f


def test_mutants_killed():
    real = DeployController.observe_motor_tau
    assert _oracle(real) == "PASS"
    # M1: no debounce (0066 behaviour)
    assert _oracle(_with_n(1)) == "FAIL(single_rail_tripped)"

    # M2: debounce also applied to stale (stale needs N consecutive stale samples)
    def debounce_stale(self, low_state, t):
        fault, n_bad, worst = real(self, low_state, t)
        cnt = getattr(self, "_m2_stale", 0)
        if fault == "motor_tau_stale":
            cnt += 1
            if cnt < self.cfg.fsm.motor_tau_invalid_steps:
                fault, n_bad = None, 0
        else:
            cnt = 0
        self._m2_stale = cnt
        return fault, n_bad, worst
    assert _oracle(debounce_stale) == "FAIL(stale_not_immediate)"

    # M3: count steps instead of consecutive (never reset on a valid sample) -> interrupted rails trip
    def no_reset(self, low_state, t):
        prev = self._tau_rail_steps.copy()
        out = real(self, low_state, t)
        self._tau_rail_steps = np.maximum(self._tau_rail_steps, prev)
        return out
    c, _ = _ctrl(motor_tau_invalid_steps=4)
    c.observe_motor_tau = no_reset.__get__(c, DeployController)  # type: ignore[method-assign]
    t, tick = _to_policy(c)
    assert _run(c, ([(KNEE,)] * 3 + [()]) * 3, t, tick)[0] is not None  # mutant trips; real does not (test above)
