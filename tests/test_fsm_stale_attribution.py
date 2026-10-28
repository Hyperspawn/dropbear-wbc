"""Item 279: a re-used LowState tick ages the IMU sample and the LowState together. The state freshness gate owns that
case (grace step re-sends the last target without evaluating the policy, then state_stale); the IMU gate must not
pre-empt it, and must still trip on a sample that arrived stale or when the state gate cannot act."""
from __future__ import annotations

import numpy as np
import pytest

import deploy_fixtures as fx
import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy.config import ObsTermCfg, hold_config
from dropbear_wbc.deploy.fsm import DeployController, FsmState
from dropbear_wbc.sdk import motors

DEF = np.asarray(motors.DEFAULT_POS)
DT = 0.02


class CountingPolicy:
    has_reference = False

    def __init__(self):
        self.calls = 0

    def act(self, obs, time_step=None):
        self.calls += 1
        return np.zeros(motors.NUM_MOTORS, np.float32)


def _policy(imu_age_us=0, **fsm):
    cfg = hold_config()
    cfg.observations = [ObsTermCfg("g", "projected_gravity")]
    for k, v in fsm.items():
        setattr(cfg.fsm, k, v)
    pol = CountingPolicy()
    ctrl = DeployController(cfg, pol)
    st = fx.low_state(0, DEF)
    ctrl.request(FsmState.MOVE_TO_DEFAULT, st, 0.0)
    ctrl.step(st, 2.0)
    assert ctrl.state == FsmState.HOLD
    st = fx.low_state(10, DEF, imu_age_us=imu_age_us)
    t = 2.0
    assert ctrl.request(FsmState.POLICY, st, t)
    ctrl.step(st, t)
    assert ctrl.state == FsmState.POLICY and pol.calls == 1
    return ctrl, pol, st, t


@pytest.mark.parametrize("jitter", [0.0, 1e-4, 2e-3])
def test_lowstate_stall_trips_as_state_stale_at_any_step_jitter(jitter):
    """Defaults (state 30 ms + 1 grace step, IMU 60 ms): both expire on the same step; base said imu_stale whenever
    the step landed past 60.000 ms (CPU load), state_stale otherwise."""
    ctrl, pol, st, t = _policy()
    _, i1 = ctrl.step(st, t + DT)
    _, i2 = ctrl.step(st, t + 2 * DT)
    assert ctrl.state == FsmState.POLICY and i2.extra.get("state_hold_last") == 1
    _, i3 = ctrl.step(st, t + 3 * DT + jitter)
    assert i3.transitions == ["policy->passive(state_stale)"], i3.transitions
    assert i3.extra["imu_fault"] == "imu_stale" and i3.extra.get("imu_stale_deferred") is True
    assert pol.calls == 2  # evaluated on the fresh sample and the 20 ms re-use only


def test_grace_step_survives_realistic_imu_age():
    """IMU age 25 ms (notes/39 range 7-34 ms): base tripped imu_stale on the first re-use step past 35 ms, so the
    state gate's hold-last grace never ran and a 2-step LowState gap dropped the robot to PASSIVE."""
    ctrl, pol, st, t = _policy(imu_age_us=25_000)
    ctrl.step(st, t + DT)
    _, info = ctrl.step(st, t + 2 * DT)  # 25 + 40 = 65 ms IMU, 40 ms state
    assert ctrl.state == FsmState.POLICY and info.extra.get("state_hold_last") == 1
    fresh = fx.low_state(13, DEF, imu_age_us=25_000)
    _, info = ctrl.step(fresh, t + 3 * DT)
    assert ctrl.state == FsmState.POLICY and info.transitions == []


def test_sample_that_arrives_stale_still_trips_imu_stale():
    ctrl, pol, st, t = _policy()
    old = fx.low_state(11, DEF, imu_age_us=70_000)
    _, info = ctrl.step(old, t + DT)
    assert info.transitions == ["policy->passive(imu_stale)"] and "imu_stale_deferred" not in info.extra


@pytest.mark.parametrize("fsm", [{"state_gate": False}, {"state_max_age_s": None}])
def test_imu_gate_keeps_reuse_when_state_gate_cannot_act(fsm):
    ctrl, pol, st, t = _policy(**fsm)
    for k in (1, 2):
        ctrl.step(st, t + k * DT)
        assert ctrl.state == FsmState.POLICY
    _, info = ctrl.step(st, t + 3 * DT + 1e-4)
    assert info.transitions == ["policy->passive(imu_stale)"]
