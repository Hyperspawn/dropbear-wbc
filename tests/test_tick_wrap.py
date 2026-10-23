"""Item 258: LowState.tick is uint32 on the wire (wraps after 99.4 days at 500 Hz).

The deploy FSM state gate must read the wrap as a forward step (not ``state_tick_regressed``), the cross-host
tick-lag estimator must stay continuous across it, and the runner/teleop loop schedulers must not stall on it.
"""
from __future__ import annotations

import numpy as np
import pytest

import deploy_fixtures as fx
import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy.config import ObsTermCfg, hold_config
from dropbear_wbc.deploy.fsm import DeployController, FsmState
from dropbear_wbc.sdk import motors
from dropbear_wbc.sdk.types import TICK_MOD, LowCmd, TickUnwrapper, tick_diff

DEF = np.asarray(motors.DEFAULT_POS)


def _st(tick):
    return fx.low_state(tick % TICK_MOD, DEF, imu_valid=1, imu_age_us=5000)


class _Policy:
    has_reference = False

    def __init__(self):
        self.calls = 0

    def act(self, obs, time_step=None):
        self.calls += 1
        return np.zeros(motors.NUM_MOTORS, np.float32)


def _run(start, n=60, lag_at=None, lag_ticks=0):
    cfg = hold_config()
    cfg.observations = [ObsTermCfg("g", "projected_gravity")]
    pol = _Policy()
    ctrl = DeployController(cfg, pol)
    ctrl.state_tick_dt_s = 0.002
    s = _st(start)
    ctrl.request(FsmState.MOVE_TO_DEFAULT, s, 0.0)
    ctrl.step(s, 2.0, state_age_s=0.001)
    t, tick = 2.0, start + 10
    assert ctrl.request(FsmState.POLICY, _st(tick), t, state_age_s=0.001)
    faults, lags = [], []
    for k in range(n):
        t += 0.02
        tick += 10
        wire = tick - (lag_ticks * (k - lag_at + 1) if lag_at is not None and k >= lag_at else 0)
        _, info = ctrl.step(_st(wire), t, state_age_s=0.001)
        faults.append(info.extra["state_fault"])
        lags.append(info.extra["state_lag_s"])
    return ctrl, pol, faults, lags


def test_tick_diff_wrap_and_restart():
    assert tick_diff(4, 0xFFFFFFF6) == 14
    assert tick_diff(0xFFFFFFF6, 4) == -14
    assert tick_diff(0, 1_000_000) == -1_000_000
    assert tick_diff(1_000_010, 1_000_000) == 10


def test_unwrapper_is_monotonic_across_the_wrap_and_masks_back_on_the_wire():
    u = TickUnwrapper()
    ticks = [(TICK_MOD - 25 + 10 * k) % TICK_MOD for k in range(6)]
    out = [u(x) for x in ticks]
    assert np.all(np.diff(out) == 10) and out[-1] > TICK_MOD
    # Echoing the unwrapped tick in a LowCmd puts the wire value back.
    assert LowCmd.from_bytes(LowCmd(tick=out[-1]).to_bytes()).tick == ticks[-1]
    assert u(5) < out[-1]  # a producer restart stays visible as a backwards step


def test_fsm_wrap_is_not_a_regression():
    ctrl, pol, faults, lags = _run(TICK_MOD - 300)
    assert all(f is None for f in faults), faults
    assert ctrl.state == FsmState.POLICY and pol.calls == 60
    assert max(lags) < 1e-9


def test_fsm_lag_estimator_still_fires_after_the_wrap():
    # After the wrap (k=28 is the wrap step) the producer slows to 5 ticks per 10-tick step: lag grows 10 ms/step
    # and must be caught as stale (30 ms limit), with no spurious regression anywhere.
    ctrl, _, faults, lags = _run(TICK_MOD - 300, lag_at=40, lag_ticks=5)
    assert "state_tick_regressed" not in faults
    first = faults.index("state_stale")
    assert 41 <= first <= 44 and lags[first] > 0.03 and ctrl.state == FsmState.PASSIVE


def test_fsm_real_restart_is_still_a_regression():
    cfg = hold_config()
    cfg.observations = [ObsTermCfg("g", "projected_gravity")]
    ctrl = DeployController(cfg, _Policy())
    ctrl.state_tick_dt_s = 0.002
    ctrl.observe_state(_st(1_000_000), 0.0, 0.001)
    fault, _, _ = ctrl.observe_state(_st(0), 0.02, 0.001)
    assert fault == "state_tick_regressed"


@pytest.mark.parametrize("tool", ["policy_runner.py", "teleop_arm.py"])
def test_tools_unwrap_every_lowstate_read(tool):
    """Grep gate: every ``sub.Read`` in the loop tools goes through the tick unwrapper (the sim-clock scheduler
    compared raw ticks and hung forever at the wrap: notes/258)."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "tools" / tool).read_text(encoding="utf-8")
    reads = re.findall(r"[\w.]*sub\.Read\(", src)
    assert reads and all(r == "sub.Read(" for r in reads)
    assert src.count("tick_unwrap.state(sub.Read(") == len(reads)
