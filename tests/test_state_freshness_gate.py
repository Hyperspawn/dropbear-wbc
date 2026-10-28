"""Item 236: LowState freshness gate in the deploy FSM (transport age, tick re-use, cross-host tick lag)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import deploy_fixtures as fx
import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy.config import ObsTermCfg, _fsm_from, hold_config
from dropbear_wbc.deploy.fsm import DeployController, FsmState
from dropbear_wbc.sdk import motors

DEF = np.asarray(motors.DEFAULT_POS)
DT, TICK_DT = 0.02, 0.002


class CountingPolicy:
    has_reference = False

    def __init__(self):
        self.calls = 0

    def act(self, obs, time_step=None):
        self.calls += 1
        return np.full(motors.NUM_MOTORS, 0.1 * self.calls, np.float32)


def _st(tick):
    return fx.low_state(tick, DEF, imu_valid=1, imu_age_us=5000)


def _policy_ctrl(**fsm):
    cfg = hold_config()
    cfg.observations = [ObsTermCfg("g", "projected_gravity")]
    for k, v in fsm.items():
        setattr(cfg.fsm, k, v)
    pol = CountingPolicy()
    ctrl = DeployController(cfg, pol)
    ctrl.state_tick_dt_s = TICK_DT
    s = _st(0)
    ctrl.request(FsmState.MOVE_TO_DEFAULT, s, 0.0)
    ctrl.step(s, 2.0, state_age_s=0.001)
    assert ctrl.state == FsmState.HOLD
    assert ctrl.request(FsmState.POLICY, _st(1000), 2.0, state_age_s=0.001)
    return ctrl, pol, 2.0, 1000


def test_late_delivery_with_new_ticks_never_reaches_the_policy():
    ctrl, pol, t, tick = _policy_ctrl()
    for _ in range(5):
        t, tick = t + DT, tick + 10
        _, info = ctrl.step(_st(tick), t, state_age_s=0.7)  # notes/202: 0.7 s old, fresh imu.age_us
    assert pol.calls == 0
    assert ctrl.state == FsmState.PASSIVE and info.extra["state_fault"] == "state_stale"


def test_one_stale_step_resends_last_target_then_recovers():
    ctrl, pol, t, tick = _policy_ctrl()
    t, tick = t + DT, tick + 10
    _, info = ctrl.step(_st(tick), t, state_age_s=0.002)
    q_good = info.q_des.copy()
    assert pol.calls == 1
    t, tick = t + DT, tick + 10
    _, info = ctrl.step(_st(tick), t, state_age_s=0.050)
    assert pol.calls == 1 and ctrl.state == FsmState.POLICY and info.extra["state_hold_last"] == 1
    np.testing.assert_array_equal(info.q_des, q_good)
    t, tick = t + DT, tick + 10
    _, info = ctrl.step(_st(tick), t, state_age_s=0.002)
    assert pol.calls == 2 and ctrl.state == FsmState.POLICY and info.extra["state_fault"] is None


def test_two_stale_steps_go_passive():
    ctrl, pol, t, tick = _policy_ctrl()
    t, tick = t + DT, tick + 10
    ctrl.step(_st(tick), t, state_age_s=0.002)
    for _ in range(2):
        t, tick = t + DT, tick + 10
        _, info = ctrl.step(_st(tick), t, state_age_s=0.050)
    assert ctrl.state == FsmState.PASSIVE and info.transition == "policy->passive(state_stale)"
    assert pol.calls == 1


def test_reused_tick_counts_as_age_without_stamp():
    ctrl, pol, t, tick = _policy_ctrl()
    s = _st(tick + 10)
    states = []
    for _ in range(4):  # same tick for 0, 20, 40, 60 ms; imu gate alone allows 60 ms - 5 ms
        t += DT
        ctrl.step(s, t)
        states.append(ctrl.state)
    assert pol.calls == 2  # fresh + 20 ms re-use; 40 ms is held, 60 ms goes passive
    assert states[-1] == FsmState.PASSIVE


def test_policy_request_refused_on_stale_state():
    cfg = hold_config()
    cfg.observations = [ObsTermCfg("g", "projected_gravity")]
    ctrl = DeployController(cfg, CountingPolicy())
    s = _st(0)
    ctrl.request(FsmState.MOVE_TO_DEFAULT, s, 0.0)
    ctrl.step(s, 2.0)
    assert not ctrl.request(FsmState.POLICY, _st(1000), 2.0, state_age_s=0.2)
    _, info = ctrl.step(_st(1000), 2.0, state_age_s=0.2)
    assert ctrl.state == FsmState.HOLD and info.transition == "hold->policy refused(state_stale)"
    assert ctrl.request(FsmState.POLICY, _st(1010), 2.02, state_age_s=0.002)


def test_cross_host_lag_trips_without_any_stamp():
    ctrl, pol, t, tick = _policy_ctrl()
    k_trip = None
    for k in range(40):
        t += DT
        tick += 10 if k < 20 else 5  # delivered tick falls behind 10 ms per step after k=20
        ctrl.step(_st(tick), t)
        if ctrl.state == FsmState.PASSIVE:
            k_trip = k
            break
    assert k_trip is not None and 22 <= k_trip <= 25


@pytest.mark.parametrize("ppm", [-400.0, 400.0])
def test_clock_drift_is_not_lag(ppm):
    ctrl, pol, t, tick = _policy_ctrl()
    t0, tick0 = t, tick
    for k in range(1, 30001):  # 600 s at 50 Hz
        t = t0 + k * DT
        tick_now = tick0 + int(round(k * DT * (1.0 - ppm * 1e-6) / TICK_DT))
        ctrl.step(_st(tick_now), t)
    assert ctrl.state == FsmState.POLICY


def test_tick_regression_trips():
    ctrl, pol, t, tick = _policy_ctrl(state_stale_grace_steps=0)
    t, tick = t + DT, tick + 10
    ctrl.step(_st(tick), t)
    _, info = ctrl.step(_st(5), t + DT)  # HAL restarted
    assert ctrl.state == FsmState.PASSIVE and info.transition == "policy->passive(state_tick_regressed)"


def test_gate_off_and_config_keys():
    ctrl, pol, t, tick = _policy_ctrl(state_gate=False)
    ctrl.step(_st(tick + 10), t + DT, state_age_s=0.7)
    assert pol.calls == 1 and ctrl.state == FsmState.POLICY
    f = _fsm_from({"state_gate": False, "state_max_age_ms": 45, "state_stale_grace_steps": 0})
    assert f.state_gate is False and f.state_max_age_s == pytest.approx(0.045) and f.state_stale_grace_steps == 0
    assert _fsm_from({"state_max_age_ms": None}).state_max_age_s is None
    d = _fsm_from(None)
    assert d.state_gate and d.state_max_age_s == pytest.approx(0.030) and d.state_stale_grace_steps == 1
    with pytest.raises(ValueError):
        _fsm_from({"state_stale_grace_steps": -1})


# --- end to end: policy_runner --clock wall over the real ZMQ transport (item 236) ---------------------------------

def _clock_tick(t0: float, now: float, dt: float = 0.002) -> int:
    """LowState.tick of a fake robot whose counter runs on its own clock, like an MCU sample counter: a late loop
    iteration skips ticks instead of falling behind wall time. Counting publishes instead made the tick trail the
    runner's wall clock whenever the fake robot thread was starved (machine load, GIL), which the tick-lag estimator
    then correctly reported as state_stale on a fresh frame (item 280)."""
    return 1 + int((now - t0) / dt + 1e-9)


def _wall_robot(ep, stall_after_cmds=None, stall_s=0.0):
    """500 Hz fake robot in real time; stamp_ns=0 so the transport stamps it. Optionally stops publishing for
    ``stall_s`` after ``stall_after_cmds`` LowCmds (a reader- or bus-side stall, notes/202 F2)."""
    import threading
    import time

    from dropbear_wbc.sdk.transport import LOWCMD, LOWSTATE, ChannelPublisher, ChannelSubscriber
    from dropbear_wbc.sdk.types import LowCmd, LowState


    class R(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.running, self.cmds, self.ticks, self.stalled = True, 0, 0, False
            self.q = np.zeros(22)

        def run(self):
            pub = ChannelPublisher(LOWSTATE, LowState, endpoints=ep, role="robot")
            sub = ChannelSubscriber(LOWCMD, LowCmd, endpoints=ep, role="robot")
            pub.Init()
            sub.Init()
            mode = np.zeros(22, dtype=np.uint8)
            nxt = t0 = time.perf_counter()
            try:
                while self.running:
                    cmd = sub.Read(timeout=0.0)
                    if cmd is not None:
                        self.q += 0.05 * (cmd.motor.q.astype(float) - self.q)
                        mode = np.asarray(cmd.motor.mode, dtype=np.uint8).copy()
                        self.cmds += 1
                    if stall_after_cmds is not None and not self.stalled and self.cmds >= stall_after_cmds:
                        self.stalled = True
                        time.sleep(stall_s)
                        nxt = time.perf_counter()
                    tick = _clock_tick(t0, time.perf_counter())
                    if tick == self.ticks:  # loop ran early: same tick, nothing new to publish
                        nxt += 0.002
                        time.sleep(max(0.0, nxt - time.perf_counter()))
                        continue
                    self.ticks = tick
                    st = fx.low_state(self.ticks, self.q)
                    st.stamp_ns = 0
                    st.motor.mode[:] = mode
                    pub.Write(st)
                    nxt += 0.002
                    time.sleep(max(0.0, nxt - time.perf_counter()))
            finally:
                pub.Close()
                sub.Close()

    return R()


def _run_wall(tmp_path, ep, robot):
    import importlib.util
    import json

    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    root = Path(__file__).resolve().parents[1]
    fx.write_beyondmimic_onnx(tmp_path / "policy.onnx", obs_dim=sum(t["dim"] for t in fx.OBS_TERMS_FULL))
    fx.write_sidecar(tmp_path / "policy.json", "policy.onnx", terms=fx.OBS_TERMS_FULL)
    spec = importlib.util.spec_from_file_location("policy_runner", root / "tools" / "policy_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    robot.start()
    try:
        runner.main(["--mode", "policy", "--sidecar", str(tmp_path / "policy.json"), "--allow-privileged",
                     "--clock", "wall", "--duration", "2.0", "--passive-s", "0.1", "--move-s", "0.4",
                     "--hold-s", "0.2", "--cmd-port", str(ep.cmd_port), "--state-port", str(ep.state_port),
                     "--connect-timeout-s", "20", "--summary", str(tmp_path / "s.json"),
                     "--log", str(tmp_path / "steps.jsonl")])
    finally:
        robot.running = False
        robot.join(timeout=5)
    return json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))


def test_runner_wall_clock_no_false_trip(tmp_path):
    from dropbear_wbc.sdk.transport import Endpoints

    ep = Endpoints(cmd_port=27591, state_port=27592)
    s = _run_wall(tmp_path, ep, _wall_robot(ep))
    names = [t["transition"] for t in s["transitions"]]
    assert "hold->policy" in names and not any("state_" in n for n in names), names
    assert s["state_gate"]["fault_steps"] == {} and s["state_gate"]["stamp_age"] is True


def test_runner_wall_clock_publisher_stall_goes_passive(tmp_path):
    from dropbear_wbc.sdk.transport import Endpoints

    ep = Endpoints(cmd_port=27593, state_port=27594)
    # 0.1+0.4+0.2 s = 35 steps before POLICY; stall 0.25 s ten steps into POLICY.
    s = _run_wall(tmp_path, ep, _wall_robot(ep, stall_after_cmds=45, stall_s=0.25))
    names = [t["transition"] for t in s["transitions"]]
    assert "hold->policy" in names, names
    assert "policy->passive(state_stale)" in names, names
    assert s["state_gate"]["fault_steps"].get("state_stale", 0) >= 2, s["state_gate"]
