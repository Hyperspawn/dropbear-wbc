"""Item 157 (notes/46 F2): FSM L4 gate on LowState.motor.mode. CPU only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import deploy_fixtures as fx
import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy.config import FsmCfg, _fsm_from, hold_config
from dropbear_wbc.deploy.fsm import DeployController, FsmState
from dropbear_wbc.sdk import motors
from dropbear_wbc.sdk.transport import Endpoints

ROOT = Path(__file__).resolve().parents[1]
N = motors.NUM_MOTORS
DT = 0.02


def _st(tick, mode):
    st = fx.low_state(tick)
    st.motor.mode[:] = np.asarray(mode, np.uint8)
    return st


def _run(ctrl, modes, t0=0.0, hold_at=0):
    """Step the controller through ``modes`` (list of 22-vectors); request MoveToDefault at step ``hold_at``."""
    out = []
    for k, m in enumerate(modes):
        st = _st(k, m)
        t = t0 + k * DT
        if k == hold_at and ctrl.state == FsmState.PASSIVE and ctrl.latched is None:
            ctrl.request(FsmState.MOVE_TO_DEFAULT, st, t)
        _, info = ctrl.step(st, t)
        out.append((ctrl.state, info.transition))
    return out


def _ctrl(**fsm):
    cfg = hold_config()
    cfg.fsm.move_to_default_s = 0.1
    for k, v in fsm.items():
        setattr(cfg.fsm, k, v)
    return DeployController(cfg)


ONES = np.ones(N)


def test_all_enabled_stays_hold():
    c = _ctrl()
    res = _run(c, [ONES] * 60)
    assert res[-1][0] == FsmState.HOLD and c.latched is None


def test_single_step_drop_is_tolerated():
    c = _ctrl()
    modes = [ONES] * 60
    m = ONES.copy(); m[3] = 0
    modes[30] = m
    res = _run(c, modes)
    assert res[-1][0] == FsmState.HOLD and c.latched is None


def test_two_step_drop_trips_passive_and_latches():
    c = _ctrl()
    m = ONES.copy(); m[3] = 0  # LL knee
    modes = [ONES] * 30 + [m] * 30
    res = _run(c, modes)
    states = [s for s, _ in res]
    assert states[30] == FsmState.HOLD           # first bad step: grace
    assert states[31] == FsmState.PASSIVE        # second bad step: trip in the same step
    assert c.latched["reason"].startswith("motor_mode_dropped:")
    assert motors.MOTOR_NAMES[3] in c.latched["reason"]
    assert res[31][1].startswith("hold->passive(motor_mode_dropped")
    # latched: requests refused even after the motor comes back
    st = _st(99, ONES)
    assert c.request(FsmState.MOVE_TO_DEFAULT, st, 2.0) is False
    assert c.state == FsmState.PASSIVE and c.refused
    c.clear_latch()
    assert c.request(FsmState.MOVE_TO_DEFAULT, st, 2.0) is True


def test_passive_command_after_trip_is_damping():
    c = _ctrl()
    m = ONES.copy(); m[0:6] = 0  # whole left leg damped by the HAL
    _run(c, [ONES] * 30 + [m] * 5)
    cmd, info = c.step(_st(40, m), 40 * DT)
    assert info.state == FsmState.PASSIVE
    assert np.all(cmd.motor.kp == 0)


def test_policy_state_trips_too():
    """Gate runs before the per-state branch, so a policy-less controller can be checked in POLICY on the trip step."""
    c = _ctrl()
    c.state = FsmState.POLICY
    c._mode_armed[:] = True
    c._mode_bad_steps[12] = 1  # one bad step already seen
    m = ONES.copy(); m[12] = 0
    cmd, info = c.step(_st(1, m), 0.02)
    assert info.state == FsmState.PASSIVE and info.transition.startswith("policy->passive(motor_mode_dropped")
    assert motors.MOTOR_NAMES[12] in c.latched["reason"] and np.all(cmd.motor.kp == 0)


def test_never_reporting_source_is_not_tripped():
    """Legacy logs / mode-less sources: mode stays 0 everywhere. The FSM gate stays quiet (item-156 runner gate owns it)."""
    c = _ctrl()
    res = _run(c, [np.zeros(N)] * 80)
    assert res[-1][0] == FsmState.HOLD and c.latched is None


def test_partial_arm_trips_after_timeout():
    c = _ctrl(motor_mode_arm_timeout_s=0.5)
    m = ONES.copy(); m[6:12] = 0  # right leg never armed
    res = _run(c, [m] * 60)
    k = next(i for i, (s, _) in enumerate(res) if s == FsmState.PASSIVE and i > 0)
    assert 0.5 < k * DT <= 0.5 + 2 * DT
    assert c.latched["reason"].startswith("motor_mode_never_armed:")
    assert all(motors.MOTOR_NAMES[i] in c.latched["reason"] for i in range(6, 12))


def test_gate_disabled_and_mask():
    m = ONES.copy(); m[3] = 0
    c = _ctrl(motor_mode_gate=False)
    assert _run(c, [ONES] * 30 + [m] * 30)[-1][0] == FsmState.HOLD
    mask = np.ones(N, bool); mask[3] = False
    c = _ctrl(motor_mode_mask=mask)
    assert _run(c, [ONES] * 30 + [m] * 30)[-1][0] == FsmState.HOLD


def test_drop_in_passive_does_not_latch():
    c = _ctrl()
    m = ONES.copy(); m[3] = 0
    res = _run(c, [ONES] * 5 + [m] * 10, hold_at=10 ** 6)  # never leave Passive
    assert all(s == FsmState.PASSIVE for s, _ in res) and c.latched is None


def test_fsm_yaml_parsing():
    f = _fsm_from({"motor_mode_gate": False, "motor_mode_grace_steps": 3, "motor_mode_arm_timeout_s": None,
                   "motor_mode_mask": [motors.MOTOR_NAMES[0], motors.MOTOR_NAMES[5]]})
    assert f.motor_mode_gate is False and f.motor_mode_grace_steps == 3 and f.motor_mode_arm_timeout_s is None
    assert f.motor_mode_mask.sum() == 2 and f.motor_mode_mask[0] and f.motor_mode_mask[5]
    with pytest.raises(ValueError):
        _fsm_from({"motor_mode_mask": ["nope"]})
    with pytest.raises(ValueError):
        _fsm_from({"motor_mode_grace_steps": -1})
    d = FsmCfg()
    assert d.motor_mode_gate is True and d.motor_mode_grace_steps == 1 and d.motor_mode_arm_timeout_s == 0.5


def test_bridge_published_mode_helper():
    src = (ROOT / "tools" / "newton_bridge.py").read_text()
    ns: dict = {"np": np}
    start = src.index("def published_mode(")
    end = src.index("\ndef ", start + 10)
    exec(src[start:end], ns)  # helper only; the bridge itself needs Newton
    pm = ns["published_mode"]
    a = np.ones(N, np.int32)
    assert np.all(pm(a, True) == 0) and np.all(pm(a, False) == 1)
    assert "published_mode(cur[\"mode\"], damping)" in src
    assert "published_mode(cur[\"mode\"], True)" in src


# ------------------------------------------------------------- runner over ZMQ with a fake robot that drops a motor

def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_trips_when_fake_robot_drops_a_motor(tmp_path):
    lb = _load("test_runner_loopback")

    class DroppingRobot(lb.FakeRobot):
        def run(self):
            import threading, time as _t

            def dropper():
                while self.running and self.ticks < 400:
                    _t.sleep(0.001)
                self.echo_mode = False
                self.mode = self.mode.copy(); self.mode[3] = 0
            threading.Thread(target=dropper, daemon=True).start()
            super().run()

    ep = Endpoints(cmd_port=47585, state_port=47586)
    robot = DroppingRobot(ep)
    robot.start()
    try:
        runner = lb._load_runner()
        sp = tmp_path / "summary.json"
        rc = runner.main(["--mode", "hold", "--duration", "1.2", "--passive-s", "0.05", "--move-s", "0.15",
                          "--cmd-port", str(ep.cmd_port), "--state-port", str(ep.state_port),
                          "--connect-timeout-s", "20", "--summary", str(sp)])
    finally:
        robot.running = False
        robot.join(timeout=5)
    s = json.loads(sp.read_text(encoding="utf-8"))
    assert s["status"] == "tripped", s.get("error")
    assert rc == 1
    assert s["final_fsm"] == "passive"
    assert s["motor_mode_gate"]["latched"]["reason"].startswith("motor_mode_dropped:")
    assert any("motor_mode_dropped" in tr["transition"] for tr in s["transitions"])
