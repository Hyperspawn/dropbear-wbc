"""CPU-only end-to-end test of tools/policy_runner.py over ZMQ against a fake robot (no simulator).

The fake robot binds rt/lowstate + rt/lowcmd on test ports, advances a tick
counter as fast as it can and moves each motor toward the commanded position
(first-order lag), publishing a privileged sim block. The real runner ``main()``
runs Passive -> MoveToDefault -> Hold -> Policy with the synthetic BeyondMimic
ONNX (full 125-dim layout with sim-only terms) from ``deploy_fixtures``.
"""
from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import deploy_fixtures as fx
import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk.transport import LOWCMD, LOWSTATE, ChannelPublisher, ChannelSubscriber, Endpoints
from dropbear_wbc.sdk.types import LowCmd, LowState, MotorMode

ROOT = Path(__file__).resolve().parents[1]
EP = Endpoints(cmd_port=47555, state_port=47556)


class FakeRobot(threading.Thread):
    def __init__(self, endpoints: Endpoints = EP, *, echo_mode: bool = True, tick_offset: int = 0,
                 freeze_after: int | None = None, restart_after: int | None = None):
        super().__init__(daemon=True)
        self.endpoints = endpoints
        self.running = True
        self.q = np.zeros(22)
        self.q_des = None
        self.ticks = 0
        self.cmds = 0
        self.echo_mode = echo_mode
        self.tick_offset = tick_offset
        self.freeze_after = freeze_after    # item 263: keep publishing, stop advancing the tick
        self.restart_after = restart_after  # item 263: tick jumps back to 0 once (producer restart)
        self.mode = np.zeros(22, dtype=np.uint8)
        self.last_kp = None

    def run(self):
        pub = ChannelPublisher(LOWSTATE, LowState, endpoints=self.endpoints, role="robot")
        sub = ChannelSubscriber(LOWCMD, LowCmd, endpoints=self.endpoints, role="robot")
        pub.Init()
        sub.Init()
        try:
            while self.running:
                cmd = sub.Read(timeout=0.0)
                if cmd is not None:
                    self.q_des = cmd.motor.q.astype(float)
                    self.last_kp = np.asarray(cmd.motor.kp, dtype=float).copy()
                    self.cmds += 1
                    if self.echo_mode:
                        self.mode = np.asarray(cmd.motor.mode, dtype=np.uint8).copy()
                if self.q_des is not None:  # paused until the first command, like the bridge
                    self.q += 0.05 * (self.q_des - self.q)
                    if self.freeze_after is None or self.ticks < self.freeze_after:
                        self.ticks += 1
                    if self.restart_after is not None and self.ticks >= self.restart_after:
                        self.ticks, self.restart_after = 0, None
                st = fx.low_state((self.ticks + self.tick_offset) & 0xFFFFFFFF, self.q)
                st.motor.mode[:] = self.mode
                pub.Write(st)
                time.sleep(0.0002)
        finally:
            pub.Close()
            sub.Close()


def _load_runner():
    spec = importlib.util.spec_from_file_location("policy_runner", ROOT / "tools" / "policy_runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_policy_mode_over_zmq(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    fx.write_beyondmimic_onnx(tmp_path / "policy.onnx", obs_dim=sum(t["dim"] for t in fx.OBS_TERMS_FULL))
    fx.write_sidecar(tmp_path / "policy.json", "policy.onnx", terms=fx.OBS_TERMS_FULL)
    robot = FakeRobot()
    robot.start()
    try:
        runner = _load_runner()
        summary_path = tmp_path / "summary.json"
        rc = runner.main(["--mode", "policy", "--sidecar", str(tmp_path / "policy.json"), "--allow-privileged",
                          "--duration", "1.6", "--passive-s", "0.1", "--move-s", "0.5", "--hold-s", "0.3",
                          "--cmd-port", str(EP.cmd_port), "--state-port", str(EP.state_port),
                          "--connect-timeout-s", "20", "--summary", str(summary_path),
                          "--log", str(tmp_path / "steps.jsonl")])
    finally:
        robot.running = False
        robot.join(timeout=5)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0, summary.get("error")
    names = [t["transition"] for t in summary["transitions"]]
    assert names[:3] == ["passive->move_to_default", "move_to_default->hold", "hold->policy"], names
    assert summary["privileged_terms"] == ["motion_anchor_pos_b", "base_lin_vel"]
    steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    policy_steps = [s for s in steps if s["fsm"] == "policy"]
    assert len(policy_steps) >= 20 and all(len(s["action"]) == 22 for s in policy_steps)
    # Sim clock: one step per 10 ticks (0.02 s / 0.002 s), never fewer.
    assert summary["ticks_per_step"]["expected"] == 10 and summary["ticks_per_step"]["mean"] >= 10
    assert robot.cmds >= len(steps)


def test_runner_fails_when_motor_mode_never_enable(tmp_path):
    """Item 156: summary must not report executed when LowState.mode stays 0 (HAL refuse / never armed)."""
    ep = Endpoints(cmd_port=47575, state_port=47576)
    robot = FakeRobot(ep, echo_mode=False)  # mode stays 0
    robot.start()
    try:
        runner = _load_runner()
        summary_path = tmp_path / "summary.json"
        rc = runner.main(["--mode", "hold", "--duration", "0.4", "--passive-s", "0.05", "--move-s", "0.15",
                          "--cmd-port", str(ep.cmd_port), "--state-port", str(ep.state_port),
                          "--connect-timeout-s", "20", "--summary", str(summary_path)])
    finally:
        robot.running = False
        robot.join(timeout=5)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert summary["status"] == "failed"
    assert summary["motor_mode_max"] == 0
    assert summary["motor_mode_enable_steps"] == 0
    assert "mode never reached" in (summary.get("error") or "")


def test_runner_hold_mode_passes_with_enable_echo(tmp_path):
    ep = Endpoints(cmd_port=47565, state_port=47566)
    robot = FakeRobot(ep, echo_mode=True)
    robot.start()
    try:
        runner = _load_runner()
        summary_path = tmp_path / "summary.json"
        rc = runner.main(["--mode", "hold", "--duration", "0.5", "--passive-s", "0.05", "--move-s", "0.2",
                          "--cmd-port", str(ep.cmd_port), "--state-port", str(ep.state_port),
                          "--connect-timeout-s", "20", "--summary", str(summary_path)])
    finally:
        robot.running = False
        robot.join(timeout=5)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0, summary.get("error")
    assert summary["status"] == "executed"
    assert summary["motor_mode_max"] == int(MotorMode.ENABLE)
    assert summary["motor_mode_enable_steps"] >= 1
    # knees clamped to motion-mode field (DEFAULT_KD was 12)
    assert max(summary["config"]["hold_kd"]) <= 5.0 + 1e-9
    assert summary["config"]["hold_kd"][3] == 5.0
    assert summary["config"]["hold_kd"][9] == 5.0


def test_runner_sim_clock_survives_the_uint32_tick_wrap(tmp_path):
    """Item 258: started 150 ticks before the uint32 wrap, the sim-clock scheduler compared raw ticks
    (``st.tick >= last_tick + decim``) and spun in its keepalive loop forever after the wrap."""
    ep = Endpoints(cmd_port=47585, state_port=47586)
    robot = FakeRobot(ep, echo_mode=True, tick_offset=(1 << 32) - 150)
    robot.start()
    try:
        runner = _load_runner()
        summary_path = tmp_path / "summary.json"
        rc = runner.main(["--mode", "hold", "--duration", "0.6", "--passive-s", "0.05", "--move-s", "0.2",
                          "--cmd-port", str(ep.cmd_port), "--state-port", str(ep.state_port),
                          "--connect-timeout-s", "20", "--summary", str(summary_path)])
    finally:
        robot.running = False
        robot.join(timeout=5)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0, summary.get("error")
    assert robot.ticks > 150  # the run crossed the wrap
    assert summary["ticks_per_step"]["mean"] >= 10 and summary["steps"] >= 25


def _run_stalled(tmp_path, ep, robot, stall_s, duration=2.0):
    robot.start()
    t0 = time.perf_counter()
    try:
        runner = _load_runner()
        summary_path = tmp_path / "summary.json"
        rc = runner.main(["--mode", "hold", "--duration", str(duration), "--passive-s", "0.05", "--move-s", "0.2",
                          "--cmd-port", str(ep.cmd_port), "--state-port", str(ep.state_port),
                          "--connect-timeout-s", "20", "--tick-stall-s", str(stall_s),
                          "--summary", str(summary_path)])
    finally:
        robot.running = False
        robot.join(timeout=5)
    return rc, json.loads(summary_path.read_text(encoding="utf-8")), time.perf_counter() - t0


def test_runner_sim_clock_aborts_on_frozen_tick(tmp_path):
    """Item 263: a producer that keeps publishing one tick used to spin the keepalive loop forever
    (frames keep last_rx fresh, so --state-timeout-s never fires)."""
    ep = Endpoints(cmd_port=47587, state_port=47588)
    robot = FakeRobot(ep, freeze_after=100)
    rc, summary, wall = _run_stalled(tmp_path, ep, robot, 0.5)
    assert rc != 0
    assert "TickStallError" in summary["error"] and "frozen at tick" in summary["error"]
    assert wall < 10
    assert summary["abort_damping_sent"] is True
    assert robot.last_kp is not None and float(np.abs(robot.last_kp).max()) == 0.0  # damping was the last command


def test_runner_sim_clock_aborts_on_producer_restart(tmp_path):
    """Item 263: a restart (tick back to 0) stalls the scheduler until the tick passes the old value
    (3000 ticks here, about 1.5 s of catch-up at the fake robot rate; 0.5 s stall limit)."""
    ep = Endpoints(cmd_port=47589, state_port=47590)
    rc, summary, wall = _run_stalled(tmp_path, ep, FakeRobot(ep, restart_after=3000), 0.5, duration=8.0)
    assert rc != 0
    assert "TickStallError" in summary["error"] and "producer restart" in summary["error"]
    assert wall < 10
