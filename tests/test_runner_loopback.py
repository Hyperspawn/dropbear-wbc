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
    def __init__(self, endpoints: Endpoints = EP, *, echo_mode: bool = True):
        super().__init__(daemon=True)
        self.endpoints = endpoints
        self.running = True
        self.q = np.zeros(22)
        self.q_des = None
        self.ticks = 0
        self.cmds = 0
        self.echo_mode = echo_mode
        self.mode = np.zeros(22, dtype=np.uint8)

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
                    self.cmds += 1
                    if self.echo_mode:
                        self.mode = np.asarray(cmd.motor.mode, dtype=np.uint8).copy()
                if self.q_des is not None:  # paused until the first command, like the bridge
                    self.q += 0.05 * (self.q_des - self.q)
                    self.ticks += 1
                st = fx.low_state(self.ticks, self.q)
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
