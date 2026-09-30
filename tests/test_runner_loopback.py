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
from dropbear_wbc.sdk.types import LowCmd, LowState

ROOT = Path(__file__).resolve().parents[1]
EP = Endpoints(cmd_port=47555, state_port=47556)


class FakeRobot(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.q = np.zeros(22)
        self.q_des = None
        self.ticks = 0
        self.cmds = 0

    def run(self):
        pub = ChannelPublisher(LOWSTATE, LowState, endpoints=EP, role="robot")
        sub = ChannelSubscriber(LOWCMD, LowCmd, endpoints=EP, role="robot")
        pub.Init()
        sub.Init()
        try:
            while self.running:
                cmd = sub.Read(timeout=0.0)
                if cmd is not None:
                    self.q_des = cmd.motor.q.astype(float)
                    self.cmds += 1
                if self.q_des is not None:  # paused until the first command, like the bridge
                    self.q += 0.05 * (self.q_des - self.q)
                    self.ticks += 1
                st = fx.low_state(self.ticks, self.q)
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
