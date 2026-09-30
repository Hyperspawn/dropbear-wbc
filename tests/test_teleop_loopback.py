"""CPU-only end-to-end test of tools/teleop_arm.py over ZMQ against a fake robot (no simulator).

The fake robot binds rt/lowstate + rt/lowcmd on test ports, advances ticks as fast as it can, moves every motor
toward its commanded position (first-order lag) and publishes a privileged sim block with the hand-plate poses
computed from the IK model (so "simulated" wrist == FK of the measured joints). The real teleop ``main()`` runs
MOVE_IN -> SETTLE -> TELEOP (scripted figure-8) -> HOME and records a LeRobot-v2-like session.
"""
from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.kinematics.semantic import DEFAULT_CALIBRATION
from dropbear_wbc.sdk import motors
from dropbear_wbc.sdk.transport import LOWCMD, LOWSTATE, ChannelPublisher, ChannelSubscriber, Endpoints
from dropbear_wbc.sdk.types import IMUState, LowCmd, LowState, MotorStateBlock, SimState

ROOT = Path(__file__).resolve().parents[1]
EP = Endpoints(cmd_port=48555, state_port=48556)
pytestmark = pytest.mark.skipif(not DEFAULT_CALIBRATION.exists(), reason="no semantic calibration JSON")


class FakeArmRobot(threading.Thread):
    def __init__(self, ik):
        super().__init__(daemon=True)
        self.ik = ik
        self.running = True
        self.q = np.asarray(motors.DEFAULT_POS, dtype=float).copy()
        self.q_des = None
        self.ticks = 0
        self.cmds = 0
        self.max_abs_tau_ff = 0.0

    def state(self) -> LowState:
        st = LowState(tick=self.ticks, stamp_ns=time.perf_counter_ns())
        st.imu = IMUState(quat_wxyz=np.array([1, 0, 0, 0], np.float32), accel=np.array([0, 0, 9.81], np.float32))
        st.motor = MotorStateBlock.zeros(22)
        st.motor.q[:] = self.q.astype(np.float32)
        sem = self.ik.motor_to_semantic_arms_fast(self.q)
        pos = []
        for k, side in enumerate(("left", "right")):
            pos.append(self.ik.fk(side, sem[5 * k:5 * k + 5])[:3, 3] + self.ik.torso_origin_root)
        st.sim = SimState(time_s=self.ticks * 0.002, root_pos_w=np.zeros(3, np.float32),
                          root_quat_w=np.array([1, 0, 0, 0], np.float32),
                          body_names=("LH_shoulder_ex_al_interface_1", "RH_shoulder_ex_al_interface_1"),
                          body_pos_w=np.asarray(pos, np.float32),
                          body_quat_w=np.tile(np.array([1, 0, 0, 0], np.float32), (2, 1)))
        return st

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
                    self.max_abs_tau_ff = max(self.max_abs_tau_ff, float(np.abs(cmd.motor.tau).max()))
                    self.cmds += 1
                if self.q_des is not None:  # paused until the first command, like the bridge
                    self.q += 0.3 * (self.q_des - self.q)
                    self.ticks += 1
                pub.Write(self.state())
                time.sleep(0.0005)
        finally:
            pub.Close()
            sub.Close()


def _load_tool():
    spec = importlib.util.spec_from_file_location("teleop_arm", ROOT / "tools" / "teleop_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_teleop_scripted_session_over_zmq(tmp_path):
    from dropbear_wbc.teleop.arm_ik import DropbearArmIK

    ik = DropbearArmIK()
    robot = FakeArmRobot(ik)
    robot.start()
    try:
        tool = _load_tool()
        rec = tmp_path / "session"
        summary_path = tmp_path / "summary.json"
        rc = tool.main(["--device", "scripted", "--duration", "3.0", "--ramp-s", "1.0", "--move-in-s", "0.3",
                        "--settle-s", "0.2", "--home-s", "0.3", "--cmd-port", str(EP.cmd_port),
                        "--state-port", str(EP.state_port), "--connect-timeout-s", "20", "--record", str(rec),
                        "--summary", str(summary_path), "--task", "loopback test"])
    finally:
        robot.running = False
        robot.join(timeout=5)
    s = json.loads(summary_path.read_text())
    assert rc == 0, s.get("traceback")
    assert s["status"] == "ok"
    assert [tr["to"] for tr in s["transitions"]] == ["settle", "teleop", "home_done"]
    info = json.loads((rec / "meta" / "info.json").read_text())
    n = info["total_frames"]
    assert 140 <= n <= 152, n  # 3 s at 50 Hz of simulated time
    assert info["features"]["observation.state"]["shape"] == [10]
    mod = json.loads((rec / "meta" / "modality.json").read_text())
    assert mod["state"]["right_arm"] == {"start": 5, "end": 10} and mod["action"]["left_arm"] == {"start": 0, "end": 5}
    assert (rec / "meta" / "episodes.jsonl").exists() and (rec / "meta" / "tasks.jsonl").exists()
    assert (rec / "extras" / "episode_000000.npz").exists()
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None
    if pq is not None:
        tab = pq.read_table(rec / "data" / "chunk-000" / "episode_000000.parquet")
        assert tab.num_rows == n
        assert len(tab.column("observation.state")[0]) == 10 and len(tab.column("action")[0]) == 10
    an = s["analysis"]
    for side in ("left", "right"):
        # fake robot: first-order lag, sim wrist == FK -> small tracking error, zero model error
        assert an[side]["wrist_tracking_fk_vs_target"]["mean_mm"] < 20.0
        assert an[side]["model_sim_vs_fk"]["max_mm"] < 0.1
        assert an[side]["target_path_length_m"] > 0.05
    assert robot.max_abs_tau_ff > 0.1  # gravity feed-forward was sent
