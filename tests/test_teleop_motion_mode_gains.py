"""tools/teleop_arm.py must only put gains inside the MIT motion-mode field on the wire (kp 0..500, kd 0..5).

The old defaults (legs DEFAULT x 2 -> kd 10/10/10/24/8/8, elbow kp 600 / kd 10, exit damping with the raw knee
kd 12) made a field-checking HAL (item 156) refuse every teleop LowCmd, the damping exit included.
"""
from __future__ import annotations

import json
import threading

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.robots.hw_motor_specs import MOTION_MODE_LIMITS
from dropbear_wbc.sdk.transport import Endpoints

import test_teleop_loopback as lb

KP_MAX, KD_MAX = MOTION_MODE_LIMITS["kp_max"], MOTION_MODE_LIMITS["kd_max"]


def test_default_gains_inside_field_and_reported():
    tool = lb._load_tool()
    args = tool.parse_args([])
    kp_req, kd_req = tool.requested_gains(args)
    assert kd_req.max() > KD_MAX and kp_req.max() > KP_MAX  # the legacy request really is out of field
    kp, kd, clamped = tool.gains(args)
    assert kp.max() <= KP_MAX and kd.max() <= KD_MAX and kp.min() >= 0 and kd.min() >= 0
    moved = {(c["motor"], c["field"]): (c["before"], c["after"]) for c in clamped}
    assert moved[("LL_knee_actuator_joint", "kd")] == (24.0, KD_MAX)
    assert moved[("LH_elbow_joint", "kp")] == (600.0, KP_MAX)
    assert moved[("RH_elbow_joint", "kd")] == (10.0, KD_MAX)
    assert len(clamped) == int(np.sum(kp != kp_req) + np.sum(kd != kd_req))
    # values already inside the field are passed through untouched
    keep = (kp_req <= KP_MAX)
    assert np.array_equal(kp[keep], kp_req[keep])


class FieldCheckRobot(lb.FakeArmRobot):
    """Fake robot that records every LowCmd's gains (what a field-checking HAL would accept or refuse)."""

    def __init__(self, ik, ep):
        super().__init__(ik)
        self.ep = ep
        self.n_out_of_field = 0
        self.max_kp = self.max_kd = 0.0
        self.last_kp = self.last_kd = None

    def run(self):  # same loop as the parent, on our own ports, plus the gain check
        from dropbear_wbc.sdk.transport import LOWCMD, LOWSTATE, ChannelPublisher, ChannelSubscriber
        from dropbear_wbc.sdk.types import LowCmd, LowState
        import time
        pub = ChannelPublisher(LOWSTATE, LowState, endpoints=self.ep, role="robot")
        sub = ChannelSubscriber(LOWCMD, LowCmd, endpoints=self.ep, role="robot")
        pub.Init()
        sub.Init()
        try:
            while self.running:
                cmd = sub.Read(timeout=0.0)
                if cmd is not None:
                    kp, kd = cmd.motor.kp.astype(float), cmd.motor.kd.astype(float)
                    self.max_kp, self.max_kd = max(self.max_kp, kp.max()), max(self.max_kd, kd.max())
                    self.n_out_of_field += int(kp.max() > KP_MAX + 1e-6 or kd.max() > KD_MAX + 1e-6)
                    self.last_kp, self.last_kd = kp, kd
                    self.q_des = cmd.motor.q.astype(float)
                    self.cmds += 1
                if self.q_des is not None:
                    self.q += 0.3 * (self.q_des - self.q)
                    self.ticks += 1
                pub.Write(self.state())
                time.sleep(0.0005)
        finally:
            pub.Close()
            sub.Close()


@pytest.mark.skipif(not lb.DEFAULT_CALIBRATION.exists(), reason="no semantic calibration JSON")
def test_every_teleop_lowcmd_inside_field(tmp_path):
    from dropbear_wbc.teleop.arm_ik import DropbearArmIK

    ep = Endpoints(cmd_port=48655, state_port=48656)
    robot = FieldCheckRobot(DropbearArmIK(), ep)
    robot.start()
    try:
        tool = lb._load_tool()
        summary_path = tmp_path / "summary.json"
        rc = tool.main(["--device", "scripted", "--duration", "1.0", "--ramp-s", "0.3", "--move-in-s", "0.2",
                        "--settle-s", "0.1", "--home-s", "0.2", "--cmd-port", str(ep.cmd_port),
                        "--state-port", str(ep.state_port), "--connect-timeout-s", "20",
                        "--summary", str(summary_path)])
    finally:
        robot.running = False
        robot.join(timeout=5)
    s = json.loads(summary_path.read_text())
    assert rc == 0, s.get("traceback")
    assert robot.cmds > 20
    assert robot.n_out_of_field == 0, (robot.max_kp, robot.max_kd)
    assert robot.last_kp.max() == 0.0 and 0.0 < robot.last_kd.max() <= KD_MAX  # damping exit is wire-legal
    assert s["gains"]["clamped_to_motion_mode_field"]
