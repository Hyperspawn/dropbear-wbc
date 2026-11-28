"""tools/teleop_arm.py must only put gains inside the MIT motion-mode field on the wire (kp 0..500, kd 0..5).

The old defaults (legs DEFAULT x 2 -> kd 10/10/10/24/8/8, elbow kp 600 / kd 10, exit damping with the raw knee
kd 12) made a field-checking HAL (item 156) refuse every teleop LowCmd, the damping exit included.

The fake robot binds OS-assigned ports and hands them to the client: the old fixed ports 28655/28656 made the test
wait 20 s and fail with a misleading "no LowState received" whenever another process (a second checkout's suite on
the shared box) held either port, because the robot thread died on EADDRINUSE without failing the test.
"""
from __future__ import annotations

import json

import zmq

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
    """Fake robot that records every LowCmd's gains (what a field-checking HAL would accept or refuse).

    Binds first like every fake robot (tests/loopback_ports.py): ``ep`` defaults to port 0 on both channels
    (OS-assigned free ports, held from bind to close, so no check-then-bind race), and :meth:`wait_bound` returns
    the endpoints actually bound or raises the bind error at once instead of letting the client time out.
    """

    def __init__(self, ik, ep=None):
        super().__init__(ik, ep)
        self.n_out_of_field = 0
        self.max_kp = self.max_kd = 0.0
        self.last_kp = self.last_kd = None

    def run(self):  # same loop as the parent, on our own ports, plus the gain check
        import time
        channels = self.bind()
        if channels is None:
            return
        pub, sub = channels
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


def _hold_port() -> tuple[zmq.Socket, int]:
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind("tcp://127.0.0.1:*")
    return sock, int(sock.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])


def test_fake_robot_binds_os_assigned_ports_even_when_the_old_fixed_ports_are_taken():
    from dropbear_wbc.teleop.arm_ik import DropbearArmIK

    held = []
    for port in (28655, 28656):  # the old fixed ports; a concurrent run of the old test would hold them
        sock = zmq.Context.instance().socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.bind(f"tcp://127.0.0.1:{port}")
        except zmq.ZMQError:
            sock.close(linger=0)  # already taken by someone else: the point of the test holds anyway
            continue
        held.append(sock)
    robot = FieldCheckRobot(DropbearArmIK())
    robot.start()
    try:
        ep = robot.wait_bound()
        assert robot.bind_error is None
        assert ep.cmd_port != ep.state_port
        assert {ep.cmd_port, ep.state_port}.isdisjoint({0, 28655, 28656})
    finally:
        robot.running = False
        robot.join(timeout=5)
        for sock in held:
            sock.close(linger=0)
    assert not robot.is_alive()


def test_fake_robot_bind_failure_is_raised_at_once_not_a_client_timeout():
    import time

    from dropbear_wbc.teleop.arm_ik import DropbearArmIK

    sock, port = _hold_port()
    try:
        robot = FieldCheckRobot(DropbearArmIK(), Endpoints(cmd_port=port, state_port=0))
        robot.start()
        t0 = time.perf_counter()
        with pytest.raises(RuntimeError, match="could not bind"):
            robot.wait_bound()
        assert time.perf_counter() - t0 < FieldCheckRobot.BIND_TIMEOUT_S
        assert isinstance(robot.bind_error, zmq.ZMQError)
        robot.join(timeout=5)
        assert not robot.is_alive()
    finally:
        sock.close(linger=0)


def test_two_fake_robots_get_distinct_ports():
    from dropbear_wbc.teleop.arm_ik import DropbearArmIK

    robots = [FieldCheckRobot(DropbearArmIK()) for _ in range(2)]
    for r in robots:
        r.start()
    try:
        eps = [r.wait_bound() for r in robots]
        ports = [p for ep in eps for p in (ep.cmd_port, ep.state_port)]
        assert len(set(ports)) == 4
    finally:
        for r in robots:
            r.running = False
            r.join(timeout=5)


@pytest.mark.skipif(not lb.DEFAULT_CALIBRATION.exists(), reason="no semantic calibration JSON")
def test_every_teleop_lowcmd_inside_field(tmp_path):
    from dropbear_wbc.teleop.arm_ik import DropbearArmIK

    robot = FieldCheckRobot(DropbearArmIK())  # OS-assigned ports: no clash with a concurrent run
    robot.start()
    try:
        ep = robot.wait_bound()
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
