"""tests/loopback_ports.py: fake robots bind OS-assigned ports first and fail fast (item 398)."""
from __future__ import annotations

import time

import pytest
import zmq

import loopback_ports as lp
import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk.transport import LOWCMD, LOWSTATE, ChannelPublisher, ChannelSubscriber, Endpoints
from dropbear_wbc.sdk.types import LowCmd, LowState

# Old fixed literals of the converted modules (test_runner_loopback EP, test_teleop_loopback EP).
OLD_PORTS = (27555, 27556, 28555, 28556)


class EchoRobot(lp.BindFirstRobot):
    """Publishes LowState(tick=n) and remembers the last LowCmd tick."""

    def __init__(self, endpoints=None):
        super().__init__(endpoints)
        self.last_cmd_tick = None

    def run(self):
        channels = self.bind()
        if channels is None:
            return
        pub, sub = channels
        n = 0
        try:
            while self.running:
                cmd = sub.Read(timeout=0.0)
                if cmd is not None:
                    self.last_cmd_tick = cmd.tick
                n += 1
                pub.Write(LowState(tick=n))
                time.sleep(0.002)
        finally:
            pub.Close()
            sub.Close()


def _hold(port: int = 0):
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.bind(f"tcp://127.0.0.1:{port or '*'}")
    except zmq.ZMQError:
        sock.close(linger=0)  # already held by someone else: what the test needs anyway
        return None, port
    return sock, int(sock.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])


def _stop(*robots):
    for r in robots:
        r.running = False
        r.join(timeout=5)


def test_any_port_is_port_zero():
    assert (lp.ANY_PORT.cmd_port, lp.ANY_PORT.state_port) == (0, 0)


def test_bind_robot_channels_reports_the_bound_ports():
    pub, sub, ep = lp.bind_robot_channels()
    try:
        assert ep.cmd_port > 0 and ep.state_port > 0 and ep.cmd_port != ep.state_port
        assert ep.cmd_port == lp.bound_port(sub) and ep.state_port == lp.bound_port(pub)
        assert ep.host == lp.ANY_PORT.host
    finally:
        pub.Close()
        sub.Close()


def test_bound_port_refuses_an_unbound_socket():
    sock = zmq.Context.instance().socket(zmq.PUB)
    try:
        holder = type("H", (), {"_sock": sock})()
        with pytest.raises(RuntimeError, match="not bound"):
            lp.bound_port(holder)
    finally:
        sock.close(linger=0)


def test_robot_round_trip_while_the_old_fixed_ports_are_held():
    held = [s for s, _ in (_hold(p) for p in OLD_PORTS) if s is not None]
    robot = EchoRobot()
    try:
        ep = robot.start_bound()
        assert {ep.cmd_port, ep.state_port}.isdisjoint({0, *OLD_PORTS})
        state_sub = ChannelSubscriber(LOWSTATE, LowState, endpoints=ep, role="client")
        cmd_pub = ChannelPublisher(LOWCMD, LowCmd, endpoints=ep, role="client")
        state_sub.Init()
        cmd_pub.Init()
        try:
            deadline, st = time.monotonic() + 5.0, None
            while time.monotonic() < deadline and (st is None or robot.last_cmd_tick != 77):
                st = state_sub.Read(timeout=0.02) or st
                cmd_pub.Write(LowCmd(tick=77))
            assert st is not None and st.tick > 0  # state port is the robot's PUB
            assert robot.last_cmd_tick == 77  # cmd port is the robot's SUB
        finally:
            state_sub.Close()
            cmd_pub.Close()
    finally:
        _stop(robot)
        for s in held:
            s.close(linger=0)


@pytest.mark.parametrize("channel", ["cmd", "state"])
def test_bind_failure_is_raised_at_once(channel):
    sock, port = _hold()
    try:
        ep = Endpoints(cmd_port=port, state_port=0) if channel == "cmd" else Endpoints(cmd_port=0, state_port=port)
        robot = EchoRobot(ep)
        t0 = time.perf_counter()
        robot.start()
        with pytest.raises(RuntimeError, match="could not bind"):
            robot.wait_bound()
        assert time.perf_counter() - t0 < 2.0  # not the client's 20 s connect timeout
        assert isinstance(robot.bind_error, zmq.ZMQError) and robot.bind_error.errno == zmq.EADDRINUSE
        robot.join(timeout=5)
        assert not robot.is_alive()
    finally:
        sock.close(linger=0)


def test_failed_bind_closes_both_channels(monkeypatch):
    closed = []

    class Pub(ChannelPublisher):
        def Close(self):
            closed.append("pub")
            super().Close()

    class Sub(ChannelSubscriber):
        def Close(self):
            closed.append("sub")
            super().Close()

    monkeypatch.setattr(lp, "ChannelPublisher", Pub)
    monkeypatch.setattr(lp, "ChannelSubscriber", Sub)
    sock, port = _hold()
    try:
        with pytest.raises(zmq.ZMQError):
            lp.bind_robot_channels(Endpoints(cmd_port=port, state_port=0))  # state PUB binds, cmd SUB fails
    finally:
        sock.close(linger=0)
    assert sorted(closed) == ["pub", "sub"]


def test_robot_that_never_binds_times_out_and_is_stopped():
    class Mute(lp.BindFirstRobot):
        def run(self):
            while self.running:
                time.sleep(0.01)

    robot = Mute()
    robot.start()
    t0 = time.perf_counter()
    with pytest.raises(TimeoutError, match="did not bind"):
        robot.wait_bound(timeout=0.2)
    assert 0.15 < time.perf_counter() - t0 < 2.0
    assert robot.running is False
    robot.join(timeout=5)
    assert not robot.is_alive()


def test_concurrent_robots_get_distinct_ports():
    robots = [EchoRobot() for _ in range(4)]
    try:
        eps = [r.start_bound() for r in robots]
        ports = [p for ep in eps for p in (ep.cmd_port, ep.state_port)]
        assert len(set(ports)) == 8 and min(ports) > 0
    finally:
        _stop(*robots)
