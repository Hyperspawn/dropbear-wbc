"""ZMQ round-trip tests for rt/lowcmd and rt/lowstate (CPU only, localhost, non-default ports)."""
from __future__ import annotations

import time

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk.transport import LOWCMD, LOWSTATE, ChannelPublisher, ChannelSubscriber, Endpoints, ZmqSubscriber
from dropbear_wbc.sdk.types import LowCmd, LowState, MotorCmdBlock


def _endpoints(offset: int) -> Endpoints:
    # Ports away from the contract defaults so tests never collide with a running bridge.
    return Endpoints(cmd_port=46555 + offset, state_port=46556 + offset)


def _wait_connected(pub, sub, make_msg, timeout=3.0):
    """PUB/SUB 'slow joiner': publish until the subscriber sees something."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pub.Write(make_msg())
        msg = sub.Read(timeout=0.02)
        if msg is not None:
            return msg
    raise TimeoutError("subscriber never connected")


def test_lowcmd_lowstate_roundtrip_through_zmq():
    ep = _endpoints(0)
    robot_state_pub = ChannelPublisher(LOWSTATE, LowState, endpoints=ep, role="robot")
    robot_cmd_sub = ChannelSubscriber(LOWCMD, LowCmd, endpoints=ep, role="robot")
    client_state_sub = ChannelSubscriber(LOWSTATE, LowState, endpoints=ep, role="client")
    client_cmd_pub = ChannelPublisher(LOWCMD, LowCmd, endpoints=ep, role="client")
    for ch in (robot_state_pub, robot_cmd_sub, client_state_sub, client_cmd_pub):
        ch.Init()
    try:
        state = LowState(tick=41)
        state.motor.q[:] = np.linspace(-1, 1, 22)
        got_state = _wait_connected(robot_state_pub, client_state_sub, lambda: state)
        assert got_state.tick == 41
        np.testing.assert_allclose(got_state.motor.q, np.linspace(-1, 1, 22), rtol=0, atol=1e-7)

        cmd = LowCmd(tick=got_state.tick)
        cmd.motor = MotorCmdBlock.from_arrays(22, q=0.1, kp=50.0, kd=2.0)
        cmd.motor_cmd[3].q = -0.5
        got_cmd = _wait_connected(client_cmd_pub, robot_cmd_sub, lambda: cmd)
        assert got_cmd.tick == 41 and got_cmd.crc == cmd.crc
        assert got_cmd.motor_cmd[3].q == np.float32(-0.5) and got_cmd.motor.kp[0] == 50.0
        assert got_cmd.stamp_ns > 0
    finally:
        for ch in (robot_state_pub, robot_cmd_sub, client_state_sub, client_cmd_pub):
            ch.Close()


def test_latest_message_semantics_and_nonblocking():
    ep = _endpoints(10)
    pub = ChannelPublisher(LOWSTATE, LowState, endpoints=ep, role="robot")
    sub = ChannelSubscriber(LOWSTATE, LowState, endpoints=ep, role="client")
    pub.Init()
    sub.Init()
    try:
        _wait_connected(pub, sub, lambda: LowState(tick=0))
        assert sub.Read(timeout=0.0) is None  # nothing new -> non-blocking None
        before = dict(sub.stats)
        for tick in range(1, 21):
            pub.Write(LowState(tick=tick))
        time.sleep(0.2)
        latest = sub.Read(timeout=0.0)
        assert latest is not None and latest.tick == 20  # newest wins
        burst = sub.stats["received"] - before["received"]
        # A burst above the PUB high-water mark (16) may lose frames in ZMQ; every frame that did
        # arrive except the newest is drained and counted as skipped.
        assert burst >= 2 and sub.stats["skipped"] - before["skipped"] == burst - 1
        t0 = time.perf_counter()
        assert sub.Read(timeout=0.0) is None
        assert time.perf_counter() - t0 < 0.01
    finally:
        pub.Close()
        sub.Close()


def test_handler_thread_and_bad_frames_counted():
    ep = _endpoints(20)
    pub = ChannelPublisher(LOWCMD, LowCmd, endpoints=ep, role="client")
    received: list[int] = []
    sub = ChannelSubscriber(LOWCMD, LowCmd, endpoints=ep, role="robot")
    sub.Init(lambda m: received.append(m.tick))
    pub.Init()
    try:
        deadline = time.monotonic() + 3.0
        while not received and time.monotonic() < deadline:
            pub.Write(LowCmd(tick=7))
            time.sleep(0.02)
        assert received and received[-1] == 7
        pub._pub.publish(b"\x92garbage")  # malformed payload on the right topic
        time.sleep(0.2)
        assert sub.stats["decode_errors"] >= 1
        assert sub.Read().tick == 7
    finally:
        pub.Close()
        sub.Close()


def test_wrong_channel_type_rejected():
    with pytest.raises(TypeError):
        ChannelPublisher(LOWCMD, LowState)


def test_topic_filtering():
    ep = _endpoints(30)
    # A stray publisher on the state port with a different topic must not be received.
    from dropbear_wbc.sdk.transport import ZmqPublisher
    stray = ZmqPublisher(ep.for_channel(LOWSTATE), "rt/other", bind=True)
    sub = ZmqSubscriber(ep.for_channel(LOWSTATE), LOWSTATE, bind=False)
    try:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            stray.publish(b"x")
            assert sub.recv_latest(timeout_ms=10) is None
    finally:
        stray.close()
        sub.close()
