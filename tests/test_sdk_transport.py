"""ZMQ round-trip tests for rt/lowcmd and rt/lowstate (CPU only, localhost, non-default ports)."""
from __future__ import annotations

import subprocess
import time
import warnings

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk.transport import LOWCMD, LOWSTATE, ChannelPublisher, ChannelSubscriber, Endpoints, ZmqSubscriber
from dropbear_wbc.sdk.types import LowCmd, LowState, MotorCmdBlock


HOST_STALL_S = 1.0  # poll-loop gap that marks a starved host, not a transport fault
RETRY_OFFSET = 90  # spare _endpoints offset for the single stall retry in the burst test


def _endpoints(offset: int) -> Endpoints:
    # Ports away from the contract defaults so tests never collide with a running bridge.
    return Endpoints(cmd_port=26555 + offset, state_port=26556 + offset)


def _tcp_states(port: int) -> list[str]:
    """TCP sockets on 127.0.0.1:port (Linux `ss`); empty where ss is unavailable."""
    try:
        out = subprocess.run(["ss", "-tanH", f"( sport = :{port} or dport = :{port} )"],
                             capture_output=True, text=True, timeout=2.0).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [" ".join(line.split()[:1] + line.split()[3:5]) for line in out.splitlines() if line.strip()]


class NotConnected(TimeoutError):
    """Slow-joiner wait ran out. `host_stall_s` is the longest gap between our own poll iterations."""

    def __init__(self, msg: str, host_stall_s: float):
        super().__init__(msg)
        self.host_stall_s = host_stall_s


def _wait_connected(pub, sub, make_msg, timeout=3.0, port: int | None = None):
    """PUB/SUB 'slow joiner': publish until the subscriber sees something.

    On timeout the error carries what is needed to classify it (notes/309): poll count, the longest gap
    between poll iterations (a host stall starves the loop; 3 s budget at 20 ms polls is ~150 polls) and
    the TCP state of the port (no ESTABLISHED pair = TCP never connected; ESTABLISHED = subscription or
    delivery failure inside ZMQ, a real transport bug)."""
    t0 = last = time.monotonic()
    deadline = t0 + timeout
    polls, max_gap = 0, 0.0
    while time.monotonic() < deadline:
        pub.Write(make_msg())
        msg = sub.Read(timeout=0.02)
        now = time.monotonic()
        max_gap, last, polls = max(max_gap, now - last), now, polls + 1
        if msg is not None:
            return msg
    tcp = _tcp_states(port) if port is not None else []
    raise NotConnected(f"subscriber never connected: {polls} polls in {last - t0:.2f} s, max poll gap {max_gap:.3f} s, "
                       f"port {port} tcp={tcp}, sub stats={dict(sub.stats)}", max_gap)


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
        # CONFLATE keeps only the newest frame on both sides: one frame reaches the reader, none skipped.
        assert burst == 1 and sub.stats["skipped"] == before["skipped"]
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


def test_latest_wins_is_deterministic_on_bursts():
    """20-frame burst > SNDHWM: the newest frame must win every time (was 62-74 % before CONFLATE, notes/202).

    The property under test is burst delivery, not connection setup. A connect timeout is retried once on a
    fresh socket pair on a spare port (closing a ZMQ socket unbinds asynchronously, so rebinding the same port
    can hit EADDRINUSE) only when our own poll loop was starved for > 1 s (host stall, CI run 41608080261);
    any other connect failure, or a second stall anywhere in the test, fails with the diagnostic (notes/309)."""
    retried = []
    for i in range(25):
        for attempt in (0, 1):
            ep = _endpoints(40 + i if attempt == 0 else RETRY_OFFSET)
            pub = ChannelPublisher(LOWSTATE, LowState, endpoints=ep, role="robot")
            sub = ChannelSubscriber(LOWSTATE, LowState, endpoints=ep, role="client")
            pub.Init()
            sub.Init()
            try:
                try:
                    _wait_connected(pub, sub, lambda: LowState(tick=0), port=ep.state_port)
                except NotConnected as e:
                    if attempt == 0 and not retried and e.host_stall_s > HOST_STALL_S:
                        retried.append((i, str(e)))
                        continue
                    raise
                for tick in range(1, 21):
                    pub.Write(LowState(tick=tick))
                time.sleep(0.05)
                latest = sub.Read(timeout=0.0)
                assert latest is not None and latest.tick == 20, (i, latest and latest.tick)
                break
            finally:
                pub.Close()
                sub.Close()
    if retried:
        warnings.warn(f"connect retried after host stall: {retried}")


def test_reader_stall_does_not_return_stale_state():
    """Publisher at 500 Hz, reader stalls 400 ms: the next Read must be fresh (was 286 ms old, notes/202)."""
    import threading

    ep = _endpoints(80)
    pub = ChannelPublisher(LOWSTATE, LowState, endpoints=ep, role="robot")
    sub = ChannelSubscriber(LOWSTATE, LowState, endpoints=ep, role="client")
    pub.Init()
    sub.Init()
    stop = threading.Event()
    sent = [0]

    def run():
        while not stop.is_set():
            sent[0] += 1
            pub.Write(LowState(tick=sent[0]))
            time.sleep(0.002)

    try:
        _wait_connected(pub, sub, lambda: LowState(tick=0))
        th = threading.Thread(target=run, daemon=True)
        th.start()
        time.sleep(0.1)
        sub.Read(timeout=0.0)
        time.sleep(0.4)  # reader stall: > (SNDHWM + RCVHWM) / 500 Hz
        st = sub.Read(timeout=0.0)
        assert st is not None
        age_ms = (time.perf_counter_ns() - st.stamp_ns) * 1e-6
        assert age_ms < 50.0 and sent[0] - st.tick <= 25, (age_ms, sent[0] - st.tick)
    finally:
        stop.set()
        pub.Close()
        sub.Close()


class _FakePub:
    def Write(self, msg):
        pass


class _FakeSub:
    def __init__(self, stall_s: float):
        self.stall_s, self.stats = stall_s, {"received": 0, "skipped": 0}

    def Read(self, timeout=0.0):
        time.sleep(self.stall_s or timeout)
        self.stall_s = 0.0  # one stall, then ordinary 20 ms polls that never see a frame
        return None


def test_wait_connected_classifies_host_stall_vs_transport_miss():
    """A starved poll loop is reported as a host stall (retry allowed); a plain miss is not (notes/309)."""
    with pytest.raises(NotConnected) as stall:
        _wait_connected(_FakePub(), _FakeSub(1.2), lambda: None, timeout=1.5, port=1)
    assert stall.value.host_stall_s > HOST_STALL_S and "max poll gap 1.2" in str(stall.value)
    with pytest.raises(NotConnected) as miss:
        _wait_connected(_FakePub(), _FakeSub(0.0), lambda: None, timeout=0.3, port=1)
    assert miss.value.host_stall_s < HOST_STALL_S
    assert "tcp=" in str(miss.value) and " polls in " in str(miss.value)
