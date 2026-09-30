"""ZMQ PUB/SUB transport for ``dropbear_hg-v1`` with Unitree-style channel names.

Topology (CONTRACTS section 6): the robot side (Newton bridge today, ESP32
gateway later) *binds* both sockets; clients *connect*.

========== ============= ============================ ==================
channel    message       endpoint (default)           robot side / client
========== ============= ============================ ==================
rt/lowstate :class:`LowState` ``tcp://127.0.0.1:5556``  PUB bind / SUB connect
rt/lowcmd   :class:`LowCmd`   ``tcp://127.0.0.1:5555``  SUB bind / PUB connect
========== ============= ============================ ==================

Every message is one ZMQ frame ``b"<channel>\\0" + msgpack payload`` so SUB-side
prefix filtering works. Sends never block (a full queue drops the message and
counts it). Reads have *latest-message* semantics: all queued frames are drained
and only the newest is decoded; the number of skipped frames is counted. (ZMQ's
``CONFLATE`` option is not used because it is incompatible with multipart and
unreliable with topic filtering.)

The ``ChannelFactoryInitialize`` / ``ChannelPublisher`` / ``ChannelSubscriber``
API mirrors ``unitree_sdk2py.core.channel`` so Unitree example code ports with
an import change::

    ChannelFactoryInitialize(0)                    # role="client" by default
    pub = ChannelPublisher("rt/lowcmd", LowCmd); pub.Init()
    sub = ChannelSubscriber("rt/lowstate", LowState); sub.Init(handler, 10)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import zmq

from .types import CrcError, LowCmd, LowState

LOWCMD = "rt/lowcmd"
LOWSTATE = "rt/lowstate"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_CMD_PORT = 5555
DEFAULT_STATE_PORT = 5556

MSG_TYPES = {LOWCMD: LowCmd, LOWSTATE: LowState}


@dataclass(frozen=True)
class Endpoints:
    """TCP endpoints of the two channels."""

    host: str = DEFAULT_HOST
    cmd_port: int = DEFAULT_CMD_PORT
    state_port: int = DEFAULT_STATE_PORT

    def for_channel(self, channel: str) -> str:
        if channel == LOWCMD:
            return f"tcp://{self.host}:{self.cmd_port}"
        if channel == LOWSTATE:
            return f"tcp://{self.host}:{self.state_port}"
        raise KeyError(f"unknown channel {channel!r}; expected {LOWCMD!r} or {LOWSTATE!r}")


def _prefix(channel: str) -> bytes:
    return channel.encode() + b"\0"


class ZmqPublisher:
    """Non-blocking publisher of raw payloads on one channel.

    Args:
        endpoint: ``tcp://host:port``.
        channel: channel name used as the frame prefix.
        bind: bind (robot side) or connect (client side).
        sndhwm: send high-water mark; beyond it messages are dropped (counted in :attr:`dropped`).
    """

    def __init__(self, endpoint: str, channel: str, bind: bool, sndhwm: int = 16,
                 context: zmq.Context | None = None):
        self.endpoint, self.channel = endpoint, channel
        self._prefix = _prefix(channel)
        self._sock = (context or zmq.Context.instance()).socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, sndhwm)
        self._sock.setsockopt(zmq.LINGER, 0)
        (self._sock.bind if bind else self._sock.connect)(endpoint)
        self.sent = 0
        self.dropped = 0

    def publish(self, payload: bytes) -> bool:
        """Send one frame without blocking. Returns False if it was dropped."""
        try:
            self._sock.send(self._prefix + payload, flags=zmq.NOBLOCK, copy=True)
        except zmq.Again:
            self.dropped += 1
            return False
        self.sent += 1
        return True

    def close(self) -> None:
        self._sock.close(linger=0)


class ZmqSubscriber:
    """Subscriber with latest-message reads on one channel.

    Args:
        endpoint: ``tcp://host:port``.
        channel: channel name (prefix filter).
        bind: bind (robot side) or connect (client side).
        rcvhwm: receive high-water mark (older frames beyond it are dropped by ZMQ).
    """

    def __init__(self, endpoint: str, channel: str, bind: bool, rcvhwm: int = 64,
                 context: zmq.Context | None = None):
        self.endpoint, self.channel = endpoint, channel
        self._prefix = _prefix(channel)
        self._sock = (context or zmq.Context.instance()).socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, rcvhwm)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SUBSCRIBE, self._prefix)
        (self._sock.bind if bind else self._sock.connect)(endpoint)
        self.received = 0
        self.skipped = 0
        """Frames drained without being returned because a newer one was queued."""

    def recv_latest(self, timeout_ms: float = 0.0) -> bytes | None:
        """Return the newest queued payload, waiting up to ``timeout_ms`` for the first one.

        Returns ``None`` if nothing arrived. ``timeout_ms < 0`` waits forever.
        """
        latest: bytes | None = None
        if timeout_ms != 0 and not self._sock.poll(None if timeout_ms < 0 else int(max(timeout_ms, 1))):
            return None
        while True:
            try:
                frame = self._sock.recv(flags=zmq.NOBLOCK, copy=True)
            except zmq.Again:
                break
            if latest is not None:
                self.skipped += 1
            latest = frame
            self.received += 1
        if latest is None:
            return None
        return latest[len(self._prefix):]

    def close(self) -> None:
        self._sock.close(linger=0)


# --------------------------------------------------------------------------------------
# unitree_sdk2py-style channel API
# --------------------------------------------------------------------------------------


@dataclass
class _FactoryConfig:
    role: str = "client"
    endpoints: Endpoints = Endpoints()


_FACTORY = _FactoryConfig()


def ChannelFactoryInitialize(domain_id: int = 0, network_interface: str | None = None, *, role: str = "client",
                             host: str = DEFAULT_HOST, cmd_port: int = DEFAULT_CMD_PORT,
                             state_port: int = DEFAULT_STATE_PORT) -> None:
    """Configure the process-wide channel factory (mirrors ``unitree_sdk2py``).

    Args:
        domain_id: ignored (DDS domain in Unitree's SDK); kept for signature compatibility.
        network_interface: ignored (DDS NIC in Unitree's SDK).
        role: ``"client"`` (connects; controllers, runners) or ``"robot"`` (binds; bridges).
        host, cmd_port, state_port: TCP endpoints (CONTRACTS section 6 defaults).
    """
    del domain_id, network_interface
    if role not in ("client", "robot"):
        raise ValueError(f"role must be 'client' or 'robot', got {role!r}")
    _FACTORY.role = role
    _FACTORY.endpoints = Endpoints(host, cmd_port, state_port)


M = TypeVar("M", LowCmd, LowState)


class ChannelPublisher(Generic[M]):
    """Typed publisher (``unitree_sdk2py.core.channel.ChannelPublisher`` equivalent)."""

    def __init__(self, name: str, msg_type: type[M], endpoints: Endpoints | None = None, role: str | None = None):
        if MSG_TYPES.get(name) is not msg_type:
            raise TypeError(f"channel {name!r} carries {MSG_TYPES.get(name)}, not {msg_type}")
        self.name, self.msg_type = name, msg_type
        self._endpoints = endpoints or _FACTORY.endpoints
        self._role = role or _FACTORY.role
        self._pub: ZmqPublisher | None = None

    def Init(self) -> None:
        # The robot side publishes state; a client publishes commands. Robot side binds.
        self._pub = ZmqPublisher(self._endpoints.for_channel(self.name), self.name, bind=self._role == "robot")

    def Write(self, msg: M, timeout: float | None = None) -> bool:
        """Serialize (fills ``crc``; sets ``stamp_ns`` if 0) and send without blocking."""
        del timeout
        if self._pub is None:
            raise RuntimeError("ChannelPublisher.Init() was not called")
        if not msg.stamp_ns:
            msg.stamp_ns = time.perf_counter_ns()
        return self._pub.publish(msg.to_bytes())

    @property
    def stats(self) -> dict[str, int]:
        return {"sent": self._pub.sent if self._pub else 0, "dropped": self._pub.dropped if self._pub else 0}

    def Close(self) -> None:
        if self._pub is not None:
            self._pub.close()
            self._pub = None


class ChannelSubscriber(Generic[M]):
    """Typed subscriber with latest-message semantics.

    Use either polling (:meth:`Read`) or a handler thread (:meth:`Init` with
    ``handler``), as in ``unitree_sdk2py``. Malformed frames and CRC failures
    are dropped and counted, never raised into the caller.
    """

    def __init__(self, name: str, msg_type: type[M], endpoints: Endpoints | None = None, role: str | None = None):
        if MSG_TYPES.get(name) is not msg_type:
            raise TypeError(f"channel {name!r} carries {MSG_TYPES.get(name)}, not {msg_type}")
        self.name, self.msg_type = name, msg_type
        self._endpoints = endpoints or _FACTORY.endpoints
        self._role = role or _FACTORY.role
        self._sub: ZmqSubscriber | None = None
        self._handler: Callable[[M], None] | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest: M | None = None
        self.decode_errors = 0
        self.crc_errors = 0

    def Init(self, handler: Callable[[M], None] | None = None, queue_len: int = 0) -> None:
        """Open the socket; with ``handler``, start a daemon thread that calls it per newest message."""
        del queue_len
        self._sub = ZmqSubscriber(self._endpoints.for_channel(self.name), self.name, bind=self._role == "robot")
        if handler is not None:
            self._handler = handler
            self._running = True
            self._thread = threading.Thread(target=self._loop, name=f"sub:{self.name}", daemon=True)
            self._thread.start()

    def _decode(self, payload: bytes) -> M | None:
        try:
            return self.msg_type.from_bytes(payload)
        except CrcError:
            self.crc_errors += 1
        except Exception:  # noqa: BLE001 - malformed input must not kill the subscriber
            self.decode_errors += 1
        return None

    def _loop(self) -> None:
        assert self._sub is not None and self._handler is not None
        while self._running:
            payload = self._sub.recv_latest(timeout_ms=50)
            if payload is None:
                continue
            msg = self._decode(payload)
            if msg is None:
                continue
            with self._lock:
                self._latest = msg
            self._handler(msg)

    def Read(self, timeout: float | None = 0.0) -> M | None:
        """Return the newest message.

        Without a handler thread this polls the socket, waiting up to ``timeout``
        seconds (``None`` = forever) for the first message; returns ``None`` if
        nothing new arrived. With a handler thread it returns the last message the
        thread decoded (possibly already seen).
        """
        if self._sub is None:
            raise RuntimeError("ChannelSubscriber.Init() was not called")
        if self._thread is not None:
            with self._lock:
                return self._latest
        timeout_ms = -1.0 if timeout is None else timeout * 1e3
        payload = self._sub.recv_latest(timeout_ms=timeout_ms)
        return None if payload is None else self._decode(payload)

    @property
    def stats(self) -> dict[str, int]:
        s = self._sub
        return {"received": s.received if s else 0, "skipped": s.skipped if s else 0,
                "decode_errors": self.decode_errors, "crc_errors": self.crc_errors}

    def Close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sub is not None:
            self._sub.close()
            self._sub = None
