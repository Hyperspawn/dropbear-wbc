"""Bind-first loopback endpoints for the fake robots in the ZMQ tests.

A fake robot binds its two channels on port 0 (the OS picks a free port and the socket holds it from bind to
close, so there is no check-then-bind race), reads the bound ports back via ZMQ_LAST_ENDPOINT and hands them to
the client. A bind failure is raised in the test thread at once instead of surfacing as the client's 20 s
"no LowState received" timeout. Fixed literal ports clashed with any other process on the host holding them
(another checkout's suite, a matrix script; notes/397, item 398).
"""
from __future__ import annotations

import threading

import zmq

import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk.transport import LOWCMD, LOWSTATE, ChannelPublisher, ChannelSubscriber, Endpoints
from dropbear_wbc.sdk.types import LowCmd, LowState

ANY_PORT = Endpoints(cmd_port=0, state_port=0)
BIND_TIMEOUT_S = 10.0


def bound_port(channel) -> int:
    """Port a bound channel (ChannelPublisher/ChannelSubscriber after Init, or a ZmqPublisher/ZmqSubscriber)
    actually listens on."""
    inner = getattr(channel, "_pub", None) or getattr(channel, "_sub", None) or channel
    endpoint = inner._sock.getsockopt_string(zmq.LAST_ENDPOINT)
    port = int(endpoint.rsplit(":", 1)[1]) if ":" in endpoint else 0
    if port <= 0:
        raise RuntimeError(f"channel is not bound to a TCP port: LAST_ENDPOINT {endpoint!r}")
    return port


def bind_robot_channels(endpoints: Endpoints | None = None):
    """Robot side: bind rt/lowstate PUB and rt/lowcmd SUB on ``endpoints`` (default port 0 on both).

    Returns ``(pub, sub, bound)`` where ``bound`` carries the ports actually bound; on any failure both channels
    are closed and the error propagates."""
    ep = endpoints if endpoints is not None else ANY_PORT
    pub = ChannelPublisher(LOWSTATE, LowState, endpoints=ep, role="robot")
    sub = ChannelSubscriber(LOWCMD, LowCmd, endpoints=ep, role="robot")
    try:
        pub.Init()
        sub.Init()
        bound = Endpoints(host=ep.host, cmd_port=bound_port(sub), state_port=bound_port(pub))
    except BaseException:
        pub.Close()
        sub.Close()
        raise
    return pub, sub, bound


class BindFirstRobot(threading.Thread):
    """Fake-robot thread that binds before the client starts.

    ``run()`` starts with ``channels = self.bind()`` and returns if it is ``None``; the test calls
    ``ep = robot.start_bound()`` and passes ``ep.cmd_port`` / ``ep.state_port`` to the client."""

    BIND_TIMEOUT_S = BIND_TIMEOUT_S

    def __init__(self, endpoints: Endpoints | None = None):
        super().__init__(daemon=True)
        self.endpoints = endpoints if endpoints is not None else ANY_PORT
        self.running = True
        self.bound_ep: Endpoints | None = None
        self.bind_error: BaseException | None = None
        self._bound = threading.Event()

    def bind(self):
        """Called first in ``run()``: ``(pub, sub)`` once bound, ``None`` after recording the bind error."""
        try:
            pub, sub, self.bound_ep = bind_robot_channels(self.endpoints)
        except Exception as e:  # noqa: BLE001 - re-raised in the test thread by wait_bound()
            self.bind_error = e
            return None
        finally:
            self._bound.set()
        return pub, sub

    def wait_bound(self, timeout: float | None = None) -> Endpoints:
        timeout = self.BIND_TIMEOUT_S if timeout is None else timeout
        if not self._bound.wait(timeout):
            self.running = False
            raise TimeoutError(f"fake robot did not bind within {timeout} s")
        if self.bind_error is not None:
            raise RuntimeError(f"fake robot could not bind {self.endpoints}: {self.bind_error!r}") from self.bind_error
        return self.bound_ep

    def start_bound(self, timeout: float | None = None) -> Endpoints:
        self.start()
        return self.wait_bound(timeout)
