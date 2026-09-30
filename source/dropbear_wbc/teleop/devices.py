"""Teleop input sources: left/right wrist target poses (4x4) in the robot **torso frame** (see :mod:`.arm_ik`).

Every source implements ``start()``, ``get(t) -> WristTargets`` and ``close()``. ``t`` is the loop clock in
seconds (the simulated time for scripted runs, wall time otherwise); a source may ignore it.

* :class:`ScriptedSource` - no hardware: figure-8 / sine / hold / pose-sequence targets around reference wrist
  poses (for tests and verification runs).
* :class:`KeyboardSource` - WASD/QE (left hand), IJKL/UO (right hand), ``[ ]`` and ``; '`` wrist roll; backends
  ``msvcrt`` (Windows console), ``pynput`` (if installed) or ``stdin`` lines; :meth:`KeyboardSource.feed` injects
  keys (tests).
* :class:`VuerXRSource` - WebXR headset through a Vuer server, the televuer pattern: ``CAMERA_MOVE`` (head),
  ``HAND_MOVE`` (hand tracking, 25 joints; the wrist joint is the arm pose) or ``CONTROLLER_MOVE`` (controllers)
  events -> head-relative wrist poses (``head_yaw`` reference, :mod:`.frames`) -> :class:`.frames.OperatorMapping`
  (scale + calibrated offset) -> torso frame. The server binds **127.0.0.1** by default; see docs/TELEOP.md for
  what a headset on the LAN needs (HTTPS certificate, explicit ``--xr-host``).
"""
from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .frames import OperatorMapping, xr_matrix, xr_to_head_relative


@dataclass
class WristTargets:
    """One sample of the operator's wrist targets.

    Attributes:
        t: source time [s] (loop clock for scripted sources, wall clock since start for live devices).
        stamp_ns: ``time.perf_counter_ns()`` when the underlying measurement arrived (latency reference).
        left, right: 4x4 torso-frame wrist targets, ``None`` = hold that arm.
        valid: False until the device has produced real data.
        events: discrete operator events since the last call (``"start"``, ``"stop"``, ``"calibrate"``,
            ``"home"``, ``"record"``).
        raw: device-specific extras recorded with the frame (e.g. head pose).
    """

    t: float
    stamp_ns: int
    left: np.ndarray | None
    right: np.ndarray | None
    valid: bool = True
    events: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# ------------------------------------------------------------------------------------------------ scripted
class ScriptedSource:
    """Deterministic targets around reference wrist poses (torso frame).

    Args:
        reference: side -> 4x4 reference wrist pose (e.g. ``ik.fk(side, q_ref)``).
        kind: ``"figure8"`` (3-D Lissajous: x ``ax sin(w t / 2)``, y ``ay sin(w t)``, z ``az sin(2 w t)``),
            ``"sine"`` (along ``axis``), ``"hold"`` (the reference), ``"poses"`` (a sequence of torso-frame poses
            per side, each held ``hold_s`` with ``move_s`` smooth transitions).
        amplitude: (ax, ay, az) [m]; the right arm mirrors y (both hands move outward together).
        period: [s] of the base frequency.
        ramp_s: amplitude ramps in with a smoothstep over this time (no jump at start).
        start_from: optional side -> 4x4 start pose; the target moves from it to the reference over ``ramp_s``.
        roll_amplitude: wrist-roll oscillation [rad] about the target's x axis (exercises wrist roll).
        orientation: ``"fixed"`` (reference orientation) or ``"none"`` (position only: rotation from the
            reference, the IK caller may set the rotation weight to 0).
    """

    name = "scripted"

    def __init__(self, reference: dict, kind: str = "figure8", amplitude=(0.04, 0.06, 0.05), period: float = 6.0,
                 ramp_s: float = 2.0, axis=(0.0, 1.0, 0.0), start_from: dict | None = None,
                 roll_amplitude: float = 0.0, poses: dict | None = None, hold_s: float = 1.5, move_s: float = 1.0):
        if kind not in ("figure8", "sine", "hold", "poses"):
            raise ValueError(f"unknown scripted kind {kind!r}")
        self.reference = {s: np.asarray(v, dtype=float) for s, v in reference.items()}
        self.kind = kind
        self.amp = np.asarray(amplitude, dtype=float)
        self.period = float(period)
        self.ramp_s = float(ramp_s)
        self.axis = np.asarray(axis, dtype=float) / max(np.linalg.norm(axis), 1e-12)
        self.start_from = start_from
        self.roll_amplitude = float(roll_amplitude)
        self.poses = poses
        self.hold_s, self.move_s = float(hold_s), float(move_s)
        self.t0: float | None = None

    def start(self) -> None:
        self.t0 = None

    def close(self) -> None:
        pass

    @property
    def duration_poses(self) -> float:
        n = len(next(iter(self.poses.values()))) if self.poses else 0
        return n * (self.hold_s + self.move_s)

    def _offset(self, tau: float, side: str) -> np.ndarray:
        w = 2.0 * math.pi / self.period
        sgn = 1.0 if side == "left" else -1.0
        if self.kind == "figure8":
            off = np.array([self.amp[0] * math.sin(0.5 * w * tau), sgn * self.amp[1] * math.sin(w * tau),
                            self.amp[2] * math.sin(2.0 * w * tau)])
        elif self.kind == "sine":
            a = self.axis * np.array([1.0, sgn, 1.0])
            off = a * float(np.linalg.norm(self.amp)) * math.sin(w * tau)
        else:
            off = np.zeros(3)
        return off

    def _poses_at(self, tau: float, side: str) -> np.ndarray:
        seq = self.poses[side]
        seg = self.hold_s + self.move_s
        i = int(tau // seg)
        if i >= len(seq):
            return np.asarray(seq[-1], dtype=float)
        u = tau - i * seg
        cur = np.asarray(seq[i], dtype=float)
        prev = np.asarray(seq[i - 1], dtype=float) if i > 0 else self.reference[side]
        a = _smoothstep(u / self.move_s) if self.move_s > 0 else 1.0
        out = cur.copy()
        out[:3, 3] = prev[:3, 3] + a * (cur[:3, 3] - prev[:3, 3])
        # orientation: slerp via the relative rotation vector
        from .arm_ik import so3_log
        rv = so3_log(cur[:3, :3] @ prev[:3, :3].T) * a
        ang = float(np.linalg.norm(rv))
        if ang > 1e-12:
            k = rv / ang
            kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            r = np.eye(3) + math.sin(ang) * kx + (1 - math.cos(ang)) * kx @ kx
        else:
            r = np.eye(3)
        out[:3, :3] = r @ prev[:3, :3]
        return out

    def get(self, t: float) -> WristTargets:
        if self.t0 is None:
            self.t0 = t
        tau = t - self.t0
        out = {}
        for side, ref in self.reference.items():
            if self.kind == "poses":
                out[side] = self._poses_at(tau, side)
                continue
            m = ref.copy()
            ramp = _smoothstep(tau / self.ramp_s) if self.ramp_s > 0 else 1.0
            if self.start_from is not None and side in self.start_from:
                m[:3, 3] = self.start_from[side][:3, 3] + ramp * (ref[:3, 3] - self.start_from[side][:3, 3])
            m[:3, 3] = m[:3, 3] + ramp * self._offset(tau, side)
            if self.roll_amplitude:
                m[:3, :3] = m[:3, :3] @ _rot_x(ramp * self.roll_amplitude * math.sin(2.0 * math.pi * tau / self.period))
            out[side] = m
        return WristTargets(t=t, stamp_ns=time.perf_counter_ns(), left=out.get("left"), right=out.get("right"))


# ------------------------------------------------------------------------------------------------ keyboard
KEYMAP = {
    # key: (side, axis or "roll", sign)
    "w": ("left", 0, +1), "s": ("left", 0, -1), "a": ("left", 1, +1), "d": ("left", 1, -1),
    "q": ("left", 2, +1), "e": ("left", 2, -1), "[": ("left", "roll", +1), "]": ("left", "roll", -1),
    "i": ("right", 0, +1), "k": ("right", 0, -1), "j": ("right", 1, +1), "l": ("right", 1, -1),
    "u": ("right", 2, +1), "o": ("right", 2, -1), ";": ("right", "roll", +1), "'": ("right", "roll", -1),
}
KEY_EVENTS = {"r": "start", "h": "home", "c": "calibrate", "x": "stop", "\x1b": "stop", "p": "record"}


class KeyboardSource:
    """Keyboard wrist jogging in the torso frame (xr_teleoperate-free fallback / debugging).

    Each key press moves that hand's target by ``step_m`` (or rotates the wrist by ``step_rad``); a held key
    auto-repeats at the OS rate, so holding gives a velocity. Targets start at ``reference`` (usually the FK of the
    robot's current arm pose) and are clamped to ``max_offset_m`` from it. ``r``/``h``/``c``/``x``/``p`` are events
    (start, home, calibrate, stop, record); Esc = stop.
    """

    name = "keyboard"

    def __init__(self, reference: dict, step_m: float = 0.01, step_rad: float = 0.05, max_offset_m: float = 0.35,
                 backend: str = "auto"):
        self.reference = {s: np.asarray(v, dtype=float) for s, v in reference.items()}
        self.step_m, self.step_rad, self.max_offset = float(step_m), float(step_rad), float(max_offset_m)
        self.offset = {s: np.zeros(3) for s in self.reference}
        self.roll = {s: 0.0 for s in self.reference}
        self.backend = backend
        self._events: list[str] = []
        self._lock = threading.Lock()
        self._stamp = time.perf_counter_ns()
        self._listener = None
        self._stdin_thread = None
        self._running = False

    def start(self) -> None:
        self._running = True
        be = self.backend
        if be == "auto":
            try:
                import msvcrt  # noqa: F401
                be = "msvcrt"
            except ImportError:
                be = "stdin"
        if be == "pynput":
            from pynput import keyboard  # type: ignore

            def on_press(k):
                ch = getattr(k, "char", None)
                if ch is None and k == keyboard.Key.esc:
                    ch = "\x1b"
                if ch:
                    self.feed(ch)
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
        elif be == "stdin":
            import sys

            def loop():
                while self._running:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    for ch in line.strip():
                        self.feed(ch)
            self._stdin_thread = threading.Thread(target=loop, daemon=True)
            self._stdin_thread.start()
        self.backend = be

    def close(self) -> None:
        self._running = False
        if self._listener is not None:
            self._listener.stop()

    def feed(self, ch: str) -> None:
        """Apply one key (also used by the msvcrt poll and by tests)."""
        ch = ch.lower() if ch not in ("\x1b",) else ch
        with self._lock:
            self._stamp = time.perf_counter_ns()
            if ch in KEYMAP:
                side, ax, sgn = KEYMAP[ch]
                if side not in self.offset:
                    return
                if ax == "roll":
                    self.roll[side] += sgn * self.step_rad
                else:
                    self.offset[side][ax] += sgn * self.step_m
                    n = np.linalg.norm(self.offset[side])
                    if n > self.max_offset:
                        self.offset[side] *= self.max_offset / n
            elif ch in KEY_EVENTS:
                ev = KEY_EVENTS[ch]
                if ev == "home":
                    for s in self.offset:
                        self.offset[s] = np.zeros(3)
                        self.roll[s] = 0.0
                self._events.append(ev)

    def _poll_msvcrt(self) -> None:
        import msvcrt

        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # function / arrow key prefix
                msvcrt.getwch()
                continue
            self.feed(ch)

    def get(self, t: float) -> WristTargets:
        if self.backend == "msvcrt":
            try:
                self._poll_msvcrt()
            except Exception:  # noqa: BLE001 - not a console (e.g. detached): no keyboard input
                pass
        with self._lock:
            out = {}
            for s, ref in self.reference.items():
                m = ref.copy()
                m[:3, 3] = ref[:3, 3] + self.offset[s]
                m[:3, :3] = ref[:3, :3] @ _rot_x(self.roll[s])
                out[s] = m
            ev, self._events = self._events, []
            stamp = self._stamp
        return WristTargets(t=t, stamp_ns=stamp, left=out.get("left"), right=out.get("right"), events=ev)


# ------------------------------------------------------------------------------------------------ WebXR (Vuer)
_STATE_KEYS = ("trigger", "triggerValue", "squeeze", "squeezeValue", "thumbstick", "aButton", "bButton", "pinch")
_META = ("stamp_wall_s", "unused", "unused2", "n_cam", "n_hand", "n_ctrl", "n_sessions", "last_delay_s")


class _SharedXR:
    """Latest XR data in shared memory (multiprocessing arrays), written by the Vuer server (thread or process)."""

    def __init__(self, ctx=None):
        import multiprocessing as mp

        ctx = ctx or mp
        self.head = ctx.Array("d", 16, lock=True)
        self.left = ctx.Array("d", 16, lock=True)
        self.right = ctx.Array("d", 16, lock=True)
        self.state = ctx.Array("d", 2 * len(_STATE_KEYS), lock=True)
        self.meta = ctx.Array("d", len(_META), lock=True)
        for a in (self.head, self.left, self.right):
            a[:] = [float("nan")] * 16

    def write_arms(self, left16, right16, st_left: dict, st_right: dict, ts_ms, kind: str) -> None:
        now = time.time()
        with self.meta.get_lock():
            self.left[:] = [float(x) for x in left16]
            self.right[:] = [float(x) for x in right16]
            vals = [float(st_left.get(k, 0.0)) for k in _STATE_KEYS] + [float(st_right.get(k, 0.0)) for k in _STATE_KEYS]
            self.state[:] = vals
            self.meta[0] = now
            self.meta[4 if kind == "hand" else 5] += 1
            if ts_ms is not None:
                self.meta[7] = now - ts_ms / 1000.0

    def write_head(self, head16) -> None:
        with self.meta.get_lock():
            self.head[:] = [float(x) for x in head16]
            self.meta[3] += 1

    def read(self):
        with self.meta.get_lock():
            return (np.array(self.head[:]), np.array(self.left[:]), np.array(self.right[:]),
                    np.array(self.state[:]), np.array(self.meta[:]))


def _serve_vuer(cfg: dict, shared: _SharedXR, ready, holder: dict | None = None) -> None:
    """Run a Vuer server (blocking) that writes CAMERA_MOVE / HAND_MOVE / CONTROLLER_MOVE data into ``shared``.

    Module-level so that it can be the target of a spawned process (televuer runs Vuer in its own process too).
    Only "/" (web client + websocket) and "/assets" (client build) are served: Vuer's "/static" (the working
    directory) and "/relay" (POST events into sessions) are deliberately not exposed.
    """
    try:
        from aiohttp import web
        from vuer import Vuer
        from vuer.schemas import Hands, MotionControllers

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if holder is not None:
            holder["loop"] = loop
        kw = dict(host=cfg["host"], port=cfg["port"], queries=dict(grid=False), queue_len=3)
        if cfg.get("cert") and cfg.get("key"):
            kw.update(cert=cfg["cert"], key=cfg["key"])
        app = Vuer(**kw)

        def _ts(event):
            ts = getattr(event, "ts", None)
            try:
                return ts.timestamp() * 1000.0 if hasattr(ts, "timestamp") else float(ts)
            except Exception:  # noqa: BLE001
                return None

        async def on_cam(event, session, fps=60):
            try:
                shared.write_head(event.value["camera"]["matrix"])
            except Exception:  # noqa: BLE001 - malformed event: ignore (televuer does the same)
                pass

        async def on_hand(event, session, fps=60):
            try:
                v = event.value
                shared.write_arms(list(v["left"])[0:16], list(v["right"])[0:16], v.get("leftState", {}) or {},
                                  v.get("rightState", {}) or {}, _ts(event), "hand")
            except Exception:  # noqa: BLE001
                pass

        async def on_ctrl(event, session, fps=60):
            try:
                v = event.value
                shared.write_arms(v["left"], v["right"], v.get("leftState", {}) or {}, v.get("rightState", {}) or {},
                                  _ts(event), "ctrl")
            except Exception:  # noqa: BLE001
                pass

        app.add_handler("CAMERA_MOVE")(on_cam)
        app.add_handler("HAND_MOVE")(on_hand)
        app.add_handler("CONTROLLER_MOVE")(on_ctrl)
        hand_tracking, fps = cfg["hand_tracking"], cfg["display_fps"]

        async def main(session):
            with shared.meta.get_lock():
                shared.meta[6] += 1
            if hand_tracking:
                session.upsert(Hands(stream=True, key="hands", hideLeft=True, hideRight=True), to="bgChildren")
            else:
                session.upsert(MotionControllers(stream=True, key="motionControllers", left=True, right=True),
                               to="bgChildren")
            while True:
                await asyncio.sleep(1.0 / fps)

        app.spawn(start=False)(main)
        app._add_route("", app.socket_index, method="GET")
        app._add_static("/assets", app.client_root / "assets")

        async def init():
            runner = web.AppRunner(app.app)
            await runner.setup()
            ssl_ctx = None
            if cfg.get("cert") and cfg.get("key"):
                import ssl

                ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_ctx.load_cert_chain(certfile=cfg["cert"], keyfile=cfg["key"])
            site = web.TCPSite(runner, cfg["host"], cfg["port"], ssl_context=ssl_ctx)
            await site.start()
            return runner

        runner = loop.run_until_complete(init())
        ready.set()
        loop.run_forever()
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(runner.cleanup())
        loop.close()
    except BaseException as e:  # noqa: BLE001
        if holder is not None:
            holder["error"] = e
        else:
            print(f"[vuer-xr] server failed: {e!r}", flush=True)
        ready.set()


class VuerXRSource:
    """WebXR wrist targets from a Vuer server (televuer's event handling; no image streaming).

    Args:
        mapping: :class:`.frames.OperatorMapping` (scale, per-hand offsets; :meth:`calibrate` sets offsets).
        hand_tracking: True = ``HAND_MOVE`` (25 joints per hand, wrist = joint 0), False = ``CONTROLLER_MOVE``.
        host, port: bind address. Default **127.0.0.1** (local only). A headset on the LAN needs
            ``host="0.0.0.0"`` (or the LAN IP) **and** HTTPS (``cert``/``key``): WebXR only runs in a secure context.
        arm_reference_mode: ``"head_yaw"`` (default, xr_teleoperate) or ``"head_position"``.
        stale_s: data older than this is reported ``valid=False`` (the loop then holds the arms).
        backend: ``"process"`` (default; like televuer, the websocket / asyncio work runs in its own process and
            the latest poses are shared through multiprocessing arrays, so it never competes with the control loop
            for the GIL) or ``"thread"``.
    """

    name = "webxr"

    def __init__(self, mapping: OperatorMapping | None = None, hand_tracking: bool = True, host: str = "127.0.0.1",
                 port: int = 8012, cert: str | None = None, key: str | None = None,
                 arm_reference_mode: str = "head_yaw", stale_s: float = 0.5, display_fps: float = 30.0,
                 backend: str = "process"):
        if backend not in ("process", "thread"):
            raise ValueError(f"unknown backend {backend!r}")
        self.mapping = mapping or OperatorMapping()
        self.hand_tracking = hand_tracking
        self.host, self.port, self.cert, self.key = host, int(port), cert, key
        self.mode = arm_reference_mode
        self.stale_s = float(stale_s)
        self.display_fps = float(display_fps)
        self.backend = backend
        self._proc = None
        self._thread: threading.Thread | None = None
        self._holder: dict = {}
        self._shared: _SharedXR | None = None
        self._prev_buttons: dict = {}

    # -- server
    def start(self) -> None:
        import multiprocessing as mp

        cfg = {"host": self.host, "port": self.port, "cert": self.cert, "key": self.key,
               "hand_tracking": self.hand_tracking, "display_fps": self.display_fps}
        if self.backend == "process":
            ctx = mp.get_context("spawn")
            self._shared = _SharedXR(ctx)
            ready = ctx.Event()
            self._proc = ctx.Process(target=_serve_vuer, args=(cfg, self._shared, ready), name="vuer-xr", daemon=True)
            self._proc.start()
            if not ready.wait(timeout=60.0) or not self._proc.is_alive():
                raise RuntimeError(f"Vuer server process did not start (alive={self._proc.is_alive()})")
        else:
            self._shared = _SharedXR()
            ready = threading.Event()
            self._thread = threading.Thread(target=_serve_vuer, args=(cfg, self._shared, ready, self._holder),
                                            name="vuer-xr", daemon=True)
            self._thread.start()
            if not ready.wait(timeout=30.0):
                raise RuntimeError("Vuer server did not start within 30 s")
            if "error" in self._holder:
                raise RuntimeError(f"Vuer server failed: {self._holder['error']!r}")

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join(timeout=3.0)
        loop = self._holder.get("loop")
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    @property
    def url(self) -> str:
        scheme = "https" if self.cert else "http"
        ws = "wss" if self.cert else "ws"
        return f"{scheme}://{self.host}:{self.port}/?ws={ws}://{self.host}:{self.port}"

    @property
    def counts(self) -> dict:
        if self._shared is None:
            return {}
        m = self._shared.read()[4]
        return {"CAMERA_MOVE": int(m[3]), "HAND_MOVE": int(m[4]), "CONTROLLER_MOVE": int(m[5]), "sessions": int(m[6]),
                "last_event_delay_s": float(m[7])}

    # -- targets
    def head_relative(self):
        """(head_robot, left_rel, right_rel, stamp_wall_s, state) from the latest XR data, or None before any data."""
        if self._shared is None:
            return None
        head, left, right, state, meta = self._shared.read()
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            return None
        head_m = xr_matrix(head) if np.all(np.isfinite(head)) else np.full((4, 4), np.nan)
        h, l_rel, r_rel = xr_to_head_relative(head_m, xr_matrix(left), xr_matrix(right), self.hand_tracking, self.mode)
        n = len(_STATE_KEYS)
        st = {"left": dict(zip(_STATE_KEYS, state[:n])), "right": dict(zip(_STATE_KEYS, state[n:]))}
        return h, l_rel, r_rel, float(meta[0]), st

    def calibrate(self, robot_ref: dict, orientation: bool = False) -> dict:
        """Map the operator's current wrists onto ``robot_ref`` (side -> torso-frame 4x4). Returns a summary."""
        hr = self.head_relative()
        if hr is None:
            raise RuntimeError("no XR data yet; cannot calibrate")
        _, l_rel, r_rel, _, _ = hr
        return self.mapping.calibrate({"left": l_rel, "right": r_rel}, robot_ref, orientation=orientation)

    def _button_events(self, state: dict) -> list[str]:
        """Controller buttons -> events (rising edges): right A = ``"toggle"`` (start / pause tracking), left X =
        ``"calibrate"``, both thumbsticks pressed = ``"stop"`` (xr_teleoperate's soft e-stop gesture)."""
        ev = []
        cur = {"rA": bool(state["right"]["aButton"]), "lA": bool(state["left"]["aButton"]),
               "sticks": bool(state["left"]["thumbstick"]) and bool(state["right"]["thumbstick"])}
        for k, name in (("rA", "toggle"), ("lA", "calibrate"), ("sticks", "stop")):
            if cur[k] and not self._prev_buttons.get(k, False):
                ev.append(name)
        self._prev_buttons = cur
        return ev

    def get(self, t: float) -> WristTargets:
        hr = self.head_relative()
        if hr is None:
            return WristTargets(t=t, stamp_ns=time.perf_counter_ns(), left=None, right=None, valid=False)
        head, l_rel, r_rel, stamp_wall, state = hr
        age = time.time() - stamp_wall
        valid = age <= self.stale_s
        ev = [] if self.hand_tracking else self._button_events(state)
        stamp_ns = time.perf_counter_ns() - int(max(age, 0.0) * 1e9)  # arrival time on this process' clock
        return WristTargets(t=t, stamp_ns=stamp_ns, left=self.mapping.apply("left", l_rel) if valid else None,
                            right=self.mapping.apply("right", r_rel) if valid else None, valid=valid, events=ev,
                            raw={"head_robot": head, "left_rel": l_rel, "right_rel": r_rel, "age_s": age})
