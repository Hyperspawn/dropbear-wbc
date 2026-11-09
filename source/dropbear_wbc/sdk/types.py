"""``dropbear_hg-v1`` message types: the Unitree ``unitree_hg`` LowCmd/LowState equivalent.

Shapes follow Unitree's IDL (``MotorCmd{mode,q,dq,tau,kp,kd}``,
``MotorState{q,dq,ddq,tau_est,temperature}``, ``IMUState{quaternion,gyroscope,
accelerometer,rpy}``) with 22 body motors in the order of
:data:`dropbear_wbc.sdk.motors.MOTOR_NAMES` and an optional 6-slot neck block
(SDK slots 22-27). Internally each block is a struct of numpy arrays, and
``cmd.motor_cmd[i].q = x`` style access (as in ``unitree_sdk2py``) works through
lightweight views.

Units and frames:
    * Joint position rad (neck: m), velocity rad/s (m/s), torque N*m (N).
    * ``IMUState`` is the torso root body (``world``): ``quat_wxyz`` maps body to
      world (z up); ``gyro`` is the body-frame angular velocity [rad/s];
      ``accel`` is the body-frame specific force [m/s^2] (reads +9.81 on z
      when upright and at rest); ``rpy`` is roll/pitch/yaw [rad] (intrinsic ZYX).
    * ``IMUState.valid`` (uint8, wire ``valid``) is 1 only when the producer holds a live, configured IMU sample
      (sim: always 1; robot HAL: WHO_AM_I-verified, configured, fresh DBI frame). ``IMUState.age_us`` (uint32,
      wire ``age_us``) is the producer's best estimate of sample time -> LowState publish [us]
      (:data:`IMU_AGE_UNKNOWN` = unknown). A default-constructed ``IMUState`` is **invalid** (valid=0, age
      unknown): the identity quaternion it carries is a placeholder, not an upright reading. Consumers must gate
      on both fields (``deploy.fsm`` refuses / leaves ``POLICY``). Added by local patch 0004 (item 137).
    * ``MotorStateBlock.tau_valid`` (uint8[n], wire ``tau_valid``) is 1 when ``tau_est`` is trusted. On hardware
      that means the MyActuator motion-mode (0x500) ``t`` field was strictly inside
      :data:`MOTION_MODE_TAU_RAIL_NM` (+-24 N*m); at the rail, use :func:`tau_est_with_iq_fallback` (``iq * Kt``
      from 0x9C / 0xA1). Sim always sends 1 (applied torque after clip). ``MotorStateBlock.age_us`` (uint32[n],
      wire ``age_us``) is sample time -> LowState publish [us] per slot (:data:`MOTOR_AGE_UNKNOWN` = unknown).
      A pre-health sender omits both keys; decode fills ``tau_valid=1`` (legacy ``tau_est`` kept) and unknown age,
      and CRC is checked over the legacy motor bytes. Unlike IMU health (fail-closed on absent), motor health
      fails open on absent so existing sim / bridge frames keep working. Added for item 85.
    * ``SimState`` is privileged, simulation-only ground truth in the world
      frame. It never exists on hardware; policies that need it are sim-only.

Wire format (msgpack map, one message per ZMQ frame)::

    {"schema": "dropbear_hg-v1", "type": "LowCmd"|"LowState", "tick": uint32,
     "stamp_ns": int64, <blocks as maps of little-endian raw array bytes>, "crc": uint32}

``tick`` in a LowState is the MCU / simulator **sample** counter at 500 Hz (time-driven:
it advances with the control period even when a publish is skipped). It must not be a
publish counter; a publish-counted tick trails wall time under load and trips the deploy
tick-lag gate on fresh frames (item 280 / 282). In a LowCmd it echoes the LowState tick
the command was computed from (0 if unknown), which lets the robot side measure command
latency in ticks. ``stamp_ns`` is ``time.perf_counter_ns()`` at send time; it is only
comparable between processes on the same host (Windows QueryPerformanceCounter is system-wide).

CRC: Unitree's CRC32 (see :mod:`dropbear_wbc.sdk.crc`) over the canonical byte
string: ``<I tick``, ``<q stamp_ns``, then every array's little-endian bytes in
the fixed field order used by :meth:`LowCmd._canonical` /
:meth:`LowState._canonical`.
"""
from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import msgpack
import numpy as np

from .crc import crc32_bytes
from .motors import NUM_MOTORS, NUM_NECK

SCHEMA = "dropbear_hg-v1"

IMU_AGE_UNKNOWN = 0xFFFFFFFF
"""``IMUState.age_us`` sentinel: the producer does not know the sample age (treated as stale)."""

MOTOR_AGE_UNKNOWN = 0xFFFFFFFF
"""``MotorStateBlock.age_us`` sentinel: the producer does not know that slot's sample age (treated as stale)."""

TAU_SRC_DIRECT = 0
"""``MotorStateBlock.tau_src``: ``tau_est`` is the producer's native torque (hardware: 0x500 ``t``; sim: applied torque)."""

TAU_SRC_IQ_KT = 1
"""``MotorStateBlock.tau_src``: ``tau_est`` was substituted by the producer as ``sign * iq * Kt`` (0x9C / 0xA1 ``iq``)."""

TAU_SRC_VALUES = (TAU_SRC_DIRECT, TAU_SRC_IQ_KT)

MOTION_MODE_TAU_RAIL_NM = 24.0
"""MyActuator motion-mode (0x500) ``t`` / ``t_ff`` field ceiling [N*m].

24 % of X10-S2 peak (100), 40 % of CEM-60 peak (60), 60 % of X10 1:7 peak (40). On a knee at
``kp = 500``, ``kp * |err|`` alone hits the rail at :data:`MOTION_MODE_KP500_RAIL_ERR_RAD` (48 mrad).
"""

MOTION_MODE_KP500_RAIL_ERR_RAD = MOTION_MODE_TAU_RAIL_NM / 500.0
"""Position error [rad] at which knee ``kp=500`` alone saturates the 0x500 ``t`` rail (0.048 rad = 48 mrad)."""


TICK_MOD = 1 << 32
"""``tick`` is ``uint32`` on the wire: at the 500 Hz LowState tick it wraps after 2**32 * 2 ms = 99.4 days."""


def tick_diff(new: int, old: int) -> int:
    """Signed modular difference ``new - old`` of two wire ticks, in ``[-2**31, 2**31)``.

    A wrap (``0xFFFFFFF6 -> 0x4``) reads as +14, a producer restart (``1_000_000 -> 0``) as -1_000_000. Only a
    restart after more than 2**31 ticks (49.7 days at 500 Hz) is ambiguous and reads as a forward jump.
    """
    return ((int(new) - int(old) + (TICK_MOD >> 1)) % TICK_MOD) - (TICK_MOD >> 1)


class TickUnwrapper:
    """Map wire ticks to a monotonic (unbounded int) tick by accumulating :func:`tick_diff`.

    Consumers that compare ticks with ``<``/``>=`` or subtract them (loop schedulers, simulated clocks) must use
    the unwrapped tick, or they stall or run backwards at the uint32 wrap. A backwards step stays backwards so a
    producer restart is still visible. ``LowCmd`` serialization masks ``tick & 0xFFFFFFFF``, so echoing an unwrapped
    tick puts the original wire value back on the wire.
    """

    def __init__(self) -> None:
        self._wire: int | None = None
        self._u = 0

    def __call__(self, tick: int) -> int:
        wire = int(tick) & (TICK_MOD - 1)
        self._u = wire if self._wire is None else self._u + tick_diff(wire, self._wire)
        self._wire = wire
        return self._u

    def state(self, low_state):
        """Return ``low_state`` with ``tick`` replaced by the unwrapped tick (``None`` passes through)."""
        if low_state is not None:
            low_state.tick = self(low_state.tick)
        return low_state


class TickStallError(TimeoutError):
    """LowState frames keep arriving but the (unwrapped) tick has not passed its previous maximum."""


class TickStallWatch:
    """Abort a tick-scheduled loop whose producer keeps publishing without advancing the tick.

    A sim-clock scheduler waits for ``tick >= target``. A frozen simulation (or a HAL stuck on one sample) keeps
    sending frames, so a receive timeout never fires and the loop spins forever on keepalives. A producer restart
    (tick back near 0) has the same effect until the tick climbs past the old value. ``check`` raises
    :class:`TickStallError` once no tick above the running maximum has been seen for ``stall_s`` seconds.
    Feed it unwrapped ticks (:class:`TickUnwrapper`). ``stall_s <= 0`` disables it.
    """

    def __init__(self, tick: int, stall_s: float, now: float | None = None) -> None:
        self.stall_s = float(stall_s)
        self.max_tick = int(tick)
        self.last_tick = int(tick)
        self.t_adv = time.perf_counter() if now is None else float(now)

    def check(self, tick: int, now: float) -> None:
        tick = int(tick)
        self.last_tick = tick
        if tick > self.max_tick:
            self.max_tick, self.t_adv = tick, float(now)
            return
        if self.stall_s > 0 and now - self.t_adv > self.stall_s:
            how = (f"frozen at tick {tick}" if tick == self.max_tick
                   else f"at tick {tick}, behind the previous maximum {self.max_tick} (producer restart?)")
            raise TickStallError(f"LowState tick has not advanced for {self.stall_s} s: {how}")


class CrcError(ValueError):
    """Raised when a decoded message's CRC does not match its contents."""


class MotorMode(IntEnum):
    """Per-motor mode, as in Unitree's ``MotorCmd.mode`` (1 enable, 0 disable)."""

    DISABLE = 0
    """Motor off: the bridge applies zero torque (joint is free)."""
    ENABLE = 1
    """Motor applies ``tau = tau_ff + kp*(q - q_meas) + kd*(dq - dq_meas)``."""


_F32 = np.dtype("<f4")
_U8 = np.dtype("u1")


def _f32(values: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name}: expected shape ({n},), got {arr.shape}")
    return arr.astype(_F32, copy=True)


def _u8(values: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name}: expected shape ({n},), got {arr.shape}")
    return arr.astype(_U8, copy=True)


def _from_bin(blob: bytes, dtype: np.dtype, n: int, name: str) -> np.ndarray:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError(f"{name}: expected raw bytes, got {type(blob).__name__}")
    arr = np.frombuffer(blob, dtype=dtype)
    if arr.shape != (n,):
        raise ValueError(f"{name}: expected {n} elements, got {arr.shape[0]}")
    return arr.copy()


_U32 = np.dtype("<u4")


def motion_mode_tau_valid(tau_nm: Any, rail_nm: float = MOTION_MODE_TAU_RAIL_NM) -> np.ndarray:
    """Per-slot ``tau_valid`` mask: 1 where ``|tau|`` is strictly inside the 0x500 ``t`` rail.

    A saturated reading (``|tau| >= rail``) is not a usable torque estimate; consumers should fall back to
    ``iq * Kt`` via :func:`tau_est_with_iq_fallback`. Returns ``uint8`` with the same shape as ``tau_nm``.
    """
    t = np.asarray(tau_nm, dtype=np.float64)
    return (np.abs(t) < float(rail_nm)).astype(_U8, copy=False)


def tau_est_with_iq_fallback(tau_est: Any, tau_valid: Any, iq_a: Any, kt_nm_per_a: Any) -> np.ndarray:
    """Torque [N*m]: ``tau_est`` where ``tau_valid==1``, else ``iq_a * kt_nm_per_a``.

    Hardware producers put the 0x500 ``t`` field in ``tau_est`` and clear ``tau_valid`` at the +-24 N*m rail;
    ``iq`` comes from 0x9C / 0xA1 (0.01 A/LSB) and ``Kt`` from the datasheet / bench. Sim leaves ``tau_valid`` at 1
    and never needs the fallback. Output is float32, broadcast against the inputs.
    """
    tau = np.asarray(tau_est, dtype=np.float64)
    valid = np.asarray(tau_valid, dtype=np.uint8)
    iq = np.asarray(iq_a, dtype=np.float64)
    kt = np.asarray(kt_nm_per_a, dtype=np.float64)
    out = np.where(valid != 0, tau, iq * kt)
    return out.astype(np.float32, copy=False)


# --------------------------------------------------------------------------------------
# Motor command block
# --------------------------------------------------------------------------------------


class MotorCmd:
    """View of one slot of a :class:`MotorCmdBlock` (Unitree ``MotorCmd_`` ergonomics)."""

    __slots__ = ("_block", "_i")

    def __init__(self, block: "MotorCmdBlock", index: int):
        self._block = block
        self._i = index

    mode = property(lambda s: int(s._block.mode[s._i]), lambda s, v: s._block.mode.__setitem__(s._i, v))
    q = property(lambda s: float(s._block.q[s._i]), lambda s, v: s._block.q.__setitem__(s._i, v))
    dq = property(lambda s: float(s._block.dq[s._i]), lambda s, v: s._block.dq.__setitem__(s._i, v))
    tau = property(lambda s: float(s._block.tau[s._i]), lambda s, v: s._block.tau.__setitem__(s._i, v))
    kp = property(lambda s: float(s._block.kp[s._i]), lambda s, v: s._block.kp.__setitem__(s._i, v))
    kd = property(lambda s: float(s._block.kd[s._i]), lambda s, v: s._block.kd.__setitem__(s._i, v))

    def __repr__(self) -> str:
        return (f"MotorCmd(mode={self.mode}, q={self.q:.4f}, dq={self.dq:.4f}, tau={self.tau:.4f}, "
                f"kp={self.kp:.3f}, kd={self.kd:.3f})")


@dataclass
class MotorCmdBlock:
    """Struct-of-arrays motor commands (``n`` slots).

    Attributes:
        mode: uint8 per slot, see :class:`MotorMode`.
        q: target position [rad | m].
        dq: target velocity [rad/s | m/s].
        tau: feed-forward torque [N*m | N].
        kp: stiffness [N*m/rad | N/m].
        kd: damping [N*m*s/rad | N*s/m].
    """

    mode: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    kp: np.ndarray
    kd: np.ndarray

    FIELDS = ("mode", "q", "dq", "tau", "kp", "kd")

    @classmethod
    def zeros(cls, n: int, mode: int = MotorMode.ENABLE) -> "MotorCmdBlock":
        """All-zero targets and gains, every slot in ``mode``."""
        return cls(mode=np.full(n, int(mode), dtype=_U8), q=np.zeros(n, _F32), dq=np.zeros(n, _F32),
                   tau=np.zeros(n, _F32), kp=np.zeros(n, _F32), kd=np.zeros(n, _F32))

    @classmethod
    def from_arrays(cls, n: int, *, mode: Any = MotorMode.ENABLE, q: Any = 0.0, dq: Any = 0.0,
                    tau: Any = 0.0, kp: Any = 0.0, kd: Any = 0.0) -> "MotorCmdBlock":
        """Build a block, broadcasting scalars to ``n`` slots."""
        b = lambda v: np.broadcast_to(np.asarray(v), (n,))  # noqa: E731
        return cls(mode=_u8(b(mode), n, "mode"), q=_f32(b(q), n, "q"), dq=_f32(b(dq), n, "dq"),
                   tau=_f32(b(tau), n, "tau"), kp=_f32(b(kp), n, "kp"), kd=_f32(b(kd), n, "kd"))

    def __len__(self) -> int:
        return int(self.q.shape[0])

    def __getitem__(self, index: int) -> MotorCmd:
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        return MotorCmd(self, index % len(self))

    def __iter__(self):
        return (MotorCmd(self, i) for i in range(len(self)))

    def copy(self) -> "MotorCmdBlock":
        return MotorCmdBlock(*(getattr(self, f).copy() for f in self.FIELDS))

    def validate(self, n: int, name: str) -> None:
        for f in self.FIELDS:
            arr = getattr(self, f)
            want = _U8 if f == "mode" else _F32
            if arr.shape != (n,) or arr.dtype != want:
                raise ValueError(f"{name}.{f}: expected {want} shape ({n},), got {arr.dtype} {arr.shape}")

    def to_wire(self) -> dict[str, bytes]:
        return {f: getattr(self, f).tobytes() for f in self.FIELDS}

    @classmethod
    def from_wire(cls, d: dict, n: int, name: str) -> "MotorCmdBlock":
        return cls(**{f: _from_bin(d[f], _U8 if f == "mode" else _F32, n, f"{name}.{f}") for f in cls.FIELDS})

    def _canonical(self) -> bytes:
        return b"".join(getattr(self, f).tobytes() for f in self.FIELDS)


# --------------------------------------------------------------------------------------
# Motor state block
# --------------------------------------------------------------------------------------


class MotorState:
    """Read-only view of one slot of a :class:`MotorStateBlock` (Unitree ``MotorState_``)."""

    __slots__ = ("_block", "_i")

    def __init__(self, block: "MotorStateBlock", index: int):
        self._block = block
        self._i = index

    mode = property(lambda s: int(s._block.mode[s._i]))
    q = property(lambda s: float(s._block.q[s._i]))
    dq = property(lambda s: float(s._block.dq[s._i]))
    ddq = property(lambda s: float(s._block.ddq[s._i]))
    tau_est = property(lambda s: float(s._block.tau_est[s._i]))
    temperature = property(lambda s: float(s._block.temperature[s._i]))
    tau_valid = property(lambda s: int(s._block.tau_valid[s._i]))
    age_us = property(lambda s: int(s._block.age_us[s._i]))

    def __repr__(self) -> str:
        return (f"MotorState(mode={self.mode}, q={self.q:.4f}, dq={self.dq:.4f}, ddq={self.ddq:.3f}, "
                f"tau_est={self.tau_est:.3f}, temperature={self.temperature:.1f}, "
                f"tau_valid={self.tau_valid}, age_us={self.age_us})")


@dataclass
class MotorStateBlock:
    """Struct-of-arrays motor feedback (``n`` slots).

    Attributes:
        mode: uint8 mode the motor is currently in (echo of the applied command mode).
        q: measured position [rad | m].
        dq: measured velocity [rad/s | m/s].
        ddq: acceleration [rad/s^2 | m/s^2] (finite difference in simulation).
        tau_est: estimated output torque [N*m | N] (simulation: last applied, clipped torque;
            hardware: 0x500 ``t`` when ``tau_valid`` else prefer ``iq * Kt``).
        temperature: winding temperature [degC] (simulation: constant).
        tau_valid: uint8[n]; 1 when ``tau_est`` is trusted (sim: always; HW: 0x500 ``t`` inside +-24 N*m).
        age_us: uint32[n]; sample -> publish age [us]; :data:`MOTOR_AGE_UNKNOWN` when unknown.
        tau_src: uint8[n]; where ``tau_est`` came from: :data:`TAU_SRC_DIRECT` (0x500 ``t`` / sim applied torque)
            or :data:`TAU_SRC_IQ_KT` (producer substituted ``iq * Kt`` for a railed ``t``). Optional on the wire;
            absent decodes as all :data:`TAU_SRC_DIRECT` and is left out of the CRC.
    """

    mode: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau_est: np.ndarray
    temperature: np.ndarray
    tau_valid: np.ndarray | None = None
    age_us: np.ndarray | None = None
    tau_src: np.ndarray | None = None

    FIELDS = ("mode", "q", "dq", "ddq", "tau_est", "temperature")
    HEALTH_KEYS = ("tau_valid", "age_us")

    def __post_init__(self) -> None:
        n = int(np.asarray(self.q).reshape(-1).shape[0])
        if self.tau_valid is None:
            self.tau_valid = np.ones(n, _U8)  # fail-open: legacy / unset trusts tau_est
        else:
            self.tau_valid = np.asarray(self.tau_valid, dtype=_U8).reshape(n).copy()
        if self.age_us is None:
            self.age_us = np.full(n, MOTOR_AGE_UNKNOWN, dtype=_U32)
        else:
            self.age_us = np.asarray(self.age_us, dtype=_U32).reshape(n).copy()
        if self.tau_src is None:
            self.tau_src = np.full(n, TAU_SRC_DIRECT, dtype=_U8)
        else:
            self.tau_src = np.asarray(self.tau_src, dtype=_U8).reshape(n).copy()

    @classmethod
    def zeros(cls, n: int) -> "MotorStateBlock":
        return cls(mode=np.zeros(n, _U8), q=np.zeros(n, _F32), dq=np.zeros(n, _F32), ddq=np.zeros(n, _F32),
                   tau_est=np.zeros(n, _F32), temperature=np.zeros(n, _F32),
                   tau_valid=np.ones(n, _U8), age_us=np.full(n, MOTOR_AGE_UNKNOWN, dtype=_U32))

    def __len__(self) -> int:
        return int(self.q.shape[0])

    def __getitem__(self, index: int) -> MotorState:
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        return MotorState(self, index % len(self))

    def __iter__(self):
        return (MotorState(self, i) for i in range(len(self)))

    def validate(self, n: int, name: str) -> None:
        for f in self.FIELDS:
            arr = getattr(self, f)
            want = _U8 if f == "mode" else _F32
            if arr.shape != (n,) or arr.dtype != want:
                raise ValueError(f"{name}.{f}: expected {want} shape ({n},), got {arr.dtype} {arr.shape}")
        tv = np.asarray(self.tau_valid)
        if tv.shape != (n,) or tv.dtype != _U8:
            raise ValueError(f"{name}.tau_valid: expected {_U8} shape ({n},), got {tv.dtype} {tv.shape}")
        if not np.isin(tv, (0, 1)).all():
            raise ValueError(f"{name}.tau_valid: entries must be 0 or 1, got {sorted(set(map(int, tv)))}")
        au = np.asarray(self.age_us)
        if au.shape != (n,) or au.dtype != _U32:
            raise ValueError(f"{name}.age_us: expected {_U32} shape ({n},), got {au.dtype} {au.shape}")
        ts = np.asarray(self.tau_src)
        if ts.shape != (n,) or ts.dtype != _U8:
            raise ValueError(f"{name}.tau_src: expected {_U8} shape ({n},), got {ts.dtype} {ts.shape}")
        if not np.isin(ts, TAU_SRC_VALUES).all():
            raise ValueError(f"{name}.tau_src: entries must be in {TAU_SRC_VALUES}, got {sorted(set(map(int, ts)))}")
        bad = (ts == TAU_SRC_IQ_KT) & (tv == 0)
        if bad.any():
            raise ValueError(f"{name}.tau_src: slots {np.flatnonzero(bad).tolist()} are iq*Kt but tau_valid=0 "
                             "(a substituted torque must be marked valid; leave tau_src=0 when no fresh iq)")

    def to_wire(self) -> dict[str, bytes]:
        out = {f: getattr(self, f).tobytes() for f in self.FIELDS}
        out["tau_valid"] = np.asarray(self.tau_valid, dtype=_U8).tobytes()
        out["age_us"] = np.asarray(self.age_us, dtype=_U32).tobytes()
        out["tau_src"] = np.asarray(self.tau_src, dtype=_U8).tobytes()
        return out

    @classmethod
    def from_wire(cls, d: dict, n: int, name: str) -> "MotorStateBlock":
        kw = {f: _from_bin(d[f], _U8 if f == "mode" else _F32, n, f"{name}.{f}") for f in cls.FIELDS}
        if "tau_valid" in d or "age_us" in d:
            if "tau_valid" not in d or "age_us" not in d:
                raise ValueError(f"{name}: 'tau_valid' and 'age_us' must both be present or both absent")
            kw["tau_valid"] = _from_bin(d["tau_valid"], _U8, n, f"{name}.tau_valid")
            kw["age_us"] = _from_bin(d["age_us"], _U32, n, f"{name}.age_us")
        if "tau_src" in d:
            if "tau_valid" not in d:
                raise ValueError(f"{name}: 'tau_src' requires 'tau_valid' and 'age_us'")
            kw["tau_src"] = _from_bin(d["tau_src"], _U8, n, f"{name}.tau_src")
        return cls(**kw)

    @staticmethod
    def wire_has_health(d: dict) -> bool:
        """True when the motor map carries both health keys (post item-85 sender)."""
        return "tau_valid" in d and "age_us" in d

    @staticmethod
    def wire_has_src(d: dict) -> bool:
        """True when the motor map carries ``tau_src`` (post item-325 sender)."""
        return "tau_src" in d

    def _canonical(self, health: bool = True, src: bool = True) -> bytes:
        body = b"".join(getattr(self, f).tobytes() for f in self.FIELDS)
        if not health:
            return body
        body += np.asarray(self.tau_valid, dtype=_U8).tobytes() + np.asarray(self.age_us, dtype=_U32).tobytes()
        return body + np.asarray(self.tau_src, dtype=_U8).tobytes() if src else body


# --------------------------------------------------------------------------------------
# IMU and privileged simulation state
# --------------------------------------------------------------------------------------


def quat_wxyz_to_rpy(q: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw [rad] (intrinsic Z-Y-X) of a unit wxyz quaternion."""
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=_F32)


@dataclass
class IMUState:
    """Torso IMU (root body ``world``); see the module docstring for frames, units and ``valid`` / ``age_us``."""

    quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], _F32))
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3, _F32))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3, _F32))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3, _F32))
    valid: int = 0
    """1 = live configured sample, 0 = no/dead/unconfigured IMU (default: a fresh IMUState is NOT a reading)."""
    age_us: int = IMU_AGE_UNKNOWN
    """Sample -> publish age [us]; :data:`IMU_AGE_UNKNOWN` when unknown (default)."""

    FIELDS = ("quat_wxyz", "gyro", "accel", "rpy")
    _SIZES = {"quat_wxyz": 4, "gyro": 3, "accel": 3, "rpy": 3}
    _WIRE = {"quat_wxyz": "quat_wxyz", "gyro": "gyro", "accel": "acc", "rpy": "rpy"}
    """Wire keys follow CONTRACTS section 6 (``imu{quat_wxyz, gyro, acc}``); ``valid`` / ``age_us`` are ints."""
    HEALTH_KEYS = ("valid", "age_us")

    # Unitree ``IMUState_`` aliases (``quaternion`` is wxyz in Unitree too).
    quaternion = property(lambda s: s.quat_wxyz)
    gyroscope = property(lambda s: s.gyro)
    accelerometer = property(lambda s: s.accel)
    acc = property(lambda s: s.accel)

    @property
    def age_s(self) -> float:
        """``age_us`` in seconds (``inf`` when unknown)."""
        return math.inf if int(self.age_us) >= IMU_AGE_UNKNOWN else int(self.age_us) * 1e-6

    def validate(self) -> None:
        for f in self.FIELDS:
            arr = getattr(self, f)
            if arr.shape != (self._SIZES[f],) or arr.dtype != _F32:
                raise ValueError(f"imu.{f}: expected float32 shape ({self._SIZES[f]},), got {arr.dtype} {arr.shape}")
        if int(self.valid) not in (0, 1):
            raise ValueError(f"imu.valid: expected 0 or 1, got {self.valid!r}")
        if not 0 <= int(self.age_us) <= IMU_AGE_UNKNOWN:
            raise ValueError(f"imu.age_us: expected uint32, got {self.age_us!r}")

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {self._WIRE[f]: getattr(self, f).tobytes() for f in self.FIELDS}
        out["valid"], out["age_us"] = int(self.valid), int(self.age_us)
        return out

    @classmethod
    def from_wire(cls, d: dict) -> "IMUState":
        """Decode; a pre-0004 sender (no ``valid`` / ``age_us`` keys) decodes as INVALID with unknown age."""
        kw = {f: _from_bin(d[cls._WIRE[f]], _F32, cls._SIZES[f], f"imu.{f}") for f in cls.FIELDS}
        if "valid" in d or "age_us" in d:
            if "valid" not in d or "age_us" not in d:
                raise ValueError("imu: 'valid' and 'age_us' must both be present or both absent")
            kw["valid"], kw["age_us"] = int(d["valid"]), int(d["age_us"])
        msg = cls(**kw)
        msg.validate()
        return msg

    @staticmethod
    def wire_has_health(d: dict) -> bool:
        return "valid" in d and "age_us" in d

    def _canonical(self, health: bool = True) -> bytes:
        body = b"".join(getattr(self, f).tobytes() for f in self.FIELDS)
        return body + struct.pack("<II", int(self.valid), int(self.age_us)) if health else body


@dataclass
class SimState:
    """Privileged simulation ground truth (world frame, z up). Not available on hardware.

    Attributes:
        time_s: simulated time since reset [s].
        root_pos_w: root body (``world``) frame origin [m].
        root_quat_w: root body orientation, wxyz.
        root_lin_vel_w: root body centre-of-mass linear velocity [m/s].
        root_ang_vel_w: root body angular velocity [rad/s].
        body_names: names of extra bodies whose poses follow (may be empty).
        body_pos_w: (k, 3) extra body frame origins [m].
        body_quat_w: (k, 4) extra body orientations, wxyz.
    """

    time_s: float = 0.0
    root_pos_w: np.ndarray = field(default_factory=lambda: np.zeros(3, _F32))
    root_quat_w: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], _F32))
    root_lin_vel_w: np.ndarray = field(default_factory=lambda: np.zeros(3, _F32))
    root_ang_vel_w: np.ndarray = field(default_factory=lambda: np.zeros(3, _F32))
    body_names: tuple[str, ...] = ()
    body_pos_w: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), _F32))
    body_quat_w: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), _F32))

    _VEC = {"root_pos_w": 3, "root_quat_w": 4, "root_lin_vel_w": 3, "root_ang_vel_w": 3}

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(pos_w, quat_wxyz)`` of an extra body by name."""
        i = self.body_names.index(name)
        return self.body_pos_w[i], self.body_quat_w[i]

    def to_wire(self) -> dict[str, Any]:
        k = len(self.body_names)
        if self.body_pos_w.shape != (k, 3) or self.body_quat_w.shape != (k, 4):
            raise ValueError("sim body arrays do not match body_names")
        out: dict[str, Any] = {"time_s": float(self.time_s)}
        for f in self._VEC:
            out[f] = np.asarray(getattr(self, f), _F32).tobytes()
        out["body_names"] = list(self.body_names)
        out["body_pos_w"] = np.asarray(self.body_pos_w, _F32).tobytes()
        out["body_quat_w"] = np.asarray(self.body_quat_w, _F32).tobytes()
        return out

    @classmethod
    def from_wire(cls, d: dict) -> "SimState":
        names = tuple(str(n) for n in d.get("body_names", ()))
        k = len(names)
        kwargs = {f: _from_bin(d[f], _F32, n, f"sim.{f}") for f, n in cls._VEC.items()}
        return cls(time_s=float(d["time_s"]), body_names=names,
                   body_pos_w=_from_bin(d["body_pos_w"], _F32, 3 * k, "sim.body_pos_w").reshape(k, 3),
                   body_quat_w=_from_bin(d["body_quat_w"], _F32, 4 * k, "sim.body_quat_w").reshape(k, 4),
                   **kwargs)

    def _canonical(self) -> bytes:
        parts = [struct.pack("<d", float(self.time_s))]
        parts += [np.asarray(getattr(self, f), _F32).tobytes() for f in self._VEC]
        parts += [("\0".join(self.body_names)).encode(), np.asarray(self.body_pos_w, _F32).tobytes(),
                  np.asarray(self.body_quat_w, _F32).tobytes()]
        return b"".join(parts)


# --------------------------------------------------------------------------------------
# LowCmd / LowState
# --------------------------------------------------------------------------------------


def _header(tick: int, stamp_ns: int) -> bytes:
    return struct.pack("<Iq", int(tick) & 0xFFFFFFFF, int(stamp_ns))


def _check_envelope(d: Any, kind: str) -> dict:
    if not isinstance(d, dict):
        raise ValueError(f"{kind}: payload is not a map")
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{kind}: schema {d.get('schema')!r} != {SCHEMA!r}")
    if d.get("type") != kind:
        raise ValueError(f"expected type {kind!r}, got {d.get('type')!r}")
    return d


@dataclass
class LowCmd:
    """Low-level command (Unitree ``LowCmd_`` equivalent).

    Attributes:
        tick: LowState tick this command was computed from (0 if unknown).
        stamp_ns: sender ``time.perf_counter_ns()`` (same-host clock).
        motor: 22 body-motor commands in motor-contract order.
        neck: optional 6 neck lead-screw commands (SDK slots 22-27); ``None`` keeps
            the robot side's current neck targets.
        crc: CRC32 filled by :meth:`to_bytes` and checked by :meth:`from_bytes`.
    """

    tick: int = 0
    stamp_ns: int = 0
    motor: MotorCmdBlock = field(default_factory=lambda: MotorCmdBlock.zeros(NUM_MOTORS))
    neck: MotorCmdBlock | None = None
    crc: int = 0

    @property
    def motor_cmd(self) -> MotorCmdBlock:
        """Unitree-style alias: ``cmd.motor_cmd[i].q = ...``."""
        return self.motor

    @property
    def neck_cmd(self) -> MotorCmdBlock | None:
        return self.neck

    def validate(self) -> None:
        self.motor.validate(NUM_MOTORS, "motor")
        if self.neck is not None:
            self.neck.validate(NUM_NECK, "neck")

    def _canonical(self) -> bytes:
        parts = [_header(self.tick, self.stamp_ns), self.motor._canonical()]
        if self.neck is not None:
            parts.append(self.neck._canonical())
        return b"".join(parts)

    def compute_crc(self) -> int:
        return crc32_bytes(self._canonical())

    def to_bytes(self) -> bytes:
        """Validate, fill :attr:`crc` and serialize to msgpack."""
        self.validate()
        self.crc = self.compute_crc()
        return msgpack.packb({
            "schema": SCHEMA, "type": "LowCmd", "tick": int(self.tick) & 0xFFFFFFFF,
            "stamp_ns": int(self.stamp_ns), "motor": self.motor.to_wire(),
            "neck": None if self.neck is None else self.neck.to_wire(), "crc": self.crc,
        }, use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes, verify_crc: bool = True) -> "LowCmd":
        """Decode msgpack; raise ``ValueError`` on malformed data and :class:`CrcError` on CRC mismatch."""
        d = _check_envelope(msgpack.unpackb(data, raw=False), "LowCmd")
        neck = d.get("neck")
        msg = cls(tick=int(d["tick"]), stamp_ns=int(d["stamp_ns"]),
                  motor=MotorCmdBlock.from_wire(d["motor"], NUM_MOTORS, "motor"),
                  neck=None if neck is None else MotorCmdBlock.from_wire(neck, NUM_NECK, "neck"),
                  crc=int(d["crc"]))
        if verify_crc and msg.compute_crc() != msg.crc:
            raise CrcError(f"LowCmd CRC mismatch: got {msg.crc:#010x}, computed {msg.compute_crc():#010x}")
        return msg


@dataclass
class LowState:
    """Low-level state (Unitree ``LowState_`` equivalent).

    Attributes:
        tick: control tick counter (500 Hz in the Newton bridge).
        stamp_ns: sender ``time.perf_counter_ns()`` (same-host clock).
        imu: torso IMU.
        motor: 22 body-motor states in motor-contract order.
        neck: optional 6 neck lead-screw states.
        sim: optional privileged simulation ground truth (never on hardware).
        crc: CRC32 filled by :meth:`to_bytes` and checked by :meth:`from_bytes`.
    """

    tick: int = 0
    stamp_ns: int = 0
    imu: IMUState = field(default_factory=IMUState)
    motor: MotorStateBlock = field(default_factory=lambda: MotorStateBlock.zeros(NUM_MOTORS))
    neck: MotorStateBlock | None = None
    sim: SimState | None = None
    crc: int = 0

    @property
    def motor_state(self) -> MotorStateBlock:
        """Unitree-style alias: ``state.motor_state[i].q``."""
        return self.motor

    @property
    def imu_state(self) -> IMUState:
        """Unitree-style alias."""
        return self.imu

    def validate(self) -> None:
        self.imu.validate()
        self.motor.validate(NUM_MOTORS, "motor")
        if self.neck is not None:
            self.neck.validate(NUM_NECK, "neck")

    def _canonical(self, imu_health: bool = True, motor_health: bool = True, motor_src: bool = True,
                   neck_src: bool = True) -> bytes:
        parts = [_header(self.tick, self.stamp_ns), self.imu._canonical(imu_health),
                 self.motor._canonical(motor_health, motor_src)]
        if self.neck is not None:
            parts.append(self.neck._canonical(motor_health, neck_src))
        if self.sim is not None:
            parts.append(self.sim._canonical())
        return b"".join(parts)

    def compute_crc(self, imu_health: bool = True, motor_health: bool = True, motor_src: bool = True,
                    neck_src: bool = True) -> int:
        return crc32_bytes(self._canonical(imu_health, motor_health, motor_src, neck_src))

    def to_bytes(self) -> bytes:
        """Validate, fill :attr:`crc` and serialize to msgpack."""
        self.validate()
        self.crc = self.compute_crc()
        return msgpack.packb({
            "schema": SCHEMA, "type": "LowState", "tick": int(self.tick) & 0xFFFFFFFF,
            "stamp_ns": int(self.stamp_ns), "imu": self.imu.to_wire(), "motor": self.motor.to_wire(),
            "neck": None if self.neck is None else self.neck.to_wire(),
            "sim": None if self.sim is None else self.sim.to_wire(), "crc": self.crc,
        }, use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes, verify_crc: bool = True) -> "LowState":
        """Decode msgpack; raise ``ValueError`` on malformed data and :class:`CrcError` on CRC mismatch."""
        d = _check_envelope(msgpack.unpackb(data, raw=False), "LowState")
        neck, sim = d.get("neck"), d.get("sim")
        # pre-0004 / pre-item-85 senders: CRC over the legacy canonical bytes (no health fields)
        imu_h = IMUState.wire_has_health(d["imu"])
        mot_h = MotorStateBlock.wire_has_health(d["motor"])
        # pre-item-325 senders: no tau_src key -> CRC without it (decoded as all TAU_SRC_DIRECT)
        src = {"motor_src": MotorStateBlock.wire_has_src(d["motor"]),
               "neck_src": neck is not None and MotorStateBlock.wire_has_src(neck)}
        msg = cls(tick=int(d["tick"]), stamp_ns=int(d["stamp_ns"]), imu=IMUState.from_wire(d["imu"]),
                  motor=MotorStateBlock.from_wire(d["motor"], NUM_MOTORS, "motor"),
                  neck=None if neck is None else MotorStateBlock.from_wire(neck, NUM_NECK, "neck"),
                  sim=None if sim is None else SimState.from_wire(sim), crc=int(d["crc"]))
        if verify_crc and msg.compute_crc(imu_health=imu_h, motor_health=mot_h, **src) != msg.crc:
            raise CrcError(
                f"LowState CRC mismatch: got {msg.crc:#010x}, "
                f"computed {msg.compute_crc(imu_health=imu_h, motor_health=mot_h, **src):#010x}")
        return msg
