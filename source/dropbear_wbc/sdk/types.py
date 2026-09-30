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
    * ``SimState`` is privileged, simulation-only ground truth in the world
      frame. It never exists on hardware; policies that need it are sim-only.

Wire format (msgpack map, one message per ZMQ frame)::

    {"schema": "dropbear_hg-v1", "type": "LowCmd"|"LowState", "tick": uint32,
     "stamp_ns": int64, <blocks as maps of little-endian raw array bytes>, "crc": uint32}

``tick`` in a LowState is the simulator/robot control tick (500 Hz). In a LowCmd
it echoes the LowState tick the command was computed from (0 if unknown), which
lets the robot side measure command latency in ticks. ``stamp_ns`` is
``time.perf_counter_ns()`` at send time; it is only comparable between
processes on the same host (Windows QueryPerformanceCounter is system-wide).

CRC: Unitree's CRC32 (see :mod:`dropbear_wbc.sdk.crc`) over the canonical byte
string: ``<I tick``, ``<q stamp_ns``, then every array's little-endian bytes in
the fixed field order used by :meth:`LowCmd._canonical` /
:meth:`LowState._canonical`.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import msgpack
import numpy as np

from .crc import crc32_bytes
from .motors import NUM_MOTORS, NUM_NECK

SCHEMA = "dropbear_hg-v1"


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

    def __repr__(self) -> str:
        return (f"MotorState(mode={self.mode}, q={self.q:.4f}, dq={self.dq:.4f}, ddq={self.ddq:.3f}, "
                f"tau_est={self.tau_est:.3f}, temperature={self.temperature:.1f})")


@dataclass
class MotorStateBlock:
    """Struct-of-arrays motor feedback (``n`` slots).

    Attributes:
        mode: uint8 mode the motor is currently in (echo of the applied command mode).
        q: measured position [rad | m].
        dq: measured velocity [rad/s | m/s].
        ddq: acceleration [rad/s^2 | m/s^2] (finite difference in simulation).
        tau_est: estimated output torque [N*m | N] (simulation: last applied, clipped torque).
        temperature: winding temperature [degC] (simulation: constant).
    """

    mode: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau_est: np.ndarray
    temperature: np.ndarray

    FIELDS = ("mode", "q", "dq", "ddq", "tau_est", "temperature")

    @classmethod
    def zeros(cls, n: int) -> "MotorStateBlock":
        return cls(mode=np.zeros(n, _U8), q=np.zeros(n, _F32), dq=np.zeros(n, _F32), ddq=np.zeros(n, _F32),
                   tau_est=np.zeros(n, _F32), temperature=np.zeros(n, _F32))

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

    def to_wire(self) -> dict[str, bytes]:
        return {f: getattr(self, f).tobytes() for f in self.FIELDS}

    @classmethod
    def from_wire(cls, d: dict, n: int, name: str) -> "MotorStateBlock":
        return cls(**{f: _from_bin(d[f], _U8 if f == "mode" else _F32, n, f"{name}.{f}") for f in cls.FIELDS})

    def _canonical(self) -> bytes:
        return b"".join(getattr(self, f).tobytes() for f in self.FIELDS)


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
    """Torso IMU (root body ``world``); see the module docstring for frames and units."""

    quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], _F32))
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3, _F32))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3, _F32))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3, _F32))

    FIELDS = ("quat_wxyz", "gyro", "accel", "rpy")
    _SIZES = {"quat_wxyz": 4, "gyro": 3, "accel": 3, "rpy": 3}
    _WIRE = {"quat_wxyz": "quat_wxyz", "gyro": "gyro", "accel": "acc", "rpy": "rpy"}
    """Wire keys follow CONTRACTS section 6 (``imu{quat_wxyz, gyro, acc}``)."""

    # Unitree ``IMUState_`` aliases (``quaternion`` is wxyz in Unitree too).
    quaternion = property(lambda s: s.quat_wxyz)
    gyroscope = property(lambda s: s.gyro)
    accelerometer = property(lambda s: s.accel)
    acc = property(lambda s: s.accel)

    def validate(self) -> None:
        for f in self.FIELDS:
            arr = getattr(self, f)
            if arr.shape != (self._SIZES[f],) or arr.dtype != _F32:
                raise ValueError(f"imu.{f}: expected float32 shape ({self._SIZES[f]},), got {arr.dtype} {arr.shape}")

    def to_wire(self) -> dict[str, bytes]:
        return {self._WIRE[f]: getattr(self, f).tobytes() for f in self.FIELDS}

    @classmethod
    def from_wire(cls, d: dict) -> "IMUState":
        return cls(**{f: _from_bin(d[cls._WIRE[f]], _F32, cls._SIZES[f], f"imu.{f}") for f in cls.FIELDS})

    def _canonical(self) -> bytes:
        return b"".join(getattr(self, f).tobytes() for f in self.FIELDS)


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

    def _canonical(self) -> bytes:
        parts = [_header(self.tick, self.stamp_ns), self.imu._canonical(), self.motor._canonical()]
        if self.neck is not None:
            parts.append(self.neck._canonical())
        if self.sim is not None:
            parts.append(self.sim._canonical())
        return b"".join(parts)

    def compute_crc(self) -> int:
        return crc32_bytes(self._canonical())

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
        msg = cls(tick=int(d["tick"]), stamp_ns=int(d["stamp_ns"]), imu=IMUState.from_wire(d["imu"]),
                  motor=MotorStateBlock.from_wire(d["motor"], NUM_MOTORS, "motor"),
                  neck=None if neck is None else MotorStateBlock.from_wire(neck, NUM_NECK, "neck"),
                  sim=None if sim is None else SimState.from_wire(sim), crc=int(d["crc"]))
        if verify_crc and msg.compute_crc() != msg.crc:
            raise CrcError(f"LowState CRC mismatch: got {msg.crc:#010x}, computed {msg.compute_crc():#010x}")
        return msg
