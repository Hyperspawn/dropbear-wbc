"""Real-sensor error model for the Newton bridge LowState (numpy only; mirror of ``tasks/tracking/mdp/sensor_dr``).

The default plant publishes perfect, zero-age sensors. The robot will not: the torso IMU on the center ESP32 is an
MPU-6050-class part read raw over I2C (no fusion on the board, no magnetometer), and joint state arrives over CAN
with a reply-format quantum and an age of up to one command period. This module turns a perfect LowState into a
plausible measured one so sim2sim evaluation sees the same errors training should be randomized over.

Per episode (``reset``), one sample of:
    * IMU mounting/tilt misalignment ``R_mis`` (roll/pitch, uniform +-``tilt_rad``);
    * residual gyro bias (uniform +-``gyro_bias``, per axis, after a boot still-stand calibration);
    * heading drift rate (uniform +-``yaw_drift_rate``): a 6-axis fusion integrates yaw from the gyro;
    * accelerometer bias (uniform +-``accel_bias``);
    * IMU and joint-state ages in policy steps (uniform, fractional; linear interpolation between steps). Call the
      model once per POLICY step (50 Hz), not per bridge tick, or the ages shrink 10x. Ranges: notes/39 (item 39).
Per tick: gyro bias random walk, white noise, full-scale clipping, LSB quantization; joint q/dq quantization and
optional stale hold (a motor's state refreshes with probability ``joint_update_prob`` per tick: polled state).

Measured orientation: ``q_meas = Rz(psi0 + rate * t) * q_true * q_mis``; gyro/accel are expressed in the
misaligned sensor frame: ``w_meas = R_mis^T w_body + b``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

G = 9.81


@dataclass(frozen=True)
class SensorErrorCfg:
    # IMU (MPU-6000/6050 PS rev 3.4 where a number is cited; ranges are post-calibration residuals)
    tilt_rad: float = math.radians(3.0)
    """Mounting + accel-offset tilt, uniform per axis (roll, pitch). +-50 mg uncalibrated offset alone = 2.9 deg."""
    gyro_bias: float = 0.02
    """Residual gyro bias [rad/s] after boot calibration (initial ZRO is +-20 dps = 0.35 rad/s uncalibrated)."""
    gyro_bias_walk: float = 5e-4
    """Gyro bias random walk [rad/s/sqrt(s)] (warm-up/temperature drift; unmeasured)."""
    gyro_noise: float = 3e-3
    """White gyro noise [rad/s rms per sample]; datasheet 0.005 dps/sqrt(Hz) at the 260 Hz default DLPF = 1.8e-3."""
    gyro_full_scale: float = math.radians(250.0)
    """Default FS_SEL=0 (no GYRO_CONFIG write in neck.ino)."""
    gyro_lsb: float = math.radians(1.0 / 131.0)
    accel_bias: float = 0.3
    accel_noise: float = 0.08
    accel_full_scale: float = 2.0 * G
    accel_lsb: float = G / 16384.0
    yaw_drift_rate: float = 0.02
    """Heading drift [rad/s] of 6-axis fusion = residual z bias integrated; 0.02 rad/s = 11.5 deg per 10 s clip."""
    imu_age_ticks: tuple[float, float] = (0.35, 1.7)
    """IMU sample (incl. DLPF group delay) -> policy input, in policy steps: 7-34 ms. notes/39 chain model, item-8
    target over DBI v1 at 115200: 13.6-33 ms (5 ch) / 8.9-28 ms (torso 1 ch); never below ~7 ms (DLPF 4.8-5.9 ms +
    wire). Per episode: the age is a boot-phase constant (within-session std ~1 ms). Replace after bench S39."""
    # joint state
    joint_q_lsb: float = 0.38e-3
    """0x400/0x500 motion-mode reply angle quantum [rad]; the 0xA1/A2/A4 replies are 1 deg = 17.5e-3 (item 34)."""
    joint_dq_lsb: float = 22e-3
    joint_age_ticks: tuple[float, float] = (0.0, 0.5)
    """Encoder sample -> policy input, in policy steps (0-10 ms). notes/39: 0.6-8.1 ms with a 200 Hz MIT round,
    12.8 ms max with 1 ms round jitter + 2 ms USB. The old (0, 1) = 0-20 ms assumed reply-per-policy-step."""
    joint_update_prob: float = 1.0
    """Per-tick probability a motor's state refreshes (0.034 = deployed 0x92 poll ~1.7 Hz/motor at 50 Hz)."""


PROFILES: dict[str, SensorErrorCfg] = {
    "ideal": SensorErrorCfg(tilt_rad=0, gyro_bias=0, gyro_bias_walk=0, gyro_noise=0, gyro_full_scale=math.inf,
                            gyro_lsb=0, accel_bias=0, accel_noise=0, accel_full_scale=math.inf, accel_lsb=0,
                            yaw_drift_rate=0, imu_age_ticks=(0, 0), joint_q_lsb=0, joint_dq_lsb=0,
                            joint_age_ticks=(0, 0), joint_update_prob=1.0),
    "motion_mode": SensorErrorCfg(),
}
PROFILES["a1_reply"] = replace(PROFILES["motion_mode"], joint_q_lsb=math.radians(1.0), joint_dq_lsb=math.radians(1.0))
PROFILES["deployed_054"] = replace(PROFILES["a1_reply"], joint_update_prob=1.7 / 50.0, imu_age_ticks=(0.0, 0.75),
                                   joint_age_ticks=(0.0, 1.0))
PROFILES["center_0010"] = replace(PROFILES["motion_mode"], gyro_full_scale=math.radians(1000.0),
                                  gyro_lsb=math.radians(1.0 / 32.8), accel_full_scale=8.0 * G, accel_lsb=G / 4096.0)
"""Center board with Dropbear-0010 (FS_SEL 2 +-1000 dps, AFS_SEL 2 +-8 g, DLPF 3)."""
"""Firmware .54 as deployed: 0x92 poll, ~1.7 Hz per motor."""


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qaxis(axis: np.ndarray, ang: float) -> np.ndarray:
    return np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * np.asarray(axis, float)])


def _rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quantize(x: np.ndarray, lsb: float, full_scale: float = math.inf) -> np.ndarray:
    x = np.clip(x, -full_scale, full_scale)
    return x if lsb <= 0 else np.round(x / lsb) * lsb


class FracDelay:
    """Ring buffer returning each channel group ``lag`` ticks in the past (fractional: linear interpolation)."""

    def __init__(self, max_lag: float):
        self.n = int(math.ceil(max_lag)) + 1
        self.buf: list[np.ndarray] = []

    def reset(self) -> None:
        self.buf = []

    def push(self, x: np.ndarray, lag: float) -> np.ndarray:
        x = np.asarray(x, float)
        if not self.buf:
            self.buf = [x.copy() for _ in range(self.n)]
        self.buf = [x.copy()] + self.buf[:-1]
        i0 = int(math.floor(lag))
        a = lag - i0
        i1 = min(i0 + 1, self.n - 1)
        return (1 - a) * self.buf[i0] + a * self.buf[i1]


@dataclass
class _Episode:
    q_mis: np.ndarray
    gyro_b: np.ndarray
    accel_b: np.ndarray
    psi0: float
    yaw_rate: float
    imu_lag: float
    joint_lag: float


@dataclass
class SensorErrorModel:
    cfg: SensorErrorCfg = field(default_factory=SensorErrorCfg)
    tick_dt: float = 0.02
    seed: int | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        c = self.cfg
        self._imu_d = FracDelay(c.imu_age_ticks[1])
        self._jq_d = FracDelay(c.joint_age_ticks[1])
        self.reset()

    def reset(self) -> _Episode:
        c, r = self.cfg, self.rng
        u = lambda a, n=None: r.uniform(-a, a, n) if a > 0 else (np.zeros(n) if n else 0.0)  # noqa: E731
        roll, pitch = u(c.tilt_rad), u(c.tilt_rad)
        q_mis = _qmul(_qaxis([1, 0, 0], roll), _qaxis([0, 1, 0], pitch))
        self.ep = _Episode(q_mis=q_mis, gyro_b=u(c.gyro_bias, 3), accel_b=u(c.accel_bias, 3), psi0=0.0,
                           yaw_rate=u(c.yaw_drift_rate), imu_lag=float(r.uniform(*c.imu_age_ticks)),
                           joint_lag=float(r.uniform(*c.joint_age_ticks)))
        self.t = 0.0
        self._imu_d.reset()
        self._jq_d.reset()
        self._held: np.ndarray | None = None
        return self.ep

    def imu(self, quat_wxyz, gyro_b, accel_b):
        """Perfect body-frame IMU -> measured (quat_wxyz, gyro, accel). Call once per control tick."""
        c, ep, r = self.cfg, self.ep, self.rng
        if c.gyro_bias_walk > 0:
            ep.gyro_b = ep.gyro_b + c.gyro_bias_walk * math.sqrt(self.tick_dt) * r.standard_normal(3)
        rm = _rot(ep.q_mis)
        q = _qmul(_qmul(_qaxis([0, 0, 1], ep.psi0 + ep.yaw_rate * self.t), np.asarray(quat_wxyz, float)), ep.q_mis)
        g = rm.T @ np.asarray(gyro_b, float) + ep.gyro_b + c.gyro_noise * r.standard_normal(3)
        a = rm.T @ np.asarray(accel_b, float) + ep.accel_b + c.accel_noise * r.standard_normal(3)
        g, a = quantize(g, c.gyro_lsb, c.gyro_full_scale), quantize(a, c.accel_lsb, c.accel_full_scale)
        out = self._imu_d.push(np.concatenate([q, g, a]), ep.imu_lag)
        q = out[:4] / np.linalg.norm(out[:4])
        self.t += self.tick_dt
        return q, out[4:7], out[7:10]

    def imu_age_s(self) -> float:
        """Modelled IMU sample age [s] of the last ``imu()`` output (episode lag x policy tick)."""
        return float(self.ep.imu_lag) * self.tick_dt

    def joints(self, q, dq):
        """Perfect motor (q, dq) -> measured (aged, quantized, optionally stale-held)."""
        c, n = self.cfg, len(q)
        x = self._jq_d.push(np.concatenate([q, dq]), self.ep.joint_lag)
        x = np.concatenate([quantize(x[:n], c.joint_q_lsb), quantize(x[n:], c.joint_dq_lsb)])
        if self._held is None or c.joint_update_prob >= 1.0:
            self._held = x
        else:
            upd = self.rng.random(n) < c.joint_update_prob
            m = np.concatenate([upd, upd])
            self._held = np.where(m, x, self._held)
        return self._held[:n].copy(), self._held[n:].copy()
