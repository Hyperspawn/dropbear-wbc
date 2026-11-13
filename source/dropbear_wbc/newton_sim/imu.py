"""Simulated torso IMU and LowState assembly for the Newton bridge (numpy only).

IMU model (root body ``world``, located at the body's centre of mass):
    * ``quat_wxyz``: body -> world orientation.
    * ``gyro`` [rad/s]: ``R^T w_world``.
    * ``accel`` [m/s^2]: specific force ``R^T (a_com_world - g_world)`` with
      ``g_world = (0, 0, -9.81)``; ``a_com_world`` is the finite difference of the
      COM velocity over one control tick. Reads ``(0, 0, +9.81)`` upright at rest.
    * ``rpy`` [rad]: intrinsic Z-Y-X roll/pitch/yaw.
No noise or bias is added unless a :class:`~dropbear_wbc.newton_sim.sensor_model.SensorErrorModel` is passed
(then quat/gyro/accel and motor q/dq are the MEASURED values; the privileged SimState block stays truth).
Tick consistency (patch 0081): ``SensorErrorModel`` ages and bias walk are in units of ITS ``tick_dt`` and advance once
per ``imu()`` call. A raw model whose ``tick_dt`` differs from the assembler's (e.g. a 50 Hz model built every 2 ms
bridge tick) silently shrinks every age by ``tick_dt / model.tick_dt`` while ``age_us`` still reports the 50 Hz value,
so the constructor refuses it; wrap it in ``sensor_gate.PolicyGatedSensorModel``. A gated model must satisfy
``model.tick_dt / steps_per_policy == tick_dt`` (1 % tolerance). ``allow_sensor_tick_mismatch=True`` downgrades the
refusal to a warning. Models without a ``tick_dt`` attribute are not checked.
IMU health (patch 0004): ``valid = 1`` always (the simulated sensor cannot die); ``age_us`` = the sensor model's
modelled sample age (``imu_age_s()``), 0 without a model.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from ..sdk import motors
from ..sdk.types import IMUState, LowState, MotorStateBlock, SimState, quat_wxyz_to_rpy

GRAVITY_W = np.array([0.0, 0.0, -9.81])
SIM_TEMPERATURE_C = 25.0


def rotate_inv_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the inverse of the unit wxyz quaternion ``q`` (world -> body)."""
    w, u = float(q[0]), -np.asarray(q[1:4], float)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


SENSOR_TICK_RTOL = 0.01


def sensor_tick_mismatch(tick_dt: float, sensor_model, rtol: float = SENSOR_TICK_RTOL) -> str | None:
    """Why ``sensor_model`` cannot be called once per ``tick_dt`` build, or ``None`` if consistent / not checkable.

    Gated models (``steps_per_policy`` + ``model``) must advance at ``model.tick_dt / steps_per_policy == tick_dt``;
    raw models (``tick_dt`` attribute) must have ``model.tick_dt == tick_dt``.
    """
    if sensor_model is None:
        return None
    inner = getattr(sensor_model, "model", None)
    spp = getattr(sensor_model, "steps_per_policy", None)
    if inner is not None and spp is not None:
        m_dt = getattr(inner, "tick_dt", None)
        if m_dt is None:
            return None
        eff = float(m_dt) / int(spp)
        if abs(eff - tick_dt) > rtol * tick_dt:
            return (f"gated sensor model advances every {spp} builds of {1e3 * tick_dt:.3f} ms = "
                    f"{1e3 * spp * tick_dt:.3f} ms but its tick_dt is {1e3 * float(m_dt):.3f} ms (held-sample age "
                    f"step {1e3 * eff:.3f} ms vs {1e3 * tick_dt:.3f} ms); pick a bridge tick that divides the policy period")
        return None
    m_dt = getattr(sensor_model, "tick_dt", None)
    if m_dt is None:
        return None
    if abs(float(m_dt) - tick_dt) > rtol * tick_dt:
        return (f"raw sensor model tick_dt {1e3 * float(m_dt):.3f} ms != assembler tick_dt {1e3 * tick_dt:.3f} ms: "
                f"every age and the bias walk would run {float(m_dt) / tick_dt:.2f}x fast while age_us reports the "
                f"{1e3 * float(m_dt):.3f} ms-tick value; wrap it in sensor_gate.PolicyGatedSensorModel(model, "
                f"policy_steps_per_bridge(tick_dt, policy_hz))")
    return None


class LowStateAssembler:
    """Builds :class:`LowState` messages from plant readouts, keeping finite-difference history.

    Args:
        tick_dt: control tick period [s] used for finite differences.
        publish_sim: include the privileged :class:`SimState` block.
        body_names: names of the extra bodies in the readout (for the sim block).
        sensor_model: optional measurement model, called once per :meth:`build`; its tick must match ``tick_dt``
            (see :func:`sensor_tick_mismatch`), else ``ValueError``.
        allow_sensor_tick_mismatch: warn instead of raising on a tick mismatch.
    """

    def __init__(self, tick_dt: float, publish_sim: bool = True, body_names: tuple[str, ...] = (),
                 sensor_model=None, allow_sensor_tick_mismatch: bool = False):
        why = sensor_tick_mismatch(tick_dt, sensor_model)
        if why is not None:
            if not allow_sensor_tick_mismatch:
                raise ValueError(why)
            warnings.warn(why, RuntimeWarning, stacklevel=2)
        self.tick_dt = tick_dt
        self.sensor_model = sensor_model
        self.publish_sim = publish_sim
        self.body_names = tuple(body_names)
        self._prev_lin_vel: np.ndarray | None = None
        self._prev_dq: np.ndarray | None = None

    def build(self, r, motor_mode: np.ndarray) -> LowState:
        """Assemble a LowState from a :class:`~dropbear_wbc.newton_sim.plant.PlantReadout`."""
        q = np.asarray(r.root_quat_wxyz, float)
        q = q / np.linalg.norm(q)
        lin_v = np.asarray(r.root_lin_vel_w, float)
        a_w = np.zeros(3) if self._prev_lin_vel is None else (lin_v - self._prev_lin_vel) / self.tick_dt
        ddq = np.zeros(motors.NUM_MOTORS) if self._prev_dq is None else (r.motor_dq - self._prev_dq) / self.tick_dt
        self._prev_lin_vel, self._prev_dq = lin_v, np.asarray(r.motor_dq, float).copy()
        f32 = lambda v: np.asarray(v, dtype=np.float32)  # noqa: E731
        gyro = rotate_inv_wxyz(q, np.asarray(r.root_ang_vel_w, float))
        acc = rotate_inv_wxyz(q, a_w - GRAVITY_W)
        mq, mdq = np.asarray(r.motor_q, float), np.asarray(r.motor_dq, float)
        qm = q
        if self.sensor_model is not None:
            qm, gyro, acc = self.sensor_model.imu(q, gyro, acc)
            mq, mdq = self.sensor_model.joints(mq, mdq)
        age_s = 0.0
        if self.sensor_model is not None and hasattr(self.sensor_model, "imu_age_s"):
            age_s = float(self.sensor_model.imu_age_s())
        imu = IMUState(quat_wxyz=f32(qm), gyro=f32(gyro), accel=f32(acc), rpy=quat_wxyz_to_rpy(f32(qm)),
                       valid=1, age_us=int(round(1e6 * max(0.0, age_s))))
        # Sim applied torque is trusted (tau_valid=1); age_us=0 = sample published in the same tick.
        n_m, n_n = motors.NUM_MOTORS, motors.NUM_NECK
        motor = MotorStateBlock(mode=np.asarray(motor_mode, np.uint8), q=f32(mq), dq=f32(mdq),
                                ddq=f32(ddq), tau_est=f32(r.motor_tau),
                                temperature=np.full(n_m, SIM_TEMPERATURE_C, np.float32),
                                tau_valid=np.ones(n_m, np.uint8), age_us=np.zeros(n_m, np.uint32))
        neck = MotorStateBlock(mode=np.ones(n_n, np.uint8), q=f32(r.neck_q), dq=f32(r.neck_dq),
                               ddq=np.zeros(n_n, np.float32), tau_est=np.zeros(n_n, np.float32),
                               temperature=np.full(n_n, SIM_TEMPERATURE_C, np.float32),
                               tau_valid=np.ones(n_n, np.uint8), age_us=np.zeros(n_n, np.uint32))
        sim = None
        if self.publish_sim:
            sim = SimState(time_s=float(r.time_s), root_pos_w=f32(r.root_pos_w), root_quat_w=f32(q),
                           root_lin_vel_w=f32(lin_v), root_ang_vel_w=f32(r.root_ang_vel_w),
                           body_names=self.body_names, body_pos_w=f32(r.body_pos_w).reshape(-1, 3),
                           body_quat_w=f32(r.body_quat_wxyz).reshape(-1, 4))
        return LowState(tick=int(r.tick), stamp_ns=time.perf_counter_ns(), imu=imu, motor=motor, neck=neck, sim=sim)
