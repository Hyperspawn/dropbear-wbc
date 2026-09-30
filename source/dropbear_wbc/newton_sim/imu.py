"""Simulated torso IMU and LowState assembly for the Newton bridge (numpy only).

IMU model (root body ``world``, located at the body's centre of mass):
    * ``quat_wxyz``: body -> world orientation.
    * ``gyro`` [rad/s]: ``R^T w_world``.
    * ``accel`` [m/s^2]: specific force ``R^T (a_com_world - g_world)`` with
      ``g_world = (0, 0, -9.81)``; ``a_com_world`` is the finite difference of the
      COM velocity over one control tick. Reads ``(0, 0, +9.81)`` upright at rest.
    * ``rpy`` [rad]: intrinsic Z-Y-X roll/pitch/yaw.
No noise or bias is added.
"""
from __future__ import annotations

import time

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


class LowStateAssembler:
    """Builds :class:`LowState` messages from plant readouts, keeping finite-difference history.

    Args:
        tick_dt: control tick period [s] used for finite differences.
        publish_sim: include the privileged :class:`SimState` block.
        body_names: names of the extra bodies in the readout (for the sim block).
    """

    def __init__(self, tick_dt: float, publish_sim: bool = True, body_names: tuple[str, ...] = ()):
        self.tick_dt = tick_dt
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
        imu = IMUState(quat_wxyz=f32(q), gyro=f32(rotate_inv_wxyz(q, np.asarray(r.root_ang_vel_w, float))),
                       accel=f32(rotate_inv_wxyz(q, a_w - GRAVITY_W)), rpy=quat_wxyz_to_rpy(q))
        motor = MotorStateBlock(mode=np.asarray(motor_mode, np.uint8), q=f32(r.motor_q), dq=f32(r.motor_dq),
                                ddq=f32(ddq), tau_est=f32(r.motor_tau),
                                temperature=np.full(motors.NUM_MOTORS, SIM_TEMPERATURE_C, np.float32))
        neck = MotorStateBlock(mode=np.ones(motors.NUM_NECK, np.uint8), q=f32(r.neck_q), dq=f32(r.neck_dq),
                               ddq=np.zeros(motors.NUM_NECK, np.float32), tau_est=np.zeros(motors.NUM_NECK, np.float32),
                               temperature=np.full(motors.NUM_NECK, SIM_TEMPERATURE_C, np.float32))
        sim = None
        if self.publish_sim:
            sim = SimState(time_s=float(r.time_s), root_pos_w=f32(r.root_pos_w), root_quat_w=f32(q),
                           root_lin_vel_w=f32(lin_v), root_ang_vel_w=f32(r.root_ang_vel_w),
                           body_names=self.body_names, body_pos_w=f32(r.body_pos_w).reshape(-1, 3),
                           body_quat_w=f32(r.body_quat_wxyz).reshape(-1, 4))
        return LowState(tick=int(r.tick), stamp_ns=time.perf_counter_ns(), imu=imu, motor=motor, neck=neck, sim=sim)
