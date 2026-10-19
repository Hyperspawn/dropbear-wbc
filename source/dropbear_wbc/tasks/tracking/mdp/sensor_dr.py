"""Sensor-side domain randomization for the tracking task (torch mirror of ``newton_sim.sensor_model``).

Training today sees perfect, zero-age observations plus white uniform noise; the robot will see an MPU-6050-class
IMU with no magnetometer (heading = integrated gyro), a tilted/misaligned mount, residual gyro bias, and joint state
that is up to one command period old with a reply-format quantum. This module adds those errors to the POLICY group.

* :func:`reset_imu_errors` (EventTerm, mode="reset") samples per env: roll/pitch misalignment, gyro bias, heading
  drift rate, and IMU / joint-state ages in policy steps (fractional).
* :class:`measured_anchor_ori_b` replaces ``motion_anchor_ori_b``: robot anchor orientation is
  ``Rz(rate * t) * q_true * q_mis`` before the frame subtraction, then aged.
* :class:`measured_base_ang_vel` replaces ``base_ang_vel``: ``R_mis^T w + b``, aged (white noise stays on the term).
* :class:`measured_joint` wraps ``joint_pos_rel`` / ``joint_vel_rel``: aged and quantized.

Ages use a short history at the policy rate with linear interpolation between steps (sub-step ages without a
physics-rate buffer). The critic group is untouched (asymmetric actor-critic keeps truth).
Not run under Isaac Lab on the authoring machine: py_compile only. Check on Lab 2.2 before training.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_mul, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_ATTR = "_dropbear_sensor_err"


class _SensorErr:
    def __init__(self, n: int, device):
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.q_mis = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1)
        self.r_mis = torch.eye(3, device=device).repeat(n, 1, 1)
        self.gyro_b, self.yaw_rate, self.imu_lag, self.joint_lag = z(n, 3), z(n), z(n), z(n)
        self.fresh = torch.ones(n, dtype=torch.bool, device=device)


def _err(env: ManagerBasedEnv) -> _SensorErr:
    e = getattr(env, _ATTR, None)
    if e is None:
        e = _SensorErr(env.num_envs, env.device)
        setattr(env, _ATTR, e)
    return e


def reset_imu_errors(env: ManagerBasedEnv, env_ids: torch.Tensor | None, tilt_rad: float = math.radians(3.0),
                     gyro_bias: float = 0.02, yaw_drift_rate: float = 0.02,
                     imu_age_steps: tuple[float, float] = (0.0, 0.75),
                     joint_age_steps: tuple[float, float] = (0.0, 1.0)) -> None:
    """EventTerm (mode="reset"): sample per-episode sensor errors for ``env_ids``."""
    e = _err(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n, dev = len(env_ids), env.device
    u = lambda a, *s: (2 * torch.rand(*s, device=dev) - 1) * a  # noqa: E731
    zero = torch.zeros(n, device=dev)
    q = quat_from_euler_xyz(u(tilt_rad, n), u(tilt_rad, n), zero)
    e.q_mis[env_ids] = q
    e.r_mis[env_ids] = matrix_from_quat(q)
    e.gyro_b[env_ids] = u(gyro_bias, n, 3)
    e.yaw_rate[env_ids] = u(yaw_drift_rate, n)
    lo, hi = imu_age_steps
    e.imu_lag[env_ids] = lo + (hi - lo) * torch.rand(n, device=dev)
    lo, hi = joint_age_steps
    e.joint_lag[env_ids] = lo + (hi - lo) * torch.rand(n, device=dev)
    e.fresh[env_ids] = True


class _FracDelay:
    """Per-env fractional delay at the policy rate (history depth ``ceil(max_lag) + 1``)."""

    def __init__(self, max_lag: float):
        self.k = int(math.ceil(max_lag)) + 1
        self.h: torch.Tensor | None = None
        self.fresh: torch.Tensor | None = None

    def reset(self, env_ids) -> None:
        if self.fresh is not None:
            self.fresh[env_ids] = True

    def __call__(self, x: torch.Tensor, lag: torch.Tensor) -> torch.Tensor:
        if self.h is None or self.h.shape[1:] != x.shape:
            self.h = x.unsqueeze(0).repeat(self.k, 1, 1).clone()
            self.fresh = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        self.h = torch.roll(self.h, 1, dims=0)
        self.h[0] = x
        if self.fresh.any():
            self.h[:, self.fresh] = x[self.fresh].unsqueeze(0)
            self.fresh[:] = False
        lag = lag.clamp(0.0, self.k - 1)
        i0 = lag.floor().long()
        a = (lag - i0).unsqueeze(-1)
        i1 = (i0 + 1).clamp(max=self.k - 1)
        ar = torch.arange(x.shape[0], device=x.device)
        return (1 - a) * self.h[i0, ar] + a * self.h[i1, ar]


class measured_anchor_ori_b(ManagerTermBase):
    """``motion_anchor_ori_b`` as a 6-axis IMU would give it: misaligned, heading-drifting, aged."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._d = _FracDelay(float(cfg.params.get("max_age_steps", 1.0)))

    def reset(self, env_ids=None) -> None:
        self._d.reset(slice(None) if env_ids is None else env_ids)

    def __call__(self, env: ManagerBasedEnv, command_name: str, max_age_steps: float = 1.0) -> torch.Tensor:
        e, cmd = _err(env), env.command_manager.get_term(command_name)
        t = env.episode_length_buf.float() * env.step_dt
        zero = torch.zeros_like(t)
        q_yaw = quat_from_euler_xyz(zero, zero, e.yaw_rate * t)
        q_meas = quat_mul(quat_mul(q_yaw, cmd.robot_anchor_quat_w), e.q_mis)
        _, ori = subtract_frame_transforms(cmd.robot_anchor_pos_w, q_meas, cmd.anchor_pos_w, cmd.anchor_quat_w)
        mat = matrix_from_quat(ori)
        return self._d(mat[..., :2].reshape(mat.shape[0], -1), e.imu_lag)


class measured_base_ang_vel(ManagerTermBase):
    """Root (IMU body) angular velocity in the misaligned sensor frame plus per-episode bias, aged."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._d = _FracDelay(float(cfg.params.get("max_age_steps", 1.0)))

    def reset(self, env_ids=None) -> None:
        self._d.reset(slice(None) if env_ids is None else env_ids)

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 max_age_steps: float = 1.0) -> torch.Tensor:
        e = _err(env)
        w = env.scene[asset_cfg.name].data.root_ang_vel_b
        w = torch.bmm(e.r_mis.transpose(1, 2), w.unsqueeze(-1)).squeeze(-1) + e.gyro_b
        return self._d(w, e.imu_lag)


class measured_joint(ManagerTermBase):
    """Wrap a joint observation (``func``): aged by the per-env joint-state age, quantized by ``lsb``.
    Quantization is applied to the RELATIVE value (grid offset by the default pose; first-order equivalent)."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._d = _FracDelay(float(cfg.params.get("max_age_steps", 1.0)))
        a = cfg.params.get("asset_cfg")
        if isinstance(a, SceneEntityCfg):
            a.resolve(env.scene)

    def reset(self, env_ids=None) -> None:
        self._d.reset(slice(None) if env_ids is None else env_ids)

    def __call__(self, env: ManagerBasedEnv, func, asset_cfg: SceneEntityCfg, lsb: float = 0.0,
                 max_age_steps: float = 1.0) -> torch.Tensor:
        x = self._d(func(env, asset_cfg=asset_cfg), _err(env).joint_lag)
        return torch.round(x / lsb) * lsb if lsb > 0 else x
