"""Tracking rewards (BeyondMimic port, unchanged maths). Errors in metres / radians / SI velocities."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    idx = _get_body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_pos_relative_w[:, idx] - command.robot_body_pos_w[:, idx]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    idx = _get_body_indexes(command, body_names)
    error = quat_error_magnitude(command.body_quat_relative_w[:, idx], command.robot_body_quat_w[:, idx]) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    idx = _get_body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_lin_vel_w[:, idx] - command.robot_body_lin_vel_w[:, idx]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    idx = _get_body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_ang_vel_w[:, idx] - command.robot_body_ang_vel_w[:, idx]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.sum((last_contact_time < threshold) * first_air, dim=-1)


def motion_anchor_height_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """exp(-(z_ref - z)^2 / std^2) of the anchor (head) height above the ground: squat depth and jump height.

    The global anchor term mixes in x/y, where a policy without a position estimate drifts by metres over a clip, so it
    sits near zero and gives no height signal; the relative body terms are measured from the anchor, so a whole-body
    rise or drop is invisible to them (docs/RESULTS.md 2026-10-16)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
    return torch.exp(-error / std**2)


def motion_feet_height_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """exp(-mean_feet (z_ref - z)^2 / std^2) of the foot heights above the ground: lifting a foot for a kick or one-leg
    stance, both feet leaving the ground in a jump, and keeping a planted foot down."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    idx = _get_body_indexes(command, body_names)
    error = torch.square(command.body_pos_w[:, idx, 2] - command.robot_body_pos_w[:, idx, 2])
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_pos_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """exp(-mean_motors (q_ref - q)^2 / std^2) over the 22 motor joints: follow the reference's joint angles (knee bend,
    hip lift) directly, not only through the body points."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.joint_pos - command.robot_joint_pos)
    return torch.exp(-error.mean(-1) / std**2)

