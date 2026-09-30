"""Tracking observations (BeyondMimic port). Anchor quantities use the offset anchor frame of the command.

Rotations are encoded as the first two columns of the rotation matrix (6D). Positions in metres.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _cmd(env: ManagerBasedEnv, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot anchor orientation in world frame, 6D (num_envs, 6)."""
    mat = matrix_from_quat(_cmd(env, command_name).robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot anchor linear velocity in world frame [m/s] (num_envs, 3)."""
    return _cmd(env, command_name).robot_anchor_lin_vel_w.view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot anchor angular velocity in world frame [rad/s] (num_envs, 3)."""
    return _cmd(env, command_name).robot_anchor_ang_vel_w.view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Tracked body positions in the robot anchor frame [m] (num_envs, 3 * num_bodies)."""
    command = _cmd(env, command_name)
    n = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, n, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, n, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Tracked body orientations in the robot anchor frame, 6D (num_envs, 6 * num_bodies)."""
    command = _cmd(env, command_name)
    n = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, n, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, n, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference anchor position in the robot anchor frame [m] (num_envs, 3)."""
    command = _cmd(env, command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference anchor orientation in the robot anchor frame, 6D (num_envs, 6)."""
    command = _cmd(env, command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)
