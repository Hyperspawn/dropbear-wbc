"""Tracking terminations (BeyondMimic port).

``bad_anchor_ori`` compares the z component of gravity projected into a *world-aligned* anchor frame
(``anchor link frame * up_offset_wxyz``). Upstream used the raw link frame, which for G1's ``torso_link`` is
z-up at rest; Dropbear's anchor link frame has its z axis pointing forward at rest (quat 0.5,0.5,0.5,0.5),
so without the offset the check would only see pitch, not roll.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
    up_offset_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    """|g_z(reference) - g_z(robot)| > threshold, gravity projected into ``anchor link frame * up_offset``."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    off = torch.tensor(up_offset_wxyz, dtype=torch.float32, device=env.device).expand(env.num_envs, 4)
    motion_q = math_utils.quat_mul(command.anchor_quat_w, off)
    robot_q = math_utils.quat_mul(command.robot_anchor_quat_w, off)
    motion_g = math_utils.quat_apply_inverse(motion_q, asset.data.GRAVITY_VEC_W)
    robot_g = math_utils.quat_apply_inverse(robot_q, asset.data.GRAVITY_VEC_W)
    return (motion_g[:, 2] - robot_g[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    idx = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, idx] - command.robot_body_pos_w[:, idx], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    idx = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, idx, -1] - command.robot_body_pos_w[:, idx, -1])
    return torch.any(error > threshold, dim=-1)
