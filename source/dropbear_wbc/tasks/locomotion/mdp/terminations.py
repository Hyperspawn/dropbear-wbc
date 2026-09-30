"""Velocity-task terminations on the anchor (chest) body (never the root frame height, which is below the soles)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import anchor_projected_gravity, ground_z

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def anchor_height_below(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, minimum_height: float,
                        ground_sensor: str | None = None) -> torch.Tensor:
    """Anchor link z above the ground (``rewards.ground_z``: env origin on flat ground) below ``minimum_height``."""
    asset = env.scene[asset_cfg.name]
    z = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2] - ground_z(env, ground_sensor)
    return z < minimum_height


def anchor_tilt_above(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_tilt_rad: float, up_offset_wxyz=(1.0, 0.0, 0.0, 0.0)
) -> torch.Tensor:
    """Tilt of the world-aligned anchor frame's z axis from vertical above ``max_tilt_rad`` (roll and pitch)."""
    g = anchor_projected_gravity(env, asset_cfg, up_offset_wxyz)
    return g[:, 2] > -math.cos(max_tilt_rad)
