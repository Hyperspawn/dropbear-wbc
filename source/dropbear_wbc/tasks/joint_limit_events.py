"""Startup event for the explicit end-stop model (``hw_motor_specs.STOP_MODEL``): move PhysX's rigid joint limits of the
modeled joints outward so that only the actuator's stop spring acts (its torque is then observable)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def widen_joint_limits(env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, margin_rad: float):
    asset = env.scene[asset_cfg.name]
    if getattr(asset.data, "stop_model_limits_widened", False):
        return
    ids = asset_cfg.joint_ids
    lim = asset.data.default_joint_pos_limits[:, ids].clone()
    lim[..., 0] -= float(margin_rad)
    lim[..., 1] += float(margin_rad)
    asset.write_joint_position_limit_to_sim(lim, joint_ids=ids, warn_limit_violation=False)
    asset.data.stop_model_limits_widened = True
