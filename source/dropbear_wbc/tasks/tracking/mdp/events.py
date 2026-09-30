"""Tracking events (BeyondMimic port) with Dropbear patches.

* ``randomize_joint_default_pos`` is meant to be configured on the 22 motors only. Upstream wrote the
  randomized values into the action term's ``_offset`` using *articulation* joint indices, which is only
  correct when the action covers all joints in articulation order. Here the offset is re-derived from
  ``default_joint_pos`` through the action term's own joint ids (22 motors, contract order).
* ``randomize_rigid_body_com`` is unchanged (configure it on ``world``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    action_name: str = "joint_pos",
):
    """Randomize default joint positions (calibration error model) and sync the action offset.

    Saves the nominal default in ``asset.data.default_joint_pos_nominal`` (all joints, env 0) for export.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(asset.data, "default_joint_pos_nominal"):
        asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]
        rows = env_ids[:, None] if joint_ids != slice(None) else env_ids
        asset.data.default_joint_pos[rows, joint_ids] = pos
        # keep JointPositionAction(use_default_offset=True) consistent: offset = default[action joint ids]
        term = env.action_manager.get_term(action_name)
        term_ids = term._joint_ids
        if term_ids == slice(None):
            term._offset[env_ids] = asset.data.default_joint_pos[env_ids]
        else:
            term_ids_t = torch.as_tensor(term_ids, dtype=torch.long, device=asset.device)
            term._offset[env_ids] = asset.data.default_joint_pos[env_ids][:, term_ids_t]


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Add a uniform random offset [m] to the CoM of the selected bodies (startup; CPU tensors)."""
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device="cpu")
    ranges = torch.tensor([com_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")], device="cpu")
    rand = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)
    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids[:, None], body_ids, :3] += rand
    asset.root_physx_view.set_coms(coms, env_ids)
