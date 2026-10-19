"""Shared mass-DR event for the tracking and velocity tasks (notes/13 T4, item 59).

Imports only torch at module level (Isaac types under ``TYPE_CHECKING``), so the event runs against a fake
articulation in the CPU tests. Re-exported by ``tasks/tracking/mdp/events.py`` and ``tasks/locomotion/mdp/events.py``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import torch

from dropbear_wbc.robots.mass_dr import LIMB_MASS_SCALE, limb_body_index_plan, limb_mass_groups

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def randomize_limb_mass_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] = LIMB_MASS_SCALE,
    limb_groups: Mapping[str, Sequence[str]] | None = None,
    include_shoulder_rotors: bool = False,
    recompute_inertia: bool = True,
):
    """Scale every body of a limb by ONE ``U(scale_range)`` draw per limb per env (startup).

    ``limb_groups`` (limb -> body names) defaults to :func:`dropbear_wbc.robots.mass_dr.limb_mass_groups` of the
    articulation's ``body_names``; ``asset_cfg.body_ids`` is ignored (the groups select the bodies).

    Same write path as Isaac Lab 2.2 ``randomize_rigid_body_mass``: start from ``data.default_mass`` (so repeated
    calls do not compound), ``root_physx_view.set_masses`` on CPU tensors, then (``recompute_inertia``) inertia =
    ``data.default_inertia`` x mass ratio (uniform-density assumption, per body) via ``set_inertias``. Bodies outside
    the groups (``world``, ``head_*``, ``torso_*``) are not touched, so this composes with ``add_base_mass`` on
    ``world`` in either order. The sampled scales are kept in ``asset.data.limb_mass_scale`` (num_envs, n_limbs, CPU)
    with ``asset.data.limb_mass_scale_names`` for logging / privileged observations.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = torch.as_tensor(env_ids).to("cpu", dtype=torch.long)
    groups = limb_mass_groups(asset.body_names, include_shoulder_rotors) if limb_groups is None else limb_groups
    body_ids_l, limb_index_l, limbs = limb_body_index_plan(asset.body_names, groups)
    body_ids = torch.tensor(body_ids_l, dtype=torch.long, device="cpu")
    limb_index = torch.tensor(limb_index_l, dtype=torch.long, device="cpu")

    lo, hi = float(scale_range[0]), float(scale_range[1])
    scales = lo + (hi - lo) * torch.rand((len(env_ids), len(limbs)), device="cpu")
    per_body = scales[:, limb_index]  # (n_env, n_bodies): one draw shared by all bodies of a limb

    rows = env_ids[:, None]
    default_mass = asset.data.default_mass.to("cpu")
    masses = asset.root_physx_view.get_masses()
    masses[rows, body_ids] = default_mass[rows, body_ids] * per_body
    asset.root_physx_view.set_masses(masses, env_ids)

    if recompute_inertia:
        inertias = asset.root_physx_view.get_inertias()
        default_inertia = asset.data.default_inertia.to("cpu")
        inertias[rows, body_ids] = default_inertia[rows, body_ids] * per_body[..., None]
        asset.root_physx_view.set_inertias(inertias, env_ids)

    store = getattr(asset.data, "limb_mass_scale", None)
    if store is None or tuple(store.shape) != (env.scene.num_envs, len(limbs)):
        store = torch.ones((env.scene.num_envs, len(limbs)), device="cpu")
    store[env_ids] = scales
    asset.data.limb_mass_scale = store
    asset.data.limb_mass_scale_names = list(limbs)


def apply_body_mass_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offsets: Mapping[str, float],
    recompute_inertia: bool = True,
):
    """Add a fixed mass [kg] to named bodies, in every env, once (startup): the plant change of a motor swap
    (:func:`dropbear_wbc.robots.hw_motor_specs.motor_mass_offsets`). Not a randomization, so it stays on in play.

    Both the live masses and ``data.default_mass`` move by the offset, so the mass-DR terms (which restart from the
    defaults) randomize around the new plant whichever order the startup terms run in. Inertia follows the mass ratio
    of each body (uniform-density assumption, as in the limb scale). A second call is a no-op.
    """
    asset = env.scene[asset_cfg.name]
    if getattr(asset.data, "body_mass_offsets", None):
        return
    n = env.scene.num_envs
    all_ids = torch.arange(n, device="cpu")
    idx = [asset.body_names.index(b) for b in offsets]
    delta = torch.tensor([float(offsets[b]) for b in offsets], dtype=torch.float32)
    default_mass = asset.data.default_mass
    old = default_mass[:, idx].to("cpu").clone()
    new = old + delta
    if (new <= 0).any():
        raise ValueError(f"mass offsets {dict(offsets)} make a body mass non-positive")
    default_mass[:, idx] = new.to(default_mass.device)
    masses = asset.root_physx_view.get_masses()
    masses[:, idx] = masses[:, idx] + delta
    asset.root_physx_view.set_masses(masses, all_ids)
    if recompute_inertia:
        ratio = (new / old)[..., None]
        inertias = asset.root_physx_view.get_inertias()
        inertias[:, idx] = inertias[:, idx] * ratio
        asset.root_physx_view.set_inertias(inertias, all_ids)
        di = asset.data.default_inertia
        di[:, idx] = (di[:, idx].to("cpu") * ratio).to(di.device)
    asset.data.body_mass_offsets = dict(offsets)
