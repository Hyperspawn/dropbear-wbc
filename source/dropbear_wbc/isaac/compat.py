"""Isaac Lab quaternion-order and root-write boundary (2.x <-> 3.x).

dropbear_wbc is wxyz everywhere outside Isaac: settled motion NPZs, ``deploy/quat.py``, the SDK
``imu.quat_wxyz`` contract and the policy sidecar. Isaac Lab 2.x is also wxyz; Isaac Lab 3.0 is xyzw
(``.data.*quat*``, ``*_pose_w``, ``write_root_pose_to_sim_index``, ``isaaclab.utils.math``) and its
``.data`` properties return ProxyArray (``.torch``).

Every Isaac quaternion read in ``source/dropbear_wbc`` goes through ``root_quat_wxyz`` /
``body_quat_wxyz`` and every root write through ``write_root_state`` / ``write_root_pose``, and every cfg ``rot=`` literal
(``InitialStateCfg`` / ``OffsetCfg``) through ``cfg_quat``, so a
3.0 port is a one-place reorder. ``tests/test_isaac_quat_boundary.py`` fails on any new raw read.

Importing this module never imports isaaclab. On 2.x every conversion is an identity (no copy).
``DROPBEAR_ISAAC_QUAT_ORDER=wxyz|xyzw`` overrides detection (tests, migration branch).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

WXYZ = "wxyz"
XYZW = "xyzw"
_ENV = "DROPBEAR_ISAAC_QUAT_ORDER"


def _isaaclab_major() -> int | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    for dist in ("isaaclab",):
        try:
            return int(version(dist).split(".")[0])
        except (PackageNotFoundError, ValueError):
            continue
    return None


@lru_cache(maxsize=1)
def _detected_order() -> str:
    major = _isaaclab_major()
    return XYZW if major is not None and major >= 3 else WXYZ


def isaac_quat_order() -> str:
    """Quaternion order of the installed Isaac Lab ('wxyz' on 2.x or when absent, 'xyzw' on >= 3)."""
    o = os.environ.get(_ENV, "").strip().lower()
    if o:
        if o not in (WXYZ, XYZW):
            raise ValueError(f"{_ENV}={o!r}: expected 'wxyz' or 'xyzw'")
        return o
    return _detected_order()


def as_tensor(x: Any) -> Any:
    """Isaac Lab 3.0 ``.data`` returns ProxyArray; take its zero-copy ``.torch`` view. Identity otherwise."""
    t = getattr(x, "torch", None)
    return x if t is None or callable(t) else t


def gravity_dir_w(asset: Any) -> Any:
    """Unit world gravity direction per env, (N, 3); (0, 0, -1) under default gravity.

    ``data.GRAVITY_VEC_W`` is a unit vector on Isaac Lab 2.x and on the 3.0 PhysX backend, but the 3.0
    Newton backend binds it to ``model.gravity`` in m/s^2 (|g| = 9.81; "consumers normalize on read",
    isaaclab_newton ArticulationData). Raw use scales every gravity-in-body-frame term by |g|.
    A zero vector (gravity disabled) stays zero.
    """
    g = as_tensor(asset.data.GRAVITY_VEC_W)
    try:
        import torch

        if isinstance(g, torch.Tensor):
            return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    except ImportError:  # pragma: no cover
        pass
    import numpy as np

    g = np.asarray(g, dtype=float)
    return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)


# --- body mass / inertia / CoM writes (2.x root_physx_view <-> 3.x set_*_index) -------------------------------
# Isaac Lab 3.0 removed the PhysX view from the Newton backend (``Articulation.root_view`` is a Newton
# ArticulationView with no get_masses/set_masses/get_coms; ``root_physx_view`` exists only on the PhysX backend,
# deprecated). Both 3.0 backends expose ``set_masses_index`` / ``set_inertias_index`` / ``set_coms_index`` (partial
# data: rows env_ids, columns body_ids) and keep ``data.body_mass`` / ``body_inertia`` / ``body_com_pos_b`` coherent.
# Callers here pass FULL (num_envs, num_bodies[, k]) CPU tensors, as the 2.x view API does.


def _index_setters(asset: Any) -> bool:
    return callable(getattr(asset, "set_masses_index", None))


def _newton_backend(asset: Any) -> bool:
    return type(asset).__module__.split(".")[0] == "isaaclab_newton"


def _rows_cols(env_ids: Any, body_ids: Any, device: Any) -> tuple[Any, Any]:
    import torch

    e = torch.as_tensor(env_ids, dtype=torch.long)
    b = None if body_ids is None else torch.as_tensor(body_ids, dtype=torch.long)
    return e, b


def _sub(full: Any, e: Any, b: Any) -> Any:
    x = full[e.cpu()] if full.device.type == "cpu" else full[e.to(full.device)]
    return x if b is None else x[:, b.to(x.device)]


def get_body_masses(asset: Any) -> Any:
    """Live body masses [kg], (num_envs, num_bodies) float32 CPU tensor (a copy)."""
    if _index_setters(asset):
        return as_tensor(asset.data.body_mass).detach().to("cpu", copy=True)
    return asset.root_physx_view.get_masses()


def set_body_masses(asset: Any, masses: Any, env_ids: Any, body_ids: Any = None) -> None:
    """Write rows ``env_ids`` (and columns ``body_ids``, all if None) of the full (N, B) ``masses``."""
    if not _index_setters(asset):
        asset.root_physx_view.set_masses(masses, env_ids)
        return
    e, b = _rows_cols(env_ids, body_ids, asset.device)
    asset.set_masses_index(masses=_sub(masses, e, b).to(asset.device).float().contiguous(),
                           body_ids=None if b is None else b.to(asset.device), env_ids=e.to(asset.device))


def get_body_inertias(asset: Any) -> Any:
    """Live flattened body inertias [kg*m^2], (num_envs, num_bodies, 9) float32 CPU tensor (a copy)."""
    if _index_setters(asset):
        return as_tensor(asset.data.body_inertia).detach().to("cpu", copy=True)
    return asset.root_physx_view.get_inertias()


def set_body_inertias(asset: Any, inertias: Any, env_ids: Any, body_ids: Any = None) -> None:
    """Write rows ``env_ids`` (and columns ``body_ids``) of the full (N, B, 9) ``inertias``."""
    if not _index_setters(asset):
        asset.root_physx_view.set_inertias(inertias, env_ids)
        return
    e, b = _rows_cols(env_ids, body_ids, asset.device)
    asset.set_inertias_index(inertias=_sub(inertias, e, b).to(asset.device).float().contiguous(),
                             body_ids=None if b is None else b.to(asset.device), env_ids=e.to(asset.device))


def get_body_com_pos(asset: Any) -> Any:
    """CoM position in each body's link frame [m], (num_envs, num_bodies, 3) float32 CPU tensor (a copy)."""
    if _index_setters(asset):
        return as_tensor(asset.data.body_com_pos_b).detach().to("cpu", copy=True)
    return asset.root_physx_view.get_coms()[..., :3].clone()


def set_body_com_pos(asset: Any, pos: Any, env_ids: Any, body_ids: Any = None) -> None:
    """Write CoM positions (rows ``env_ids``, columns ``body_ids``) of the full (N, B, 3) ``pos``.

    The CoM orientation is kept: 2.x and 3.0 PhysX re-send the live principal-axes quaternion (backend order,
    untouched); 3.0 Newton has no CoM orientation (always body-aligned) and takes positions only.
    """
    if not _index_setters(asset):
        coms = asset.root_physx_view.get_coms().clone()
        coms[..., :3] = pos
        asset.root_physx_view.set_coms(coms, env_ids)
        return
    e, b = _rows_cols(env_ids, body_ids, asset.device)
    if _newton_backend(asset):
        sub = _sub(pos, e, b)
    else:
        pose = as_tensor(asset.data.body_com_pose_b).detach().to("cpu", copy=True)
        pose[..., :3] = pos
        sub = _sub(pose, e, b)
    asset.set_coms_index(coms=sub.to(asset.device).float().contiguous(),
                         body_ids=None if b is None else b.to(asset.device), env_ids=e.to(asset.device))


# --- body mass / inertia defaults (Isaac Lab 3.0 deprecates data.default_mass / default_inertia, 4.0 removes) -----
# 2.x: ``data.default_mass`` / ``default_inertia`` are spawn-time copies; Isaac's ``randomize_rigid_body_mass``
# re-reads them on every call. 3.0: they are lazy snapshots of the LIVE buffers on first access (deprecated,
# base_articulation_data.py:1506-1530) and Isaac's ``randomize_rigid_body_mass`` keeps its own snapshot of
# ``body_mass`` taken at its first call. Our mass events share ONE per-asset snapshot taken at the first call of
# any of them (before that event writes), so the motor-mass offset can move it for the limb scale.
_BODY_DEFAULTS = "_dropbear_body_defaults"


def body_prop_defaults(asset: Any) -> dict:
    """Per-asset default body masses (N, B) and flattened inertias (N, B, 9), float32 CPU, snapshotted once.

    2.x: copies of ``data.default_mass`` / ``default_inertia`` (spawn values). 3.x: copies of the live
    ``data.body_mass`` / ``body_inertia`` at the first call (no deprecated property read). Callers must take it
    before their first write. :func:`shift_body_mass_defaults` is the only writer.
    """
    d = getattr(asset, _BODY_DEFAULTS, None)
    if d is None:
        if _index_setters(asset):
            d = {"mass": get_body_masses(asset), "inertia": get_body_inertias(asset)}
        else:
            d = {"mass": as_tensor(asset.data.default_mass).detach().to("cpu", copy=True),
                 "inertia": as_tensor(asset.data.default_inertia).detach().to("cpu", copy=True)}
        setattr(asset, _BODY_DEFAULTS, d)
    return d


def shift_body_mass_defaults(asset: Any, body_ids: Any, new_mass: Any, inertia_ratio: Any) -> None:
    """Move the default plant of ``body_ids`` (all envs) to ``new_mass`` (N, k), inertia x ``inertia_ratio``
    (N, k, 1). 2.x also moves ``data.default_mass`` / ``default_inertia`` in place (Isaac 2.x
    ``randomize_rigid_body_mass`` restarts from them); 3.x Isaac keeps its own snapshot, so they are not touched."""
    import torch

    d = body_prop_defaults(asset)
    b = torch.as_tensor(body_ids, dtype=torch.long)
    d["mass"][:, b] = new_mass.to("cpu", d["mass"].dtype)
    d["inertia"][:, b] = d["inertia"][:, b] * inertia_ratio.to("cpu", d["inertia"].dtype)
    if not _index_setters(asset):
        dm, di = as_tensor(asset.data.default_mass), as_tensor(asset.data.default_inertia)
        dm[:, b.to(dm.device)] = d["mass"][:, b].to(dm.device)
        di[:, b.to(di.device)] = d["inertia"][:, b].to(di.device)


# --- PhysX-only diagnostics -----------------------------------------------------------------------------------
class PhysXOnlyError(RuntimeError):
    """A diagnostic needs the PhysX tensor view (DOF drive / friction read-back) and the backend has none."""


def physx_view(asset: Any, purpose: str) -> Any:
    """The PhysX articulation view: 2.x ``root_physx_view``; 3.x PhysX ``root_view`` (``root_physx_view`` is a
    deprecated alias). Raises :class:`PhysXOnlyError` on the 3.x Newton backend (``root_view`` is a Newton
    ArticulationView without the PhysX getters) and when a 3.x joint ordering is active (view DOF columns are in
    backend order, not ``joint_names`` order)."""
    if _newton_backend(asset):
        raise PhysXOnlyError(f"{purpose} is PhysX-only: it reads the PhysX tensor view (DOF drive / friction / mass "
                             f"read-back) and the Isaac Lab Newton backend ({type(asset).__module__}) has none. "
                             "Run it with the PhysX physics backend.")
    if getattr(getattr(asset, "data", None), "has_joint_ordering", False) is True:
        raise PhysXOnlyError(f"{purpose}: ArticulationCfg.joint_ordering is active, so PhysX view DOF columns are in "
                             "backend order, not joint_names order; unset joint_ordering for this diagnostic.")
    if _index_setters(asset):
        return asset.root_view
    return asset.root_physx_view


def _roll(q: Any, shift: int) -> Any:
    try:
        import torch

        if isinstance(q, torch.Tensor):
            return torch.roll(q, shifts=shift, dims=-1)
    except ImportError:  # pragma: no cover
        pass
    import numpy as np

    return np.roll(np.asarray(q), shift, axis=-1)


def _check4(q: Any) -> None:
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion last dim must be 4, got shape {tuple(q.shape)}")


def wxyz_from_isaac(q: Any) -> Any:
    """Isaac-native quaternion(s) [..., 4] -> wxyz. Identity on 2.x."""
    q = as_tensor(q)
    _check4(q)
    return q if isaac_quat_order() == WXYZ else _roll(q, 1)


def isaac_from_wxyz(q: Any) -> Any:
    """wxyz quaternion(s) [..., 4] -> Isaac-native order. Identity on 2.x."""
    _check4(q)
    return q if isaac_quat_order() == WXYZ else _roll(q, -1)


def cfg_quat(q_wxyz: Any) -> tuple[float, float, float, float]:
    """wxyz quaternion -> float 4-tuple in Isaac Lab's cfg order, for ``rot=`` fields of
    ``InitialStateCfg`` / sensor ``OffsetCfg``. Lab 3.0 reads those as xyzw (default (0, 0, 0, 1)), so a
    wxyz identity (1, 0, 0, 0) would spawn the asset rotated 180 deg about x. Identity order on 2.x."""
    q = tuple(float(v) for v in q_wxyz)
    if len(q) != 4:
        raise ValueError(f"cfg_quat expects 4 values (wxyz), got {len(q)}")
    return q if isaac_quat_order() == WXYZ else (q[1], q[2], q[3], q[0])


def root_quat_wxyz(asset: Any, *, link: bool = False) -> Any:
    """Root orientation (num_envs, 4) in wxyz. ``link=True`` reads ``root_link_quat_w``."""
    d = asset.data
    return wxyz_from_isaac(d.root_link_quat_w if link else d.root_quat_w)


def body_quat_wxyz(asset: Any, ids: Any = slice(None), *, link: bool = False) -> Any:
    """Body orientations (num_envs, len(ids), 4) in wxyz. ``link=True`` reads ``body_link_quat_w``."""
    d = asset.data
    q = as_tensor(d.body_link_quat_w if link else d.body_quat_w)
    return wxyz_from_isaac(q[:, ids])


def _cat(parts: list) -> Any:
    import torch

    return torch.cat(parts, dim=-1)


def write_root_pose(asset: Any, pos: Any, quat_wxyz: Any, env_ids: Any = None) -> None:
    """Write root pose (pos (n, 3), quat wxyz (n, 4)); picks the 3.x ``*_index`` API when present."""
    pose = _cat([pos, isaac_from_wxyz(quat_wxyz)])
    fn = getattr(asset, "write_root_pose_to_sim_index", None)
    if fn is not None:
        fn(root_pose=pose, env_ids=env_ids)
    else:
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)


def write_root_state(asset: Any, pos: Any, quat_wxyz: Any, lin_vel: Any, ang_vel: Any, env_ids: Any = None) -> None:
    """Write root pose + velocity; on 3.x as pose_index + velocity_index (write_root_state_to_sim is deprecated)."""
    fn = getattr(asset, "write_root_pose_to_sim_index", None)
    if fn is not None:
        fn(root_pose=_cat([pos, isaac_from_wxyz(quat_wxyz)]), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=_cat([lin_vel, ang_vel]), env_ids=env_ids)
    else:
        asset.write_root_state_to_sim(_cat([pos, isaac_from_wxyz(quat_wxyz), lin_vel, ang_vel]), env_ids=env_ids)
