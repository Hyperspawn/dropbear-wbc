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
