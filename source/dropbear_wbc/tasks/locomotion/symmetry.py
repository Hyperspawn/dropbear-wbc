"""Left/right mirror map for the velocity task (rsl-rl 2.3.3 PPO ``symmetry_cfg``).

The 22-motor mirror is derived from the semantic calibration, not from motor names. Semantic rule (root frame,
x forward, y left, z up; mirror plane x-z): partner side, pitch / knee / elbow keep sign, roll and yaw flip. In motor
space the map is ``M = J^-1 S P J`` with ``J`` = d(semantic)/d(motor) at ``standing_motor_pos`` (finite differences
through :class:`SemanticMap`, so the serial hip groups and the ankle pairs' tie-rod Jacobians are included).

Several motor names do not say what the motor does, so a name rule is wrong: ``PG_*_leg_pitch`` is hip roll (sign
flips) and ``*H_roll`` keeps its sign (mirrored mounting). The knee and elbow pairs carry a small L/R gain from the
calibration (1.016 / 1.052); the ankle pairs keep off-diagonal terms <= 0.0063. ``M @ M`` is the identity exactly.

Usage (Isaac Lab ``RslRlSymmetryCfg``)::

    symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=True,
        data_augmentation_func="dropbear_wbc.tasks.locomotion.symmetry:compute_symmetric_states")
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from dropbear_wbc.kinematics.semantic import DEFAULT_CALIBRATION, SemanticMap
from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES

NUM_MOTORS = len(MOTOR_NAMES)

BASE_TERM_SIGNS: dict[str, tuple[float, ...]] = {
    "base_lin_vel": (1.0, -1.0, 1.0),  # vy flips
    "base_ang_vel": (-1.0, 1.0, -1.0),  # pseudo-vector: wx, wz flip
    "projected_gravity": (1.0, -1.0, 1.0),
    "velocity_commands": (1.0, -1.0, -1.0),  # (vx, vy, wz)
}
"""Per-component sign of the root-frame observation terms under the x-z mirror."""

MOTOR_TERMS: tuple[str, ...] = ("joint_pos", "joint_vel")
ACTION_TERMS: tuple[str, ...] = ("actions",)

DEFAULT_POLICY_LAYOUT: tuple[tuple[str, int], ...] = (
    ("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_commands", 3),
    ("joint_pos", NUM_MOTORS), ("joint_vel", NUM_MOTORS), ("actions", NUM_MOTORS),
)
"""``velocity_env_cfg.ObservationsCfg`` policy and critic groups (same terms, 78 values)."""


def semantic_mirror(calibration: str | Path = DEFAULT_CALIBRATION) -> np.ndarray:
    """22x22 ``S P``: q_sem' = sign * q_sem[partner] (left_* <-> right_*; roll/yaw flip)."""
    sem = list(json.loads(Path(calibration).read_text())["semantic_names"])
    partner = [sem.index(n.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")) for n in sem]
    sign = np.array([-1.0 if ("roll" in n or "yaw" in n) else 1.0 for n in sem])
    return np.diag(sign) @ np.eye(len(sem))[partner]


@functools.lru_cache(maxsize=4)
def _motor_mirror_cached(calibration: str, h: float) -> np.ndarray:
    calib = json.loads(Path(calibration).read_text())
    if list(calib["motor_names"]) != list(MOTOR_NAMES):
        raise ValueError("calibration motor order differs from dropbear_names.MOTOR_NAMES")
    sm = SemanticMap(calib)
    q0 = np.asarray(calib["standing_motor_pos"], dtype=float)
    f = lambda q: np.asarray(sm.motor_to_semantic(q, clip=False), dtype=float)  # noqa: E731
    jac = np.stack([(f(q0 + h * e) - f(q0 - h * e)) / (2 * h) for e in np.eye(NUM_MOTORS)], axis=1)
    m = np.linalg.solve(jac, semantic_mirror(calibration) @ jac)
    m = 0.5 * (m + np.linalg.inv(m))  # remove finite-difference residue; keeps M @ M = I
    m.setflags(write=False)
    return m


def motor_mirror_matrix(calibration: str | Path = DEFAULT_CALIBRATION, h: float = 1e-4) -> np.ndarray:
    """Linear motor-space mirror ``dq' = M dq`` (positions relative to the default pose, velocities, PD targets)."""
    return _motor_mirror_cached(str(calibration), float(h))


def motor_mirror_pairs(calibration: str | Path = DEFAULT_CALIBRATION) -> list[tuple[str, str, float]]:
    """Dominant term per row: (motor, mirrored-from motor, signed gain). Documentation / tests only."""
    m = motor_mirror_matrix(calibration)
    out = []
    for i, n in enumerate(MOTOR_NAMES):
        j = int(np.argmax(np.abs(m[i])))
        out.append((n, MOTOR_NAMES[j], float(m[i, j])))
    return out


def action_mirror_matrix(action_scale=None, calibration: str | Path = DEFAULT_CALIBRATION) -> np.ndarray:
    """Mirror for raw actions (target = default + scale * a): ``diag(1/s) M diag(s)``."""
    m = motor_mirror_matrix(calibration)
    if action_scale is None:
        return m
    s = np.broadcast_to(np.asarray(action_scale, dtype=float), (NUM_MOTORS,))
    return (m * s[None, :]) / s[:, None]


def observation_mirror_matrix(layout=DEFAULT_POLICY_LAYOUT, action_scale=None,
                              calibration: str | Path = DEFAULT_CALIBRATION) -> np.ndarray:
    """Block-diagonal mirror for a flat observation built from ``layout`` = [(term name, dim), ...].

    Unknown terms raise, so a new observation term cannot silently pass through un-mirrored."""
    blocks = []
    for name, dim in layout:
        if name in BASE_TERM_SIGNS and dim == len(BASE_TERM_SIGNS[name]):
            blocks.append(np.diag(BASE_TERM_SIGNS[name]))
        elif name in MOTOR_TERMS and dim == NUM_MOTORS:
            blocks.append(motor_mirror_matrix(calibration))
        elif name in ACTION_TERMS and dim == NUM_MOTORS:
            blocks.append(action_mirror_matrix(action_scale, calibration))
        else:
            raise KeyError(f"no mirror rule for observation term {name!r} (dim {dim})")
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n))
    k = 0
    for b in blocks:
        d = b.shape[0]
        out[k:k + d, k:k + d] = b
        k += d
    return out


def _env_layout(env, group: str):
    """[(term, dim)] for one observation group of an Isaac Lab ManagerBasedRLEnv (unwrapped or wrapped)."""
    base = getattr(env, "unwrapped", env)
    om = base.observation_manager
    names = om.active_terms[group]
    dims = om.group_obs_term_dim[group]
    return [(n, int(np.prod(d))) for n, d in zip(names, dims)]


def _env_action_scale(env):
    base = getattr(env, "unwrapped", env)
    try:
        term = base.action_manager.get_term("joint_pos")
        s = term._scale  # noqa: SLF001 (Isaac Lab JointAction keeps the resolved per-joint scale here)
        s = s[0] if getattr(s, "ndim", 0) == 2 else s
        return np.asarray(s.detach().cpu() if hasattr(s, "detach") else s, dtype=float)
    except Exception:  # noqa: BLE001 - scalar or missing scale -> plain motor mirror
        return None


@functools.lru_cache(maxsize=8)
def _torch_matrix(key, device, dtype, mat_bytes, n):
    import torch

    return torch.from_numpy(np.frombuffer(mat_bytes, dtype=np.float64).reshape(n, n).copy()).to(device, dtype)


def _mirror(x, mat: np.ndarray):
    if x is None:
        return None
    t = _torch_matrix(None, x.device, x.dtype, mat.tobytes(), mat.shape[0])
    return x @ t.T


_NORMALIZERS: dict[str, object] = {}


def set_observation_normalizers(policy=None, critic=None) -> None:
    """Register the runner's rsl-rl ``EmpiricalNormalization`` modules (None clears).

    rsl-rl stores NORMALIZED observations in the rollout and hands those to the augmentation. Mirroring a normalized
    vector is only right if the running mean is itself mirror symmetric: M (x - mu) / s != (M x - mu) / s otherwise.
    The walkers hold their arms asymmetrically, so the mirrored samples were wrong and the asymmetry fed on itself
    (local AMP runs: rsl-rl symmetry loss 4 -> 10 in 500 iterations). With a normalizer registered the observation
    is un-normalized, mirrored and re-normalized."""
    _NORMALIZERS.clear()
    if policy is not None:
        _NORMALIZERS["policy"] = policy
    if critic is not None:
        _NORMALIZERS["critic"] = critic


def _mirror_obs(x, mat: np.ndarray, obs_type: str):
    norm = _NORMALIZERS.get(obs_type)
    mean, std = getattr(norm, "_mean", None), getattr(norm, "_std", None)
    if mean is None or std is None:  # no normalizer (or torch.nn.Identity)
        return _mirror(x, mat)
    scale = std.to(x.dtype) + float(getattr(norm, "eps", 0.0))
    mean = mean.to(x.dtype)
    return (_mirror(x * scale + mean, mat) - mean) / scale


def compute_symmetric_states(obs=None, actions=None, env=None, obs_type: str = "policy"):
    """rsl-rl 2.3.3 ``data_augmentation_func``: returns ``cat([x, mirror(x)])`` for obs and actions."""
    import torch

    if env is not None:
        layout = _env_layout(env, obs_type)
        scale = _env_action_scale(env)
    else:
        layout, scale = DEFAULT_POLICY_LAYOUT, None
    obs_aug = act_aug = None
    if obs is not None:
        obs_aug = torch.cat([obs, _mirror_obs(obs, observation_mirror_matrix(tuple(layout), scale), obs_type)], dim=0)
    if actions is not None:
        act_aug = torch.cat([actions, _mirror(actions, action_mirror_matrix(scale))], dim=0)
    return obs_aug, act_aug
