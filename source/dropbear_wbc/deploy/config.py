"""Deployment configuration for the Dropbear policy runner.

Two input formats map onto one :class:`DeployConfig`:

1. **Unitree ``deploy.yaml``** as written by unitree_rl_lab (``joint_ids_map``,
   ``step_dt``, ``stiffness``, ``damping``, ``default_joint_pos``,
   ``actions.JointPositionAction.{scale,offset,clip}``, ordered
   ``observations: {func_name: {scale, clip, history_length, params}}``), plus
   optional Dropbear keys: ``joint_names`` (preferred over ``joint_ids_map``),
   ``policy`` (ONNX path relative to the YAML), ``motion`` and ``fsm``.
   As in unitree_rl_lab, ``joint_ids_map[i]`` is the SDK slot of policy joint ``i``.
   ``stiffness``/``damping`` are in **SDK slot order** (22 values or a scalar; 0 for slots
   outside the policy). ``default_joint_pos`` and the action ``scale``/``offset``/``clip`` are
   in **policy order**.
   ``default_joint_pos`` may also be ``"legacy"`` (``DROPBEAR_CFG`` init pose) or
   ``{"calibration": <path>}`` (``standing_motor_pos`` of the semantic calibration).

2. **Policy sidecar JSON** ``dropbear-policy-sidecar-v1`` written next to an
   exported ONNX (schema in :data:`SIDECAR_SCHEMA` and CONTRACTS section 6.1).

All per-joint arrays in a DeployConfig are in *policy order* (the order of
``joint_names``); ``motor_ids[i]`` is the SDK slot of policy joint ``i``. Motors
not driven by the policy are held at the default pose with the hold gains.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..sdk import motors

SIDECAR_SCHEMA = "dropbear-policy-sidecar-v1"

# BeyondMimic's policy ObsGroup attribute names -> observation function names. Used only when
# a sidecar/ONNX metadata entry gives a term name without an explicit "func".
BEYONDMIMIC_ALIASES: dict[str, str] = {
    "command": "motion_command",
    "motion_anchor_pos_b": "motion_anchor_pos_b",
    "motion_anchor_ori_b": "motion_anchor_ori_b",
    "base_lin_vel": "base_lin_vel",
    "base_ang_vel": "base_ang_vel",
    "joint_pos": "joint_pos_rel",
    "joint_vel": "joint_vel_rel",
    "actions": "last_action",
}


@dataclass
class ObsTermCfg:
    """One observation term, concatenated in list order.

    Attributes:
        name: term label (as in the source file).
        func: observation function key in :mod:`dropbear_wbc.deploy.observations`.
        scale: scalar or per-element multiplier (applied after ``clip``, as in Isaac Lab).
        clip: optional ``(lo, hi)``.
        history_length: number of stacked frames (oldest first); 1 = no history.
        params: extra parameters (e.g. ``command_name``).
        dim: expected per-frame dimension if known (validated at runtime).
    """

    name: str
    func: str
    scale: Any = 1.0
    clip: tuple[float, float] | None = None
    history_length: int = 1
    params: dict = field(default_factory=dict)
    dim: int | None = None


@dataclass
class MotionCfg:
    """Reference motion for tracking policies.

    Attributes:
        file: contract NPZ (``dropbear-motion-npz-v1``) or Unitree-style CSV; ``None`` when the
            reference comes from the ONNX outputs (``source="onnx"``).
        source: ``"npz"``/``"csv"`` (read ``file``) or ``"onnx"`` (BeyondMimic export outputs).
        anchor_body: anchor body name (CONTRACTS section 5: chest-level torso body).
        fps: frame rate of ``file`` (NPZ carries its own ``fps``).
        time_start, time_end: playback window [s].
        align: ``"yaw_xy"`` (default), ``"yaw"`` or ``"none"`` alignment of the reference to the
            robot's anchor when the policy starts (like unitree_rl_lab's ``State_Mimic``).
        anchor_offset_pos, anchor_offset_quat: rigid anchor pose in the root (``world``) body
            frame; derived from the NPZ when ``None``.
    """

    file: Path | None = None
    source: str = "npz"
    anchor_body: str | None = None
    fps: float | None = None
    time_start: float = 0.0
    time_end: float | None = None
    align: str = "yaw_xy"
    anchor_offset_pos: tuple[float, float, float] | None = None
    anchor_offset_quat: tuple[float, float, float, float] | None = None
    requirements: dict = field(default_factory=dict)
    """``source="runtime"`` exports (motion-library policies, CONTRACTS 5.3/6.1): what a reference fed at runtime must
    satisfy (``fps``, ``usd_sha256``, ``authored_ankle_tierods``, ``motor_names``, ``anchor_body_name``); checked by
    :func:`dropbear_wbc.deploy.motion.load_runtime_motion`."""


MOTION_SOURCE_RUNTIME = "runtime"
"""``motion.source`` of an export that embeds no reference: the runner needs ``--motion <npz|csv>``."""


@dataclass
class FsmCfg:
    """Unitree-deploy-style state machine parameters (SDK slot order, 22 entries each).

    Attributes:
        move_to_default_s: MoveToDefault interpolation time [s] (Unitree FixStand ``ts``).
        passive_kd: damping used in Passive [N*m*s/rad].
        hold_kp, hold_kd: gains for MoveToDefault/Hold and for motors outside the policy.
        bad_orientation_rad: Policy -> Passive when the torso tilt exceeds this [rad]
            (unitree_rl_lab ``bad_orientation(1.0)``); ``None`` disables the check.
    """

    move_to_default_s: float = 2.0
    passive_kd: np.ndarray = field(default_factory=lambda: np.asarray(motors.DEFAULT_KD, float))
    hold_kp: np.ndarray = field(default_factory=lambda: np.asarray(motors.DEFAULT_KP, float))
    hold_kd: np.ndarray = field(default_factory=lambda: np.asarray(motors.DEFAULT_KD, float))
    bad_orientation_rad: float | None = 1.0


@dataclass
class DeployConfig:
    """Everything the runner needs besides the ONNX weights. Arrays in policy order."""

    joint_names: tuple[str, ...]
    motor_ids: np.ndarray
    step_dt: float
    kp: np.ndarray
    kd: np.ndarray
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    action_offset: np.ndarray
    action_clip: float | None = None
    target_clip: np.ndarray | None = None
    """(n, 2) motor position-target limits [rad] applied after scale + offset (sidecar ``target_clip``; the policy was
    trained with them). ``None`` = unclipped."""
    target_interp_steps: int = 0
    """Sidecar ``target_interp_steps``: the policy was trained with each new target ramped linearly over this many
    5 ms physics steps. The 50 Hz FSM cannot do it; the motor-side loop must (``tools/newton_bridge.py --target-ramp-ms``;
    on the robot, the ESP32 / motor firmware). 0 = step targets (zero-order hold)."""
    observations: list[ObsTermCfg] = field(default_factory=list)
    policy_path: Path | None = None
    onnx_obs_input: str | None = None
    onnx_time_input: str | None = None
    motion: MotionCfg | None = None
    fsm: FsmCfg = field(default_factory=FsmCfg)
    default_pose_sdk: np.ndarray = field(default_factory=lambda: np.asarray(motors.DEFAULT_POS, float))
    obs_dim: int | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def num_actions(self) -> int:
        return len(self.joint_names)

    def validate(self) -> None:
        n = self.num_actions
        for name in ("kp", "kd", "default_joint_pos", "action_scale", "action_offset"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name}: expected ({n},), got {arr.shape}")
        if len(set(self.joint_names)) != n or any(j not in motors.MOTOR_NAMES for j in self.joint_names):
            raise ValueError(f"joint_names must be distinct motor-contract names: {self.joint_names}")
        if self.default_pose_sdk.shape != (motors.NUM_MOTORS,):
            raise ValueError("default_pose_sdk must have 22 entries")
        if not 0.001 <= self.step_dt <= 0.1:
            raise ValueError(f"implausible step_dt {self.step_dt}")
        if self.target_clip is not None and (self.target_clip.shape != (n, 2)
                                             or np.any(self.target_clip[:, 0] > self.target_clip[:, 1])):
            raise ValueError(f"target_clip: expected ({n}, 2) with lo <= hi, got {self.target_clip.shape}")
        if self.target_interp_steps < 0:
            raise ValueError(f"target_interp_steps must be >= 0, got {self.target_interp_steps}")


def _vec(value: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.full(n, float(arr[0]))
    if arr.shape != (n,):
        raise ValueError(f"{name}: expected {n} values, got {arr.size}")
    return arr


def _csv_or_list(value: Any) -> Any:
    """ONNX metadata stores lists as comma-separated strings (BeyondMimic ``list_to_csv_str``)."""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return parts
    return value


def load_standing_pose(calibration_path: Path) -> np.ndarray:
    """``standing_motor_pos`` (22, SDK order) from ``dropbear-semantic-calibration-v1``."""
    data = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    pose = np.asarray(data["standing_motor_pos"], dtype=float)
    if pose.shape != (motors.NUM_MOTORS,):
        raise ValueError(f"standing_motor_pos in {calibration_path} has shape {pose.shape}")
    return pose


def _resolve_default(value: Any, joint_names: tuple[str, ...], base: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (policy-order default, SDK-order full default pose)."""
    full = np.asarray(motors.DEFAULT_POS, dtype=float)
    ids = [motors.MOTOR_NAMES.index(j) for j in joint_names]
    if value is None or value == "legacy":
        return full[ids].copy(), full
    if isinstance(value, dict) and "calibration" in value:
        full = load_standing_pose((base / value["calibration"]).resolve())
        return full[ids].copy(), full
    pol = _vec(_csv_or_list(value), len(joint_names), "default_joint_pos")
    full = full.copy()
    full[ids] = pol
    return pol, full


def _fsm_from(d: dict | None) -> FsmCfg:
    fsm = FsmCfg()
    if not d:
        return fsm
    n = motors.NUM_MOTORS
    if "move_to_default_s" in d:
        fsm.move_to_default_s = float(d["move_to_default_s"])
    for key in ("passive_kd", "hold_kp", "hold_kd"):
        if key in d:
            setattr(fsm, key, _vec(d[key], n, f"fsm.{key}"))
    if "bad_orientation_rad" in d:
        fsm.bad_orientation_rad = None if d["bad_orientation_rad"] is None else float(d["bad_orientation_rad"])
    return fsm


def _motion_from(d: dict | None, base: Path) -> MotionCfg | None:
    if not d:
        return None
    file = d.get("file") or d.get("npz") or d.get("motion_file")
    off_p, off_q = d.get("anchor_offset_pos"), d.get("anchor_offset_quat")
    return MotionCfg(
        file=None if file is None else (base / file).resolve(),
        source=d.get("source", "onnx" if file is None else ("csv" if str(file).endswith(".csv") else "npz")),
        anchor_body=d.get("anchor_body") or d.get("anchor_body_name"), fps=d.get("fps"),
        time_start=float(d.get("time_start", 0.0)),
        time_end=None if d.get("time_end") is None else float(d["time_end"]), align=d.get("align", "yaw_xy"),
        anchor_offset_pos=None if off_p is None else tuple(float(v) for v in off_p),
        anchor_offset_quat=None if off_q is None else tuple(float(v) for v in off_q),
        requirements=dict(d.get("requirements") or {}))


def _obs_from_mapping(obs: dict) -> list[ObsTermCfg]:
    """Unitree ``deploy.yaml``: ordered ``{func_name: {params, clip, scale, history_length}}``."""
    terms = []
    for name, spec in (obs or {}).items():
        spec = spec or {}
        clip = spec.get("clip")
        terms.append(ObsTermCfg(name=name, func=spec.get("func", name), scale=spec.get("scale", 1.0) or 1.0,
                                clip=None if clip is None else (float(clip[0]), float(clip[1])),
                                history_length=int(spec.get("history_length", 1) or 1),
                                params=dict(spec.get("params") or {}), dim=spec.get("dim")))
    return terms


def load_deploy_yaml(path: Path) -> DeployConfig:
    """Parse a unitree_rl_lab-style ``deploy.yaml`` (with optional Dropbear extensions)."""
    path = Path(path)
    base = path.parent
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "joint_names" in d:
        joint_names = tuple(d["joint_names"])
    elif "joint_ids_map" in d:
        joint_names = tuple(motors.MOTOR_NAMES[int(i)] for i in d["joint_ids_map"])
    else:
        joint_names = motors.MOTOR_NAMES
    n = len(joint_names)
    motor_ids = [motors.MOTOR_NAMES.index(j) for j in joint_names]
    default_pol, default_full = _resolve_default(d.get("default_joint_pos"), joint_names, base)
    act = (d.get("actions") or {}).get("JointPositionAction", {}) or {}
    offset = act.get("offset")
    # unitree_rl_lab writes stiffness/damping in *SDK slot order* (export_deploy_cfg.py:
    # ``stiffness[joint_ids_map] = default_joint_stiffness``; deploy C++ ``joint_stiffness; // sdk order``,
    # applied as ``motor_cmd()[i].kp() = joint_stiffness[i]``). Every other per-joint array is in policy order.
    sdk_kp = _vec(d.get("stiffness", motors.DEFAULT_KP), motors.NUM_MOTORS, "stiffness (SDK slot order)")
    sdk_kd = _vec(d.get("damping", motors.DEFAULT_KD), motors.NUM_MOTORS, "damping (SDK slot order)")
    cfg = DeployConfig(
        joint_names=joint_names,
        motor_ids=np.asarray(motor_ids, dtype=np.int64),
        step_dt=float(d.get("step_dt", 0.02)),
        kp=sdk_kp[motor_ids],
        kd=sdk_kd[motor_ids],
        default_joint_pos=default_pol,
        action_scale=_vec(act.get("scale", 0.25), n, "actions.scale"),
        action_offset=default_pol.copy() if offset is None else _vec(offset, n, "actions.offset"),
        action_clip=None if act.get("clip") is None else float(np.max(np.abs(act["clip"]))),
        observations=_obs_from_mapping(d.get("observations")),
        policy_path=None if d.get("policy") is None else (base / d["policy"]).resolve(),
        motion=_motion_from(d.get("motion"), base), fsm=_fsm_from(d.get("fsm")), default_pose_sdk=default_full,
        source=str(path))
    cfg.validate()
    return cfg


def load_sidecar(path: Path, onnx_metadata: dict[str, str] | None = None) -> DeployConfig:
    """Parse a ``dropbear-policy-sidecar-v1`` JSON (see module docstring and CONTRACTS 6.1).

    ``onnx_metadata`` (BeyondMimic ``attach_onnx_metadata`` keys) fills fields the sidecar omits.
    """
    path = Path(path)
    base = path.parent
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"{path}: schema {d.get('schema')!r} != {SIDECAR_SCHEMA!r}")
    meta = {k: _csv_or_list(v) for k, v in (onnx_metadata or {}).items()}
    get = lambda k, alt=None: d.get(k, meta.get(alt or k))  # noqa: E731
    joint_names = tuple(get("joint_names"))
    n = len(joint_names)
    default_pol, default_full = _resolve_default(get("default_joint_pos"), joint_names, base)
    terms = []
    for t in d.get("observations") or [{"name": nm} for nm in meta.get("observation_names", [])]:
        func = t.get("func") or BEYONDMIMIC_ALIASES.get(t["name"], t["name"])
        clip = t.get("clip")
        terms.append(ObsTermCfg(name=t["name"], func=func, scale=t.get("scale", 1.0) or 1.0,
                                clip=None if clip is None else (float(clip[0]), float(clip[1])),
                                history_length=int(t.get("history_length", 1) or 1),
                                params=dict(t.get("params") or {}), dim=t.get("dim")))
    offset = d.get("action_offset")
    onnx_in = d.get("onnx_inputs") or {}
    motion = dict(d.get("motion") or {})
    if "anchor_body_name" not in motion and meta.get("anchor_body_name"):
        motion["anchor_body_name"] = meta["anchor_body_name"][0] if isinstance(meta["anchor_body_name"], list) \
            else meta["anchor_body_name"]
    cfg = DeployConfig(
        joint_names=joint_names,
        motor_ids=np.asarray([motors.MOTOR_NAMES.index(j) for j in joint_names], dtype=np.int64),
        step_dt=float(d.get("step_dt", 0.02)),
        kp=_vec(get("joint_stiffness"), n, "joint_stiffness"),
        kd=_vec(get("joint_damping"), n, "joint_damping"),
        default_joint_pos=default_pol,
        action_scale=_vec(get("action_scale"), n, "action_scale"),
        action_offset=default_pol.copy() if offset is None else _vec(offset, n, "action_offset"),
        action_clip=None if d.get("action_clip") is None else float(d["action_clip"]),
        target_clip=None if d.get("target_clip") is None else np.asarray(d["target_clip"], dtype=float).reshape(-1, 2),
        target_interp_steps=int(d.get("target_interp_steps") or 0),
        observations=terms,
        policy_path=None if d.get("policy_onnx") is None else (base / d["policy_onnx"]).resolve(),
        onnx_obs_input=onnx_in.get("obs"), onnx_time_input=onnx_in.get("time_step"),
        motion=_motion_from(motion or None, base), fsm=_fsm_from(d.get("fsm")), default_pose_sdk=default_full,
        obs_dim=d.get("obs_dim"), source=str(path), meta={k: d[k] for k in ("task", "usd_sha256", "run_path")
                                                            if k in d})
    cfg.validate()
    return cfg


def hold_config(default_pose_sdk: np.ndarray | None = None) -> DeployConfig:
    """A policy-less configuration for the ``hold`` mode (all 22 motors, legacy gains)."""
    full = np.asarray(motors.DEFAULT_POS if default_pose_sdk is None else default_pose_sdk, dtype=float)
    cfg = DeployConfig(joint_names=motors.MOTOR_NAMES, motor_ids=np.arange(motors.NUM_MOTORS), step_dt=0.02,
                       kp=np.asarray(motors.DEFAULT_KP, float), kd=np.asarray(motors.DEFAULT_KD, float),
                       default_joint_pos=full.copy(), action_scale=np.zeros(motors.NUM_MOTORS),
                       action_offset=full.copy(), default_pose_sdk=full, source="builtin:hold")
    cfg.validate()
    return cfg
