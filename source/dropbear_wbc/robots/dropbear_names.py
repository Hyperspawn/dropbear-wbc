"""Dropbear name constants (joints, bodies, groups). Pure Python: no Isaac/torch imports.

Everything here refers to names in the user's USD (plant authority, see ``docs/CONTRACTS.md``
section 0) with SHA-256 ``45586414...``. Isaac Lab exposes 90 articulation bodies and 91 DOFs for it
(22 body motors + 6 neck lead screws + 63 passive DOFs, where the spherical joints
``*_Revolute115``/``*_Revolute117`` appear as three DOFs each: ``<name>:0``, ``:1``, ``:2``).

Frames and units: world frame x forward, y left, z up (metres). The articulation root body is
``world`` (the rigid torso+pelvis, 19.5 kg). Its frame origin sits about 12.5 cm *below* the
soles at the authored rest pose (foot collision AABB min z = 0.12497 m in the rest pose, see
``logs/calibrate_settle/usd_joint_frames.json``), so root height is negative when standing.
Angles are radians unless a name ends in ``_DEG``.
"""
from __future__ import annotations

import os
from pathlib import Path

from dropbear_wbc import paths as _paths

DEFAULT_USD_PATH: str = str(_paths.usd_path())
"""The plant USD (read-only plant authority): ``$DROPBEAR_USD`` / ``.dropbear.env``, else ``assets/dropbear.usd``
(``python tools/fetch_assets.py`` downloads it). See :mod:`dropbear_wbc.paths`."""

USD_SHA256: str = "45586414b065cd982d487cbd868fe982108b3b8ccec64d3dfcf629652ed8db0f"
"""SHA-256 of the USD all names below were verified against."""

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
"""The repository root (resolved from this file's location)."""


def resolve_usd_path() -> str:
    """Return the plant USD path (:func:`dropbear_wbc.paths.usd_path`: ``$DROPBEAR_USD``, ``.dropbear.env``, then
    ``assets/dropbear.usd``)."""
    return str(_paths.usd_path())


# ---------------------------------------------------------------------------------------------
# Joints
# ---------------------------------------------------------------------------------------------

MOTOR_NAMES: tuple[str, ...] = (
    # left leg (SDK slots 0-5)
    "PG_left_leg_pitch",
    "PG_left_leg_roll",
    "LL_hip_joint",
    "LL_knee_actuator_joint",
    "LL_Revolute67",
    "LL_Revolute81",
    # right leg (slots 6-11)
    "PG_right_leg_pitch",
    "PG_right_leg_roll",
    "RL_hip_joint",
    "RL_knee_actuator_joint",
    "RL_Revolute67",
    "RL_Revolute81",
    # left arm (slots 12-16)
    "LH_yaw",
    "LH_pitch",
    "LH_roll",
    "LH_elbow_joint",
    "LH_wrist_roll",
    # right arm (slots 17-21)
    "RH_yaw",
    "RH_pitch",
    "RH_roll",
    "RH_elbow_joint",
    "RH_wrist_roll",
)
"""The 22 body motors, contract ``dropbear-wbc-motors-v1`` order (index = SDK slot).

Always resolve with ``find_joints(MOTOR_NAMES, preserve_order=True)``."""

NUM_MOTORS: int = len(MOTOR_NAMES)

NECK_NAMES: tuple[str, ...] = tuple(f"head_LeadScrew{i}" for i in range(1, 7))
"""Six prismatic Stewart-platform lead screws (metres). Held by PD at their init value; never actions."""

# Legacy DROPBEAR_CFG actuator groups (dropbear_walk/isaaclab_asset/dropbear.py).
HIP_MOTORS: tuple[str, ...] = (
    "PG_left_leg_pitch", "PG_left_leg_roll", "PG_right_leg_pitch", "PG_right_leg_roll",
    "LL_hip_joint", "RL_hip_joint",
)
KNEE_MOTORS: tuple[str, ...] = ("LL_knee_actuator_joint", "RL_knee_actuator_joint")
ANKLE_MOTORS: tuple[str, ...] = ("LL_Revolute67", "LL_Revolute81", "RL_Revolute67", "RL_Revolute81")
ARM_MOTORS: tuple[str, ...] = (
    "LH_yaw", "LH_pitch", "LH_roll", "LH_elbow_joint", "LH_wrist_roll",
    "RH_yaw", "RH_pitch", "RH_roll", "RH_elbow_joint", "RH_wrist_roll",
)

PASSIVE_LEG_REGEX: str = r"(LL|RL)_(?!(?:hip_joint|knee_actuator_joint|Revolute67|Revolute81)$).+"
"""Every leg joint that is not a leg motor (four-bar, ankle U-joint, tie-rod spherical DOFs)."""
PASSIVE_ARM_REGEX: str = r"(LH|RH)_(?!(?:yaw|pitch|roll|elbow_joint|wrist_roll)$).+"
"""Every arm joint that is not an arm motor (elbow four-bar links)."""
PASSIVE_HEAD_REGEX: str = r"head_(?!LeadScrew[1-6]$).+"
"""Every head joint that is not a lead screw (Stewart U-joints)."""

EXPECTED_NUM_JOINTS: int = 91
EXPECTED_NUM_PASSIVE: int = 63
EXPECTED_NUM_BODIES: int = 90
"""Articulation links. The USD has 93 rigid bodies; 3 of them have no joints (see :data:`ORPHAN_BODIES`)."""
NUM_USD_RIGID_BODIES: int = 93

ORPHAN_BODIES: tuple[str, ...] = (
    "LL_skateboard_bearing__10__1",
    "LL_skateboard_bearing__11__1",
    "RL_skateboard_bearing__11__1",
)
"""18 g knee bearings with colliders but no joints: PhysX treats them as free bodies inside the knee
(not articulation links, so their contacts with the leg are NOT filtered by ``enabled_self_collisions``).
Deactivated at spawn by :mod:`dropbear_wbc.robots.spawn` (in memory; the USD file is untouched)."""

USD_JOINT_FRICTION: dict[str, float] = {
    "PG_left_leg_pitch": 0.5, "PG_left_leg_roll": 0.5, "PG_right_leg_pitch": 0.5, "PG_right_leg_roll": 0.5,
    "LL_Revolute87": 0.1, "RL_Revolute87": 0.1,
}
"""``physxJoint:jointFriction`` authored in the USD (``logs/robot_task/usd_joint_attrs_probe.log``). PhysX scales
this coefficient by the joint *constraint* force, so the loaded hip joints lock up; the robot config zeroes
joint friction by default (``make_dropbear_cfg(joint_friction=None)`` keeps these USD values)."""
JOINT_AXIS_FIXES: dict[str, str] = {"LL_Revolute121": "Z"}
"""Left-knee closure authored with axis X (mirror ``RL_Revolute121`` and all other knee joints: Z, identical
local frames); the X axis locks the left knee four-bar. Applied in memory at spawn."""
MIN_PRINCIPAL_INERTIA: float = 2.0e-6
"""``LH_bicep_1``/``RH_bicep_1`` author diagonal inertia (2e-6, 0, 2e-6) kg*m^2; PhysX rejects zero components
for articulation links. Zero components are raised to this value at spawn."""
SPHERICAL_JOINT_FIXES: tuple[str, ...] = ("LL_Revolute111", "LL_Revolute112", "RL_Revolute111", "RL_Revolute112")
"""Ankle crank->tie-rod loop closures authored as REVOLUTE joints with their axis parallel to the calf-motor axis:
each tie rod stays planar, so foot roll over-constrains the parallel ankle (Grubler mobility 0 instead of 2;
docs/CONTRACTS.md 0.2). Retyped to ``PhysicsSphericalJoint`` in memory at spawn by default (rod-end bearings);
opt out with :data:`AUTHORED_ANKLE_ENV` = 1 or ``authored_ankle_tierods=True``. They are excluded from the
articulation (loop closures), so the articulation DOF/body counts do not change."""
AUTHORED_ANKLE_ENV: str = "DROPBEAR_AUTHORED_ANKLE"
"""Environment variable: ``1`` keeps the authored revolute ankle tie-rod closures (opt-out of the 0.2 fix)."""


def authored_ankle_requested(explicit: bool | None = None) -> bool:
    """True if the authored (over-constrained, revolute) ankle tie rods should be kept.

    ``explicit`` (a config flag) wins when not ``None``; otherwise ``$DROPBEAR_AUTHORED_ANKLE`` in
    ``{"1", "true", "yes"}`` (case-insensitive) opts out of the default spherical retype.
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(AUTHORED_ANKLE_ENV, "").strip().lower() in ("1", "true", "yes")


def spherical_joint_fixes(authored_ankle_tierods: bool | None = None) -> tuple[str, ...]:
    """The closures to retype to spherical at spawn (empty when the authored ankle is requested)."""
    return () if authored_ankle_requested(authored_ankle_tierods) else SPHERICAL_JOINT_FIXES
USD_MOTOR_MAX_JOINT_VELOCITY_RAD_S: float = 10.0
"""``physxJoint:maxJointVelocity`` = 572.96 deg/s authored on the motors; kept (legacy config never overrode it)."""

# Actuator parameters of the legacy DROPBEAR_CFG (the config that produced the only walking policy).
# (effort limit [N*m or N], stiffness [N*m/rad or N/m], damping [N*m*s/rad or N*s/m], armature)
ACTUATOR_PARAMS: dict[str, tuple[float, float, float, float]] = {
    "arms": (40.0, 50.0, 2.0, 0.01),
    "hips": (200.0, 150.0, 5.0, 0.01),
    "knees": (300.0, 200.0, 12.0, 0.01),
    "ankles": (80.0, 80.0, 4.0, 0.01),
    "neck": (100.0, 5000.0, 50.0, 0.001),
    "passive": (50.0, 0.0, 50.0, 0.01),
}
LEGACY_PASSIVE_DAMPING: float = ACTUATOR_PARAMS["passive"][2]
"""The legacy config's passive-joint damping request (50 N*m*s/rad). On this USD it has NO effect in Isaac: none of the
55 movable passive tree joints has an authored ``UsdPhysics.DriveAPI``, and PhysX ignores the drive damping that Isaac
Lab writes through ``set_dof_dampings`` for such joints (read-back shows the value, dynamics are identical for 0 / 0.5 /
5 / 50: ``logs/review_fixes/passive_damping/passive_damping_nodrive.json``)."""
EFFECTIVE_PASSIVE_DAMPING: float = 0.0
"""Passive-joint damping the contract plant ACTUALLY has in Isaac (armature 0.01 only). Other simulators (Newton) use
this value so that sim2sim compares the plant the policies were trained on (docs/CONTRACTS.md 0.3)."""

ACTUATOR_GROUP_MOTORS: dict[str, tuple[str, ...]] = {
    "arms": ARM_MOTORS,
    "hips": HIP_MOTORS,
    "knees": KNEE_MOTORS,
    "ankles": ANKLE_MOTORS,
}
CLOSURE_MOTORS: tuple[str, ...] = (
    "LL_knee_actuator_joint", "RL_knee_actuator_joint",
    "LL_Revolute67", "LL_Revolute81", "RL_Revolute67", "RL_Revolute81",
    "LH_elbow_joint", "RH_elbow_joint",
)
"""Motors that drive a loop closure (knee/elbow four-bars, ankle tie rods). Perturbing one of them without
re-solving the passive joints violates the closure (2.4-3.9 cm anchor gaps after RSI with +/-0.1 rad noise,
``logs/robot_task/smoke_env16_200steps.json``), so RSI noise on them is configured separately (default 0).
The other 14 motors (hips, shoulders, wrists) are serial and can be perturbed freely."""

LEGACY_VELOCITY_LIMITS: dict[str, float] = {"arms": 20.0, "hips": 23.0, "knees": 14.0, "ankles": 9.0, "neck": 0.5}
"""Legacy ``velocity_limit`` values [rad/s, m/s]. Isaac Lab 2.2 does NOT apply ``velocity_limit`` of an
implicit actuator to the simulation (only ``velocity_limit_sim``), so the legacy config never used them.
We keep that behaviour and record them as metadata only."""


def motor_group(name: str) -> str:
    """Return the legacy actuator group (arms/hips/knees/ankles) owning motor ``name``."""
    for group, names in ACTUATOR_GROUP_MOTORS.items():
        if name in names:
            return group
    raise KeyError(f"{name!r} is not a body motor")


def motor_param(name: str, index: int) -> float:
    """Return ``ACTUATOR_PARAMS[group][index]`` for a motor (0 effort, 1 kp, 2 kd, 3 armature)."""
    return ACTUATOR_PARAMS[motor_group(name)][index]


MOTOR_HARD_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "PG_left_leg_pitch": (-15.0, 30.0),
    "PG_left_leg_roll": (-30.0, 30.0),
    "LL_hip_joint": (-50.0, 30.0),
    "LL_knee_actuator_joint": (0.0, 30.0),
    "LL_Revolute67": (-50.0, 60.0),
    "LL_Revolute81": (-50.0, 60.0),
    "PG_right_leg_pitch": (-30.0, 15.0),
    "PG_right_leg_roll": (-30.0, 30.0),
    "RL_hip_joint": (-50.0, 30.0),
    "RL_knee_actuator_joint": (0.0, 30.0),
    "RL_Revolute67": (-50.0, 60.0),
    "RL_Revolute81": (-60.0, 50.0),
    "LH_yaw": (-180.0, 180.0),
    "LH_pitch": (-180.0, 10.0),
    "LH_roll": (-180.0, 180.0),
    "LH_elbow_joint": (0.0, 30.0),
    "LH_wrist_roll": (-180.0, 180.0),
    "RH_yaw": (-180.0, 180.0),
    "RH_pitch": (-180.0, 10.0),
    "RH_roll": (-180.0, 180.0),
    "RH_elbow_joint": (0.0, 30.0),
    "RH_wrist_roll": (-180.0, 180.0),
}
"""Authored USD hard limits of the 22 motors in degrees (``docs/usd_tree_45586414.json``)."""

LEGACY_DEFAULT_MOTOR_POS: dict[str, float] = {
    "PG_left_leg_pitch": -0.1, "PG_left_leg_roll": 0.0, "LL_hip_joint": 0.0,
    "LL_knee_actuator_joint": 0.3, "LL_Revolute67": -0.2, "LL_Revolute81": 0.0,
    "PG_right_leg_pitch": -0.1, "PG_right_leg_roll": 0.0, "RL_hip_joint": 0.0,
    "RL_knee_actuator_joint": 0.3, "RL_Revolute67": -0.2, "RL_Revolute81": 0.0,
    "LH_yaw": 0.0, "LH_pitch": 0.0, "LH_roll": 0.0, "LH_elbow_joint": 0.3, "LH_wrist_roll": 0.0,
    "RH_yaw": 0.0, "RH_pitch": 0.0, "RH_roll": 0.0, "RH_elbow_joint": 0.3, "RH_wrist_roll": 0.0,
}
"""``DROPBEAR_CFG.init_state.joint_pos`` for the motors [rad]. Fallback default when no calibration
JSON provides ``standing_motor_pos`` (contract section 5)."""

# ---------------------------------------------------------------------------------------------
# Bodies
# ---------------------------------------------------------------------------------------------

ROOT_BODY: str = "world"
"""Articulation root (rigid torso + pelvis). Frame origin ~12.5 cm below the soles."""

ANCHOR_BODY: str = "head_5mm_ujoint_base__5__1"
"""Tracking anchor: chest-level (z = 1.639 m at rest) Stewart-base body attached to ``world`` by the
fixed joint ``head_Platform_Base_Joint1`` (a zero-DOF articulation link, so it moves rigidly with the
torso). Verified in Isaac by ``tools/inspect_dropbear_articulation.py``."""

ANCHOR_FRAME_OFFSET_WXYZ: tuple[float, float, float, float] = (0.5, -0.5, -0.5, -0.5)
"""Right-multiplied rotation from the anchor *link* frame to a world-aligned frame (``q_up = q_link * offset``).
The link's authored rest orientation is (0.5, 0.5, 0.5, 0.5) (Fusion CAD frame: link z = world x), so
``q_up`` is x forward / y left / z up at rest, like G1's ``torso_link``. Used ONLY by ``bad_anchor_ori``
(projected-gravity z). Observations and rewards use the raw link frame (BeyondMimic convention, which is
what a deploy runner reconstructs from the root pose and the rigid anchor-in-root offset)."""

FOOT_BODIES: tuple[str, ...] = (
    "LL_skateboard_bearing_left_2", "LL_basis_left_1",
    "RL_skateboard_bearing_left_2", "RL_basis_left_1",
)
"""Bodies allowed to touch the ground at the feet (sole plate + ankle cross)."""

FOOT_EE_BODIES: tuple[str, ...] = ("LL_skateboard_bearing_left_2", "RL_skateboard_bearing_left_2")
"""Foot end-effectors (sole plates; G1 ``*_ankle_roll_link`` analogue)."""

HAND_BODIES: tuple[str, ...] = ("LH_shoulder_ex_al_interface_1", "RH_shoulder_ex_al_interface_1")
"""Hand end-effectors (after the wrist-roll motor; G1 ``*_wrist_yaw_link`` analogue)."""

KEY_BODIES: dict[str, str] = {
    "root": ROOT_BODY,
    "anchor": ANCHOR_BODY,
    "pelvis": ROOT_BODY,  # the pelvis is part of the rigid 'world' body (its frame origin is below the feet)
    "torso": ANCHOR_BODY,
    "left_thigh": "LL_RMD_X10_S2_MIR4__3_Stator_1",
    "right_thigh": "RL_RMD_X10_S2_MIR4__3_Stator_1",
    "left_shank": "LL_double_bracket_10deg_MIR_MIR_MIR_1",
    "right_shank": "RL_double_bracket_10deg_MIR_MIR_MIR_1",
    "left_foot": "LL_skateboard_bearing_left_2",
    "right_foot": "RL_skateboard_bearing_left_2",
    "left_upper_arm": "LH_RMD_X8_Pro_MIR8_MIR1__3__1",
    "right_upper_arm": "RH_RMD_X8_Pro_MIR8_MIR1__3__1",
    "left_forearm": "LH_6mm_bearing__4__1",
    "right_forearm": "RH_6mm_bearing__4__1",
    "left_hand": "LH_shoulder_ex_al_interface_1",
    "right_hand": "RH_shoulder_ex_al_interface_1",
    "head": "head_u_joint_center__8__1",
}
"""Semantic key bodies (G1 analogues). Thigh = hip-pitch child (knee-motor stator housing),
shank = four-bar output bracket carrying the calf motors, foot = sole plate, upper arm = shoulder-roll
child, forearm = elbow four-bar output, hand = wrist-roll child, head = Stewart platform top."""

TRACKED_BODIES: tuple[str, ...] = (
    KEY_BODIES["torso"],
    KEY_BODIES["left_thigh"], KEY_BODIES["left_shank"], KEY_BODIES["left_foot"],
    KEY_BODIES["right_thigh"], KEY_BODIES["right_shank"], KEY_BODIES["right_foot"],
    KEY_BODIES["left_upper_arm"], KEY_BODIES["left_forearm"], KEY_BODIES["left_hand"],
    KEY_BODIES["right_upper_arm"], KEY_BODIES["right_forearm"], KEY_BODIES["right_hand"],
    KEY_BODIES["head"],
)
"""14 tracked bodies (BeyondMimic ``body_names`` analogue of G1's 14-body list)."""

_LEFT_HIP = ("PG_left_leg_pitch", "PG_left_leg_roll", "LL_hip_joint")
_RIGHT_HIP = ("PG_right_leg_pitch", "PG_right_leg_roll", "RL_hip_joint")
_LEFT_SHOULDER = ("LH_yaw", "LH_pitch", "LH_roll")
_RIGHT_SHOULDER = ("RH_yaw", "RH_pitch", "RH_roll")

EXPECTED_BODY_DRIVERS: dict[str, tuple[str, ...]] = {
    KEY_BODIES["torso"]: (),
    KEY_BODIES["head"]: (),
    KEY_BODIES["left_thigh"]: _LEFT_HIP,
    KEY_BODIES["left_shank"]: _LEFT_HIP + ("LL_knee_actuator_joint",),
    KEY_BODIES["left_foot"]: _LEFT_HIP + ("LL_knee_actuator_joint", "LL_Revolute67", "LL_Revolute81"),
    KEY_BODIES["right_thigh"]: _RIGHT_HIP,
    KEY_BODIES["right_shank"]: _RIGHT_HIP + ("RL_knee_actuator_joint",),
    KEY_BODIES["right_foot"]: _RIGHT_HIP + ("RL_knee_actuator_joint", "RL_Revolute67", "RL_Revolute81"),
    KEY_BODIES["left_upper_arm"]: _LEFT_SHOULDER,
    KEY_BODIES["left_forearm"]: _LEFT_SHOULDER + ("LH_elbow_joint",),
    KEY_BODIES["left_hand"]: _LEFT_SHOULDER + ("LH_elbow_joint", "LH_wrist_roll"),
    KEY_BODIES["right_upper_arm"]: _RIGHT_SHOULDER,
    KEY_BODIES["right_forearm"]: _RIGHT_SHOULDER + ("RH_elbow_joint",),
    KEY_BODIES["right_hand"]: _RIGHT_SHOULDER + ("RH_elbow_joint", "RH_wrist_roll"),
}
"""For the motor-sweep check: the motors that are expected to move each tracked body relative to the
root (a motor NOT listed must leave the body still). Derived from the USD kinematic tree
(``docs/usd_tree_45586414.json``)."""


def undesired_contact_body_regex() -> str:
    """Regex (Isaac Lab ``re.fullmatch`` semantics) matching every body except feet and hands."""
    allowed = "|".join(FOOT_BODIES + HAND_BODIES)
    return rf"^(?!(?:{allowed})$).+$"
