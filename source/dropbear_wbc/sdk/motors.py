"""Motor contract ``dropbear-wbc-motors-v1`` and legacy actuator parameters.

This module is dependency-free (stdlib only) so that the SDK, the Newton bridge,
the policy runner and unit tests can all import it without Isaac Lab.

Slot order is the SDK slot order from ``docs/CONTRACTS.md`` section 1 (legs first,
then arms, like the G1 SDK). Units: radians, rad/s, N*m (revolute motors);
metres, m/s, N (prismatic neck lead screws).

Actuator parameters are the starting point taken from the user's Isaac Lab
``DROPBEAR_CFG`` (``dropbear_walk/
isaaclab_asset/dropbear.py``, the configuration that produced the only walking
policy). They are simulation parameters, not measured motor characteristics.
"""
from __future__ import annotations

from dataclasses import dataclass

CONTRACT_ID = "dropbear-wbc-motors-v1"

MOTOR_NAMES: tuple[str, ...] = (
    # left leg (slots 0-5)
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
"""The 22 body motors in SDK slot order (``dropbear-wbc-motors-v1``)."""

NECK_NAMES: tuple[str, ...] = tuple(f"head_LeadScrew{i}" for i in range(1, 7))
"""The 6 prismatic neck lead screws (SDK slots 22-27, reserved; held by PD)."""

NUM_MOTORS: int = len(MOTOR_NAMES)
NUM_NECK: int = len(NECK_NAMES)

PASSIVE_LEG_JOINTS: tuple[str, ...] = (
    "LL_Revolute28", "LL_Revolute29", "LL_Revolute33", "LL_Revolute37",
    "LL_Revolute47", "LL_Revolute49", "LL_Revolute87", "LL_Revolute88", "LL_Revolute119",
    "LL_Revolute115", "LL_Revolute117",
    "RL_Revolute28", "RL_Revolute29", "RL_Revolute33", "RL_Revolute37",
    "RL_Revolute47", "RL_Revolute49", "RL_Revolute87", "RL_Revolute88", "RL_Revolute119",
    "RL_Revolute115", "RL_Revolute117",
)
"""Passive leg joints (``*_Revolute115/117`` are 3-DOF spherical joints)."""

PASSIVE_ARM_HEAD_JOINTS: tuple[str, ...] = (
    "LH_Revolute32", "LH_Revolute41", "LH_Revolute42", "LH_Revolute44", "LH_Revolute123",
    "RH_Revolute32", "RH_Revolute41", "RH_Revolute42", "RH_Revolute44", "RH_Revolute123",
    "head_Revolute_26", "head_Revolute_27", "head_Revolute_34", "head_Revolute_37",
    "head_Revolute_38", "head_Revolute_44", "head_Revolute_45", "head_Revolute_47",
    "head_Revolute_48", "head_Revolute_54", "head_Revolute_55", "head_Revolute_57",
    "head_Revolute_58", "head_Revolute_64", "head_Revolute_65", "head_Revolute_67",
    "head_Revolute_68", "head_Revolute_74", "head_Revolute_75", "head_Revolute_77",
    "head_Revolute_78", "head_Revolute_84", "head_Revolute_85",
)
"""Passive elbow four-bar and Stewart U-joint joints."""

PASSIVE_JOINTS: tuple[str, ...] = PASSIVE_LEG_JOINTS + PASSIVE_ARM_HEAD_JOINTS


@dataclass(frozen=True)
class ActuatorGroup:
    """One ``ImplicitActuatorCfg`` group of the legacy ``DROPBEAR_CFG``.

    Attributes:
        name: Group name as in ``DROPBEAR_CFG.actuators``.
        joints: USD joint names in the group.
        effort_limit: Torque [N*m] (or force [N] for prismatic joints) limit.
        velocity_limit: Speed limit [rad/s or m/s] (informational in the bridge).
        stiffness: Default PD stiffness kp [N*m/rad or N/m].
        damping: Default PD damping kd [N*m*s/rad or N*s/m].
        armature: Reflected rotor inertia [kg*m^2 or kg].
    """

    name: str
    joints: tuple[str, ...]
    effort_limit: float
    velocity_limit: float
    stiffness: float
    damping: float
    armature: float


LEGACY_ACTUATOR_GROUPS: tuple[ActuatorGroup, ...] = (
    ActuatorGroup("arms", ("LH_yaw", "LH_pitch", "LH_roll", "LH_elbow_joint", "LH_wrist_roll",
                           "RH_yaw", "RH_pitch", "RH_roll", "RH_elbow_joint", "RH_wrist_roll"),
                  effort_limit=40.0, velocity_limit=20.0, stiffness=50.0, damping=2.0, armature=0.01),
    ActuatorGroup("hips", ("PG_left_leg_pitch", "PG_left_leg_roll", "PG_right_leg_pitch",
                           "PG_right_leg_roll", "LL_hip_joint", "RL_hip_joint"),
                  effort_limit=200.0, velocity_limit=23.0, stiffness=150.0, damping=5.0, armature=0.01),
    ActuatorGroup("knees", ("LL_knee_actuator_joint", "RL_knee_actuator_joint"),
                  effort_limit=300.0, velocity_limit=14.0, stiffness=200.0, damping=12.0, armature=0.01),
    ActuatorGroup("ankles", ("LL_Revolute67", "LL_Revolute81", "RL_Revolute67", "RL_Revolute81"),
                  effort_limit=80.0, velocity_limit=9.0, stiffness=80.0, damping=4.0, armature=0.01),
    ActuatorGroup("head", NECK_NAMES,
                  effort_limit=100.0, velocity_limit=0.5, stiffness=5000.0, damping=50.0, armature=0.001),
)

LEGACY_PASSIVE_DAMPING: float = 50.0
"""The legacy ``parasitic_*`` request (stiffness 0, damping 50). Isaac never applied it (no DriveAPI on the passive
joints; ``logs/review_fixes/passive_damping/``, docs/CONTRACTS.md 0.3), so it is NOT the contract plant's value."""
PASSIVE_DAMPING: float = 0.0
"""Passive-joint DOF damping [N*m*s/rad] of the contract plant as Isaac actually simulates it (0; armature only).
The Newton plant uses it so that sim2sim compares the plant the policies were trained on. Changed from 50 on
2026-09-24 (review fix); runs before that used 50 (``--passive-damping 50`` reproduces them)."""
PASSIVE_ARMATURE: float = 0.01
"""Passive-joint armature [kg*m^2]."""

LEGACY_INIT_POS: dict[str, float] = {
    "LH_yaw": 0.0, "LH_pitch": 0.0, "LH_roll": 0.0, "LH_elbow_joint": 0.3, "LH_wrist_roll": 0.0,
    "RH_yaw": 0.0, "RH_pitch": 0.0, "RH_roll": 0.0, "RH_elbow_joint": 0.3, "RH_wrist_roll": 0.0,
    "PG_left_leg_pitch": -0.1, "PG_left_leg_roll": 0.0,
    "PG_right_leg_pitch": -0.1, "PG_right_leg_roll": 0.0,
    "LL_hip_joint": 0.0, "LL_knee_actuator_joint": 0.3,
    "RL_hip_joint": 0.0, "RL_knee_actuator_joint": 0.3,
    "LL_Revolute67": -0.2, "LL_Revolute81": 0.0,
    "RL_Revolute67": -0.2, "RL_Revolute81": 0.0,
}
"""``DROPBEAR_CFG.init_state.joint_pos`` for the 22 motors [rad].

Contract section 5 says the default pose is ``standing_motor_pos`` from the
semantic calibration when available; this legacy pose is only the fallback.
"""


def group_of(joint_name: str) -> ActuatorGroup:
    """Return the legacy actuator group that owns ``joint_name``.

    Raises:
        KeyError: if the joint is not an actuated joint.
    """
    for group in LEGACY_ACTUATOR_GROUPS:
        if joint_name in group.joints:
            return group
    raise KeyError(f"{joint_name!r} is not in any actuated group")


def per_motor(attribute: str) -> list[float]:
    """Return ``attribute`` of the owning group for each of the 22 motors, in slot order."""
    return [float(getattr(group_of(name), attribute)) for name in MOTOR_NAMES]


EFFORT_LIMIT: tuple[float, ...] = tuple(per_motor("effort_limit"))
"""Per-motor effort limit [N*m], slot order."""
VELOCITY_LIMIT: tuple[float, ...] = tuple(per_motor("velocity_limit"))
"""Per-motor velocity limit [rad/s], slot order."""
DEFAULT_KP: tuple[float, ...] = tuple(per_motor("stiffness"))
"""Per-motor default stiffness [N*m/rad], slot order."""
DEFAULT_KD: tuple[float, ...] = tuple(per_motor("damping"))
"""Per-motor default damping [N*m*s/rad], slot order."""
ARMATURE: tuple[float, ...] = tuple(per_motor("armature"))
"""Per-motor armature [kg*m^2], slot order."""
DEFAULT_POS: tuple[float, ...] = tuple(LEGACY_INIT_POS[name] for name in MOTOR_NAMES)
"""Legacy default motor pose [rad], slot order."""

NECK_KP: float = group_of(NECK_NAMES[0]).stiffness
NECK_KD: float = group_of(NECK_NAMES[0]).damping
NECK_EFFORT_LIMIT: float = group_of(NECK_NAMES[0]).effort_limit
NECK_ARMATURE: float = group_of(NECK_NAMES[0]).armature

LEFT_LEG = tuple(range(0, 6))
RIGHT_LEG = tuple(range(6, 12))
LEFT_ARM = tuple(range(12, 17))
RIGHT_ARM = tuple(range(17, 22))


def motor_index(name: str) -> int:
    """Return the SDK slot of a motor by USD joint name."""
    return MOTOR_NAMES.index(name)
