"""Isaac Lab ``ArticulationCfg`` for the Dropbear humanoid (user's USD, contract ``dropbear-wbc-motors-v1``).

Derived from the user's legacy ``DROPBEAR_CFG``
(``dropbear_walk/isaaclab_asset/dropbear.py``): same actuator
groups, stiffness/damping/effort limits/armature and neck PD, self-collisions off. Differences (all deliberate;
evidence in ``logs/robot_task/``):

* passive-joint damping: the legacy config asks for d=50 on every passive joint, but that is a *drive* parameter and
  none of the passive tree joints has an authored DriveAPI, so PhysX never applied it (measured:
  ``logs/review_fixes/passive_damping/``; CONTRACTS 0.3). The default is therefore the EFFECTIVE value 0 (armature
  0.01 only), which is physically identical to the legacy request; ``passive_drive_api=True`` makes a non-zero value
  act on the passive revolute joints (opt-in variant, not the contract plant);

* solver iterations are a parameter (default 32 position / 4 velocity, contract section 5);
* ``soft_joint_pos_limit_factor = 1.0`` (soft limits == authored hard limits);
* passive joints are selected by regex (every non-motor, non-neck joint) instead of a name list, and
  the resolved count is checked at runtime (63);
* ``max_angular_velocity`` is 50 rad/s expressed in deg/s (Isaac Lab's unit). The legacy config wrote
  ``50.0`` while its comment says rad/s, which caps every link at 50 deg/s;
* default motor pose = calibration ``standing_motor_pos`` if present, else the legacy init pose;
  passive joints start at 0 (authored, closure-consistent rest) and neck lead screws at 0;
* joint friction is 0 by default: the USD authors PhysX ``jointFriction`` 0.5 on the four ``PG_*`` hip
  motors (and 0.1 on the ankle U-joints ``*_Revolute87``), which PhysX scales by the joint constraint force;
  the legacy config inherited it and the loaded hip motors barely move (``logs/robot_task/inspect_*``).
  It is overridden on the stage because Isaac Lab 2.2's friction setter is a no-op on Isaac Sim 5.x;
* the 3 joint-less USD bodies (``ORPHAN_BODIES``) are deactivated at spawn (in memory);
* the left-knee closure ``LL_Revolute121`` axis is overridden X -> Z (it locks the left knee otherwise) and
  the zero principal inertia of ``*_bicep_1`` is raised to 2e-6 kg*m^2 (PhysX rejects it) -- in memory;
* the ankle crank->tie-rod closures ``*_Revolute111/112`` are retyped revolute -> spherical (rod ends) by default
  (docs/CONTRACTS.md 0.2, over-constrained parallel ankle); opt out with ``authored_ankle_tierods=True`` or
  ``$DROPBEAR_AUTHORED_ANKLE=1``.

Import only after the Isaac Sim app is running.
"""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .defaults import DefaultPose, resolve_default_pose
from .spawn import DropbearUsdFileCfg
from .dropbear_names import (  # noqa: F401  (re-exported for the contract path dropbear_wbc.robots.dropbear.*)
    ACTUATOR_GROUP_MOTORS,
    ACTUATOR_PARAMS,
    ANCHOR_BODY,
    ANCHOR_FRAME_OFFSET_WXYZ,
    ANKLE_MOTORS,
    ARM_MOTORS,
    EXPECTED_BODY_DRIVERS,
    EXPECTED_NUM_BODIES,
    EXPECTED_NUM_JOINTS,
    EXPECTED_NUM_PASSIVE,
    EFFECTIVE_PASSIVE_DAMPING,
    LEGACY_PASSIVE_DAMPING,
    FOOT_BODIES,
    FOOT_EE_BODIES,
    HAND_BODIES,
    HIP_MOTORS,
    KEY_BODIES,
    KNEE_MOTORS,
    LEGACY_DEFAULT_MOTOR_POS,
    MOTOR_HARD_LIMITS_DEG,
    MOTOR_NAMES,
    JOINT_AXIS_FIXES,
    MIN_PRINCIPAL_INERTIA,
    NECK_NAMES,
    NUM_MOTORS,
    ORPHAN_BODIES,
    SPHERICAL_JOINT_FIXES,
    spherical_joint_fixes,
    USD_JOINT_FRICTION,
    PASSIVE_ARM_REGEX,
    PASSIVE_HEAD_REGEX,
    PASSIVE_LEG_REGEX,
    ROOT_BODY,
    TRACKED_BODIES,
    USD_SHA256,
    motor_param,
    resolve_usd_path,
    undesired_contact_body_regex,
)

MAX_LINEAR_VELOCITY_M_S: float = 20.0
MAX_ANGULAR_VELOCITY_DEG_S: float = math.degrees(50.0)
"""50 rad/s in Isaac Lab's deg/s unit (legacy intent: 'Humans max out at ~30 rad/s ... capping at 50 rad/s')."""

DEFAULT_ROOT_HEIGHT: float = -0.10
"""Initial root z [m] before the first reset (feet ~2.5 cm above ground at the rest pose). Every reset of
the tracking task overwrites the root with the motion NPZ state."""


def _actuators(passive_damping: float) -> dict[str, ImplicitActuatorCfg]:
    """Legacy actuator groups. Joint friction is NOT set here: Isaac Lab 2.2 does not apply
    ``ImplicitActuatorCfg.friction`` on Isaac Sim 5.x, so it is handled on the stage (``spawn.py``)."""
    groups: dict[str, ImplicitActuatorCfg] = {}
    for group, motors in ACTUATOR_GROUP_MOTORS.items():
        effort, kp, kd, armature = ACTUATOR_PARAMS[group]
        groups[group] = ImplicitActuatorCfg(
            joint_names_expr=list(motors),
            effort_limit_sim=effort,
            stiffness=kp,
            damping=kd,
            armature=armature,
        )
    effort, kp, kd, armature = ACTUATOR_PARAMS["neck"]
    groups["neck"] = ImplicitActuatorCfg(
        joint_names_expr=list(NECK_NAMES),
        effort_limit_sim=effort,
        stiffness=kp,
        damping=kd,
        armature=armature,
    )
    effort, kp, _, armature = ACTUATOR_PARAMS["passive"]
    groups["passive_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[PASSIVE_LEG_REGEX],
        effort_limit_sim=effort,
        stiffness=kp,
        damping=passive_damping,
        armature=armature,
    )
    groups["passive_arms_head"] = ImplicitActuatorCfg(
        joint_names_expr=[PASSIVE_ARM_REGEX, PASSIVE_HEAD_REGEX],
        effort_limit_sim=effort,
        stiffness=kp,
        damping=passive_damping,
        armature=armature,
    )
    return groups


def make_dropbear_cfg(
    *,
    usd_path: str | None = None,
    solver_position_iterations: int = 32,
    solver_velocity_iterations: int = 4,
    default_pose: DefaultPose | None = None,
    fix_root_link: bool = False,
    activate_contact_sensors: bool = True,
    max_angular_velocity_deg_s: float = MAX_ANGULAR_VELOCITY_DEG_S,
    joint_friction: float | None = 0.0,
    passive_damping: float = EFFECTIVE_PASSIVE_DAMPING,
    passive_drive_api: bool = False,
    remove_orphan_bodies: bool = True,
    fix_left_knee_closure: bool = True,
    fix_zero_inertia: bool = True,
    authored_ankle_tierods: bool | None = None,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> ArticulationCfg:
    """Build the Dropbear articulation config.

    Args:
        usd_path: USD file; default ``$DROPBEAR_USD`` or the user's P: path.
        solver_position_iterations: PhysX TGS position iterations (contract default 32).
        solver_velocity_iterations: PhysX velocity iterations (contract default 4).
        default_pose: Default motor pose; default = calibration standing pose if present, else legacy.
        fix_root_link: Fix the root ``world`` body to the world (inspection/settling only).
        activate_contact_sensors: Apply PhysX contact reporting to all bodies (needed by ContactSensor).
        max_angular_velocity_deg_s: Per-link angular velocity cap [deg/s].
        joint_friction: PhysX joint friction coefficient for every joint; ``None`` keeps the USD values
            (0.5 on the four ``PG_*`` hip motors, 0.1 on ``*_Revolute87``), which lock the loaded hips.
        passive_damping: Damping written to the 63 passive DOFs [N*m*s/rad]. It only acts with ``passive_drive_api``;
            default = the effective contract value 0 (the legacy 50 was never applied by PhysX, CONTRACTS 0.3).
        passive_drive_api: OPT-IN variant: author ``DriveAPI`` on the passive revolute tree joints so that
            ``passive_damping`` acts (the spherical rod-end DOFs stay undamped). Not the contract plant.
        remove_orphan_bodies: Deactivate the 3 joint-less USD bodies at spawn (see ``spawn.py``).
        fix_left_knee_closure: Override ``LL_Revolute121`` axis X -> Z (see ``JOINT_AXIS_FIXES``).
        fix_zero_inertia: Raise zero principal inertia components to ``MIN_PRINCIPAL_INERTIA``.
        authored_ankle_tierods: ``True`` keeps the authored revolute ankle tie-rod closures (over-constrained,
            CONTRACTS 0.2); ``False`` retypes ``SPHERICAL_JOINT_FIXES`` to spherical (the default plant);
            ``None`` reads ``$DROPBEAR_AUTHORED_ANKLE`` (``1`` = authored).
        prim_path: Spawn prim path.
    """
    pose = default_pose if default_pose is not None else resolve_default_pose()
    # Isaac Lab rejects a joint matched by two keys, and unmatched joints default to 0.0, so only the
    # motors are listed: passive joints and neck lead screws start at their authored rest (0).
    joint_pos: dict[str, float] = dict(pose.motor_pos)
    cfg = ArticulationCfg(
        prim_path=prim_path,
        spawn=DropbearUsdFileCfg(
            usd_path=usd_path or resolve_usd_path(),
            deactivate_prims=tuple(ORPHAN_BODIES) if remove_orphan_bodies else (),
            joint_friction_override=joint_friction,
            joint_axis_overrides=dict(JOINT_AXIS_FIXES) if fix_left_knee_closure else {},
            min_principal_inertia=MIN_PRINCIPAL_INERTIA if fix_zero_inertia else None,
            spherical_joint_overrides=spherical_joint_fixes(authored_ankle_tierods),
            passive_drive_api=passive_drive_api,
            passive_drive_exclude=tuple(MOTOR_NAMES) + tuple(NECK_NAMES),
            activate_contact_sensors=activate_contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=MAX_LINEAR_VELOCITY_M_S,
                max_angular_velocity=max_angular_velocity_deg_s,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=solver_position_iterations,
                solver_velocity_iteration_count=solver_velocity_iterations,
                fix_root_link=fix_root_link,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, DEFAULT_ROOT_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators=_actuators(passive_damping),
    )
    return cfg


def default_pose_source(cfg: ArticulationCfg) -> str:
    """Best-effort description of where ``cfg``'s motor defaults came from (for logs/exports)."""
    legacy = all(abs(cfg.init_state.joint_pos.get(n, 0.0) - v) < 1e-9 for n, v in LEGACY_DEFAULT_MOTOR_POS.items())
    return "legacy_DROPBEAR_CFG" if legacy else "calibration_or_custom"


def apply_calibration_default_pos(cfg: ArticulationCfg, path: str | None = None) -> tuple[ArticulationCfg, str]:
    """Override ``cfg.init_state.joint_pos`` motors with the calibration ``standing_motor_pos``.

    Returns:
        ``(cfg, source)``; ``cfg`` is modified in place and returned. If the JSON does not exist the
        config is unchanged and ``source`` says so.

    Raises:
        dropbear_wbc.robots.defaults.CalibrationError: if the JSON exists but is invalid (fail closed).
    """
    from .defaults import load_standing_motor_pos

    pose = load_standing_motor_pos(path)
    if pose is None:
        return cfg, "no_calibration_json"
    cfg.init_state.joint_pos.update(pose)
    return cfg, f"calibration:{path or 'default'}"


DROPBEAR_CFG: ArticulationCfg = make_dropbear_cfg()
"""Default config: 32/4 solver iterations, calibration default pose if present (else legacy)."""
