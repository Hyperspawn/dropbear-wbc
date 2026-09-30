"""Dropbear velocity-command locomotion environment (contract ``dropbear-velocity-v1``, docs/CONTRACTS.md section 7).

Port of Isaac Lab's ``Isaac-Velocity-Flat-H1-v0`` (``velocity_env_cfg.py`` + ``config/h1/{rough,flat}_env_cfg.py``,
BSD-3) with unitree_rl_lab's command-range curriculum (``Unitree-H1-Velocity``). Dropbear patches, each for a
measured reason (docs/CONTRACTS.md 0.1-0.3, 5.1; dropbear-research AGENTS.md lessons):

* **plant**: ``make_dropbear_cfg`` (spawn fixes: orphans off, hip joint friction 0, ``LL_Revolute121`` axis Z, bicep
  inertia, spherical ankle tie rods, passive damping 0), calibration standing pose as the action offset;
* **actions**: ``JointPositionAction`` on the 22 motors in contract order (the same action interface as the tracking
  task and the SDK sidecar). Legs use BeyondMimic's ``0.25 * effort / kp`` scale (hips 0.333, knees 0.375, ankles
  0.25 rad); arms use a small 0.1 rad scale plus a deviation penalty, so they can swing/counter-balance a little (H1
  keeps its arms in the action with a deviation penalty too) but are not a primary balance actuator (research
  REWARD_DESIGN.md "Arms"). A 12-leg-motor action would need the deploy runner to hold the arms separately;
* **reset**: FULL settled joint row + root state from a standing contract NPZ (``mdp.reset_from_standing_npz``),
  rigid yaw/xy randomization, noise on the serial motors only -- never the airborne CAD pose, never passive joints;
* **torso signals from the anchor (chest) body** ``head_5mm_ujoint_base__5__1`` (height, orientation). The root
  ``world`` frame origin sits ~12.5 cm below the soles, so no root height anywhere. Policy observations use the root
  IMU quantities (``base_ang_vel``, ``projected_gravity``): the root is the rigid torso+pelvis, world-aligned at rest,
  so its orientation IS the torso orientation (the anchor is fixed to it) and they are what ``LowState.imu`` reports;
* **feet**: the explicit contact bodies ``LL/RL_skateboard_bearing_left_2`` (sole plates) + ``LL/RL_basis_left_1``
  (ankle crosses), grouped per foot (never a ``.*skateboard.*`` regex, which also matches knee bearings);
* **terminations**: anchor height / anchor tilt / any non-foot ground contact;
* **commands**: yaw-rate (no heading) commands, 10 % standing envs, initial ranges vx 0..0.5, vy +-0.2, yaw +-0.5
  widened by the curriculum towards vx -0.3..1.0, vy +-0.4, yaw +-1.0;
* pushes OFF initially (``events.push_robot`` configured but disabled; enable for robustness later);
* PhysX: 8/4 solver iterations for training (CONTRACTS 5 measurement), GPU buffers sized from ``num_envs``.

Timing: sim dt 0.005 s, decimation 4 (50 Hz policy), 20 s episodes.
"""
from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from dropbear_wbc.robots.dropbear_names import (
    ANCHOR_BODY,
    ANCHOR_FRAME_OFFSET_WXYZ,
    ARM_MOTORS,
    FOOT_BODIES,
    FOOT_EE_BODIES,
    MOTOR_NAMES,
    ROOT_BODY,
    motor_param,
)
from dropbear_wbc.tasks.tracking.tracking_env_cfg import size_physx_buffers

from . import mdp
from .stand import DEFAULT_STAND_NPZ

MOTORS: list[str] = list(MOTOR_NAMES)
LEG_MOTORS: list[str] = [n for n in MOTOR_NAMES if n not in ARM_MOTORS]
HIP_ROLL_YAW_MOTORS: list[str] = ["PG_left_leg_pitch", "PG_left_leg_roll", "PG_right_leg_pitch", "PG_right_leg_roll"]
"""Dropbear's ``PG_*`` hip motors are the roll-like and yaw-like DOFs (CONTRACTS section 1; ``*_hip_joint`` is pitch)."""
ARM_ACTION_SCALE: float = 0.1
ACTION_SCALE: dict[str, float] = {
    n: (ARM_ACTION_SCALE if n in ARM_MOTORS else 0.25 * motor_param(n, 0) / motor_param(n, 1)) for n in MOTOR_NAMES
}
"""Per-motor action scale [rad per unit action]: legs BeyondMimic rule, arms 0.1."""

FOOT_SENSOR_BODIES: list[str] = [FOOT_BODIES[0], FOOT_BODIES[1], FOOT_BODIES[2], FOOT_BODIES[3]]
"""Left sole, left ankle cross, right sole, right ankle cross (grouped per foot, 2 bodies each)."""
SOLE_BODIES: list[str] = list(FOOT_EE_BODIES)
STAND_ANCHOR_Z_FALLBACK: float = 1.4978
"""Anchor height of ``dropbear_static_stand.npz`` (calibration v3 plant); replaced by the NPZ value at config time."""


def motors_cfg(names: list[str] | None = None) -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(names or MOTORS), preserve_order=True)


def anchor_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=[ANCHOR_BODY], preserve_order=True)


def non_foot_body_regex() -> str:
    """Every body except the four foot bodies (Isaac Lab ``re.fullmatch``)."""
    allowed = "|".join(FOOT_BODIES)
    return rf"^(?!(?:{allowed})$).+$"


@configclass
class VelocitySceneCfg(InteractiveSceneCfg):
    """Flat ground, Dropbear, lights, contact sensor with air-time tracking on every robot body."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot: ArticulationCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0)
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.5), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.5, 0.5)),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=MOTORS, preserve_order=True, scale=ACTION_SCALE, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations (order preserved; names follow ``dropbear_wbc.deploy.observations`` functions).
        ``base_lin_vel`` is sim-only (as in Isaac Lab's H1 flat task); everything else is available from LowState."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": motors_cfg()}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": motors_cfg()}, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic: the policy terms without noise."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": motors_cfg()})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": motors_cfg()})
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROOT_BODY),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    # reset
    reset_robot = EventTerm(
        func=mdp.reset_from_standing_npz,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "npz_path": str(DEFAULT_STAND_NPZ),
            "frame_mode": "random",
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
            "leg_serial_noise": (-0.02, 0.02),
            "arm_serial_noise": (-0.1, 0.1),
        },
    )
    # interval (disabled initially: the config sets it to None in __post_init__ unless enable_pushes)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    # -- task (H1)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # -- base (root CoM velocity = torso+pelvis CoM; orientation/height from the anchor body)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(
        func=mdp.anchor_flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": anchor_cfg(), "up_offset_wxyz": ANCHOR_FRAME_OFFSET_WXYZ},
    )
    anchor_height_l2 = RewTerm(
        func=mdp.anchor_height_l2, weight=-5.0, params={"asset_cfg": anchor_cfg(), "target_height": STAND_ANCHOR_Z_FALLBACK}
    )
    # -- joints (motors only; passive four-bar/tie-rod DOFs are never penalized or commanded)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6, params={"asset_cfg": motors_cfg()})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7, params={"asset_cfg": motors_cfg()})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": motors_cfg(LEG_MOTORS)})
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": motors_cfg(HIP_ROLL_YAW_MOTORS)}
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": motors_cfg(list(ARM_MOTORS))}
    )
    # -- feet (H1 flat: air time weight 1.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_SENSOR_BODIES, preserve_order=True),
            "bodies_per_foot": 2,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_SENSOR_BODIES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=SOLE_BODIES, preserve_order=True),
            "bodies_per_foot": 2,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_height = DoneTerm(
        func=mdp.anchor_height_below, params={"asset_cfg": anchor_cfg(), "minimum_height": STAND_ANCHOR_Z_FALLBACK - 0.3}
    )
    anchor_tilt = DoneTerm(
        func=mdp.anchor_tilt_above,
        params={"asset_cfg": anchor_cfg(), "max_tilt_rad": 0.8, "up_offset_wxyz": ANCHOR_FRAME_OFFSET_WXYZ},
    )
    non_foot_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[non_foot_body_regex()]), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    command_levels = CurrTerm(func=mdp.velocity_cmd_levels)


@configclass
class VelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Base velocity environment (robot set by the Dropbear subclass)."""

    scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.6, 2.6, 0.9), lookat=(0.0, 0.0, -0.5), origin_type="asset_body", asset_name="robot", body_name=ANCHOR_BODY
    )

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        size_physx_buffers(self.sim.physx, self.scene.num_envs)

    def set_num_envs(self, num_envs: int) -> None:
        """Change ``scene.num_envs`` and re-size the PhysX GPU buffers accordingly."""
        self.scene.num_envs = int(num_envs)
        size_physx_buffers(self.sim.physx, self.scene.num_envs)
