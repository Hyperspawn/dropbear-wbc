"""Dropbear motion-tracking environment (BeyondMimic ``tracking_env_cfg.py`` port, contract ``dropbear-tracking-v1``).

Upstream: ``whole_body_tracking/tasks/tracking/tracking_env_cfg.py`` (MIT). Dropbear patches:

* actions: ``JointPositionAction`` on the 22 motors (``preserve_order=True``, ``use_default_offset=True``),
  per-motor scale ``0.25 * effort_limit / stiffness`` (BeyondMimic's G1 rule);
* joint observations (``joint_pos_rel``, ``joint_vel_rel``) and ``joint_pos_limits`` penalty: motors only;
* ``add_joint_default_pos``: motors only; ``base_com``: on the root ``world``;
* RSI joint noise (+/-0.1 rad, BeyondMimic) only on the 14 serial motors; the 8 closure-coupled motors
  (knees, ankles, elbows; ``CLOSURE_MOTORS``) get none by default (it would tear the loop closures);
* ``undesired_contacts``: every body except the feet (sole plate + ankle cross) and the hands;
* terminations: anchor height/orientation and foot + hand height; the orientation check uses a world-aligned
  anchor frame (``ANCHOR_FRAME_OFFSET_WXYZ``), observations/rewards use the raw anchor link frame (as upstream);
* PhysX GPU buffers sized from ``num_envs`` (90 mesh-collider links per robot), see :func:`size_physx_buffers`;
* no MDL ground material (Nucleus download); the default preview-surface plane is used.

Timing: sim dt 0.005 s, decimation 4 (50 Hz policy), 10 s episodes.
"""
from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from dropbear_wbc.robots.dropbear_names import (
    ANCHOR_BODY,
    CLOSURE_MOTORS,
    ANCHOR_FRAME_OFFSET_WXYZ,
    FOOT_EE_BODIES,
    HAND_BODIES,
    MOTOR_NAMES,
    ROOT_BODY,
    TRACKED_BODIES,
    motor_param,
    undesired_contact_body_regex,
)

from . import mdp

MOTORS: list[str] = list(MOTOR_NAMES)
ACTION_SCALE: dict[str, float] = {n: 0.25 * motor_param(n, 0) / motor_param(n, 1) for n in MOTOR_NAMES}
"""BeyondMimic rule ``0.25 * effort_limit / stiffness`` per motor [rad per unit action]."""

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


def motors_cfg(asset: str = "robot") -> SceneEntityCfg:
    """SceneEntityCfg selecting the 22 motors in contract order."""
    return SceneEntityCfg(asset, joint_names=MOTORS, preserve_order=True)


def size_physx_buffers(physx: PhysxCfg, num_envs: int) -> PhysxCfg:
    """Scale PhysX GPU buffers with ``num_envs`` (never below Isaac Lab / BeyondMimic defaults).

    Dropbear has 90 articulation links with convex-hull mesh colliders; self-collisions are off, so pairs are
    robot-vs-ground only, but every body's shapes enter the broadphase aggregate. Values were checked for
    overflow warnings in the throughput logs (``logs/robot_task/throughput_*``).
    """
    n = max(int(num_envs), 1)
    physx.gpu_max_rigid_contact_count = max(2**23, n * 2**11)
    physx.gpu_max_rigid_patch_count = max(10 * 2**15, n * 2**8)
    physx.gpu_found_lost_pairs_capacity = max(2**21, n * 2**10)
    physx.gpu_found_lost_aggregate_pairs_capacity = max(2**25, n * 2**13)
    physx.gpu_total_aggregate_pairs_capacity = max(2**21, n * 2**10)
    physx.gpu_collision_stack_size = max(2**26, n * 2**15)
    physx.gpu_heap_capacity = max(2**26, n * 2**14)
    physx.gpu_temp_buffer_capacity = max(2**24, n * 2**12)
    return physx


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    """Flat ground, Dropbear, lights, contact sensor on every robot body."""

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
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=False
    )


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=MISSING,
        anchor_body_name=ANCHOR_BODY,
        body_names=list(TRACKED_BODIES),
        joint_names=MOTORS,
        root_body_name=ROOT_BODY,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        closure_joint_names=list(CLOSURE_MOTORS),
        closure_joint_position_range=(0.0, 0.0),
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
        """Actor observations (order preserved; see the export sidecar for the layout)."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": motors_cfg()}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": motors_cfg()}, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": motors_cfg()})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": motors_cfg()})
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={"asset_cfg": motors_cfg(), "pos_distribution_params": (-0.01, 0.01), "operation": "add"},
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROOT_BODY),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.3}
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.4}
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.3}
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.4}
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 1.0}
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 3.14}
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": motors_cfg()})
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[undesired_contact_body_regex()]),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(func=mdp.bad_anchor_pos_z_only, params={"command_name": "motion", "threshold": 0.25})
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
            "up_offset_wxyz": ANCHOR_FRAME_OFFSET_WXYZ,
        },
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25, "body_names": list(FOOT_EE_BODIES + HAND_BODIES)},
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Base tracking environment (robot set by the Dropbear subclass)."""

    scene: TrackingSceneCfg = TrackingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.2, 2.2, 0.6), lookat=(0.0, 0.0, -0.4), origin_type="asset_body", asset_name="robot", body_name=ANCHOR_BODY
    )

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        size_physx_buffers(self.sim.physx, self.scene.num_envs)

    def set_num_envs(self, num_envs: int) -> None:
        """Change ``scene.num_envs`` and re-size the PhysX GPU buffers accordingly."""
        self.scene.num_envs = int(num_envs)
        size_physx_buffers(self.sim.physx, self.scene.num_envs)
