"""Dropbear fixed-base tabletop push task (Isaac Lab 2.2), the GR00T data-collection scene.

Modelled on unitree_sim_isaaclab's base-fixed G1 tabletop tasks (``tasks/g1_tasks/pick_place_redblock_*``: fixed
robot, table, 6 cm red cube, front + wrist cameras at 640 x 480, uniform object resets, pose-window success). Dropbear
has no hand or gripper, so the task is PUSHING (decision in docs/DECISIONS.md, 2026-09-24): push a 5 cm block into a
10 cm target zone on the table with the cylindrical hand body.

Scene (env frame = Isaac world, per env origin): ground plane; Dropbear with ``fix_root_link`` at
``LAYOUT.root_pos_w`` (legs at the calibrated standing pose, feet ~4 cm above the ground); a static table slab (top
1.13 m); the block (dynamic cube); the zone (visual-only green square, moved by USD xform ops); dome + distant light; a head
camera on the anchor body (640 x 480), optional wrist cameras (320 x 240) and an optional third-person scene camera.

Actions: absolute position targets of the 10 arm motors (SDK slots 12..21, motor-contract order). The runner maps
semantic arm angles to them with the calibration (``SemanticMap``); legs and neck hold their defaults.
Arm gains: the teleop SDK gains (``tools/teleop_arm.py``: shoulder 200/5, elbow motor 600/10, wrist 60/2; effort
limit 40 N*m as the legacy config), not the legacy tracking gains (50/2), so that teleop, scripted and GR00T
rollouts share one low-level controller. The runner adds the teleop gravity feed-forward (``teleop.gravity``) as an
effort target.

Control 20 Hz (sim dt 5 ms, decimation 10); 20 s episodes; terminations: success (block in zone, still, 0.5 s),
block dropped / off the table, time out.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from dropbear_wbc.robots.dropbear import make_dropbear_cfg
from dropbear_wbc.robots.dropbear_names import ARM_MOTORS

from . import cameras as cam
from . import mdp
from .layout import LAYOUT

ARM_JOINTS: list[str] = list(ARM_MOTORS)
"""The 10 arm motors in motor-contract order (LH_yaw .. LH_wrist_roll, RH_yaw .. RH_wrist_roll)."""
TELEOP_ARM_GAINS = {  # group: (joints, kp, kd) -- tools/teleop_arm.py defaults
    "arm_shoulders": ([j for j in ARM_JOINTS if not j.endswith(("elbow_joint", "wrist_roll"))], 200.0, 5.0),
    "arm_elbows": ([j for j in ARM_JOINTS if j.endswith("elbow_joint")], 600.0, 10.0),
    "arm_wrists": ([j for j in ARM_JOINTS if j.endswith("wrist_roll")], 60.0, 2.0),
}


def make_tabletop_robot_cfg(solver_iters: tuple[int, int] = (32, 4)) -> ArticulationCfg:
    cfg = make_dropbear_cfg(fix_root_link=True, activate_contact_sensors=False,
                            solver_position_iterations=solver_iters[0], solver_velocity_iterations=solver_iters[1])
    cfg.init_state.pos = tuple(LAYOUT.root_pos_w)
    arms = cfg.actuators.pop("arms")
    armature = arms.armature
    for name, (joints, kp, kd) in TELEOP_ARM_GAINS.items():
        cfg.actuators[name] = ImplicitActuatorCfg(joint_names_expr=joints, effort_limit_sim=arms.effort_limit_sim,
                                                  stiffness=kp, damping=kd, armature=armature)
    return cfg


def _cuboid(size, color, *, collision: bool, rigid: bool = False, kinematic: bool = False, mass: float | None = None,
            static_friction: float = 0.5, dynamic_friction: float = 0.4) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=tuple(size),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic, disable_gravity=kinematic,
                                                     max_depenetration_velocity=1.0) if rigid else None,
        mass_props=sim_utils.MassPropertiesCfg(mass=mass) if mass is not None else None,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision, contact_offset=0.005,
                                                         rest_offset=0.0) if (collision or rigid) else None,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=static_friction,
                                                        dynamic_friction=dynamic_friction, restitution=0.0)
        if collision else None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), roughness=0.6),
    )


def _root_w(p_root):
    return tuple(float(a + b) for a, b in zip(p_root, LAYOUT.root_pos_w))


@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    robot: ArticulationCfg = make_tabletop_robot_cfg()
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=_cuboid((LAYOUT.table_depth, LAYOUT.table_width, LAYOUT.table_thickness), (0.55, 0.42, 0.30),
                      collision=True, static_friction=LAYOUT.table_static_friction,
                      dynamic_friction=LAYOUT.table_dynamic_friction),
        init_state=AssetBaseCfg.InitialStateCfg(pos=_root_w(LAYOUT.table_center)),
    )
    table_base = AssetBaseCfg(  # visual-only pedestal under the far half of the slab (no collider: keeps the legs free)
        prim_path="{ENV_REGEX_NS}/TableBase",
        spawn=_cuboid((0.20, LAYOUT.table_width * 0.8, LAYOUT.table_top_z + LAYOUT.root_pos_w[2] - LAYOUT.table_thickness),
                      (0.35, 0.27, 0.20), collision=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(
            LAYOUT.table_front_x + LAYOUT.table_depth - 0.12 + LAYOUT.root_pos_w[0], LAYOUT.table_center_y,
            0.5 * (LAYOUT.table_top_z + LAYOUT.root_pos_w[2] - LAYOUT.table_thickness))),
    )
    block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Block",
        spawn=_cuboid((LAYOUT.block_size,) * 3, LAYOUT.block_color, collision=True, rigid=True, mass=LAYOUT.block_mass,
                      static_friction=LAYOUT.block_static_friction, dynamic_friction=LAYOUT.block_dynamic_friction),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_root_w((0.29, 0.34, LAYOUT.block_rest_z + 0.001))),
    )
    zone = AssetBaseCfg(  # visual only (no physics); moved per reset by USD xform ops (mdp._author_zone_prims)
        prim_path="{ENV_REGEX_NS}/Zone",
        spawn=_cuboid((LAYOUT.zone_size, LAYOUT.zone_size, LAYOUT.zone_thickness), LAYOUT.zone_color, collision=False),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_root_w((0.27, 0.24, LAYOUT.table_top_z + 0.5 * LAYOUT.zone_thickness + 0.0005))),
    )
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DistantLightCfg(color=(0.95, 0.95, 0.95), intensity=2500.0, angle=1.0))
    sky_light = AssetBaseCfg(prim_path="/World/skyLight",
                             spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.85), intensity=900.0))
    head_cam: TiledCameraCfg | None = None
    left_wrist_cam: TiledCameraCfg | None = None
    right_wrist_cam: TiledCameraCfg | None = None
    scene_cam: TiledCameraCfg | None = None


def head_camera_cfg() -> TiledCameraCfg:
    pos, rot = cam.head_cam_offset()
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{cam.ANCHOR_BODY}/head_cam",
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=cam.HEAD_CAM_FOCAL_MM, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.03, 20.0)),
        width=cam.HEAD_CAM_RES[0], height=cam.HEAD_CAM_RES[1],
    )


def wrist_camera_cfg(side: str) -> TiledCameraCfg:
    pos, rot = cam.wrist_cam_offset(side)
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{cam.HAND_BODY[side]}/wrist_cam",
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=cam.WRIST_CAM_FOCAL_MM, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.01, 10.0)),
        width=cam.WRIST_CAM_RES[0], height=cam.WRIST_CAM_RES[1],
    )


def scene_camera_cfg() -> TiledCameraCfg:
    pos, rot = cam.scene_cam_pose(LAYOUT.root_pos_w)
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/SceneCam",
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 30.0)),
        width=640, height=480,
    )


@configclass
class ActionsCfg:
    arm_pos = base_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINTS, preserve_order=True,
                                              scale=1.0, use_default_offset=False)


def _arm_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=ARM_JOINTS, preserve_order=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Privileged low-dimensional state (for RL / debugging; the GR00T dataset uses images + arm state)."""

        arm_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": _arm_cfg()})
        arm_vel = ObsTerm(func=base_mdp.joint_vel, params={"asset_cfg": _arm_cfg()})
        block_pose = ObsTerm(func=mdp.block_pose_root)
        zone_pos = ObsTerm(func=mdp.zone_pos_root)
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_tabletop, mode="reset", params={"npz_path": str(mdp.DEFAULT_STAND_NPZ)})


@configclass
class RewardsCfg:
    success = RewTerm(func=mdp.success_reward, weight=1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.block_in_zone_stable)
    block_dropped = DoneTerm(func=mdp.block_dropped)


@configclass
class DropbearTabletopEnvCfg(ManagerBasedRLEnvCfg):
    scene: TabletopSceneCfg = TabletopSceneCfg(num_envs=4, env_spacing=8.0, replicate_physics=True)  # 8 m: wrist cameras must not see neighbour envs
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    curriculum = None
    viewer: ViewerCfg = ViewerCfg(eye=(1.6, 1.2, 1.9), lookat=(0.3, -0.07, 1.1), origin_type="env", env_index=0)

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = LAYOUT.episode_s
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.003
        self.rerender_on_reset = True
        assert abs(1.0 / (self.sim.dt * self.decimation) - LAYOUT.control_hz) < 1e-6

    def enable_cameras(self, head: bool = True, wrists: bool = False, scene: bool = False) -> None:
        self.scene.head_cam = head_camera_cfg() if head else None
        self.scene.left_wrist_cam = wrist_camera_cfg("left") if wrists else None
        self.scene.right_wrist_cam = wrist_camera_cfg("right") if wrists else None
        self.scene.scene_cam = scene_camera_cfg() if scene else None

    def set_solver_iterations(self, pos: int, vel: int) -> None:
        props = self.scene.robot.spawn.articulation_props
        props.solver_position_iteration_count = int(pos)
        props.solver_velocity_iteration_count = int(vel)
