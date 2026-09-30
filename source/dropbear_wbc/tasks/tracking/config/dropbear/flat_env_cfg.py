"""Dropbear flat-ground tracking configs: training (``Dropbear-Tracking-Flat-v0``) and play (``...-Play-v0``)."""
from __future__ import annotations

from isaaclab.utils import configclass

from dropbear_wbc.robots.defaults import resolve_default_pose
from dropbear_wbc.robots.dropbear import make_dropbear_cfg

from ...tracking_env_cfg import TrackingEnvCfg


@configclass
class DropbearFlatEnvCfg(TrackingEnvCfg):
    """Training config: 4096 envs by default, solver 32/4, calibration default pose if present."""

    default_pose_source: str = ""
    """Where the motor default pose (action offset) came from (logged and exported)."""
    default_pose_info: dict = {}
    """Calibration provenance (path, sha256, created, usd_sha256, plant_variant); empty for the legacy pose."""

    def __post_init__(self):
        super().__post_init__()
        # calibration standing_motor_pos (data/calibration/dropbear_semantic_calibration.json or
        # $DROPBEAR_CALIBRATION_JSON) if present -- validated fail-closed -- else the legacy DROPBEAR_CFG pose
        pose = resolve_default_pose()
        self.default_pose_source = pose.source
        self.default_pose_info = dict(pose.info)
        self.scene.robot = make_dropbear_cfg(prim_path="{ENV_REGEX_NS}/Robot", default_pose=pose)

    actuator_profile: str = "legacy"
    """``legacy`` (the implicit groups every tracking policy so far was trained with) or a real-actuator twin profile
    of ``robots.hw_motor_specs.HW_PROFILE_MAPS`` (docs/ACTUATORS.md section 11)."""

    target_clamp_margin_deg: float | None = None
    """Motor position targets clipped to the hard limits +- this [deg] (None = unclipped, every run through 2026-09-26)."""

    def clamp_targets_to_limits(self, margin_deg: float = 0.0) -> None:
        """Clip the motor position targets (after scale + offset) to each motor's hard limits +- ``margin_deg``, as a
        deploy runtime that clamps ``p_des`` to the motor range would. Unclipped tracking policies sent targets past a
        stop in 20-84 % of steps (knee crank up to 79 deg past it), which also squeezes the ankle rods into their stops
        (docs/ISSUES.md #25). No authority is lost: from anywhere in the range, kp times the range still saturates the
        leg motors."""
        import math

        from dropbear_wbc.robots.dropbear_names import MOTOR_HARD_LIMITS_DEG

        m = math.radians(margin_deg)
        self.actions.joint_pos.clip = {n: (math.radians(lo) - m, math.radians(hi) + m)
                                       for n, (lo, hi) in MOTOR_HARD_LIMITS_DEG.items()}
        self.target_clamp_margin_deg = float(margin_deg)

    def set_solver_iterations(self, position: int, velocity: int) -> None:
        """Override the PhysX articulation solver iterations (contract default 32/4)."""
        props = self.scene.robot.spawn.articulation_props
        props.solver_position_iteration_count = int(position)
        props.solver_velocity_iteration_count = int(velocity)

    def enable_hw_regularizers(self, thermal: float = -2.0e-4, torque_rate: float = -0.02, knee_stop: float = 0.0,
                               knee_stop_margin_deg: float = 3.0, feet_slide: float = 0.0) -> None:
        """Real-hardware regularizers for an ``hw_*`` profile (docs/ACTUATORS.md sections 11-13): ``motor_over_rated``
        (``thermal`` * sum relu(|tau| - rated)^2 with the profile's rated torques) and ``motor_torque_rate``
        (``torque_rate`` * sum ((tau_t - tau_{t-1}) / peak)^2 between policy steps). Same terms as the velocity task.

        ``knee_stop`` (0 = off): ``knee_stop`` * sum over the two knee cranks of how deep [rad] each sits inside the last
        ``knee_stop_margin_deg`` before a hard stop. Policies on the CEM-60 knee rested the stance leg on the knee's
        flexion stop 30-50 % of the time, the stop carrying the load the motor could not (docs/ISSUES.md #19). The
        references reach that stop only briefly in (unloaded) swing, so the ankle rods, whose references sit on their
        stops, are left out.

        ``feet_slide`` (0 = off): ``feet_slide`` * sum over feet of the sole's planar speed while that foot touches the
        ground (the locomotion task's term). The deployable Kimodo walk landed its right foot at about 2.7 m/s and skid
        18 cm per step, and scuffed the right toe mid-swing (docs/ISSUES.md #20); both are contact at speed."""
        from isaaclab.managers import RewardTermCfg as RewTerm
        from isaaclab.managers import SceneEntityCfg

        from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES
        from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS, joint_hw_params
        from dropbear_wbc.tasks.locomotion.mdp.rewards import motor_torque_over_rated_l2, motor_torque_rate_l2

        if self.actuator_profile not in HW_PROFILE_MAPS:
            raise ValueError(f"hw regularizers need an hw_* actuator profile, not {self.actuator_profile!r}")
        params = joint_hw_params(HW_PROFILE_MAPS[self.actuator_profile])
        motors = SceneEntityCfg("robot", joint_names=list(MOTOR_NAMES), preserve_order=True)
        self.rewards.motor_over_rated = RewTerm(
            func=motor_torque_over_rated_l2, weight=float(thermal),
            params={"asset_cfg": motors, "rated": [float(params[m]["rated_torque"]) for m in MOTOR_NAMES]})
        self.rewards.motor_torque_rate = RewTerm(func=motor_torque_rate_l2, weight=float(torque_rate),
                                                 params={"asset_cfg": motors})
        if knee_stop:
            import math

            from dropbear_wbc.tasks.locomotion.mdp.rewards import joint_near_limit_l1

            self.rewards.knee_stop = RewTerm(
                func=joint_near_limit_l1, weight=float(knee_stop),
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LL_knee_actuator_joint", "RL_knee_actuator_joint"],
                                                    preserve_order=True),
                        "margin": math.radians(float(knee_stop_margin_deg))})
        if feet_slide:
            from dropbear_wbc.robots.dropbear_names import FOOT_BODIES, FOOT_EE_BODIES
            from dropbear_wbc.tasks.locomotion.mdp.rewards import feet_slide as feet_slide_term

            self.rewards.feet_slide = RewTerm(
                func=feet_slide_term, weight=float(feet_slide),
                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODIES), preserve_order=True),
                        "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_EE_BODIES), preserve_order=True),
                        "bodies_per_foot": 2})

    def set_actuator_profile(self, name: str) -> None:
        """Switch the body-motor groups to ``legacy`` or an ``hw_*`` twin profile (neck and passive groups unchanged)."""
        from dropbear_wbc.robots.hw_actuators import legacy_motor_groups, make_hw_profile_groups, set_motor_groups
        from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS

        if name == "legacy":
            set_motor_groups(self.scene.robot.actuators, legacy_motor_groups())
        elif name in HW_PROFILE_MAPS:
            set_motor_groups(self.scene.robot.actuators, make_hw_profile_groups(name))
        else:
            raise ValueError(f"unknown tracking actuator profile {name!r}; known: legacy, {sorted(HW_PROFILE_MAPS)}")
        self.actuator_profile = name


@configclass
class DropbearFlatPlayEnvCfg(DropbearFlatEnvCfg):
    """Play config: few envs, no pushes / observation noise / startup randomization / RSI noise; the motion
    always (re)starts at frame 0 and loops; command markers visible."""

    def __post_init__(self):
        super().__post_init__()
        self.set_num_envs(4)
        self.scene.env_spacing = 3.0
        self.episode_length_s = 3600.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        motion = self.commands.motion
        motion.pose_range = {}
        motion.velocity_range = {}
        motion.joint_position_range = (0.0, 0.0)
        motion.closure_joint_position_range = (0.0, 0.0)
        motion.start_at_zero = True
        motion.debug_vis = True
