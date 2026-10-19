"""Dropbear flat-ground tracking configs: training (``Dropbear-Tracking-Flat-v0``) and play (``...-Play-v0``)."""
from __future__ import annotations

from isaaclab.utils import configclass

from dropbear_wbc.robots.defaults import resolve_default_pose
from dropbear_wbc.robots.dropbear import make_dropbear_cfg

from ...tracking_env_cfg import TrackingEnvCfg


DEPLOYABLE_TRACKING_PROFILE: str = "hw_v1ie"
"""Default actuator profile of the NoState (deployable) tracking tasks: the profile of every deployable walk that
passed the hardware gate (docs/OVERNIGHT_BREV_2026-09-25.md). ``legacy`` puts armature 0.01 on every body motor,
2.1-2.4x under the datasheet calf/ankle inertia, with gains no motor can run (notes/13 T1)."""


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
    """``legacy`` (the implicit groups of the state-observing tracking tasks), ``legacy_hwarm`` (legacy gains + the
    datasheet armature, an A/B control) or a real-actuator twin profile of ``robots.hw_motor_specs.HW_PROFILE_MAPS``
    (docs/archive/ACTUATORS_HISTORY.md section 11). The NoState (deployable) tasks default to :data:`DEPLOYABLE_TRACKING_PROFILE`."""

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
        """Real-hardware regularizers for an ``hw_*`` profile (docs/archive/ACTUATORS_HISTORY.md sections 11-13): ``motor_over_rated``
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

    def enable_height_tracking(self, anchor: float = 1.0, feet: float = 1.0, anchor_std: float = 0.1,
                               feet_std: float = 0.1) -> None:
        """Reward matching the reference's HEIGHTS above the ground: the anchor (head) for squat depth and jump height,
        the feet for foot lift, one-leg stance and flight (0 = off). Without them the library tracker kept both feet
        planted through jumps, kicks, one-leg balance and squats while holding 99/138 clips without a fall
        (tools/motion_fidelity.py, docs/RESULTS.md 2026-10-16)."""
        from isaaclab.managers import RewardTermCfg as RewTerm

        from dropbear_wbc.robots.dropbear_names import FOOT_EE_BODIES
        from dropbear_wbc.tasks.tracking import mdp

        if anchor:
            self.rewards.motion_anchor_height = RewTerm(
                func=mdp.motion_anchor_height_error_exp, weight=float(anchor),
                params={"command_name": "motion", "std": float(anchor_std)})
        if feet:
            self.rewards.motion_feet_height = RewTerm(
                func=mdp.motion_feet_height_error_exp, weight=float(feet),
                params={"command_name": "motion", "std": float(feet_std), "body_names": list(FOOT_EE_BODIES)})

    def set_tracking_strictness(self, body_pos_std: float | None = None, joint_pos: float = 0.0,
                                joint_pos_std: float = 0.25, end_threshold_m: float | None = None) -> None:
        """Make "follow the reference" harder to skip (docs/ROADMAP.md, the tracker's objective).

        ``body_pos_std`` [m]: tolerance of the body-position term (default 0.3; the feet are 2 of its 14 points, so
        missing a 21 cm foot lift cost ~3 %). ``joint_pos`` (> 0 = on): reward on the 22 motor angles with tolerance
        ``joint_pos_std`` [rad]. ``end_threshold_m``: end the episode when the head or a foot/hand height is this far
        from the reference (default 0.25 m, more than the hard motions ever missed by)."""
        from isaaclab.managers import RewardTermCfg as RewTerm

        from dropbear_wbc.tasks.tracking import mdp

        if body_pos_std is not None:
            self.rewards.motion_body_pos.params["std"] = float(body_pos_std)
        if joint_pos:
            self.rewards.motion_joint_pos = RewTerm(func=mdp.motion_joint_pos_error_exp, weight=float(joint_pos),
                                                    params={"command_name": "motion", "std": float(joint_pos_std)})
        if end_threshold_m is not None:
            self.terminations.anchor_pos.params["threshold"] = float(end_threshold_m)
            self.terminations.ee_body_pos.params["threshold"] = float(end_threshold_m)

    def enable_knee_band(self, stiffness: float, engage_deg: float) -> None:
        """An elastic beside each knee motor (a theraband or tendon): ``-stiffness * max(0, crank - engage)`` on the knee
        crank, so it helps hold the robot up once the knee is bent past ``engage_deg`` of crank (0-30 deg = knee
        0-47 deg). Logged as telemetry ``tau_spring``; not motor torque. Call after ``set_actuator_profile``."""
        self._knee_band = (float(stiffness), float(engage_deg))
        self._rebuild_hw_groups()

    def enable_safety_penalties(self, stop_load: float = 0.0, impact: float = 0.0, impact_threshold_n: float = 900.0) -> None:
        """``stop_load`` (< 0): penalty on end-stop torque above a safe value (``stop_load_l2``; needs the stop model).
        ``impact`` (< 0): penalty on each foot's contact force above ``impact_threshold_n`` [N] (Isaac Lab
        ``contact_forces``): landings drove 2-3 kN (4-5x body weight) through the ankle and knee pins
        (``tools/joint_loads.py``, 2026-10-17)."""
        from isaaclab.envs import mdp as il_mdp
        from isaaclab.managers import RewardTermCfg as RewTerm
        from isaaclab.managers import SceneEntityCfg

        from dropbear_wbc.tasks.locomotion.mdp.rewards import stop_load_l2

        if stop_load:
            self.rewards.stop_load = RewTerm(func=stop_load_l2, weight=float(stop_load),
                                             params={"asset_cfg": SceneEntityCfg("robot")})
        if impact:
            self.rewards.foot_impact = RewTerm(
                func=il_mdp.contact_forces, weight=float(impact),
                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(__import__('dropbear_wbc.robots.dropbear_names', fromlist=['x']).FOOT_BODIES), preserve_order=True),
                        "threshold": float(impact_threshold_n)})

    def _rebuild_hw_groups(self) -> None:
        from dropbear_wbc.robots.hw_actuators import make_hw_profile_groups, set_motor_groups
        from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS

        if self.actuator_profile not in HW_PROFILE_MAPS:
            raise ValueError(f"needs an hw_* actuator profile, not {self.actuator_profile!r}")
        set_motor_groups(self.scene.robot.actuators,
                         make_hw_profile_groups(self.actuator_profile, stop_model=getattr(self, "_stop_model", False),
                                                knee_band=getattr(self, "_knee_band", None)))

    def enable_stop_model(self) -> None:
        """Model the motor joints' end stops explicitly (``hw_motor_specs.STOP_MODEL``): the actuator's stop spring acts
        past each mechanical limit and PhysX's rigid limit moves outward, so the torque the stop parts carry is known
        (``DatasheetMotor.stop_effort``, telemetry ``tau_stop``). Needs an ``hw_*`` profile; call after
        ``set_actuator_profile``."""
        import math

        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES
        from dropbear_wbc.robots.hw_actuators import make_hw_profile_groups, set_motor_groups
        from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS, STOP_MODEL
        from dropbear_wbc.tasks.joint_limit_events import widen_joint_limits

        if self.actuator_profile not in HW_PROFILE_MAPS:
            raise ValueError(f"the stop model needs an hw_* actuator profile, not {self.actuator_profile!r}")
        from dropbear_wbc.robots.hw_motor_specs import motor_stop_limits

        self._stop_model = True
        self._rebuild_hw_groups()
        stopped = [m for m, (lo, hi) in motor_stop_limits().items() if math.degrees(hi - lo) < 350.0 and m in MOTOR_NAMES]
        self.events.widen_stop_limits = EventTerm(
            func=widen_joint_limits, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=stopped, preserve_order=True),
                    "margin_rad": math.radians(STOP_MODEL["margin_deg"])})

    def enable_armature_dr(self) -> None:
        """Install per-reset log-uniform armature DR on the notes/13 T2 critical joints (``*_armdr`` profiles).

        Requires the matching ``make_hw_motor_groups(..., armature_dr=True)`` groups (ImplicitActuator on calves +
        wrist roll). Play configs should clear ``self.events.armature_dr`` like the other startup/reset DR terms.
        """
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        from dropbear_wbc.robots.hw_motor_specs import ARMATURE_DR_CRITICAL_JOINTS, HW_PROFILE_MAPS, armature_dr_ranges
        from dropbear_wbc.tasks.tracking import mdp

        if self.actuator_profile not in HW_PROFILE_MAPS:
            raise ValueError(f"armature DR needs an hw_* profile, not {self.actuator_profile!r}")
        ranges = armature_dr_ranges(HW_PROFILE_MAPS[self.actuator_profile], joints=ARMATURE_DR_CRITICAL_JOINTS)
        self.events.armature_dr = EventTerm(
            func=mdp.randomize_joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(ranges), preserve_order=True),
                "armature_ranges": ranges,
            },
        )

    def set_actuator_profile(self, name: str) -> None:
        """Switch the body-motor groups to ``legacy`` or an ``hw_*`` twin profile (neck and passive groups unchanged).
        ``*_armdr`` profiles also install :meth:`enable_armature_dr`."""
        from dropbear_wbc.robots.hw_actuators import legacy_motor_groups, make_hw_profile_groups, set_motor_groups
        from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS, HW_PROFILE_OPTIONS, LEGACY_HW_ARMATURE_PROFILES

        if name == "legacy":
            set_motor_groups(self.scene.robot.actuators, legacy_motor_groups())
        elif name in LEGACY_HW_ARMATURE_PROFILES:
            set_motor_groups(self.scene.robot.actuators, legacy_motor_groups(LEGACY_HW_ARMATURE_PROFILES[name]))
        elif name in HW_PROFILE_MAPS:
            set_motor_groups(self.scene.robot.actuators, make_hw_profile_groups(name))
        else:
            raise ValueError(f"unknown tracking actuator profile {name!r}; known: legacy, "
                             f"{sorted(LEGACY_HW_ARMATURE_PROFILES)}, {sorted(HW_PROFILE_MAPS)}")
        self.actuator_profile = name
        if HW_PROFILE_OPTIONS.get(name, {}).get("armature_dr"):
            self.enable_armature_dr()
        elif getattr(self.events, "armature_dr", None) is not None:
            self.events.armature_dr = None
        self._set_motor_mass_offsets(name)

    def _set_motor_mass_offsets(self, profile: str) -> None:
        """Startup term ``motor_mass``: the plant mass change of the profile's motors vs the CAD (twin v3 profiles
        only; ``hw_motor_specs.motor_mass_offsets``). Kept in play: it is the plant, not a randomization."""
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        from dropbear_wbc.robots.hw_motor_specs import motor_mass_offsets
        from dropbear_wbc.tasks.mass_events import apply_body_mass_offsets

        offsets = motor_mass_offsets(profile)
        self.events.motor_mass = (EventTerm(func=apply_body_mass_offsets, mode="startup",
                                            params={"asset_cfg": SceneEntityCfg("robot"), "offsets": offsets})
                                  if offsets else None)


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
        self.events.add_base_mass = None
        self.events.scale_limb_mass = None
        if getattr(self.events, "armature_dr", None) is not None:
            self.events.armature_dr = None
        motion = self.commands.motion
        motion.pose_range = {}
        motion.velocity_range = {}
        motion.joint_position_range = (0.0, 0.0)
        motion.closure_joint_position_range = (0.0, 0.0)
        motion.start_at_zero = True
        motion.debug_vis = True
