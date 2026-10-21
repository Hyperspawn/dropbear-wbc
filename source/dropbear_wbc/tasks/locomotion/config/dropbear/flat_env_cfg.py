"""Dropbear flat-ground velocity configs: training (``Dropbear-Velocity-Flat-v0``) and play (``...-Play-v0``)."""
from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from dropbear_wbc.robots.defaults import resolve_default_pose
from dropbear_wbc.robots.dropbear import make_dropbear_cfg
from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS

from ...stand import DEFAULT_STAND_NPZ, summarize_stand_npz
from ...velocity_env_cfg import VelocityEnvCfg

ANCHOR_FALL_DROP_M: float = 0.30
"""Terminate when the anchor is this far below its settled standing height (knee range allows only ~15 cm squat)."""

ACTUATOR_PROFILES: dict[str, dict[str, dict[str, float]]] = {
    # legacy DROPBEAR_CFG gains (== tracking task): knee crank kp 200 / kd 12 / 300 N*m
    "legacy": {},
    # knee crank stiffness raised so the REFLECTED knee stiffness matches H1's knee (kp 200 / kd 5 at the joint):
    # 600 / 20 on the crank -> ~186 / ~6.2 at the knee through the four-bar (LUT slope 1.79-1.82 at the stand);
    # effort kept at the legacy 300 N*m (3x the RMD-X10-S2 V3 datasheet peak: NOT physically plausible)
    "stiff_knee": {"knees": {"stiffness": 600.0, "damping": 20.0}},
    # as stiff_knee, knee crank effort capped at the RMD-X10-S2 V3 (X10-100, 1:35) datasheet PEAK torque 100 N*m
    # (rated 50 N*m); assumes the X10-S2 drives the crank directly (USD: knee joint = X10-S2 stator -> rotor)
    "stiff_knee_hw": {"knees": {"stiffness": 600.0, "damping": 20.0, "effort_limit_sim": 100.0}},
}
"""Per-task leg actuator settings (Unitree deploy.yaml-style per-joint stiffness). The velocity task only; the tracking
task keeps the legacy gains its policies were trained with. The action interface (offset + 0.375 rad knee scale) is
the same for every profile. The export reads the LIVE sim gains into the sidecar (``joint_stiffness/joint_damping``).
Decision + evidence: docs/DECISIONS.md, docs/LOCOMOTION.md section 5."""
DEFAULT_ACTUATOR_PROFILE: str = "stiff_knee_hw"

HW_ACTUATOR_PROFILES: dict[str, str] = dict(HW_PROFILE_MAPS)
"""Real-actuator twin profiles (robots/hw_motor_specs.py; docs/archive/ACTUATORS_HISTORY.md section 11): explicit PD at the physics rate,
per-joint datasheet torque-speed envelope, rotor armature, gearbox friction and 0-10 ms command delay, motion-mode
gains (kp <= 500, kd <= 5). They REPLACE the legacy implicit motor groups (``hw_*`` groups)."""


@configclass
class DropbearVelocityFlatEnvCfg(VelocityEnvCfg):
    """Training config: 2048 envs, solver 8/4 (CONTRACTS 5 measurement), calibration default pose, pushes off."""

    default_pose_source: str = ""
    """Where the motor default pose (action offset) came from (logged and exported)."""
    default_pose_info: dict = {}
    """Calibration provenance (path, sha256, created, usd_sha256, plant_variant)."""
    stand_info: dict = {}
    """Summary of the reset NPZ (path, anchor_z, ...), see :func:`summarize_stand_npz`."""
    robot_mass_kg: float | None = None
    """Measured total robot mass [kg] the plant is re-centred on (:meth:`set_robot_mass`); ``None`` = CAD + motor swap."""
    actuator_profile: str = ""
    """Name of the applied :data:`ACTUATOR_PROFILES` entry (logged in run_info and exported)."""

    def __post_init__(self):
        super().__post_init__()
        pose = resolve_default_pose()
        self.default_pose_source = pose.source
        self.default_pose_info = dict(pose.info)
        self.scene.robot = make_dropbear_cfg(
            prim_path="{ENV_REGEX_NS}/Robot", default_pose=pose, solver_position_iterations=8, solver_velocity_iterations=4
        )
        self.events.push_robot = None  # pushes off initially (enable_pushes() for a robustness stage)
        self.set_stand_npz(DEFAULT_STAND_NPZ)
        self.set_actuator_profile(DEFAULT_ACTUATOR_PROFILE)

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

        from ...velocity_env_cfg import FOOT_SENSOR_BODIES

        if stop_load:
            self.rewards.stop_load = RewTerm(func=stop_load_l2, weight=float(stop_load),
                                             params={"asset_cfg": SceneEntityCfg("robot")})
        if impact:
            self.rewards.foot_impact = RewTerm(
                func=il_mdp.contact_forces, weight=float(impact),
                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_SENSOR_BODIES), preserve_order=True),
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

    def set_actuator_profile(self, name: str) -> None:
        """Apply ``ACTUATOR_PROFILES[name]`` on top of the legacy actuator groups (fresh legacy values first, so
        switching profiles never accumulates)."""
        from dropbear_wbc.robots.dropbear_names import ACTUATOR_PARAMS
        from dropbear_wbc.robots.hw_actuators import legacy_motor_groups, make_hw_profile_groups, set_motor_groups

        if name in HW_ACTUATOR_PROFILES:
            set_motor_groups(self.scene.robot.actuators, make_hw_profile_groups(name))
            self.actuator_profile = name
            self._set_motor_mass_offsets(name)
            return
        if name not in ACTUATOR_PROFILES:
            known = sorted(ACTUATOR_PROFILES) + sorted(HW_ACTUATOR_PROFILES)
            raise ValueError(f"unknown actuator profile {name!r}; known: {known}")
        if any(k.startswith("hw_") for k in self.scene.robot.actuators):
            set_motor_groups(self.scene.robot.actuators, legacy_motor_groups())
        for group, act in self.scene.robot.actuators.items():
            if group in ACTUATOR_PARAMS:
                effort, kp, kd, _ = ACTUATOR_PARAMS[group]
                act.stiffness, act.damping, act.effort_limit_sim = kp, kd, effort
        for group, values in ACTUATOR_PROFILES[name].items():
            act = self.scene.robot.actuators[group]
            for key, value in values.items():
                setattr(act, key, float(value))
        self.actuator_profile = name
        self._set_motor_mass_offsets(name)

    def _set_motor_mass_offsets(self, profile: str) -> None:
        """Startup term ``motor_mass``: the plant mass change of the profile's motors vs the CAD (twin v3 profiles
        only; ``hw_motor_specs.motor_mass_offsets``), plus the unmodelled mass on ``world`` when :meth:`set_robot_mass`
        gave one (``mass_dr.plant_mass_offsets``). Kept in play: it is the plant, not a randomization."""
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        from dropbear_wbc.robots.mass_dr import plant_mass_offsets
        from dropbear_wbc.tasks.mass_events import apply_body_mass_offsets

        offsets = plant_mass_offsets(profile, self.robot_mass_kg)
        self.events.motor_mass = (EventTerm(func=apply_body_mass_offsets, mode="startup",
                                            params={"asset_cfg": SceneEntityCfg("robot"), "offsets": offsets})
                                  if offsets else None)

    def set_robot_mass(self, robot_mass_kg: float | None) -> None:
        """Re-centre the plant on a measured total mass [kg] (``--robot_mass_kg``): the gap to CAD + the profile's
        motor swap becomes a fixed offset on ``world`` (kept in play) and ``add_base_mass`` a band around it
        (``mass_dr.root_mass_add_range``). ``None`` keeps the default plant and ranges. Order-independent with
        :meth:`set_actuator_profile` (the offset is recomputed there)."""
        from dropbear_wbc.robots.mass_dr import check_robot_mass_kg, root_mass_add_range

        self.robot_mass_kg = check_robot_mass_kg(robot_mass_kg)
        if getattr(self.events, "add_base_mass", None) is not None:
            self.events.add_base_mass.params["mass_distribution_params"] = root_mass_add_range("locomotion",
                                                                                                self.robot_mass_kg)
        self._set_motor_mass_offsets(self.actuator_profile)

    terrain: str = "flat"
    """``flat`` (every run through 2026-09-26) or ``rough`` (:meth:`enable_rough_terrain`)."""

    def enable_rough_terrain(self, max_init_level: int = 3, stair_max_m: float = 0.10, slope_max: float = 0.2) -> None:
        """Blind rough-terrain walking sized for Dropbear (docs/TERRAIN.md): a curriculum grid (10 difficulty rows x
        20 columns of 8 m tiles) of flat ground, random roughness (1-4 cm), pyramid slopes up / down (up to
        ``slope_max``, 0.2 = 11 deg) and pyramid stairs up / down (steps 2 cm -> ``stair_max_m``; the CEM-60 knee and
        the 48 deg knee range make 10 cm the honest ceiling). A down-facing ray caster ``ground_scan`` (0.4 m grid under
        the base) gives the ground height for the height reward, the fall termination and the swing-height term; it is
        NOT an observation, so the policy stays blind and deployable. Isaac Lab's ``terrain_levels_vel`` moves robots
        up / down the rows by the distance walked. Call AFTER :meth:`enable_gait_shaping` (it re-points that term)."""
        import isaaclab.sim as sim_utils
        import isaaclab.terrains as tg
        from isaaclab.managers import CurriculumTermCfg as CurrTerm
        from isaaclab.sensors import RayCasterCfg, patterns
        from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
        from isaaclab_tasks.manager_based.locomotion.velocity.mdp import terrain_levels_vel

        gen = TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20, horizontal_scale=0.1, vertical_scale=0.005,
            slope_threshold=0.75, use_cache=False, curriculum=True,
            sub_terrains={
                "flat": tg.MeshPlaneTerrainCfg(proportion=0.15),
                "rough": tg.HfRandomUniformTerrainCfg(proportion=0.25, noise_range=(0.01, 0.04), noise_step=0.01,
                                                      border_width=0.25),
                "slope_up": tg.HfPyramidSlopedTerrainCfg(proportion=0.15, slope_range=(0.0, slope_max),
                                                         platform_width=2.0, border_width=0.25),
                "slope_down": tg.HfInvertedPyramidSlopedTerrainCfg(proportion=0.15, slope_range=(0.0, slope_max),
                                                                   platform_width=2.0, border_width=0.25),
                "stairs_up": tg.MeshPyramidStairsTerrainCfg(proportion=0.15, step_height_range=(0.02, stair_max_m),
                                                            step_width=0.35, platform_width=3.0, border_width=1.0),
                "stairs_down": tg.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.15, step_height_range=(0.02, stair_max_m), step_width=0.35, platform_width=3.0,
                    border_width=1.0),
            },
        )
        old = self.scene.terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="generator", terrain_generator=gen,
            max_init_terrain_level=int(max_init_level), collision_group=-1,
            physics_material=old.physics_material if old.physics_material is not None else sim_utils.RigidBodyMaterialCfg(),
        )
        self.scene.ground_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/world", offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0)),
            ray_alignment="yaw", pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.4, 0.4]),
            mesh_prim_paths=["/World/ground"], debug_vis=False)
        self.rewards.anchor_height_l2.params["ground_sensor"] = "ground_scan"
        self.terminations.anchor_height.params["ground_sensor"] = "ground_scan"
        if getattr(self.rewards, "feet_swing_height", None) is not None:
            self.rewards.feet_swing_height.params["ground_sensor"] = "ground_scan"
        self.curriculum.terrain_levels = CurrTerm(func=terrain_levels_vel)
        self.terrain = "rough"

    def enable_thermal_penalty(self, weight: float = -2.0e-4) -> None:
        """Add ``motor_over_rated``: ``weight * sum relu(|tau| - rated)^2`` over the 22 motors, with the RATED
        (continuous) torques of the active ``hw_*`` profile. Call after :meth:`set_actuator_profile` (hw profiles only:
        the legacy groups have no rated torque)."""
        from isaaclab.managers import RewardTermCfg as RewTerm
        from isaaclab.managers import SceneEntityCfg

        from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES
        from dropbear_wbc.robots.hw_motor_specs import joint_hw_params

        from ... import mdp

        if self.actuator_profile not in HW_ACTUATOR_PROFILES:
            raise ValueError(f"thermal penalty needs an hw_* actuator profile, not {self.actuator_profile!r}")
        params = joint_hw_params(HW_ACTUATOR_PROFILES[self.actuator_profile])
        self.rewards.motor_over_rated = RewTerm(
            func=mdp.motor_torque_over_rated_l2,
            weight=float(weight),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(MOTOR_NAMES), preserve_order=True),
                    "rated": [float(params[m]["rated_torque"]) for m in MOTOR_NAMES]},
        )

    def enable_knee_stop_penalty(self, weight: float = -10.0, margin_deg: float = 3.0) -> None:
        """Add ``knee_stop``: ``weight`` * how deep [rad] each knee crank sits inside the last ``margin_deg`` before a
        hard stop (the tracking task's term, docs/ISSUES.md #19). The rough-terrain walkers rested the knee on its
        flexion stop in 91-97 % of stance with the motor nearly idle (docs/RESULTS.md, session 3): the printed stop
        carried the weight. Gait v3's all-motor ``joint_near_limit`` (-0.5, 5 deg) is about 20x too weak for that."""
        import math

        from isaaclab.managers import RewardTermCfg as RewTerm
        from isaaclab.managers import SceneEntityCfg

        from ... import mdp

        self.rewards.knee_stop = RewTerm(
            func=mdp.joint_near_limit_l1, weight=float(weight),
            params={"margin": math.radians(float(margin_deg)),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=["LL_knee_actuator_joint", "RL_knee_actuator_joint"],
                                                preserve_order=True)})

    def enable_gait_shaping(self, knee_flex: float = 0.5, swing_height: float = -20.0, torque_rate: float = -0.02,
                            mode_time: float = -2.0, air_standing: float = -0.5, knee_stance: float = 0.0,
                            feet_width: float = 0.0, near_limit: float = 0.0, hip_deviation: float | None = None,
                            target_flex: float = 0.15, clearance: float = 0.08, clamp_targets: bool = True,
                            clamp_margin_deg: float = 3.0) -> None:
        """Gait terms against the stiff-leg gait and for smooth torques (docs/archive/ACTUATORS_HISTORY.md section 12):

        * ``knee_flex_swing`` (+``knee_flex``): each swinging foot's knee crank flexed ``target_flex`` rad above the
          stand default (the H1 air-time reward alone is met by a straight leg swung from the hip);
        * ``feet_swing_height`` (``swing_height``, Unitree G1): swing sole ``clearance`` m above its standing height;
        * ``motor_torque_rate`` (``torque_rate``): ``sum ((tau_t - tau_{t-1}) / peak)^2`` between policy steps;
        * ``feet_mode_time`` (``mode_time``): a foot down > 1.0 s or up > 0.6 s while moving (no one-leg hopping);
        * ``feet_air_standing`` (``air_standing``): feet off the ground under a zero command;
        * ``knee_stance_flex`` (``knee_stance``, v2; 0 = off): stance knee flexed > 0.05 rad above the stand default
          (without it the policy walks crouched on the max-flexion stops);
        * ``feet_lateral`` (``feet_width``, v3; 0 = off): soles closer than 0.10 m sideways (no crossover gait);
        * ``joint_near_limit`` (``near_limit``, v3; 0 = off): motors within 5 deg of a hard limit;
        * ``hip_deviation`` (v3): overrides the ``joint_deviation_hip`` weight (hip roll/yaw centering);
        * ``clamp_targets``: :meth:`clamp_targets_to_limits` with ``clamp_margin_deg`` (3 deg for every run through
          2026-09-26; 0 from then on, docs/ISSUES.md #23).

        Same observations/actions, so a trained policy can be warm-started."""
        from isaaclab.managers import RewardTermCfg as RewTerm
        from isaaclab.managers import SceneEntityCfg

        from dropbear_wbc.robots.dropbear_names import FOOT_EE_BODIES, KNEE_MOTORS, MOTOR_NAMES

        from ... import mdp
        from ...velocity_env_cfg import FOOT_SENSOR_BODIES

        sensor = SceneEntityCfg("contact_forces", body_names=FOOT_SENSOR_BODIES, preserve_order=True)
        self.rewards.knee_flex_swing = RewTerm(
            func=mdp.knee_flexion_in_swing, weight=float(knee_flex),
            params={"command_name": "base_velocity", "sensor_cfg": sensor, "target_flex": float(target_flex),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=list(KNEE_MOTORS), preserve_order=True)})
        self.rewards.feet_swing_height = RewTerm(
            func=mdp.feet_swing_height_l2, weight=float(swing_height),
            params={"command_name": "base_velocity", "sensor_cfg": sensor, "stand_z": float(self.stand_info["sole_z"]),
                    "clearance": float(clearance),
                    "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_EE_BODIES), preserve_order=True)})
        self.rewards.motor_torque_rate = RewTerm(
            func=mdp.motor_torque_rate_l2, weight=float(torque_rate),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(MOTOR_NAMES), preserve_order=True)})
        self.rewards.feet_mode_time = RewTerm(
            func=mdp.feet_mode_time_exceeded, weight=float(mode_time),
            params={"command_name": "base_velocity", "sensor_cfg": sensor, "max_contact_s": 1.0, "max_air_s": 0.6})
        self.rewards.feet_air_standing = RewTerm(
            func=mdp.feet_air_when_standing, weight=float(air_standing),
            params={"command_name": "base_velocity", "sensor_cfg": sensor})
        if knee_stance:
            self.rewards.knee_stance_flex = RewTerm(
                func=mdp.knee_flexion_in_stance, weight=float(knee_stance),
                params={"sensor_cfg": sensor, "margin": 0.05,
                        "asset_cfg": SceneEntityCfg("robot", joint_names=list(KNEE_MOTORS), preserve_order=True)})
        if feet_width:
            self.rewards.feet_lateral = RewTerm(
                func=mdp.feet_lateral_distance_below, weight=float(feet_width),
                params={"min_dist": 0.10,
                        "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_EE_BODIES), preserve_order=True)})
        if near_limit:
            self.rewards.joint_near_limit = RewTerm(
                func=mdp.joint_near_limit_l1, weight=float(near_limit),
                params={"margin": 0.0873,
                        "asset_cfg": SceneEntityCfg("robot", joint_names=list(MOTOR_NAMES), preserve_order=True)})
        if hip_deviation is not None:
            self.rewards.joint_deviation_hip.weight = float(hip_deviation)
        if clamp_targets:
            self.clamp_targets_to_limits(clamp_margin_deg)

    def clamp_targets_to_limits(self, margin_deg: float = 3.0) -> None:
        """Clip the motor position targets (processed actions) to each motor's hard limits +- ``margin_deg``: a target
        far beyond a stop (the stiff-leg policies sent the right knee crank -300 deg) is just full torque into the stop;
        clamped, pressing a stop costs at most kp * margin (knee crank 500 * 0.052 = 26 N*m). A real deployment clamps
        p_des to the motor range too. The telemetry scan found the 3 deg walkers doing exactly that: the knee cranks
        held 3 deg past a stop, 26 N*m (0.77x rated) of continuous squeeze (docs/ISSUES.md #23), so new runs use 0."""
        import math

        from dropbear_wbc.robots.dropbear_names import MOTOR_HARD_LIMITS_DEG

        m = math.radians(margin_deg)
        self.actions.joint_pos.clip = {n: (math.radians(lo) - m, math.radians(hi) + m)
                                       for n, (lo, hi) in MOTOR_HARD_LIMITS_DEG.items()}

    def set_stand_npz(self, path: str | Path) -> None:
        """Use ``path`` as the reset state and derive the anchor height target / fall threshold from it."""
        s = summarize_stand_npz(path)
        self.stand_info = dict(s.__dict__)
        self.events.reset_robot.params["npz_path"] = s.path
        self.rewards.anchor_height_l2.params["target_height"] = s.anchor_z
        self.terminations.anchor_height.params["minimum_height"] = s.anchor_z - ANCHOR_FALL_DROP_M

    def set_solver_iterations(self, position: int, velocity: int) -> None:
        props = self.scene.robot.spawn.articulation_props
        props.solver_position_iteration_count = int(position)
        props.solver_velocity_iteration_count = int(velocity)

    def set_arm_freedom(self, action_scale: float | None = None, deviation_weight: float | None = None) -> None:
        """Let the arms take part in balance. The default walker holds them near the standing pose: a 0.1 rad action
        scale (legs 0.25-0.375) and ``joint_deviation_arms`` -0.2. ``action_scale`` [rad per unit action] sets all 10
        arm motors; ``deviation_weight`` (<= 0; 0 removes the term) sets the pull back to the standing pose. Evaluation
        must apply the same values (play_locomotion / robustness_bench read them from the run's run_info)."""
        from dropbear_wbc.robots.dropbear_names import ARM_MOTORS

        if action_scale is not None:
            scale = dict(self.actions.joint_pos.scale)
            for n in ARM_MOTORS:
                scale[n] = float(action_scale)
            self.actions.joint_pos.scale = scale
        if deviation_weight is not None:
            if float(deviation_weight) == 0.0:
                self.rewards.joint_deviation_arms = None
            else:
                self.rewards.joint_deviation_arms.weight = float(deviation_weight)

    def enable_arm_swing(self, weight: float = 0.5, k: float = 0.45, std: float = 0.12, abduction: float = 0.0) -> None:
        """Natural walking arms: ``arm_swing`` rewards each shoulder pitch following ``k`` x the opposite hip pitch while
        moving (``mdp.arm_swing_counter_phase``; k 0.40-0.55 in the retargeted human walking clips); ``abduction`` (< 0)
        adds an L1 pull of both shoulder-roll motors back to the standing pose, against the tightrope-walker arm the free-
        arm walkers learned (session 9). Use with :meth:`set_arm_freedom`."""
        from isaaclab.envs import mdp as il_mdp
        from isaaclab.managers import RewardTermCfg as RewTerm
        from isaaclab.managers import SceneEntityCfg

        from dropbear_wbc.tasks.locomotion.mdp.rewards import ARM_SWING_JOINTS, arm_swing_counter_phase

        self.rewards.arm_swing = RewTerm(
            func=arm_swing_counter_phase, weight=float(weight),
            params={"command_name": "base_velocity", "k": float(k), "std": float(std),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_SWING_JOINTS), preserve_order=True)})
        if abduction:
            self.rewards.arm_abduction = RewTerm(
                func=il_mdp.joint_deviation_l1, weight=float(abduction),
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LH_pitch", "RH_pitch"])})

    def enable_pushes(self, velocity_xy: float = 0.3, interval_s: tuple[float, float] = (8.0, 12.0)) -> None:
        from isaaclab.managers import EventTermCfg as EventTerm

        from ... import mdp

        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval_s,
            params={"velocity_range": {"x": (-velocity_xy, velocity_xy), "y": (-velocity_xy, velocity_xy)}},
        )


@configclass
class DropbearVelocityFlatPlayEnvCfg(DropbearVelocityFlatEnvCfg):
    """Play/eval config: 16 envs, solver 32/4, no observation noise / startup randomization / reset noise, fixed
    commands set by the caller (default: the curriculum limit ranges), long episodes, command arrows visible."""

    def __post_init__(self):
        super().__post_init__()
        self.set_num_envs(16)
        self.set_solver_iterations(32, 4)
        self.scene.env_spacing = 3.0
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.scale_limb_mass = None
        self.events.push_robot = None
        p = self.events.reset_robot.params
        p["frame_mode"] = "first"
        p["pose_range"] = {}
        p["leg_serial_noise"] = (0.0, 0.0)
        p["arm_serial_noise"] = (0.0, 0.0)
        self.curriculum.command_levels = None
        cmd = self.commands.base_velocity
        cmd.ranges = cmd.limit_ranges
        cmd.debug_vis = True
