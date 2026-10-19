"""Isaac Lab actuator model for the real-actuator ("hw") profiles of :mod:`dropbear_wbc.robots.hw_motor_specs`.

:class:`DatasheetMotor` is an explicit PD actuator (the PD runs at the physics rate, like the motor's own motion-mode
loop) with

* a per-joint DC-motor torque-speed envelope (Isaac Lab ``DCMotor`` law, but with per-joint saturation effort),
* Coulomb + viscous gearbox friction subtracted after the envelope (randomized per episode by a scale),
* a per-episode random delay of the position/velocity/effort targets (``min_delay..max_delay`` physics steps),
* optionally an explicit end-stop spring past each joint's mechanical limit (``hw_motor_specs.STOP_MODEL``), whose torque
  ``stop_effort`` is what the stop parts carry,
* optionally a parallel elastic (an elastic band beside the motor, like a tendon): ``-k * max(0, q - engage)``, pushing
  the joint back once it is past ``engage``; its torque ``spring_effort`` is not motor torque.

``applied_effort`` is the MOTOR torque (after the envelope, before friction), so ``robot.data.applied_torque`` shows
what the motor delivers; ``friction_effort`` holds the friction torque of the last step.

Import only after the Isaac Sim app is running.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions

from .dropbear_names import ACTUATOR_GROUP_MOTORS, ACTUATOR_PARAMS
from .hw_motor_specs import (
    FRICTION_SCALE_RANGE,
    LATENCY_STEPS,
    VELOCITY_LIMIT_SIM_FACTOR,
    joint_hw_params,
)


class DatasheetMotor(IdealPDActuator):
    """Explicit PD + per-joint torque-speed envelope + friction + command delay (see the module docstring)."""

    cfg: DatasheetMotorCfg

    def __init__(self, cfg: DatasheetMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._saturation = self._parse_joint_parameter(cfg.saturation_effort, None)
        self._coulomb = self._parse_joint_parameter(cfg.coulomb_friction, 0.0)
        self._viscous = self._parse_joint_parameter(cfg.viscous_friction_coef, 0.0)
        if torch.any(self._saturation < self.effort_limit):
            raise ValueError(f"{self.joint_names}: saturation_effort must be >= effort_limit (peak)")
        # velocity where the torque-speed line meets -effort_limit (DCMotor's clip for back-driven joints)
        self._vel_at_effort_lim = self.velocity_limit * (1.0 + self.effort_limit / self._saturation)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self.friction_effort = torch.zeros_like(self.computed_effort)
        self._friction_scale = torch.ones(self._num_envs, 1, device=self._device)
        self._pos_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._vel_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._eff_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # optional linear interpolation of a NEW position target over ``target_interp_steps`` physics steps (what the
        # motor-side firmware can do between 50 Hz policy targets; docs/ISSUES.md #11)
        self._interp_n = int(cfg.target_interp_steps)
        self._last_tgt = None
        self._snap = torch.ones(self._num_envs, 1, dtype=torch.bool, device=self._device)
        self.stop_effort = torch.zeros_like(self.computed_effort)
        self.spring_effort = torch.zeros_like(self.computed_effort)
        self._springs = cfg.spring_stiffness is not None
        if self._springs:
            self._spring_k = self._parse_joint_parameter(cfg.spring_stiffness, 0.0)
            self._spring_q0 = self._parse_joint_parameter(cfg.spring_engage, 0.0)
        self._stops = cfg.stop_stiffness is not None
        if self._stops:
            self._stop_lo = self._parse_joint_parameter(cfg.stop_lower, None)
            self._stop_hi = self._parse_joint_parameter(cfg.stop_upper, None)
            self._stop_k = self._parse_joint_parameter(cfg.stop_stiffness, None)
            self._stop_d = self._parse_joint_parameter(cfg.stop_damping, 0.0)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            env_ids = torch.arange(self._num_envs, device=self._device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        n = len(env_ids)
        lo, hi = self.cfg.friction_scale_range
        self._friction_scale[env_ids, 0] = lo + (hi - lo) * torch.rand(n, device=self._device)
        lags = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (n,), dtype=torch.int, device=self._device)
        for buf in (self._pos_delay, self._vel_delay, self._eff_delay):
            buf.set_time_lag(lags, env_ids)
            buf.reset(env_ids)
        self._snap[env_ids] = True  # a reset env jumps straight to its first target (no ramp from the old pose)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        if self.cfg.max_delay > 0:  # a zero-lag DelayBuffer is the identity (skipped: ~2.7 ms per 50 Hz step on CPU)
            control_action.joint_positions = self._pos_delay.compute(control_action.joint_positions)
            control_action.joint_velocities = self._vel_delay.compute(control_action.joint_velocities)
            control_action.joint_efforts = self._eff_delay.compute(control_action.joint_efforts)
        if self._interp_n > 1:
            control_action.joint_positions = self._interpolate(control_action.joint_positions)
        self._joint_vel[:] = joint_vel
        out = super().compute(control_action, joint_pos, joint_vel)
        self.friction_effort[:] = self._friction_scale * (
            self._coulomb * torch.tanh(joint_vel / self.cfg.friction_vel_eps) + self._viscous * joint_vel
        )
        out.joint_efforts = self.applied_effort - self.friction_effort
        if self._stops:
            below = torch.clamp(self._stop_lo - joint_pos, min=0.0)
            above = torch.clamp(joint_pos - self._stop_hi, min=0.0)
            touching = (below > 0) | (above > 0)
            self.stop_effort[:] = self._stop_k * (below - above) - torch.where(touching, self._stop_d * joint_vel, 0.0)
            out.joint_efforts = out.joint_efforts + self.stop_effort
        if self._springs:
            self.spring_effort[:] = -self._spring_k * torch.clamp(joint_pos - self._spring_q0, min=0.0)
            out.joint_efforts = out.joint_efforts + self.spring_effort
        return out

    def _interpolate(self, tgt: torch.Tensor) -> torch.Tensor:
        if self._last_tgt is None:
            self._last_tgt, self._from, self._to = tgt.clone(), tgt.clone(), tgt.clone()
            self._cmd = tgt.clone()
            self._k = torch.full((tgt.shape[0], 1), float(self._interp_n), device=tgt.device)
        new = (tgt != self._last_tgt).any(dim=1, keepdim=True)
        self._cmd = torch.where(self._snap, tgt, self._cmd)
        self._snap[:] = False
        self._from = torch.where(new, self._cmd, self._from)
        self._to = torch.where(new, tgt, self._to)
        self._k = torch.where(new, torch.zeros_like(self._k), self._k)
        self._k = torch.clamp(self._k + 1.0, max=float(self._interp_n))
        self._cmd = self._from + (self._k / self._interp_n) * (self._to - self._from)
        self._last_tgt = tgt.clone()
        return self._cmd

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        vel = torch.clip(self._joint_vel, min=-self._vel_at_effort_lim, max=self._vel_at_effort_lim)
        top = torch.clip(self._saturation * (1.0 - vel / self.velocity_limit), max=self.effort_limit)
        bottom = torch.clip(self._saturation * (-1.0 - vel / self.velocity_limit), min=-self.effort_limit)
        return torch.clip(effort, min=bottom, max=top)


@configclass
class DatasheetMotorCfg(IdealPDActuatorCfg):
    """Config of :class:`DatasheetMotor`. ``effort_limit`` = peak torque, ``velocity_limit`` = no-load speed."""

    class_type: type = DatasheetMotor
    saturation_effort: dict[str, float] | float | None = None
    """Torque-speed line intercept at zero speed [N*m] (>= ``effort_limit``)."""
    coulomb_friction: dict[str, float] | float = 0.0
    """Output-side Coulomb friction [N*m]."""
    viscous_friction_coef: dict[str, float] | float = 0.0
    """Output-side viscous friction [N*m*s/rad]."""
    friction_vel_eps: float = 0.05
    """tanh smoothing speed of the Coulomb term [rad/s]."""
    friction_scale_range: tuple[float, float] = (1.0, 1.0)
    """Per-episode uniform scale on both friction terms."""
    min_delay: int = 0
    max_delay: int = 0
    """Target delay range in physics steps (per episode, uniform integer)."""
    target_interp_steps: int = 0
    """> 1: a new position target is reached linearly over this many physics steps (0/1 = step change)."""
    stop_lower: dict[str, float] | float | None = None
    stop_upper: dict[str, float] | float | None = None
    """Mechanical joint limits [rad] for the explicit stop model (with ``stop_stiffness``)."""
    stop_stiffness: dict[str, float] | float | None = None
    """End-stop spring [N*m/rad]; None = off (PhysX's rigid joint limit acts, its torque is not observable)."""
    stop_damping: dict[str, float] | float = 0.0
    """End-stop damping [N*m*s/rad], only while the joint is past a stop."""
    spring_stiffness: dict[str, float] | float | None = None
    """Parallel elastic stiffness [N*m/rad] (e.g. a knee band); None = off."""
    spring_engage: dict[str, float] | float = 0.0
    """Joint angle [rad] where the elastic starts to pull (slack below it)."""


def make_hw_profile_groups(profile: str, stop_model: bool = False,
                           knee_band: tuple[float, float] | None = None) -> dict:
    """Motor groups of an ``hw_*`` actuator profile: its motor map plus its options (``HW_PROFILE_OPTIONS``).
    ``knee_band``: (stiffness N*m/rad, engage deg) of an elastic on each knee crank (see ``make_hw_motor_groups``).
    Armature-DR profiles (``*_armdr``) mix DatasheetMotor + ImplicitActuator on the notes/13 T3 joints."""
    from .hw_motor_specs import HW_PROFILE_MAPS, HW_PROFILE_OPTIONS

    return make_hw_motor_groups(HW_PROFILE_MAPS[profile], stop_model=stop_model, knee_band=knee_band,
                                **HW_PROFILE_OPTIONS.get(profile, {}))


def make_hw_motor_groups(map_name: str = "user_2026_09_25", *, randomize: bool = True,
                         target_interp_steps: int = 0,
                         gain_overrides: dict[str, tuple[float, float]] | None = None,
                         stop_model: bool = False,
                         knee_band: tuple[float, float] | None = None,
                         armature_dr: bool = False) -> dict:
    """One :class:`DatasheetMotorCfg` group per motor model of ``map_name`` (group name ``hw_<model>``).
    ``gain_overrides``: motor name -> (kp, kd) replacing the role gains (checked against the motion-mode limits).
    ``armature_dr``: put :data:`ARMATURE_DR_IMPLICIT_JOINTS` on ImplicitActuator (stable at the alt armature under
    explicit 5 ms PD) and leave the rest on DatasheetMotor; callers must also install the armature DR EventTerm
    (``DropbearFlatEnvCfg.enable_armature_dr``). The explicit stop model and the knee band act on the DatasheetMotor
    groups only: with ``armature_dr`` the implicit joints (calves, wrist roll) keep PhysX's rigid limits."""
    from .hw_motor_specs import MOTION_MODE_LIMITS, STOP_MODEL, armature_dr_group_plan, motor_stop_limits

    params = joint_hw_params(map_name)
    for name, (kp, kd) in (gain_overrides or {}).items():
        if name not in params:
            raise ValueError(f"gain override for unknown motor {name!r}")
        if kp > MOTION_MODE_LIMITS["kp_max"] or kd > MOTION_MODE_LIMITS["kd_max"]:
            raise ValueError(f"{name}: gains {kp}/{kd} outside the motion-mode range {MOTION_MODE_LIMITS}")
        params[name] = {**params[name], "kp": float(kp), "kd": float(kd)}

    plan = armature_dr_group_plan(map_name, gain_overrides=gain_overrides) if armature_dr else None
    by_model: dict[str, list[str]] = {}
    if plan is None:
        for joint, p in params.items():
            by_model.setdefault(p["model"], []).append(joint)
    else:
        by_model = {m: list(js) for m, js in plan["explicit"].items()}

    groups: dict = {}
    for model, joints in by_model.items():
        key = "hw_" + model.lower().replace("-", "_").replace(":", "_").replace(".", "_")

        def per_joint(field: str, scale: float = 1.0, _joints=joints) -> dict[str, float]:
            return {j: float(params[j][field]) * scale for j in _joints}

        groups[key] = DatasheetMotorCfg(
            joint_names_expr=list(joints),
            effort_limit=per_joint("peak_torque"),
            velocity_limit=per_joint("no_load_speed"),
            velocity_limit_sim=per_joint("no_load_speed", VELOCITY_LIMIT_SIM_FACTOR),
            saturation_effort=per_joint("saturation_effort"),
            stiffness=per_joint("kp"),
            damping=per_joint("kd"),
            armature=per_joint("armature"),
            friction=0.0,
            coulomb_friction=per_joint("coulomb_friction"),
            viscous_friction_coef=per_joint("viscous_friction"),
            friction_scale_range=FRICTION_SCALE_RANGE if randomize else (1.0, 1.0),
            min_delay=LATENCY_STEPS[0] if randomize else 0,
            max_delay=LATENCY_STEPS[1] if randomize else 0,
            target_interp_steps=int(target_interp_steps),
        )
        if stop_model:
            import math

            lim = motor_stop_limits()
            stopped = [j for j in joints if math.degrees(lim[j][1] - lim[j][0]) < 350.0]  # rolls without a stop: off
            k, d = {}, {}
            for j in joints:
                arm = float(params[j]["armature"])
                if j not in stopped:
                    k[j], d[j] = 0.0, 0.0
                elif "knee" in j:
                    k[j], d[j] = STOP_MODEL["knee_stiffness"], STOP_MODEL["knee_damping"]
                else:
                    k[j] = STOP_MODEL["omega"] ** 2 * arm
                    d[j] = 2.0 * STOP_MODEL["zeta"] * math.sqrt(k[j] * arm)
            groups[key].stop_lower = {j: lim[j][0] for j in joints}
            groups[key].stop_upper = {j: lim[j][1] for j in joints}
            groups[key].stop_stiffness = k
            groups[key].stop_damping = d
        if knee_band is not None:
            import math

            knees = [j for j in joints if "knee_actuator" in j]
            if knees:  # an elastic beside each knee motor, pulling the crank back toward a straight knee
                groups[key].spring_stiffness = {j: (float(knee_band[0]) if j in knees else 0.0) for j in joints}
                groups[key].spring_engage = {j: (math.radians(float(knee_band[1])) if j in knees else 0.0) for j in joints}
    if plan is not None:
        # one ImplicitActuator group for the T3-unstable joints (calves + wrist roll)
        joints = list(plan["implicit"])
        groups["hw_armdr_implicit"] = ImplicitActuatorCfg(
            joint_names_expr=joints,
            effort_limit_sim={j: plan["implicit"][j]["effort"] for j in joints},
            stiffness={j: plan["implicit"][j]["kp"] for j in joints},
            damping={j: plan["implicit"][j]["kd"] for j in joints},
            armature={j: plan["implicit"][j]["armature"] for j in joints},
        )
    return groups


def legacy_motor_groups(armature_map: str | None = None) -> dict[str, ImplicitActuatorCfg]:
    """The legacy implicit motor groups (``arms``/``hips``/``knees``/``ankles``) exactly as ``make_dropbear_cfg``.
    ``armature_map``: a motor map name; each motor then gets that map's datasheet armature instead of the legacy 0.01
    (``hw_motor_specs.LEGACY_HW_ARMATURE_PROFILES``, notes/13 T1). Gains and effort limits stay legacy."""
    from .hw_motor_specs import legacy_group_armature

    per_motor = legacy_group_armature(armature_map) if armature_map else {}
    groups: dict[str, ImplicitActuatorCfg] = {}
    for group, motors in ACTUATOR_GROUP_MOTORS.items():
        effort, kp, kd, armature = ACTUATOR_PARAMS[group]
        groups[group] = ImplicitActuatorCfg(
            joint_names_expr=list(motors), effort_limit_sim=effort, stiffness=kp, damping=kd,
            armature=dict(per_motor[group]) if armature_map else armature,
        )
    return groups


# kept importable from here for older callers; the Isaac-free versions live in hw_introspect
from .hw_introspect import overlay_explicit_gains, target_interp_steps  # noqa: E402,F401


def set_motor_groups(actuators: dict, motor_groups: dict) -> None:
    """Replace every body-motor group of ``actuators`` (legacy or ``hw_*``) by ``motor_groups``, in place.
    Neck and passive groups are kept."""
    for key in [k for k in actuators if k in ACTUATOR_GROUP_MOTORS or k.startswith("hw_")]:
        del actuators[key]
    actuators.update(motor_groups)
