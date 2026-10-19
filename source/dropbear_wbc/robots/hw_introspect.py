"""Read the explicit (``hw_*``) motor models off an articulation without importing Isaac Lab.

The exports (``tasks/*/export.py``) use these; they also run on CPU with Isaac stubs and a mocked robot that has no
``actuators`` (``scripts/emulate_tracking_export.py``), which reads as "no explicit models".
"""
from __future__ import annotations


def actuator_effort_limit(act):
    """Actuator-model effort limit tensor (num_envs, num_joints) on Isaac Lab 2.2 (``effort_limit``) and 3.0
    (``actuator_effort_limit``; ``effort_limit`` there is a DeprecationWarning alias removed in 3.1)."""
    lim = getattr(act, "actuator_effort_limit", None)
    return act.effort_limit if lim is None else lim


def actuator_velocity_limit(act):
    """Actuator-model velocity limit tensor; see :func:`actuator_effort_limit`."""
    lim = getattr(act, "actuator_velocity_limit", None)
    return act.velocity_limit if lim is None else lim


def _explicit_actuators(robot) -> list:
    return [a for a in (getattr(robot, "actuators", None) or {}).values() if not a.is_implicit_model]


def overlay_explicit_gains(robot, motor_names: list[str], kp: list, kd: list, effort: list) -> None:
    """In place: for motors driven by an explicit model (``hw_*``), replace the solver-side values read from
    ``robot.data`` (kp = kd = 0, effort = the huge safety limit) by the model's own kp, kd and peak torque."""
    for act in _explicit_actuators(robot):
        for k, name in enumerate(act.joint_names):
            if name in motor_names:
                i = motor_names.index(name)
                kp[i] = float(act.stiffness[0, k])
                kd[i] = float(act.damping[0, k])
                effort[i] = float(actuator_effort_limit(act)[0, k])


def target_interp_steps(robot) -> int:
    """The position-target interpolation (physics steps) of ``robot``'s explicit motor models (0 if none): the deploy
    runtime / motor firmware must ramp each new 50 Hz target over this many control ticks (docs/ISSUES.md #11)."""
    vals = {int(getattr(a.cfg, "target_interp_steps", 0) or 0) for a in _explicit_actuators(robot)}
    return max(vals) if vals else 0
