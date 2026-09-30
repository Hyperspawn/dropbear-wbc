"""Read the explicit (``hw_*``) motor models off an articulation without importing Isaac Lab.

The exports (``tasks/*/export.py``) use these; they also run on CPU with Isaac stubs and a mocked robot that has no
``actuators`` (``scripts/emulate_tracking_export.py``), which reads as "no explicit models".
"""
from __future__ import annotations


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
                effort[i] = float(act.effort_limit[0, k])


def target_interp_steps(robot) -> int:
    """The position-target interpolation (physics steps) of ``robot``'s explicit motor models (0 if none): the deploy
    runtime / motor firmware must ramp each new 50 Hz target over this many control ticks (docs/ISSUES.md #11)."""
    vals = {int(getattr(a.cfg, "target_interp_steps", 0) or 0) for a in _explicit_actuators(robot)}
    return max(vals) if vals else 0
