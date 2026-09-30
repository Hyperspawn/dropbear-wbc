"""Observation terms computed from ``LowState`` (numpy), mirroring Isaac Lab / BeyondMimic names.

Each term is ``f(ctx, params) -> 1-D float array``. :class:`ObservationBuilder`
concatenates terms in configuration order, applying ``clip`` then ``scale`` and
stacking ``history_length`` frames oldest-first (Isaac Lab's CircularBuffer,
filled with the first frame after a reset).

Frames:
    * Body frame = root body ``world`` (the IMU body). ``base_ang_vel`` is the IMU
      gyro; ``projected_gravity`` is ``R^T [0, 0, -1]``.
    * Joint terms are in policy joint order (``DeployConfig.joint_names``).
    * Tracking terms follow BeyondMimic: ``motion_anchor_pos_b`` /
      ``motion_anchor_ori_b`` express the reference anchor pose in the robot's
      anchor frame; ``motion_anchor_ori_b`` is the first two columns of the
      rotation matrix, flattened row-major (6 values).

Privileged terms (need ``LowState.sim``; not observable on hardware):
``base_lin_vel``, ``motion_anchor_pos_b``. The runner refuses them unless
explicitly allowed, and reports that the policy is sim-only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..sdk.types import LowState
from . import quat as Q
from .config import DeployConfig, ObsTermCfg

PRIVILEGED_TERMS = frozenset({"base_lin_vel", "motion_anchor_pos_b"})


class PrivilegedObservationError(RuntimeError):
    """A privileged (sim-only) term was requested without permission or without a sim block."""


@dataclass
class ReferenceFrame:
    """Reference motion sample in the (aligned) world frame, policy joint order."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    anchor_pos_w: np.ndarray
    anchor_quat_w: np.ndarray


@dataclass
class ObsContext:
    """Inputs available to observation terms at one policy step.

    Attributes:
        state: latest LowState.
        cfg: deployment config.
        last_action: previous raw policy output (policy order).
        velocity_command: (vx, vy, wz) command for velocity policies [m/s, m/s, rad/s].
        reference: current reference frame for tracking policies (or ``None``).
        anchor_offset_pos, anchor_offset_quat: rigid anchor pose in the root body frame.
        allow_privileged: permit sim-only terms.
    """

    state: LowState
    cfg: DeployConfig
    last_action: np.ndarray
    velocity_command: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference: ReferenceFrame | None = None
    anchor_offset_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    anchor_offset_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    allow_privileged: bool = False
    reference_ahead: Callable[[int], ReferenceFrame] | None = None
    """``k -> `` reference ``k`` policy steps after the current one (clamped to the clip end), for commands with
    future frames (``motion_command`` ``params.future_steps``; motion-library ``-Future`` policies, CONTRACTS 5.3)."""

    # ---- derived robot quantities
    def root_quat(self) -> np.ndarray:
        return Q.normalize(self.state.imu.quat_wxyz.astype(np.float64))

    def joint_pos(self) -> np.ndarray:
        return self.state.motor.q.astype(np.float64)[self.cfg.motor_ids]

    def joint_vel(self) -> np.ndarray:
        return self.state.motor.dq.astype(np.float64)[self.cfg.motor_ids]

    def robot_anchor_quat_w(self) -> np.ndarray:
        return Q.mul(self.root_quat(), self.anchor_offset_quat)

    def robot_anchor_pos_w(self) -> np.ndarray:
        sim = self._sim("robot anchor position")
        return sim.root_pos_w.astype(np.float64) + Q.rotate(self.root_quat(), self.anchor_offset_pos)

    def _sim(self, what: str):
        if not self.allow_privileged:
            raise PrivilegedObservationError(f"{what} needs privileged sim state; pass --allow-privileged")
        if self.state.sim is None:
            raise PrivilegedObservationError(f"{what} needs LowState.sim, which this robot side does not publish")
        return self.state.sim

    def ref(self) -> ReferenceFrame:
        if self.reference is None:
            raise RuntimeError("tracking observation requested but no reference motion is loaded")
        return self.reference

    def ref_ahead(self, k: int) -> ReferenceFrame:
        if self.reference_ahead is None:
            raise RuntimeError("observation needs future reference frames but the controller provides none")
        return self.reference_ahead(int(k))


TermFn = Callable[[ObsContext, dict], np.ndarray]
TERMS: dict[str, TermFn] = {}


def term(*names: str):
    def deco(fn: TermFn) -> TermFn:
        for n in names:
            TERMS[n] = fn
        return fn
    return deco


@term("base_ang_vel")
def base_ang_vel(ctx: ObsContext, params: dict) -> np.ndarray:
    return ctx.state.imu.gyro.astype(np.float64)


@term("projected_gravity")
def projected_gravity(ctx: ObsContext, params: dict) -> np.ndarray:
    return Q.rotate_inv(ctx.root_quat(), np.array([0.0, 0.0, -1.0]))


@term("base_lin_vel")
def base_lin_vel(ctx: ObsContext, params: dict) -> np.ndarray:
    sim = ctx._sim("base_lin_vel")
    return Q.rotate_inv(ctx.root_quat(), sim.root_lin_vel_w.astype(np.float64))


@term("joint_pos_rel")
def joint_pos_rel(ctx: ObsContext, params: dict) -> np.ndarray:
    return ctx.joint_pos() - ctx.cfg.default_joint_pos


@term("joint_pos")
def joint_pos_abs(ctx: ObsContext, params: dict) -> np.ndarray:
    return ctx.joint_pos()


@term("joint_vel_rel", "joint_vel")
def joint_vel_rel(ctx: ObsContext, params: dict) -> np.ndarray:
    # Default joint velocity is zero, so rel == abs.
    return ctx.joint_vel()


@term("last_action", "actions")
def last_action(ctx: ObsContext, params: dict) -> np.ndarray:
    return ctx.last_action.astype(np.float64)


@term("velocity_commands")
def velocity_commands(ctx: ObsContext, params: dict) -> np.ndarray:
    return ctx.velocity_command.astype(np.float64)


@term("motion_command")
def motion_command(ctx: ObsContext, params: dict) -> np.ndarray:
    """Reference motor pos + vel; with ``params.future_steps`` (k1, k2, ...) also pos + vel at steps t+k1, t+k2, ...
    (the motion-library ``-Future`` command layout, CONTRACTS 5.3)."""
    r = ctx.ref()
    parts = [r.joint_pos, r.joint_vel]
    for k in params.get("future_steps") or ():
        f = ctx.ref_ahead(int(k))
        parts += [f.joint_pos, f.joint_vel]
    return np.concatenate(parts)


@term("motion_joint_pos")
def motion_joint_pos(ctx: ObsContext, params: dict) -> np.ndarray:
    return ctx.ref().joint_pos


@term("motion_joint_vel")
def motion_joint_vel(ctx: ObsContext, params: dict) -> np.ndarray:
    return ctx.ref().joint_vel


@term("generated_commands")
def generated_commands(ctx: ObsContext, params: dict) -> np.ndarray:
    name = params.get("command_name", "motion")
    if name == "motion":
        return motion_command(ctx, params)
    if name in ("base_velocity", "velocity"):
        return velocity_commands(ctx, params)
    raise KeyError(f"unknown command_name {name!r}")


@term("motion_anchor_pos_b")
def motion_anchor_pos_b(ctx: ObsContext, params: dict) -> np.ndarray:
    r = ctx.ref()
    pos, _ = Q.subtract_frame_transforms(ctx.robot_anchor_pos_w(), ctx.robot_anchor_quat_w(),
                                         r.anchor_pos_w, r.anchor_quat_w)
    return pos


@term("motion_anchor_ori_b")
def motion_anchor_ori_b(ctx: ObsContext, params: dict) -> np.ndarray:
    r = ctx.ref()
    rel = Q.mul(Q.conj(ctx.robot_anchor_quat_w()), r.anchor_quat_w)
    return Q.matrix(rel)[:, :2].reshape(-1)


class ObservationBuilder:
    """Concatenates configured terms into the policy input vector."""

    def __init__(self, terms: list[ObsTermCfg], allow_privileged: bool = False):
        unknown = [t.func for t in terms if t.func not in TERMS]
        if unknown:
            raise KeyError(f"unknown observation functions {unknown}; known: {sorted(TERMS)}")
        privileged = [t.func for t in terms if t.func in PRIVILEGED_TERMS]
        if privileged and not allow_privileged:
            raise PrivilegedObservationError(
                f"policy observes sim-only terms {privileged}; rerun with --allow-privileged (sim2sim only)")
        self.terms = terms
        self.privileged_terms = privileged
        self._hist: list[deque] = [deque(maxlen=max(1, t.history_length)) for t in terms]
        self._dims: list[int | None] = [t.dim for t in terms]

    def reset(self) -> None:
        for h in self._hist:
            h.clear()

    def compute(self, ctx: ObsContext) -> np.ndarray:
        parts = []
        for k, (t, hist) in enumerate(zip(self.terms, self._hist)):
            v = np.asarray(TERMS[t.func](ctx, t.params), dtype=np.float64).reshape(-1)
            if self._dims[k] is not None and v.size != self._dims[k]:
                raise ValueError(f"observation {t.name} ({t.func}) has {v.size} values, config says {self._dims[k]}")
            self._dims[k] = v.size
            if t.clip is not None:
                v = np.clip(v, t.clip[0], t.clip[1])
            scale = np.asarray(t.scale, dtype=np.float64).reshape(-1)
            if scale.size not in (1, v.size):
                raise ValueError(f"observation {t.name}: scale has {scale.size} entries for {v.size} values")
            v = v * scale
            if not hist:
                for _ in range(hist.maxlen):
                    hist.append(v)
            else:
                hist.append(v)
            parts.extend(hist)
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)

    def layout(self) -> list[dict]:
        """Per-term (name, func, dim, history) after the first :meth:`compute`."""
        return [{"name": t.name, "func": t.func, "dim": d, "history_length": t.history_length}
                for t, d in zip(self.terms, self._dims)]
