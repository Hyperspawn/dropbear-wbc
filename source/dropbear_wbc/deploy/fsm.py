"""Unitree-deploy-style finite state machine: Passive -> MoveToDefault -> Hold -> Policy.

Pattern from unitree_rl_lab ``deploy/include/FSM`` (``State_Passive``,
``State_FixStand``, ``State_RLBase``/``State_Mimic``) and unitree_rl_gym
``deploy_real``: the controller turns the latest :class:`LowState` into a
:class:`LowCmd` once per control step. It has no transport or clock of its own,
so it runs unchanged against the Newton bridge, a recorded log or (later) the
ESP32 gateway.

States (all 22 motors, SDK slot order in the command):
    * ``PASSIVE``: ``kp = 0``, ``kd = passive_kd``, ``q* = q`` (damping only).
    * ``MOVE_TO_DEFAULT``: linear interpolation from the measured pose to the
      default pose over ``move_to_default_s`` with hold gains; then ``HOLD``.
    * ``HOLD``: default pose with hold gains.
    * ``POLICY``: ``q* = offset + scale * action`` on the policy joints (policy
      gains); other motors hold the default pose. Leaves to ``PASSIVE`` on bad
      orientation and to ``HOLD`` when a reference motion ends.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..sdk import motors
from ..sdk.types import LowCmd, LowState, MotorCmdBlock, MotorMode
from . import quat as Q
from .config import DeployConfig
from .motion import MotionReference
from .observations import ObsContext, ObservationBuilder
from .policy import OnnxPolicy


class FsmState(str, Enum):
    PASSIVE = "passive"
    MOVE_TO_DEFAULT = "move_to_default"
    HOLD = "hold"
    POLICY = "policy"


@dataclass
class StepInfo:
    """Diagnostics of one controller step."""

    state: FsmState
    q_des: np.ndarray
    transition: str | None = None
    policy_ms: float = 0.0
    obs: np.ndarray | None = None
    action: np.ndarray | None = None
    motion_frame: int | None = None
    extra: dict = field(default_factory=dict)


class DeployController:
    """Turns LowState into LowCmd following the FSM above.

    Args:
        cfg: deployment configuration.
        policy: ONNX policy (required to enter ``POLICY``).
        motion: reference motion for tracking policies (optional).
        allow_privileged: allow sim-only observation terms.
        velocity_command: (vx, vy, wz) for velocity policies.
    """

    def __init__(self, cfg: DeployConfig, policy: OnnxPolicy | None = None, motion: MotionReference | None = None,
                 allow_privileged: bool = False, velocity_command: np.ndarray | None = None):
        self.cfg, self.policy, self.motion = cfg, policy, motion
        self.allow_privileged = allow_privileged
        self.velocity_command = np.zeros(3) if velocity_command is None else np.asarray(velocity_command, float)
        self.obs_builder = ObservationBuilder(cfg.observations, allow_privileged) if policy is not None else None
        self.state = FsmState.PASSIVE
        self._t_enter = 0.0
        self._q_start = np.zeros(motors.NUM_MOTORS)
        self._last_action = np.zeros(cfg.num_actions)
        self._policy_step = 0
        self.full_kp = cfg.fsm.hold_kp.copy()
        self.full_kd = cfg.fsm.hold_kd.copy()
        self.full_kp[cfg.motor_ids] = cfg.kp
        self.full_kd[cfg.motor_ids] = cfg.kd
        self._pending: str | None = None

    # ------------------------------------------------------------------ transitions

    def request(self, new: FsmState, low_state: LowState, t: float) -> None:
        """Enter ``new`` (runs its ``enter`` logic)."""
        if new == FsmState.POLICY and self.policy is None:
            raise RuntimeError("POLICY requested but no ONNX policy is loaded")
        self._pending = f"{self.state.value}->{new.value}"
        self.state = new
        self._t_enter = t
        q = low_state.motor.q.astype(np.float64)
        if new == FsmState.MOVE_TO_DEFAULT:
            self._q_start = q.copy()
        elif new == FsmState.POLICY:
            self._last_action = np.zeros(self.cfg.num_actions)
            self._policy_step = 0
            self.obs_builder.reset()
            if self.motion is not None:
                ctx = self._ctx(low_state)
                pos = ctx.robot_anchor_pos_w() if (self.allow_privileged and low_state.sim is not None) else None
                self.motion.align(ctx.robot_anchor_quat_w(), pos)

    def _ctx(self, low_state: LowState) -> ObsContext:
        ctx = ObsContext(state=low_state, cfg=self.cfg, last_action=self._last_action,
                         velocity_command=self.velocity_command, allow_privileged=self.allow_privileged)
        if self.motion is not None:
            ctx.anchor_offset_pos = self.motion.anchor_offset_pos
            ctx.anchor_offset_quat = self.motion.anchor_offset_quat
            ctx.reference = self.motion.sample(self._policy_step)
            step = self._policy_step
            ctx.reference_ahead = lambda k: self.motion.sample(step + k)  # frame_index clamps at the clip end
        return ctx

    # ------------------------------------------------------------------ step

    def step(self, low_state: LowState, t: float) -> tuple[LowCmd, StepInfo]:
        """Compute the command for controller time ``t`` [s]."""
        n = motors.NUM_MOTORS
        cfg = self.cfg
        q_meas = low_state.motor.q.astype(np.float64)
        kp, kd = self.full_kp.copy(), self.full_kd.copy()
        info = StepInfo(state=self.state, q_des=np.zeros(n))
        if self.state == FsmState.PASSIVE:
            q_des, kp, kd = q_meas.copy(), np.zeros(n), cfg.fsm.passive_kd.copy()
        elif self.state == FsmState.MOVE_TO_DEFAULT:
            ratio = float(np.clip((t - self._t_enter) / max(cfg.fsm.move_to_default_s, 1e-6), 0.0, 1.0))
            q_des = self._q_start + ratio * (cfg.default_pose_sdk - self._q_start)
            kp, kd = cfg.fsm.hold_kp.copy(), cfg.fsm.hold_kd.copy()
            info.extra["ratio"] = ratio
            if ratio >= 1.0:
                self.state = FsmState.HOLD
                self._pending = "move_to_default->hold"
        elif self.state == FsmState.HOLD:
            q_des = cfg.default_pose_sdk.copy()
            kp, kd = cfg.fsm.hold_kp.copy(), cfg.fsm.hold_kd.copy()
        else:  # POLICY
            q_des = self._policy_step_q(low_state, info)
            tilt = Q.tilt_angle(low_state.imu.quat_wxyz.astype(np.float64))
            info.extra["tilt_rad"] = tilt
            if cfg.fsm.bad_orientation_rad is not None and tilt > cfg.fsm.bad_orientation_rad:
                self.state, self._pending = FsmState.PASSIVE, "policy->passive(bad_orientation)"
            elif self.motion is not None and self.motion.done(self._policy_step):
                self.state, self._pending = FsmState.HOLD, "policy->hold(motion_end)"
        cmd = LowCmd(tick=low_state.tick)
        cmd.motor = MotorCmdBlock.from_arrays(n, mode=MotorMode.ENABLE, q=q_des, dq=0.0, tau=0.0, kp=kp, kd=kd)
        info.q_des = q_des
        info.transition, self._pending = self._pending, None
        return cmd, info

    def _policy_step_q(self, low_state: LowState, info: StepInfo) -> np.ndarray:
        cfg = self.cfg
        t0 = time.perf_counter()
        poll = getattr(self.motion, "poll", None)  # live reference (deploy.motion.LiveMotion): splice new clips first
        if poll is not None:
            for ev in poll(self._policy_step):
                info.extra.setdefault("live", []).append(ev)
        ctx = self._ctx(low_state)
        obs = self.obs_builder.compute(ctx)
        frame = self.motion.frame_index(self._policy_step) if self.motion is not None else self._policy_step
        action = self.policy.act(obs, time_step=frame)
        if action.shape != (cfg.num_actions,):
            raise ValueError(f"policy returned {action.shape}, expected ({cfg.num_actions},)")
        if cfg.action_clip is not None:
            action = np.clip(action, -cfg.action_clip, cfg.action_clip)
        q_des = cfg.default_pose_sdk.copy()
        q_pol = cfg.action_offset + cfg.action_scale * action
        if cfg.target_clip is not None:  # the training-time target clamp (sidecar ``target_clip``)
            q_pol = np.clip(q_pol, cfg.target_clip[:, 0], cfg.target_clip[:, 1])
        q_des[cfg.motor_ids] = q_pol
        self._last_action = action
        info.policy_ms = 1e3 * (time.perf_counter() - t0)
        info.obs, info.action, info.motion_frame = obs, action, frame
        self._policy_step += 1
        return q_des
