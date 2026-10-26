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

IMU health gate (local patch 0004, item 137): ``POLICY`` is refused, and left for
``PASSIVE`` in the same step (the policy is not evaluated on the bad sample), when
``LowState.imu`` is invalid (``valid != 1``), stale (``age_us`` plus the time the
same LowState tick has been re-used exceeds ``fsm.imu_max_age_s``) or, if enabled,
frozen (``fsm.imu_frozen_steps``). A default-constructed ``IMUState`` is invalid, so a
dead IMU's identity quaternion can no longer pass as "upright".

LowState freshness gate (item 236, notes/202 F2): ``POLICY`` is refused, and left, when the LowState itself is
stale, independent of the HAL-side ``imu.age_us``: (a) transport age (runner receive time minus publisher
``stamp_ns``, same host only) or the time the same tick has been re-used exceeds ``fsm.state_max_age_s``; (b) the
tick lags the controller clock by more than ``fsm.state_max_age_s`` relative to the best (lowest-latency) sample seen,
which works across hosts without a shared clock (needs ``state_tick_dt_s``); (c) the tick went backwards (HAL
restart). Ticks are compared modulo 2**32 (``sdk.types.tick_diff``, item 258), so the uint32 wrap after 99.4 days at
500 Hz is a normal forward step, not (c). The policy is not evaluated on a stale sample: for up to ``fsm.state_stale_grace_steps`` consecutive stale
steps the last policy target is re-sent, then the FSM goes to ``PASSIVE``. Before this gate a LowState 0.7 s old
with a fresh ``imu.age_us`` drove the policy.

L4 motor-mode gate (item 157, notes/46 F2): in every active state (MoveToDefault / Hold / Policy) a gated motor
that the robot side had armed (``LowState.motor.mode == 1``) and that then reports mode != 1 for more than
``fsm.motor_mode_grace_steps`` steps (HAL: stale reply, latched group fault, watchdog damping) sends the FSM to
``PASSIVE`` and latches it; a partially armed robot (some motors never 1 within ``motor_mode_arm_timeout_s``)
trips the same way. Before this gate the FSM never read ``mode``: a damped/faulted leg stayed "held" with q frozen
at the last reply.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..sdk import motors
from ..sdk.types import LowCmd, LowState, MotorCmdBlock, MotorMode, tick_diff
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
    transition: str | None = None   # last transition of this step (== transitions[-1]); kept for callers
    transitions: list[str] = field(default_factory=list)  # every transition of this step, in order (item 260)
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
        self._pending: list[str] = []  # transitions since the last step() (request() + step), reported in order
        self._last_refusal: str | None = None
        self._imu_tick: int | None = None
        self._imu_tick_t0 = 0.0
        self._imu_last_raw: bytes | None = None
        self._imu_same = 0
        self._imu_health: tuple[str | None, float] = ("imu_unseen", float("inf"))
        self.state_tick_dt_s: float | None = None  # LowState tick period [s]; enables the cross-host lag check
        self._st_tick: int | None = None
        self._st_utick = 0                         # unwrapped LowState tick (uint32 wrap, item 258)
        self._st_tick_t0 = 0.0
        self._st_base: float | None = None         # lowest (t - tick * dt) seen, drift-relaxed
        self._st_t_base = 0.0
        self._st_lag = 0.0
        self._st_regressed = False
        self._st_bad = 0                           # consecutive stale steps in POLICY
        self._last_q_policy: np.ndarray | None = None

        n = motors.NUM_MOTORS
        mask = cfg.fsm.motor_mode_mask
        self._mode_mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool).reshape(n)
        self._mode_armed = np.zeros(n, bool)      # motor reported mode 1 at least once
        self._mode_bad_steps = np.zeros(n, np.int64)  # consecutive steps at mode != 1 after armed
        self._t_active = None                     # controller time the current active stretch started
        self.latched: dict | None = None          # set by the motor-mode gate; cleared only by clear_latch()
        self.refused: list[dict] = []             # requests refused while latched

    # ------------------------------------------------------------------ IMU health

    def observe_imu(self, low_state: LowState, t: float) -> tuple[str | None, float]:
        """Update the IMU-health tracker with ``low_state`` at controller time ``t``.

        Returns ``(fault, age_s)``: ``fault`` is ``None`` (healthy), ``"imu_invalid"``, ``"imu_stale"`` or
        ``"imu_frozen"``; ``age_s`` is the effective sample age (``imu.age_us`` + time this LowState tick has
        been re-used, ``inf`` when unknown). Idempotent for repeated calls with the same (tick, t).
        """
        fsm, imu = self.cfg.fsm, low_state.imu
        tick = int(low_state.tick)
        new_tick = tick != self._imu_tick
        if new_tick:
            self._imu_tick, self._imu_tick_t0 = tick, t
            raw = imu.gyro.tobytes() + imu.accel.tobytes()
            self._imu_same = self._imu_same + 1 if raw == self._imu_last_raw else 0
            self._imu_last_raw = raw
        age = imu.age_s + max(0.0, t - self._imu_tick_t0)
        fault = None
        if fsm.imu_gate:
            if int(imu.valid) != 1:
                fault = "imu_invalid"
            elif fsm.imu_max_age_s is not None and not age <= fsm.imu_max_age_s:
                fault = "imu_stale"
            elif fsm.imu_frozen_steps > 0 and self._imu_same >= fsm.imu_frozen_steps:
                fault = "imu_frozen"
        self._imu_health = (fault, age)
        return fault, age

    # ------------------------------------------------------------------ LowState freshness

    def observe_state(self, low_state: LowState, t: float,
                      state_age_s: float | None = None) -> tuple[str | None, float, float]:
        """Update the LowState freshness tracker (item 236) at controller time ``t``.

        Args:
            state_age_s: transport age of ``low_state`` measured by the caller now (receive clock minus publisher
                ``stamp_ns``; ``None`` when unknown or cross-host).

        Returns ``(fault, age_s, lag_s)``: ``fault`` is ``None``, ``"state_stale"`` or ``"state_tick_regressed"``;
        ``age_s = max(transport age, time this tick has been re-used)``; ``lag_s`` is how far the tick trails the
        controller clock relative to the best sample seen (0 without ``state_tick_dt_s``). Idempotent for repeated
        calls with the same (tick, t).
        """
        fsm = self.cfg.fsm
        tick = int(low_state.tick)
        if tick != self._st_tick:
            # Modular difference: the uint32 wrap (99.4 days at 500 Hz) is a forward step, not a regression.
            d = 0 if self._st_tick is None else tick_diff(tick, self._st_tick)
            self._st_regressed = d < 0
            self._st_utick = tick if (self._st_tick is None or self._st_regressed) else self._st_utick + d
            self._st_tick, self._st_tick_t0 = tick, t
            dt = self.state_tick_dt_s
            if dt:
                off = t - self._st_utick * dt
                if self._st_base is None or self._st_regressed:
                    base = off
                else:  # relax the baseline by the allowed clock drift so ppm-level drift is not read as lag
                    base = min(off, self._st_base + fsm.state_drift_ppm * 1e-6 * max(0.0, t - self._st_t_base))
                self._st_base, self._st_t_base = base, t
                self._st_lag = off - base
        reuse = max(0.0, t - self._st_tick_t0)
        transport = state_age_s if state_age_s is not None and np.isfinite(state_age_s) and state_age_s >= 0 else 0.0
        age = max(reuse, float(transport))
        fault = None
        if fsm.state_gate:
            if self._st_regressed:
                fault = "state_tick_regressed"
            elif fsm.state_max_age_s is not None and (age > fsm.state_max_age_s or self._st_lag > fsm.state_max_age_s):
                fault = "state_stale"
        return fault, age, self._st_lag

    # ------------------------------------------------------------------ latch

    def clear_latch(self) -> None:
        """Operator reset of the motor-mode latch (after the robot side was re-armed). Stays in ``PASSIVE``."""
        self.latched = None
        self._mode_armed[:] = False
        self._mode_bad_steps[:] = 0

    def _motor_mode_check(self, low_state: LowState, t: float, info: StepInfo) -> str | None:
        """Update the per-motor arm/drop counters; return a trip reason or ``None``."""
        fsm = self.cfg.fsm
        mode = np.asarray(low_state.motor.mode).reshape(-1)
        if mode.shape[0] != motors.NUM_MOTORS:
            return f"motor_mode(size {mode.shape[0]} != {motors.NUM_MOTORS})"
        on = mode == MotorMode.ENABLE
        self._mode_armed |= on
        active = self.state != FsmState.PASSIVE
        if not active:
            self._mode_bad_steps[:] = 0
            self._t_active = None
            return None
        if self._t_active is None:
            self._t_active = t
        g = self._mode_mask
        self._mode_bad_steps = np.where(self._mode_armed & ~on, self._mode_bad_steps + 1, 0)
        info.extra["motor_mode_bad"] = int(np.count_nonzero(g & (self._mode_bad_steps > 0)))
        dropped = np.flatnonzero(g & (self._mode_bad_steps > fsm.motor_mode_grace_steps))
        if dropped.size:
            return "motor_mode_dropped:" + ",".join(motors.MOTOR_NAMES[i] for i in dropped)
        if (fsm.motor_mode_arm_timeout_s is not None and np.any(self._mode_armed & g)
                and t - self._t_active > fsm.motor_mode_arm_timeout_s):
            never = np.flatnonzero(g & ~self._mode_armed)
            if never.size:
                return "motor_mode_never_armed:" + ",".join(motors.MOTOR_NAMES[i] for i in never)
        return None

    # ------------------------------------------------------------------ transitions

    def request(self, new: FsmState, low_state: LowState, t: float, state_age_s: float | None = None) -> bool:
        """Enter ``new`` (runs its ``enter`` logic). Returns ``False`` when ``POLICY`` is refused by the IMU gate, the
        state freshness gate or the tilt check (``bad_orientation``, item 275), or (for any non-Passive state) by the
        motor-mode latch; the state is then unchanged.

        Every transition (including refusals) is reported in the next step's ``StepInfo.transitions`` in order, so a
        request followed by a trip in the same step keeps both. An IMU refusal reports
        ``"<state>->policy refused(<fault>)"`` as the next step's transition (once per distinct
        fault; callers may retry every step).
        """
        if self.latched is not None and new != FsmState.PASSIVE:
            self.refused.append({"t": t, "requested": new.value, "latched": self.latched["reason"]})
            self._pending.append(f"{self.state.value}->{new.value}(refused:motor_mode_latched)")
            return False
        if new == FsmState.POLICY and self.policy is None:
            raise RuntimeError("POLICY requested but no ONNX policy is loaded")
        if new == FsmState.POLICY:
            fault, _ = self.observe_imu(low_state, t)
            if fault is None:
                fault, _, _ = self.observe_state(low_state, t, state_age_s)
            if fault is None and self._tilted(low_state) is not None:
                fault = "bad_orientation"
            if fault is not None:
                if fault != self._last_refusal:
                    self._pending.append(f"{self.state.value}->policy refused({fault})")
                self._last_refusal = fault
                return False
        self._last_refusal = None
        self._pending.append(f"{self.state.value}->{new.value}")
        self.state = new
        self._t_enter = t
        q = low_state.motor.q.astype(np.float64)
        if new == FsmState.MOVE_TO_DEFAULT:
            self._q_start = q.copy()
        elif new == FsmState.POLICY:
            self._last_action = np.zeros(self.cfg.num_actions)
            self._policy_step = 0
            self._st_bad = 0
            self._last_q_policy = None
            self.obs_builder.reset()
            if self.motion is not None:
                ctx = self._ctx(low_state)
                pos = ctx.robot_anchor_pos_w() if (self.allow_privileged and low_state.sim is not None) else None
                self.motion.align(ctx.robot_anchor_quat_w(), pos)
        return True

    def _tilted(self, low_state: LowState) -> float | None:
        """Torso tilt [rad] when it exceeds ``fsm.bad_orientation_rad`` (else ``None``; ``None`` also when disabled)."""
        lim = self.cfg.fsm.bad_orientation_rad
        if lim is None:
            return None
        tilt = Q.tilt_angle(low_state.imu.quat_wxyz.astype(np.float64))
        return tilt if tilt > lim else None

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

    def step(self, low_state: LowState, t: float, state_age_s: float | None = None) -> tuple[LowCmd, StepInfo]:
        """Compute the command for controller time ``t`` [s] (``state_age_s``: see :meth:`observe_state`)."""
        n = motors.NUM_MOTORS
        cfg = self.cfg
        q_meas = low_state.motor.q.astype(np.float64)
        kp, kd = self.full_kp.copy(), self.full_kd.copy()
        fault, age = self.observe_imu(low_state, t)
        info_extra = {"imu_valid": int(low_state.imu.valid), "imu_age_s": age, "imu_fault": fault}
        if self.state == FsmState.POLICY and fault is not None:
            # Leave before evaluating the policy: its observations (gravity, gyro, anchor) are not trustworthy.
            self.state = FsmState.PASSIVE
            self._pending.append(f"policy->passive({fault})")
        tilt = self._tilted(low_state) if self.state == FsmState.POLICY else None
        if tilt is not None:
            # Item 275: leave before evaluating the policy, so the trip step itself sends Passive damping, not a policy
            # target at policy kp (the IMU is valid here, so the quaternion is trusted).
            info_extra["tilt_rad"] = tilt
            self.state = FsmState.PASSIVE
            self._pending.append("policy->passive(bad_orientation)")
        sfault, sage, slag = self.observe_state(low_state, t, state_age_s)
        info_extra.update(state_age_s=sage, state_lag_s=slag, state_fault=sfault)
        hold_last = False
        if self.state == FsmState.POLICY and sfault is not None:
            self._st_bad += 1
            if self._last_q_policy is None or self._st_bad > cfg.fsm.state_stale_grace_steps:
                self.state = FsmState.PASSIVE
                self._pending.append(f"policy->passive({sfault})")
            else:
                hold_last = True
        else:
            self._st_bad = 0
        info = StepInfo(state=self.state, q_des=np.zeros(n), extra=info_extra)
        if cfg.fsm.motor_mode_gate:
            reason = self._motor_mode_check(low_state, t, info)
            if reason is not None:
                self._pending.append(f"{self.state.value}->passive({reason})")
                self.latched = {"t": t, "tick": int(low_state.tick), "from": self.state.value, "reason": reason}
                self.state = FsmState.PASSIVE
                self._t_active = None
                info.state = self.state
        if self.state == FsmState.PASSIVE:
            q_des, kp, kd = q_meas.copy(), np.zeros(n), cfg.fsm.passive_kd.copy()
        elif self.state == FsmState.MOVE_TO_DEFAULT:
            ratio = float(np.clip((t - self._t_enter) / max(cfg.fsm.move_to_default_s, 1e-6), 0.0, 1.0))
            q_des = self._q_start + ratio * (cfg.hold_target_sdk - self._q_start)
            kp, kd = cfg.fsm.hold_kp.copy(), cfg.fsm.hold_kd.copy()
            info.extra["ratio"] = ratio
            if ratio >= 1.0:
                self.state = FsmState.HOLD
                self._pending.append("move_to_default->hold")
        elif self.state == FsmState.HOLD:
            q_des = cfg.hold_target_sdk.copy()
            kp, kd = cfg.fsm.hold_kp.copy(), cfg.fsm.hold_kd.copy()
        elif hold_last:  # POLICY on a stale sample: re-send the last policy target, do not evaluate the policy
            q_des = self._last_q_policy.copy()
            info.extra["state_hold_last"] = self._st_bad
        else:  # POLICY
            q_des = self._policy_step_q(low_state, info)
            self._last_q_policy = q_des.copy()
            info.extra["tilt_rad"] = Q.tilt_angle(low_state.imu.quat_wxyz.astype(np.float64))
            if self.motion is not None and self.motion.done(self._policy_step):
                self.state = FsmState.HOLD
                self._pending.append("policy->hold(motion_end)")
        cmd = LowCmd(tick=low_state.tick)
        cmd.motor = MotorCmdBlock.from_arrays(n, mode=MotorMode.ENABLE, q=q_des, dq=0.0, tau=0.0, kp=kp, kd=kd)
        info.q_des = q_des
        info.transitions, self._pending = self._pending, []
        info.transition = info.transitions[-1] if info.transitions else None
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
        if cfg.ankle_roll_bind:  # pinned tie-rod ankle: semantic roll clamp, as in training (sidecar ankle_roll_bind)
            if getattr(self, "_ankle_bind", None) is None:
                from dropbear_wbc.kinematics.ankle_bind import AnkleRollBind

                self._ankle_bind = AnkleRollBind(roll_limit_deg=cfg.ankle_roll_bind["roll_limit_deg"],
                                                 margin_deg=cfg.ankle_roll_bind["margin_deg"])
            q_des, hit = self._ankle_bind.clamp(q_des, return_mask=True)
            if hit.any():
                info.extra["ankle_roll_bind"] = hit.tolist()
        self._last_action = action
        info.policy_ms = 1e3 * (time.perf_counter() - t0)
        info.obs, info.action, info.motion_frame = obs, action, frame
        self._policy_step += 1
        return q_des
