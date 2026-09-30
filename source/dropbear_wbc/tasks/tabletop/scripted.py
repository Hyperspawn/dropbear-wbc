"""IK-based scripted push policy for the tabletop task (pure numpy; one instance per env).

It reads privileged state (block pose, zone pose, measured arm joints; root frame) and outputs the commanded SEMANTIC
arm angles (10, left then right; CONTRACTS section 2) that the env maps to the 10 arm motors with
``SemanticMap.semantic_to_motor``. The inactive arm holds the calibrated standing pose.

Motion (Cartesian goals for the TOOL point on the hand axis, :mod:`.kinematics`; the IK is the teleop priority solver):

    LIFT      tool up beside the torso to hover height, behind the table edge
    TRANSIT   to the pre-push point above the table (stand-off behind the block, opposite the zone)
    DESCEND   hand's lowest point to push height (8 mm above the table)
    PUSH      closed loop on the block pose: the tool target leads the block's trailing face by ``penetration``
              along u = unit(zone - block), rate-limited to ``push_speed``; re-approach if the hand leaves the push line
    RETREAT   back off along -u, RISE to hover, BACK to the lift point, HOME (joint-space to the standing pose), IDLE

An outer-loop integrator corrects the commanded tool point by the error between the commanded and the MEASURED tool
point (FK of the measured semantic arm angles), which removes steady-state sag of the position-controlled arm.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .kinematics import HAND_LENGTH_M, HAND_RADIUS_M, TabletopArmIK
from .layout import LAYOUT, TabletopLayout, block_half_extent_along

PHASES = ("LIFT", "TRANSIT", "DESCEND", "PUSH", "RETREAT", "RISE", "BACK", "HOME", "IDLE")
PHASE_ID = {p: i for i, p in enumerate(PHASES)}


@dataclass
class PushParams:
    push_clear: float = 0.008       # hand lowest point above the table while pushing [m]
    hover_clear: float = 0.075      # ... while hovering (block top + 2.5 cm)
    standoff: float = 0.04          # gap between hand and block face at the pre-push point (> tracking error)
    penetration: float = 0.008      # the push target leads the modelled contact point by this much ...
    max_lead: float = 0.04          # ... growing up to this while the block does not move (model / sag error)
    lead_rate: float = 0.02         # lead growth [m/s] while stalled
    stall_move: float = 0.002       # block displacement [m] over stall_window_s below which the push is "stalled"
    stall_window_s: float = 0.5
    transit_speed: float = 0.18     # [m/s]
    descend_speed: float = 0.10
    push_speed: float = 0.07
    settle_tol: float = 0.012       # phase transitions wait until the MEASURED tool point is this close to the goal
    settle_timeout_s: float = 2.0   # ... or this long (then proceed anyway)
    phase_timeout_s: float = 4.0    # TRANSIT / DESCEND end after this long even if the goal kept moving
    retreat: float = 0.035
    stop_tol: float = 0.008         # stop when the block centre is this close to the zone centre (or past it)
    realign_lat: float = 0.03       # re-approach when the hand is this far off the push line ...
    realign_deg: float = 40.0       # ... or the hand->block direction deviates this much from u
    max_reapproach: int = 3
    axis_tilt_deg: float = 60.0     # preferred hand axis: tilted forward from straight down (soft; best in workspace.log)
    lift_x: float = 0.10            # lift waypoint: behind the table edge ...
    lift_y_out: float = 0.10        # ... and outward of the shoulder
    home_s: float = 1.5
    integrator_gain: float = 0.25   # per step (horizontal only: a vertical integrator winds up against the table)
    integrator_limit: float = 0.02  # [m]
    max_joint_step: float = 0.12    # rad per control step (semantic), a safety rate limit

    def to_dict(self) -> dict:
        return asdict(self)


def hand_support(ik: TabletopArmIK, side: str, q5: np.ndarray, u: np.ndarray, z_band_top: float) -> float:
    """Extent of the hand cylinder beyond the tool point along horizontal unit ``u``, for the part of the hand whose
    lower surface is below ``z_band_top`` (the part that can touch a block of that height)."""
    hp = ik.hand(side, q5)
    a = hp.axis
    u3 = np.array([u[0], u[1], 0.0])
    rad_u = HAND_RADIUS_M * math.sqrt(max(0.0, 1.0 - float(a @ u3) ** 2))
    rad_z = HAND_RADIUS_M * math.sqrt(max(0.0, 1.0 - float(a[2]) ** 2))
    best = None
    for s in np.linspace(0.0, HAND_LENGTH_M, 9):
        p = hp.wrist + s * a
        if p[2] - rad_z <= z_band_top:
            v = float((p - hp.tool) @ u3)
            best = v if best is None else max(best, v)
    return (best if best is not None else 0.0) + rad_u


class ScriptedPushPolicy:
    """One env's scripted pusher. Call :meth:`reset` at each episode start, then :meth:`act` every control step."""

    def __init__(self, ik: TabletopArmIK, layout: TabletopLayout = LAYOUT, params: PushParams | None = None,
                 control_hz: float | None = None):
        self.ik = ik
        self.L = layout
        self.p = params or PushParams()
        self.dt = 1.0 / float(control_hz or layout.control_hz)
        t = math.radians(self.p.axis_tilt_deg)
        self.axis_pref = np.array([math.sin(t), 0.0, -math.cos(t)])
        self.q_rest = ik.rest_q()

    # ------------------------------------------------------------------ geometry helpers
    def lift_point(self, side: str) -> np.ndarray:
        sh = self.ik.chains[side].shoulder
        out = 1.0 if side == "left" else -1.0
        return np.array([self.p.lift_x, sh[1] + out * self.p.lift_y_out])

    def contact_offset(self, side: str, q5, u, yaw: float) -> float:
        """Tool-point distance behind the block centre (along u) at which the hand touches the block."""
        return (block_half_extent_along(u, yaw, self.L.block_size)
                + hand_support(self.ik, side, q5, np.asarray(u, float), self.L.table_top_z + self.L.block_size))

    # ------------------------------------------------------------------ episode
    def reset(self, side: str, zone_xy, q_sem10_measured=None) -> None:
        self.side = side
        self.sl = slice(0, 5) if side == "left" else slice(5, 10)
        self.zone_xy = np.asarray(zone_xy, float)[:2].copy()
        q = self.q_rest.copy() if q_sem10_measured is None else np.asarray(q_sem10_measured, float).copy()
        self.q_cmd = q.copy()
        for s in ("left", "right"):
            ssl = slice(0, 5) if s == "left" else slice(5, 10)
            self.ik.q_prev[s] = self.ik.chains[s].clip(q[ssl])
        hp = self.ik.hand(side, q[self.sl])
        self.tool_cmd = hp.tool.copy()            # commanded tool point (before the integrator)
        self.integ = np.zeros(3)
        self.phase = "LIFT"
        self.phase_t = 0.0
        self.t = 0.0
        self.n_reapproach = 0
        self.u = None
        self.home_from = None
        self.events: list[tuple[float, str]] = [(0.0, "LIFT")]
        self.done_push = False
        self.reapproach = False
        self._retreat_xy = None
        self.lead = self.p.penetration
        self._block_hist: list[tuple[float, np.ndarray]] = []
        self._goal = None  # (xy, lowest_z) of the current waypoint, for the measured-convergence gate

    def _set_phase(self, ph: str) -> None:
        self.phase, self.phase_t = ph, 0.0
        self.events.append((round(self.t, 3), ph))

    def _lowest_goal(self, xy, clear: float) -> tuple[np.ndarray, float]:
        return np.asarray(xy, float)[:2], self.L.table_top_z + clear

    def _settled(self, q_meas, tol: float | None = None) -> bool:
        """The MEASURED tool point is within ``settle_tol`` of the commanded one (or the phase timed out). Phase
        transitions use it, so a lagging arm never starts to descend or push from the wrong place."""
        if self.phase_t >= self.p.settle_timeout_s:
            return True
        meas = self.ik.hand(self.side, np.asarray(q_meas, float)[self.sl]).tool
        cmd = self.ik.hand(self.side, self.q_cmd[self.sl]).tool  # where the commanded joints put the tool
        return float(np.linalg.norm(meas - cmd)) <= (tol or self.p.settle_tol)

    def _move(self, goal_xy, goal_lowest_z, speed) -> bool:
        """Advance the commanded tool point toward (xy, lowest z) at ``speed``; returns True when there."""
        # target tool height from the current hand geometry (tool - lowest offset)
        hp = self.ik.hand(self.side, self.q_cmd[self.sl])
        dz = float(hp.tool[2] - hp.lowest_z)
        goal = np.array([goal_xy[0], goal_xy[1], goal_lowest_z + dz])
        d = goal - self.tool_cmd
        n = float(np.linalg.norm(d))
        step = speed * self.dt
        if n <= step:
            self.tool_cmd = goal
            return True
        self.tool_cmd = self.tool_cmd + d * (step / n)
        return False

    def act(self, block_pos, block_yaw: float, q_sem10_measured) -> tuple[np.ndarray, dict]:
        """One control step. ``block_pos`` (3,) root frame; returns (commanded semantic arm q (10,), info)."""
        p, L = self.p, self.L
        q_meas = np.asarray(q_sem10_measured, float)
        b = np.asarray(block_pos, float)
        self.t += self.dt
        self.phase_t += self.dt
        side, sl = self.side, self.sl
        info: dict = {}
        if self.phase in ("LIFT", "TRANSIT", "DESCEND", "PUSH", "RETREAT", "RISE", "BACK"):
            if self.u is None or self.phase in ("LIFT", "TRANSIT"):
                v = self.zone_xy - b[:2]
                self.u = v / max(np.linalg.norm(v), 1e-9)
            u = self.u
            c_off = self.contact_offset(side, self.q_cmd[sl], u, block_yaw)
            pre_xy = b[:2] - u * (c_off + p.standoff)
            if self.phase == "LIFT":
                if self._move(self.lift_point(side), L.table_top_z + p.hover_clear, p.transit_speed)                         and self._settled(q_meas, 3 * p.settle_tol):
                    self._set_phase("TRANSIT")
            elif self.phase == "TRANSIT":
                there = self._move(pre_xy, L.table_top_z + p.hover_clear, p.transit_speed)
                if (there and self.phase_t > 0.2 and self._settled(q_meas)) or self.phase_t > p.phase_timeout_s:
                    self._set_phase("DESCEND")
            elif self.phase == "DESCEND":
                there = self._move(pre_xy, L.table_top_z + p.push_clear, p.descend_speed)
                # the goal is re-derived from the block pose and the hand geometry every step, so it can creep and
                # never be "reached" exactly: proceed after phase_timeout_s
                if (there and self._settled(q_meas)) or self.phase_t > p.phase_timeout_s:
                    self._set_phase("PUSH")
                    self.lead = p.penetration
                    self._block_hist = []
            elif self.phase == "PUSH":
                v = self.zone_xy - b[:2]
                dist = float(np.linalg.norm(v))
                along = float(v @ u)
                tool_meas = self.ik.hand(side, q_meas[sl]).tool
                w = b[:2] - tool_meas[:2]
                lat = abs(float(w[0] * u[1] - w[1] * u[0]))
                ang = math.degrees(math.acos(float(np.clip((w @ u) / max(np.linalg.norm(w), 1e-9), -1, 1))))
                info.update(push_dist=dist, push_lat=lat, push_ang=ang)
                if dist < p.stop_tol or along < p.stop_tol * 0.5:
                    self.done_push = True
                    self._set_phase("RETREAT")
                elif (lat > p.realign_lat or ang > p.realign_deg) and self.phase_t > 0.3 \
                        and self.n_reapproach < p.max_reapproach:
                    self.n_reapproach += 1
                    self.u = None
                    self._set_phase("RISE")
                    self.reapproach = True
                else:
                    if dist > 1e-6:  # re-aim at the zone continuously (small corrections)
                        self.u = v / dist
                        u = self.u
                    # stall detection: the block has not moved over the last window -> lead further into it (the
                    # hand model / arm sag leave the real hand short of the modelled contact); moving -> relax
                    self._block_hist.append((self.t, b[:2].copy()))
                    while self._block_hist and self._block_hist[0][0] < self.t - p.stall_window_s:
                        self._block_hist.pop(0)
                    moved = float(np.linalg.norm(b[:2] - self._block_hist[0][1]))
                    if self.phase_t > p.stall_window_s and moved < p.stall_move:
                        self.lead = min(p.max_lead, self.lead + p.lead_rate * self.dt)
                    elif moved > 3 * p.stall_move:
                        self.lead = max(p.penetration, self.lead - p.lead_rate * self.dt)
                    info.update(push_lead=self.lead)
                    tgt_xy = b[:2] - u * (c_off - self.lead)
                    self._move(tgt_xy, L.table_top_z + p.push_clear, p.push_speed)
            elif self.phase == "RETREAT":
                back = self.tool_cmd[:2] - u * p.retreat if self.phase_t <= self.dt + 1e-9 else self._retreat_xy
                self._retreat_xy = back
                if self._move(back, L.table_top_z + p.push_clear, p.descend_speed):
                    self._set_phase("RISE")
                    self.reapproach = False
            elif self.phase == "RISE":
                if self._move(self.tool_cmd[:2], L.table_top_z + p.hover_clear, p.descend_speed):
                    if getattr(self, "reapproach", False) and not self.done_push:
                        self._set_phase("TRANSIT")
                    else:
                        self._set_phase("BACK")
            elif self.phase == "BACK":
                if self._move(self.lift_point(side), L.table_top_z + p.hover_clear, p.transit_speed):
                    self._set_phase("HOME")
                    self.home_from = self.q_cmd.copy()
            # outer-loop integrator on the measured tool point (not while transiting fast)
            meas = self.ik.hand(side, q_meas[sl]).tool
            cmd_fk = self.ik.hand(side, self.q_cmd[sl]).tool
            if self.phase in ("DESCEND", "PUSH", "RETREAT"):
                d_int = p.integrator_gain * (cmd_fk - meas)
                d_int[2] = 0.0  # horizontal only
                self.integ = np.clip(self.integ + d_int, -p.integrator_limit, p.integrator_limit)
            res = self.ik.solve(side, self.tool_cmd + self.integ, self.axis_pref, q_init=self.q_cmd[sl],
                                restarts="light")
            q_new = self.q_cmd.copy()
            q_new[sl] = res.q
            info.update(ik_err=res.pos_err_m)
        elif self.phase == "HOME":
            a = min(1.0, self.phase_t / p.home_s)
            s = a * a * (3 - 2 * a)
            q_new = (1 - s) * self.home_from + s * self.q_rest
            if a >= 1.0:
                self._set_phase("IDLE")
        else:  # IDLE
            q_new = self.q_rest.copy()
        # the inactive arm holds the standing pose; semantic rate limit (safety)
        other = slice(5, 10) if side == "left" else slice(0, 5)
        q_new[other] = self.q_rest[other]
        dq = np.clip(q_new - self.q_cmd, -p.max_joint_step, p.max_joint_step)
        self.q_cmd = self.q_cmd + dq
        info.update(phase=self.phase, phase_id=PHASE_ID[self.phase], tool_cmd=self.tool_cmd.copy(),
                    integ=self.integ.copy(), n_reapproach=self.n_reapproach)
        return self.q_cmd.copy(), info


def plan_check_points(ik: TabletopArmIK, pol: ScriptedPushPolicy, side: str, block_xy, block_yaw: float, zone_xy):
    """Kinematic feasibility points of a placement (root frame): list of (name, xy, lowest-point clearance)."""
    b, z = np.asarray(block_xy, float), np.asarray(zone_xy, float)
    v = z - b
    d = float(np.linalg.norm(v))
    u = v / d
    q5 = ik.q_prev[side]
    c_off = pol.contact_offset(side, q5, u, block_yaw)
    p = pol.p
    pts = [("lift", pol.lift_point(side), p.hover_clear),
           ("prepush_hover", b - u * (c_off + p.standoff), p.hover_clear),
           ("prepush", b - u * (c_off + p.standoff), p.push_clear)]
    for k, s in enumerate(np.arange(0.0, d + 1e-9, 0.02)):
        pts.append((f"push_{k}", b + u * s - u * c_off, p.push_clear))
    end = z - u * c_off
    pts += [("push_end", end, p.push_clear), ("retreat", end - u * p.retreat, p.push_clear),
            ("retreat_hover", end - u * p.retreat, p.hover_clear)]
    return pts
