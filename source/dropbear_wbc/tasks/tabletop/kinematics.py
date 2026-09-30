"""Hand-tool kinematics for the tabletop task (pure numpy; importable without Isaac).

Dropbear has no hand: each arm ends in a cylindrical hand body (``LH/RH_shoulder_ex_al_interface_1``) that extends
from the wrist point along the forearm axis. Its collision hull (``tools/tabletop_geometry_probe.py`` ->
``logs/tabletop/geometry_probe.json``) is a cylinder of radius ~0.04 m (max radial distance of the hull 0.043 m) and
length 0.1675 m along the body -y axis, which is the IK end-effector x axis (``x_ee``, "along the forearm toward the
fingers", teleop/arm_ik.py): at the authored rest the two agree within 1.5 deg (probe + ``DropbearArmIK`` FK).

The teleop IK (:mod:`dropbear_wbc.teleop.arm_ik`) solves for the WRIST point. For pushing, the contact happens near
the far end of the hand, so this module re-uses the same calibrated chains with the end-effector moved to a TOOL
point on the hand axis (``x_ee * tool_d`` from the wrist). The wrist-roll axis is the hand axis, so the tool point
does not depend on wrist roll: 4 joints place it and 1 swivel DoF is left for the orientation preference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

from dropbear_wbc.teleop.arm_ik import (
    ARM_SEMANTIC_INDEX,
    SIDES,
    ArmChain,
    ArmIKResult,
    DropbearArmIK,
    IKWeights,
    _rot_batch,  # noqa: F401  (re-exported for tests)
    solve_chain_robust,
)

HAND_LENGTH_M: float = 0.1675
"""Hand cylinder length along x_ee from the wrist point (collision hull, geometry probe)."""
HAND_RADIUS_M: float = 0.043
"""Hand cylinder radius (max radial distance of the collision hull vertices from the hand axis)."""
TOOL_D_M: float = 0.13
"""Default tool point: on the hand axis, 0.13 m from the wrist (3.75 cm before the hand's end face)."""


@dataclass
class ToolChain(ArmChain):
    """An :class:`ArmChain` whose end-effector point is ``wrist + R_ee @ tool`` (``tool`` in the EE frame)."""

    tool: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def fk(self, q: np.ndarray, jacobian: bool = False):
        out = ArmChain.fk(self, q, jacobian=jacobian)
        off = out[1] @ self.tool
        if not jacobian:
            return out[0] + off, out[1]
        pos, r, jac = out
        jac = jac.copy()
        # v_tool = v_wrist + w x (R tool): add w_i x off to every joint column (wrist roll's w is parallel to the
        # hand axis, so a tool point on that axis gets 0 from it, as it should)
        jac[:3] += np.cross(jac[3:].T, off).T
        return pos + off, r, jac

    def fk_batch(self, q: np.ndarray, with_elbow: bool = False):
        out = ArmChain.fk_batch(self, q, with_elbow=with_elbow)
        off = np.einsum("...ij,j->...i", out[1], self.tool)
        return (out[0] + off,) + tuple(out[1:])


def tool_chain(chain: ArmChain, tool_d: float) -> ToolChain:
    """The same calibrated chain with its end-effector moved ``tool_d`` metres along x_ee."""
    kw = {f.name: getattr(chain, f.name) for f in fields(ArmChain)}
    return ToolChain(**kw, tool=np.array([float(tool_d), 0.0, 0.0]))


def rot_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector ``a`` onto unit vector ``b``."""
    a = np.asarray(a, float) / max(np.linalg.norm(a), 1e-12)
    b = np.asarray(b, float) / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    s, c = float(np.linalg.norm(v)), float(np.clip(a @ b, -1.0, 1.0))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        k = np.cross(a, perp)
        k /= np.linalg.norm(k)
        return 2.0 * np.outer(k, k) - np.eye(3)
    k = v / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * kx + (1 - math.cos(ang)) * kx @ kx


@dataclass
class HandPose:
    """Hand cylinder of one arm in the ROOT frame."""

    wrist: np.ndarray       # (3,) wrist point (hand body origin)
    tip: np.ndarray         # (3,) centre of the hand's end face
    axis: np.ndarray        # (3,) unit x_ee
    tool: np.ndarray        # (3,) tool point
    lowest_z: float         # lowest point of the cylinder
    tilt_rad: float         # angle between x_ee and straight down


class TabletopArmIK:
    """Tool-point IK for pushing, on top of the teleop :class:`DropbearArmIK` (same calibration, limits, motor map).

    ``solve(side, p_tool_root, axis_pref)``: position of the tool point (strict priority), then in its null space the
    hand axis toward ``axis_pref`` (orientation weight), posture toward the calibrated standing pose and smoothing
    toward the previous solution -- the teleop solver (``solve_chain_robust``) unchanged, on a :class:`ToolChain`.
    Positions are in the ROOT (``world`` body) frame; ``DropbearArmIK.torso_origin_root`` converts to the teleop
    torso frame.
    """

    def __init__(self, calibration=None, tool_d: float = TOOL_D_M, weights: IKWeights | None = None,
                 base: DropbearArmIK | None = None):
        self.base = base or DropbearArmIK(calibration)
        self.tool_d = float(tool_d)
        self.weights = weights or IKWeights()
        self.chains = {s: tool_chain(self.base.chains[s], self.tool_d) for s in SIDES}
        self.q_prev: dict[str, np.ndarray] = {s: self.base.chains[s].q_rest.copy() for s in SIDES}

    # ------------------------------------------------------------------ kinematics
    def hand(self, side: str, q5: np.ndarray) -> HandPose:
        wrist, r = self.base.chains[side].fk(np.asarray(q5, float))
        ax = r[:, 0]
        tip = wrist + HAND_LENGTH_M * ax
        rad = HAND_RADIUS_M * math.sqrt(max(0.0, 1.0 - float(ax[2]) ** 2))
        lowest = min(float(wrist[2]), float(tip[2])) - rad
        tilt = math.acos(float(np.clip(-ax[2], -1.0, 1.0)))
        return HandPose(wrist=wrist, tip=tip, axis=ax, tool=wrist + self.tool_d * ax, lowest_z=lowest, tilt_rad=tilt)

    def elbow_root(self, side: str, q5: np.ndarray) -> np.ndarray:
        """Elbow-axis point (root frame) for the arm-over-table clearance checks."""
        _, _, elbow = self.base.chains[side].fk_batch(np.asarray(q5, float)[None], with_elbow=True)
        return elbow[0]

    # ------------------------------------------------------------------ IK
    def solve(self, side: str, p_tool_root: np.ndarray, axis_pref: np.ndarray | None = None,
              q_init: np.ndarray | None = None, max_iters: int = 60, restarts: str = "full") -> ArmIKResult:
        """Tool-point IK for one arm (root frame). ``axis_pref``: preferred hand axis direction (default down)."""
        ch = self.chains[side]
        q0 = self.q_prev[side] if q_init is None else ch.clip(np.asarray(q_init, float))
        _, r0 = ch.fk(q0)
        a_des = np.array([0.0, 0.0, -1.0]) if axis_pref is None else np.asarray(axis_pref, float)
        t = np.eye(4)
        t[:3, :3] = rot_align(r0[:, 0], a_des) @ r0  # axis-only error: keep the current roll about the hand axis
        t[:3, 3] = np.asarray(p_tool_root, float)
        res = solve_chain_robust(ch, t, q0, self.q_prev[side], self.weights, max_iters=max_iters, restarts=restarts)
        self.q_prev[side] = res.q.copy()
        return res

    def solve_lowest(self, side: str, xy_root: np.ndarray, lowest_z: float, axis_pref=None, q_init=None,
                     iters: int = 3, **kw) -> tuple[ArmIKResult, HandPose]:
        """Place the tool point over ``xy_root`` with the hand's LOWEST point at ``lowest_z`` (fixed-point on the
        height offset between the tool point and the cylinder's lowest point, which depends on the hand tilt)."""
        dz = HAND_LENGTH_M - self.tool_d  # vertical hand: the tool point sits this far above the end face
        res = hp = None
        q = q_init
        for _ in range(iters):
            p = np.array([xy_root[0], xy_root[1], lowest_z + dz])
            res = self.solve(side, p, axis_pref, q_init=q, **kw)
            hp = self.hand(side, res.q)
            dz_new = float(hp.tool[2] - hp.lowest_z)
            q = res.q
            if abs(dz_new - dz) < 5e-4:
                break
            dz = dz_new
        return res, hp

    def to_motor10(self, q_sem10: np.ndarray) -> np.ndarray:
        """Semantic arm angles (10, left then right) -> arm motor targets (10, SDK slots 12..21)."""
        m, _ = self.base.to_motor_fast(np.asarray(q_sem10, float))
        return m

    def motor_to_sem10(self, motor_q22: np.ndarray) -> np.ndarray:
        return self.base.motor_to_semantic_arms_fast(np.asarray(motor_q22, float))

    def rest_q(self) -> np.ndarray:
        return self.base.rest_q()

    @staticmethod
    def arm_slice(side: str) -> slice:
        return slice(0, 5) if side == "left" else slice(5, 10)


__all__ = ["HAND_LENGTH_M", "HAND_RADIUS_M", "TOOL_D_M", "ToolChain", "tool_chain", "TabletopArmIK", "HandPose",
           "rot_align", "ARM_SEMANTIC_INDEX"]
