"""Dropbear arm IK in the semantic (G1-named) joint space, xr_teleoperate ``R1_A5_ArmIK`` style.

Each Dropbear arm has 5 DoF (CONTRACTS section 2): ``shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
wrist_roll`` and no hand. The IK solves in that semantic space and the result goes through the measured map
``SemanticMap.semantic_to_motor`` to the 10 arm motors (``LH_yaw .. RH_wrist_roll``, SDK slots 12-21).

Kinematic model (all numbers from ``data/calibration/dropbear_semantic_calibration.json``)
-----------------------------------------------------------------------------------------
A serial product-of-exponentials chain in the root (``world`` body) frame, x forward, y left, z up, written at
the **semantic zero** (upper arm hanging, forearm pointing forward = G1 elbow 0, wrist roll 0):

====  ==============  ==================  =====================================================================
 #    semantic DoF    axis (zero frame)   point on the axis
====  ==============  ==================  =====================================================================
 1    shoulder_pitch  +y                  ``geometry.per_side.<s>.arm_points_rest_root.shoulder_center`` (S)
 2    shoulder_roll   +x                  S
 3    shoulder_yaw    +z                  the measured shoulder-yaw screw (``serial_screw_fits.<LH|RH>_roll``)
 4    elbow           +y                  ``arm_points_rest_root.elbow_center`` (E, best-fit pivot)
 5    wrist_roll      +x                  the wrist point W0 (wrist-roll joint anchor)
====  ==============  ==================  =====================================================================

* Joints 1-3 are the ``serial3`` shoulder: the semantic triple is the YXZ Euler decomposition of the upper-arm
  rotation, and on Dropbear the motor chain order is the same (``LH_yaw`` -> +-y, ``LH_pitch`` -> +-x,
  ``LH_roll`` -> +-z; ``serial_groups.<s>_shoulder.axes_root``), so this PoE chain is the exact semantic
  shoulder. The measured axis directions differ from the ideal ones by < 1e-4 rad; ``semantic_to_motor`` absorbs
  that. The shoulder-yaw (upper-arm roll) axis is offset from S by about 14 mm, which this model keeps.
* Joint 4 uses the G1 elbow convention (0 = forearm forward, +pi/2 = straight arm, positive = extension about
  +y). The real elbow is a four-bar: the forearm body rotation is exact (it defines the semantic angle) but the
  wrist moves about a best-fit pivot with ``elbow_wrist_point_pivot_rms_m`` (about 5 mm) of residual. The
  loader reports this model's error at the two other calibrated wrist positions (``geometry.per_side.<s>.zero``
  and ``.standing``) in :attr:`ArmChain.info`.
* W0 = E + Ry(0 - e_rest) (W_rest - E): the authored rest wrist (``arm_points_rest_root.wrist``, straight arm)
  rotated back to the semantic zero. ``e_rest = pi/2 - upper_forearm_angle`` (the authored arm is straight).
* End-effector frame: the Unitree humanoid arm convention used by xr_teleoperate / televuer (x along the
  forearm from wrist toward the fingers, z from pinky toward index, y completes). It is the identity at the
  semantic zero, so ``R_ee(q) = Ry(pitch) Rx(roll) Rz(yaw) Ry(elbow) Rx(wrist_roll)``.

Target frame ("torso" frame)
----------------------------
The IK takes wrist targets in the **torso frame**: root (``world`` body) axes, origin at the pelvis centre
``rest_transforms.pelvis_in_root`` (the midpoint of the hip-pitch anchors; it is also on the shoulder midline).
The frame is left/right symmetric: mirroring is ``y -> -y``.

Solver
------
xr_teleoperate ``R1_A5_ArmIK`` minimises one weighted cost with casadi/ipopt,

    50 |p(q) - p*|^2 + 0.5 |log(R* R(q)^T)|^2 + 0.02 |q|^2 + 0.1 |q - q_last|^2,

i.e. position >> orientation, a regulariser toward zero and smoothing toward the last solution. Here the same
terms and weights are used, but by default (``IKWeights.mode = "priority"``) as a strict task hierarchy, solved by
a bounded damped Gauss-Newton iteration with backtracking:

1. position: damped least squares on the wrist position error;
2. in the exact null space of the position task (SVD basis; 2-D for 5 joints): the weighted sum
   ``0.5 |e_rot|^2 + 0.02 |q - q_stand|^2 + 0.1 |q - q_prev|^2``, i.e. orientation, the **null-space posture bias
   toward the calibrated standing pose**, and smoothing toward the previous solution.

Reason: with a 5-DoF arm most operator wrist orientations are unreachable, and in the single weighted cost every
radian of orientation residual moves the wrist by about (0.5/50) / arm length ~ 3 cm (measured: up to 9 mm on
FK-generated, exactly reachable targets from a nearby warm start; see tests/test_teleop_arm_ik.py). The priority
form keeps the wrist on target and spends the 2 redundant DoF (elbow swivel and wrist roll) on orientation and
posture. ``mode="weighted"`` keeps the R1_A5 single cost for comparison.

Joint limits are box constraints (active set). Solutions are warm-started from the previous one; when the result
is still > 3 mm off a reachable target, the solver restarts from closed-form seeds (:func:`analytic_seed`).
Units: metres, radians.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from dropbear_wbc.kinematics.semantic import DEFAULT_CALIBRATION, MOTOR_NAMES, SEMANTIC_NAMES, SemanticMap

ARM_JOINTS: tuple[str, ...] = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
SIDES: tuple[str, ...] = ("left", "right")
MIRROR_SIGN = np.array([1.0, -1.0, -1.0, 1.0, -1.0])
"""G1 mirror convention per arm joint: roll / yaw flip sign, pitch / elbow keep (``calib_build`` symmetry)."""

ARM_SEMANTIC_NAMES: dict[str, tuple[str, ...]] = {s: tuple(f"{s}_{j}" for j in ARM_JOINTS) for s in SIDES}
ARM_SEMANTIC_INDEX: dict[str, tuple[int, ...]] = {
    s: tuple(SEMANTIC_NAMES.index(n) for n in ARM_SEMANTIC_NAMES[s]) for s in SIDES}
ARM_MOTOR_NAMES: dict[str, tuple[str, ...]] = {
    "left": ("LH_yaw", "LH_pitch", "LH_roll", "LH_elbow_joint", "LH_wrist_roll"),
    "right": ("RH_yaw", "RH_pitch", "RH_roll", "RH_elbow_joint", "RH_wrist_roll"),
}
ARM_MOTOR_INDEX: dict[str, tuple[int, ...]] = {s: tuple(MOTOR_NAMES.index(n) for n in ARM_MOTOR_NAMES[s]) for s in SIDES}
ARM_MOTOR_SLOTS: tuple[int, ...] = ARM_MOTOR_INDEX["left"] + ARM_MOTOR_INDEX["right"]
"""SDK slots of the 10 arm motors (12..21), left arm first."""

_AXIS_IDS = (1, 0, 2, 1, 0)  # joint axes at the semantic zero: y, x, z, y, x

# Teleop soft limits for the LEFT arm (semantic, rad), intersected with the calibration's valid ranges.
# The right arm uses the mirror image. They keep the IK out of self-contact / wrap-around regions; G1 values
# for comparison: pitch [-3.09, 2.67], roll [-1.59, 2.25], yaw [-2.62, 2.62], wrist roll [-1.97, 1.97].
DEFAULT_SOFT_LIMITS_LEFT = np.array([
    [-3.0, 1.0],       # shoulder_pitch: up to 57 deg backward, 172 deg forward/up
    [-np.inf, 2.6],    # shoulder_roll: calibration lower (-10 deg, LH_pitch motor limit), 149 deg abduction
    [-2.6, 2.6],       # shoulder_yaw
    [-np.inf, np.inf], # elbow: calibration valid range (four-bar)
    [-2.6, 2.6],       # wrist_roll
])


def mirror_limits(lim_left: np.ndarray) -> np.ndarray:
    """Left-arm (5, 2) limits -> right-arm limits under :data:`MIRROR_SIGN`."""
    lo, hi = lim_left[:, 0] * MIRROR_SIGN, lim_left[:, 1] * MIRROR_SIGN
    return np.stack([np.minimum(lo, hi), np.maximum(lo, hi)], axis=1)


# ------------------------------------------------------------------------------------------------ small math
def _rot(axis_id: int, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    if axis_id == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis_id == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_batch(axis_id: int, a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis_id == 0:
        m = [o, z, z, z, c, -s, z, s, c]
    elif axis_id == 1:
        m = [c, z, s, z, o, z, -s, z, c]
    else:
        m = [c, -s, z, s, c, z, z, z, o]
    return np.stack(m, axis=-1).reshape(np.shape(a) + (3, 3))


def so3_log(r: np.ndarray) -> np.ndarray:
    """Rotation vector of one rotation matrix (robust near 0 and pi)."""
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    c = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
    ang = math.acos(c)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    s = math.sin(ang)
    if s > 1e-6:
        return v * (ang / (2.0 * s))
    if ang < 1.0:
        return 0.5 * v
    # near pi: axis from the symmetric part
    diag = np.clip((np.diag(r) + 1.0) / 2.0, 0.0, None)
    axis = np.sqrt(diag)
    k = int(np.argmax(axis))
    for j in range(3):
        if j != k:
            axis[j] = math.copysign(axis[j], r[k, j] + r[j, k])
    return axis / np.linalg.norm(axis) * ang


def so3_log_batch(r: np.ndarray) -> np.ndarray:
    from dropbear_wbc.motion.rotations import rotation_log

    return rotation_log(r)


def pose(pos, rot=None) -> np.ndarray:
    """4x4 homogeneous transform from a position (3,) and an optional rotation (3, 3)."""
    t = np.eye(4)
    t[:3, 3] = np.asarray(pos, dtype=float)
    if rot is not None:
        t[:3, :3] = np.asarray(rot, dtype=float)
    return t


def mirror_pose(t: np.ndarray) -> np.ndarray:
    """Mirror a pose (4, 4) through the torso x-z plane (y -> -y); keeps a proper rotation."""
    m = np.diag([1.0, -1.0, 1.0])
    out = np.eye(4)
    out[:3, :3] = m @ t[:3, :3] @ m
    out[:3, 3] = m @ t[:3, 3]
    return out


# ------------------------------------------------------------------------------------------------ the chain
@dataclass
class ArmChain:
    """One arm's semantic serial chain in the root frame (see the module docstring).

    Attributes:
        side: ``"left"`` or ``"right"``.
        points: (5, 3) one point on each joint axis at the semantic zero, root frame [m].
        wrist0: (3,) wrist (end-effector) position at the semantic zero, root frame [m].
        lower, upper: (5,) semantic joint limits used by the IK [rad].
        q_rest: (5,) posture-bias target (the calibration's standing pose) [rad].
        info: provenance and model-residual diagnostics.
    """

    side: str
    points: np.ndarray
    wrist0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    q_rest: np.ndarray
    info: dict = field(default_factory=dict)
    elbow_table: tuple | None = None
    """Optional measured wrist path ``(e_grid (K,), w (K, 3))``: the wrist position (root frame, shoulder angles 0,
    wrist roll irrelevant) as a function of the semantic elbow angle, from the calibration's measured forward model.
    Replaces the rigid best-fit pivot for the wrist POSITION (the four-bar elbow is polycentric); the rotation is
    unchanged (the semantic elbow angle is the forearm rotation by definition)."""

    def _wrist_zero(self, e):
        """Wrist position at shoulder angles 0 for elbow angle(s) ``e``, and d/de (table: piecewise linear)."""
        eg, w = self.elbow_table
        e = np.asarray(e, dtype=float)
        pos = np.stack([np.interp(e, eg, w[:, k]) for k in range(3)], axis=-1)
        i = np.clip(np.searchsorted(eg, e) - 1, 0, len(eg) - 2)
        dw = (w[i + 1] - w[i]) / (eg[i + 1] - eg[i])[..., None]
        return pos, dw

    # -- single configuration (fast path used by the solver)
    def fk(self, q: np.ndarray, jacobian: bool = False):
        """Semantic q (5,) -> (wrist position (3,), EE rotation (3, 3)[, J (6, 5)]), root frame.

        ``J`` is the space Jacobian at the wrist point: rows 0-2 linear velocity, rows 3-5 angular velocity.
        """
        r = np.eye(3)
        t = np.zeros(3)
        ws, ps = [], []
        r_sh = t_sh = None
        for i in range(5):
            a = _AXIS_IDS[i]
            if i == 3:
                r_sh, t_sh = r.copy(), t.copy()
            if jacobian:
                ws.append(r[:, a].copy())
                ps.append(r @ self.points[i] + t)
            ri = _rot(a, float(q[i]))
            t = t + r @ (self.points[i] - ri @ self.points[i])
            r = r @ ri
        if self.elbow_table is None:
            pos = r @ self.wrist0 + t
        else:
            w_e, dw_e = self._wrist_zero(float(q[3]))
            pos = r_sh @ w_e + t_sh
        if not jacobian:
            return pos, r
        jac = np.zeros((6, 5))
        for i in range(5):
            jac[:3, i] = np.cross(ws[i], pos - ps[i])
            jac[3:, i] = ws[i]
        if self.elbow_table is not None:
            jac[:3, 3] = r_sh @ dw_e
            jac[:3, 4] = 0.0  # the wrist point lies on the wrist-roll axis
        return pos, r, jac

    # -- batched (tests, analysis)
    def fk_batch(self, q: np.ndarray, with_elbow: bool = False):
        """Semantic q (..., 5) -> wrist positions (..., 3) and EE rotations (..., 3, 3) [, elbow centres]."""
        q = np.asarray(q, dtype=float)
        shape = q.shape[:-1]
        r = np.broadcast_to(np.eye(3), shape + (3, 3)).copy()
        t = np.zeros(shape + (3,))
        elbow = None
        r_sh = t_sh = None
        for i in range(5):
            if i == 3:
                elbow = np.einsum("...ij,j->...i", r, self.points[3]) + t
                r_sh, t_sh = r.copy(), t.copy()
            ri = _rot_batch(_AXIS_IDS[i], q[..., i])
            t = t + np.einsum("...ij,...j->...i", r, self.points[i] - np.einsum("...ij,j->...i", ri, self.points[i]))
            r = r @ ri
        if self.elbow_table is None:
            pos = np.einsum("...ij,j->...i", r, self.wrist0) + t
        else:
            w_e, _ = self._wrist_zero(q[..., 3])
            pos = np.einsum("...ij,...j->...i", r_sh, w_e) + t_sh
        return (pos, r, elbow) if with_elbow else (pos, r)

    def clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)

    @property
    def shoulder(self) -> np.ndarray:
        return self.points[0]

    @property
    def reach(self) -> float:
        """Upper bound of the shoulder-centre -> wrist distance [m] (straight arm)."""
        s = self.points[0]
        return float(np.linalg.norm(self.points[3] - s) + np.linalg.norm(self.wrist0 - self.points[3])
                     + np.linalg.norm((self.points[2] - s)[:2]))

    @property
    def shell(self) -> tuple[float, float]:
        """Exact (min, max) shoulder-centre -> wrist distance over the joint limits [m].

        The shoulder pitch / roll axes pass through the shoulder centre and wrist roll does not move the wrist, so the
        distance depends only on (shoulder yaw, elbow): evaluated on a 73 x 65 grid of those two joints."""
        if "_shell" not in self.__dict__:
            yaw = np.linspace(self.lower[2], self.upper[2], 73)
            el = np.linspace(self.lower[3], self.upper[3], 65)
            yy, ee = np.meshgrid(yaw, el, indexing="ij")
            q = np.zeros(yy.shape + (5,))
            q[..., 2], q[..., 3] = yy, ee
            w, _ = self.fk_batch(q)
            dist = np.linalg.norm(w - self.points[0], axis=-1)
            self.__dict__["_shell"] = (float(dist.min()), float(dist.max()))
        return self.__dict__["_shell"]

    @property
    def reach_min(self) -> float:
        return self.shell[0]


# ------------------------------------------------------------------------------------------------ building
def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


HAND_BODY = {"left": "LH_shoulder_ex_al_interface_1", "right": "RH_shoulder_ex_al_interface_1"}


def measured_elbow_tables(calib_path: str | Path, calib: dict, n: int = 129) -> tuple[dict, dict]:
    """Wrist path vs elbow angle for both arms from the calibration's measured forward model
    (``kinematics.serial_model.CalibrationFK``: the raw sweep named in the calibration's provenance, interpolated
    four-bar tables). Returns ``({side: (e_grid, w_root)}, info)``. Raises if the raw sweep is unavailable."""
    from dropbear_wbc.kinematics.serial_model import CalibrationFK

    cfk = CalibrationFK(calib_path)
    stand = np.asarray(calib["standing_semantic_pos"], dtype=float)
    out = {}
    for side in SIDES:
        idx = list(ARM_SEMANTIC_INDEX[side])
        lo, hi = calib["dofs"][f"{side}_elbow"]["valid_range"]
        e = np.linspace(min(lo, hi), max(lo, hi), n)
        q22 = np.tile(stand, (n, 1))
        q22[:, idx] = 0.0
        q22[:, idx[3]] = e
        segs, used, _ = cfk.segments(q22)
        eg = used[:, idx[3]]
        if np.any(np.diff(eg) <= 0):
            raise ValueError(f"{side}: elbow grid not increasing after clipping")
        out[side] = (eg.copy(), np.asarray(segs[HAND_BODY[side]][1], dtype=float).copy())
    return out, {"source": "kinematics.serial_model.CalibrationFK", "raw_sweep": str(cfk.raw_path), "samples": n}


def build_chain(calib: dict, side: str, lower: np.ndarray, upper: np.ndarray, q_rest: np.ndarray,
                elbow_table: tuple | None = None) -> ArmChain:
    """Build one :class:`ArmChain` from a ``dropbear-semantic-calibration-v1`` dict (optionally with a measured
    elbow wrist-path table, see :attr:`ArmChain.elbow_table`)."""
    g = calib["geometry"]["per_side"][side]
    ap = g["arm_points_rest_root"]
    s = np.asarray(ap["shoulder_center"], dtype=float)
    e = np.asarray(ap["elbow_center"], dtype=float)
    w_rest = np.asarray(ap["wrist"], dtype=float)
    arm = "LH" if side == "left" else "RH"
    # shoulder-yaw (upper-arm roll) screw: point on the axis; the fitted point is the one closest to the origin
    # (root frame), its axis is +-z, so its x/y locate the axis and z is arbitrary -> put it at S's height.
    yaw_fit = calib["serial_screw_fits"][f"{arm}_roll"]
    yaw_axis = np.asarray(yaw_fit["axis_root"], dtype=float)
    if abs(abs(yaw_axis[2]) - 1.0) > 1e-3:
        raise ValueError(f"{side} shoulder-yaw axis {yaw_axis} is not vertical at the semantic zero")
    p_yaw = np.array([yaw_fit["point_root_or_parent"][0], yaw_fit["point_root_or_parent"][1], s[2]])
    if np.linalg.norm((p_yaw - s)[:2]) > 0.05:
        raise ValueError(f"{side} shoulder-yaw axis point {p_yaw} is > 5 cm from the shoulder centre {s}; "
                         "serial_screw_fits.point_root_or_parent is probably not in the root frame")
    arm_rest = calib["geometry"].get("solves", {}).get(side, {}).get("arm_rest", {})
    e_rest = math.pi / 2 - math.radians(float(arm_rest.get("upper_forearm_angle_deg", 0.0)))
    w0 = e + _rot(1, 0.0 - e_rest) @ (w_rest - e)
    points = np.stack([s, s, p_yaw, e, w0])
    chain = ArmChain(side=side, points=points, wrist0=w0, lower=np.asarray(lower, float),
                     upper=np.asarray(upper, float), q_rest=np.asarray(q_rest, float), elbow_table=elbow_table)
    # model residuals at the other calibrated wrist positions
    checks = {}
    sem_idx = ARM_SEMANTIC_INDEX[side]
    for name, key in (("semantic_zero", "zero"), ("standing", "standing")):
        wr = g.get(key, {}).get("wrist_root")
        if wr is None:
            continue
        qs = np.asarray(calib["semantic_zero_semantic_pos" if key == "zero" else "standing_semantic_pos"])[list(sem_idx)]
        pos, _ = chain.fk(qs)
        checks[name] = {"q_sem": qs.round(6).tolist(), "calibration_wrist_root": list(map(float, wr)),
                        "model_wrist_root": pos.round(6).tolist(),
                        "error_m": float(np.linalg.norm(pos - np.asarray(wr)))}
    chain.info = {
        "shoulder_center": s.tolist(), "yaw_axis_point": p_yaw.tolist(), "elbow_center": e.tolist(),
        "wrist_rest": w_rest.tolist(), "elbow_rest_rad": e_rest, "wrist0": w0.tolist(),
        "upper_arm_m": float(np.linalg.norm(e - s)), "forearm_m": float(np.linalg.norm(w_rest - e)),
        "elbow_wrist_point_pivot_rms_m": ap.get("elbow_wrist_point_pivot_rms_m"),
        "elbow_model": "measured table" if elbow_table is not None else "best-fit pivot",
        "model_vs_calibration": checks,
    }
    return chain


@dataclass
class IKWeights:
    """Cost weights. Defaults = xr_teleoperate ``R1_A5_ArmIK`` (50 translation, 0.5 rotation, 0.02 regulariser,
    0.1 smoothing); here the regulariser pulls toward the standing pose and both it and the smoothing act only in
    the null space of the position task."""

    position: float = 50.0
    rotation: float = 0.5
    posture: float = 0.02
    smooth: float = 0.1
    damping: float = 1e-6       # initial Levenberg-Marquardt damping
    nullspace_eps: float = 1e-3  # damping [m] of the position pseudo-inverse in the null-space projector
    max_step: float = 0.4       # rad per iteration (trust region)
    mode: str = "priority"      # "priority" (position > orientation > posture) or "weighted" (R1_A5 cost)
    sing_eps: float = 0.03      # [m/rad] smallest position singular value below which damping ramps up
    sing_damping: float = 0.03  # [m/rad] damping at an exact singularity (Nakamura/Chiaverini style)


@dataclass
class ArmIKResult:
    """Result for one arm (semantic q, errors at the solution, solver statistics)."""

    q: np.ndarray
    pos_err_m: float
    rot_err_rad: float
    iterations: int
    converged: bool
    at_limit: np.ndarray
    target_used: bool = True
    restarted: bool = False          # closed-form-seed restarts were tried
    restart_improved: bool = False   # ... and replaced the warm-start branch


@dataclass
class IKResult:
    """Both arms: semantic arm angles (10: left 5 then right 5), per-arm results and timing."""

    q_sem: np.ndarray
    arms: dict
    solve_ms: float


def solve_chain(chain: ArmChain, target: np.ndarray, q_init: np.ndarray, q_prev: np.ndarray | None = None,
                weights: IKWeights = IKWeights(), max_iters: int = 60, mode: str | None = None,
                step_tol: float = 1e-8) -> ArmIKResult:
    """Solve one arm. ``target`` is a 4x4 pose in the ROOT frame. Returns semantic q within the limits.

    ``mode="priority"`` (default): strict task priority, position > orientation > posture/smoothing (each lower
    task lives in the null space of the position task). ``mode="weighted"``: the single weighted
    xr_teleoperate R1_A5 cost (orientation residuals leak into position at the 100:1 weight ratio).
    """
    mode = mode or weights.mode
    if mode == "weighted":
        return solve_chain_weighted(chain, target, q_init, q_prev, weights, max_iters=max_iters)
    if mode != "priority":
        raise ValueError(f"unknown IK mode {mode!r}")
    w = weights
    lo, hi = chain.lower, chain.upper
    q = np.clip(np.asarray(q_init, dtype=float), lo, hi)
    q_prev = q.copy() if q_prev is None else np.clip(np.asarray(q_prev, dtype=float), lo, hi)
    p_t, r_t = target[:3, 3], target[:3, :3]
    lam2 = w.nullspace_eps ** 2

    def evaluate(qq, with_jac=True):
        if with_jac:
            pos, rr, jac = chain.fk(qq, jacobian=True)
        else:
            (pos, rr), jac = chain.fk(qq), None
        e_p = p_t - pos
        e_r = so3_log(r_t @ rr.T)
        d_rest, d_prev = qq - chain.q_rest, qq - q_prev
        sec = w.rotation * e_r @ e_r + w.posture * d_rest @ d_rest + w.smooth * d_prev @ d_prev
        return e_p, e_r, jac, float(np.linalg.norm(e_p)), float(sec)

    e_p, e_r, jac, perr, sec = evaluate(q)
    pos_tol = 1e-5
    it, converged = 0, False
    for it in range(1, max_iters + 1):
        jv, jw = jac[:3], jac[3:]
        free = np.ones(5, dtype=bool)
        for _as in range(6):  # active set: freeze joints pushed against a bound
            jvf, jwf = jv[:, free], jw[:, free]
            nf = int(free.sum())
            u, sv, vt = np.linalg.svd(jvf, full_matrices=True)
            k = min(3, nf)
            # stage 1: damped least squares on the position error (SVD form), singularity-robust damping:
            # lambda^2 grows smoothly as the smallest singular value drops below sing_eps (stretched arm)
            smin = float(sv[k - 1]) if k else 0.0
            lam2_eff = lam2 + (w.sing_damping ** 2) * max(0.0, 1.0 - (smin / w.sing_eps) ** 2)
            dq1 = vt[:k].T @ ((sv[:k] / (sv[:k] ** 2 + lam2_eff)) * (u[:, :k].T @ e_p))
            dqf = dq1
            if nf > k:  # stage 2: orientation, posture and smoothing in the exact null space of the position task
                zb = vt[k:].T                          # (nf, nf - 3) orthonormal null-space basis
                qf, rest_f, prev_f = q[free], chain.q_rest[free], q_prev[free]
                h2 = w.rotation * jwf.T @ jwf + (w.posture + w.smooth) * np.eye(nf)
                g2 = (w.rotation * jwf.T @ (e_r - jwf @ dq1) - w.posture * (qf + dq1 - rest_f)
                      - w.smooth * (qf + dq1 - prev_f))
                y = np.linalg.solve(zb.T @ h2 @ zb + 1e-12 * np.eye(nf - k), zb.T @ g2)
                dqf = dq1 + zb @ y
            dq = np.zeros(5)
            dq[free] = dqf
            blocked = free & (((q <= lo + 1e-12) & (dq < 0)) | ((q >= hi - 1e-12) & (dq > 0)))
            if not blocked.any():
                break
            free &= ~blocked
            if not free.any():
                dq = np.zeros(5)
                break
        n = float(np.max(np.abs(dq))) if dq.any() else 0.0
        if n < step_tol:
            converged = True
            break
        if n > w.max_step:
            dq *= w.max_step / n
        accepted = False
        alpha = 1.0
        for _ls in range(12):  # backtracking, lexicographic acceptance (position first)
            q_new = np.clip(q + alpha * dq, lo, hi)
            e_p_n, e_r_n, _, perr_n, sec_n = evaluate(q_new, with_jac=False)
            if perr_n < perr - 1e-12 or (perr_n <= max(perr, pos_tol) + 1e-9 and sec_n < sec - 1e-14):
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            converged = True
            break
        step = float(np.max(np.abs(q_new - q)))
        q, e_p, e_r, perr, sec = q_new, e_p_n, e_r_n, perr_n, sec_n
        if step >= step_tol and it < max_iters:
            jac = chain.fk(q, jacobian=True)[2]
        if step < step_tol:
            converged = True
            break
    at_limit = (q <= lo + 1e-9) | (q >= hi - 1e-9)
    return ArmIKResult(q=q, pos_err_m=perr, rot_err_rad=float(np.linalg.norm(e_r)), iterations=it,
                       converged=converged, at_limit=at_limit)


def solve_chain_weighted(chain: ArmChain, target: np.ndarray, q_init: np.ndarray, q_prev: np.ndarray | None = None,
                         weights: IKWeights = IKWeights(), max_iters: int = 60, tol: float = 1e-10) -> ArmIKResult:
    """The xr_teleoperate R1_A5 single weighted cost (posture / smoothing projected into the position null space),
    solved by bounded Levenberg-Marquardt. Kept for comparison (``mode="weighted"``)."""
    w = weights
    lo, hi = chain.lower, chain.upper
    q = np.clip(np.asarray(q_init, dtype=float), lo, hi)
    q_prev = q.copy() if q_prev is None else np.clip(np.asarray(q_prev, dtype=float), lo, hi)
    p_t, r_t = target[:3, 3], target[:3, :3]
    eye5 = np.eye(5)
    lam = w.damping

    def terms(qq):
        pos, rr, jac = chain.fk(qq, jacobian=True)
        return pos, rr, jac, p_t - pos, so3_log(r_t @ rr.T)

    pos, rr, jac, e_p, e_r = terms(q)
    it = 0
    converged = False
    for it in range(1, max_iters + 1):
        jv, jw = jac[:3], jac[3:]
        nproj = eye5 - jv.T @ np.linalg.solve(jv @ jv.T + (w.nullspace_eps ** 2) * np.eye(3), jv)
        nn = nproj.T @ nproj
        d_rest, d_prev = q - chain.q_rest, q - q_prev
        h = w.position * jv.T @ jv + w.rotation * jw.T @ jw + (w.posture + w.smooth) * nn
        g = w.position * jv.T @ e_p + w.rotation * jw.T @ e_r - nn @ (w.posture * d_rest + w.smooth * d_prev)

        def merit(e_p_, e_r_, q_):
            dr, dp = q_ - chain.q_rest, q_ - q_prev
            return (w.position * e_p_ @ e_p_ + w.rotation * e_r_ @ e_r_
                    + w.posture * dr @ nn @ dr + w.smooth * dp @ nn @ dp)

        f0 = merit(e_p, e_r, q)
        accepted = False
        for _ in range(8):
            free = np.ones(5, dtype=bool)
            dq = np.zeros(5)
            for _as in range(6):  # active set: freeze joints pushed against a bound
                hf = h[np.ix_(free, free)] + lam * (1.0 + np.diag(h)[free]) * np.eye(int(free.sum()))
                dq = np.zeros(5)
                dq[free] = np.linalg.solve(hf, g[free])
                blocked = free & (((q <= lo + 1e-12) & (dq < 0)) | ((q >= hi - 1e-12) & (dq > 0)))
                if not blocked.any():
                    break
                free &= ~blocked
                if not free.any():
                    break
            n = float(np.max(np.abs(dq))) if dq.size else 0.0
            if n > w.max_step:
                dq *= w.max_step / n
            q_new = np.clip(q + dq, lo, hi)
            pos_n, rr_n, jac_n, e_p_n, e_r_n = terms(q_new)
            f1 = merit(e_p_n, e_r_n, q_new)
            if f1 <= f0:
                accepted = True
                step = float(np.max(np.abs(q_new - q)))
                q, pos, rr, jac, e_p, e_r = q_new, pos_n, rr_n, jac_n, e_p_n, e_r_n
                lam = max(lam / 3.0, 1e-9)
                break
            lam *= 10.0
        if not accepted or step < 1e-9 or f0 - merit(e_p, e_r, q) < tol * max(1.0, f0):
            converged = True
            break
    at_limit = (q <= lo + 1e-9) | (q >= hi - 1e-9)
    return ArmIKResult(q=q, pos_err_m=float(np.linalg.norm(e_p)), rot_err_rad=float(np.linalg.norm(e_r)),
                       iterations=it, converged=converged, at_limit=at_limit)


def analytic_seed(chain: ArmChain, target_pos_root: np.ndarray, swivel: float = 0.0, branch: int = 0) -> np.ndarray:
    """Closed-form initial guess for a wrist position (root frame).

    The elbow comes from the shoulder-centre -> target distance (the table ``|W(e) - S|`` at zero shoulder
    angles, which is monotonic over the elbow range); the shoulder is the minimal rotation taking the
    zero-shoulder arm direction onto the target direction, then turned by ``swivel`` about that direction,
    decomposed as YXZ Euler (pitch, roll, yaw) on ``branch`` 0 (|roll| <= pi/2) or 1 (the other branch).
    Wrist roll = rest value. Used to escape the straight-elbow local minimum of a cold start.
    """
    from dropbear_wbc.kinematics.semantic import euler_yxz

    s = chain.points[0]
    v = np.asarray(target_pos_root, dtype=float) - s
    d = float(np.linalg.norm(v))
    lo, hi = chain.lower[3], chain.upper[3]
    es = np.linspace(lo, hi, 64)
    q = np.zeros((64, 5))
    q[:, 3] = es
    w, _ = chain.fk_batch(q)
    dist = np.linalg.norm(w - s, axis=-1)
    order = np.argsort(dist)
    e = float(np.interp(d, dist[order], es[order]))
    q0 = np.zeros(5)
    q0[3] = e
    q0[4] = chain.q_rest[4]
    u0 = chain.fk(q0)[0] - s
    u0 /= max(np.linalg.norm(u0), 1e-12)
    vt = v / max(d, 1e-12)
    ax = np.cross(u0, vt)
    sa, ca = np.linalg.norm(ax), float(np.clip(u0 @ vt, -1.0, 1.0))
    if sa < 1e-9:
        r = np.eye(3) if ca > 0 else _rot(1, math.pi)
    else:
        k = ax / sa
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        ang = math.atan2(sa, ca)
        r = np.eye(3) + math.sin(ang) * kx + (1 - math.cos(ang)) * kx @ kx
    if swivel:
        kx = np.array([[0, -vt[2], vt[1]], [vt[2], 0, -vt[0]], [-vt[1], vt[0], 0]])
        r = (np.eye(3) + math.sin(swivel) * kx + (1 - math.cos(swivel)) * kx @ kx) @ r
    e = euler_yxz(r)
    if branch:  # the other YXZ branch (roll beyond +-pi/2: arm raised above the shoulder)
        e = np.array([e[0] + math.pi, math.pi - e[1], e[2] + math.pi])
        e = (e + math.pi) % (2 * math.pi) - math.pi
    q0[:3] = e
    return chain.clip(q0)


def solve_chain_robust(chain: ArmChain, target: np.ndarray, q_init: np.ndarray, q_prev: np.ndarray | None = None,
                       weights: IKWeights = IKWeights(), max_iters: int = 60, restart_above_m: float = 0.003,
                       restarts: str = "full") -> ArmIKResult:
    """:func:`solve_chain` from the warm start; if the wrist error stays above ``restart_above_m`` and the target
    is within reach, retry from analytic seeds (swivel 0, +-0.8, +-1.6, 3.0 rad; both YXZ branches). A restart
    replaces the warm-start branch only if it at least halves the wrist error and gains > 0.5 mm (continuity).
    Typical trap it fixes: a near-straight arm whose target moves closer (the elbow sticks at the straight limit,
    where d|wrist - shoulder| / d elbow ~ 0).
    ``restarts``: ``"full"`` (12 seeds), ``"light"`` (2 seeds: swivel 0, both branches; real-time loop) or ``"none"``."""
    res = solve_chain(chain, target, q_init, q_prev, weights, max_iters=max_iters)
    if res.pos_err_m <= restart_above_m or restarts == "none":
        return res
    d = float(np.linalg.norm(target[:3, 3] - chain.points[0]))
    lo_d, hi_d = chain.shell
    if d > hi_d + 1e-3 or d < lo_d - 1e-3:  # outside the reachable shell: seeds cannot help
        return res
    best = res
    best.restarted = True
    iters = res.iterations
    swivels = (0.0,) if restarts == "light" else (0.0, 0.8, -0.8, 1.6, -1.6, 3.0)
    for sw in swivels:
        for br in (0, 1):
            seed = analytic_seed(chain, target[:3, 3], swivel=sw, branch=br)
            r = solve_chain(chain, target, seed, q_prev, weights, max_iters=max_iters)
            iters += r.iterations
            # switch branch only for a clear improvement: at least halve the error and gain > 0.5 mm (continuity)
            if r.pos_err_m < 0.5 * best.pos_err_m and best.pos_err_m - r.pos_err_m > 5e-4:
                best = r
                best.restarted = best.restart_improved = True
            if best.pos_err_m <= restart_above_m:
                best.iterations = iters
                return best
    best.iterations = iters
    return best


# ------------------------------------------------------------------------------------------------ public API
class DropbearArmIK:
    """Two-arm IK in the semantic space with warm start, reload and the motor-space conversion.

    Typical use::

        ik = DropbearArmIK()                               # default calibration
        res = ik.solve(left_T, right_T)                    # 4x4 wrist targets, torso frame
        motor10, sat = ik.to_motor(res.q_sem)              # arm motor targets, SDK slots 12..21
        ik.reload_if_changed()                             # pick up a re-run calibration

    Args:
        calibration: path of the semantic calibration JSON (default ``$DROPBEAR_CALIBRATION_JSON`` or
            ``data/calibration/dropbear_semantic_calibration.json``).
        weights: :class:`IKWeights`.
        soft_limits_left: (5, 2) left-arm teleop limits (mirrored for the right arm), intersected with the
            calibration's semantic valid ranges. ``None`` = :data:`DEFAULT_SOFT_LIMITS_LEFT`.
    """

    def __init__(self, calibration: str | Path | None = None, weights: IKWeights | None = None,
                 soft_limits_left: np.ndarray | None = None, elbow_model: str = "table"):
        import os

        self.path = Path(calibration or os.environ.get("DROPBEAR_CALIBRATION_JSON") or DEFAULT_CALIBRATION)
        self.weights = weights or IKWeights()
        self.soft_limits_left = np.array(DEFAULT_SOFT_LIMITS_LEFT if soft_limits_left is None else soft_limits_left,
                                         dtype=float)
        if elbow_model not in ("table", "pivot"):
            raise ValueError(f"elbow_model must be 'table' or 'pivot', got {elbow_model!r}")
        self.elbow_model = elbow_model
        self.q_prev: dict[str, np.ndarray] | None = None
        self._restart_failed_at: dict[str, np.ndarray] = {}
        self._load()

    # -- loading
    def _load(self) -> None:
        raw = self.path.read_bytes()
        calib = json.loads(raw)
        self.smap = SemanticMap(calib)
        self.calib = calib
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.mtime = self.path.stat().st_mtime
        self.standing_semantic = np.asarray(calib["standing_semantic_pos"], dtype=float)
        self.standing_motor = np.asarray(calib["standing_motor_pos"], dtype=float)
        self.torso_origin_root = np.asarray(calib["rest_transforms"]["pelvis_in_root"]["pos"], dtype=float)
        soft = {"left": self.soft_limits_left, "right": mirror_limits(self.soft_limits_left)}
        tables, self.elbow_model_info = {}, {"requested": self.elbow_model, "used": "pivot"}
        if self.elbow_model == "table":
            try:
                tables, tinfo = measured_elbow_tables(self.path, calib)
                self.elbow_model_info.update(used="table", **tinfo)
            except Exception as e:  # noqa: BLE001 - raw sweep missing / stale: fall back to the pivot model
                self.elbow_model_info["fallback_reason"] = repr(e)
        self.chains: dict[str, ArmChain] = {}
        for side in SIDES:
            idx = list(ARM_SEMANTIC_INDEX[side])
            cal = self.smap.semantic_limits[idx]
            lo = np.maximum(cal[:, 0], soft[side][:, 0])
            hi = np.minimum(cal[:, 1], soft[side][:, 1])
            if np.any(lo > hi):
                raise ValueError(f"{side}: empty IK joint range (calibration {cal.tolist()}, soft {soft[side].tolist()})")
            q_rest = np.clip(self.standing_semantic[idx], lo, hi)
            self.chains[side] = build_chain(calib, side, lo, hi, q_rest, elbow_table=tables.get(side))
        if self.q_prev is not None:  # keep the warm start across reloads, inside the new limits
            self.q_prev = {s: self.chains[s].clip(self.q_prev[s]) for s in SIDES}

    def reload_if_changed(self) -> bool:
        """Reload when the calibration file changed on disk (mtime and SHA-256). Returns True if reloaded."""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return False
        if mtime == self.mtime:
            return False
        if _sha256(self.path) == self.sha256:
            self.mtime = mtime
            return False
        self._load()
        return True

    def info(self) -> dict:
        return {"calibration": str(self.path), "calibration_sha256": self.sha256,
                "calibration_created": self.calib.get("created"), "plant_variant": self.calib.get("plant_variant"),
                "torso_origin_root": self.torso_origin_root.tolist(), "elbow_model": self.elbow_model_info,
                "weights": self.weights.__dict__,
                "limits": {s: np.stack([c.lower, c.upper], 1).round(6).tolist() for s, c in self.chains.items()},
                "q_rest": {s: c.q_rest.round(6).tolist() for s, c in self.chains.items()},
                "chains": {s: c.info for s, c in self.chains.items()}}

    # -- frames
    def torso_to_root(self, t: np.ndarray) -> np.ndarray:
        out = np.array(t, dtype=float, copy=True)
        out[:3, 3] = out[:3, 3] + self.torso_origin_root
        return out

    def root_to_torso(self, t: np.ndarray) -> np.ndarray:
        out = np.array(t, dtype=float, copy=True)
        out[:3, 3] = out[:3, 3] - self.torso_origin_root
        return out

    def fk(self, side: str, q5: np.ndarray) -> np.ndarray:
        """Wrist pose (4x4) in the torso frame for one arm's semantic q (5,)."""
        pos, rot = self.chains[side].fk(np.asarray(q5, dtype=float))
        return pose(pos - self.torso_origin_root, rot)

    def fk_both(self, q_sem10: np.ndarray) -> dict[str, np.ndarray]:
        q = np.asarray(q_sem10, dtype=float)
        return {"left": self.fk("left", q[:5]), "right": self.fk("right", q[5:])}

    def rest_q(self) -> np.ndarray:
        return np.concatenate([self.chains["left"].q_rest, self.chains["right"].q_rest])

    # -- solve
    def reset(self, q_sem10: np.ndarray | None = None) -> None:
        """Set the warm start (default: the standing pose)."""
        q = self.rest_q() if q_sem10 is None else np.asarray(q_sem10, dtype=float)
        self.q_prev = {"left": self.chains["left"].clip(q[:5]), "right": self.chains["right"].clip(q[5:])}

    def solve(self, left: np.ndarray | None, right: np.ndarray | None, q_init: np.ndarray | None = None,
              frame: str = "torso", max_iters: int = 60, restarts: str = "full",
              restart_above_m: float = 0.003) -> IKResult:
        """Solve both arms. ``left`` / ``right`` are 4x4 wrist targets (``None`` holds that arm).

        Args:
            q_init: optional (10,) warm start (e.g. the measured semantic arm pose, as xr_teleoperate passes the
                current motor q); default is the previous solution (or the standing pose on the first call).
            frame: ``"torso"`` (default) or ``"root"``.
        """
        t0 = time.perf_counter()
        if self.q_prev is None:
            self.reset()
        arms = {}
        for side, tgt in (("left", left), ("right", right)):
            ch = self.chains[side]
            prev = self.q_prev[side]
            init = prev if q_init is None else np.asarray(q_init, dtype=float)[:5] if side == "left" else \
                np.asarray(q_init, dtype=float)[5:]
            if tgt is None:
                pos, rot = ch.fk(prev)
                arms[side] = ArmIKResult(q=prev.copy(), pos_err_m=0.0, rot_err_rad=0.0, iterations=0, converged=True,
                                         at_limit=(prev <= ch.lower + 1e-9) | (prev >= ch.upper - 1e-9),
                                         target_used=False)
                continue
            t_root = self.torso_to_root(tgt) if frame == "torso" else np.asarray(tgt, dtype=float)
            # after a restart that did not help, do not retry until the target has moved >= 1 cm (unreachable
            # targets would otherwise pay for the seeds every control step)
            rs = restarts
            failed_at = self._restart_failed_at.get(side)
            if rs != "none" and failed_at is not None and np.linalg.norm(t_root[:3, 3] - failed_at) < 0.01:
                rs = "none"
            res = solve_chain_robust(ch, t_root, init, prev, self.weights, max_iters=max_iters, restarts=rs,
                                     restart_above_m=restart_above_m)
            if res.restarted and not res.restart_improved:
                self._restart_failed_at[side] = t_root[:3, 3].copy()
            elif res.pos_err_m <= restart_above_m:
                self._restart_failed_at.pop(side, None)
            arms[side] = res
            self.q_prev[side] = res.q.copy()
        q = np.concatenate([arms["left"].q, arms["right"].q])
        return IKResult(q_sem=q, arms=arms, solve_ms=1e3 * (time.perf_counter() - t0))

    # -- motor space
    def to_motor(self, q_sem10: np.ndarray):
        """Semantic arm angles (10,) -> (arm motor targets (10,) for SDK slots 12..21, saturation summary).

        Goes through ``SemanticMap.semantic_to_motor`` on the full 22-vector (legs at the standing semantic pose,
        whose motor values are not used here)."""
        s = self.standing_semantic.copy()
        q = np.asarray(q_sem10, dtype=float)
        for k, side in enumerate(SIDES):
            s[list(ARM_SEMANTIC_INDEX[side])] = q[5 * k:5 * k + 5]
        m, rep = self.smap.semantic_to_motor(s, return_report=True)
        arm_names = set(ARM_SEMANTIC_NAMES["left"] + ARM_SEMANTIC_NAMES["right"])
        sat = {k: v for k, v in rep.summary().items() if k in arm_names}
        return m[list(ARM_MOTOR_SLOTS)], sat

    def to_motor_fast(self, q_sem10: np.ndarray):
        """Same result as :meth:`to_motor` (tested equal), but only evaluates the arm entries of the
        ``SemanticMap`` (shoulder ``serial3`` groups, elbow ``lut1d``, wrist ``linear``): about 10x cheaper because
        the leg maps (hip ``serial3`` Newton, ankle ``lut2d`` Newton) are skipped. Used by the control loop."""
        smap = self.smap
        q = np.asarray(q_sem10, dtype=float)
        s = self.standing_semantic.copy()
        idx = list(ARM_SEMANTIC_INDEX["left"] + ARM_SEMANTIC_INDEX["right"])
        s[idx] = q
        used = np.clip(s, smap.semantic_limits[:, 0], smap.semantic_limits[:, 1])
        out = np.zeros(len(MOTOR_NAMES))
        arm = set(idx)
        groups = [g for g in smap._serial if set(g.sem) <= arm]
        for i, obj in smap._single:
            if i in arm:
                out[obj.m] = obj.inv(used[i])
        for g in groups:
            out[g.chain] = g.inv(used[g.sem])
        out = np.clip(out, smap.motor_limits[:, 0], smap.motor_limits[:, 1])
        for g in groups:
            used[g.sem] = g.fwd(out)
        clipped = np.abs(s - used)[idx] > 1e-6
        names = ARM_SEMANTIC_NAMES["left"] + ARM_SEMANTIC_NAMES["right"]
        sat = {n: {"fraction": 1.0, "max_excess_rad": float(abs(s[i] - used[i]))}
               for n, i, c in zip(names, idx, clipped) if c}
        return out[list(ARM_MOTOR_SLOTS)], sat

    def motor_to_semantic_arms_fast(self, motor_q22: np.ndarray) -> np.ndarray:
        """Same as :meth:`motor_to_semantic_arms` (tested equal) but evaluates only the arm entries of the map."""
        smap = self.smap
        q = np.asarray(motor_q22, dtype=float)
        m = np.clip(q, smap.motor_limits[:, 0], smap.motor_limits[:, 1])
        idx = list(ARM_SEMANTIC_INDEX["left"] + ARM_SEMANTIC_INDEX["right"])
        arm = set(idx)
        out = np.zeros(q.shape)
        for i, obj in smap._single:
            if i in arm:
                out[..., i] = obj.fwd(m)
        for g in smap._serial:
            if set(g.sem) <= arm:
                out[..., g.sem] = g.fwd(m)
        return out[..., idx]

    def motor_to_semantic_arms(self, motor_q22: np.ndarray) -> np.ndarray:
        """Measured 22 motor angles -> semantic arm angles (10,) (left 5, right 5)."""
        sem = self.smap.motor_to_semantic(np.asarray(motor_q22, dtype=float))
        return sem[..., list(ARM_SEMANTIC_INDEX["left"] + ARM_SEMANTIC_INDEX["right"])]

    def motor22(self, arm_motor10: np.ndarray, legs_motor: np.ndarray | None = None) -> np.ndarray:
        """Full 22 motor vector: legs at ``legs_motor`` (default the standing pose), arms from ``arm_motor10``."""
        out = (self.standing_motor if legs_motor is None else np.asarray(legs_motor, float)).copy()
        out[list(ARM_MOTOR_SLOTS)] = arm_motor10
        return out


def mirrored_chain(chain: ArmChain, side: str, y_mid: float) -> ArmChain:
    """The mirror image of ``chain`` through the plane ``y = y_mid`` (root frame) as an arm of ``side``.

    Used by the symmetry tests to separate solver symmetry from calibration asymmetry."""
    def m(p):
        p = np.array(p, dtype=float, copy=True)
        p[..., 1] = 2.0 * y_mid - p[..., 1]
        return p
    lim = np.stack([chain.lower, chain.upper], 1)
    lo_hi = mirror_limits(lim)
    table = None if chain.elbow_table is None else (chain.elbow_table[0].copy(), m(chain.elbow_table[1]))
    return replace(chain, side=side, points=m(chain.points), wrist0=m(chain.wrist0), lower=lo_hi[:, 0],
                   upper=lo_hi[:, 1], q_rest=chain.q_rest * MIRROR_SIGN, info={"mirrored_from": chain.side},
                   elbow_table=table)
