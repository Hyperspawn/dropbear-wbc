"""Foot-contact-aware stance-leg IK stage (post-retarget; shared by the G1-intermediate and the GMR route).

Why: both retarget routes copy *joint angles* (G1 route) or solve IK on a *derived serial surrogate* (GMR route); neither
puts Dropbear's real feet where the source feet were. With Dropbear's short knee range (flexion ~0-48 deg) and the
serial/skeleton models' foot errors (up to 47 mm / 112 mm, ``logs/foot_contact/foot_fk_accuracy.json``) stance feet
of whole-body clips floated 3-14 cm and slid. This stage fixes the legs and the pelvis height so that every stance foot
is flat on the ground at a world-locked pose, using the most accurate leg FK available.

Input and output live in the SEMANTIC space (``dropbear-semantic-v1``): pelvis pose + 22 semantic angles + source
contacts. Internally the legs are parametrised by their 6 MOTORS each (hip roll/yaw/pitch motors, knee crank, the
two calf motors), because

* the plant forward model is a function of the motors (``calib_fit.MeasuredFK``: fitted hip screws, the knee
  four-bar sweep table, the calf-motor grid; physics error <= 3.2 mm, sole z <= 2.2 mm, see
  ``tools/foot_fk_accuracy.py``), and
* the feasible set is exactly the motor box (USD motor limits intersected with the calibration's valid tables);
  semantic hip/ankle ranges are only bounding boxes of it.

The cost's "stay close to the retargeted pose" term is measured in semantic angles (``SemanticMap.motor_to_semantic``),
and the result is returned as semantic angles (legs replaced, arms untouched) plus the leg motors that realise them.

Algorithm (all frames vectorised, units m / rad / s):

1. **Contact phases** from the source contacts (the route supplies them, ideally with hysteresis:
   :func:`dropbear_wbc.motion.contacts.detect_contacts_hysteresis`), cleaned with minimum stance / swing durations.
2. **Baseline**: the retargeted pose's plant soles (MeasuredFK + sole hull) are grounded -- per frame the lowest
   contact-foot sole to z = 0, interpolated through flight, Gaussian-smoothed (``settle.ground.ground_correction``).
3. **Targets per foot**: per stance phase a world-locked, flat sole pose -- the sole centre's xy from the phase's first
   frame, on z = 0, yaw = circular mean of the source foot yaw over the phase. Swing frames follow the retargeted foot
   plus an offset that blends (min-jerk) from the end-of-previous-stance offset to the start-of-next-stance offset (so
   every swing starts and ends exactly at its stance targets), lifted to keep the lowest sole point >= ``clearance``
   (ramped in over ``clearance_ramp_s`` next to stance).
4. **Per-frame damped Gauss-Newton** (box-projected) over 17 variables: pelvis correction (dx, dy, dz, pitch, roll) and
   12 leg motors. Residuals: stance foot pose (position of the sole centre; tilt; yaw -- the stiffest terms), swing
   foot pose (soft), swing clearance (hinge), leg semantic angles vs. the retargeted pose, and the pelvis correction vs
   a prior. Pass A: prior 0. Pass B: prior = pass-A correction interpolated over flight and smoothed. The correction is
   then smoothed once more and Pass C re-solves the legs only (pelvis fixed), which re-establishes the stance
   constraints exactly wherever the legs can reach.
5. Reports: stance residuals, clearance, pelvis correction, legs at the motor box, motor speed / step (the
   validator's 10 rad/s and 0.3 rad/frame gates).

The stage never touches the arms; their motor speeds remain the route's responsibility (serial continuity).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dropbear_wbc.kinematics.calib_fit import seg_motors

from .contacts import contact_segments
from .names import MOTOR_NAMES, SEMANTIC_INDEX
from .rotations import matrix_to_quat, quat_continuous, quat_to_matrix, rot_x, rot_y, rot_z

__all__ = ["STAGE_VERSION", "FootContactParams", "PlantLegModel", "FootStageResult", "clean_contacts",
           "foot_contact_stage", "resample_semantic", "leg_motor_columns", "get_calibration_fk"]

STAGE_VERSION = "foot_contact-1"
SIDES = ("left", "right")
LEG_SEM = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
LEG_MOTOR_ROLES = ("hip_roll", "hip_yaw", "hip_pitch", "knee", "calf_a", "calf_b")
REPO = Path(__file__).resolve().parents[3]
SOLE_HULLS = REPO / "data/calibration/dropbear_foot_sole_hulls.json"


_CFK_CACHE: dict[str, Any] = {}


def get_calibration_fk(path: str | Path | None = None):
    """Cached :class:`~dropbear_wbc.kinematics.serial_model.CalibrationFK` (loads the raw sweep once per process)."""
    from dropbear_wbc.kinematics.serial_model import DEFAULT_CALIBRATION, CalibrationFK

    key = str(Path(path or DEFAULT_CALIBRATION).resolve())
    if key not in _CFK_CACHE:
        _CFK_CACHE[key] = CalibrationFK(key)
    return _CFK_CACHE[key]


def leg_motor_columns() -> np.ndarray:
    """Motor-contract column indices of the 12 leg motors, [left LEG_MOTOR_ROLES, right LEG_MOTOR_ROLES]."""
    return np.array([MOTOR_NAMES.index(seg_motors(s)[r]) for s in SIDES for r in LEG_MOTOR_ROLES])


@dataclass(frozen=True)
class FootContactParams:
    """Tunables of :func:`foot_contact_stage` (recorded in every sidecar)."""

    # contact phases
    min_stance_s: float = 0.10  # shorter stance runs become swing
    min_swing_s: float = 0.08  # shorter swing gaps are filled (foot stays planted)
    # targets
    yaw_source: str = "source"  # 'source' (route-supplied source foot yaw) | 'dropbear' (retargeted foot yaw)
    clearance_m: float = 0.010  # min lowest-sole height of a swing foot
    clearance_ramp_s: float = 0.06  # clearance ramps from 0 at stance to clearance_m over this time
    ground_sigma_s: float = 0.10  # baseline grounding smoothing
    # residual scales (a residual equal to the scale costs 1)
    stance_pos_m: float = 0.0005
    stance_tilt_rad: float = 0.004
    stance_yaw_rad: float = 0.02
    swing_pos_m: float = 0.02
    swing_rot_rad: float = 0.10
    clearance_scale_m: float = 0.001
    sem_rad: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2)  # hip p/r/y, knee, ankle p/r
    root_xy_m: float = 0.01
    root_z_m: float = 0.05
    root_tilt_rad: float = 0.03
    landing_ramp_s: float = 0.10  # swing-foot weights ramp from the stance weights over this time
    yaw_margin_deg: float = 3.0  # stance yaw targets stay inside the hip-yaw range minus this margin
    # minimum lateral sole-centre separation in the pelvis heading frame (0 = off). Dropbear's pelvis is much narrower
    # than G1's / a human's (stance width 0.138 m), so G1-route walks land the feet on top of each other (kimodo walk:
    # feet < 5 cm apart in 66 % of frames); the retargeted sole paths are widened symmetrically before the targets
    min_stance_width_m: float = 0.10
    # temporal terms (banded least squares): second-difference scales per frame, motor-step hinge
    acc_root_xy_m: float = 0.002
    acc_root_z_m: float = 0.001
    acc_root_tilt_rad: float = 0.005
    acc_motor_rad: float = 0.02
    max_motor_step_rad: float = 0.18  # 9 rad/s at 50 Hz (validator gate: 10 rad/s, 0.3 rad/frame)
    step_hinge_scale_rad: float = 0.002
    # solver
    iters: int = 30
    # quality thresholds (report only)
    stance_ok_pos_m: float = 0.003
    stance_ok_tilt_deg: float = 1.0


# ------------------------------------------------------------------------------------------------------------------
def _vee_skew(a: np.ndarray) -> np.ndarray:
    """0.5 * vee(A - A^T) (..., 3): sin(angle) * axis of a rotation matrix A (small-angle rotation vector)."""
    return 0.5 * np.stack([a[..., 2, 1] - a[..., 1, 2], a[..., 0, 2] - a[..., 2, 0], a[..., 1, 0] - a[..., 0, 1]], -1)


def _rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    ang = np.linalg.norm(v, axis=-1)
    axis = np.where(ang[..., None] > 1e-12, v / np.maximum(ang[..., None], 1e-12), np.array([0.0, 0.0, 1.0]))
    k = np.zeros(v.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2], k[..., 1, 2] = -axis[..., 2], axis[..., 1], -axis[..., 0]
    k[..., 1, 0], k[..., 2, 0], k[..., 2, 1] = axis[..., 2], -axis[..., 1], axis[..., 0]
    s, c = np.sin(ang)[..., None, None], np.cos(ang)[..., None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def _matrix_to_rotvec(m: np.ndarray) -> np.ndarray:
    from .rotations import rotation_log

    return rotation_log(m)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)


def _moving_max(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k |= 1
    p = k // 2
    xp = np.pad(x, (p, p), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(xp, k).max(axis=1)


def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k |= 1
    p = k // 2
    xp = np.pad(x, (p, p), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def _gauss(x: np.ndarray, sigma_frames: float) -> np.ndarray:
    from dropbear_wbc.settle.ground import gaussian_smooth

    if x.ndim == 1:
        return gaussian_smooth(x, sigma_frames)
    return np.stack([gaussian_smooth(x[:, k], sigma_frames) for k in range(x.shape[1])], axis=1)


def clean_contacts(contacts: np.ndarray, fps: float, min_stance_s: float, min_swing_s: float) -> np.ndarray:
    """Fill swing gaps shorter than ``min_swing_s`` (between two stances), then drop stance runs shorter than
    ``min_stance_s`` (boundary runs included). (T, 2) bool."""
    out = np.asarray(contacts, dtype=bool).copy()
    n_sw = max(1, int(round(min_swing_s * fps)))
    n_st = max(1, int(round(min_stance_s * fps)))
    for s in range(out.shape[1]):
        c = out[:, s]
        for a, b in contact_segments(~c):
            if b - a < n_sw and a > 0 and b < len(c):
                c[a:b] = True
        for a, b in contact_segments(c):
            if b - a < n_st:
                c[a:b] = False
        out[:, s] = c
    return out


def plant_gate_metrics(cfk, motor_q: np.ndarray, root_pos: np.ndarray, root_quat_wxyz: np.ndarray,
                       contacts: np.ndarray | None, fps: float, max_contact_z: float = 0.015, max_vel: float = 10.0,
                       max_step: float = 0.3, sole_hulls: str | Path = SOLE_HULLS,
                       settle_ground_sigma_s: float | None = None) -> dict[str, Any]:
    """Kinematic prediction of the validator gates (``tools/validate_motion_npz.py``) on a retargeted clip, BEFORE the
    settle: contact-foot lowest sole z (plant forward model + the full sole hulls of the sole plate), contact-foot
    slip (foot-link origin), lowest sole of any foot, motor reference speed (central differences) and step.
    ``contacts`` None = the lower foot (the settle's rule). ``settle_ground_sigma_s`` first applies the settle's own
    ground fix (``settle.ground.ground_correction``: per-frame contact-foot lowest sole to z = 0, Gaussian-smoothed with
    this sigma; ``tools/settle_motion.py --ground-sigma`` default 0.1 s), which is what the settle does to a clip whose
    root height is only approximate."""
    from dropbear_wbc.kinematics.calib_fit import seg_bodies

    soles = json.loads(Path(sole_hulls).read_text())
    r_root = quat_to_matrix(root_quat_wxyz)
    t = len(motor_q)
    low = np.zeros((t, 2))
    orig = np.zeros((t, 2, 3))
    for k, s in enumerate(SIDES):
        v = np.asarray(soles["feet"][seg_bodies(s)["foot"]]["vertices_b"], dtype=np.float64)
        r_f, p_f = cfk.fk.leg(s, motor_q)["foot"]
        rw = r_root @ r_f
        pw = root_pos + np.einsum("tij,tj->ti", r_root, p_f)
        low[:, k] = (np.einsum("tj,vj->tv", rw[:, 2, :], v) + pw[:, 2:3]).min(axis=1)
        orig[:, k] = pw
    c = np.stack([low[:, 0] <= low[:, 1], low[:, 1] < low[:, 0]], -1) if contacts is None else np.asarray(contacts, bool)
    dz_info = None
    if settle_ground_sigma_s is not None:
        from dropbear_wbc.settle.ground import ground_correction

        g = ground_correction({"left": low[:, 0], "right": low[:, 1]}, fps, c if c.any() else None,
                              sigma_s=settle_ground_sigma_s)
        low = low + g.dz[:, None]
        dz_info = {"sigma_s": settle_ground_sigma_s, "dz_min_m": float(g.dz.min()), "dz_max_m": float(g.dz.max())}
    hc = low[c]
    slips = []
    for k in range(2):
        v = np.linalg.norm(np.diff(orig[:, k, :2], axis=0), axis=-1) * fps
        slips.append(v[c[1:, k] & c[:-1, k]])
    sl = np.concatenate(slips)
    vel = np.abs(np.gradient(motor_q, 1.0 / fps, axis=0)) if t > 1 else np.zeros_like(motor_q)
    step = np.abs(np.diff(motor_q, axis=0)) if t > 1 else np.zeros((1, motor_q.shape[1]))
    out = {
        "contact_sole_z_mm": {"min": float(1e3 * hc.min()) if hc.size else None, "max": float(1e3 * hc.max()) if hc.size else None,
                              "abs_p95": float(1e3 * np.percentile(np.abs(hc), 95)) if hc.size else None,
                              "frac_over_limit": float((np.abs(hc) > max_contact_z).mean()) if hc.size else 0.0},
        "contact_foot_slip_mps": {"p95": float(np.percentile(sl, 95)) if sl.size else None,
                                  "max": float(sl.max()) if sl.size else None},
        "min_sole_z_any_foot_mm": float(1e3 * low.min()),
        "motor_max_abs_vel_rad_s": float(vel.max()), "motor_max_abs_vel_joint": MOTOR_NAMES[int(vel.max(0).argmax())],
        "motor_max_step_rad": float(step.max()),
        "contact_frac": {"left": float(c[:, 0].mean()), "right": float(c[:, 1].mean()), "none": float((~c.any(1)).mean())},
        "note": "kinematic prediction (MeasuredFK + sole hull of the sole plate) of the validator gates before the settle",
        "settle_ground_fix": dz_info,
    }
    out["gates_pass_predicted"] = bool((not hc.size or np.abs(hc).max() < max_contact_z) and vel.max() <= max_vel
                                       and step.max() <= max_step)
    return out


def project_motor_steps(m: np.ndarray, max_step: float, iters: int = 3000, tol: float = 1e-6) -> tuple[np.ndarray, dict]:
    """Closest trajectory (L2, per column) whose frame-to-frame change is <= ``max_step`` everywhere.

    Dykstra's alternating projection between the even-pair and odd-pair difference constraints (each a set of
    independent 2-variable projections), which converges to the exact Euclidean projection onto their intersection.
    Columns already within the limit are returned unchanged. Used to bring the ARM motors of dance clips under the motor
    speed gate (validator: 10 rad/s, 0.3 rad/frame) with the smallest possible change instead of a lagging rate
    limiter. ``m`` (T, J) [rad]; returns (projected (T, J), info)."""
    m = np.asarray(m, dtype=np.float64)
    out = m.copy()
    info: dict[str, Any] = {"max_step_rad": max_step, "columns": {}}
    if len(m) < 2:
        return out, info
    d0 = np.abs(np.diff(m, axis=0)).max(axis=0)
    cols = np.flatnonzero(d0 > max_step)
    if not cols.size:
        return out, info
    x = m[:, cols].copy()

    def proj(y: np.ndarray, start: int) -> np.ndarray:
        z = y.copy()
        a = z[start:-1:2] if (len(z) - start) % 2 == 0 else z[start:-2:2]
        b = z[start + 1::2][: len(a)]
        d = b - a
        over = np.abs(d) > max_step
        mid = 0.5 * (a + b)
        sg = np.sign(d)
        a_new = np.where(over, mid - 0.5 * sg * max_step, a)
        b_new = np.where(over, mid + 0.5 * sg * max_step, b)
        n = len(a)
        z[start:start + 2 * n:2] = a_new
        z[start + 1:start + 1 + 2 * n:2] = b_new
        return z

    p_, q_ = np.zeros_like(x), np.zeros_like(x)
    it = 0
    for it in range(iters):
        y = proj(x + p_, 0)
        p_ = x + p_ - y
        x_new = proj(y + q_, 1)
        q_ = y + q_ - x_new
        done = np.abs(np.diff(x_new, axis=0)).max() <= max_step + tol and np.abs(x_new - x).max() < 1e-9
        x = x_new
        if done:
            break
    out[:, cols] = x
    for k, c in enumerate(cols):
        info["columns"][int(c)] = {"max_step_before": float(d0[c]), "max_step_after": float(np.abs(np.diff(x[:, k])).max()),
                                   "max_change_rad": float(np.abs(x[:, k] - m[:, c]).max()),
                                   "frames_changed": int((np.abs(x[:, k] - m[:, c]) > 1e-6).sum())}
    info["iters"] = it + 1
    return out, info


def resample_semantic(fps_in: float, fps_out: float, pelvis_pos: np.ndarray, pelvis_quat_wxyz: np.ndarray,
                      q: np.ndarray, extra: dict[str, np.ndarray] | None = None):
    """Resample a semantic trajectory (lerp positions / angles / extra arrays, slerp quaternions). Keeps frame 0 and
    drops the tail beyond the last full output sample. Returns (pelvis_pos, pelvis_quat, q, extra, index_float)."""
    from .rotations import quat_slerp

    t_in = len(q)
    dur = (t_in - 1) / fps_in
    times = np.arange(0.0, dur + 1e-9, 1.0 / fps_out)
    x = times * fps_in
    i0 = np.clip(np.floor(x).astype(int), 0, t_in - 1)
    i1 = np.clip(i0 + 1, 0, t_in - 1)
    a = x - i0
    lerp = lambda arr: arr[i0] * (1 - a).reshape((-1,) + (1,) * (arr.ndim - 1)) + \
        arr[i1] * a.reshape((-1,) + (1,) * (arr.ndim - 1))  # noqa: E731
    qc = quat_continuous(pelvis_quat_wxyz)
    quat = quat_slerp(qc[i0], qc[i1], a)
    ex = {k: (lerp(np.asarray(v, dtype=np.float64)) if v is not None else None) for k, v in (extra or {}).items()}
    return lerp(pelvis_pos), quat_continuous(quat), lerp(q), ex, x


# ------------------------------------------------------------------------------------------------------------------
class PlantLegModel:
    """Legs of the plant forward model (``CalibrationFK.fk`` = MeasuredFK) with the sole-plate geometry.

    Motor order per leg: ``LEG_MOTOR_ROLES`` (hip roll, hip yaw, hip pitch, knee crank, calf A, calf B), as in
    ``kinematics.calib_fit.seg_motors``. Poses are root(``world`` body)-relative unless stated otherwise.
    """

    def __init__(self, cfk, sole_hulls: str | Path = SOLE_HULLS, cone_deg: float = 70.0, n_dirs: int = 3000,
                 max_support: int = 96):
        self.cfk = cfk
        self.smap = cfk.smap
        self.midx = {s: np.array([MOTOR_NAMES.index(seg_motors(s)[r]) for r in LEG_MOTOR_ROLES]) for s in SIDES}
        self.sem_idx = {s: np.array([SEMANTIC_INDEX[f"{s}_{j}"] for j in LEG_SEM]) for s in SIDES}
        lim = np.asarray(self.smap.motor_limits, dtype=np.float64)
        self.lo = {s: lim[self.midx[s], 0].copy() for s in SIDES}
        self.hi = {s: lim[self.midx[s], 1].copy() for s in SIDES}
        eps = 1e-4
        for s in SIDES:  # valid tables of the closed loops (the FK clamps outside them)
            sm = seg_motors(s)
            g = cfk.fk.loops[sm["knee"]]["grid"]["m"]
            self.lo[s][3], self.hi[s][3] = max(self.lo[s][3], g[0] + eps), min(self.hi[s][3], g[-1] - eps)
            gr = cfk.fk.grids[s]
            self.lo[s][4], self.hi[s][4] = max(self.lo[s][4], gr["a_grid"][0] + eps), min(self.hi[s][4], gr["a_grid"][-1] - eps)
            self.lo[s][5], self.hi[s][5] = max(self.lo[s][5], gr["b_grid"][0] + eps), min(self.hi[s][5], gr["b_grid"][-1] - eps)
        self.qz = np.asarray(cfk.qz, dtype=np.float64)
        self.r_ref = {s: cfk.fk.leg(s, self.qz)["foot"][0] for s in SIDES}
        soles = json.loads(Path(sole_hulls).read_text())
        from dropbear_wbc.kinematics.calib_fit import seg_bodies

        self.c_loc, self.support, self.sole_offset = {}, {}, {}
        for s in SIDES:
            v = np.asarray(soles["feet"][seg_bodies(s)["foot"]]["vertices_b"], dtype=np.float64)
            vr = v @ self.r_ref[s].T  # vertices in the level (flat) orientation
            zmin = vr[:, 2].min()
            bottom = vr[:, 2] < zmin + 1.5e-3
            cx = 0.5 * (vr[bottom, 0].min() + vr[bottom, 0].max())
            cy = 0.5 * (vr[bottom, 1].min() + vr[bottom, 1].max())
            self.c_loc[s] = self.r_ref[s].T @ np.array([cx, cy, zmin])
            # support points for "lowest point" queries: argmax_v v.u for down-directions u within cone_deg of the
            # sole normal (in the foot frame) -> a few dozen vertices instead of ~5000
            n = self.r_ref[s].T @ np.array([0.0, 0.0, -1.0])
            k = np.arange(n_dirs) + 0.5
            phi = np.arccos(1 - 2 * k / n_dirs)
            th = np.pi * (1 + 5 ** 0.5) * k
            dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)
            dirs = dirs[dirs @ n > np.cos(np.radians(cone_deg))]
            sup = v[np.unique(np.argmax(dirs @ v.T, axis=1))]
            if len(sup) > max_support:  # farthest-point subsample (stance feet are flat: any bottom vertex is exact)
                keep = [int(np.argmin(sup @ n))]
                d = np.linalg.norm(sup - sup[keep[0]], axis=1)
                for _ in range(max_support - 1):
                    keep.append(int(np.argmax(d)))
                    d = np.minimum(d, np.linalg.norm(sup - sup[keep[-1]], axis=1))
                sup = sup[keep]
            self.support[s] = sup
            self.sole_offset[s] = float(zmin)
        self.motor_bounds = {s: (self.lo[s], self.hi[s]) for s in SIDES}
        # foot heading relative to the pelvis that the hip yaw can reach (single-DOF semantic hip-yaw range)
        dofs = cfk.cal.get("dofs", {})
        self.yaw_band = {}
        for s in SIDES:
            rng = dofs.get(f"{s}_hip_yaw", {}).get("single_dof_range") or dofs.get(f"{s}_hip_yaw", {}).get("valid_range")
            self.yaw_band[s] = (float(min(rng)), float(max(rng))) if rng else (-0.5, 0.5)

    # -- kinematics ------------------------------------------------------------------------------------------------
    def _m22(self, m12: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
        t = m12.shape[0]
        m = np.broadcast_to(self.qz, (t, 22)).copy() if base is None else base.copy()
        m[:, self.midx["left"]] = m12[:, :6]
        m[:, self.midx["right"]] = m12[:, 6:]
        return m

    def feet_rel(self, m12: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Root-relative foot (R, p) of both legs for leg motors (T, 12) [left 6, right 6]."""
        m = self._m22(m12)
        return {s: self.cfk.fk.leg(s, m)["foot"] for s in SIDES}

    def leg_semantic(self, m12: np.ndarray) -> np.ndarray:
        """(T, 12) semantic leg angles [left LEG_SEM, right LEG_SEM] of leg motors (T, 12)."""
        q = self.smap.motor_to_semantic(self._m22(m12))
        return np.concatenate([q[:, self.sem_idx["left"]], q[:, self.sem_idx["right"]]], axis=1)

    def sole_centre(self, s: str, rw: np.ndarray, pw: np.ndarray) -> np.ndarray:
        return pw + rw @ self.c_loc[s]

    def lowest_z(self, s: str, rw: np.ndarray, pw: np.ndarray) -> np.ndarray:
        return (np.einsum("tj,vj->tv", rw[:, 2, :], self.support[s]) + pw[:, 2:3]).min(axis=1)

    def foot_yaw(self, s: str, rw: np.ndarray) -> np.ndarray:
        """Heading of the zero-referenced foot frame (flat foot: exact yaw)."""
        f = rw @ self.r_ref[s].T
        return np.arctan2(f[:, 1, 0], f[:, 0, 0])

    def flat(self, s: str, yaw: np.ndarray) -> np.ndarray:
        return rot_z(yaw) @ self.r_ref[s]


# ------------------------------------------------------------------------------------------------------------------
@dataclass
class FootStageResult:
    pelvis_pos: np.ndarray
    pelvis_quat_wxyz: np.ndarray
    q_sem: np.ndarray  # (T, 22) legs replaced
    leg_motors: np.ndarray  # (T, 12) [left 6, right 6] LEG_MOTOR_ROLES order
    motor_q: np.ndarray  # (T, 22): leg columns = leg_motors, others = the reference motors
    contacts: np.ndarray  # (T, 2) cleaned phases used as stance
    report: dict[str, Any] = field(default_factory=dict)


class _Problem:
    """Trajectory least squares over x_t = [pelvis correction (5), leg motors (12)] for all frames t.

    Per-frame residuals (feet, semantics, pelvis prior) are coupled in time by a second-difference smoothness term on
    every variable and a hinge on the leg-motor step (motor speed limit), which makes the normal matrix block-banded;
    it is solved by damped Gauss-Newton with a sparse factorisation and an active set for the motor box."""

    NV = 17

    def __init__(self, model: PlantLegModel, p: FootContactParams, fps: float, pel0: np.ndarray, r_pel0: np.ndarray,
                 pelvis_t_root: np.ndarray, stance: np.ndarray, tgt_c: np.ndarray, tgt_r: np.ndarray,
                 clear: np.ndarray, q_ref: np.ndarray):
        self.model, self.p, self.fps = model, p, fps
        self.pel0, self.r_pel0 = pel0, r_pel0
        self.r_pr, self.t_pr = pelvis_t_root[:3, :3], pelvis_t_root[:3, 3]
        self.stance, self.tgt_c, self.tgt_r, self.clear, self.q_ref = stance, tgt_c, tgt_r, clear, q_ref
        self.t = t = len(pel0)
        self.prior = np.zeros((t, 5))
        w_root = np.array([1 / p.root_xy_m, 1 / p.root_xy_m, 1 / p.root_z_m, 1 / p.root_tilt_rad, 1 / p.root_tilt_rad])
        self.w_root = np.broadcast_to(w_root, (t, 5)).copy()
        self.w_sem = np.concatenate([1 / np.asarray(p.sem_rad), 1 / np.asarray(p.sem_rad)])
        lo = np.concatenate([model.lo["left"], model.lo["right"]])
        hi = np.concatenate([model.hi["left"], model.hi["right"]])
        self.lo = np.concatenate([np.full(5, -np.inf), lo])
        self.hi = np.concatenate([np.full(5, np.inf), hi])
        # foot weights: stance = stiff; swing ramps (log-linearly) from stiff at touchdown/lift-off to soft
        self.w_pos = np.zeros((t, 2, 1))
        self.w_rot = np.zeros((t, 2, 3))
        ramp = max(1.0, p.landing_ramp_s * fps)
        idx = np.arange(t)
        for k in range(2):
            st = stance[:, k]
            if st.any():
                sidx = np.flatnonzero(st)
                j = np.clip(np.searchsorted(sidx, idx), 0, len(sidx) - 1)
                dist = np.minimum(np.abs(idx - sidx[j]), np.abs(idx - sidx[np.maximum(j - 1, 0)]))
            else:
                dist = np.full(t, np.inf)
            f = np.clip(dist / ramp, 0.0, 1.0)
            lp = (1 - f) * np.log(1 / p.stance_pos_m) + f * np.log(1 / p.swing_pos_m)
            self.w_pos[:, k, 0] = np.exp(lp)
            for c, st_w in enumerate((p.stance_tilt_rad, p.stance_tilt_rad, p.stance_yaw_rad)):
                self.w_rot[:, k, c] = np.exp((1 - f) * np.log(1 / st_w) + f * np.log(1 / p.swing_rot_rad))
        # temporal smoothness (second differences) and the motor-step hinge
        import scipy.sparse as sp

        acc = np.array([p.acc_root_xy_m] * 2 + [p.acc_root_z_m] + [p.acc_root_tilt_rad] * 2 + [p.acc_motor_rad] * 12)
        self.a_acc = 1.0 / acc
        if t >= 3:
            d2 = sp.diags([np.ones(t - 2), -2 * np.ones(t - 2), np.ones(t - 2)], [0, 1, 2], shape=(t - 2, t))
            self.S = sp.kron(d2.T @ d2, sp.diags(self.a_acc ** 2)).tocsr()
        else:
            self.S = sp.csr_matrix((self.NV * t, self.NV * t))

    def root(self, xr: np.ndarray):
        pel = self.pel0 + xr[:, :3]
        r_pel = self.r_pel0 @ rot_y(xr[:, 3]) @ rot_x(xr[:, 4])
        return r_pel @ self.r_pr, pel + r_pel @ self.t_pr, pel, r_pel

    def _foot_res(self, r_root, p_root, rel):
        """Weighted pose + clearance residuals of both feet, per side (T, 7)."""
        p = self.p
        out = []
        for k, s in enumerate(SIDES):
            r_rel, p_rel = rel[s]
            rw = r_root @ r_rel
            pw = p_root + np.einsum("tij,tj->ti", r_root, p_rel)
            c = self.model.sole_centre(s, rw, pw)
            e_p = (c - self.tgt_c[:, k]) * self.w_pos[:, k]
            e_r = _vee_skew(rw @ np.swapaxes(self.tgt_r[:, k], -1, -2)) * self.w_rot[:, k]
            low = self.model.lowest_z(s, rw, pw)
            e_c = np.where(self.stance[:, k], 0.0, np.maximum(0.0, self.clear[:, k] - low)) / p.clearance_scale_m
            out.append(np.concatenate([e_p, e_r, e_c[:, None]], axis=1))
        return out

    def residual(self, x: np.ndarray, rel=None, sem=None):
        xr, m = x[:, :5], x[:, 5:]
        r_root, p_root, _, _ = self.root(xr)
        rel = self.model.feet_rel(m) if rel is None else rel
        sem = self.model.leg_semantic(m) if sem is None else sem
        fl, fr = self._foot_res(r_root, p_root, rel)
        e_sem = (sem - self.q_ref) * self.w_sem
        e_root = (xr - self.prior) * self.w_root
        return np.concatenate([fl, e_sem[:, :6], fr, e_sem[:, 6:], e_root], axis=1)  # (T, 7+6+7+6+5 = 31)

    def jacobian(self, x: np.ndarray, r0: np.ndarray, rel0, sem0, fix_root: bool):
        jac = np.zeros((self.t, r0.shape[1], self.NV))
        eps = 1e-6
        if not fix_root:
            for j in range(5):
                xp = x.copy()
                xp[:, j] += eps
                jac[:, :, j] = (self.residual(xp, rel=rel0, sem=sem0) - r0) / eps
        for k in range(6):  # perturb motor k of both legs at once (the legs are independent)
            xp = x.copy()
            xp[:, 5 + k] += eps
            xp[:, 11 + k] += eps
            d = (self.residual(xp) - r0) / eps
            jac[:, :13, 5 + k] = d[:, :13]
            jac[:, 13:26, 11 + k] = d[:, 13:26]
        return jac

    def _hinge(self, x: np.ndarray):
        """Motor-step hinge: residuals (n,), frame and variable index of each active (t, t+1) pair."""
        d = np.diff(x[:, 5:], axis=0)
        vmax = self.p.max_motor_step_rad
        tt, jj = np.nonzero(np.abs(d) > vmax)
        sg = np.sign(d[tt, jj])
        res = (d[tt, jj] - sg * vmax) / self.p.step_hinge_scale_rad
        return res, tt, jj + 5

    def total_cost(self, x: np.ndarray, r: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        r = self.residual(x) if r is None else r
        xf = x.reshape(-1)
        smooth = float(xf @ (self.S @ xf))
        hres, _, _ = self._hinge(x)
        per = (r * r).sum(1)
        return float(per.sum() + smooth + (hres ** 2).sum()), per

    def solve(self, x0: np.ndarray, fix_root: bool = False, iters: int | None = None) -> tuple[np.ndarray, dict]:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl

        n, t = self.NV, self.t
        x = np.clip(x0.copy(), self.lo, self.hi)
        r = self.residual(x)
        cost, per = self.total_cost(x, r)
        cost0 = cost
        lam = 1e-3
        hist = []
        free_var = np.ones(n, dtype=bool)
        if fix_root:
            free_var[:5] = False
        for it in range(iters or self.p.iters):
            rel = self.model.feet_rel(x[:, 5:])
            sem = self.model.leg_semantic(x[:, 5:])
            jac = self.jacobian(x, r, rel, sem, fix_root)
            jt = np.swapaxes(jac, 1, 2)
            hb = jt @ jac
            gb = (jt @ r[..., None])[..., 0]
            h = sp.bsr_matrix((hb, np.arange(t), np.arange(t + 1)), shape=(n * t, n * t)).tocsr() + self.S
            g = gb.reshape(-1) + self.S @ x.reshape(-1)
            hres, ht, hj = self._hinge(x)
            if hres.size:
                rows = np.arange(hres.size)
                c = 1.0 / self.p.step_hinge_scale_rad
                jh = sp.csr_matrix((np.concatenate([np.full(hres.size, -c), np.full(hres.size, c)]),
                                    (np.concatenate([rows, rows]), np.concatenate([ht * n + hj, (ht + 1) * n + hj]))),
                                   shape=(hres.size, n * t))
                h = h + jh.T @ jh
                g = g + jh.T @ hres
            gv = g.reshape(t, n)
            fixed = ((x <= self.lo + 1e-9) & (gv > 0)) | ((x >= self.hi - 1e-9) & (gv < 0)) | ~free_var[None, :]
            free = ~fixed.reshape(-1)
            hf = h[free][:, free].tocsc()
            dg = hf.diagonal()
            accepted, step = False, 0.0
            for _ in range(8):
                a = (hf + sp.diags(lam * (dg + 1e-9))).tocsc()
                dx = np.zeros(n * t)
                dx[free] = -spl.spsolve(a, g[free])
                x_new = np.clip(x + dx.reshape(t, n), self.lo, self.hi)
                r_new = self.residual(x_new)
                c_new, per_new = self.total_cost(x_new, r_new)
                if c_new < cost:
                    step = float(np.abs(x_new - x).max())
                    rel_impr = (cost - c_new) / max(cost, 1e-12)
                    x, r, cost, per = x_new, r_new, c_new, per_new
                    lam = max(lam * 0.3, 1e-8)
                    accepted = True
                    break
                lam = min(lam * 10.0, 1e8)
            hist.append({"it": it, "cost": cost, "lam": lam, "accepted": accepted, "max_step": step})
            if not accepted or (it > 2 and (step < 1e-7 or rel_impr < 1e-7)):
                break
        return x, {"iters": len(hist), "cost_start": cost0, "cost": cost, "cost_frame_max": float(per.max()),
                   "last": hist[-1] if hist else None}


def _interp_over(mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Replace rows where ``mask`` is False by linear interpolation from the True rows (edges held)."""
    if mask.all() or not mask.any():
        return x.copy()
    idx = np.arange(len(x))
    out = x.copy()
    for k in range(x.shape[1]):
        out[:, k] = np.interp(idx, idx[mask], x[mask, k])
    return out


def _circ_mean(a: np.ndarray) -> float:
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _widen_stance(centres: dict, r_root: np.ndarray, pelvis_yaw: np.ndarray | None, min_width: float) -> dict:
    """In place: push the left/right sole-centre paths apart symmetrically along the pelvis heading's lateral axis
    wherever their lateral separation is below ``min_width`` [m]. Returns a report."""
    left, right = centres[SIDES[0]], centres[SIDES[1]]
    psi = pelvis_yaw if pelvis_yaw is not None else np.arctan2(r_root[:, 1, 0], r_root[:, 0, 0])
    lat = np.stack([-np.sin(psi), np.cos(psi)], axis=1)  # (T, 2) world direction of the pelvis +y
    sep = np.einsum("ti,ti->t", left[:, :2] - right[:, :2], lat)
    rep = {"min_width_m": float(min_width), "sep_before_m": {"min": float(sep.min()), "p10": float(np.percentile(sep, 10)),
                                                             "mean": float(sep.mean())},
           "frames_below_before": float((sep < min_width).mean())}
    if min_width > 0.0:
        deficit = np.clip(min_width - sep, 0.0, None)
        left[:, :2] += 0.5 * deficit[:, None] * lat
        right[:, :2] -= 0.5 * deficit[:, None] * lat
        rep["max_shift_per_foot_m"] = float(0.5 * deficit.max())
    return rep


def build_targets(model: PlantLegModel, rel_ref, r_root, p_root, contacts: np.ndarray, fps: float,
                  p: FootContactParams, source_yaw: np.ndarray | None, pelvis_yaw: np.ndarray | None = None):
    """World foot targets (see module doc). Returns dict with tgt_c (T,2,3), tgt_r (T,2,3,3), clear (T,2), info.

    Stance yaw: the circular mean of the source yaw over the phase, clamped into the band that Dropbear's hip yaw can
    reach relative to the pelvis heading (``PlantLegModel.yaw_band`` minus ``yaw_margin_deg``) for every frame of the
    phase. If no single yaw fits the whole phase (the pelvis turns by more than the band while the foot is down), the
    yaw follows the band per frame (a pivot on the sole centre; smoothed), which is reported."""
    t = len(contacts)
    tgt_c = np.zeros((t, 2, 3))
    tgt_r = np.zeros((t, 2, 3, 3))
    clear = np.zeros((t, 2))
    info: dict[str, Any] = {"phases": {}}
    ramp_n = max(1.0, p.clearance_ramp_s * fps)
    centres = {}
    for s in SIDES:
        r_rel, p_rel = rel_ref[s]
        centres[s] = model.sole_centre(s, r_root @ r_rel, p_root + np.einsum("tij,tj->ti", r_root, p_rel))
    info["min_width"] = _widen_stance(centres, r_root, pelvis_yaw, p.min_stance_width_m)
    for k, s in enumerate(SIDES):
        r_rel, p_rel = rel_ref[s]
        rw = r_root @ r_rel
        pw = p_root + np.einsum("tij,tj->ti", r_root, p_rel)
        c = centres[s]
        yaw_db = model.foot_yaw(s, rw)
        yaw_src = yaw_db if (source_yaw is None or p.yaw_source == "dropbear") else source_yaw[:, k]
        phases = contact_segments(contacts[:, k])
        d_start, d_end, rv_start, rv_end = [], [], [], []
        yaw_spread, clamped, pivots, drift = [], 0, [], []
        m = np.radians(p.yaw_margin_deg)
        band_lo, band_hi = model.yaw_band[s][0] + m, model.yaw_band[s][1] - m
        for a, b in phases:
            # how far the RETARGETED foot moves during the phase: large values mean the contacts merged real steps
            drift.append(float(np.linalg.norm(c[a:b, :2] - c[a, :2], axis=1).max()))
            psi = _circ_mean(yaw_src[a:b])
            yaw_spread.append(float(np.degrees(np.abs(_wrap(yaw_src[a:b] - psi)).max())))
            psi_t = np.full(b - a, psi)
            if pelvis_yaw is not None:
                rel = _wrap(psi - pelvis_yaw[a:b])  # foot heading relative to the pelvis, per frame
                lo_ok = np.max(band_lo - rel)  # psi must rise by at least this (if > 0)
                hi_ok = np.min(band_hi - rel)  # psi may rise by at most this (if < 0: must fall)
                if lo_ok <= hi_ok:  # a single yaw fits the whole phase
                    shift = min(max(0.0, lo_ok), hi_ok) if lo_ok > 0 else max(min(0.0, hi_ok), lo_ok)
                    if shift != 0.0:
                        clamped += 1
                    psi_t = psi_t + shift
                else:  # pelvis turns more than the band: pivot, yaw follows the band per frame
                    rel_c = np.clip(rel, band_lo, band_hi)
                    psi_t = pelvis_yaw[a:b] + rel_c
                    psi_t = np.unwrap(psi_t)
                    if b - a >= 3:
                        from dropbear_wbc.settle.ground import gaussian_smooth

                        psi_t = gaussian_smooth(psi_t, 0.04 * fps)
                    pivots.append(float(np.degrees(np.ptp(psi_t))))
            rts = model.flat(s, psi_t)
            ct = np.array([c[a, 0], c[a, 1], 0.0])
            tgt_c[a:b, k] = ct
            tgt_r[a:b, k] = rts
            d_start.append(ct - c[a])
            d_end.append(ct - c[b - 1])
            rv_start.append(_matrix_to_rotvec(rts[0] @ rw[a].T))
            rv_end.append(_matrix_to_rotvec(rts[-1] @ rw[b - 1].T))
        # swing: offsets blended between neighbouring phases
        dp = np.zeros((t, 3))
        drv = np.zeros((t, 3))
        swing = ~contacts[:, k]
        for a, b in contact_segments(swing):
            prev = next((i for i, (pa, pb) in enumerate(phases) if pb == a), None)
            nxt = next((i for i, (pa, pb) in enumerate(phases) if pa == b), None)
            n = b - a
            if prev is None and nxt is None:
                continue  # never in contact: zero offset
            if prev is None:
                dp[a:b], drv[a:b] = d_start[nxt], rv_start[nxt]
            elif nxt is None:
                dp[a:b], drv[a:b] = d_end[prev], rv_end[prev]
            else:
                w = _smoothstep((np.arange(n) + 1.0) / (n + 1.0))[:, None]
                dp[a:b] = (1 - w) * d_end[prev] + w * d_start[nxt]
                drv[a:b] = (1 - w) * rv_end[prev] + w * rv_start[nxt]
            # clearance ramp: distance (frames) to the nearest stance frame
            idx = np.arange(a, b)
            dist = np.minimum(idx - a + 1 if prev is not None else np.inf, b - idx if nxt is not None else np.inf)
            clear[a:b, k] = p.clearance_m * np.minimum(1.0, dist / ramp_n)
        r_sw = _rotvec_to_matrix(drv) @ rw
        c_sw = c + dp
        pw_sw = c_sw - np.einsum("tij,j->ti", r_sw, model.c_loc[s])
        low = model.lowest_z(s, r_sw, pw_sw)
        need = np.where(swing, np.maximum(0.0, clear[:, k] - low), 0.0)
        win = max(1, int(round(0.04 * fps)))
        lift = _moving_avg(_moving_max(need, 2 * win + 1), 2 * win + 1)
        lift = np.where(swing, lift, 0.0)
        c_sw[:, 2] += lift
        tgt_c[swing, k] = c_sw[swing]
        tgt_r[swing, k] = r_sw[swing]
        info["phases"][s] = {"n_stance_phases": len(phases),
                             "stance_frac": float(contacts[:, k].mean()),
                             "stance_yaw_spread_deg_max": max(yaw_spread) if yaw_spread else None,
                             "stance_yaw_spread_deg_p50": float(np.median(yaw_spread)) if yaw_spread else None,
                             "phases_yaw_clamped_to_hip_band": clamped,
                             "phases_pivot": len(pivots), "pivot_deg_max": max(pivots) if pivots else 0.0,
                             "swing_lift_max_m": float(lift.max()),
                             "baseline_stance_sole_z_m": {
                                 "min": float(model.lowest_z(s, rw, pw)[contacts[:, k]].min()) if contacts[:, k].any() else None,
                                 "max": float(model.lowest_z(s, rw, pw)[contacts[:, k]].max()) if contacts[:, k].any() else None},
                             "baseline_stance_slip_m_max": float(max((np.linalg.norm(d[:2]) for d in d_end), default=0.0)),
                             "retargeted_foot_drift_in_phase_m": {"p50": float(np.median(drift)) if drift else None,
                                                                  "max": float(max(drift)) if drift else None},
                             "yaw_source": "source" if (source_yaw is not None and p.yaw_source == "source") else "dropbear"}
    info["min_width_targets"] = _widen_swing_targets(tgt_c, contacts, r_root, pelvis_yaw, p.min_stance_width_m)
    return {"tgt_c": tgt_c, "tgt_r": tgt_r, "clear": clear, "info": info}


def _widen_swing_targets(tgt_c: np.ndarray, contacts: np.ndarray, r_root: np.ndarray, pelvis_yaw: np.ndarray | None,
                         min_width: float) -> dict:
    """Second width pass, on the final targets (in place): a stance target is locked where the phase started, so the
    first pass (on the retargeted paths) cannot see it; here a SWING foot closer than ``min_width`` to the other foot's
    target is pushed out by the whole deficit (both swinging: half each; both in stance: untouched -- the locks were
    widened by the first pass). Smoothed over 5 frames so the swing path stays continuous."""
    psi = pelvis_yaw if pelvis_yaw is not None else np.arctan2(r_root[:, 1, 0], r_root[:, 0, 0])
    lat = np.stack([-np.sin(psi), np.cos(psi)], axis=1)
    sep = np.einsum("ti,ti->t", tgt_c[:, 0, :2] - tgt_c[:, 1, :2], lat)
    rep = {"frames_below_before": float((sep < min_width).mean()), "sep_min_before_m": float(sep.min())}
    if min_width <= 0.0:
        return rep
    deficit = np.clip(min_width - sep, 0.0, None)
    swing = ~contacts
    share = np.zeros((len(sep), 2))
    both = swing[:, 0] & swing[:, 1]
    share[both] = 0.5
    only = swing & ~both[:, None]
    share[only] = 1.0
    push = deficit[:, None] * share  # (T, 2) metres per foot
    k = 5
    ker = np.ones(k) / k
    for f in range(2):
        sm = np.convolve(np.pad(push[:, f], (k // 2, k // 2), mode="edge"), ker, mode="valid")
        push[:, f] = np.where(swing[:, f], np.maximum(push[:, f], sm), 0.0)
    tgt_c[:, 0, :2] += push[:, 0:1] * lat
    tgt_c[:, 1, :2] -= push[:, 1:2] * lat
    sep2 = np.einsum("ti,ti->t", tgt_c[:, 0, :2] - tgt_c[:, 1, :2], lat)
    rep.update({"frames_below_after": float((sep2 < min_width).mean()), "sep_min_after_m": float(sep2.min()),
                "max_push_m": float(push.max())})
    return rep


def _stance_errors(model, prob: _Problem, x, contacts):
    r_root, p_root, _, _ = prob.root(x[:, :5])
    rel = model.feet_rel(x[:, 5:])
    out = {}
    pos_all, tilt_all, yaw_all, low_sw = [], [], [], []
    for k, s in enumerate(SIDES):
        rw = r_root @ rel[s][0]
        pw = p_root + np.einsum("tij,tj->ti", r_root, rel[s][1])
        c = model.sole_centre(s, rw, pw)
        st = contacts[:, k]
        e = rw @ np.swapaxes(prob.tgt_r[:, k], -1, -2)
        tilt = np.degrees(np.arccos(np.clip(e[:, 2, 2], -1, 1)))
        yaw = np.degrees(np.abs(np.arctan2(e[:, 1, 0], e[:, 0, 0])))
        pos = np.linalg.norm(c - prob.tgt_c[:, k], axis=1)
        low = model.lowest_z(s, rw, pw)
        pos_all.append(pos[st])
        tilt_all.append(tilt[st])
        yaw_all.append(yaw[st])
        low_sw.append(low[~st])
        out[s] = {"stance_pos_mm_max": float(1e3 * pos[st].max()) if st.any() else None,
                  "stance_sole_z_mm": [float(1e3 * low[st].min()), float(1e3 * low[st].max())] if st.any() else None,
                  "swing_min_sole_z_mm": float(1e3 * low[~st].min()) if (~st).any() else None}
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)  # noqa: E731
    pa, ta, ya = cat(pos_all), cat(tilt_all), cat(yaw_all)
    out["stance_pos_mm"] = {"p50": float(1e3 * np.median(pa)) if pa.size else None,
                            "p95": float(1e3 * np.percentile(pa, 95)) if pa.size else None,
                            "max": float(1e3 * pa.max()) if pa.size else None}
    out["stance_tilt_deg"] = {"p95": float(np.percentile(ta, 95)) if ta.size else None,
                              "max": float(ta.max()) if ta.size else None}
    out["stance_yaw_err_deg"] = {"p95": float(np.percentile(ya, 95)) if ya.size else None,
                                 "max": float(ya.max()) if ya.size else None}
    ls = cat(low_sw)
    out["swing_min_sole_z_mm"] = float(1e3 * ls.min()) if ls.size else None
    return out


def foot_contact_stage(fps: float, pelvis_pos: np.ndarray, pelvis_quat_wxyz: np.ndarray, q_sem: np.ndarray,
                       contacts: np.ndarray, cfk, pelvis_t_root: np.ndarray, params: FootContactParams = FootContactParams(),
                       source_yaw: np.ndarray | None = None, model: PlantLegModel | None = None) -> FootStageResult:
    """Run the stage (module doc). ``q_sem`` (T, 22) is the retargeted (achieved) semantic pose, ``contacts`` (T, 2) the
    source contact phases, ``source_yaw`` (T, 2) optional source foot headings [rad], ``pelvis_t_root`` the 4x4 pose of
    the root body in the semantic pelvis frame (``CalibrationView.pelvis_T_root``)."""
    import time

    t0 = time.time()
    p = params
    model = model or PlantLegModel(cfk)
    t = len(q_sem)
    contacts = clean_contacts(contacts, fps, p.min_stance_s, p.min_swing_s)
    r_pel = quat_to_matrix(pelvis_quat_wxyz)
    pel = np.asarray(pelvis_pos, dtype=np.float64).copy()
    # reference motors (continuity-seeded for the serial hips); legs only are optimised
    m_ref, seq_info = cfk.smap.semantic_to_motor_sequence(q_sem)
    m12_ref = np.concatenate([m_ref[:, model.midx["left"]], m_ref[:, model.midx["right"]]], axis=1)
    lo12 = np.concatenate([model.lo["left"], model.lo["right"]])
    hi12 = np.concatenate([model.hi["left"], model.hi["right"]])
    m12_ref = np.clip(m12_ref, lo12, hi12)
    q_ref12 = model.leg_semantic(m12_ref)
    rel_ref = model.feet_rel(m12_ref)
    r_pr, t_pr = pelvis_t_root[:3, :3], pelvis_t_root[:3, 3]
    # ---- baseline grounding with the plant feet ------------------------------------------------------------------
    from dropbear_wbc.settle.ground import ground_correction

    r_root = r_pel @ r_pr
    p_root = pel + r_pel @ t_pr
    low = {}
    for k, s in enumerate(SIDES):
        rw = r_root @ rel_ref[s][0]
        pw = p_root + np.einsum("tij,tj->ti", r_root, rel_ref[s][1])
        low[s] = model.lowest_z(s, rw, pw)
    g = ground_correction(low, fps, contacts if contacts.any() else None, sigma_s=p.ground_sigma_s)
    pel[:, 2] += g.dz
    p_root = pel + r_pel @ t_pr
    base_contact_z = np.where(contacts, np.stack([low["left"] + g.dz, low["right"] + g.dz], 1), np.nan)
    # ---- targets ---------------------------------------------------------------------------------------------------
    from .rotations import yaw_of_matrix

    tg = build_targets(model, rel_ref, r_root, p_root, contacts, fps, p, source_yaw, pelvis_yaw=yaw_of_matrix(r_pel))
    prob = _Problem(model, p, fps, pel, r_pel, pelvis_t_root, contacts, tg["tgt_c"], tg["tgt_r"], tg["clear"], q_ref12)
    x0 = np.concatenate([np.zeros((t, 5)), m12_ref], axis=1)
    before = _stance_errors(model, prob, x0, contacts)
    # ---- one banded (trajectory) Gauss-Newton solve --------------------------------------------------------------
    xc, info_c = prob.solve(x0)
    any_c = contacts.any(axis=1)
    after = _stance_errors(model, prob, xc, contacts)
    # ---- outputs -----------------------------------------------------------------------------------------------------
    _, _, pel_out, r_pel_out = prob.root(xc[:, :5])
    m12 = xc[:, 5:]
    q_out = np.asarray(q_sem, dtype=np.float64).copy()
    sem12 = model.leg_semantic(m12)
    q_out[:, model.sem_idx["left"]] = sem12[:, :6]
    q_out[:, model.sem_idx["right"]] = sem12[:, 6:]
    motor_q = m_ref.copy()
    motor_q[:, model.midx["left"]] = m12[:, :6]
    motor_q[:, model.midx["right"]] = m12[:, 6:]
    dm = np.abs(np.diff(m12, axis=0))
    # ankle poses at the edge of the feasible set: the SemanticMap's lut2d inverse (whole-cell validity) clips them,
    # i.e. semantic_to_motor(q_out) would NOT return these calf motors (the forward map and MeasuredFK are exact there)
    _, rep_out = cfk.smap.semantic_to_motor(q_out, return_report=True)
    clipped_out = np.asarray(rep_out.clipped).reshape(t, -1)
    ankle_cols = [SEMANTIC_INDEX[f"{s}_ankle_{j}"] for s in SIDES for j in ("pitch", "roll")]
    ankle_edge = {s: float(clipped_out[:, [SEMANTIC_INDEX[f"{s}_ankle_pitch"], SEMANTIC_INDEX[f"{s}_ankle_roll"]]]
                           .any(axis=1).mean()) for s in SIDES}
    at_lo = np.abs(m12 - lo12) < 1e-3
    at_hi = np.abs(m12 - hi12) < 1e-3
    roles = [f"{s}_{r}" for s in SIDES for r in LEG_MOTOR_ROLES]
    rc = xc[:, :5]
    report = {
        "stage": STAGE_VERSION, "params": asdict(p), "fk": "calib_fit.MeasuredFK (CalibrationFK) + sole hull",
        "calibration_sha256": getattr(cfk, "sha256", None), "frames": t, "fps": fps,
        "contacts_clean": {"left": float(contacts[:, 0].mean()), "right": float(contacts[:, 1].mean()),
                           "none": float((~any_c).mean())},
        "baseline_ground_dz_m": {"min": float(g.dz.min()), "max": float(g.dz.max())},
        "baseline_contact_sole_z_mm": {"min": float(1e3 * np.nanmin(base_contact_z)) if contacts.any() else None,
                                       "max": float(1e3 * np.nanmax(base_contact_z)) if contacts.any() else None},
        "targets": tg["info"],
        "before": before, "after": after,
        "solver": info_c,
        "pelvis_correction": {"dz_m": [float(rc[:, 2].min()), float(rc[:, 2].max())],
                              "dxy_m_max": float(np.linalg.norm(rc[:, :2], axis=1).max()),
                              "tilt_deg_max": float(np.degrees(np.abs(rc[:, 3:]).max())),
                              "incl_baseline_ground_dz_m": [float((rc[:, 2] + g.dz).min()), float((rc[:, 2] + g.dz).max())]},
        "leg_motor_at_bound_frac": {n: float((at_lo[:, i] | at_hi[:, i]).mean()) for i, n in enumerate(roles)
                                    if (at_lo[:, i] | at_hi[:, i]).any()},
        "leg_motor_speed": {"max_rad_s": float(dm.max() * fps) if len(dm) else 0.0,
                            "max_joint": roles[int(dm.max(axis=0).argmax())] if len(dm) else None,
                            "frames_over_10rad_s": int((dm * fps > 10.0).any(axis=1).sum()) if len(dm) else 0,
                            "max_step_rad": float(dm.max()) if len(dm) else 0.0},
        "ankle_at_feasible_edge_frac": {**ankle_edge, "note": "frames whose ankle (pitch, roll) lies in an inverse cell "
                                        "touching the feasible boundary (SemanticMap would clip it); calf motors valid"},
        "leg_semantic_change_deg": {f"{s}_{j}": float(np.degrees(np.abs(sem12[:, 6 * k + i] - q_ref12[:, 6 * k + i]).max()))
                                    for k, s in enumerate(SIDES) for i, j in enumerate(LEG_SEM)},
        "serial_continuity": {k: v for k, v in seq_info.items() if k != "serial_orientation_error_deg"},
        "wall_s": round(time.time() - t0, 2),
    }
    return FootStageResult(pelvis_pos=pel_out, pelvis_quat_wxyz=quat_continuous(matrix_to_quat(r_pel_out)),
                           q_sem=q_out, leg_motors=m12, motor_q=motor_q, contacts=contacts, report=report)
