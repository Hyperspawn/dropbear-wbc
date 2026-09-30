"""Derived SERIAL ("output-space") Dropbear model: fit, MJCF writer and runtime helpers.

DERIVED, NOT CANONICAL. The plant is the USD (docs/CONTRACTS.md section 0). Dropbear's knees, elbows and
ankles are closed loops; this model collapses every loop into the semantic joint it drives
(``dropbear-semantic-v1``, CONTRACTS section 2) so that serial-tree tools (GMR, SONIC motion_lib, mjlab,
ProtoMotions, MuJoCo viewers) can use Dropbear. Motor commands still go through
``dropbear_wbc.kinematics.semantic.SemanticMap`` (semantic -> 22 motors).

What is exact and what is fitted (numbers in the metadata JSON ``fit`` block):

* **Joint values = semantic values.** 22 hinges named exactly like the semantic DOFs; ranges = the
  calibration's semantic valid ranges (bounding boxes for the ``serial3`` and ``lut2d`` DOFs).
* **Orientations are exact** (to the calibration forward model): hips/shoulders are the YXZ decomposition
  (pitch about +y, roll about +x, yaw about +z of the zero-referenced segment frame), knee/elbow/wrist
  hinges use the measured rotation axes, the ankle is pitch (+y) then roll (+x) as in the calibration.
* **Positions are fitted.** Joint anchors are least-squares fits to the calibration's measured forward model (:class:`~dropbear_wbc.kinematics.calib_fit.
  MeasuredFK`, built from the raw physics sweep). Hips, shoulders, ankle and wrist are anchored at the
  semantic zero (exact there); the knee/elbow four-bars use the least-squares fixed pivot over their valid
  range with a free zero position (their subtrees are shifted by a few mm at the zero, reported as
  ``zero_pose_shift_mm``). Two approximations remain and are reported:
  the hip chain is physically roll -> yaw -> pitch while the semantic chain is pitch -> roll -> yaw (the
  roll axis sits ~8 cm above the pitch axis), and the four-bar knee/elbow are polycentric (no fixed pivot).
* **Mass/inertia**: every USD rigid body (except the 3 orphan bodies, CONTRACTS 0.1) is lumped onto the
  serial body it moves with (motion analysis over the calibration sweep), at the semantic-zero pose.
  Masses/COMs/inertias come from the USD (``tools/extract_usd_body_properties.py``); the 0.1 minimum
  principal inertia fix is applied.

Frames: the model world frame is the USD root-body (``world``) frame at the semantic zero; at the model zero
(all 22 joints = 0: legs straight, soles level, arms hanging, forearms pointing forward = G1 elbow 0) every serial
body frame is world-aligned, except the knee, elbow and wrist-roll links, which are rotated (knee ~10 deg) so that
their measured joint axis is a principal axis: every MJCF joint axis is an integer unit vector, as PHC-style parsers
(SONIC motion_lib, ProtoMotions) require. The root body ``pelvis`` sits at the calibration's
``rest_transforms.pelvis_in_root`` (hip-centre midpoint). Units: m, kg, rad; quaternions wxyz.

Two MJCF variants are written: ``dropbear_serial.xml`` (GMR, MuJoCo, mjlab: welded ``torso_link``/``head_link``
bodies, position servos) and ``dropbear_serial_motionlib.xml`` (one hinge per non-root body, ``<motor>`` actuators).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dropbear_wbc.kinematics.calib_fit import MeasuredFK, Raw, seg_bodies, seg_motors
from dropbear_wbc.kinematics.rigid import axis_angle_matrix, euler_yxz, mat_to_quat, quat_mat, rotvec, signed_angle_about
from dropbear_wbc.kinematics.semantic import MOTOR_NAMES, SEMANTIC_NAMES, SemanticMap, euler_yxz_matrix

REPO = Path(__file__).resolve().parents[3]
SCHEMA = "dropbear-serial-model-v1"
MODEL_NAME = "dropbear_serial"
DEFAULT_CALIBRATION = REPO / "data/calibration/dropbear_semantic_calibration.json"
DEFAULT_XML = REPO / "data/robot/dropbear_serial.xml"
DEFAULT_META = REPO / "data/robot/dropbear_serial.json"
DEFAULT_SCENE = REPO / "data/robot/dropbear_serial_scene.xml"
DEFAULT_SOLE_HULLS = REPO / "data/calibration/dropbear_foot_sole_hulls.json"
ORPHAN_BODIES = ("LL_skateboard_bearing__10__1", "LL_skateboard_bearing__11__1", "RL_skateboard_bearing__11__1")
MIN_PRINCIPAL_INERTIA = 2.0e-6
ANCHOR_BODY = "head_5mm_ujoint_base__5__1"
HEAD_BODY = "head_u_joint_center__8__1"
ROOT_BODY = "world"
DERIVED_LABEL = ("DERIVED, NOT CANONICAL: serial output-space surrogate of the Dropbear USD plant (sha 45586414...). "
                 "Closed loops (knee/elbow four-bars, parallel ankle, head Stewart platform) are collapsed into "
                 "the dropbear-semantic-v1 joints; motors are reached through SemanticMap.semantic_to_motor.")

SIDES = ("left", "right")
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
EX, EY, EZ = np.eye(3)
DOWN = np.array([0.0, 0.0, -1.0])

# serial body -> physical segment key (see _physical_segments); None = virtual link (no physical body)
PHYS_OF = {"hip_pitch": None, "hip_roll": None, "hip_yaw": "thigh", "knee": "shank", "ankle_pitch": "cross",
           "ankle_roll": "foot", "shoulder_pitch": "sh1", "shoulder_roll": "sh2", "shoulder_yaw": "upper_arm",
           "elbow": "forearm", "wrist_roll": "hand"}
KEY_SITE_BODIES = {  # tracked / key USD bodies exposed as sites (docs/CONTRACTS.md 5.1)
    "world": "pelvis", ANCHOR_BODY: "torso", HEAD_BODY: "head",
    **{seg_bodies(s)[k]: f"{s}_{k}" for s in SIDES for k in ("thigh", "shank", "foot", "upper_arm", "forearm", "hand")},
}


def link_name(side: str, joint: str) -> str:
    return f"{side}_{joint}_link"


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_path(p: str | Path) -> Path:
    """Absolute path; relative paths are taken relative to the repository root."""
    q = Path(str(p))
    return q if q.is_absolute() else REPO / q


def _mv(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...j->...i", r, v)


def _rot(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float64)
    return axis_angle_matrix(np.broadcast_to(np.asarray(axis, dtype=np.float64), angle.shape + (3,)), angle)


def _closest_on_line(p: np.ndarray, d: np.ndarray, x: np.ndarray) -> np.ndarray:
    d = d / np.linalg.norm(d)
    return p + d * np.dot(x - p, d)


def _stats_mm(err_m: np.ndarray) -> dict[str, float]:
    e = 1e3 * np.asarray(err_m, dtype=np.float64).reshape(-1)
    return {"rms_mm": float(np.sqrt(np.mean(e ** 2))), "p50_mm": float(np.percentile(e, 50)),
            "p95_mm": float(np.percentile(e, 95)), "max_mm": float(e.max()), "n": int(e.size)}


# ------------------------------------------------------------------------------------------------------
class CalibrationFK:
    """The calibration's own forward model on SEMANTIC poses.

    ``SemanticMap.semantic_to_motor`` (with clipping; the reachable semantic pose is the report's ``used``)
    followed by :class:`MeasuredFK` (products of exponentials for serial hinges, interpolated sweep tables for
    the closed loops), built from the raw sweep named in the calibration's provenance. Root-relative poses.
    """

    def __init__(self, calibration: str | Path = DEFAULT_CALIBRATION):
        self.path = resolve_path(calibration)
        self.cal = json.loads(self.path.read_text())
        self.sha256 = sha256_file(self.path)
        self.smap = SemanticMap(self.cal)
        prov = self.cal.get("provenance", {})
        if "raw_sweep" not in prov:
            raise RuntimeError(f"{self.path}: provenance.raw_sweep missing; cannot rebuild the forward model")
        self.raw_path = resolve_path(prov["raw_sweep"])
        if not self.raw_path.is_file():
            raise FileNotFoundError(f"raw sweep {self.raw_path} (named by {self.path}) not found")
        self.raw = Raw(self.raw_path)
        if self.raw.usd_sha() != self.cal["usd_sha256"]:
            raise RuntimeError("raw sweep and calibration come from different USDs")
        self.findings: list[str] = []
        self.fk = MeasuredFK(self.raw, self.findings, float(prov.get("max_gap_m", 3e-3)))
        self.qz = np.asarray(self.cal["semantic_zero_motor_pos"], dtype=np.float64)
        self.refs = {s: {k: np.asarray(v, dtype=np.float64) for k, v in self.cal["reference_rotations_root"][s].items()}
                     for s in SIDES}
        # consistency: the forward model must reproduce the calibration's reference rotations (legs at the
        # semantic zero, upper arm at rest; the forearm/hand references are the straight-arm rest record, which
        # lies ~1.5 deg outside the valid elbow table, so the table clips there and they are not compared)
        worst = 0.0
        for s in SIDES:
            leg = self.fk.leg(s, self.qz)
            arm = self.fk.arm(s, np.zeros(22))
            for k, (r, _) in {**leg, "upper_arm": arm["upper_arm"]}.items():
                worst = max(worst, float(np.abs(r - self.refs[s][k]).max()))
        if worst > 1e-6:
            raise RuntimeError(f"raw sweep {self.raw_path} does not reproduce the calibration reference rotations "
                               f"(max diff {worst:.2e}); stale sweep?")
        self.rigid_root_bodies = self._rigid_root_poses()

    def _rigid_root_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        i = self.fk.rest_idx
        return {b: (self.raw.R(b, i), self.raw.P(b, i)) for b in (ROOT_BODY, ANCHOR_BODY, HEAD_BODY)}

    def segments(self, q_sem: np.ndarray, clip: bool = True):
        """Semantic (..., 22) -> ({USD body name: (R (...,3,3), p (...,3))}, used semantic (..., 22), motors)."""
        q = np.asarray(q_sem, dtype=np.float64)
        m, rep = self.smap.semantic_to_motor(q, clip=clip, return_report=True)
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for s in SIDES:
            sb = seg_bodies(s)
            for k, v in {**self.fk.leg(s, m), **self.fk.arm(s, m)}.items():
                out[sb[k]] = v
        shape = q.shape[:-1]
        for b, (r, p) in self.rigid_root_bodies.items():
            out[b] = (np.broadcast_to(r, shape + (3, 3)).copy(), np.broadcast_to(p, shape + (3,)).copy())
        return out, rep.used, m


# ------------------------------------------------------------------------------------------------------
def _fit_chain_anchors(axes: list[np.ndarray], angles: np.ndarray, x_meas: np.ndarray, x0: np.ndarray):
    """Least-squares anchors of a serial hinge chain (axes fixed, world-aligned at zero).

    Model: x = a_1 + R_1[(a_2 - a_1) + R_2[... + R_k (x0 - a_k)]] = sum_i (P_{i-1} - P_i) a_i + P_k x0,
    P_i = R_1...R_i. Linear in the anchors; the along-axis components are unobservable (min-norm solution).
    Returns (anchors (k, 3), residual vectors (N, 3)).
    """
    n, k = angles.shape
    rs = [_rot(axes[i], angles[:, i]) for i in range(k)]
    p_prev = np.broadcast_to(np.eye(3), (n, 3, 3))
    blocks = []
    for i in range(k):
        p_i = p_prev @ rs[i]
        blocks.append(p_prev - p_i)
        p_prev = p_i
    a = np.concatenate(blocks, axis=-1).reshape(-1, 3 * k)
    b = (x_meas - _mv(p_prev, x0)).reshape(-1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    res = (a @ sol - b).reshape(n, 3)
    return sol.reshape(k, 3), res


def _fit_hinge_free(axis: np.ndarray, angles: np.ndarray, x_meas: np.ndarray):
    """Least-squares fixed-pivot hinge with a free zero position: x = c + R(q) (x0 - c), unknown c and x0.

    Used for the polycentric four-bars (knee, elbow): minimises the position error over the whole valid range
    instead of forcing exactness at q = 0. Returns (c, x0, residual vectors)."""
    r = _rot(axis, angles)
    a = np.concatenate([np.eye(3) - r, r], axis=-1).reshape(-1, 6)
    sol, *_ = np.linalg.lstsq(a, x_meas.reshape(-1), rcond=None)
    c, x0 = sol[:3], sol[3:]
    return c, x0, (a @ sol).reshape(-1, 3) - x_meas


def _chain_positions(axes, anchors, angles, x0):
    n, k = angles.shape
    x = np.broadcast_to(x0, (n, 3)).copy()
    for i in reversed(range(k)):
        x = anchors[i] + _mv(_rot(axes[i], angles[:, i]), x - anchors[i])
    return x


def _principal_axis(rv: np.ndarray, ref: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(rv, full_matrices=False)
    ax = vt[0] / np.linalg.norm(vt[0])
    return ax if np.dot(ax, ref) >= 0 else -ax


def _child_body(raw: Raw, joint: str) -> str:
    j = raw.jf[joint]
    return raw.body_names[int(raw.d["jf_body1"][j])]


@dataclass
class SerialSpec:
    """Everything needed to write the MJCF + metadata (world-aligned frames at the model zero)."""

    bodies: dict[str, dict] = field(default_factory=dict)   # name -> {parent, origin(3), joint?, ...}
    order: list[str] = field(default_factory=list)          # tree order (parents first)
    meta: dict[str, Any] = field(default_factory=dict)


def fit_serial_model(calibration: str | Path = DEFAULT_CALIBRATION, body_props: str | Path | None = None,
                     sole_hulls: str | Path | None = None, n_uniform: int = 3000, seed: int = 0,
                     log=print) -> SerialSpec:
    """Fit the serial model to the calibration (see module doc). Deterministic for a given input set."""
    cfk = CalibrationFK(calibration)
    cal, fk, raw, smap, qz = cfk.cal, cfk.fk, cfk.raw, cfk.smap, cfk.qz
    si = {n: i for i, n in enumerate(SEMANTIC_NAMES)}
    lim = smap.semantic_limits
    rng = np.random.default_rng(seed)
    if body_props is None:
        body_props = REPO / f"data/robot/usd_body_properties_{cal['usd_sha256'][:8]}.json"
    body_props = resolve_path(body_props)
    props = json.loads(body_props.read_text())
    if props["usd_sha256"] != cal["usd_sha256"]:
        raise RuntimeError(f"{body_props} is for USD {props['usd_sha256'][:12]}, calibration for {cal['usd_sha256'][:12]}")
    sole_hulls = resolve_path(sole_hulls or cal.get("provenance", {}).get("sole_hulls", DEFAULT_SOLE_HULLS))
    soles = json.loads(sole_hulls.read_text())

    pel = cal["rest_transforms"]["pelvis_in_root"]
    pelvis_origin = np.asarray(pel["pos"], dtype=np.float64)
    if np.abs(quat_mat(np.asarray(pel["quat_wxyz"], dtype=np.float64)) - np.eye(3)).max() > 1e-9:
        raise RuntimeError("pelvis_in_root is not axis-aligned with the root; unsupported")

    # ---- physical segments and their model-zero poses ------------------------------------------------
    rest_i = fk.rest_idx
    phys: dict[str, dict[str, str]] = {}
    zero_pose: dict[str, tuple[np.ndarray, np.ndarray]] = {ROOT_BODY: (np.eye(3), np.zeros(3))}
    for s in SIDES:
        sb, sm = seg_bodies(s), seg_motors(s)
        g = cal["serial_groups"][f"{s}_shoulder"]["chain_motors"]
        phys[s] = {"thigh": sb["thigh"], "shank": sb["shank"], "cross": sb["ankle_cross"], "foot": sb["foot"],
                   "sh1": _child_body(raw, g[0]), "sh2": _child_body(raw, g[1]), "upper_arm": sb["upper_arm"],
                   "forearm": sb["forearm"], "hand": sb["hand"]}
        leg, arm = fk.leg(s, qz), fk.arm(s, qz)
        for k in ("thigh", "shank", "foot"):
            zero_pose[sb[k]] = leg[k]
        for k in ("upper_arm", "forearm", "hand"):
            zero_pose[sb[k]] = arm[k]
        for k in ("sh1", "sh2"):  # shoulder motors are 0 at the model zero = authored rest
            zero_pose[phys[s][k]] = (raw.R(phys[s][k], rest_i), raw.P(phys[s][k], rest_i))
    for b in (ANCHOR_BODY, HEAD_BODY):
        zero_pose[b] = (raw.R(b, rest_i), raw.P(b, rest_i))

    # reference physics sample for bodies that are not segment bodies (relative placement)
    ref_src = f"sweep rest record ({cfk.raw_path.name})"
    ref_r, ref_p = raw.r[rest_i], raw.p[rest_i]
    ver = cal.get("verification", {}).get("verify_npz")
    if ver and resolve_path(ver).is_file():
        vr = Raw(resolve_path(ver))
        if "semantic_zero" in vr.program_names and vr.usd_sha() == cal["usd_sha256"]:
            vi = int(vr.rec("semantic_zero")[-1])
            if np.abs(vr.motor_pos[vi] - qz).max() < math.radians(0.5) and vr.body_names == raw.body_names:
                ref_r, ref_p = vr.r[vi], vr.p[vi]
                ref_src = f"verify semantic_zero record ({resolve_path(ver).name})"
    bidx = raw.bidx
    # cross (ankle U-joint cross): placed relative to the shank from the reference sample
    for s in SIDES:
        cr, sh = phys[s]["cross"], phys[s]["shank"]
        rs, ps = ref_r[bidx[sh]], ref_p[bidx[sh]]
        rc, pc = ref_r[bidx[cr]], ref_p[bidx[cr]]
        zs = zero_pose[sh]
        zero_pose[cr] = (zs[0] @ rs.T @ rc, zs[1] + zs[0] @ (rs.T @ (pc - ps)))

    # ---- segment assignment by motion analysis over the sweep ---------------------------------------
    cand = [ROOT_BODY] + [phys[s][k] for s in SIDES for k in phys[s]]
    com_b = {n: np.asarray(props["bodies"][n]["com_b"]) for n in props["bodies"]}
    assign: dict[str, str] = {}
    spread_tab: dict[str, dict[str, float]] = {}
    for b in raw.body_names:
        if b in cand:
            assign[b] = b
            continue
        c = raw.p[:, bidx[b]] + _mv(raw.r[:, bidx[b]], com_b[b])
        best, scores = None, {}
        for s in cand:
            rs, ps = raw.r[:, bidx[s]], raw.p[:, bidx[s]]
            loc = np.einsum("nji,nj->ni", rs, c - ps)
            pos_spread = float(np.sqrt(np.mean(np.sum((loc - loc.mean(0)) ** 2, -1))))
            rrel = np.swapaxes(rs, -1, -2) @ raw.r[:, bidx[b]]
            ang = np.linalg.norm(rotvec(rrel @ rrel[rest_i].T), axis=-1)
            score = pos_spread + 0.02 * float(np.sqrt(np.mean(ang ** 2)))
            scores[s] = score
            if best is None or score < scores[best]:
                best = s
        assign[b] = best
        spread_tab[b] = {"assigned": best, "score": scores[best]}
    # physical segment key -> serial body
    seg_to_serial = {ROOT_BODY: "pelvis"}
    for s in SIDES:
        for j, k in PHYS_OF.items():
            if k is not None and (j in LEG or j in ARM):
                seg_to_serial[phys[s][k]] = link_name(s, j)

    # ---- joint fits ---------------------------------------------------------------------------------
    fit: dict[str, Any] = {}
    joints: dict[str, dict] = {}   # joint name -> {axis, anchor, range, body}
    shift: dict[str, dict[str, np.ndarray]] = {s: {} for s in SIDES}   # four-bar subtree zero shifts
    anchor_ref: dict[str, np.ndarray] = {}

    def serial3_samples(s: str, grp: str, dofs: tuple[str, str, str]):
        ids = [si[f"{s}_{d}"] for d in dofs]
        q = np.zeros((n_uniform + 3 * 101, 22))
        q[:n_uniform, ids] = rng.uniform(lim[ids, 0], lim[ids, 1], (n_uniform, 3))
        for k, i in enumerate(ids):
            q[n_uniform + 101 * k:n_uniform + 101 * (k + 1), i] = np.linspace(lim[i, 0], lim[i, 1], 101)
        segs, used, _ = cfk.segments(q)
        return ids, segs, used

    for s in SIDES:
        sb, sm = seg_bodies(s), seg_motors(s)
        leg_pre, arm_pre = s[0].upper() + "L", s[0].upper() + "H"
        th0 = zero_pose[sb["thigh"]]
        hip_c = th0[1] + th0[0] @ raw.anchor_local(f"{leg_pre}_hip_joint", 1)[1]
        sh0 = zero_pose[sb["shank"]]
        ank_c = sh0[1] + sh0[0] @ raw.anchor_local(f"{leg_pre}_Revolute87", 0)[1]
        fo0 = zero_pose[sb["forearm"]]
        wr_loc = raw.anchor_local(f"{arm_pre}_wrist_roll", 0)[1]
        wrist_c = fo0[1] + fo0[0] @ wr_loc
        # -- hip (serial3, YXZ) --
        ids, segs, used = serial3_samples(s, "hip", ("hip_pitch", "hip_roll", "hip_yaw"))
        r_t, p_t = segs[sb["thigh"]]
        f_model = euler_yxz_matrix(used[:, ids])
        ori = np.degrees(np.linalg.norm(rotvec(np.swapaxes(f_model @ cfk.refs[s]["thigh"], -1, -2) @ r_t), axis=-1))
        anchors, res = _fit_chain_anchors([EY, EX, EZ], used[:, ids], p_t, th0[1])
        anchors = np.stack([_closest_on_line(anchors[i], ax, hip_c) for i, ax in enumerate((EY, EX, EZ))])
        res = _chain_positions([EY, EX, EZ], anchors, used[:, ids], th0[1]) - p_t
        fit[f"{s}_hip"] = {"position_error": _stats_mm(np.linalg.norm(res, axis=-1)),
                           "position_error_uniform_only": _stats_mm(np.linalg.norm(res[:n_uniform], axis=-1)),
                           "orientation_error_max_deg": float(ori.max()),
                           "samples": "uniform in the serial3 semantic box (clipped to reachable) + single-DOF sweeps",
                           "note": "chain-order approximation: physical hip is roll->yaw->pitch, semantic is "
                                   "pitch->roll->yaw; the error is a pure translation of the thigh"}
        for i, d in enumerate(("hip_pitch", "hip_roll", "hip_yaw")):
            joints[f"{s}_{d}"] = {"axis": (EY, EX, EZ)[i], "anchor": anchors[i]}
        # -- shoulder (serial3, YXZ; physical chain has the same order) --
        ids, segs, used = serial3_samples(s, "shoulder", ("shoulder_pitch", "shoulder_roll", "shoulder_yaw"))
        r_u, p_u = segs[sb["upper_arm"]]
        ua0 = zero_pose[sb["upper_arm"]]
        f_model = euler_yxz_matrix(used[:, ids])
        ori = np.degrees(np.linalg.norm(rotvec(np.swapaxes(f_model @ cfk.refs[s]["upper_arm"], -1, -2) @ r_u), axis=-1))
        anchors, _ = _fit_chain_anchors([EY, EX, EZ], used[:, ids], p_u, ua0[1])
        g0 = cal["serial_groups"][f"{s}_shoulder"]["chain_motors"][0]   # physical shoulder-pitch joint frame
        s1 = zero_pose[phys[s]["sh1"]]
        sh_ref = s1[1] + s1[0] @ raw.anchor_local(g0, 1)[1]
        anchors = np.stack([_closest_on_line(anchors[i], ax, sh_ref) for i, ax in enumerate((EY, EX, EZ))])
        res = _chain_positions([EY, EX, EZ], anchors, used[:, ids], ua0[1]) - p_u
        fit[f"{s}_shoulder"] = {"position_error": _stats_mm(np.linalg.norm(res, axis=-1)),
                                "orientation_error_max_deg": float(ori.max())}
        for i, d in enumerate(("shoulder_pitch", "shoulder_roll", "shoulder_yaw")):
            joints[f"{s}_{d}"] = {"axis": (EY, EX, EZ)[i], "anchor": anchors[i]}
        # -- knee (four-bar: fixed-axis hinge, LS pivot anchored at the zero) --
        kd = cal["dofs"][f"{s}_knee"]
        mg = np.asarray(kd["motor_grid"], dtype=np.float64)
        q = np.tile(qz, (241, 1))
        q[:, cfk.raw.midx[sm["knee"]]] = np.linspace(mg[0], mg[-1], 241)
        leg = fk.leg(s, q)
        g = leg["shank"][0] @ sh0[0].T
        ang, _ = signed_angle_about(g, EY)
        k_ax = _principal_axis(rotvec(g)[np.abs(ang) > math.radians(3)], EY)
        ori = np.degrees(np.linalg.norm(rotvec(np.swapaxes(_rot(k_ax, ang), -1, -2) @ g), axis=-1))
        a_meas = leg["shank"][1] + _mv(leg["shank"][0], sh0[0].T @ (ank_c - sh0[1]))
        piv, ank_fit, _ = _fit_hinge_free(k_ax, ang, a_meas)
        piv = _closest_on_line(piv, k_ax, 0.5 * (hip_c + ank_c))
        res = _chain_positions([k_ax], piv[None], ang[:, None], ank_fit) - a_meas
        shift[s]["knee"] = ank_fit - ank_c
        by_angle = [{"knee_deg": float(np.degrees(a)), "ankle_error_mm": float(1e3 * np.linalg.norm(e))}
                    for a, e in zip(ang[::20], res[::20])]
        fit[f"{s}_knee"] = {"axis_world_at_zero": k_ax.tolist(),
                            "axis_tilt_from_y_deg": float(np.degrees(np.arccos(abs(k_ax[1])))),
                            "ankle_centre_error": _stats_mm(np.linalg.norm(res, axis=-1)),
                            "orientation_error_max_deg": float(ori.max()), "knee_range_deg": np.degrees([ang.min(), ang.max()]).tolist(),
                            "error_by_angle": by_angle,
                            "zero_pose_shift_mm": float(1e3 * np.linalg.norm(ank_fit - ank_c)),
                            "zero_pose_shift_vector_m": (ank_fit - ank_c).tolist(),
                            "note": "polycentric four-bar knee approximated by the least-squares fixed-pivot hinge over "
                                    "the valid knee range (pivot and zero position fitted; the shank subtree is "
                                    "translated by zero_pose_shift at the model zero)"}
        joints[f"{s}_knee"] = {"axis": k_ax, "anchor": piv}
        # -- ankle (pitch +y then roll +x; LS anchors over the feasible calf-motor grid) --
        pair = cal["ankle_pairs"][s]
        ag, bgr = np.asarray(pair["a_grid"]), np.asarray(pair["b_grid"])
        valid = np.asarray(pair["valid"], dtype=bool)
        aa, bb = np.meshgrid(ag, bgr, indexing="ij")
        q = np.tile(qz, (int(valid.sum()), 1))
        q[:, cfk.raw.midx[sm["calf_a"]]] = aa[valid]
        q[:, cfk.raw.midx[sm["calf_b"]]] = bb[valid]
        leg = fk.leg(s, q)
        fo_z = zero_pose[sb["foot"]]
        f_foot = leg["foot"][0] @ fo_z[0].T   # shank is at its zero pose
        e3 = euler_yxz(f_foot)
        pr = e3[:, :2]
        pts = [np.zeros(3), np.array([0.08, 0.0, 0.0]), np.array([0.0, 0.04, 0.0])]
        xm = np.concatenate([leg["foot"][1] + _mv(leg["foot"][0], p) for p in pts])
        x0s = [fo_z[1] + fo_z[0] @ p for p in pts]
        # stack the three points into one LS problem (shared anchors)
        n = len(pr)
        rp, rr = _rot(EY, pr[:, 0]), _rot(EX, pr[:, 1])
        a_mat = np.concatenate([np.concatenate([np.eye(3) - rp, rp - rp @ rr], -1)] * 3).reshape(-1, 6)
        b_vec = (xm - np.concatenate([_mv(rp @ rr, x0) for x0 in x0s])).reshape(-1)
        sol, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        a_p = _closest_on_line(sol[:3], EY, ank_c)
        a_r = _closest_on_line(sol[3:], EX, ank_c)
        pos = np.concatenate([_chain_positions([EY, EX], np.stack([a_p, a_r]), pr, x0) for x0 in x0s])
        fit[f"{s}_ankle"] = {"foot_point_error": _stats_mm(np.linalg.norm(pos - xm, axis=-1)),
                             "yaw_residual_max_deg": float(np.degrees(np.abs(e3[:, 2]).max())),
                             "grid_nodes_used": int(n),
                             "pitch_anchor_to_roll_anchor_m": float(np.linalg.norm(
                                 _closest_on_line(a_p, EY, a_r) - a_r))}
        joints[f"{s}_ankle_pitch"] = {"axis": EY, "anchor": a_p}
        joints[f"{s}_ankle_roll"] = {"axis": EX, "anchor": a_r}
        # -- elbow (four-bar: fixed-axis hinge; value = G1 elbow, 0 = forearm forward) --
        ed = cal["dofs"][f"{s}_elbow"]
        mg = np.asarray(ed["motor_grid"], dtype=np.float64)
        q = np.tile(qz, (241, 1))
        q[:, cfk.raw.midx[sm["elbow"]]] = np.linspace(mg[0], mg[-1], 241)
        arm = fk.arm(s, q)
        f_e = arm["forearm"][0] @ cfk.refs[s]["forearm"].T  # upper arm at rest (F_upper = I)
        th, _ = signed_angle_about(f_e, EY)
        el = np.pi / 2 + th
        g = arm["forearm"][0] @ fo0[0].T
        e_ax = _principal_axis(rotvec(f_e)[np.abs(th) > math.radians(3)], EY)
        ori = np.degrees(np.linalg.norm(rotvec(np.swapaxes(_rot(e_ax, el), -1, -2) @ g), axis=-1))
        w_meas = arm["forearm"][1] + _mv(arm["forearm"][0], wr_loc)
        piv, wr_fit, _ = _fit_hinge_free(e_ax, el, w_meas)
        el_body, el_loc = raw.anchor_local(f"{arm_pre}_elbow_joint", 0)   # elbow motor frame (upper arm)
        ez = zero_pose[el_body] if el_body in zero_pose else zero_pose[sb["upper_arm"]]
        piv = _closest_on_line(piv, e_ax, ez[1] + ez[0] @ el_loc)
        res = _chain_positions([e_ax], piv[None], el[:, None], wr_fit) - w_meas
        shift[s]["elbow"] = wr_fit - wrist_c
        fit[f"{s}_elbow"] = {"axis_world_at_zero": e_ax.tolist(),
                             "wrist_error": _stats_mm(np.linalg.norm(res, axis=-1)),
                             "orientation_error_max_deg": float(ori.max()),
                             "elbow_range_deg": np.degrees([el.min(), el.max()]).tolist(),
                             "zero_pose_shift_mm": float(1e3 * np.linalg.norm(wr_fit - wrist_c)),
                             "zero_pose_shift_vector_m": (wr_fit - wrist_c).tolist(),
                             "note": "polycentric four-bar elbow approximated by the least-squares fixed-pivot hinge over "
                                     "the valid elbow range (forearm subtree translated by zero_pose_shift)"}
        joints[f"{s}_elbow"] = {"axis": e_ax, "anchor": piv}
        # -- wrist roll (serial hinge, measured screw in the forearm frame) --
        sc = fk.screws[sm["wrist_roll"]]
        w_ax = fo0[0] @ np.asarray(sc["axis"])
        down0 = _rot(e_ax, -np.pi / 2) @ DOWN   # elbow->hand direction at the model zero
        if np.dot(w_ax, down0) < 0:
            w_ax = -w_ax
        w_pt = _closest_on_line(fo0[1] + fo0[0] @ np.asarray(sc["point"]), w_ax, wrist_c)
        wd = cal["dofs"][f"{s}_wrist_roll"]
        mw = np.linspace(-1.0, 1.0, 81)
        q = np.tile(qz, (81, 1))
        q[:, cfk.raw.midx[sm["wrist_roll"]]] = mw
        arm = fk.arm(s, q)
        hz = zero_pose[sb["hand"]]
        g = arm["hand"][0] @ hz[0].T
        wr_sem = smap.motor_to_semantic(q, clip=False)[:, si[f"{s}_wrist_roll"]]
        ori = np.degrees(np.linalg.norm(rotvec(np.swapaxes(_rot(w_ax, wr_sem), -1, -2) @ g), axis=-1))
        res = w_pt + _mv(_rot(w_ax, wr_sem), hz[1] - w_pt) - arm["hand"][1]
        fit[f"{s}_wrist"] = {"axis_world_at_zero": w_ax.tolist(), "hand_error": _stats_mm(np.linalg.norm(res, axis=-1)),
                             "orientation_error_max_deg": float(ori.max()),
                             "semantic_map_type": wd.get("type")}
        joints[f"{s}_wrist_roll"] = {"axis": w_ax, "anchor": w_pt}
        # translate the subtrees below the four-bars to the least-squares zero positions
        dk, de = shift[s]["knee"], shift[s]["elbow"]
        for k in ("shank", "cross", "foot"):
            zr, zp = zero_pose[phys[s][k]]
            zero_pose[phys[s][k]] = (zr, zp + dk)
        for k in ("forearm", "hand"):
            zr, zp = zero_pose[phys[s][k]]
            zero_pose[phys[s][k]] = (zr, zp + de)
        for jn in ("ankle_pitch", "ankle_roll"):
            joints[f"{s}_{jn}"]["anchor"] = joints[f"{s}_{jn}"]["anchor"] + dk
        joints[f"{s}_wrist_roll"]["anchor"] = joints[f"{s}_wrist_roll"]["anchor"] + de
        anchor_ref[s] = {"hip_centre": hip_c, "ankle_centre": ank_c + dk, "wrist_centre": wrist_c + de}

    # ---- bodies ---------------------------------------------------------------------------------------
    spec = SerialSpec()
    spec.bodies["pelvis"] = {"parent": None, "origin": pelvis_origin, "joint": None, "phys": ROOT_BODY}
    spec.order.append("pelvis")
    for s in SIDES:
        parent = "pelvis"
        for j in LEG:
            name = link_name(s, j)
            jn = f"{s}_{j}"
            spec.bodies[name] = {"parent": parent, "origin": joints[jn]["anchor"], "joint": jn,
                                 "phys": phys[s][PHYS_OF[j]] if PHYS_OF[j] else None}
            spec.order.append(name)
            parent = name
    for s in SIDES:
        parent = "pelvis"
        for j in ARM:
            name = link_name(s, j)
            jn = f"{s}_{j}"
            spec.bodies[name] = {"parent": parent, "origin": joints[jn]["anchor"], "joint": jn,
                                 "phys": phys[s][PHYS_OF[j]]}
            spec.order.append(name)
            parent = name
    for name, b in (("torso_link", ANCHOR_BODY), ("head_link", HEAD_BODY)):
        spec.bodies[name] = {"parent": "pelvis", "origin": zero_pose[b][1], "joint": None, "phys": None, "welded": True}
        spec.order.append(name)
    for jn, j in joints.items():
        i = si[jn]
        j["range"] = lim[i].copy()
    # body frames: world-aligned at the model zero, except where a joint axis is not a principal axis (knee: the
    # measured axis is tilted ~10 deg; elbow, wrist: measured screw axes). Those bodies get the minimal rotation that
    # maps the nearest principal axis onto the measured axis, so every MJCF joint axis is an exact integer unit
    # vector (PHC / SONIC motion_lib / ProtoMotions parsers read axes with int()), like G1's tilted links.
    for n, bd in spec.bodies.items():
        bd["R0"] = np.eye(3)
        bd["axis_local"] = None
        if bd["joint"]:
            a = np.asarray(joints[bd["joint"]]["axis"], dtype=np.float64)
            a = a / np.linalg.norm(a)
            k = int(np.argmax(np.abs(a)))
            e = np.eye(3)[k] * np.sign(a[k])
            c = np.cross(e, a)
            sn = np.linalg.norm(c)
            if sn > 1e-12:
                bd["R0"] = _rot(c / sn, np.arctan2(sn, float(np.dot(e, a))))
            bd["axis_local"] = e
            if np.abs(bd["R0"] @ e - a).max() > 1e-9:
                raise RuntimeError(f"frame rotation failed for {n}")

    # ---- inertials (lumped at the model zero) ---------------------------------------------------------
    body_pose0: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for b in raw.body_names:
        seg = assign[b]
        zs = zero_pose[seg]
        rs, ps = ref_r[bidx[seg]], ref_p[bidx[seg]]
        rb, pb = ref_r[bidx[b]], ref_p[bidx[b]]
        body_pose0[b] = (zs[0] @ rs.T @ rb, zs[1] + zs[0] @ (rs.T @ (pb - ps)))
    lumps: dict[str, list[str]] = {n: [] for n in spec.bodies}
    for b in raw.body_names:
        if b in ORPHAN_BODIES:
            continue
        lumps[seg_to_serial[assign[b]]].append(b)
    total = 0.0
    for name, members in lumps.items():
        bd = spec.bodies[name]
        if bd.get("welded"):
            bd["inertial"] = None
            continue
        m_tot, c_acc, entries = 0.0, np.zeros(3), []
        for b in members:
            pr = props["bodies"][b]
            m = float(pr["mass"])
            r0, p0 = body_pose0[b]
            c = p0 + r0 @ np.asarray(pr["com_b"])
            ib = np.asarray(pr["inertia_b"], dtype=np.float64)
            w, v = np.linalg.eigh(ib)
            ib = v @ np.diag(np.maximum(w, MIN_PRINCIPAL_INERTIA)) @ v.T   # CONTRACTS 0.1 fix
            entries.append((m, c, r0 @ ib @ r0.T))
            m_tot += m
            c_acc += m * c
        if m_tot <= 0:
            bd["inertial"] = {"mass": 0.01, "com": bd["origin"].copy(), "inertia": np.eye(3) * 1e-5,
                              "members": [], "placeholder": True}
            continue
        com = c_acc / m_tot
        inertia = np.zeros((3, 3))
        for m, c, iw in entries:
            d = c - com
            inertia += iw + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        bd["inertial"] = {"mass": m_tot, "com": com, "inertia": inertia, "members": members, "placeholder": False}
        total += m_tot
    placeholders = [n for n, b in spec.bodies.items() if b.get("inertial") and b["inertial"]["placeholder"]]
    if placeholders:  # keep the total mass: take the placeholder mass out of the pelvis
        pin = spec.bodies["pelvis"]["inertial"]
        pin["mass"] -= 0.01 * len(placeholders)

    # ---- visual meshes: convex hull of each USD body's collision hull, at its model-zero pose ---------------
    # (one mesh per USD body, baked into the serial body frame at write time; gives viewers a Dropbear-like look
    # and gives PHC/SONIC Humanoid_Batch the <asset> meshes its load_mesh()/mesh_fk() height fix requires)
    meshes: list[dict] = []
    for b in raw.body_names:
        if b in ORPHAN_BODIES:
            continue
        hull = props["bodies"][b]["collision_hull_b"]
        if not hull or len(hull) < 4:
            continue
        r0, p0 = body_pose0[b]
        meshes.append({"name": "usd_" + b, "usd_body": b, "body": seg_to_serial[assign[b]],
                       "vertices_world0": np.asarray(hull, dtype=np.float64) @ r0.T + p0})

    # ---- sites (USD key bodies at their frames) and collision geoms ------------------------------------
    sites: dict[str, dict] = {}
    for b, label in KEY_SITE_BODIES.items():
        seg = assign[b] if b in assign else ROOT_BODY
        serial = seg_to_serial[seg]
        if b in (ANCHOR_BODY,):
            serial = "torso_link"
        if b in (HEAD_BODY,):
            serial = "head_link"
        r0, p0 = body_pose0[b] if b != ROOT_BODY else (np.eye(3), np.zeros(3))
        sites[f"usd_{b}"] = {"body": serial, "pos_world0": p0, "rot_world0": r0, "usd_body": b, "label": label}
    geoms: list[dict] = []
    for s in SIDES:
        sb = seg_bodies(s)
        foot = sb["foot"]
        fz = zero_pose[foot]
        v = np.asarray(soles["feet"][foot]["vertices_b"], dtype=np.float64) @ fz[0].T + fz[1]
        zmin = float(v[:, 2].min())
        plate = v[v[:, 2] < zmin + 0.005]   # sole footprint: hull vertices within 5 mm of the bottom plane
        lo, hi = plate.min(0), plate.max(0)
        h = 0.02
        geoms.append({"body": link_name(s, "ankle_roll"), "name": f"{s}_foot_sole", "type": "box", "class": "collision",
                      "center_world0": np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, zmin + h / 2]),
                      "half": np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, h / 2])})
        sites[f"{s}_sole"] = {"body": link_name(s, "ankle_roll"), "pos_world0": np.array([(lo[0] + hi[0]) / 2,
                              (lo[1] + hi[1]) / 2, zmin]), "rot_world0": np.eye(3), "usd_body": None,
                              "label": f"{s} sole centre (bottom plane)"}
        hand = sb["hand"]
        hb = props["bodies"][hand]["collision_hull_b"]
        if hb:
            hz = zero_pose[hand]
            hv = np.asarray(hb) @ hz[0].T + hz[1]
            geoms.append({"body": link_name(s, "wrist_roll"), "name": f"{s}_hand", "type": "box", "class": "collision",
                          "center_world0": (hv.min(0) + hv.max(0)) / 2, "half": (hv.max(0) - hv.min(0)) / 2})
    tb = props["bodies"][ROOT_BODY]["collision_hull_b"]
    if tb:
        tv = np.asarray(tb)
        geoms.append({"body": "pelvis", "name": "torso", "type": "box", "class": "collision",
                      "center_world0": (tv.min(0) + tv.max(0)) / 2, "half": (tv.max(0) - tv.min(0)) / 2})
    # visual capsules along the limbs (the knee capsule joint is drawn on the hip-ankle line, not at the fitted
    # polycentric pivot, so the straight leg looks straight)
    for s in SIDES:
        o = {j: spec.bodies[link_name(s, j)]["origin"] for j in LEG + ARM}
        hc, ac = anchor_ref[s]["hip_centre"], anchor_ref[s]["ankle_centre"]
        knee_vis = hc + (ac - hc) * np.dot(o["knee"] - hc, ac - hc) / np.dot(ac - hc, ac - hc)
        segs = [("hip_yaw", hc, knee_vis, 0.045), ("knee", knee_vis, ac, 0.04),
                ("shoulder_yaw", o["shoulder_yaw"], o["elbow"], 0.035),
                ("elbow", o["elbow"], anchor_ref[s]["wrist_centre"], 0.03)]
        for j, a, b_, r in segs:
            geoms.append({"body": link_name(s, j), "name": f"{s}_{j}_vis", "type": "capsule", "class": "visual",
                          "from_world0": a, "to_world0": b_, "radius": r})

    # ---- surrogate actuator parameters (semantic-space equivalents of the motor PD, via J = dm/ds) -----
    stand_sem = np.asarray(cal.get("standing_semantic_pos", smap.motor_to_semantic(np.asarray(cal["standing_motor_pos"]))))
    jac = np.zeros((22, 22))
    base = smap.semantic_to_motor(stand_sem, clip=False)
    for i in range(22):
        h = 1e-3
        dq = np.zeros(22)
        dq[i] = h
        jac[:, i] = (smap.semantic_to_motor(stand_sem + dq, clip=False) - smap.semantic_to_motor(stand_sem - dq, clip=False)) / (2 * h)
    from dropbear_wbc.robots.dropbear_names import ACTUATOR_GROUP_MOTORS, ACTUATOR_PARAMS
    grp = {m: g for g, ms in ACTUATOR_GROUP_MOTORS.items() for m in ms}
    eff = np.array([ACTUATOR_PARAMS[grp[m]][0] for m in MOTOR_NAMES])
    kp = np.array([ACTUATOR_PARAMS[grp[m]][1] for m in MOTOR_NAMES])
    kd = np.array([ACTUATOR_PARAMS[grp[m]][2] for m in MOTOR_NAMES])
    arm_ = np.array([ACTUATOR_PARAMS[grp[m]][3] for m in MOTOR_NAMES])
    act = {}
    for jn in joints:
        i = si[jn]
        col = jac[:, i]
        act[jn] = {"effort": float(np.sum(np.abs(col) * eff)), "kp": float(np.sum(col ** 2 * kp)),
                   "kd": float(np.sum(col ** 2 * kd)), "armature": float(np.sum(col ** 2 * arm_)),
                   "motors": [MOTOR_NAMES[k] for k in np.nonzero(np.abs(col) > 1e-3)[0]]}

    # ---- metadata ---------------------------------------------------------------------------------------
    hip_height = float(pelvis_origin[2] - min(g["center_world0"][2] - g["half"][2] for g in geoms
                                              if g["name"].endswith("foot_sole")))
    spec.meta = {
        "schema": SCHEMA, "model": MODEL_NAME, "status": DERIVED_LABEL,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {
            "calibration": str(cfk.path).replace("\\", "/"), "calibration_sha256": cfk.sha256,
            "calibration_created": cal.get("created"), "calibration_plant_variant": cal.get("plant_variant"),
            "raw_sweep": str(cfk.raw_path).replace("\\", "/"), "raw_sweep_sha256": sha256_file(cfk.raw_path),
            "usd_sha256": cal["usd_sha256"], "body_properties": str(body_props).replace("\\", "/"),
            "body_properties_sha256": sha256_file(body_props), "sole_hulls": str(sole_hulls).replace("\\", "/"),
            "reference_sample_for_linkage_bodies": ref_src,
        },
        "frames": {
            "world": "USD root body 'world' frame at the semantic zero (x fwd, y left, z up)",
            "pelvis_in_root": {"pos": pelvis_origin.tolist(), "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "model_zero": "all 22 joints 0 = semantic zero (legs straight, soles level, arms hanging, forearms "
                          "forward: G1 elbow 0); body frames are world-aligned there except knee/elbow/wrist-roll links "
                          "(bodies[*].frame_rotation_world0), rotated so every joint axis is an integer unit vector",
            "root_conversion": "T_world(USD root) = T_pelvis * inv(pelvis_in_root)",
            "pelvis_height_above_sole_at_zero_m": hip_height,
        },
        "fit": fit, "anchor_reference_points": {s: {k: v.tolist() for k, v in d.items()} for s, d in anchor_ref.items()},
        "segment_assignment": {b: {"segment": assign[b], "serial_body": seg_to_serial[assign[b]]}
                               for b in raw.body_names if b not in ORPHAN_BODIES},
        "excluded_bodies": list(ORPHAN_BODIES),
        "placeholder_bodies": placeholders,
        "surrogate_actuators": {"method": "J = d(motor)/d(semantic) at standing_semantic_pos (SemanticMap, finite "
                                          "differences); effort = sum|J_ji| effort_j, kp = sum J_ji^2 kp_j, kd, "
                                          "armature likewise; legacy DROPBEAR_CFG motor groups",
                                "joints": act},
        "calibration_findings": list(cal.get("findings", [])),
        "fk_model_findings": cfk.findings,
    }
    spec.meta["joints"] = {jn: {"semantic_index": si[jn], "body": link_name(jn.split("_", 1)[0], jn.split("_", 1)[1]),
                                "axis": np.asarray(j["axis"]).tolist(), "anchor_world0": np.asarray(j["anchor"]).tolist(),
                                "range": np.asarray(j["range"]).tolist()} for jn, j in joints.items()}
    spec.meta["_sites"] = sites
    spec.meta["_geoms"] = geoms
    spec.meta["_joints"] = joints
    spec.meta["_act"] = act
    spec.meta["_meshes"] = meshes
    spec.meta["total_mass_kg"] = float(sum(b["inertial"]["mass"] for b in spec.bodies.values() if b.get("inertial")))
    log(f"[serial_model] fitted: total mass {spec.meta['total_mass_kg']:.3f} kg, "
        f"reference sample: {ref_src}, placeholders {placeholders}")
    return spec


# ------------------------------------------------------------------------------------------------------
def _fmt(v, nd=8) -> str:
    return " ".join(f"{float(x):.{nd}f}".rstrip("0").rstrip(".") if abs(float(x)) > 1e-12 else "0" for x in np.ravel(v))


def _quat_attr(r: np.ndarray) -> str:
    if np.abs(r - np.eye(3)).max() < 1e-12:
        return ""
    return f' quat="{_fmt(mat_to_quat(r), 12)}"'


def _mesh_local(spec: SerialSpec, mesh: dict) -> np.ndarray:
    bd = spec.bodies[mesh["body"]]
    return (mesh["vertices_world0"] - np.asarray(bd["origin"], dtype=np.float64)) @ bd["R0"]


def _hull_triangles(vertices: np.ndarray) -> np.ndarray | None:
    """Triangles (F, 3, 3) of the convex hull of ``vertices`` with outward winding; None if degenerate."""
    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(vertices)
    except Exception:
        return None
    tri = vertices[hull.simplices]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("fi,fi->f", n, tri.mean(1) - vertices.mean(0)) < 0
    tri[flip] = tri[flip][:, ::-1]
    return tri


def _write_stl(path: Path, tri: np.ndarray) -> None:
    """Binary STL of triangles (F, 3, 3)."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as f:
        f.write(b"dropbear_serial derived hulls".ljust(80, b" "))
        f.write(np.uint32(len(tri)).tobytes())
        f.write(rec.tobytes())


def _mjcf_text(spec: SerialSpec, joints: dict, sites: dict, geoms: list, act: dict, variant: str,
               meshes: list | None = None) -> str:
    """MJCF text. ``variant`` = ``full`` (GMR / MuJoCo / mjlab: welded ``torso_link`` + ``head_link`` bodies,
    position servos) or ``motionlib`` (PHC-style ``Humanoid_Batch`` parsers used by SONIC motion_lib and
    ProtoMotions: every non-root body has exactly one hinge, no jointless bodies, ``<motor>`` actuators)."""
    b = spec.bodies
    hip_h = spec.meta["frames"]["pelvis_height_above_sole_at_zero_m"]
    keep = [n for n in spec.order if variant == "full" or not b[n].get("welded")]
    children: dict[str, list[str]] = {n: [] for n in keep}
    for n in keep:
        if b[n]["parent"]:
            children[b[n]["parent"]].append(n)
    site_body = {}
    for sn, st in sites.items():
        tgt = st["body"]
        if tgt not in children:  # welded body dropped in the motionlib variant: its sites go to the pelvis
            tgt = "pelvis"
        site_body[sn] = tgt
    lines: list[str] = []
    add = lines.append
    add(f'<mujoco model="{MODEL_NAME}{"" if variant == "full" else "_motionlib"}">')
    add(f"  <!-- {DERIVED_LABEL}")
    add(f"       Generated by tools/build_serial_mjcf.py from {spec.meta['inputs']['calibration']}")
    add(f"       (sha256 {spec.meta['inputs']['calibration_sha256'][:16]}..., "
        f"created {spec.meta['inputs']['calibration_created']}).")
    add("       Joint values are dropbear-semantic-v1 angles (docs/CONTRACTS.md section 2); see dropbear_serial.json")
    add("       and docs/SERIAL_MODEL_AND_GMR.md for what is exact and what is fitted. Do not edit by hand.")
    if variant == "motionlib":
        add("       VARIANT motionlib: no jointless bodies (torso/head are sites on the pelvis; add them as extend_config")
        add("       entries, see dropbear_serial.json 'motionlib_extend_config'), <motor> actuators (ctrlrange = effort).")
    add("  -->")
    meshes = meshes or []
    meshdir = ' meshdir="meshes"' if meshes else ''
    add(f'  <compiler angle="radian" inertiafromgeom="false" autolimits="true"{meshdir}/>')
    add('  <option timestep="0.005" integrator="implicitfast"/>')
    add("  <default>")
    add('    <joint damping="0.5" frictionloss="0"/>')
    add('    <geom contype="0" conaffinity="0" group="2" rgba="0.62 0.64 0.68 1"/>')
    add('    <site size="0.012" group="4" rgba="0.9 0.2 0.2 1"/>')
    add('    <default class="collision">')
    add('      <geom contype="1" conaffinity="0" group="3" condim="3" friction="1.0 0.005 0.0001" rgba="0.2 0.5 0.8 0.5"/>')
    add("    </default>")
    add('    <default class="visual">')
    add('      <geom contype="0" conaffinity="0" group="2"/>')
    add("    </default>")
    add("  </default>")
    if meshes:
        add("  <asset>")
        for mh in meshes:
            add(f'    <mesh name="{mh["name"]}" file="{mh["file"]}"/>')
        add("  </asset>")
    add("  <worldbody>")

    def emit(name: str, indent: int) -> None:
        bd = b[name]
        pad = " " * indent
        rb = bd["R0"]
        ob = np.asarray(bd["origin"], dtype=np.float64)
        if bd["parent"]:
            pr = b[bd["parent"]]
            rp, op = pr["R0"], np.asarray(pr["origin"], dtype=np.float64)
            pos, rel = rp.T @ (ob - op), rp.T @ rb
        else:
            pos, rel = np.array([0.0, 0.0, hip_h]), rb
        add(f'{pad}<body name="{name}" pos="{_fmt(pos)}"{_quat_attr(rel)}>')
        if bd["parent"] is None:
            add(f'{pad}  <freejoint name="floating_base_joint"/>')
        inn = bd.get("inertial")
        if inn:
            ii = rb.T @ inn["inertia"] @ rb
            add(f'{pad}  <inertial pos="{_fmt(rb.T @ (inn["com"] - ob))}" mass="{inn["mass"]:.6f}" '
                f'fullinertia="{_fmt([ii[0, 0], ii[1, 1], ii[2, 2], ii[0, 1], ii[0, 2], ii[1, 2]], 9)}"/>')
        if bd["joint"]:
            jn = bd["joint"]
            j = joints[jn]
            a = act[jn]
            ax = " ".join(str(int(round(v))) for v in bd["axis_local"])
            add(f'{pad}  <joint name="{jn}" axis="{ax}" range="{_fmt(j["range"], 9)}" '
                f'armature="{a["armature"]:.5f}" actuatorfrcrange="{-a["effort"]:.2f} {a["effort"]:.2f}"/>')
        for g in geoms:
            if g["body"] != name:
                continue
            if g["type"] == "box":
                add(f'{pad}  <geom name="{g["name"]}" class="{g["class"]}" type="box" '
                    f'pos="{_fmt(rb.T @ (g["center_world0"] - ob))}"{_quat_attr(rb.T)} '
                    f'size="{_fmt(np.maximum(g["half"], 1e-3))}"/>')
            else:
                fr, to = rb.T @ (g["from_world0"] - ob), rb.T @ (g["to_world0"] - ob)
                add(f'{pad}  <geom name="{g["name"]}" class="{g["class"]}" type="capsule" size="{g["radius"]:.3f}" '
                    f'fromto="{_fmt(np.concatenate([fr, to]))}"/>')
        for mh in meshes:
            if mh["body"] == name:
                add(f'{pad}  <geom name="{mh["name"]}" class="visual" type="mesh" mesh="{mh["name"]}" group="1" '
                    f'rgba="0.55 0.57 0.62 1"/>')
        for sn, st in sites.items():
            if site_body[sn] != name:
                continue
            q = mat_to_quat(rb.T @ st["rot_world0"])
            add(f'{pad}  <site name="{sn}" pos="{_fmt(rb.T @ (st["pos_world0"] - ob))}" quat="{_fmt(q, 10)}"/>')
        for c in children[name]:
            emit(c, indent + 2)
        add(f"{pad}</body>")

    emit("pelvis", 4)
    add("  </worldbody>")
    # comments stay OUTSIDE <actuator>: SONIC's Humanoid_Batch iterates actuator children with lxml getchildren(),
    # which returns comments too (KeyError 'name'; found by tools/check_sonic_humanoid_batch.py)
    if variant == "full":
        add("  <!-- Surrogate semantic-space position servos (effort/kp/kd mapped from the motor PD by J = dm/ds at")
        add("       the standing pose). For kinematic tools they are irrelevant; for simulation they are a rough")
        add("       surrogate only: the plant's torque limits live in motor space. -->")
    else:
        add("  <!-- Torque motors (ctrlrange = surrogate semantic-space effort limit), one per joint, tree order. -->")
    add("  <actuator>")
    for n in keep:
        jn = b[n]["joint"]
        if not jn:
            continue
        a = act[jn]
        if variant == "full":
            r = joints[jn]["range"]
            add(f'    <general name="{jn}" joint="{jn}" ctrlrange="{_fmt(r, 9)}" '
                f'forcerange="{-a["effort"]:.2f} {a["effort"]:.2f}" gainprm="{a["kp"]:.3f}" biastype="affine" '
                f'biasprm="0 {-a["kp"]:.3f} {-a["kd"]:.4f}"/>')
        else:
            add(f'    <motor name="{jn}" joint="{jn}" ctrlrange="{-a["effort"]:.2f} {a["effort"]:.2f}"/>')
    add("  </actuator>")
    add("</mujoco>")
    return "\n".join(lines) + "\n"


def write_mjcf(spec: SerialSpec, xml_path: Path | str = DEFAULT_XML, meta_path: Path | str = DEFAULT_META,
               scene_path: Path | str | None = DEFAULT_SCENE, motionlib_path: Path | str | None = None) -> dict:
    """Write the robot MJCF, its motion-lib variant, the scene wrapper and the metadata JSON. Returns the metadata."""
    xml_path, meta_path = Path(xml_path), Path(meta_path)
    motionlib_path = Path(motionlib_path) if motionlib_path else xml_path.with_name(xml_path.stem + "_motionlib.xml")
    b = spec.bodies
    joints, sites, geoms, act = (spec.meta.pop("_joints"), spec.meta.pop("_sites"), spec.meta.pop("_geoms"),
                                 spec.meta.pop("_act"))
    raw_meshes = spec.meta.pop("_meshes", [])
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir = xml_path.parent / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    for old_stl in mesh_dir.glob("*.stl"):  # regenerate the whole set
        old_stl.unlink()
    # ONE mesh per serial body (the union of its USD members' convex hulls): PHC/SONIC mesh_fk adds every mesh of a
    # body once per mesh geom of that body, so several mesh geoms per body would be duplicated quadratically
    per_body: dict[str, list] = {}
    for mh in raw_meshes:
        body = mh["body"] if not b[mh["body"]].get("welded") else "pelvis"
        tri = _hull_triangles(_mesh_local(spec, dict(mh, body=body)))
        if tri is not None:
            per_body.setdefault(body, []).append((mh["usd_body"], tri))
    meshes = []
    for body in spec.order:
        if body not in per_body:
            continue
        tri = np.concatenate([t for _, t in per_body[body]])
        name = f"mesh_{body}"
        _write_stl(mesh_dir / f"{name}.stl", tri)
        meshes.append({"name": name, "body": body, "file": f"{name}.stl", "usd_bodies": [u for u, _ in per_body[body]],
                       "triangles": int(len(tri))})
    xml_path.write_text(_mjcf_text(spec, joints, sites, geoms, act, "full", meshes), encoding="utf-8")
    motionlib_path.write_text(_mjcf_text(spec, joints, sites, geoms, act, "motionlib", meshes), encoding="utf-8")
    if scene_path is not None:
        scene = [
            '<mujoco model="dropbear_serial_scene">',
            f"  <!-- Scene wrapper for {xml_path.name} (floor + light). {DERIVED_LABEL} -->",
            f'  <include file="{xml_path.name}"/>',
            '  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/></visual>',
            "  <asset>",
            '    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" '
            'width="512" height="512"/>',
            '    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.1"/>',
            "  </asset>",
            "  <worldbody>",
            '    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>',
            '    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="1" group="0"/>',
            "  </worldbody>",
            "</mujoco>",
        ]
        Path(scene_path).write_text("\n".join(scene) + "\n", encoding="utf-8")
    meta = spec.meta
    meta["visual_meshes"] = {"dir": str(mesh_dir).replace("\\", "/"), "count": len(meshes),
                             "source": "per serial body: union of the convex hulls of its USD members' collision hulls "
                                       "(usd_body_properties), baked into the body frame at the model zero; visual "
                                       "only (contype 0)",
                             "meshes": {mh["name"]: {"usd_bodies": mh["usd_bodies"], "triangles": mh["triangles"]}
                                        for mh in meshes}}
    meta["files"] = {"xml": str(xml_path).replace("\\", "/"), "motionlib_xml": str(motionlib_path).replace("\\", "/"),
                     "scene": str(scene_path).replace("\\", "/") if scene_path else None}
    meta["sites"] = {sn: {"body": st["body"], "usd_body": st["usd_body"], "label": st["label"]} for sn, st in sites.items()}
    meta["bodies"] = {n: {"parent": b[n]["parent"], "joint": b[n]["joint"], "physical_segment": b[n].get("phys"),
                          "origin_world0": np.asarray(b[n]["origin"]).tolist(),
                          "frame_rotation_world0": np.asarray(b[n]["R0"]).tolist(),
                          "welded": bool(b[n].get("welded", False)),
                          "mass": (b[n]["inertial"]["mass"] if b[n].get("inertial") else 0.0),
                          "usd_members": (b[n]["inertial"]["members"] if b[n].get("inertial") else [])}
                      for n in spec.order}
    # extend_config entries (PHC / SONIC motion_lib convention: joint_name, parent_name, pos, rot wxyz in the parent
    # body frame) for the key USD bodies, parented to serial bodies that exist in the motionlib variant
    ext = []
    for sn, st in sites.items():
        if st["usd_body"] is None:
            continue
        parent = st["body"] if not b.get(st["body"], {}).get("welded") else "pelvis"
        pb = b[parent]
        rp, op = pb["R0"], np.asarray(pb["origin"], dtype=np.float64)
        ext.append({"joint_name": f"usd_{st['label']}", "parent_name": parent,
                    "pos": (rp.T @ (st["pos_world0"] - op)).tolist(),
                    "rot": mat_to_quat(rp.T @ st["rot_world0"]).tolist(), "usd_body": st["usd_body"]})
    meta["motionlib_extend_config"] = ext
    meta["xml_sha256"] = sha256_file(xml_path)
    meta["motionlib_xml_sha256"] = sha256_file(motionlib_path)
    Path(meta_path).write_text(json.dumps(meta, indent=1, default=lambda o: np.asarray(o).tolist()), encoding="utf-8")
    return meta


# ------------------------------------------------------------------------------------------------------
class SerialModel:
    """Runtime helper around the built MJCF (needs ``mujoco``).

    Maps between the semantic vector (22, ``SEMANTIC_NAMES`` order) and MuJoCo ``qpos`` (free root on
    ``pelvis`` + 22 hinges in tree order), converts the root between the pelvis frame and the USD root body
    ``world``, and evaluates FK of bodies and sites.
    """

    def __init__(self, xml: str | Path = DEFAULT_XML, meta: str | Path | None = None):
        import mujoco

        self.mj = mujoco
        self.xml = Path(xml)
        self.meta = json.loads(Path(meta or self.xml.with_suffix(".json")).read_text())
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        self.data = mujoco.MjData(self.model)
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in SEMANTIC_NAMES]
        if min(jid) < 0:
            raise RuntimeError("MJCF lacks some semantic joints")
        self.sem_qadr = np.array([self.model.jnt_qposadr[j] for j in jid])
        self.sem_dadr = np.array([self.model.jnt_dofadr[j] for j in jid])
        pel = self.meta["frames"]["pelvis_in_root"]
        self.pelvis_in_root = np.eye(4)
        self.pelvis_in_root[:3, :3] = quat_mat(np.asarray(pel["quat_wxyz"], dtype=np.float64))
        self.pelvis_in_root[:3, 3] = pel["pos"]

    def qpos(self, pelvis_pos: np.ndarray, pelvis_quat_wxyz: np.ndarray, q_sem: np.ndarray) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[0:3] = pelvis_pos
        q[3:7] = pelvis_quat_wxyz
        q[self.sem_qadr] = q_sem
        return q

    def split(self, qpos: np.ndarray):
        qpos = np.asarray(qpos)
        return qpos[..., 0:3], qpos[..., 3:7], qpos[..., self.sem_qadr]

    def root_from_pelvis(self, pelvis_pos: np.ndarray, pelvis_quat_wxyz: np.ndarray):
        """USD root ('world' body) pose from the pelvis pose (vectorised). Returns (pos, quat_wxyz)."""
        r = quat_mat(pelvis_quat_wxyz)
        inv = np.linalg.inv(self.pelvis_in_root)
        pos = pelvis_pos + _mv(r, inv[:3, 3])
        rr = r @ inv[:3, :3]
        return pos, mat_to_quat(rr)

    def fk(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        self.mj.mj_kinematics(self.model, self.data)

    def site_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        i = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, name)
        if i < 0:
            raise KeyError(name)
        return self.data.site_xmat[i].reshape(3, 3).copy(), self.data.site_xpos[i].copy()

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        i = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise KeyError(name)
        return self.data.xmat[i].reshape(3, 3).copy(), self.data.xpos[i].copy()


# ------------------------------------------------------------------------------------------------------
def evaluate_parity(smod: "SerialModel", targets: dict[str, tuple[np.ndarray, np.ndarray]], q_sem: np.ndarray,
                    pelvis_pos: np.ndarray | None = None) -> dict[str, dict]:
    """Position/orientation errors of the key-body sites vs reference poses (root-relative).

    Args:
        smod: loaded serial model.
        targets: {USD body name: (R (N,3,3), p (N,3))} reference poses in the USD root frame.
        q_sem: (N, 22) semantic poses (SEMANTIC_NAMES order) to put the serial model in.
    Returns {label: {"usd_body", "site", "position": stats_mm, "orientation_deg": {...}, "frac_below_10mm"}}.
    """
    q_sem = np.asarray(q_sem, dtype=np.float64)
    n = len(q_sem)
    pel = np.asarray(smod.meta["frames"]["pelvis_in_root"]["pos"]) if pelvis_pos is None else pelvis_pos
    site_of = {st["usd_body"]: (sn, st["label"]) for sn, st in smod.meta["sites"].items() if st["usd_body"]}
    names = [b for b in targets if b in site_of]
    pos = {b: np.zeros((n, 3)) for b in names}
    rot = {b: np.zeros((n, 3, 3)) for b in names}
    for k in range(n):
        smod.fk(smod.qpos(pel, np.array([1.0, 0.0, 0.0, 0.0]), q_sem[k]))
        for b in names:
            rot[b][k], pos[b][k] = smod.site_pose(site_of[b][0])
    out: dict[str, dict] = {}
    for b in names:
        rt, pt = targets[b]
        dp = np.linalg.norm(pos[b] - pt, axis=-1)
        da = np.degrees(np.linalg.norm(rotvec(np.swapaxes(rot[b], -1, -2) @ rt), axis=-1))
        out[site_of[b][1]] = {"usd_body": b, "site": site_of[b][0], "position": _stats_mm(dp),
                              "orientation_deg": {"p50": float(np.median(da)), "p95": float(np.percentile(da, 95)),
                                                  "max": float(da.max())},
                              "frac_below_10mm": float(np.mean(dp < 0.010))}
    return out
