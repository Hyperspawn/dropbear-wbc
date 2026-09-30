"""Fit the semantic calibration from raw physics sweeps (CPU, numpy/scipy).

Input: the raw NPZ of ``tools/calibrate_semantics.py --stage sweep`` (link poses of all 93 bodies relative
to the root link ``world`` after quasi-static settling, for 1D motor sweeps and the two 2D calf-motor
grids), plus joint frames read from the live stage.

Output: ``dropbear-semantic-calibration-v1`` JSON (see :mod:`dropbear_wbc.kinematics.semantic`).

Definitions (all rotations are root-relative; the root frame is the pelvis frame x fwd / y left / z up):

* Segment bodies: thigh = hip-pitch child, shank = knee four-bar output bracket, foot = sole plate,
  upper arm = shoulder-yaw child, forearm = elbow four-bar output, hand = wrist-roll child.
* Measured forward model (:class:`MeasuredFK`): hips / shoulders / wrists are serial hinges, modelled by
  products of exponentials with screw axes fitted to the 1D sweeps; the closed-loop relative transforms
  thigh->shank (knee crank), upper arm->forearm (elbow motor) and shank->foot (two calf motors) are
  interpolated from the sweep tables. The model is validated against the physics by the ``verify`` stage.
* Leg semantic zero ``q_zero`` (per side): knee crank at maximal hip-ankle distance (straightest leg),
  hip (roll, yaw, pitch) motors such that the ankle centre is directly below the hip centre and the foot
  heading is forward, calf motors such that the sole is level. Arm semantic zero: authored rest (upper arm
  vertical, elbow axis along y), elbow at G1 0 (clipped to the reachable range), wrist 0.
* Frames: ``F_X = R_X R_X(ref)^T`` with ref = ``q_zero`` for leg segments and rest for arm segments.
  hip / shoulder (pitch, roll, yaw) = intrinsic YXZ Euler of ``F_thigh`` / ``F_upper``;
  knee = signed rotation angle of ``F_thigh^T F_shank`` (sign: axis . +y);
  ankle (pitch, roll) = YXZ Euler of ``F_shank^T F_foot`` (the residual yaw is reported);
  elbow = pi/2 + signed angle of ``F_upper^T F_forearm`` about +y (G1 convention, straight arm = pi/2);
  wrist roll = signed angle of ``F_forearm^T F_hand`` about the elbow->hand direction.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dropbear_wbc.kinematics.closures_np import ClosureTable, closure_residuals
from dropbear_wbc.kinematics.rigid import (axis_angle_matrix, euler_yxz, fit_screw_axis, mat_to_quat, quat_mat,
                                           rotvec, signed_angle_about)
from dropbear_wbc.kinematics.semantic import MOTOR_NAMES, SCHEMA, SEMANTIC_NAMES

DEG = np.pi / 180.0
Y = np.array([0.0, 1.0, 0.0])
SIDES = {"left": ("LL", "PG_left_leg", "LH"), "right": ("RL", "PG_right_leg", "RH")}


def seg_bodies(side: str) -> dict[str, str]:
    leg, _, arm = SIDES[side]
    return {
        "thigh": f"{leg}_RMD_X10_S2_MIR4__3_Stator_1", "thigh_strap": f"{leg}_bearingstrap_1",
        "shank": f"{leg}_double_bracket_10deg_MIR_MIR_MIR_1", "foot": f"{leg}_skateboard_bearing_left_2",
        "ankle_cross": f"{leg}_basis_left_1",
        "upper_arm": f"{arm}_RMD_X8_Pro_MIR8_MIR1__3__1", "forearm": f"{arm}_6mm_bearing__4__1",
        "hand": f"{arm}_shoulder_ex_al_interface_1",
    }


def seg_motors(side: str) -> dict[str, str]:
    leg, pg, arm = SIDES[side]
    return {
        "hip_roll": f"{pg}_pitch", "hip_yaw": f"{pg}_roll", "hip_pitch": f"{leg}_hip_joint",
        "knee": f"{leg}_knee_actuator_joint", "calf_a": f"{leg}_Revolute67", "calf_b": f"{leg}_Revolute81",
        "shoulder_pitch": f"{arm}_yaw", "shoulder_roll": f"{arm}_pitch", "shoulder_yaw": f"{arm}_roll",
        "elbow": f"{arm}_elbow_joint", "wrist_roll": f"{arm}_wrist_roll",
    }


# ------------------------------------------------------------------------------------------------------
@dataclass
class Poses:
    """Root-relative link poses: R (..., B, 3, 3), p (..., B, 3)."""

    r: np.ndarray
    p: np.ndarray


class Raw:
    """Accessor for a raw sweep / verify NPZ."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        d = np.load(self.path, allow_pickle=False)
        self.d = {k: d[k] for k in d.files}
        self.body_names = [str(x) for x in self.d["body_names"]]
        self.joint_names = [str(x) for x in self.d["joint_names"]]
        self.program_names = [str(x) for x in self.d["program_names"]]
        self.bidx = {n: i for i, n in enumerate(self.body_names)}
        self.midx = {n: i for i, n in enumerate(MOTOR_NAMES)}
        self.r = quat_mat(self.d["body_quat"])  # (R, B, 3, 3)
        self.p = self.d["body_pos"]
        self.motor_pos = self.d["motor_pos"]
        self.motor_target = self.d["motor_target"]
        self.program = self.d["program"]
        self.passno = self.d["passno"]
        self.closures = ClosureTable.from_npz(self.d)
        self.gaps, self.angs = closure_residuals(self.p, self.d["body_quat"], self.closures)
        self.jf = {str(n): i for i, n in enumerate(self.d["jf_names"])} if "jf_names" in self.d else {}

    def usd_sha(self) -> str:
        return str(self.d["usd_sha256"])

    def rec(self, program_name: str, passno: int | None = 0) -> np.ndarray:
        pid = self.program_names.index(program_name)
        sel = self.program == pid
        if passno is not None:
            sel &= self.passno == passno
        idx = np.nonzero(sel)[0]
        return idx[np.argsort(self.d["sample"][idx])]

    def R(self, body: str, idx) -> np.ndarray:
        return self.r[idx, self.bidx[body]]

    def P(self, body: str, idx) -> np.ndarray:
        return self.p[idx, self.bidx[body]]

    def anchor_local(self, joint: str, which: int) -> tuple[str, np.ndarray]:
        """(body name, local anchor position) of joint frame ``which`` (0/1)."""
        j = self.jf[joint]
        b = int(self.d[f"jf_body{which}"][j])
        return self.body_names[b], np.asarray(self.d[f"jf_local_pos{which}"][j], dtype=np.float64)

    def axis_local(self, joint: str, which: int) -> np.ndarray:
        j = self.jf[joint]
        e = {"X": [1.0, 0, 0], "Y": [0, 1.0, 0], "Z": [0, 0, 1.0]}[str(self.d["jf_axis"][j])]
        return quat_mat(np.asarray(self.d[f"jf_local_rot{which}"][j])) @ np.asarray(e)


# ------------------------------------------------------------------------------------------------------
def _interp_transforms(x_grid: np.ndarray, r_tab: np.ndarray, p_tab: np.ndarray, x: np.ndarray):
    """Piecewise interpolation of rigid transforms tabulated at increasing ``x_grid`` (nlerp + lerp)."""
    x = np.clip(np.asarray(x, dtype=np.float64), x_grid[0], x_grid[-1])
    i = np.clip(np.searchsorted(x_grid, x) - 1, 0, len(x_grid) - 2)
    t = (x - x_grid[i]) / (x_grid[i + 1] - x_grid[i])
    q = mat_to_quat(r_tab)
    q0, q1 = q[i], q[i + 1]
    q1 = np.where(np.sum(q0 * q1, -1, keepdims=True) < 0, -q1, q1)
    qi = q0 * (1 - t)[..., None] + q1 * t[..., None]
    return quat_mat(qi / np.linalg.norm(qi, axis=-1, keepdims=True)), p_tab[i] * (1 - t)[..., None] + p_tab[i + 1] * t[..., None]


def _interp_transforms_2d(a_grid, b_grid, r_tab, p_tab, a, b):
    """Bilinear (nlerp) interpolation of transforms tabulated on a regular (a, b) grid (na, nb, ...)."""
    a = np.clip(np.asarray(a, dtype=np.float64), a_grid[0], a_grid[-1])
    b = np.clip(np.asarray(b, dtype=np.float64), b_grid[0], b_grid[-1])
    ia = np.clip(np.searchsorted(a_grid, a) - 1, 0, len(a_grid) - 2)
    ib = np.clip(np.searchsorted(b_grid, b) - 1, 0, len(b_grid) - 2)
    ta = (a - a_grid[ia]) / (a_grid[ia + 1] - a_grid[ia])
    tb = (b - b_grid[ib]) / (b_grid[ib + 1] - b_grid[ib])
    q = mat_to_quat(r_tab)
    ref = q[ia, ib]
    acc_q = np.zeros(np.shape(a) + (4,))
    acc_p = np.zeros(np.shape(a) + (3,))
    for da, db, w in ((0, 0, (1 - ta) * (1 - tb)), (1, 0, ta * (1 - tb)), (0, 1, (1 - ta) * tb), (1, 1, ta * tb)):
        qq = q[ia + da, ib + db]
        qq = np.where(np.sum(qq * ref, -1, keepdims=True) < 0, -qq, qq)
        acc_q += qq * w[..., None]
        acc_p += p_tab[ia + da, ib + db] * w[..., None]
    return quat_mat(acc_q / np.linalg.norm(acc_q, axis=-1, keepdims=True)), acc_p


class MeasuredFK:
    """Forward kinematics of the key segments from the sweeps (root-relative, see module doc)."""

    def __init__(self, raw: Raw, findings: list[str], max_gap: float = 3e-3):
        self.raw = raw
        self.findings = findings
        self.max_gap = max_gap
        self.rest_idx = self._rest_record()
        self.screws: dict[str, dict] = {}
        self.loops: dict[str, dict] = {}
        self.grids: dict[str, dict] = {}
        for side in SIDES:
            sb, sm = seg_bodies(side), seg_motors(side)
            for j in ("hip_roll", "hip_yaw", "hip_pitch"):
                self.screws[sm[j]] = self._fit_serial(sm[j], sb["thigh"], None)
            for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw"):
                self.screws[sm[j]] = self._fit_serial(sm[j], sb["upper_arm"], None)
            self.screws[sm["wrist_roll"]] = self._fit_serial(sm["wrist_roll"], sb["hand"], sb["forearm"])
            self.loops[sm["knee"]] = self._loop_table(sm["knee"], sb["thigh"], sb["shank"])
            self.loops[sm["elbow"]] = self._loop_table(sm["elbow"], sb["upper_arm"], sb["forearm"])
            self.grids[side] = self._grid_table(side, sb["shank"], sb["foot"])

    # -- construction ---------------------------------------------------------------------------------
    def _rest_record(self) -> int:
        """The ``rest`` program's record (all motors 0) -> the authored rest configuration."""
        if "rest" in self.raw.program_names:
            idx = self.raw.rec("rest")
            if len(idx):
                return int(idx[-1])
        err = np.abs(self.raw.motor_pos).max(axis=1)
        i = int(np.argmin(err))
        if err[i] > 0.5 * DEG:
            raise RuntimeError(f"no rest record found (closest max|m| = {err[i]:.4f} rad)")
        return i

    def rest(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        return self.raw.R(body, self.rest_idx), self.raw.P(body, self.rest_idx)

    def _fit_serial(self, motor: str, body: str, parent: str | None) -> dict:
        idx = self.raw.rec(f"sweep1d:{motor}")
        m = self.raw.motor_pos[idx, self.raw.midx[motor]]
        idx_all = np.concatenate([[self.rest_idx], idx])
        r, p = self.raw.R(body, idx_all), self.raw.P(body, idx_all)
        if parent is not None:  # express in the parent's frame
            rp, pp = self.raw.R(parent, idx_all), self.raw.P(parent, idx_all)
            r = np.swapaxes(rp, -1, -2) @ r
            p = np.einsum("nji,nj->ni", rp, p - pp)
        # reference = the authored rest record (all motors 0)
        r_rel = r[1:] @ r[0].T
        p_rel = p[1:] - np.einsum("nij,j->ni", r_rel, p[0])
        fit = fit_screw_axis(r_rel, p_rel, m)
        fit.update(motor=motor, body=body, parent=parent, m0=0.0, n=int(len(idx)),
                   range=[float(m.min()), float(m.max())])
        return fit

    def _loop_table(self, motor: str, parent: str, child: str) -> dict:
        """Relative transform parent->child vs achieved motor position (forward pass, sorted)."""
        out = {}
        for pas in (0, 1):
            idx = self.raw.rec(f"sweep1d:{motor}", passno=pas)
            if len(idx) == 0:
                continue
            m = self.raw.motor_pos[idx, self.raw.midx[motor]]
            rp, pp = self.raw.R(parent, idx), self.raw.P(parent, idx)
            rc, pc = self.raw.R(child, idx), self.raw.P(child, idx)
            r = np.swapaxes(rp, -1, -2) @ rc
            p = np.einsum("nji,nj->ni", rp, pc - pp)
            order = np.argsort(m)
            out[pas] = {"m": m[order], "r": r[order], "p": p[order], "idx": idx[order],
                        "target": self.raw.motor_target[idx[order], self.raw.midx[motor]],
                        "dq": self.raw.d["dq_window"][idx[order]], "gap": self.raw.gaps[idx[order]].max(-1),
                        "err": np.abs(self.raw.motor_target[idx[order], self.raw.midx[motor]] - m[order])}
        ok = out[0]["gap"] < self.max_gap
        g = {k: v[ok] for k, v in out[0].items()}
        keep = np.concatenate([[True], np.diff(g["m"]) > 1e-5])
        out["grid"] = {k: v[keep] for k, v in g.items()}
        out["n_invalid_gap"] = int((~ok).sum())
        return out

    def _grid_table(self, side: str, shank: str, foot: str) -> dict:
        names = [n for n in self.raw.program_names if n.startswith(f"grid2d:{SIDES[side][0]}:")]
        a_vals = np.array([float(n.split(":")[-1]) for n in names])
        order = np.argsort(a_vals)
        rows = [self.raw.rec(names[i]) for i in order]
        nb = min(len(r) for r in rows)
        idx = np.stack([r[:nb] for r in rows])  # (na, nb)
        sm = seg_motors(side)
        ia, ib = self.raw.midx[sm["calf_a"]], self.raw.midx[sm["calf_b"]]
        rs, ps = self.raw.R(shank, idx), self.raw.P(shank, idx)
        rf, pf = self.raw.R(foot, idx), self.raw.P(foot, idx)
        r = np.swapaxes(rs, -1, -2) @ rf
        p = np.einsum("...ji,...j->...i", rs, pf - ps)
        tgt_a, tgt_b = self.raw.motor_target[idx, ia], self.raw.motor_target[idx, ib]
        ach_a, ach_b = self.raw.motor_pos[idx, ia], self.raw.motor_pos[idx, ib]
        return {"idx": idx, "a_grid": tgt_a[:, 0], "b_grid": tgt_b[0, :], "r": r, "p": p,
                "track_err": np.maximum(np.abs(ach_a - tgt_a), np.abs(ach_b - tgt_b)),
                "gap": self.raw.gaps[idx].max(-1), "dq": self.raw.d["dq_window"][idx],
                "ach_a": ach_a, "ach_b": ach_b}

    # -- evaluation -------------------------------------------------------------------------------------
    def _poe(self, motors: list[str], values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Product of exponentials exp(xi_1 q_1) ... exp(xi_n q_n) (root frame)."""
        shape = np.broadcast_shapes(*[np.shape(v) for v in values])
        r = np.broadcast_to(np.eye(3), shape + (3, 3)).copy()
        p = np.zeros(shape + (3,))
        for mname, q in zip(motors, values):
            s = self.screws[mname]
            ang = s["slope"] * (np.asarray(q) - s["m0"])
            rj = axis_angle_matrix(np.broadcast_to(s["axis"], np.shape(ang) + (3,)), ang)
            pj = s["point"] - np.einsum("...ij,j->...i", rj, s["point"])
            p = p + np.einsum("...ij,...j->...i", r, pj)
            r = r @ rj
        return r, p

    def leg(self, side: str, m: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Root-relative (R, p) of thigh, shank, foot for motor vectors ``m`` (..., 22)."""
        sb, sm = seg_bodies(side), seg_motors(side)
        g = lambda k: m[..., self.raw.midx[sm[k]]]  # noqa: E731
        r0, p0 = self._poe([sm["hip_roll"], sm["hip_yaw"], sm["hip_pitch"]], [g("hip_roll"), g("hip_yaw"), g("hip_pitch")])
        rs0, ps0 = self.rest(sb["thigh"])
        r_t, p_t = r0 @ rs0, p0 + np.einsum("...ij,j->...i", r0, ps0)
        lt = self.loops[sm["knee"]]["grid"]
        rk, pk = _interp_transforms(lt["m"], lt["r"], lt["p"], g("knee"))
        r_s, p_s = r_t @ rk, p_t + np.einsum("...ij,...j->...i", r_t, pk)
        gr = self.grids[side]
        ra, pa = _interp_transforms_2d(gr["a_grid"], gr["b_grid"], gr["r"], gr["p"], g("calf_a"), g("calf_b"))
        r_f, p_f = r_s @ ra, p_s + np.einsum("...ij,...j->...i", r_s, pa)
        return {"thigh": (r_t, p_t), "shank": (r_s, p_s), "foot": (r_f, p_f)}

    def arm(self, side: str, m: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        sb, sm = seg_bodies(side), seg_motors(side)
        g = lambda k: m[..., self.raw.midx[sm[k]]]  # noqa: E731
        r0, p0 = self._poe([sm["shoulder_pitch"], sm["shoulder_roll"], sm["shoulder_yaw"]],
                           [g("shoulder_pitch"), g("shoulder_roll"), g("shoulder_yaw")])
        ru0, pu0 = self.rest(sb["upper_arm"])
        r_u, p_u = r0 @ ru0, p0 + np.einsum("...ij,j->...i", r0, pu0)
        lt = self.loops[sm["elbow"]]["grid"]
        re, pe = _interp_transforms(lt["m"], lt["r"], lt["p"], g("elbow"))
        r_e, p_e = r_u @ re, p_u + np.einsum("...ij,...j->...i", r_u, pe)
        # wrist: hand relative to forearm, screw fitted in the forearm frame
        s = self.screws[sm["wrist_roll"]]
        ang = s["slope"] * (g("wrist_roll") - s["m0"])
        rw = axis_angle_matrix(np.broadcast_to(s["axis"], np.shape(ang) + (3,)), ang)
        pw = s["point"] - np.einsum("...ij,j->...i", rw, s["point"])
        rf0, pf0 = self.rest(sb["forearm"])
        rh0, ph0 = self.rest(sb["hand"])
        rh_rel0 = rf0.T @ rh0
        ph_rel0 = rf0.T @ (ph0 - pf0)
        r_hrel = rw @ rh_rel0
        p_hrel = pw + np.einsum("...ij,j->...i", rw, ph_rel0)
        r_h, p_h = r_e @ r_hrel, p_e + np.einsum("...ij,...j->...i", r_e, p_hrel)
        return {"upper_arm": (r_u, p_u), "forearm": (r_e, p_e), "hand": (r_h, p_h)}
