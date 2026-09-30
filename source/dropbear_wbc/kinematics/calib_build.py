"""Assemble ``dropbear-semantic-calibration-v1`` from the raw sweeps (CPU; numpy + scipy).

Steps (see :mod:`dropbear_wbc.kinematics.calib_fit` for the measured FK and
:mod:`dropbear_wbc.kinematics.calib_semantics` for the angle definitions):

1. Leg semantic zero per side: knee crank at maximal hip-ankle distance (straightest leg); hip roll / yaw
   / pitch and calf motors solved (bounded least squares on the measured FK) so that the ankle centre
   (``*_Revolute87`` anchor) is directly below the hip centre (``*_hip_joint`` anchor), the sole is level
   and the foot heads forward. Arms: authored rest (checked to be hanging straight), elbow at G1 0 clipped.
2. Per-DOF maps measured from the physics sweeps, expressed in the zero-referenced frames:
   ``linear`` (hips, shoulders, wrists; scale = least-squares slope, offset = zero-pose motor value,
   cross-talk reported), ``lut1d`` (knee crank, elbow), ``lut2d`` (calf motor pair).
3. Standing pose: semantic knee flexion ``standing_knee`` (clipped to 60% of the achievable range); hip
   pitch and calf motors solved so the ankle is below the hip and the sole is level; arms at rest with
   ``standing_elbow_flex`` rad of elbow flexion.
4. Geometry: key bodies, hip/knee/ankle/shoulder/elbow/wrist points, segment lengths, sole heights.
5. Optional verification (``--stage verify`` NPZ): physics-measured semantic angles vs the map's prediction.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np

from dropbear_wbc.kinematics.calib_fit import SIDES, MeasuredFK, Raw, seg_bodies, seg_motors
from dropbear_wbc.kinematics.calib_pair import build_pair
from dropbear_wbc.kinematics.calib_semantics import semantics_from_rotations
from dropbear_wbc.kinematics.rigid import euler_yxz, fit_screw_axis
from dropbear_wbc.kinematics.semantic import MOTOR_NAMES, SCHEMA, SEMANTIC_NAMES

DEG = math.pi / 180.0
STANDING_KNEE = 0.35  # rad, G1-like slight knee flexion (G1 default 0.669, H1 0.79)
STANDING_ELBOW_FLEX = 0.30  # rad of flexion from straight (G1 value pi/2 - 0.30)


def _lin_fit(x: np.ndarray, y: np.ndarray) -> dict:
    a = np.stack([x, np.ones_like(x)], -1)
    (s, c), *_ = np.linalg.lstsq(a, y, rcond=None)
    res = y - (s * x + c)
    return {"slope": float(s), "intercept": float(c), "rms_rad": float(np.sqrt(np.mean(res ** 2))),
            "max_abs_rad": float(np.max(np.abs(res))), "n": int(len(x))}


def _monotonic_run(y: np.ndarray) -> tuple[int, int]:
    """Longest index run [i, j) on which ``y`` is strictly monotonic."""
    best = (0, 1)
    for sign in (1.0, -1.0):
        start = 0
        for k in range(1, len(y) + 1):
            if k == len(y) or sign * (y[k] - y[k - 1]) <= 1e-7:
                if k - start > best[1] - best[0]:
                    best = (start, k)
                start = k
    return best


def _sanitize(o):
    """JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class Builder:
    def __init__(self, raw_path, sole_path, findings: list[str], max_gap: float = 3e-3, dq_tol: float = 5e-3):
        self.raw = Raw(raw_path)
        self.findings = findings
        self.max_gap = max_gap
        self.dq_tol = dq_tol
        self.fk = MeasuredFK(self.raw, findings, max_gap)
        self.sole = json.loads(Path(sole_path).read_text())
        if self.sole.get("usd_sha256") != self.raw.usd_sha():
            raise RuntimeError("sole hull JSON and sweep come from different USDs")
        self.lim = np.asarray(self.raw.d["motor_limits"], dtype=np.float64)
        self.mi = self.raw.midx
        self.rest_r = {s: {k: self.fk.rest(b)[0] for k, b in seg_bodies(s).items()} for s in SIDES}
        self.pts = {}
        for side in SIDES:
            leg, _, arm = SIDES[side]
            sb = seg_bodies(side)
            self.pts[side] = {"hip": self._local(f"{leg}_hip_joint", 1, sb["thigh"]),
                              "ankle": self._local(f"{leg}_Revolute87", 0, sb["shank"]),
                              "wrist": self._local(f"{arm}_wrist_roll", 0, sb["forearm"])}
        self.refs: dict = {}
        self.geo: dict = {s: {} for s in SIDES}
        # sole bottom plane (normal n_b) and forward direction f_b in the foot body frame, from the rest pose
        self.sole_nb, self.sole_fb = {}, {}
        self.knee_tables: dict = {}
        for side in SIDES:
            foot = seg_bodies(side)["foot"]
            r0, p0 = self.fk.rest(foot)
            vw = np.asarray(self.sole["feet"][foot]["vertices_b"]) @ r0.T + p0
            n_w, _ = sole_plane(vw)
            f_w = np.array([1.0, 0.0, 0.0]) - n_w * n_w[0]
            f_w /= np.linalg.norm(f_w)
            self.sole_nb[side], self.sole_fb[side] = r0.T @ n_w, r0.T @ f_w
            tilt = np.degrees(np.arctan2(np.array([n_w[0], n_w[1]]), n_w[2]))
            self.geo[side]["sole_plane_rest"] = {"normal_root": n_w.tolist(), "tilt_x_y_deg": tilt.tolist()}
            if np.max(np.abs(tilt)) > 0.5:
                self.findings.append(f"{side} sole is not level in the authored rest pose (sole-plane normal tilted "
                                     f"{tilt[0]:+.2f} deg in x, {tilt[1]:+.2f} deg in y); the semantic zero levels the sole")

    def _local(self, joint: str, which: int, body: str) -> np.ndarray:
        b, lp = self.raw.anchor_local(joint, which)
        if b != body:
            raise RuntimeError(f"joint {joint} frame {which} is on {b}, expected {body}")
        return lp

    @staticmethod
    def xform(rp, v):
        r, p = rp
        return p + np.einsum("...ij,j->...i", r, v)

    def foot_level_resid(self, side, r_f) -> np.ndarray:
        """(normal_x, normal_y, forward_y) of the sole frame in the root frame; all 0 = sole level, heading +x."""
        n = r_f @ self.sole_nb[side]
        f = r_f @ self.sole_fb[side]
        return np.array([n[0], n[1], f[1]])

    def sole_points(self, side, r_f, p_f):
        v = np.asarray(self.sole["feet"][seg_bodies(side)["foot"]]["vertices_b"])
        return p_f[..., None, :] + np.einsum("...ij,vj->...vi", r_f, v)

    # -- step 1 -------------------------------------------------------------------------------------------
    def solve_zero(self) -> np.ndarray:
        from scipy.optimize import least_squares

        fk, mi, lim = self.fk, self.mi, self.lim
        q = np.zeros(22)
        for side in SIDES:
            sm = seg_motors(side)
            lt = fk.loops[sm["knee"]]["grid"]
            a_in_thigh = lt["p"] + np.einsum("nij,j->ni", lt["r"], self.pts[side]["ankle"])
            dist = np.linalg.norm(a_in_thigh - self.pts[side]["hip"], axis=-1)
            k = int(np.argmax(dist))
            q[mi[sm["knee"]]] = lt["m"][k]
            self.geo[side]["knee_straight_crank_rad"] = float(lt["m"][k])
            self.geo[side]["hip_ankle_distance_m"] = {"at_straight": float(dist[k]), "min": float(dist.min()),
                                                      "at_crank_lower": float(dist[0]), "at_crank_upper": float(dist[-1])}
            if k in (0, len(dist) - 1):
                self.findings.append(f"{side} knee: the leg is straightest (max hip-ankle distance) at the crank "
                                     f"limit {np.degrees(lt['m'][k]):.2f} deg; semantic knee 0 is at that limit")
        for side in SIDES:
            sm = seg_motors(side)
            names = [sm["hip_roll"], sm["hip_yaw"], sm["hip_pitch"], sm["calf_a"], sm["calf_b"]]
            ids = [mi[n] for n in names]
            rf_rest = self.rest_r[side]["foot"]

            def resid(x, ids=ids, side=side, rf_rest=rf_rest):
                m = q.copy()
                m[ids] = x
                segs = fk.leg(side, m)
                h = self.xform(segs["thigh"], self.pts[side]["hip"])
                a = self.xform(segs["shank"], self.pts[side]["ankle"])
                tilt = self.foot_level_resid(side, segs["foot"][0])
                return np.array([(a[0] - h[0]) / 0.005, (a[1] - h[1]) / 0.005,
                                 tilt[0] / (0.1 * DEG), tilt[1] / (0.1 * DEG), tilt[2] / (0.1 * DEG)])

            sol = least_squares(resid, np.zeros(5), bounds=(lim[ids, 0], lim[ids, 1]), xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, max_nfev=2000)
            q[ids] = sol.x
            r = resid(sol.x)
            self.geo[side]["zero_solve"] = {
                "motors": names, "values_rad": sol.x.tolist(),
                "residual_ankle_minus_hip_xy_mm": (r[:2] * 5.0).tolist(),
                "residual_sole_normal_x_y_forward_y_deg": (r[2:] * 0.1).tolist(), "status": int(sol.status)}
            if np.max(np.abs(r)) > 1.0:
                self.findings.append(f"{side} leg zero pose not exactly reachable: ankle-hip offset "
                                     f"{np.round(r[:2] * 5, 2).tolist()} mm, foot tilt {np.round(r[2:] * 0.1, 3).tolist()} deg")
        return q

    def set_refs(self, q_zero: np.ndarray) -> None:
        for side in SIDES:
            segs = self.fk.leg(side, q_zero)
            self.refs[side] = {"thigh": segs["thigh"][0], "shank": segs["shank"][0], "foot": segs["foot"][0],
                               "upper_arm": self.rest_r[side]["upper_arm"], "forearm": self.rest_r[side]["forearm"],
                               "hand": self.rest_r[side]["hand"]}
        for side in SIDES:
            _, _, arm = SIDES[side]
            au = self.rest_r[side]["upper_arm"] @ self.raw.axis_local(f"{arm}_roll", 1)
            af = self.rest_r[side]["forearm"] @ self.raw.axis_local(f"{arm}_wrist_roll", 0)
            ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(au, af)), -1, 1))))
            el_axis = self.rest_r[side]["upper_arm"] @ self.raw.axis_local(f"{arm}_elbow_joint", 1)
            self.geo[side]["arm_rest"] = {
                "upper_arm_axis_root": au.tolist(), "forearm_axis_root": af.tolist(),
                "upper_forearm_angle_deg": ang, "elbow_motor_axis_root": el_axis.tolist(),
                "upper_arm_tilt_from_vertical_deg": float(np.degrees(np.arccos(min(1.0, abs(au[2])))))}
            if ang > 0.5:
                self.findings.append(f"{side} arm not straight at rest ({ang:.2f} deg between upper-arm and forearm axes)")

    # -- semantics ---------------------------------------------------------------------------------------
    def phys_sem(self, raw: Raw, idx) -> tuple[np.ndarray, dict]:
        return semantics_from_rotations(lambda s, k: raw.R(seg_bodies(s)[k], idx), self.refs)

    def model_sem(self, m: np.ndarray) -> tuple[np.ndarray, dict]:
        cache = {s: {**self.fk.leg(s, m), **self.fk.arm(s, m)} for s in SIDES}
        return semantics_from_rotations(lambda s, k: cache[s][k][0], self.refs)

    # -- step 2 -------------------------------------------------------------------------------------------
    def linear_maps(self, q_zero) -> dict:
        raw, mi, lim = self.raw, self.mi, self.lim
        roles = ("hip_roll", "hip_yaw", "hip_pitch", "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "wrist_roll")
        dofs = {}
        for side in SIDES:
            sm = seg_motors(side)
            for role in roles:
                motor, name = sm[role], f"{side}_{role}"
                idx = raw.rec(f"sweep1d:{motor}")
                m = raw.motor_pos[idx, mi[motor]]
                sem, _ = self.phys_sem(raw, idx)
                j = SEMANTIC_NAMES.index(name)
                # linearity is checked where Euler angles are well defined (|pitch-like| < 80 deg, |v| < 170 deg)
                ok = (np.abs(sem[:, j]) < np.deg2rad(170))
                for k in range(22):
                    if SEMANTIC_NAMES[k].startswith(side) and SEMANTIC_NAMES[k].endswith("pitch") and k != j and \
                            (role.startswith("hip") == SEMANTIC_NAMES[k].startswith(f"{side}_hip")) and \
                            (role.startswith("shoulder") == SEMANTIC_NAMES[k].startswith(f"{side}_shoulder")):
                        ok &= np.abs(sem[:, k]) < np.deg2rad(80)
                if role in ("hip_roll", "hip_yaw", "shoulder_roll", "shoulder_yaw"):
                    pj = SEMANTIC_NAMES.index(f"{side}_{role.split('_')[0]}_pitch")
                    ok &= np.abs(sem[:, pj]) < np.deg2rad(80)
                fit = _lin_fit(m[ok], sem[ok, j])
                scale, offset = fit["slope"], float(q_zero[mi[motor]])
                cross = {}
                for k in range(22):
                    if k == j or not SEMANTIC_NAMES[k].startswith(side):
                        continue
                    span = float(np.ptp(sem[ok, k]))
                    if span > 0.2 * DEG:
                        cross[SEMANTIC_NAMES[k]] = {"slope": _lin_fit(m[ok], sem[ok, k])["slope"],
                                                    "span_deg": float(np.degrees(span))}
                vr = sorted([scale * (lim[mi[motor], 0] - offset), scale * (lim[mi[motor], 1] - offset)])
                dofs[name] = {"type": "linear" if role == "wrist_roll" else "serial3", "motors": [motor],
                              "scale": scale, "offset": offset,
                              "valid_range": [max(vr[0], -math.pi), min(vr[1], math.pi)],
                              "fit": {**fit, "implied_offset": float(-fit["intercept"] / scale),
                                      "samples_used": int(ok.sum()), "samples_total": int(len(m)),
                                      "swept_motor_range_rad": [float(m.min()), float(m.max())]},
                              "cross_talk": cross}
                if fit["max_abs_rad"] > 1.0 * DEG:
                    self.findings.append(f"{name}: linear-fit max residual {np.degrees(fit['max_abs_rad']):.2f} deg "
                                         f"(rms {np.degrees(fit['rms_rad']):.2f} deg)")
                big = {k: v for k, v in cross.items() if abs(v["slope"]) > 0.05}
                if big:
                    self.findings.append(f"{name} ({motor}) cross-talk d(other)/d(motor): "
                                         + ", ".join(f"{k} {v['slope']:+.3f}" for k, v in big.items()))
        return dofs

    def serial_groups(self, dofs: dict, q_zero: np.ndarray) -> dict:
        """``serial_groups`` blocks (hip and shoulder chains, see ``semantic._Serial3``) + semantic boxes."""
        from dropbear_wbc.kinematics.semantic import _Serial3

        groups = {}
        for side in SIDES:
            sm = seg_motors(side)
            for seg, chain_roles, body in (("hip", ("hip_roll", "hip_yaw", "hip_pitch"), "thigh"),
                                           ("shoulder", ("shoulder_pitch", "shoulder_roll", "shoulder_yaw"), "upper_arm")):
                chain = [sm[r] for r in chain_roles]
                sem = [f"{side}_{seg}_pitch", f"{side}_{seg}_roll", f"{side}_{seg}_yaw"]
                role_motors = [dofs[n]["motors"][0] for n in sem]
                g = {"chain_motors": chain, "semantic": sem, "euler": "YXZ", "role_motors": role_motors,
                     "linear_scale": [dofs[n]["scale"] for n in sem],
                     "linear_offset": [dofs[n]["offset"] for n in sem],
                     "axes_root": [self.fk.screws[m]["axis"].tolist() for m in chain],
                     "slopes": [self.fk.screws[m]["slope"] for m in chain],
                     "R_rest": self.rest_r[side][body].tolist(), "R_ref": self.refs[side][body].tolist(),
                     "semantic_box": [[-1.0, 1.0]] * 3, "newton_iters": 40,
                     "note": "R_seg = Rot(a0,k0 m0) Rot(a1,k1 m1) Rot(a2,k2 m2) R_rest (chain order, root frame); "
                             "semantic (pitch, roll, yaw) = YXZ Euler of R_seg R_ref^T on the branch nearest the "
                             "linear estimate scale*(m_role - offset)"}
                obj = _Serial3(g, self.mi)
                n = 17
                grids = [np.linspace(self.lim[self.mi[m], 0], self.lim[self.mi[m], 1], n) for m in chain]
                mm = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)
                full = np.broadcast_to(q_zero, (len(mm), 22)).copy()
                full[:, [self.mi[m] for m in chain]] = mm
                sv = obj.fwd(full)
                box = np.stack([sv.min(0), sv.max(0)], -1)
                g["semantic_box"] = box.tolist()
                for k, n_ in enumerate(sem):
                    dofs[n_]["single_dof_range"] = list(dofs[n_]["valid_range"])
                    dofs[n_]["valid_range"] = box[k].tolist()
                    dofs[n_]["group"] = f"{side}_{seg}"
                    dofs[n_]["range_note"] = ("valid_range = bounding box of the 3-motor chain over its motor limits "
                                              "(combinations); single_dof_range = this DOF alone from the semantic zero")
                groups[f"{side}_{seg}"] = g
        return groups

    def lut1d_maps(self) -> dict:
        raw, fk = self.raw, self.fk
        dofs = {}
        for side in SIDES:
            sm = seg_motors(side)
            for role in ("knee", "elbow"):
                motor, name = sm[role], f"{side}_{role}"
                j = SEMANTIC_NAMES.index(name)
                tab = fk.loops[motor]
                curves, info = {}, {}
                for pas in (0, 1):
                    if pas not in tab:
                        continue
                    sem, diag = self.phys_sem(raw, tab[pas]["idx"])
                    curves[pas] = (tab[pas]["m"], sem[:, j])
                    if pas == 0:
                        ax = diag[f"{side}_{'knee' if role == 'knee' else 'elbow'}_axis"]
                        far = int(np.argmax(np.abs(sem[:, j] - sem[0, j])))
                        info["rotation_axis_root_at_range_end"] = ax[far].tolist()
                        info["tracking_error_max_rad"] = float(np.abs(tab[0]["target"] - tab[0]["m"]).max())
                        info["closure_gap_max_m"] = float(tab[0]["gap"].max())
                        info["dq_window_max"] = float(tab[0]["dq"].max())
                        info["closure_gap_invalid_samples"] = int(tab.get("n_invalid_gap", 0))
                m0, s0 = curves[0]
                okg = tab[0]["gap"] < self.max_gap
                if not okg.all():
                    self.findings.append(f"{name}: {int((~okg).sum())} of {len(okg)} sweep samples have a closure "
                                         f"gap > {1e3 * self.max_gap:.1f} mm and are outside the valid range")
                m0, s0 = m0[okg], s0[okg]
                keep = np.concatenate([[True], np.diff(m0) > 1e-5])
                m0, s0 = m0[keep], s0[keep]
                i0, i1 = _monotonic_run(s0)
                if (i1 - i0) < len(s0):
                    self.findings.append(f"{name}: semantic curve is monotonic only on samples {i0}..{i1 - 1} of "
                                         f"{len(s0)}; the lut1d is truncated to that run")
                mg, sv = m0[i0:i1], s0[i0:i1]
                if 1 in curves:
                    m1, s1 = curves[1]
                    o = np.argsort(m1)
                    info["hysteresis_max_rad"] = float(np.max(np.abs(np.interp(mg, m1[o], s1[o]) - sv)))
                info.update(motor_range_achieved_rad=[float(m0.min()), float(m0.max())],
                            motor_range_commanded_rad=[float(tab[0]["target"].min()), float(tab[0]["target"].max())],
                            semantic_range_rad=[float(sv.min()), float(sv.max())],
                            mean_gain=float((sv[-1] - sv[0]) / max(mg[-1] - mg[0], 1e-9)),
                            gain_min=float(np.min(np.diff(sv) / np.diff(mg))) if len(sv) > 1 else None,
                            gain_max=float(np.max(np.diff(sv) / np.diff(mg))) if len(sv) > 1 else None)
                achieved = float(np.ptp(m0))
                commanded = float(np.ptp(tab[0]["target"]))
                if achieved < 0.8 * commanded:
                    self.findings.append(f"{name}: motor achieved only {np.degrees(achieved):.1f} of "
                                         f"{np.degrees(commanded):.1f} deg commanded (mechanism jams)")
                if len(sv) < 2 or np.ptp(sv) < 1.0 * DEG or achieved < 1.0 * DEG:
                    k0 = int(np.argmin(np.abs(m0)))
                    self.findings.append(
                        f"{name}: LOCKED -- commanded {np.degrees(commanded):.1f} deg, motor moved "
                        f"{np.degrees(achieved):.2f} deg, semantic range {np.degrees(np.ptp(s0)):.2f} deg; "
                        "mapped as type 'fixed'")
                    dofs[name] = {"type": "fixed", "motors": [motor], "motor_value": float(m0[k0]),
                                  "semantic_value": float(s0[k0]), "valid_range": [float(s0[k0]), float(s0[k0])],
                                  "info": {**info, "raw_motor": m0, "raw_semantic": s0}}
                    continue
                dofs[name] = {"type": "lut1d", "motors": [motor], "motor_grid": mg, "semantic_values": sv,
                              "valid_range": [float(sv.min()), float(sv.max())], "info": info}
                if role == "knee":
                    self.knee_tables[side] = dofs[name]
        return dofs

    def pair_maps(self) -> tuple[dict, dict]:
        raw, fk, mi, lim = self.raw, self.fk, self.mi, self.lim
        pairs, dofs = {}, {}
        for side in SIDES:
            sm = seg_motors(side)
            gr = fk.grids[side]
            idx = gr["idx"]
            sem, diag = self.phys_sem(raw, idx.reshape(-1))
            jp, jr = SEMANTIC_NAMES.index(f"{side}_ankle_pitch"), SEMANTIC_NAMES.index(f"{side}_ankle_roll")
            pitch, roll = sem[:, jp].reshape(idx.shape), sem[:, jr].reshape(idx.shape)
            yaw_res = diag[f"{side}_ankle_yaw_residual"].reshape(idx.shape)
            valid = (gr["track_err"] < 0.5 * DEG) & (gr["gap"] < self.max_gap) & (gr["dq"] < self.dq_tol)
            pair = build_pair(side, (sm["calf_a"], sm["calf_b"]), gr["a_grid"], gr["b_grid"], pitch, roll, valid,
                              lim[mi[sm["calf_a"]]], lim[mi[sm["calf_b"]]], self.findings)
            pair["info"]["ankle_yaw_residual_max_deg"] = float(np.degrees(np.abs(yaw_res[valid]).max()))
            pair["info"]["tracking_error_max_deg"] = float(np.degrees(gr["track_err"].max()))
            pair["info"]["closure_gap_max_mm_all_nodes"] = float(1e3 * gr["gap"].max())
            pairs[side] = pair
            for nm in ("pitch", "roll"):
                dofs[f"{side}_ankle_{nm}"] = {"type": "lut2d", "motors": [sm["calf_a"], sm["calf_b"]], "pair": side,
                                              "valid_range": pair[f"{nm}_range"]}
        return pairs, dofs

    # -- step 3 -------------------------------------------------------------------------------------------
    def solve_standing(self, q_zero: np.ndarray, smap) -> tuple[np.ndarray, np.ndarray]:
        from scipy.optimize import least_squares

        fk, mi, lim = self.fk, self.mi, self.lim
        q = q_zero.copy()
        for side in SIDES:
            sm = seg_motors(side)
            kmax = min(max(smap.calib["dofs"][f"{s}_knee"]["valid_range"]) for s in SIDES)
            k_target = min(STANDING_KNEE, 0.6 * kmax)  # symmetric: limited by the stiffer knee
            q_sem = smap.motor_to_semantic(q)
            q_sem[SEMANTIC_NAMES.index(f"{side}_knee")] = k_target
            q[mi[sm["knee"]]] = smap.semantic_to_motor(q_sem)[mi[sm["knee"]]]
            ids = [mi[sm["hip_pitch"]], mi[sm["calf_a"]], mi[sm["calf_b"]]]
            rf_rest = self.rest_r[side]["foot"]

            def resid(x, ids=ids, side=side, rf_rest=rf_rest):
                m = q.copy()
                m[ids] = x
                segs = fk.leg(side, m)
                h = self.xform(segs["thigh"], self.pts[side]["hip"])
                a = self.xform(segs["shank"], self.pts[side]["ankle"])
                tilt = self.foot_level_resid(side, segs["foot"][0])
                return np.array([(a[0] - h[0]) / 0.005, tilt[0] / (0.1 * DEG), tilt[1] / (0.1 * DEG)])

            sol = least_squares(resid, q[ids], bounds=(lim[ids, 0], lim[ids, 1]), xtol=1e-12, ftol=1e-12, gtol=1e-12)
            q[ids] = sol.x
            r = resid(sol.x)
            self.geo[side]["standing_solve"] = {"knee_semantic_rad": k_target,
                                                "residual_ankle_minus_hip_x_mm": float(r[0] * 5),
                                                "residual_sole_normal_x_y_deg": (r[1:] * 0.1).tolist()}
        for side in SIDES:
            sm = seg_motors(side)
            q_sem = smap.motor_to_semantic(q)
            el = smap.calib["dofs"][f"{side}_elbow"]
            target = max(math.pi / 2 - STANDING_ELBOW_FLEX, min(el["valid_range"]) + 0.5 * (math.pi / 2 - min(el["valid_range"])))
            q_sem[SEMANTIC_NAMES.index(f"{side}_elbow")] = min(target, max(el["valid_range"]))
            q[mi[sm["elbow"]]] = smap.semantic_to_motor(q_sem)[mi[sm["elbow"]]]
        return q, smap.motor_to_semantic(q)

    # -- step 4 -------------------------------------------------------------------------------------------
    def geometry(self, q_zero: np.ndarray, q_stand: np.ndarray) -> dict:
        fk, raw = self.fk, self.raw
        out: dict = {"per_side": {}}
        hips, heights = {}, {}
        for side in SIDES:
            sm, sb = seg_motors(side), seg_bodies(side)
            _, _, arm = SIDES[side]
            g: dict = {}
            for label, q in (("zero", q_zero), ("standing", q_stand)):
                segs = fk.leg(side, q)
                armsg = fk.arm(side, q)
                h = self.xform(segs["thigh"], self.pts[side]["hip"])
                a = self.xform(segs["shank"], self.pts[side]["ankle"])
                sp = self.sole_points(side, *segs["foot"])
                zmin = float(sp[:, 2].min())
                w = self.xform(armsg["forearm"], self.pts[side]["wrist"])
                g[label] = {"hip_center_root": h.tolist(), "ankle_center_root": a.tolist(), "sole_min_z_root": zmin,
                            "hip_height_above_sole": float(h[2] - zmin), "wrist_root": w.tolist()}
                if label == "zero":
                    hips[side] = h
                    bottom = sp[sp[:, 2] < zmin + 0.002]
                    g["foot_front"] = float(bottom[:, 0].max() - a[0])
                    g["foot_back"] = float(a[0] - bottom[:, 0].min())
                    g["foot_half_width"] = float(0.5 * np.ptp(bottom[:, 1]))
                    g["ankle_to_sole"] = float(a[2] - zmin)
                heights[(side, label)] = zmin
            # knee: the four-bar knee is polycentric (no fixed pivot). Effective pivot = least-squares fixed
            # centre that best reproduces the ANKLE-CENTRE trajectory relative to the thigh over the valid knee
            # sweep, taken on the knee axis where it is closest to the hip-ankle line at the semantic zero.
            lt = fk.loops[sm["knee"]]["grid"]
            kz = int(np.argmin(np.abs(lt["m"] - q_zero[self.mi[sm["knee"]]])))
            r_rel = lt["r"] @ lt["r"][kz].T
            p_rel = lt["p"] - np.einsum("nij,j->ni", r_rel, lt["p"][kz])
            piv = fit_screw_axis(r_rel, p_rel, lt["m"] - lt["m"][kz])
            ank_th = lt["p"] + np.einsum("nij,j->ni", lt["r"], self.pts[side]["ankle"])  # ankle in thigh frame
            c_th, ank_rms = point_pivot(r_rel, ank_th, ank_th[kz])
            segs0 = fk.leg(side, q_zero)
            r_th, _ = segs0["thigh"]
            hip0 = self.xform(segs0["thigh"], self.pts[side]["hip"])
            ank0 = self.xform(segs0["shank"], self.pts[side]["ankle"])
            free_knee_root = _closest_point_on_line_to_line(self.xform(segs0["thigh"], c_th), r_th @ piv["axis"],
                                                            hip0, ank0 - hip0)
            # skeleton knee: ON the hip-ankle line at the zero pose (L1 + L2 = straight leg length exactly), split
            # chosen to minimise the RMS ankle-centre error of the pivot model over the valid knee sweep
            hip_l, a0 = self.pts[side]["hip"], ank_th[kz]
            u = (a0 - hip_l) / np.linalg.norm(a0 - hip_l)
            d0 = float(np.linalg.norm(a0 - hip_l))
            cand = np.linspace(0.02, d0 - 0.02, 2000)
            kk = hip_l[None] + cand[:, None] * u[None]
            pred = kk[:, None, :] + np.einsum("nij,knj->kni", r_rel, a0[None, None, :] - kk[:, None, :])
            rms = np.sqrt(np.mean(np.sum((pred - ank_th[None]) ** 2, -1), -1))
            l1 = float(cand[int(np.argmin(rms))])
            knee_root = hip0 + l1 * (ank0 - hip0) / np.linalg.norm(ank0 - hip0)
            d_ha = np.linalg.norm(ank_th - self.pts[side]["hip"], axis=-1)
            sem_knee = np.interp(lt["m"], *_knee_curve(self, side))
            g["knee_pivot_fit"] = {"axis_thigh_frame": piv["axis"].tolist(), "body_screw_rot_rms_rad": piv["rot_rms"],
                                   "body_screw_trans_rms_m": piv["trans_rms"],
                                   "free_pivot_ankle_rms_m": ank_rms, "free_pivot_root_at_zero": free_knee_root.tolist(),
                                   "knee_center_root_at_zero": knee_root.tolist(),
                                   "knee_center_thigh_local": (hip_l + l1 * u).tolist(),
                                   "skeleton_pivot_ankle_rms_m": float(rms.min()),
                                   "method": "skeleton knee = point on the hip-ankle line (semantic zero) whose fixed-pivot "
                                             "model best reproduces the ankle-centre trajectory relative to the thigh over "
                                             "the valid knee sweep (the four-bar knee is polycentric: long rockers from "
                                             "the hip to z~0.75 m carry the shank). free_pivot = unconstrained LS centre."}
            g["hip_ankle_distance_vs_knee"] = {"knee_semantic_rad": sem_knee.tolist(), "distance_m": d_ha.tolist(),
                                               "note": "exact measured leg length (hip-pitch anchor to ankle U-joint "
                                                       "centre) vs semantic knee flexion; the pivot model is approximate"}
            # elbow: same method with the wrist-roll anchor trajectory relative to the upper arm
            et = fk.loops[sm["elbow"]]["grid"]
            e0 = int(np.argmin(np.abs(et["m"])))
            r_rel = et["r"] @ et["r"][e0].T
            p_rel = et["p"] - np.einsum("nij,j->ni", r_rel, et["p"][e0])
            epiv = fit_screw_axis(r_rel, p_rel, et["m"] - et["m"][e0])
            wr_ua = et["p"] + np.einsum("nij,j->ni", et["r"], self.pts[side]["wrist"])
            ce_ua, wr_rms = point_pivot(r_rel, wr_ua, wr_ua[e0])
            arm0 = fk.arm(side, np.zeros(22))
            # shoulder centre: common perpendicular midpoint of the shoulder pitch / roll screw axes (rest)
            s1, s2 = fk.screws[sm["shoulder_pitch"]], fk.screws[sm["shoulder_roll"]]
            shoulder = _closest_point_between_lines(s1["point"], s1["axis"], s2["point"], s2["axis"])
            ua_axis = self.rest_r[side]["upper_arm"] @ raw.axis_local(f"{arm}_roll", 1)
            wrist = self.xform(arm0["forearm"], self.pts[side]["wrist"])
            r_ua, _ = arm0["upper_arm"]
            elbow_c = _closest_point_on_line_to_line(self.xform(arm0["upper_arm"], ce_ua), r_ua @ epiv["axis"],
                                                     shoulder, wrist - shoulder)
            elbow_local = r_ua.T @ (elbow_c - arm0["upper_arm"][1])
            hand_root = arm0["hand"][1]
            g["arm_points_rest_root"] = {"shoulder_center": shoulder.tolist(), "elbow_center": elbow_c.tolist(),
                                         "elbow_center_upper_arm_local": elbow_local.tolist(),
                                         "wrist": wrist.tolist(), "hand_body_origin": hand_root.tolist(),
                                         "upper_arm_axis": ua_axis.tolist(),
                                         "elbow_body_screw_rot_rms_rad": epiv["rot_rms"],
                                         "elbow_wrist_point_pivot_rms_m": wr_rms,
                                         "wrist_definition": "wrist-roll joint anchor on the forearm body"}
            g["lengths"] = {"hip_to_knee": float(np.linalg.norm(knee_root - hips[side])),
                            "knee_to_ankle": float(np.linalg.norm(np.array(g["zero"]["ankle_center_root"]) - knee_root)),
                            "shoulder_to_elbow": float(np.linalg.norm(elbow_c - shoulder)),
                            "elbow_to_wrist": float(np.linalg.norm(wrist - elbow_c))}
            out["per_side"][side] = g
        hip_mid = 0.5 * (hips["left"] + hips["right"])
        zero_sole = min(heights[("left", "zero")], heights[("right", "zero")])
        stand_sole = min(heights[("left", "standing")], heights[("right", "standing")])
        L, R = out["per_side"]["left"], out["per_side"]["right"]
        avg = lambda k: 0.5 * (L["lengths"][k] + R["lengths"][k])  # noqa: E731
        out["segment_lengths"] = {
            "hip_width": float(np.linalg.norm(hips["left"] - hips["right"])),
            "hip_to_knee": avg("hip_to_knee"), "knee_to_ankle": avg("knee_to_ankle"),
            "ankle_to_sole": 0.5 * (L["ankle_to_sole"] + R["ankle_to_sole"]),
            "ankle_forward_of_hip": 0.5 * (L["zero"]["ankle_center_root"][0] - L["zero"]["hip_center_root"][0]
                                           + R["zero"]["ankle_center_root"][0] - R["zero"]["hip_center_root"][0]),
            "foot_front": 0.5 * (L["foot_front"] + R["foot_front"]), "foot_back": 0.5 * (L["foot_back"] + R["foot_back"]),
            "foot_half_width": 0.5 * (L["foot_half_width"] + R["foot_half_width"]),
            "shoulder_to_elbow": avg("shoulder_to_elbow"), "elbow_to_wrist": avg("elbow_to_wrist"),
            "standing_hip_height": float(hip_mid[2] - zero_sole),
            "standing_pose_hip_height": float(0.5 * (L["standing"]["hip_center_root"][2] + R["standing"]["hip_center_root"][2]) - stand_sole),
            "units": "m; standing_hip_height = hip-pitch-axis height above the sole at the semantic zero (legs straight)",
            "per_side": {s: out["per_side"][s]["lengths"] for s in SIDES},
        }
        out["rest_transforms"] = {
            "pelvis_in_root": {"pos": hip_mid.tolist(), "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                               "note": "semantic pelvis frame = midpoint of the hip-pitch anchors at the semantic zero; "
                                       "axes = root (world body) axes"},
        }
        out["hip_height"] = {"root_z_at_semantic_zero": -zero_sole, "root_z_at_standing": -stand_sole,
                             "standing_root_z_estimate": -stand_sole,
                             "note": "root ('world' body) frame height when the lowest sole point touches z=0"}
        return out


def sole_plane(vertices_w: np.ndarray, cell: float = 0.01, tol: float = 1.5e-3) -> tuple[np.ndarray, np.ndarray]:
    """Bottom plane of a sole hull given in a z-up frame: (unit normal pointing up, point on the plane).

    Lower envelope = lowest vertex of every ``cell`` x ``cell`` (x, y) cell; a plane z = a x + b y + c is fitted
    to it and refitted on the envelope points within ``tol`` of the plane (the flat bottom face)."""
    v = np.asarray(vertices_w, dtype=np.float64)
    ij = np.floor(v[:, :2] / cell).astype(np.int64)
    keys, inv = np.unique(ij, axis=0, return_inverse=True)
    env = np.array([v[inv == k][np.argmin(v[inv == k, 2])] for k in range(len(keys))])
    sel = np.ones(len(env), dtype=bool)
    for _ in range(5):
        a = np.c_[env[sel, 0], env[sel, 1], np.ones(sel.sum())]
        (cx, cy, c0), *_ = np.linalg.lstsq(a, env[sel, 2], rcond=None)
        res = env[:, 2] - (cx * env[:, 0] + cy * env[:, 1] + c0)
        # keep the lowest points: the bottom face, not raised features
        sel = res < max(tol, np.percentile(res, 5) + tol)
        sel &= res > np.percentile(res, 0) - tol
    n = np.array([-cx, -cy, 1.0])
    n /= np.linalg.norm(n)
    pt = env[sel].mean(axis=0)
    return n, pt


def point_pivot(r_rel: np.ndarray, pts: np.ndarray, pt0: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares fixed pivot c for a point moving as ``pts_i = c + R_i (pt0 - c)`` (R_i relative rotations).

    Returns (c with no component along the mean rotation axis beyond the LS minimum-norm solution, rms [m])."""
    a = (np.eye(3)[None] - r_rel).reshape(-1, 3)
    b = (pts - np.einsum("nij,j->ni", r_rel, pt0)).reshape(-1)
    c, *_ = np.linalg.lstsq(a, b, rcond=None)
    res = (np.eye(3)[None] - r_rel) @ c - (pts - np.einsum("nij,j->ni", r_rel, pt0))
    return c, float(np.sqrt(np.mean(np.sum(res ** 2, axis=-1))))


def _closest_point_on_line_to_line(p1, d1, p2, d2) -> np.ndarray:
    """Point on line 1 closest to line 2."""
    d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
    w = p1 - p2
    b, d, e = d1 @ d2, d1 @ w, d2 @ w
    den = 1.0 - b * b
    t = 0.0 if abs(den) < 1e-12 else (b * e - d) / den
    return p1 + t * d1


def _closest_point_between_lines(p1, d1, p2, d2) -> np.ndarray:
    """Midpoint of the common perpendicular of two lines."""
    d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
    w = p1 - p2
    a, b, c, d, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w, d2 @ w
    den = a * c - b * b
    if abs(den) < 1e-12:
        return 0.5 * (p1 + p2)
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return 0.5 * ((p1 + s * d1) + (p2 + t * d2))


def _knee_curve(b: Builder, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(motor grid, semantic knee) of the fitted lut1d (set by lut1d_maps before geometry)."""
    d = b.knee_tables[side]
    return np.asarray(d["motor_grid"]), np.asarray(d["semantic_values"])


def _fk_diagnostics(b: Builder) -> dict:
    """Serial screw fits vs the USD joint axes, and the knee/elbow gear ratios."""
    out = {}
    for side in SIDES:
        for role, motor in seg_motors(side).items():
            if motor in b.fk.screws:
                s = b.fk.screws[motor]
                out[motor] = {"axis_root": s["axis"].tolist(), "point_root_or_parent": s["point"].tolist(),
                              "slope": s["slope"], "rot_rms_rad": s["rot_rms"], "trans_rms_m": s["trans_rms"],
                              "range": s["range"]}
    return out


def raw_plant_variant(raw) -> dict:
    """Plant-variant keys of a raw sweep/verify NPZ (``cfg_json``): ankle tie rods, contract fixes, diagnostics."""
    import json as _json

    cfg = {}
    if "cfg_json" in raw.d:
        try:
            cfg = _json.loads(str(raw.d["cfg_json"]))
        except ValueError:
            cfg = {}
    return {k: cfg.get(k) for k in ("authored_ankle_tierods", "contract_fixes", "diag_spherical_joints",
                                    "extra_axis_overrides")}


def plant_variant_mismatch(raw_a, raw_b) -> dict:
    """Keys whose values differ between two raws' plant variants (empty = same plant)."""
    va, vb = raw_plant_variant(raw_a), raw_plant_variant(raw_b)
    return {k: [va[k], vb[k]] for k in va if va[k] != vb[k]}


def _verification(b: Builder, verify_path, smap) -> dict:
    """Compare physics-measured semantic angles with the map's prediction at the verify poses."""
    vr = Raw(verify_path)
    if vr.usd_sha() != b.raw.usd_sha():
        raise RuntimeError("verify NPZ comes from a different USD")
    # the verify poses must come from the same plant variant as the sweep (review fix 2026-09-24)
    mismatch = plant_variant_mismatch(b.raw, vr)
    if mismatch:
        raise RuntimeError(f"verify NPZ {verify_path} and sweep NPZ {b.raw.path} come from different plant variants: "
                           f"{mismatch}")
    n = len(vr.program)
    sem_phys, diag = b.phys_sem(vr, np.arange(n))
    sem_map = smap.motor_to_semantic(vr.motor_pos)
    sem_model, _ = b.model_sem(vr.motor_pos)
    err = sem_phys - sem_map
    err = (err + np.pi) % (2 * np.pi) - np.pi
    okv = vr.gaps.max(-1) < b.max_gap
    per_dof = {name: {"max_abs_deg": float(np.degrees(np.abs(err[okv, j]).max())),
                      "rms_deg": float(np.degrees(np.sqrt(np.mean(err[okv, j] ** 2)))),
                      "max_abs_deg_all_poses": float(np.degrees(np.abs(err[:, j]).max()))}
               for j, name in enumerate(SEMANTIC_NAMES)}
    # FK model body-position error for the key segments
    pos_err = []
    for side in SIDES:
        segs = {**b.fk.leg(side, vr.motor_pos), **b.fk.arm(side, vr.motor_pos)}
        for k, (r, p) in segs.items():
            pos_err.append(np.linalg.norm(p - vr.P(seg_bodies(side)[k], np.arange(n)), axis=-1))
    pos_err = np.stack(pos_err, -1)
    names = [str(x) for x in vr.program_names]
    track = np.abs(vr.motor_pos - vr.motor_target).max(-1)
    gaps = vr.gaps.max(-1)
    req_block = {}
    if "pose_semantic_requested" in vr.d:
        req = np.asarray(vr.d["pose_semantic_requested"])[vr.program]
        e2 = (sem_phys - req + np.pi) % (2 * np.pi) - np.pi
        ok = gaps < b.max_gap
        req_block = {
            "semantic_error_request_vs_physics_max_deg": float(np.degrees(np.abs(e2[ok]).max())) if ok.any() else None,
            "semantic_error_request_vs_physics": {
                name: {"max_abs_deg": float(np.degrees(np.abs(e2[ok, j]).max())) if ok.any() else None,
                       "rms_deg": float(np.degrees(np.sqrt(np.mean(e2[ok, j] ** 2)))) if ok.any() else None}
                for j, name in enumerate(SEMANTIC_NAMES)},
            "poses_with_closure_gap_above_threshold": [names[vr.program[i]] for i in np.nonzero(~ok)[0]],
            "note": "requested = SemanticMap-projected request (SaturationReport.used); physics = semantic angles "
                    "measured from the settled link orientations; includes motor tracking error"}
    poses = {}
    for i in range(n):
        nm = names[vr.program[i]]
        if not nm.startswith("random"):
            poses[nm] = {"closure_gap_max_mm": float(1e3 * gaps[i]), "motor_tracking_err_max_deg": float(np.degrees(track[i])),
                         "semantic_measured": sem_phys[i].tolist(), "semantic_map": sem_map[i].tolist()}
    return {
        "verify_npz": str(verify_path).replace("\\", "/"), "num_poses": n,
        "semantic_error_map_vs_physics": per_dof,
        "semantic_error_max_deg": float(np.degrees(np.abs(err[okv]).max())),
        "semantic_error_max_deg_all_poses": float(np.degrees(np.abs(err).max())),
        "poses_valid": int(okv.sum()),
        "semantic_error_model_vs_physics_max_deg": float(np.degrees(np.abs(((sem_phys - sem_model + np.pi) % (2 * np.pi)) - np.pi).max())),
        "fk_model_position_error_max_mm": float(1e3 * pos_err.max()),
        "fk_model_position_error_rms_mm": float(1e3 * np.sqrt(np.mean(pos_err ** 2))),
        "closure_gap_max_mm": float(1e3 * gaps.max()), "motor_tracking_err_max_deg": float(np.degrees(track.max())),
        "ankle_yaw_residual_max_deg": float(np.degrees(max(np.abs(diag["left_ankle_yaw_residual"]).max(),
                                                            np.abs(diag["right_ankle_yaw_residual"]).max()))),
        "poses": poses, **req_block,
    }


MIRROR_SIGN = {"hip_yaw": -1, "hip_roll": -1, "hip_pitch": 1, "knee": 1, "ankle_pitch": 1, "ankle_roll": -1,
               "shoulder_pitch": 1, "shoulder_roll": -1, "shoulder_yaw": -1, "elbow": 1, "wrist_roll": -1}


def _symmetry(calib: dict, findings: list[str]) -> dict:
    """Left vs right map comparison with G1 mirror conventions (roll/yaw flip sign, pitch/knee/elbow keep)."""
    d = calib["dofs"]
    out = {"valid_range_mirror_diff_deg": {}}
    for base, sgn in MIRROR_SIGN.items():
        lr = np.array(d[f"left_{base}"]["valid_range"], dtype=float)
        rr = np.sort(sgn * np.array(d[f"right_{base}"]["valid_range"], dtype=float))
        diff = float(np.degrees(np.abs(lr - rr).max()))
        out["valid_range_mirror_diff_deg"][base] = diff
        if diff > 2.0:
            findings.append(f"L/R asymmetry in {base}: left valid range {np.round(np.degrees(lr), 1).tolist()} deg vs "
                            f"mirrored right {np.round(np.degrees(rr), 1).tolist()} deg (max diff {diff:.1f} deg)")
    for base in ("hip_roll", "hip_yaw", "hip_pitch", "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "wrist_roll"):
        l, r = d[f"left_{base}"], d[f"right_{base}"]
        out[base] = {"left_scale": l["scale"], "right_scale": r["scale"], "left_offset": l["offset"],
                     "right_offset": r["offset"], "left_range": l["valid_range"], "right_range": r["valid_range"],
                     "mirror_expected": "range(right) = -range(left)" if ("roll" in base or "yaw" in base) else "equal"}
    for base in ("knee", "elbow"):
        l, r = d[f"left_{base}"], d[f"right_{base}"]
        if l["type"] != "lut1d" or r["type"] != "lut1d":
            out[base] = {"left_type": l["type"], "right_type": r["type"]}
            continue
        mg = np.linspace(max(min(l["motor_grid"]), min(r["motor_grid"])), min(max(l["motor_grid"]), max(r["motor_grid"])), 50)
        dl = np.interp(mg, l["motor_grid"], l["semantic_values"])
        dr = np.interp(mg, r["motor_grid"], r["semantic_values"])
        out[base] = {"max_left_right_diff_deg": float(np.degrees(np.abs(dl - dr).max())),
                     "left_range": l["valid_range"], "right_range": r["valid_range"]}
    for nm in ("pitch", "roll"):
        out[f"ankle_{nm}"] = {"left_range": d[f"left_ankle_{nm}"]["valid_range"],
                              "right_range": d[f"right_ankle_{nm}"]["valid_range"]}
    return out


def fit_calibration(raw_path, out_path, sole_path, verify_path=None, script="tools/calibrate_semantics.py",
                    log_path: str = "", max_gap: float = 3e-3) -> dict:
    """Fit and write the calibration JSON (see module doc). Returns the JSON dict."""
    from dropbear_wbc.kinematics.semantic import SemanticMap

    findings: list[str] = []
    b = Builder(raw_path, sole_path, findings, max_gap=max_gap)
    q_zero = b.solve_zero()
    b.set_refs(q_zero)
    dofs = {}
    dofs.update(b.linear_maps(q_zero))
    serial_groups = b.serial_groups(dofs, q_zero)
    dofs.update(b.lut1d_maps())
    pairs, pdofs = b.pair_maps()
    dofs.update(pdofs)
    calib = {
        "schema": SCHEMA,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "usd_sha256": b.raw.usd_sha(),
        "semantic_names": list(SEMANTIC_NAMES), "motor_names": list(MOTOR_NAMES),
        "motor_limits_rad": b.lim, "dofs": {n: dofs[n] for n in SEMANTIC_NAMES}, "ankle_pairs": pairs,
        "serial_groups": serial_groups,
        "semantic_zero_motor_pos": q_zero, "standing_motor_pos": q_zero,
    }
    smap = SemanticMap(_sanitize(calib))
    # elbows at G1 zero (forearm forward) clipped to the reachable range
    q_sem0 = smap.motor_to_semantic(q_zero)
    for side in SIDES:
        q_sem0[SEMANTIC_NAMES.index(f"{side}_elbow")] = 0.0
    q_zero_full, rep0 = smap.semantic_to_motor(q_sem0, return_report=True)
    for side in SIDES:  # only the elbows change
        i = MOTOR_NAMES.index(seg_motors(side)["elbow"])
        q_zero[i] = q_zero_full[i]
    calib["semantic_zero_motor_pos"] = q_zero
    calib["semantic_zero_saturation"] = rep0.summary()
    smap = SemanticMap(_sanitize(calib))
    q_stand, s_stand = b.solve_standing(q_zero, smap)
    calib["standing_motor_pos"] = q_stand
    calib["standing_semantic_pos"] = s_stand
    calib["semantic_zero_semantic_pos"] = smap.motor_to_semantic(q_zero)
    geom = b.geometry(q_zero, q_stand)
    calib["segment_lengths"] = geom["segment_lengths"]
    calib["rest_transforms"] = geom["rest_transforms"]
    calib["hip_height"] = geom["hip_height"]
    calib["geometry"] = {"per_side": geom["per_side"], "solves": b.geo}
    calib["key_bodies"] = {
        "root": "world", "anchor": "head_5mm_ujoint_base__5__1", "pelvis": "world", "head": "head_u_joint_center__8__1",
        **{f"{s}_{k}": v for s in SIDES for k, v in seg_bodies(s).items()},
        "note": "anchor = chest-level Stewart-base body fixed to 'world' (same choice as "
                "dropbear_wbc.robots.dropbear_names.ANCHOR_BODY); pelvis is part of the rigid 'world' body; "
                "thigh = hip-pitch child (knee-motor stator), thigh_strap = four-bar thigh link, "
                "ankle_cross = ankle U-joint cross",
    }
    calib["reference_rotations_root"] = {s: {k: v for k, v in b.refs[s].items()} for s in SIDES}
    calib["serial_screw_fits"] = _fk_diagnostics(b)
    calib["symmetry"] = _symmetry(_sanitize(calib), findings)
    calib["conventions"] = {
        "frames": "root = articulation root body 'world' (rigid torso+pelvis); x forward, y left, z up. All "
                  "semantic angles are measured from root-relative link orientations after quasi-static settling.",
        "hip_shoulder": "intrinsic YXZ Euler of F = R R_ref^T (G1 order pitch->roll->yaw about +y,+x,+z)",
        "knee": "signed rotation angle of the shank relative to the thigh, sign = axis . +y; flexion > 0; "
                "0 = straightest leg (max hip-ankle distance)",
        "ankle": "YXZ Euler (pitch, roll) of the foot relative to the shank; 0 = sole level with the leg at zero",
        "elbow": "EXACT G1 *_elbow_joint convention: straight arm = +pi/2, 0 = forearm forward (90 deg flexion), "
                 "positive = extension (rotation about +y). NOTE: docs/CONTRACTS.md said 'elbow flexion positive'; "
                 "that is not the G1 convention (see findings). G1's own forearm link is 0.1 rad off its joint "
                 "frame, so the G1 forearm is vertical at elbow = 1.471.",
        "wrist_roll": "rotation about the elbow->hand direction (G1 wrist-roll axis +x of the elbow link)",
        "semantic_zero": "legs: straightest knee, ankle centre below hip centre, sole level, foot heading forward; "
                         "arms: authored rest (upper arm vertical, elbow axis along y), elbow at G1 0 clipped to "
                         "the reachable range, wrist 0",
        "idealised_axes": "G1's real hip-roll/yaw axes are tilted 10 deg and its shoulder-pitch axis 16 deg; the "
                          "semantic space uses exact +x/+y/+z. Dropbear's own axes (hip yaw/pitch, knee) are "
                          "tilted 10 deg; that shows up as cross-talk in dofs[*].cross_talk.",
    }
    if verify_path is not None and Path(verify_path).exists():
        calib["verification"] = _verification(b, verify_path, smap)
    cfg_json = json.loads(str(b.raw.d["cfg_json"]))
    diag = {k: cfg_json.get(k) for k in ("extra_axis_overrides", "diag_spherical_joints", "patch_joint_axes")
            if cfg_json.get(k)}
    if not cfg_json.get("contract_fixes", False):
        diag["contract_fixes"] = False
    # sweeps recorded before the CONTRACTS 0.2 adoption (2026-09-24) have no key: they used the authored ankle
    authored_ankle = bool(cfg_json.get("authored_ankle_tierods", True))
    if diag:
        calib["plant_variant"] = f"DIAGNOSTIC plant variant {diag} -- NOT the contract plant"
    elif authored_ankle:
        calib["plant_variant"] = ("authored-ankle plant (CONTRACTS 0.2 opt-out, not the default): authoritative USD + "
                                  "docs/CONTRACTS.md 0.1 spawn fixes (orphan bodies deactivated, joint friction 0, "
                                  "LL_Revolute121 axis Z, min principal inertia 2e-6); ankle tie rods *_Revolute111/112 "
                                  "revolute as authored")
    else:
        calib["plant_variant"] = ("contract plant: authoritative USD + docs/CONTRACTS.md 0.1 spawn fixes (orphan bodies "
                                  "deactivated, joint friction 0, LL_Revolute121 axis Z, min principal inertia 2e-6) + "
                                  "0.2 ankle tie rods *_Revolute111/112 spherical")
    calib["authored_ankle_tierods"] = authored_ankle
    if diag:
        findings.insert(0, f"DIAGNOSTIC calibration of a modified plant {diag}; do not use for the contract")
    calib["findings"] = findings
    calib["provenance"] = {
        "script": script, "log": log_path, "raw_sweep": str(raw_path).replace("\\", "/"),
        "sole_hulls": str(sole_path).replace("\\", "/"), "usd_path": str(b.raw.d["usd_path"]),
        "usd_sha256": b.raw.usd_sha(), "sweep_settle_cfg": json.loads(str(b.raw.d["cfg_json"])),
        "sweep_wall_s": float(b.raw.d["wall_s"]), "sweep_physics_steps": int(b.raw.d["physics_steps"]),
        "sweep_ms_per_step": float(b.raw.d["ms_per_step"]) if "ms_per_step" in b.raw.d else None,
        "sweep_run": json.loads(str(b.raw.d["run_json"])) if "run_json" in b.raw.d else None,
        "max_gap_m": b.max_gap,
        "fit_modules": ["dropbear_wbc.kinematics.calib_build", "dropbear_wbc.kinematics.calib_fit",
                        "dropbear_wbc.kinematics.calib_pair", "dropbear_wbc.kinematics.calib_semantics"],
    }
    calib = _sanitize(calib)
    SemanticMap(calib)  # must load
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(calib, indent=1))
    print(f"[fit] wrote {out_path}; {len(findings)} findings")
    for f in findings:
        print(f"[fit][finding] {f}")
    return calib
