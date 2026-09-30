"""Foot forward-kinematics accuracy of the available Dropbear leg models against PHYSICS poses (CPU only).

Question (foot_contact track, step 1): which leg FK should the foot-contact retarget stage use? Candidates:

* ``calibration`` -- ``kinematics.serial_model.CalibrationFK`` / ``calib_fit.MeasuredFK`` (used by teleop and the GMR
  route): products of exponentials for the serial hip motors + the knee four-bar sweep table (crank -> shank) + the
  2D calf-motor grid (shank -> foot), all from the calibration's raw physics sweep, evaluated at the MOTOR angles.
* ``serial_mjcf`` -- ``data/robot/dropbear_serial.xml`` (derived serial model, 22 semantic hinges) at
  ``q_sem = SemanticMap.motor_to_semantic(motors)``.
* ``skeleton`` -- ``motion.semantic_skeleton.leg_fk`` (idealised pivot leg from segment lengths), which the
  G1-intermediate route used for its ground fix / foot lock before this track.

Physics poses (never the model's own training data):

* the calibration VERIFY set (``provenance``/``verification.verify_npz``, 82 settled poses, root fixed);
* settled motion NPZs on the contract plant (``tools/settle_motion.py``; 91 DOF joint state + 90 link poses composed
  with the CSV root). The settled motor positions are the model inputs; the settled foot link poses are the truth.

Metrics per model (both feet pooled): foot-link origin position error in the root frame [mm], foot orientation error
[deg], and the error of the LOWEST SOLE POINT z in the world (collision-hull vertices,
``data/calibration/dropbear_foot_sole_hulls.json``, both foot bodies) -- the quantity the contact gate checks.

    python tools/foot_fk_accuracy.py --out logs/foot_contact/foot_fk_accuracy.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.kinematics.calib_fit import seg_bodies  # noqa: E402
from dropbear_wbc.kinematics.rigid import quat_mat, rotvec  # noqa: E402
from dropbear_wbc.kinematics.semantic import MOTOR_NAMES, SEMANTIC_NAMES  # noqa: E402
from dropbear_wbc.kinematics.serial_model import CalibrationFK  # noqa: E402
from dropbear_wbc.settle.ground import FOOT_GROUPS, SoleModel  # noqa: E402

SIDES = ("left", "right")


def _stats(x_mm: np.ndarray) -> dict:
    x = np.abs(np.asarray(x_mm, dtype=np.float64)).reshape(-1)
    if not x.size:
        return {}
    return {"p50": round(float(np.percentile(x, 50)), 3), "p95": round(float(np.percentile(x, 95)), 3),
            "max": round(float(x.max()), 3), "rms": round(float(np.sqrt(np.mean(x ** 2))), 3), "n": int(x.size)}


def load_verify(cfk: CalibrationFK) -> dict:
    ver = cfk.cal.get("verification", {})
    path = REPO / ver["verify_npz"]
    d = np.load(path, allow_pickle=False)
    names = [str(b) for b in d["body_names"]]
    motor_idx = [list(map(str, d["joint_names"])).index(m) for m in MOTOR_NAMES]
    return {"name": f"verify:{path.name}", "path": str(path), "motors": np.asarray(d["joint_pos"])[:, motor_idx],
            "body_names": names, "rel_pos": np.asarray(d["body_pos"], dtype=np.float64),
            "rel_quat": np.asarray(d["body_quat"], dtype=np.float64),
            "root_pos": np.zeros((len(d["joint_pos"]), 3)), "root_quat": np.tile([1.0, 0, 0, 0], (len(d["joint_pos"]), 1))}


def load_settled(path: Path) -> dict | None:
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    if meta.get("authored_ankle_tierods", True):
        return None  # other plant variant (or pre-0.2): not the contract plant
    names = [str(b) for b in d["body_names"]]
    jn = [str(j) for j in d["joint_names"]]
    motors = np.asarray(d["joint_pos"], dtype=np.float64)[:, [jn.index(m) for m in MOTOR_NAMES]]
    pos = np.asarray(d["body_pos_w"], dtype=np.float64)
    quat = np.asarray(d["body_quat_w"], dtype=np.float64)
    r_root = quat_mat(quat[:, 0])
    rel_pos = np.einsum("tji,tbj->tbi", r_root, pos - pos[:, :1])
    rel_rot = np.einsum("tji,tbjk->tbik", r_root, quat_mat(quat))
    from dropbear_wbc.kinematics.rigid import mat_to_quat

    return {"name": f"settled:{path.relative_to(REPO).as_posix()}", "path": str(path), "motors": motors,
            "body_names": names, "rel_pos": rel_pos, "rel_quat": mat_to_quat(rel_rot),
            "root_pos": pos[:, 0], "root_quat": quat[:, 0], "status": meta.get("status")}


def lowest_sole_z(sole: SoleModel, side: str, root_pos, root_rot, rel: dict[str, tuple[np.ndarray, np.ndarray]]):
    """World z of the lowest hull vertex of both foot bodies of ``side``; ``rel`` = {body: (R_rel, p_rel)}."""
    zs = []
    for b in FOOT_GROUPS[side]:
        if b not in rel:
            continue
        r_rel, p_rel = rel[b]
        rw = root_rot @ r_rel
        pw = root_pos + np.einsum("tij,tj->ti", root_rot, p_rel)
        zs.append((np.einsum("tj,vj->tv", rw[:, 2, :], sole.vertices[b]) + pw[:, 2:3]).min(axis=1))
    return np.min(np.stack(zs), axis=0)


def evaluate(ds: dict, cfk: CalibrationFK, sole: SoleModel, smod, geom) -> dict:
    m = ds["motors"]
    t = len(m)
    names = ds["body_names"]
    r_root = quat_mat(ds["root_quat"])
    truth = {}
    for s in SIDES:
        sb = seg_bodies(s)
        rel = {b: (quat_mat(ds["rel_quat"][:, names.index(b)]), ds["rel_pos"][:, names.index(b)])
               for b in (sb["foot"], sb["ankle_cross"])}
        truth[s] = {"foot": rel[sb["foot"]], "low": lowest_sole_z(sole, s, ds["root_pos"], r_root, rel), "rel": rel}
    q_sem = cfk.smap.motor_to_semantic(m)
    out = {"frames": t}
    # --- calibration forward model (MeasuredFK) ---------------------------------------------------------
    res = {"pos_mm": [], "ori_deg": [], "sole_z_mm": []}
    for s in SIDES:
        sb = seg_bodies(s)
        r_f, p_f = cfk.fk.leg(s, m)["foot"]
        rt, pt = truth[s]["foot"]
        res["pos_mm"].append(1e3 * np.linalg.norm(p_f - pt, axis=-1))
        res["ori_deg"].append(np.degrees(np.linalg.norm(rotvec(np.swapaxes(r_f, -1, -2) @ rt), axis=-1)))
        # ankle cross body: shank pose from the model and the physics shank->cross transform is unknown to the model,
        # so evaluate the sole plate alone for the model and the truth alike (the plate is the lowest body when flat)
        low_model = lowest_sole_z(sole, s, ds["root_pos"], r_root, {sb["foot"]: (r_f, p_f)})
        low_truth = lowest_sole_z(sole, s, ds["root_pos"], r_root, {sb["foot"]: truth[s]["rel"][sb["foot"]]})
        res["sole_z_mm"].append(1e3 * (low_model - low_truth))
    out["calibration"] = {k: _stats(np.concatenate(v)) for k, v in res.items()}
    # --- serial MJCF -------------------------------------------------------------------------------------
    if smod is not None:
        res = {"pos_mm": [], "ori_deg": [], "sole_z_mm": []}
        site = {s: next(sn for sn, st in smod.meta["sites"].items() if st["usd_body"] == seg_bodies(s)["foot"])
                for s in SIDES}
        pel = np.asarray(smod.meta["frames"]["pelvis_in_root"]["pos"])
        rf = {s: np.zeros((t, 3, 3)) for s in SIDES}
        pf = {s: np.zeros((t, 3)) for s in SIDES}
        for k in range(t):
            smod.fk(smod.qpos(pel, np.array([1.0, 0, 0, 0]), q_sem[k]))
            for s in SIDES:
                rf[s][k], pf[s][k] = smod.site_pose(site[s])
        for s in SIDES:
            sb = seg_bodies(s)
            rt, pt = truth[s]["foot"]
            res["pos_mm"].append(1e3 * np.linalg.norm(pf[s] - pt, axis=-1))
            res["ori_deg"].append(np.degrees(np.linalg.norm(rotvec(np.swapaxes(rf[s], -1, -2) @ rt), axis=-1)))
            low_model = lowest_sole_z(sole, s, ds["root_pos"], r_root, {sb["foot"]: (rf[s], pf[s])})
            low_truth = lowest_sole_z(sole, s, ds["root_pos"], r_root, {sb["foot"]: truth[s]["rel"][sb["foot"]]})
            res["sole_z_mm"].append(1e3 * (low_model - low_truth))
        out["serial_mjcf"] = {k: _stats(np.concatenate(v)) for k, v in res.items()}
    # --- idealised semantic skeleton (4 sole corners; only the lowest-sole-z comparison is meaningful) ----
    from dropbear_wbc.motion.rotations import quat_to_matrix
    from dropbear_wbc.motion.semantic_skeleton import leg_fk

    p_in_r = cfk.cal["rest_transforms"]["pelvis_in_root"]
    r_pr = quat_to_matrix(np.asarray(p_in_r["quat_wxyz"], dtype=np.float64))
    p_pr = np.asarray(p_in_r["pos"], dtype=np.float64)
    pel_pos = ds["root_pos"] + np.einsum("tij,j->ti", r_root, p_pr)
    pel_rot = r_root @ r_pr
    lf = leg_fk(pel_pos, pel_rot, q_sem, geom)
    res = {"sole_z_mm": []}
    for k, s in enumerate(SIDES):
        sb = seg_bodies(s)
        low_truth = lowest_sole_z(sole, s, ds["root_pos"], r_root, {sb["foot"]: truth[s]["rel"][sb["foot"]]})
        res["sole_z_mm"].append(1e3 * (lf.sole_height[:, k] - low_truth))
    out["skeleton"] = {k: _stats(np.concatenate(v)) for k, v in res.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, nargs="*", default=None, help="settled NPZs (default: all under data/motions)")
    ap.add_argument("--no-serial", action="store_true", help="skip the serial MJCF (needs mujoco)")
    ap.add_argument("--out", type=Path, default=REPO / "logs/foot_contact/foot_fk_accuracy.json")
    args = ap.parse_args()
    t0 = time.time()
    cfk = CalibrationFK()
    sole = SoleModel.load(REPO / "data/calibration/dropbear_foot_sole_hulls.json")
    smod = None
    if not args.no_serial:
        from dropbear_wbc.kinematics.serial_model import SerialModel

        smod = SerialModel()
    from dropbear_wbc.motion.calibration_view import load_calibration
    from dropbear_wbc.motion.semantic_skeleton import LegGeometry

    geom = LegGeometry.from_calibration(load_calibration())
    npzs = args.npz if args.npz else sorted(p for p in (REPO / "data/motions").rglob("*.npz")
                                            if "archive" not in p.parts and not p.name.endswith(".rejected.npz"))
    sets = [load_verify(cfk)]
    skipped = []
    for p in npzs:
        ds = load_settled(p)
        (sets.append(ds) if ds is not None else skipped.append(str(p.relative_to(REPO).as_posix())))
    report = {"tool": "tools/foot_fk_accuracy.py", "calibration": str(cfk.path), "calibration_sha256": cfk.sha256,
              "raw_sweep": str(cfk.raw_path), "skipped_other_plant": skipped, "sets": {}}
    pooled: dict[str, dict[str, list]] = {}
    for ds in sets:
        r = evaluate(ds, cfk, sole, smod, geom)
        report["sets"][ds["name"]] = r
        print(json.dumps({ds["name"]: r}), flush=True)
        for model, stats in r.items():
            if model == "frames":
                continue
            for metric, st in stats.items():
                pooled.setdefault(model, {}).setdefault(metric, []).append(st)
    # pooled max / frame-weighted p95 summary
    summ = {}
    for model, metrics in pooled.items():
        summ[model] = {met: {"max": max(s["max"] for s in lst), "p95_worst_set": max(s["p95"] for s in lst),
                             "p50_median_set": float(np.median([s["p50"] for s in lst]))} for met, lst in metrics.items()}
    report["summary"] = summ
    report["wall_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summ}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
