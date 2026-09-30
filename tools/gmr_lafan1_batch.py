"""Batch: LAFAN1 clips -> GMR (Dropbear serial model AND Unitree G1) -> contract CSVs + route comparison.

For each clip segment below:

1. ``tools/gmr_retarget.py`` with robot ``dropbear`` and ``unitree_g1`` -> ``logs/serial_mjcf_gmr/gmr/<clip>_{dropbear,g1}.npz``
2. ``tools/gmr_to_dropbear_csv.py`` -> ``data/motions/gmr_lafan1/<clip>.csv`` (gmr-serial-v1) and, from the G1
   retarget through the G1-intermediate route, ``data/motions/gmr_lafan1_via_g1/<clip>.csv`` (comparison)
3. per-clip report ``logs/serial_mjcf_gmr/route_compare/<clip>.json`` and a montage PNG of the serial model
   with the scaled human targets (``logs/serial_mjcf_gmr/renders/<clip>.png``)

and finally ``logs/serial_mjcf_gmr/route_comparison.{json,md}`` and ``data/motions/gmr_lafan1/catalog_gmr_lafan1.json``.

Run in ``.venv-gmr`` (CPU, ~1 min)::

    .venv-gmr/Scripts/python.exe tools/gmr_lafan1_batch.py

Review fixes (2026-09-24): both robots are solved with a ``--preroll`` (default 1 s of the clip before the window,
not recorded; GMR's G1 warm start at dance2 8.0 s had converged to a twisted local minimum), and every GMR solve passes
an IK-HEALTH gate (:func:`ik_health`: root/chest rotation error p50 <= 15 deg, hand/forearm position error p95 <=
0.15 m, no joint at a limit on > 50 % of frames). A route whose solve fails the gate is marked INVALID in the
comparison instead of being counted as evidence. ``--log-dir`` / ``--out-root`` redirect the outputs.

LAFAN1 is CC BY-NC-ND 4.0: every output is internal R&D only (never redistribute).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "source"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
LAFAN = _paths.upstream_dir() / "lafan1" / "bvh"
LOG = REPO / "logs/serial_mjcf_gmr"
# name, bvh stem, start s, end s, category, why this window
CLIPS = [
    ("lafan1_walk1_subject1", "walk1_subject1", 13.0, 28.0, "walk", "steady walking with turns (0.6-0.7 m/s), after the T-pose"),
    ("lafan1_armraise_dance2_subject1", "dance2_subject1", 8.0, 17.0, "wave-like",
     "right arm raised overhead (elevation ~125 deg) for 4 s, then lowered with elbow flexion; LAFAN1 has no wave clip"),
    ("lafan1_dance2_subject1", "dance2_subject1", 18.0, 38.0, "dance", "dance with steps and arm motion"),
    ("lafan1_run1_subject2", "run1_subject2", 28.0, 40.0, "run", "jogging 1.6-1.8 m/s"),
    ("lafan1_jumps1_subject1", "jumps1_subject1", 8.0, 20.0, "jump", "repeated two-leg jumps (both feet airborne ~30% of frames)"),
]


def render_montage(npz: Path, out_png: Path, n: int = 6) -> None:
    import mujoco
    from PIL import Image

    from dropbear_wbc.kinematics.serial_model import DEFAULT_SCENE

    d = np.load(npz, allow_pickle=False)
    qpos, hp = d["qpos"], d["human_pos"]
    m = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    data = mujoco.MjData(m)
    r = mujoco.Renderer(m, 360, 300)
    opt = mujoco.MjvOption()
    for i in range(6):
        opt.geomgroup[i] = 1 if i < 3 else 0  # visual meshes + capsules; hide collision (group 3)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.2, -12.0, 135.0
    tiles = []
    for k in np.linspace(0, len(qpos) - 1, n).astype(int):
        data.qpos[:] = qpos[k]
        mujoco.mj_forward(m, data)
        cam.lookat[:] = [qpos[k, 0], qpos[k, 1], 0.9]
        r.update_scene(data, cam, opt)
        for p in hp[k]:
            if r.scene.ngeom >= r.scene.maxgeom:
                break
            mujoco.mjv_initGeom(r.scene.geoms[r.scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0, 0]),
                                np.asarray(p, dtype=np.float64), np.eye(3).reshape(-1), np.array([1.0, 0.5, 0.0, 0.9]))
            r.scene.ngeom += 1
        tiles.append(r.render())
    Image.fromarray(np.concatenate(tiles, axis=1)).save(out_png)


HEALTH = {"root_rot_p50_deg": 15.0, "arm_pos_p95_m": 0.15, "joint_at_limit_frac": 0.5}


def ik_health(npz: Path) -> dict:
    """IK-health gate of one GMR solve (see module doc). ``ok`` False = the solve did not track the human."""
    import mujoco

    d = np.load(npz, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    tb = [str(x) for x in d["task_bodies"]]
    rot50 = np.degrees(np.median(d["task_rot_err"], axis=0))
    pos95 = np.percentile(d["task_pos_err"], 95, axis=0)
    m = mujoco.MjModel.from_xml_path(meta["xml"])
    names = [str(n) for n in d["qpos_joint_names"]]
    q = np.asarray(d["qpos"])[:, 7:]
    at_lim = {}
    for k, n in enumerate(names):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j < 0 or not m.jnt_limited[j]:
            continue
        lo, hi = m.jnt_range[j]
        frac = float(np.mean((q[:, k] < lo + np.radians(0.5)) | (q[:, k] > hi - np.radians(0.5))))
        if frac > 0:
            at_lim[n] = round(frac, 3)
    root = {b: float(rot50[tb.index(b)]) for b in ("Hips", "Spine2") if b in tb}
    arm = {b: float(pos95[tb.index(b)]) for b in ("LeftForeArm", "LeftHand", "RightForeArm", "RightHand") if b in tb}
    problems = [f"{b} rotation error p50 {v:.1f} deg" for b, v in root.items() if v > HEALTH["root_rot_p50_deg"]]
    problems += [f"{b} position error p95 {v:.2f} m" for b, v in arm.items() if v > HEALTH["arm_pos_p95_m"]]
    problems += [f"{n} at a joint limit on {100 * f:.0f}% of frames" for n, f in at_lim.items()
                 if f > HEALTH["joint_at_limit_frac"]]
    out = {"ok": not problems, "problems": problems, "root_rot_err_p50_deg": root, "arm_pos_err_p95_m": arm,
           "joints_at_limit_frac": at_lim, "thresholds": HEALTH}
    if "arm_dir_err" in d.files:
        ad = np.degrees(d["arm_dir_err"])
        out["arm_direction_error_deg"] = {
            f"{side}_{seg}": {"p50": float(np.median(ad[:, i, k])), "p95": float(np.percentile(ad[:, i, k], 95))}
            for i, side in enumerate(("left", "right")) for k, seg in enumerate(("upper_arm", "forearm"))}
    return out


def _contact_cell(ca: dict | None) -> str:
    dis = (ca or {}).get("disagreement_frac")
    if not dis:
        return "-"
    return (f"no-contact frames src {100 * ca.get('no_foot_in_contact_frac_source', 0):.0f}%; "
            f"disagree L/R {100 * dis['left']:.0f}/{100 * dis['right']:.0f}%{' FLAG' if ca.get('flag') else ''}")


def _arm_cell(h: dict) -> str:
    ad = h.get("arm_direction_error_deg")
    if not ad:
        return "-"
    return (f"{ad['left_upper_arm']['p50']:.1f}/{ad['left_upper_arm']['p95']:.1f}, "
            f"{ad['right_upper_arm']['p50']:.1f}/{ad['right_upper_arm']['p95']:.1f}")


def md_table(summary: list[dict]) -> str:
    rows = ["| clip | category | frames | IK health serial / G1 | GMR serial: frames clipped / worst clip (deg) | "
            "G1 route: frames clipped / worst clip (deg) | IK at a joint limit (serial, frac >= 5%) | "
            "stance slip p95 m/s (serial / G1) | stance float p95 m (serial / G1) | frames over motor vel. limit "
            "(serial / G1) | GMR foot pos err p95 m | serial root rot err p50 deg (Hips, Spine2) | "
            "contact hint vs source | upper-arm dir. err p50/p95 deg (L, R) |",
            "|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        hs, hg = s["health"]["serial"], s["health"]["g1"]
        g1_sat = (f"{100 * s['g1_sat']:.1f}% / {s['g1_worst']}" if hg["ok"] else
                  f"INVALID (G1 IK failed: {'; '.join(hg['problems'][:2])})")
        rr = hs["root_rot_err_p50_deg"]
        rows.append(
            f"| {s['clip']} | {s['category']} | {s['frames']} | {'ok' if hs['ok'] else 'FAIL'} / "
            f"{'ok' if hg['ok'] else 'FAIL'} | {100 * s['serial_sat']:.1f}% / {s['serial_worst']} | {g1_sat} | "
            f"{s['serial_at_limit']} | {s['slip'][0]:.2f} / {s['slip'][1]:.2f} | {s['float'][0]:.3f} / {s['float'][1]:.3f} | "
            f"{s['vel'][0]} / {s['vel'][1]} | {s['foot_err']:.3f} | {rr.get('Hips', 0):.1f}, {rr.get('Spine2', 0):.1f} | "
            f"{_contact_cell(s.get('contact_agreement'))} | {_arm_cell(hs)} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--preroll", type=float, default=1.0, help="s solved before each window (both robots; not recorded)")
    ap.add_argument("--log-dir", type=Path, default=LOG)
    ap.add_argument("--reuse-gmr", action="store_true", help="reuse existing <log-dir>/gmr/*.npz (skip the IK)")
    ap.add_argument("--out-root", type=Path, default=REPO / "data/motions")
    ap.add_argument("--gmr-dir", type=Path, default=None,
                    help="where the GMR npz live / are written (default <log-dir>/gmr); with --reuse-gmr, read only")
    ap.add_argument("--csv-args", nargs=argparse.REMAINDER, default=[],
                    help="extra args for tools/gmr_to_dropbear_csv.py, e.g. --foot-contact --fps-out 50 (must be last)")
    ap.add_argument("--no-g1", action="store_true", help="skip the G1 comparison route")
    args = ap.parse_args(argv)
    import gmr_retarget
    import gmr_to_dropbear_csv

    t0 = time.time()
    log = args.log_dir
    (log / "gmr").mkdir(parents=True, exist_ok=True)
    (log / "route_compare").mkdir(parents=True, exist_ok=True)
    (log / "renders").mkdir(parents=True, exist_ok=True)
    summary, catalog = [], []
    for name, stem, a, b, cat, why in CLIPS:
        if args.only and name not in args.only:
            continue
        bvh = LAFAN / f"{stem}.bvh"
        gdir = args.gmr_dir or (log / "gmr")
        npz_db = gdir / f"{name}_dropbear.npz"
        npz_g1 = gdir / f"{name}_g1.npz"
        for robot, out in (("dropbear", npz_db), ("unitree_g1", npz_g1)):
            if args.reuse_gmr and out.is_file():
                continue
            gmr_retarget.main(["--bvh", str(bvh), "--robot", robot, "--start", str(a), "--end", str(b), "--out", str(out),
                               "--preroll", str(args.preroll)])
        rep = log / "route_compare" / f"{name}.json"
        gmr_to_dropbear_csv.main([str(npz_db), "--clip", name, "--report", str(rep), "--out-root", str(args.out_root)]
                                 + ([] if args.no_g1 else ["--g1-npz", str(npz_g1)]) + list(args.csv_args))
        if args.no_g1:
            rj = json.loads(rep.read_text())
            rj["ik_health"] = {"serial": ik_health(npz_db)}
            rep.write_text(json.dumps(rj, indent=1), encoding="utf-8")
            side_path = Path(rj["gmr_serial"]["csv"]).with_suffix(".json")
            side = json.loads(side_path.read_text(encoding="utf-8"))
            side["ik_health"] = rj["ik_health"]["serial"]
            if rj["ik_health"]["serial"]["ok"]:
                side.pop("INVALID", None)
            else:
                side["INVALID"] = "GMR IK-health gate failed: " + "; ".join(rj["ik_health"]["serial"]["problems"])
            side_path.write_text(json.dumps(side, indent=1), encoding="utf-8")
            summary.append({"clip": name, "category": cat, "csv": rj["gmr_serial"]["csv"],
                            "ik_health_ok": rj["ik_health"]["serial"]["ok"],
                            "plant_gates_predicted": rj["gmr_serial"].get("plant_gates_predicted")})
            catalog.append({"clip": name, "csv": rj["gmr_serial"]["csv"], "category": cat,
                            "source_file": side["source_file"], "window_s": [a, b], "fps": side["fps"],
                            "num_frames": side["num_frames"], "license": side["source_license"]["license"],
                            "redistributable": False, "retarget_method": side["retarget_method"],
                            "frames_with_any_saturation_frac": rj["gmr_serial"]["frames_with_any_saturation_frac"],
                            "ik_health_ok": rj["ik_health"]["serial"]["ok"]})
            continue
        health = {"serial": ik_health(npz_db), "g1": ik_health(npz_g1)}
        rj = json.loads(rep.read_text())
        rj["ik_health"] = health
        if not health["g1"]["ok"]:
            rj["g1_route"]["INVALID"] = "G1 GMR solve failed the IK-health gate: " + "; ".join(health["g1"]["problems"])
        rep.write_text(json.dumps(rj, indent=1), encoding="utf-8")
        # flag the clips themselves, so nobody settles a route whose IK failed (tools/settle_motion.py refuses INVALID)
        for route, key in ((rj["gmr_serial"]["csv"], "serial"), (rj["g1_route"]["csv"], "g1")):
            side_path = Path(route).with_suffix(".json")
            side = json.loads(side_path.read_text(encoding="utf-8"))
            side["ik_health"] = health[key]
            if health[key]["ok"]:
                side.pop("INVALID", None)
            else:
                side["INVALID"] = "GMR IK-health gate failed: " + "; ".join(health[key]["problems"])
            side_path.write_text(json.dumps(side, indent=1), encoding="utf-8")
        r = json.loads(rep.read_text())
        g, h = r["gmr_serial"], r["g1_route"]
        worst = max(h["saturation_max_excess_deg"].items(), key=lambda kv: kv[1]) if h["saturation_max_excess_deg"] else ("-", 0)
        foot_err = max(g["gmr_ik_task_errors"][k]["pos_err_p95_m"] for k in ("LeftFootMod", "RightFootMod"))
        side_sat = json.loads(Path(g["csv"]).with_suffix(".json").read_text())["saturation"]
        at_lim = {k: round(v, 2) for k, v in side_sat["ik_at_limit"].items() if v >= 0.05}
        sw = max(side_sat["per_joint"].items(), key=lambda kv: kv[1]["max_excess_deg"])
        summary.append({"clip": name, "category": cat, "bvh": stem, "window_s": [a, b], "why": why,
                        "frames": int(np.load(npz_db)["qpos"].shape[0]),
                        "serial_sat": g["frames_with_any_saturation_frac"], "g1_sat": h["frames_with_any_saturation_frac"],
                        "g1_worst": f"{worst[0]} {worst[1]:.1f}", "serial_at_limit": at_lim,
                        "serial_worst": f"{sw[0]} {sw[1]['max_excess_deg']:.1f}",
                        "slip": [g["plant"]["stance_foot_slip_p95_mps"] or 0.0, h["plant"]["stance_foot_slip_p95_mps"] or 0.0],
                        "float": [g["plant"]["stance_float_p95_m"] or 0.0, h["plant"]["stance_float_p95_m"] or 0.0],
                        "vel": [g["plant"]["frames_motor_over_velocity_limit"], h["plant"]["frames_motor_over_velocity_limit"]],
                        "foot_err": foot_err, "report": str(rep).replace("\\", "/"), "health": health,
                        "contact_agreement": json.loads(Path(g["csv"]).with_suffix(".json").read_text())["metrics"].get(
                            "contact_agreement")})
        side = json.loads(Path(g["csv"]).with_suffix(".json").read_text())
        catalog.append({"clip": name, "csv": g["csv"], "category": cat, "source_file": side["source_file"],
                        "window_s": [a, b], "fps": side["fps"], "num_frames": side["num_frames"],
                        "license": side["source_license"]["license"], "redistributable": False,
                        "retarget_method": "gmr-serial-v1",
                        "frames_with_any_saturation_frac": g["frames_with_any_saturation_frac"],
                        "g1_route_csv": h["csv"], "g1_route_valid": health["g1"]["ok"],
                        "ik_health_ok": health["serial"]["ok"]})
        if not args.no_render:
            render_montage(npz_db, log / "renders" / f"{name}.png")
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "clips": summary, "wall_s": time.time() - t0,
           "license": "LAFAN1 CC BY-NC-ND 4.0 -- internal R&D only; never redistribute outputs"}
    if args.no_g1:
        (log / "gmr_serial_batch.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        cat_path = args.out_root / "gmr_lafan1/catalog_gmr_lafan1.json"
        cat_path.parent.mkdir(parents=True, exist_ok=True)
        cat_path.write_text(json.dumps({"schema": "dropbear-motion-catalog-v1", "source": "gmr_lafan1",
                                        "license_note": out["license"], "clips": catalog}, indent=1), encoding="utf-8")
        print(json.dumps(summary, indent=1))
        print(f"[gmr_lafan1_batch] {len(summary)} clips in {time.time() - t0:.1f} s")
        return 0
    if not args.only:
        (log / "route_comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        (log / "route_comparison.md").write_text(md_table(summary) + "\n", encoding="utf-8")
        cat_path = args.out_root / "gmr_lafan1/catalog_gmr_lafan1.json"
        cat_path.write_text(json.dumps({"schema": "dropbear-motion-catalog-v1", "source": "gmr_lafan1",
                                        "license_note": out["license"], "clips": catalog}, indent=1), encoding="utf-8")
    print(md_table(summary))
    print(f"[gmr_lafan1_batch] {len(summary)} clips in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
