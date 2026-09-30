"""Validate settled motion NPZs (``dropbear-motion-npz-v1``) and write a per-clip summary table.

CPU only (numpy). For each NPZ (``tools/settle_motion.py`` output):

* loads it with the tracking env's reader (``dropbear_wbc.tasks.tracking.motion_npz``) and checks provenance
  (schema, USD SHA, ankle variant of the current default plant unless ``--ankle any``);
* ``fps`` == 50, no NaN/Inf anywhere;
* ``closure_residual_m`` p50 / p95 / max (threshold ``--max-closure``, default 5 mm, on the max);
* feet on the ground while in contact: the lowest sole point (collision convex hulls,
  ``data/calibration/dropbear_foot_sole_hulls.json``) of every foot flagged in contact (NPZ ``contact``, i.e. the
  sidecar ``contact_hint`` or the lower foot) must satisfy |z| < ``--max-contact-z`` (1.5 cm); also reports the
  contact-foot horizontal slip speed (p95) and the deepest penetration of any foot;
* root (``world``) and anchor heights: finite, anchor z within [``--anchor-z-min``, ``--anchor-z-max``];
* saturation summary from the source sidecar (``meta.sidecar.saturation``): the joints clipped by
  ``SemanticMap.semantic_to_motor`` (fraction of frames, max excess);
* reference dynamics of the 22 MOTORS (added 2026-09-24, review fix): max |joint_vel| must be <= ``--max-motor-vel``
  (default 10 rad/s = the USD motor ``physxJoint:maxJointVelocity`` 572.96 deg/s, CONTRACTS 0.1; the legacy config's
  looser 20 rad/s arm velocity limit is NOT used) and the largest frame-to-frame motor step must be <=
  ``--max-motor-step`` (default 0.3 rad per 20 ms frame). Both catch retarget branch flips (e.g. a serial3 shoulder
  solution jumping by pi) that the settle faithfully passes through; these values enter the command observation and
  the RSI joint_vel writes.

``--write-verdicts`` writes ``<clip>.validation.json`` (schema ``dropbear-motion-validation-v1``, with the NPZ's
``npz_sha256``) next to each NPZ; the tracking env refuses a ``rejected`` verdict unless ``allow_rejected_motion``
(``dropbear_wbc.tasks.tracking.motion_npz.load_validation_verdict``).

Writes ``--out-json`` (all numbers) and ``--out-md`` (table). ``--body-heights CLIP`` additionally dumps hand /
foot / anchor heights over time for that clip (e.g. to show that wave_right raises the right hand while the feet
stay planted).

    python tools/validate_motion_npz.py data/motions/synthetic/wave_right.npz ... \
        --out-json logs/gpu_pipeline/settle_validation.json --out-md logs/gpu_pipeline/settle_summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.robots.dropbear_names import (  # noqa: E402
    ANCHOR_BODY,
    HAND_BODIES,
    USD_SHA256,
    authored_ankle_requested,
)
from dropbear_wbc.settle.ground import SoleModel  # noqa: E402
from dropbear_wbc.tasks.tracking.motion_npz import (  # noqa: E402
    MotionFormatError,
    load_motion_npz,
    npz_authored_ankle,
    validate_provenance,
)


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else float("nan")


def saturation_summary(sidecar: dict) -> dict:
    sat = (sidecar or {}).get("saturation") or {}
    per = sat.get("per_joint") or {}
    rows = sorted(((k, v.get("frac_frames", 0.0), v.get("max_excess_deg", 0.0)) for k, v in per.items()),
                  key=lambda r: -r[1])
    sat_rows = [r for r in rows if r[1] > 0]
    return {
        "any_frac": float(sat.get("frames_with_any_saturation_frac", max([r[1] for r in rows], default=0.0))),
        "joints_saturated": len(sat_rows),
        "worst": [{"joint": k, "frac_frames": round(f, 3), "max_excess_deg": round(e, 1)} for k, f, e in sat_rows[:4]],
        "max_excess_deg": round(max([r[2] for r in rows], default=0.0), 1),
        "keys": sorted(k for k in sat.keys() if k != "per_joint"),
    }


def validate(path: Path, sole: SoleModel, args) -> dict:
    rec: dict = {"npz": str(path).replace("\\", "/"), "problems": []}
    try:
        m = load_motion_npz(path)
    except Exception as e:  # noqa: BLE001
        rec["problems"].append(f"load failed: {e}")
        rec["ok"] = False
        return rec
    meta = m.meta or {}
    rec["status"] = meta.get("status")
    rec["authored_ankle_tierods"] = npz_authored_ankle(meta)
    expected_ankle = None if args.ankle == "any" else (args.ankle == "authored" or
                                                        (args.ankle == "default" and authored_ankle_requested()))
    try:
        validate_provenance(m, expected_usd_sha256=USD_SHA256, expected_authored_ankle=expected_ankle)
    except MotionFormatError as e:
        rec["problems"].append(f"provenance: {e}")
    t = m.joint_pos.shape[0]
    rec["frames"] = int(t)
    rec["fps"] = float(m.fps)
    rec["duration_s"] = round((t - 1) / float(m.fps), 3)
    if abs(float(m.fps) - 50.0) > 1e-6:
        rec["problems"].append(f"fps {m.fps} != 50")
    arrays = {"joint_pos": m.joint_pos, "joint_vel": m.joint_vel, "body_pos_w": m.body_pos_w,
              "body_quat_w": m.body_quat_w, "body_lin_vel_w": m.body_lin_vel_w, "body_ang_vel_w": m.body_ang_vel_w,
              "closure_residual_m": m.closure_residual_m}
    nonfinite = [k for k, v in arrays.items() if not np.isfinite(v).all()]
    rec["nonfinite_keys"] = nonfinite
    if nonfinite:
        rec["problems"].append(f"non-finite values in {nonfinite}")
    cr = np.asarray(m.closure_residual_m, dtype=np.float64)
    rec["closure_mm"] = {"p50": 1e3 * _pct(cr, 50), "p95": 1e3 * _pct(cr, 95), "max": 1e3 * float(cr.max())}
    if cr.max() > args.max_closure:
        rec["problems"].append(f"closure residual max {1e3 * cr.max():.2f} mm > {1e3 * args.max_closure:.1f} mm")
    names = list(m.body_names)
    pos = np.asarray(m.body_pos_w, dtype=np.float64)
    quat = np.asarray(m.body_quat_w, dtype=np.float64)
    low = sole.lowest_z(pos, quat, names)
    with np.load(path, allow_pickle=False) as d:
        contact = np.asarray(d["contact"], dtype=bool) if "contact" in d.files else None
    if contact is None:
        contact = np.stack([low["left"] <= low["right"], low["right"] < low["left"]], -1)
        rec["contact_source"] = "lower foot (NPZ has no contact key)"
    else:
        rec["contact_source"] = (meta.get("ground") or {}).get("contact_source", "npz contact")
    h = np.stack([low["left"], low["right"]], -1)
    hc = h[contact]
    rec["contact_frac"] = {"left": float(contact[:, 0].mean()), "right": float(contact[:, 1].mean()),
                           "double": float(contact.all(-1).mean()), "none": float((~contact.any(-1)).mean())}
    rec["contact_sole_z_mm"] = {"min": 1e3 * float(hc.min()) if hc.size else None,
                                "max": 1e3 * float(hc.max()) if hc.size else None,
                                "abs_p95": 1e3 * _pct(np.abs(hc), 95),
                                "frac_over_limit": float((np.abs(hc) > args.max_contact_z).mean()) if hc.size else 0.0}
    rec["min_sole_z_any_foot_mm"] = 1e3 * float(h.min())
    if hc.size and np.abs(hc).max() > args.max_contact_z:
        rec["problems"].append(f"contact foot |z| up to {1e3 * np.abs(hc).max():.1f} mm > {1e3 * args.max_contact_z:.0f} mm "
                               f"on {100 * rec['contact_sole_z_mm']['frac_over_limit']:.1f}% of contact samples")
    # horizontal slip of contact feet (lowest sole point is not tracked; use the foot body origin)
    from dropbear_wbc.settle.ground import FOOT_GROUPS

    slips = []
    for k, side in enumerate(("left", "right")):
        i = names.index(FOOT_GROUPS[side][0])
        v = np.linalg.norm(np.diff(pos[:, i, :2], axis=0), axis=-1) * float(m.fps)
        both = contact[1:, k] & contact[:-1, k]
        slips.append(v[both])
    s = np.concatenate(slips) if slips else np.zeros(0)
    rec["contact_foot_slip_mps"] = {"p95": _pct(s, 95), "max": float(s.max()) if s.size else None}
    ia = names.index(ANCHOR_BODY)
    root_z = pos[:, 0, 2]
    anchor_z = pos[:, ia, 2]
    rec["root_z_m"] = {"min": float(root_z.min()), "max": float(root_z.max())}
    rec["anchor_z_m"] = {"min": float(anchor_z.min()), "max": float(anchor_z.max()), "first": float(anchor_z[0])}
    if not (args.anchor_z_min <= anchor_z.min() and anchor_z.max() <= args.anchor_z_max):
        rec["problems"].append(f"anchor z range [{anchor_z.min():.3f}, {anchor_z.max():.3f}] m outside "
                               f"[{args.anchor_z_min}, {args.anchor_z_max}]")
    # reference dynamics of the motors (branch flips / impossible speeds)
    from dropbear_wbc.settle.quality import motor_dynamics

    md = motor_dynamics(m.joint_pos, m.joint_vel, m.joint_names, m.motor_names, args.max_motor_vel, args.max_motor_step)
    rec["problems"] += md.pop("problems")
    rec["motor_dynamics"] = md
    rec["saturation"] = saturation_summary(meta.get("sidecar") or {})
    rec["settle"] = {k: meta.get(k) for k in ("motor_clipping_frames", "motor_tracking_error_max_rad", "dq_window_max")}
    rec["settle_runtime_s"] = (meta.get("runtime") or {}).get("settle_s")
    rec["source"] = (meta.get("sidecar") or {}).get("source")
    # tools/make_static_npz.py writes no meta.status (not a settle output); only "rejected" is disqualifying
    rec["ok"] = not rec["problems"] and rec["status"] != "rejected"
    return rec


def write_verdict(path: Path, rec: dict, args) -> Path:
    """``<clip>.validation.json`` next to the NPZ (read by the tracking env, fail closed on 'rejected')."""
    import datetime as _dt
    import hashlib

    from dropbear_wbc.tasks.tracking.motion_npz import VALIDATION_SCHEMA, validation_sidecar_path

    out = validation_sidecar_path(path)
    try:
        npz_rel = str(path.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        npz_rel = str(path).replace("\\", "/")
    data = {
        "schema": VALIDATION_SCHEMA, "tool": "tools/validate_motion_npz.py",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "npz": npz_rel, "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "verdict": "accepted" if rec.get("ok") else "rejected",
        "reasons": list(rec.get("problems") or []) + (["settle meta.status == 'rejected'"]
                                                      if rec.get("status") == "rejected" else []),
        "gates": {"closure_max_mm": 1e3 * args.max_closure, "contact_sole_abs_z_mm": 1e3 * args.max_contact_z, "fps": 50,
                  "max_motor_vel_rad_s": args.max_motor_vel, "max_motor_step_rad": args.max_motor_step},
        **{k: rec.get(k) for k in ("closure_mm", "contact_sole_z_mm", "contact_foot_slip_mps", "anchor_z_m",
                                   "motor_dynamics", "saturation")},
        "log": args.log,
        "note": "settle meta.status refers to the settle's closure gate; this file is the additional validation gate",
    }
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return out


def body_heights(path: Path, sole: SoleModel, every_s: float) -> dict:
    m = load_motion_npz(path)
    names = list(m.body_names)
    pos = np.asarray(m.body_pos_w, dtype=np.float64)
    low = sole.lowest_z(pos, np.asarray(m.body_quat_w, dtype=np.float64), names)
    il, ir = names.index(HAND_BODIES[0]), names.index(HAND_BODIES[1])
    ia = names.index(ANCHOR_BODY)
    fps = float(m.fps)
    step = max(1, int(round(every_s * fps)))
    rows = []
    for f in range(0, pos.shape[0], step):
        rows.append({"t": round(f / fps, 2), "right_hand_z": round(float(pos[f, ir, 2]), 4),
                     "left_hand_z": round(float(pos[f, il, 2]), 4), "anchor_z": round(float(pos[f, ia, 2]), 4),
                     "left_sole_z": round(float(low["left"][f]), 4), "right_sole_z": round(float(low["right"][f]), 4)})
    rh, lh = pos[:, ir, 2], pos[:, il, 2]
    return {"npz": str(path).replace("\\", "/"), "hand_bodies": list(HAND_BODIES), "rows": rows,
            "right_hand_rise_m": float(rh.max() - rh[0]), "left_hand_rise_m": float(lh.max() - lh[0]),
            "right_hand_z_range": [float(rh.min()), float(rh.max())], "left_hand_z_range": [float(lh.min()), float(lh.max())],
            "right_hand_minus_left_max_m": float((rh - lh).max()),
            "sole_z_range_left": [float(low["left"].min()), float(low["left"].max())],
            "sole_z_range_right": [float(low["right"].min()), float(low["right"].max())],
            "right_hand_above_anchor_frames_frac": float((rh > pos[:, ia, 2]).mean())}


def to_markdown(recs: list[dict], heights: dict | None) -> str:
    out = ["| clip | status | dur s | frames | frames with any saturation | closure p50/p95/max mm | "
           "contact sole z min/max mm (% of contact samples with abs(z) > 15 mm) | slip p95 m/s | anchor z min/max m | "
           "saturated joints; worst 2 (frac of frames / max excess deg) | problems |",
           "|---|---|---:|---:|---:|---|---|---:|---|---|---|"]
    for r in recs:
        name = Path(r["npz"]).stem.replace(".rejected", " (rejected)")
        if "closure_mm" not in r:
            out.append(f"| {name} | load failed | | | | | | | | {'; '.join(r['problems'])} |")
            continue
        c, z, sat = r["closure_mm"], r["contact_sole_z_mm"], r["saturation"]
        worst = ", ".join(f"{w['joint']} {w['frac_frames']:.2f}/{w['max_excess_deg']:.0f}" for w in sat["worst"][:2])
        out.append(
            f"| {name} | {r['status'] or 'n/a'}{' PASS' if r['ok'] else ' FAIL'} | {r['duration_s']:.2f} | {r['frames']} | "
            f"{100 * sat['any_frac']:.0f}% | "
            f"{c['p50']:.2f} / {c['p95']:.2f} / {c['max']:.2f} | {z['min']:.1f} / {z['max']:.1f} "
            f"({100 * z['frac_over_limit']:.1f}) | {r['contact_foot_slip_mps']['p95']:.3f} | "
            f"{r['anchor_z_m']['min']:.3f} / {r['anchor_z_m']['max']:.3f} | {sat['joints_saturated']}; {worst or '-'} | "
            f"{'; '.join(r['problems']) or '-'} |")
    if heights:
        out += ["", f"### Body heights: {Path(heights['npz']).stem}", "",
                f"Hands = `{heights['hand_bodies'][0]}` (left), `{heights['hand_bodies'][1]}` (right). "
                f"Right hand rises {heights['right_hand_rise_m']:.3f} m above its start, left hand "
                f"{heights['left_hand_rise_m']:.3f} m. Sole z range left {heights['sole_z_range_left'][0] * 1e3:.1f}.."
                f"{heights['sole_z_range_left'][1] * 1e3:.1f} mm, right {heights['sole_z_range_right'][0] * 1e3:.1f}.."
                f"{heights['sole_z_range_right'][1] * 1e3:.1f} mm.", "",
                "| t s | right hand z | left hand z | anchor z | left sole z | right sole z |", "|---:|---:|---:|---:|---:|---:|"]
        for row in heights["rows"]:
            out.append(f"| {row['t']:.2f} | {row['right_hand_z']:.3f} | {row['left_hand_z']:.3f} | {row['anchor_z']:.3f} | "
                       f"{row['left_sole_z']:.4f} | {row['right_sole_z']:.4f} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, nargs="+")
    ap.add_argument("--sole", type=Path, default=REPO / "data/calibration/dropbear_foot_sole_hulls.json")
    ap.add_argument("--max-closure", type=float, default=0.005)
    ap.add_argument("--max-contact-z", type=float, default=0.015)
    ap.add_argument("--max-motor-vel", type=float, default=10.0,
                    help="max |motor joint_vel| of the reference [rad/s] (USD motor maxJointVelocity)")
    ap.add_argument("--max-motor-step", type=float, default=0.3, help="max motor change between frames [rad]")
    ap.add_argument("--write-verdicts", action="store_true", help="write <clip>.validation.json next to each NPZ")
    ap.add_argument("--log", default=None, help="log path recorded in the verdict files")
    ap.add_argument("--anchor-z-min", type=float, default=0.8)
    ap.add_argument("--anchor-z-max", type=float, default=2.0)
    ap.add_argument("--ankle", choices=["default", "spherical", "authored", "any"], default="default",
                    help="expected ankle variant (default: the current default plant)")
    ap.add_argument("--body-heights", type=str, default=None, help="clip stem for the body-height dump")
    ap.add_argument("--every-s", type=float, default=0.5)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()
    sole = SoleModel.load(args.sole)
    recs = [validate(p, sole, args) for p in args.npz]
    heights = None
    if args.body_heights:
        match = [p for p in args.npz if p.stem == args.body_heights]
        if match:
            heights = body_heights(match[0], sole, args.every_s)
    for r in recs:
        print(json.dumps({k: r.get(k) for k in ("npz", "ok", "status", "duration_s", "closure_mm", "contact_sole_z_mm",
                                                "anchor_z_m", "problems")}), flush=True)
    if args.write_verdicts:
        for p, r in zip(args.npz, recs):
            if "closure_mm" in r:
                print(f"[validate] verdict -> {write_verdict(p, r, args)}", flush=True)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({"clips": recs, "body_heights": heights}, indent=1) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(to_markdown(recs, heights), encoding="utf-8")
    return 0 if all(r["ok"] for r in recs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
