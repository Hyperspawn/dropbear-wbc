"""Per-clip gate report (before/after the foot-contact stage) and the gate-annotated motion catalog.

For every retargeted clip ``<root>/<source>/<clip>.csv`` (library v4; archived previous versions ``<clip>_v3.csv``):

* **predicted** gates (kinematic, before the settle): ``foot_contact.plant_gate_metrics`` on the CSV resampled to 50 Hz
  exactly as ``tools/settle_motion.py`` does (``settle.motion_io.read_motion_csv`` + ``resample``), with the sidecar's
  contact hint: contact-foot lowest sole z (plant forward model MeasuredFK + sole hulls; |z| < 15 mm gate), contact-foot
  slip, motor speed (<= 10 rad/s) and step (<= 0.3 rad/frame). Computed for v4 AND v3, so every clip has a before/after
  even where the v3 clip was never settled.
* **physics** verdicts (after the settle): ``<clip>_v4.validation.json`` of the settled v4 NPZ and, where one exists, the
  legacy ``<clip>.validation.json`` of the NPZ settled from the v3 CSV (``tools/validate_motion_npz.py --write-verdicts``;
  stale verdicts are reported as such).

``--write-catalog`` writes ``dropbear-motion-catalog-v1`` (+ gate fields) for the v4 library; synthetic clips (not
retargeted) are listed with their own NPZ verdicts. ``--out-json`` / ``--out-md`` write the report.

    python tools/motion_gate_report.py --root data/motions --write-catalog data/motions/catalog.json \
        --out-json logs/foot_contact/gate_report.json --out-md logs/foot_contact/gate_report.md

foot_contact track, 2026-09-24. CPU only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

SOURCES = ("unitree_rl_lab_mimic", "kimodo_g1", "soma_retargeter_g1", "gmr_lafan1", "gmr_lafan1_via_g1", "asap_g1")
GATES = {"contact_sole_abs_z_mm": 15.0, "max_motor_vel_rad_s": 10.0, "max_motor_step_rad": 0.3, "closure_max_mm": 5.0}
SLIP_NOTE = ("slip is reported, not gated by tools/validate_motion_npz.py; this report calls p95 <= 0.10 m/s "
             "(2 mm per 50 Hz frame) 'planted'")


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def predict(csv: Path, cfk) -> dict[str, Any] | None:
    from dropbear_wbc.motion.foot_contact import plant_gate_metrics
    from dropbear_wbc.settle.motion_io import read_motion_csv, resample

    if not csv.is_file():
        return None
    clip = resample(read_motion_csv(csv), 50.0)
    # the settle re-grounds every frame (contact-foot lowest sole -> z = 0, sigma 0.1 s): predict what it will see
    g = plant_gate_metrics(cfk, clip.motor_pos, clip.root_pos, clip.root_quat, clip.contact, 50.0,
                           settle_ground_sigma_s=0.1)
    g["input_fps"] = float(read_motion_csv(csv).fps)
    return g


def verdict(npz: Path) -> dict[str, Any] | None:
    from dropbear_wbc.tasks.tracking.motion_npz import load_validation_verdict, validation_sidecar_path

    cand = [npz, npz.with_name(npz.stem + ".rejected.npz")]
    for p in cand:
        if p.is_file():
            v = load_validation_verdict(p)
            out = {"npz": _rel(p), "verdict": None if v is None else v["verdict"],
                   "stale": None if v is None else v.get("stale"),
                   "reasons": [] if v is None else v.get("reasons", [])}
            vp = validation_sidecar_path(p)
            if vp.is_file():
                d = json.loads(vp.read_text(encoding="utf-8"))
                out["gates"] = {k: d.get(k) for k in ("closure_mm", "contact_sole_z_mm", "contact_foot_slip_mps",
                                                       "anchor_z_m")}
                md = d.get("motor_dynamics") or {}
                out["gates"]["motor_dynamics"] = {k: md.get(k) for k in ("max_abs_vel_rad_s", "max_abs_vel_joint",
                                                                         "max_step_rad", "max_step_joint")}
            return out
    return None


def stage_summary(side: dict) -> dict[str, Any] | None:
    st = (side.get("g1_mapping") or {}).get("foot_contact_stage") or (side.get("metrics") or {}).get("foot_contact_stage")
    if not st:
        return None
    a, pc = st.get("after") or {}, st.get("pelvis_correction") or {}
    return {"stage": st.get("stage"), "stance_pos_mm_max": (a.get("stance_pos_mm") or {}).get("max"),
            "stance_tilt_deg_max": (a.get("stance_tilt_deg") or {}).get("max"),
            "stance_yaw_err_deg_max": (a.get("stance_yaw_err_deg") or {}).get("max"),
            "swing_min_sole_z_mm": a.get("swing_min_sole_z_mm"), "pelvis_dz_m": pc.get("dz_m"),
            "pelvis_dxy_m_max": pc.get("dxy_m_max"), "pelvis_tilt_deg_max": pc.get("tilt_deg_max"),
            "contacts_clean": st.get("contacts_clean"), "solver_iters": (st.get("solver") or {}).get("iters"),
            "wall_s": st.get("wall_s")}


def ankle_edge(csv: Path, side: dict, cfk) -> dict[str, float] | None:
    """Fraction of frames whose ankle (pitch, roll) lies where the SemanticMap's lut2d inverse clips (inverse cells
    touching the feasible boundary): the calf motors are valid there, but semantic_to_motor would not return them."""
    from dropbear_wbc.motion.names import SEMANTIC_NAMES

    rel = side.get("semantic_trajectory")
    if not rel or not (csv.parent / rel).is_file():
        return None
    sem = np.loadtxt(csv.parent / rel, delimiter=",", skiprows=1)[:, 7:29]
    _, rep = cfk.smap.semantic_to_motor(sem, return_report=True)
    cl = np.asarray(rep.clipped).reshape(len(sem), -1)
    return {s: float(cl[:, [SEMANTIC_NAMES.index(f"{s}_ankle_pitch"), SEMANTIC_NAMES.index(f"{s}_ankle_roll")]]
                     .any(axis=1).mean()) for s in ("left", "right")}


def clip_record(csv: Path, cfk, with_v3: bool = True) -> dict[str, Any]:
    side = json.loads(csv.with_suffix(".json").read_text(encoding="utf-8"))
    stem = csv.stem
    rec: dict[str, Any] = {"clip": side.get("clip", stem), "source": side.get("source"), "csv": _rel(csv),
                           "fps": side.get("fps"), "num_frames": side.get("num_frames"),
                           "duration_s": round(float(side.get("duration_s") or 0.0), 3),
                           "INVALID": side.get("INVALID"), "retarget_method": side.get("retarget_method")}
    rec["predicted"] = predict(csv, cfk)
    rec["physics"] = verdict(csv.with_name(stem + "_v4.npz"))
    rec["foot_contact_stage"] = stage_summary(side)
    rec["ankle_at_feasible_edge_frac"] = ankle_edge(csv, side, cfk)
    if with_v3:
        v3 = csv.with_name(stem + "_v3.csv")
        rec["v3"] = {"csv": _rel(v3) if v3.is_file() else None, "predicted": predict(v3, cfk),
                     "physics": verdict(csv.with_name(stem + ".npz"))}
    return rec


def status(rec: dict) -> str:
    ph = rec.get("physics")
    if rec.get("INVALID"):
        return "invalid (retarget IK-health gate)"
    if ph is None or ph.get("verdict") is None:
        return "not settled"
    if ph.get("stale"):
        return f"{ph['verdict']} (STALE)"
    return ph["verdict"]


def _z(g: dict | None) -> str:
    if not g:
        return "-"
    z = g.get("contact_sole_z_mm") or {}
    if z.get("min") is None:
        return "no contact"
    return f"{z['min']:.1f}/{z['max']:.1f} ({100 * (z.get('frac_over_limit') or 0):.1f}%)"


def _dyn_pred(g: dict | None) -> str:
    if not g:
        return "-"
    return f"{g['motor_max_abs_vel_rad_s']:.1f} / {g['motor_max_step_rad']:.2f}"


def _dyn_phys(ph: dict | None) -> str:
    md = ((ph or {}).get("gates") or {}).get("motor_dynamics") or {}
    if md.get("max_abs_vel_rad_s") is None:
        return "-"
    return f"{md['max_abs_vel_rad_s']:.1f} / {md['max_step_rad']:.2f}"


def _slip(g: dict | None, key="contact_foot_slip_mps") -> str:
    s = (g or {}).get(key) or {}
    return "-" if s.get("p95") is None else f"{s['p95']:.3f}"


def to_markdown(recs: list[dict]) -> str:
    out = ["| source | clip | s | status (v4 physics) | contact sole z min/max mm (% > 15 mm): v3 pred -> v4 pred -> v4 physics "
           "| v3 physics | slip p95 m/s: v3 pred -> v4 physics | motor vel rad/s / step rad: v3 pred -> v4 physics "
           "| closure max mm | reasons |",
           "|---|---|---:|---|---|---|---|---|---:|---|"]
    for r in recs:
        v3 = r.get("v3") or {}
        ph = r.get("physics") or {}
        pz = ((ph.get("gates") or {}).get("contact_sole_z_mm")) or {}
        phz = "-" if pz.get("min") is None else f"{pz['min']:.1f}/{pz['max']:.1f} ({100 * (pz.get('frac_over_limit') or 0):.1f}%)"
        v3ph = v3.get("physics") or {}
        v3z = ((v3ph.get("gates") or {}).get("contact_sole_z_mm")) or {}
        v3phs = "-" if not v3ph else (f"{v3ph.get('verdict')}: {v3z['min']:.1f}/{v3z['max']:.1f}"
                                      if v3z.get("min") is not None else str(v3ph.get("verdict")))
        cl = ((ph.get("gates") or {}).get("closure_mm")) or {}
        slip_ph = ((ph.get("gates") or {}).get("contact_foot_slip_mps")) or {}
        slip_ph_s = "-" if slip_ph.get("p95") is None else f"{slip_ph['p95']:.3f}"
        cl_s = "-" if cl.get("max") is None else f"{cl['max']:.2f}"
        reasons = "; ".join((ph.get("reasons") or [])[:2]) or ("-" if not r.get("INVALID") else r["INVALID"][:80])
        out.append(
            f"| {r['source']} | {r['clip']} | {r['duration_s']:.1f} | {status(r)} | "
            f"{_z(v3.get('predicted'))} -> {_z(r.get('predicted'))} -> {phz} | {v3phs} | "
            f"{_slip(v3.get('predicted'))} -> {slip_ph_s} | "
            f"{_dyn_pred(v3.get('predicted'))} -> {_dyn_phys(ph)} | {cl_s} | {reasons} |")
    return "\n".join(out) + "\n"


def catalog_entry(rec: dict, side: dict) -> dict[str, Any]:
    sat = side.get("saturation") or {}
    per = sat.get("per_joint") or {}
    worst = sat.get("worst_joint")
    lic = side.get("source_license") or {}
    return {
        "clip": rec["clip"], "source": rec["source"], "csv": rec["csv"],
        "npz": (rec.get("physics") or {}).get("npz"), "status": status(rec),
        "verdict": (rec.get("physics") or {}).get("verdict"),
        "verdict_reasons": (rec.get("physics") or {}).get("reasons"),
        "gates_physics": (rec.get("physics") or {}).get("gates"),
        "gates_predicted": {k: (rec.get("predicted") or {}).get(k) for k in
                            ("contact_sole_z_mm", "contact_foot_slip_mps", "motor_max_abs_vel_rad_s",
                             "motor_max_step_rad", "gates_pass_predicted", "contact_frac")},
        "foot_contact_stage": rec.get("foot_contact_stage"),
        "ankle_at_feasible_edge_frac": rec.get("ankle_at_feasible_edge_frac"),
        "previous_v3": rec.get("v3") and {"csv": rec["v3"]["csv"], "npz": (rec["v3"].get("physics") or {}).get("npz"),
                                          "verdict": (rec["v3"].get("physics") or {}).get("verdict"),
                                          "predicted_contact_sole_z_mm": (rec["v3"].get("predicted") or {}).get(
                                              "contact_sole_z_mm")},
        "license": lic.get("license"), "redistributable": lic.get("redistributable"),
        "fps": rec["fps"], "num_frames": rec["num_frames"], "duration_s": rec["duration_s"],
        "suitability": side.get("suitability", []),
        "category": side.get("category"),
        "INVALID": side.get("INVALID"),
        "saturation_summary": {"frames_with_any_saturation_frac": sat.get("frames_with_any_saturation_frac"),
                               "worst_joint": worst,
                               "worst_joint_max_excess_deg": (per.get(worst) or {}).get("max_excess_deg")},
        "mock_calibration": (side.get("calibration") or {}).get("calibration_is_mock"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "data/motions")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--only", nargs="*", default=None, help="clip stems")
    ap.add_argument("--no-v3", action="store_true")
    ap.add_argument("--write-catalog", type=Path, default=None)
    ap.add_argument("--previous-catalog", type=Path, default=None, help="for skipped/licenses (default <root>/catalog_v3.json)")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args(argv)
    from dropbear_wbc.motion.foot_contact import get_calibration_fk

    cfk = get_calibration_fk()
    recs, sides = [], []
    for src in args.sources.split(","):
        d = args.root / src
        for sp in sorted(d.glob("*.json")):
            if "." in sp.stem or sp.stem.startswith("catalog") or sp.stem.rsplit("_", 1)[-1][:1] == "v" and \
                    sp.stem.rsplit("_", 1)[-1][1:].isdigit():
                continue
            side = json.loads(sp.read_text(encoding="utf-8"))
            if side.get("schema") != "dropbear-motion-csv-v1" or not sp.with_suffix(".csv").is_file():
                continue
            if args.only and sp.stem not in args.only:
                continue
            rec = clip_record(sp.with_suffix(".csv"), cfk, with_v3=not args.no_v3)
            recs.append(rec)
            sides.append(side)
            print(f"[gate_report] {src}/{sp.stem}: {status(rec)}", flush=True)
    order = {s: i for i, s in enumerate(SOURCES)}
    idx = sorted(range(len(recs)), key=lambda i: (order.get(recs[i]["source"], 99), recs[i]["clip"]))
    recs, sides = [recs[i] for i in idx], [sides[i] for i in idx]
    counts: dict[str, int] = {}
    for r in recs:
        counts[status(r)] = counts.get(status(r), 0) + 1
    report = {"created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "gates": GATES,
              "slip_note": SLIP_NOTE, "counts": counts, "clips": recs}
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.out_md:
        args.out_md.write_text(to_markdown(recs), encoding="utf-8")
    if args.write_catalog:
        prev_path = args.previous_catalog or (args.root / "catalog_v3.json")
        prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.is_file() else {}
        entries = [catalog_entry(r, s) for r, s in zip(recs, sides)]
        syn = []
        for sp in sorted((args.root / "synthetic").glob("*.json")):
            if "." in sp.stem:
                continue
            s = json.loads(sp.read_text(encoding="utf-8"))
            if s.get("schema") != "dropbear-motion-csv-v1":
                continue
            ph = verdict(sp.with_suffix(".npz"))
            syn.append({"clip": s.get("clip", sp.stem), "source": "synthetic", "csv": _rel(sp.with_suffix(".csv")),
                        "npz": (ph or {}).get("npz"), "status": "not settled" if ph is None else
                        (f"{ph['verdict']} (STALE)" if ph.get("stale") else ph["verdict"]),
                        "verdict": (ph or {}).get("verdict"), "gates_physics": (ph or {}).get("gates"),
                        "license": (s.get("source_license") or {}).get("license")
                        if isinstance(s.get("source_license"), dict) else s.get("source_license"),
                        "fps": s.get("fps"), "num_frames": s.get("num_frames"),
                        "duration_s": round(float(s.get("duration_s") or 0.0), 3)})
        first = next((s for s in sides if s.get("source") != "gmr_lafan1"), sides[0] if sides else {})
        licenses = dict(prev.get("licenses") or {})
        for s in sides:
            if s.get("source") not in licenses and isinstance(s.get("source_license"), dict):
                licenses[s["source"]] = s["source_license"]
        allc = entries + syn
        cat = {
            "schema": "dropbear-motion-catalog-v1", "library_version": "v4",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "created_by": "tools/motion_gate_report.py (foot_contact track)",
            "description": ("Library v4: every retargeted clip rebuilt with the foot-contact stage "
                            "(dropbear_wbc.motion.foot_contact) + serial continuity (review_fixes), 50 Hz, motor-step "
                            "projection 0.18 rad/frame; settled NPZs are <clip>_v4.npz with <clip>_v4.validation.json. "
                            "The previous (v3) CSVs are kept as <clip>_v3.csv; legacy NPZs <clip>.npz were settled from "
                            "them. 'status' = the v4 physics verdict (tools/validate_motion_npz.py)."),
            "pipeline_version": first.get("pipeline_version"),
            "calibration": first.get("calibration"),
            "retarget_options": first.get("retarget_options"),
            "gates": GATES, "slip_note": SLIP_NOTE,
            "gate_summary": {"retargeted": counts,
                             "all": {k: sum(1 for e in allc if e["status"] == k) for k in sorted({e["status"] for e in allc})}},
            "num_clips": len(allc),
            "total_duration_s": round(sum(float(e.get("duration_s") or 0) for e in allc), 2),
            "clips": allc,
            "skipped": prev.get("skipped", []), "failed": [],
            "licenses": licenses,
            "previous_catalog": _rel(prev_path) if prev_path.is_file() else None,
        }
        args.write_catalog.write_text(json.dumps(cat, indent=1), encoding="utf-8")
        print(f"[gate_report] catalog -> {args.write_catalog} ({len(allc)} clips; {counts})", flush=True)
    print(json.dumps(counts), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
