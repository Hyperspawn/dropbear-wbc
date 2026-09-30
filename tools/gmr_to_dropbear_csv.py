"""GMR retarget on the DERIVED serial Dropbear model -> contract motion CSV + sidecar (``gmr-serial-v1``).

    GMR output (root pos/rot + serial dof) -> dropbear-semantic-v1 trajectory -> SemanticMap.semantic_to_motor
    -> ``<out-root>/<source>/<clip>.csv`` + ``<clip>.json`` + ``semantic/<clip>.semantic.csv`` (CONTRACTS section 3)

Optionally (``--g1-npz``) the SAME human clip retargeted by GMR to Unitree G1 is pushed through the existing
G1-intermediate route (``dropbear_wbc.motion.g1_to_dropbear``) into ``<out-root>/<source>_via_g1/`` and both
routes are compared (joint ranges, saturation, plant-FK foot metrics) in ``--report``.

Runs with any interpreter that has numpy + the dropbear_wbc sources (``.venv-gmr``, ``.venv-newton`` or the
system Python)::

    python tools/gmr_to_dropbear_csv.py logs/serial_mjcf_gmr/gmr/walk1_subject1_dropbear.npz --clip lafan1_walk1_subject1 \
        --g1-npz logs/serial_mjcf_gmr/gmr/walk1_subject1_g1.npz --report logs/serial_mjcf_gmr/route_compare_walk.json

Library v4 (foot_contact track, 2026-09-24), DEFAULT: resample to the settle rate (``--fps-out`` default 50), run the
foot-contact stage (``dropbear_wbc.motion.foot_contact``: stance feet flat at world-locked poses with the plant leg FK,
human-foot contacts with height + toe/ankle-speed hysteresis) and project the motors under the 10 rad/s gate
(``--max-motor-step`` default 0.18). ``--no-foot-contact`` = the previous route. The ``--g1-npz`` comparison route
(G1-intermediate) runs the stage too unless ``--no-foot-contact``.

LAFAN1 (CC BY-NC-ND 4.0): outputs are Adapted Material for internal non-commercial R&D; never redistribute.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "source", REPO / "third_party" / "pydeps"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dropbear_wbc.kinematics.serial_model import DEFAULT_XML, CalibrationFK, sha256_file  # noqa: E402
from dropbear_wbc.motion.calibration_view import DEFAULT_CALIBRATION, load_calibration  # noqa: E402
from dropbear_wbc.motion.contacts import detect_contacts, horizontal_speed  # noqa: E402
from dropbear_wbc.motion.g1_model import G1_JOINT_NAMES  # noqa: E402
from dropbear_wbc.motion.g1_sources import G1Motion, LicenseInfo  # noqa: E402
from dropbear_wbc.motion.g1_to_dropbear import RetargetOptions, retarget_g1_motion  # noqa: E402
from dropbear_wbc.motion.gmr_serial import (RETARGET_METHOD, gmr_to_dropbear, joint_range_summary,  # noqa: E402
                                            load_gmr_npz, plant_feet, serial_provenance)
from dropbear_wbc.motion.motion_csv import validate_motion_files, write_motion  # noqa: E402
from dropbear_wbc.motion.names import SEMANTIC_NAMES  # noqa: E402
from dropbear_wbc.motion.rotations import quat_continuous, quat_to_matrix  # noqa: E402

LAFAN_LICENSE = LicenseInfo(
    "CC-BY-NC-ND-4.0",
    "Ubisoft La Forge Animation Dataset (LAFAN1), https://github.com/ubisoft/ubisoft-laforge-animation-dataset. "
    "Non-commercial; retargeted clips are Adapted Material: internal R&D use only, never share or redistribute.",
    redistributable=False,
    url="https://github.com/ubisoft/ubisoft-laforge-animation-dataset",
)


def plant_metrics(cfk: CalibrationFK, motor_q, root_pos, root_quat_wxyz, fps: float) -> dict:
    """Foot metrics on the plant kinematics, identical for both routes (contacts re-detected here)."""
    low, cen = plant_feet(cfk, motor_q, root_pos, quat_to_matrix(root_quat_wxyz))
    contacts, _ = detect_contacts(low, cen, fps, ground=0.0)
    slip, pen, flt = [], [], []
    for s in range(2):
        c = contacts[:, s]
        if c.any():
            v = horizontal_speed(cen[:, s], fps)
            slip.append(float(np.percentile(v[c], 95)))
            pen.append(float(max(0.0, -low[c, s].min())))
            flt.append(float(np.percentile(low[c, s], 95)))
    from dropbear_wbc.motion.gmr_serial import motor_velocity_limits

    dq = np.abs(np.diff(motor_q, axis=0)) * fps
    over = dq > motor_velocity_limits()
    return {"stance_foot_slip_p95_mps": max(slip) if slip else None, "stance_penetration_max_m": max(pen) if pen else None,
            "stance_float_p95_m": max(flt) if flt else None, "min_sole_height_m": float(low.min()),
            "contact_frac": [float(contacts[:, 0].mean()), float(contacts[:, 1].mean())],
            "frames_motor_over_velocity_limit": int(over.any(axis=1).sum()),
            "motor_speed_max_rad_s": float(dq.max())}


def g1_route(g1_npz: Path, clip: str, source: str, out_root: Path, cal, cfk, gmr_meta: dict,
             opts: RetargetOptions | None = None) -> dict:
    d = np.load(g1_npz, allow_pickle=False)
    if str(d["robot"]) != "unitree_g1":
        raise ValueError(f"{g1_npz}: robot {d['robot']} is not unitree_g1")
    names = [str(n) for n in d["qpos_joint_names"]]
    qpos = np.asarray(d["qpos"], dtype=np.float64)
    dof = np.stack([qpos[:, 7 + names.index(n)] for n in G1_JOINT_NAMES], axis=1)
    quat = qpos[:, 3:7] / np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    motion = G1Motion(fps=float(d["fps"]), root_pos=qpos[:, :3].copy(), root_quat_wxyz=quat_continuous(quat), dof=dof,
                      source=f"{source}_via_g1", license=LAFAN_LICENSE, source_file=gmr_meta["bvh"], clip=clip,
                      fmt="gmr_g1_npz", notes=[f"G1 input = GMR unitree_g1 retarget {str(g1_npz).replace(chr(92), '/')}"])
    motion.validate()
    opts = opts or RetargetOptions()
    res = retarget_g1_motion(motion, cal, opts)
    sem = res.semantic
    contacts = sem.contacts
    sidecar = {
        "source": motion.source, "source_file": motion.source_file, "source_license": LAFAN_LICENSE.as_dict(),
        "retarget_method": ("GMR(unitree_g1) + dropbear-wbc g1_to_dropbear v1 (G1-intermediate route; comparison only)"
                            + ("; foot_contact stage (dropbear_wbc.motion.foot_contact)" if opts.foot_contact is not None
                               else "")),
        "retarget_options": {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v) for k, v in asdict(opts).items()},
        "contact_hint": None if contacts is None else {"left": [bool(v) for v in contacts[:, 0]],
                                                       "right": [bool(v) for v in contacts[:, 1]],
                                                       "method": "G1 route (G1 FK sole height + speed)"},
        "notes": motion.notes + ["comparison output of tools/gmr_to_dropbear_csv.py --g1-npz"],
        "saturation": res.saturation, "metrics": res.metrics, "g1_mapping": sem.meta,
        "calibration": cal.provenance(), "gmr_frame_range": gmr_meta["frame_range"],
    }
    csv, _ = write_motion(out_root / motion.source, clip, fps=res.fps, root_pos=res.root_pos,
                          root_quat_wxyz=res.root_quat_wxyz, motor_q=res.motor_q, sidecar=sidecar,
                          semantic=(sem.pelvis_pos, sem.pelvis_quat_wxyz, sem.q, contacts))
    probs = validate_motion_files(csv)
    if probs:
        raise RuntimeError(f"{csv}: {probs}")
    q_used = cal.motor_to_semantic(res.motor_q)
    return {"csv": str(csv).replace("\\", "/"), "saturation": res.saturation, "q_used": q_used,
            "q_requested": sem.q_requested if sem.q_requested is not None else sem.q,
            "plant": plant_metrics(cfk, res.motor_q, res.root_pos, res.root_quat_wxyz, res.fps),
            "motor_q": res.motor_q}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gmr_npz", type=Path, help="tools/gmr_retarget.py output for robot 'dropbear'")
    ap.add_argument("--clip", required=True, help="output clip name")
    ap.add_argument("--source", default="gmr_lafan1")
    ap.add_argument("--out-root", type=Path, default=REPO / "data/motions")
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    ap.add_argument("--serial-xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--g1-npz", type=Path, default=None, help="GMR unitree_g1 retarget of the same clip (comparison)")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--foot-contact", action=argparse.BooleanOptionalAction, default=True,
                    help="run the foot-contact stage (library v4; default ON, --no-foot-contact = legacy route)")
    ap.add_argument("--fps-out", type=float, default=None,
                    help="resample the GMR clip first (default: 50 with the foot stage, else the GMR rate)")
    ap.add_argument("--max-motor-step", type=float, default=None,
                    help="project motors to this max step [rad/frame] (default: 0.18 with the foot stage; <= 0 = off)")
    args = ap.parse_args(argv)
    if args.foot_contact and args.fps_out is None:
        args.fps_out = 50.0
    if args.max_motor_step is None:
        args.max_motor_step = 0.18 if args.foot_contact else None
    elif args.max_motor_step <= 0:
        args.max_motor_step = None

    cal = load_calibration(args.calibration)
    if cal.is_mock:
        raise SystemExit("refusing a MOCK calibration")
    cfk = CalibrationFK(args.calibration)
    sprov = serial_provenance(args.serial_xml)
    if sprov["serial_model_calibration_sha256"] != cfk.sha256:
        raise SystemExit("serial model was built from another calibration: run tools/build_serial_mjcf.py first")
    serial_meta = json.loads(args.serial_xml.with_suffix(".json").read_text())
    clip = load_gmr_npz(args.gmr_npz)
    if clip.meta.get("xml") and Path(clip.meta["xml"]).resolve() != args.serial_xml.resolve():
        raise SystemExit(f"GMR used {clip.meta['xml']}, not {args.serial_xml}")
    from dropbear_wbc.motion.foot_contact import STAGE_VERSION, FootContactParams, plant_gate_metrics

    res = gmr_to_dropbear(clip, cal, cfk, serial_meta, foot_stage=FootContactParams() if args.foot_contact else None,
                          fps_out=args.fps_out, max_motor_step_rad=args.max_motor_step)
    res.metrics["plant_gates_predicted"] = plant_gate_metrics(cfk, res.motor_q, res.root_pos, res.root_quat_wxyz,
                                                              res.contacts, res.fps)
    commit = (REPO / "third_party/gmr_dropbear/gmr_commit.txt")
    sidecar = {
        "source": args.source, "source_file": clip.meta["bvh"], "source_format": f"bvh_{clip.meta['format']}",
        "source_fps": clip.fps, "source_license": LAFAN_LICENSE.as_dict(),
        "retarget_method": (f"{RETARGET_METHOD}: GMR IK (mink, patched: tools/setup_gmr.py) on the DERIVED serial model "
                            f"{sprov['serial_model']} with {clip.meta['ik_config']}; serial dof = dropbear-semantic-v1 -> "
                            "SemanticMap.semantic_to_motor; ground/contacts from the plant forward model"
                            + (f"; foot_contact={STAGE_VERSION} (stance feet pinned flat at world-locked poses with the "
                               "plant leg FK, dropbear_wbc.motion.foot_contact)" if args.foot_contact else "")
                            + (f"; motor_step_projection={args.max_motor_step} rad/frame" if args.max_motor_step else "")),
        "retarget_options": {"frame_range": clip.meta["frame_range"], "recenter_xy": True,
                             "ground": res.metrics["ground_method"], "gmr_npz": clip.path,
                             "foot_contact": asdict(FootContactParams()) if args.foot_contact else None,
                             "fps_out": args.fps_out, "max_motor_step_rad": args.max_motor_step},
        "contact_hint": {"left": [bool(v) for v in res.contacts[:, 0]], "right": [bool(v) for v in res.contacts[:, 1]],
                         "method": res.metrics["contact_agreement"].get("source")},
        "notes": ["LAFAN1 CC BY-NC-ND 4.0: internal R&D only, never redistribute this clip or derived policies' data.",
                  "Joint values come from IK on a DERIVED serial surrogate (fitted hip/knee geometry; see "
                  "docs/SERIAL_MODEL_AND_GMR.md); settle (tools/settle_motion.py) must run before tracking."],
        "saturation": res.saturation, "metrics": res.metrics,
        "serial_model": sprov, "gmr": {"commit": commit.read_text().strip() if commit.is_file() else None,
                                       "meta": clip.meta},
        "calibration": cal.provenance(), "pipeline_version": "gmr_serial-0.1",
    }
    csv, js = write_motion(args.out_root / args.source, args.clip, fps=res.fps, root_pos=res.root_pos,
                           root_quat_wxyz=res.root_quat_wxyz, motor_q=res.motor_q, sidecar=sidecar,
                           semantic=(res.pelvis_pos, res.pelvis_quat_wxyz, res.q_used, res.contacts))
    probs = validate_motion_files(csv)
    if probs:
        raise RuntimeError(f"{csv}: {probs}")
    sat = res.saturation
    ca = res.metrics["contact_agreement"]
    if ca.get("flag"):
        print(f"[gmr_to_dropbear_csv] WARNING {args.clip}: source contacts and plant feet disagree on "
              f"{ca['disagreement_frac']} of frames (> {ca['disagreement_limit']}); check the IK feet", flush=True)
    print(f"[gmr_to_dropbear_csv] {csv} ({len(res.motor_q)} frames @ {res.fps:g} Hz); saturated frames "
          f"{100 * sat['frames_with_any_saturation_frac']:.1f}% (worst {sat['worst_joint']} "
          f"{sat['per_joint'][sat['worst_joint']]['max_excess_deg']:.1f} deg); IK at limit {sat['ik_at_limit']}; "
          f"slip p95 {res.metrics['stance_foot_slip_p95_mps']}, pen {res.metrics['stance_penetration_max_m']}")
    report = {"clip": args.clip, "gmr_serial": {
        "csv": str(csv).replace("\\", "/"), "joint_range_used_deg": joint_range_summary(res.q_used),
        "saturation_frac": {n: v["frac_frames"] for n, v in sat["per_joint"].items() if v["frac_frames"] > 0},
        "frames_with_any_saturation_frac": sat["frames_with_any_saturation_frac"], "ik_at_limit": sat["ik_at_limit"],
        "plant": plant_metrics(cfk, res.motor_q, res.root_pos, res.root_quat_wxyz, res.fps),
        "plant_gates_predicted": res.metrics["plant_gates_predicted"],
        "gmr_ik_task_errors": res.metrics["gmr_ik_task_errors"]}}
    if args.g1_npz:
        from dropbear_wbc.motion.foot_contact import FootContactParams as _FCP

        g1_opts = RetargetOptions(output_fps=args.fps_out, foot_contact=_FCP() if args.foot_contact else None,
                                  max_motor_step_rad=args.max_motor_step)
        g1 = g1_route(args.g1_npz, args.clip, args.source, args.out_root, cal, cfk, clip.meta, g1_opts)
        s1 = g1["saturation"]
        report["g1_route"] = {
            "csv": g1["csv"], "joint_range_used_deg": joint_range_summary(g1["q_used"]),
            "joint_range_requested_deg": joint_range_summary(g1["q_requested"]),
            "saturation_frac": {n: v["frac_frames"] for n, v in s1["per_joint"].items() if v["frac_frames"] > 0},
            "saturation_max_excess_deg": {n: v["max_excess_deg"] for n, v in s1["per_joint"].items() if v["frac_frames"] > 0},
            "frames_with_any_saturation_frac": s1["frames_with_any_saturation_frac"], "plant": g1["plant"]}
        n = min(len(res.motor_q), len(g1["motor_q"]))
        d = np.degrees(np.abs(res.q_used[:n] - g1["q_used"][:n]))
        report["semantic_difference_deg"] = {nm: {"p50": float(np.median(d[:, i])), "p95": float(np.percentile(d[:, i], 95))}
                                             for i, nm in enumerate(SEMANTIC_NAMES)}
        print(f"[gmr_to_dropbear_csv] G1 route -> {g1['csv']}; saturated frames "
              f"{100 * s1['frames_with_any_saturation_frac']:.1f}% (GMR serial {100 * sat['frames_with_any_saturation_frac']:.1f}%)")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
