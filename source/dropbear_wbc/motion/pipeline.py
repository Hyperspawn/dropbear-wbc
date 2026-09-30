"""File-level retarget pipeline shared by ``tools/retarget_g1.py`` and ``tools/build_motion_library.py``."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .calibration_view import CalibrationView, load_calibration
from .g1_model import G1_JOINT_INDEX
from .g1_sources import G1Motion, load_g1_motion
from .g1_to_dropbear import RetargetOptions, retarget_g1_motion
from .motion_csv import validate_motion_files, write_motion
from .suitability import THRESHOLDS, suitability_flags

__all__ = ["RETARGET_METHOD", "retarget_file", "catalog_entry"]

RETARGET_METHOD = (
    "dropbear-wbc g1_to_dropbear v1: G1 -> dropbear-semantic-v1 joint mapping, waist fold into the body, "
    "hip-height root scaling, stance-foot lock, ground fix, calibration SemanticMap"
)
PIPELINE_VERSION = "motion_pipeline-0.1"

LEG_G1_JOINTS = [
    G1_JOINT_INDEX[n]
    for n in G1_JOINT_INDEX
    if any(k in n for k in ("hip", "knee", "ankle"))
]


def _method_string(cal: CalibrationView, opts: RetargetOptions) -> str:
    s = (f"{RETARGET_METHOD}; joint_mapping={opts.joint_mapping}; waist={opts.waist_mode}; "
         f"foot_lock_xy={opts.foot_lock_xy}; ground_fix={opts.ground_fix}")
    if opts.foot_contact is not None:
        from .foot_contact import STAGE_VERSION

        s += (f"; foot_contact={STAGE_VERSION} (stance feet pinned flat at world-locked poses with the plant leg FK, "
              "dropbear_wbc.motion.foot_contact)")
    if opts.max_motor_step_rad is not None:
        s += f"; motor_step_projection={opts.max_motor_step_rad} rad/frame"
    if cal.is_mock:
        s = "MOCK-CALIBRATION (NOT FOR TRAINING): " + s
    return s


def retarget_file(
    path: Path | str,
    out_root: Path | str,
    *,
    source: str | None = None,
    cal: CalibrationView | None = None,
    calibration_path: Path | str | None = None,
    allow_mock: bool = False,
    opts: RetargetOptions = RetargetOptions(),
) -> dict[str, Any]:
    """Load one G1 clip, retarget, write CSV/sidecar/semantic, validate. Returns a catalog entry."""
    t0 = time.perf_counter()
    cal = cal or load_calibration(calibration_path, allow_mock=allow_mock)
    motion = load_g1_motion(path, source)
    if opts.time_scale != 1.0:
        import dataclasses

        if not 0.25 <= opts.time_scale <= 4.0:
            raise ValueError(f"implausible time_scale {opts.time_scale}")
        motion = dataclasses.replace(
            motion, fps=motion.fps / opts.time_scale, clip=f"{motion.clip}_ts{opts.time_scale:g}",
            notes=motion.notes + [f"time-stretched x{opts.time_scale:g}: source {motion.fps:g} fps read as "
                                  f"{motion.fps / opts.time_scale:g} fps (RetargetOptions.time_scale)"])
    res = retarget_g1_motion(motion, cal, opts)
    sem = res.semantic
    if opts.foot_contact is not None:
        from .foot_contact import get_calibration_fk, plant_gate_metrics

        res.metrics["plant_gates_predicted"] = plant_gate_metrics(
            get_calibration_fk(cal.path), res.motor_q, res.root_pos, res.root_quat_wxyz, sem.contacts, res.fps)
    leg_range = np.ptp(motion.dof[:, LEG_G1_JOINTS], axis=0)
    sat_frac = res.saturation["frames_with_any_saturation_frac"]
    flags = suitability_flags(sem.meta, sem.contacts, res.fps, leg_range, sat_frac)
    notes = list(motion.notes)
    if cal.is_mock:
        notes.insert(0, "MOCK calibration: motor values are placeholders; do not train on this clip.")
    if res.saturation["usd_motor_limit_violation_deg"]:
        notes.append(f"motor values exceed USD limits: {res.saturation['usd_motor_limit_violation_deg']}")
    contacts = sem.contacts
    sidecar = {
        "source": motion.source,
        "source_file": motion.source_file,
        "source_format": motion.fmt,
        "source_fps": motion.fps,
        "source_license": motion.license.as_dict(),
        "retarget_method": _method_string(cal, opts),
        "retarget_options": {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v) for k, v in asdict(opts).items()},
        "contact_hint": None
        if contacts is None
        else {
            "left": [bool(v) for v in contacts[:, 0]],
            "right": [bool(v) for v in contacts[:, 1]],
            "method": ("G1 FK sole height + horizontal speed with hysteresis (contacts.HysteresisParams), cleaned by the "
                       "foot-contact stage (min stance / swing); these are the stance phases the stage pinned"
                       if opts.foot_contact is not None else
                       "G1 FK sole height + horizontal speed thresholds (source-robot scale)"),
        },
        "notes": notes,
        "suitability": flags,
        "suitability_thresholds": THRESHOLDS,
        "saturation": res.saturation,
        "metrics": res.metrics,
        "g1_mapping": sem.meta,
        "g1_dof_present": [bool(v) for v in motion.dof_present],
        "calibration": cal.provenance(),
        "pipeline_version": PIPELINE_VERSION,
    }
    out_dir = Path(out_root) / motion.source
    csv_path, json_path = write_motion(
        out_dir,
        motion.clip,
        fps=res.fps,
        root_pos=res.root_pos,
        root_quat_wxyz=res.root_quat_wxyz,
        motor_q=res.motor_q,
        sidecar=sidecar,
        semantic=(sem.pelvis_pos, sem.pelvis_quat_wxyz, sem.q, contacts),
    )
    problems = validate_motion_files(csv_path)
    if problems:
        raise RuntimeError(f"{csv_path} failed validation: {problems}")
    n_out = int(res.motor_q.shape[0])
    return catalog_entry(csv_path, sidecar | {"fps": res.fps, "clip": motion.clip, "num_frames": n_out,
                                              "duration_s": (n_out - 1) / float(res.fps)}, motion,
                         time.perf_counter() - t0)


def catalog_entry(csv_path: Path, side: dict[str, Any], motion: G1Motion | None, seconds: float) -> dict[str, Any]:
    sat = side["saturation"]
    per = sat["per_joint"]
    top = sorted(per.items(), key=lambda kv: kv[1]["frac_frames"], reverse=True)[:5]
    return {
        "clip": side["clip"],
        "source": side["source"],
        "csv": str(csv_path).replace("\\", "/"),
        "license": side["source_license"]["license"],
        "redistributable": side["source_license"]["redistributable"],
        "fps": side["fps"],
        "num_frames": side.get("num_frames", None if motion is None else motion.num_frames),
        "duration_s": round(side["duration_s"], 3) if "duration_s" in side else
        (None if motion is None else round(motion.duration, 3)),
        "suitability": side["suitability"],
        "saturation_summary": {
            "frames_with_any_saturation_frac": round(sat["frames_with_any_saturation_frac"], 4),
            "worst_joint": sat["worst_joint"],
            "worst_joint_max_excess_deg": round(per[sat["worst_joint"]]["max_excess_deg"], 2),
            "top_joints_by_frac": {k: round(v["frac_frames"], 4) for k, v in top if v["frac_frames"] > 0},
        },
        "mock_calibration": side["calibration"]["calibration_is_mock"],
        "retarget_seconds": round(seconds, 3),
    }
