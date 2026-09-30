"""GMR output on the DERIVED serial Dropbear model -> contract motion CSV (``retarget_method`` gmr-serial-v1).

Pipeline (see docs/SERIAL_MODEL_AND_GMR.md):

1. GMR (``tools/gmr_retarget.py``, robot ``dropbear``) solves IK on ``data/robot/dropbear_serial.xml``. Its
   joints are the 22 dropbear-semantic-v1 DOFs, so the GMR joint vector IS the requested semantic trajectory
   (mapped by joint NAME, not by index), and its free root is the semantic pelvis frame.
2. ``SemanticMap.semantic_to_motor`` (contract API) -> 22 motor angles; clipping (serial3 motor limits, the
   feasible ankle set) is reported per joint with the same statistics as the G1-intermediate route
   (:func:`dropbear_wbc.motion.g1_to_dropbear.saturation_stats`). GMR itself already respects the serial
   model's joint limits (bounding boxes); frames at a limit are reported separately (``ik_at_limit``).
3. Root: USD root body ``world`` = pelvis pose * inv(``pelvis_in_root``) (calibration ``rest_transforms``).
4. Contacts come from the HUMAN source feet (GMR's scaled ``LeftFootMod``/``RightFootMod`` task positions, height
   above their rolling ground + horizontal speed, :func:`dropbear_wbc.motion.contacts.detect_contacts`), because the
   IK robot feet are not trustworthy on frames where the knee limit (47 deg) and the pelvis position weight push
   them through the floor (review finding 2026-09-24: with IK-foot contacts, dance/run/jump had 49/86/81 % "no foot
   in contact" frames against 2/4/46 % in the source).
5. Ground: per frame, the plant contact-foot sole (calibration forward model
   :class:`~dropbear_wbc.kinematics.serial_model.CalibrationFK` / MeasuredFK at the motor angles, sole points from
   ``data/calibration/dropbear_foot_sole_hulls.json``) is put on z = 0 on the source-contact frames, interpolated
   over flight frames and Gaussian-smoothed (the settle tool's :func:`dropbear_wbc.settle.ground.ground_correction`),
   instead of one clip-wide 5th-percentile shift. xy is re-centred on the first frame. The sidecar records the
   agreement between the source contacts and the plant feet after grounding; > 10 % disagreement is flagged.
   Clips without human foot positions (old GMR npz) fall back to the previous clip-wide grounding + plant contacts.

Units: m, rad; quaternions wxyz internally (xyzw in the CSV per the contract).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dropbear_wbc.kinematics.calib_fit import seg_bodies
from dropbear_wbc.kinematics.serial_model import SIDES, CalibrationFK, resolve_path, sha256_file

from .calibration_view import CalibrationView
from .contacts import ContactParams, HysteresisParams, detect_contacts, detect_contacts_hysteresis, horizontal_speed
from .foot_contact import FootContactParams
from .g1_to_dropbear import saturation_stats, usd_motor_limits
from .names import MOTOR_NAMES, SEMANTIC_NAMES
from .rotations import matrix_to_quat, quat_continuous, quat_to_matrix

__all__ = ["RETARGET_METHOD", "GmrClip", "load_gmr_npz", "gmr_to_dropbear", "GmrResult", "plant_feet",
           "resample_gmr_clip", "human_foot_yaw", "human_foot_speeds", "HUMAN_HYSTERESIS"]

RETARGET_METHOD = "gmr-serial-v1"
SOLE_HULLS = Path(__file__).resolve().parents[3] / "data/calibration/dropbear_foot_sole_hulls.json"


@dataclass
class GmrClip:
    fps: float
    pelvis_pos: np.ndarray       # (T, 3)
    pelvis_quat_wxyz: np.ndarray  # (T, 4)
    q_sem: np.ndarray            # (T, 22) SEMANTIC_NAMES order (requested = GMR IK output)
    meta: dict[str, Any]
    task_bodies: list[str]
    task_pos_err: np.ndarray     # (T, K)
    task_rot_err: np.ndarray     # (T, K)
    path: str = ""
    human_pos: np.ndarray | None = None  # (T, K, 3) GMR-scaled human task-body positions (task_bodies order)
    human_feet_raw: np.ndarray | None = None  # (T, 2, 2, 3) raw human [ankle, toe] positions, left/right


def load_gmr_npz(path: str | Path) -> GmrClip:
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    if str(d["robot"]) != "dropbear":
        raise ValueError(f"{path}: robot {d['robot']!r} is not the Dropbear serial model")
    names = [str(n) for n in d["qpos_joint_names"]]
    missing = [n for n in SEMANTIC_NAMES if n not in names]
    if missing:
        raise ValueError(f"{path}: GMR joints lack semantic DOFs {missing}")
    qpos = np.asarray(d["qpos"], dtype=np.float64)
    q = qpos[:, 7:]
    q_sem = np.stack([q[:, names.index(n)] for n in SEMANTIC_NAMES], axis=1)
    quat = qpos[:, 3:7] / np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    return GmrClip(fps=float(d["fps"]), pelvis_pos=qpos[:, :3].copy(), pelvis_quat_wxyz=quat_continuous(quat),
                   q_sem=q_sem, meta=meta, task_bodies=[str(x) for x in d["task_bodies"]],
                   task_pos_err=np.asarray(d["task_pos_err"]), task_rot_err=np.asarray(d["task_rot_err"]),
                   path=str(path).replace("\\", "/"),
                   human_pos=np.asarray(d["human_pos"], dtype=np.float64) if "human_pos" in d.files else None,
                   human_feet_raw=(np.asarray(d["human_feet_raw"], dtype=np.float64)
                                   if "human_feet_raw" in d.files and np.isfinite(d["human_feet_raw"]).all() else None))


def motor_velocity_limits() -> np.ndarray:
    """Legacy per-motor velocity limits [rad/s] in motor-contract order (robots.dropbear_names; no zmq import)."""
    from dropbear_wbc.robots.dropbear_names import LEGACY_VELOCITY_LIMITS, motor_group

    return np.array([LEGACY_VELOCITY_LIMITS[motor_group(n)] for n in MOTOR_NAMES])


def plant_feet(cfk: CalibrationFK, motor_q: np.ndarray, root_pos: np.ndarray, root_rot: np.ndarray,
               soles: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Lowest sole z (T, 2) and sole centres (T, 2, 3) in the world, from the plant forward model."""
    soles = soles or json.loads(SOLE_HULLS.read_text())
    low = np.zeros((len(motor_q), 2))
    cen = np.zeros((len(motor_q), 2, 3))
    for k, s in enumerate(SIDES):
        foot = seg_bodies(s)["foot"]
        v = np.asarray(soles["feet"][foot]["vertices_b"], dtype=np.float64)
        r_f, p_f = cfk.fk.leg(s, motor_q)["foot"]
        rw = root_rot @ r_f
        pw = root_pos + np.einsum("tij,tj->ti", root_rot, p_f)
        vw = pw[:, None, :] + np.einsum("tij,vj->tvi", rw, v)
        low[:, k] = vw[..., 2].min(axis=1)
        bottom = vw[..., 2] < low[:, k, None] + 0.005
        cen[:, k] = (vw * bottom[..., None]).sum(1) / np.maximum(bottom.sum(1, keepdims=True), 1)
    return low, cen


@dataclass
class GmrResult:
    fps: float
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    motor_q: np.ndarray
    pelvis_pos: np.ndarray
    pelvis_quat_wxyz: np.ndarray
    q_used: np.ndarray
    contacts: np.ndarray
    saturation: dict[str, Any]
    metrics: dict[str, Any]
    notes: list[str] = field(default_factory=list)


HUMAN_FEET: tuple[str, str] = ("LeftFootMod", "RightFootMod")
HUMAN_CONTACT = ContactParams(height_thresh=0.03, speed_thresh=100.0)
"""Human-foot contact rule: a foot is in contact when its sole proxy (lower of toe height above the clip's toe ground and
ankle height above its standing height) is within 3 cm of the ground. HEIGHT ONLY: LAFAN stance feet slide and pivot
(near-ground foot speed p95 2.3-2.5 m/s on dance/run/jump), so a speed gate removed real stance frames (run 39-45 %
"flight"). With this rule the no-contact fraction equals the source's "both feet > 3 cm up" fraction (dance 2.7 %,
run 10.6 %, jump 48.9 %; logs/review_fixes/gmr_v2). Slip is measured separately (stance_foot_slip_p95_mps)."""


STAGE_PLANT_CONTACT = ContactParams(height_thresh=0.005, speed_thresh=100.0)
"""Plant-foot contact rule for the source-vs-plant agreement AFTER the foot-contact stage (soles within 5 mm)."""


HUMAN_HYSTERESIS = HysteresisParams(h_on=0.03, h_off=0.05, v_on=0.3, v_off=0.6, min_stance_s=0.10,
                                    min_swing_s=0.08)
"""Contact rule of the foot-contact stage (which world-locks every stance foot, so a stance must be a PLANTED foot):
touch down below 3 cm AND slower than 0.3 m/s, lift off above 5 cm OR faster than 0.6 m/s; min stance 0.1 s / swing
0.08 s. The speed is the slower of the foot's toe and ankle (:func:`human_foot_speeds`): a foot rolling or pivoting on
its toe or heel stays planted, and the lower-of-toe/ankle proxy point switching between the two (~0.17 m apart) does not
create speed spikes. A height-only rule (as :data:`HUMAN_CONTACT`) merges LAFAN's low-clearance walking steps: on
walk1 it gave 85-90 % stance with the source ankle drifting 0.37-0.62 m (p50) per "stance" phase, against 0.01-0.02 m
with this rule (``logs/foot_contact/dev/diag_speed2.out``)."""


def human_foot_contacts(clip: GmrClip, params: ContactParams = HUMAN_CONTACT,
                        hysteresis: HysteresisParams | None = None) -> np.ndarray | None:
    """(T, 2) contacts of the human source feet (toe + ankle; fallback: GMR FootMod ankles), or ``None``."""
    def detect(h, pos, speeds=None, **kw):
        if hysteresis is not None:
            return detect_contacts_hysteresis(h, pos, clip.fps, hysteresis, speeds=speeds, **kw)[0]
        return detect_contacts(h, pos, clip.fps, params, **kw)[0]

    if clip.human_feet_raw is not None:
        ank, toe = clip.human_feet_raw[:, :, 0], clip.human_feet_raw[:, :, 1]
        g = float(np.percentile(toe[..., 2], 1.0))
        rest = float(np.percentile(ank[..., 2], 5.0))
        toe_h, ank_h = toe[..., 2] - g, ank[..., 2] - rest
        proxy = np.minimum(toe_h, ank_h)
        pos = np.where((toe_h < ank_h)[..., None], toe, ank)
        return detect(proxy, pos, speeds=human_foot_speeds(clip), ground=0.0)
    if clip.human_pos is None or not all(f in clip.task_bodies for f in HUMAN_FEET):
        return None
    k = [clip.task_bodies.index(f) for f in HUMAN_FEET]
    pos = clip.human_pos[:, k, :]
    return detect(pos[..., 2], pos)


def human_foot_speeds(clip: GmrClip) -> np.ndarray | None:
    """(T, 2) horizontal speed [m/s] of each human foot = the slower of its toe and ankle (raw LAFAN), or ``None``."""
    if clip.human_feet_raw is None:
        return None
    return np.stack([np.minimum(horizontal_speed(clip.human_feet_raw[:, k, 1], clip.fps),
                                horizontal_speed(clip.human_feet_raw[:, k, 0], clip.fps)) for k in range(2)], axis=1)


def human_foot_yaw(clip: GmrClip) -> np.ndarray | None:
    """(T, 2) heading [rad] of the human feet (ankle -> toe, raw LAFAN) or ``None`` without raw feet."""
    if clip.human_feet_raw is None:
        return None
    d = clip.human_feet_raw[:, :, 1] - clip.human_feet_raw[:, :, 0]
    return np.arctan2(d[..., 1], d[..., 0])


def resample_gmr_clip(clip: GmrClip, fps: float) -> GmrClip:
    """Resample every per-frame array of a GMR clip to ``fps`` (lerp; slerp for the pelvis quaternion)."""
    from dataclasses import replace

    from .foot_contact import resample_semantic

    if abs(fps - clip.fps) < 1e-9:
        return clip
    extra = {"task_pos_err": clip.task_pos_err, "task_rot_err": clip.task_rot_err, "human_pos": clip.human_pos,
             "human_feet_raw": clip.human_feet_raw}
    pos, quat, q, ex, _ = resample_semantic(clip.fps, fps, clip.pelvis_pos, clip.pelvis_quat_wxyz, clip.q_sem, extra)
    meta = dict(clip.meta)
    meta["resampled"] = {"from_fps": clip.fps, "to_fps": fps, "method": "lerp / slerp (foot_contact.resample_semantic)"}
    return replace(clip, fps=float(fps), pelvis_pos=pos, pelvis_quat_wxyz=quat, q_sem=q, meta=meta,
                   task_pos_err=ex["task_pos_err"], task_rot_err=ex["task_rot_err"], human_pos=ex["human_pos"],
                   human_feet_raw=ex["human_feet_raw"])


def gmr_to_dropbear(clip: GmrClip, cal: CalibrationView, cfk: CalibrationFK, serial_meta: dict,
                    contact: ContactParams = ContactParams(), recenter_xy: bool = True,
                    ground_sigma_s: float = 0.1, foot_stage: FootContactParams | None = None,
                    fps_out: float | None = None, max_motor_step_rad: float | None = None) -> GmrResult:
    """Convert a GMR (dropbear serial) clip into motors + root + contacts + saturation/quality stats.

    ``fps_out`` resamples the clip first. ``foot_stage`` (foot_contact track 2026-09-24) runs
    :func:`dropbear_wbc.motion.foot_contact.foot_contact_stage` on the achieved semantic pose with the human-foot
    contacts (hysteresis) and headings; its legs, pelvis and contacts replace the IK ones (the serial-model feet are
    off by up to 47 mm, CONTRACTS 2.2) and the ground step below is skipped (the stage grounds on the plant feet)."""
    if fps_out is not None:
        clip = resample_gmr_clip(clip, fps_out)
    q_req = clip.q_sem
    stage_report = None
    stage_contacts = None
    if foot_stage is not None:
        from dataclasses import replace

        from .foot_contact import foot_contact_stage

        q_motor0 = cfk.smap.semantic_to_motor_sequence(q_req)[0]
        q_used0 = cfk.smap.motor_to_semantic(q_motor0)
        contacts0 = human_foot_contacts(clip, hysteresis=HUMAN_HYSTERESIS)
        if contacts0 is None:
            raise ValueError(f"{clip.path}: the foot-contact stage needs human foot positions in the GMR npz")
        pel0 = clip.pelvis_pos.copy()
        if recenter_xy:
            pel0[:, :2] -= pel0[0, :2]
        st = foot_contact_stage(clip.fps, pel0, clip.pelvis_quat_wxyz, q_used0, contacts0, cfk, cal.pelvis_T_root,
                                foot_stage, source_yaw=human_foot_yaw(clip))
        stage_report = st.report
        q_motor = st.motor_q
        _, rep = cfk.smap.semantic_to_motor(q_req, return_report=True)
        clip = replace(clip, pelvis_pos=st.pelvis_pos, pelvis_quat_wxyz=st.pelvis_quat_wxyz)
        recenter_xy = False
        stage_contacts = st.contacts
    else:
        q_motor, rep = cfk.smap.semantic_to_motor(q_req, return_report=True)
    step_proj = None
    q_motor_pre_proj = q_motor
    if max_motor_step_rad is not None:
        from .foot_contact import project_motor_steps

        q_motor, step_proj = project_motor_steps(q_motor, float(max_motor_step_rad))
        step_proj["columns"] = {MOTOR_NAMES[c]: v for c, v in step_proj["columns"].items()}
    q_used = cfk.smap.motor_to_semantic(q_motor)
    q_sat = q_used  # saturation = range clipping, measured before the rate projection (reported separately)
    if step_proj is not None and step_proj["columns"]:
        q_sat = cfk.smap.motor_to_semantic(q_motor_pre_proj)
        dq = np.degrees(np.abs(q_used - q_sat).max(axis=0))
        step_proj["semantic_change_deg"] = {n: float(dq[i]) for i, n in enumerate(SEMANTIC_NAMES) if dq[i] > 1e-6}
        step_proj["note"] = ("Euler-angle difference; ill-conditioned near the shoulder YXZ singularity (90 deg "
                             "abduction), where a small motor change reads as a large pitch/yaw change")
    if stage_report is not None:
        # saturation = clipping of the IK request by the valid range (SemanticMap report), independent of the stage's
        # intentional leg changes (metrics.foot_contact_stage.leg_semantic_change_deg), of the serial-continuity branch
        # choice (near the shoulder YXZ singularity the same orientation has other pitch/yaw Euler pairs: 9-14 deg
        # Euler difference at 0.02 deg orientation difference on dance2) and of the rate projection (reported above)
        q_sat = np.asarray(rep.used, dtype=np.float64).reshape(q_req.shape)
    sat = saturation_stats(q_req, q_sat)
    frames_sat = float(np.mean(np.abs(q_req - q_sat).max(axis=1) > 1e-4))
    # frames where GMR's IK sits at the serial model's joint limit (the analogue of clipping, done by the IK)
    lim = np.array([serial_meta["joints"][n]["range"] for n in SEMANTIC_NAMES])
    at_lim = (np.abs(q_req - lim[:, 0]) < np.radians(0.5)) | (np.abs(q_req - lim[:, 1]) < np.radians(0.5))
    usd_lim = usd_motor_limits()
    lim_viol = {}
    for i, n in enumerate(MOTOR_NAMES):
        lo, hi = usd_lim[n]
        over = np.maximum(q_motor[:, i] - hi, 0.0) + np.maximum(lo - q_motor[:, i], 0.0)
        if over.max() > 1e-6:
            lim_viol[n] = float(np.degrees(over.max()))
    # root (USD 'world' body) from the pelvis
    r_pel = quat_to_matrix(clip.pelvis_quat_wxyz)
    p_tr = cal.pelvis_T_root
    pelvis_pos = clip.pelvis_pos.copy()
    if recenter_xy:
        pelvis_pos[:, :2] -= pelvis_pos[0, :2]
    root_pos = pelvis_pos + np.einsum("tij,j->ti", r_pel, p_tr[:3, 3])
    root_rot = r_pel @ p_tr[:3, :3]
    # contacts from the human source feet; ground per frame from the plant contact-foot soles
    soles = json.loads(SOLE_HULLS.read_text())
    low, cen = plant_feet(cfk, q_motor, root_pos, root_rot, soles)
    human = human_foot_contacts(clip) if stage_report is None else stage_contacts
    contact_info: dict[str, Any] = {}
    if stage_report is not None:
        contacts = human
        dz = np.zeros(len(low))
        ground_desc = ("foot-contact stage (dropbear_wbc.motion.foot_contact): stance soles pinned to z=0 with the plant "
                       "forward model; no extra ground shift")
    elif human is not None:
        from dropbear_wbc.settle.ground import ground_correction

        contacts = human
        gfix = ground_correction({"left": low[:, 0], "right": low[:, 1]}, clip.fps, contacts, sigma_s=ground_sigma_s)
        dz = gfix.dz
        ground_desc = (f"per-frame: source-contact plant sole -> z=0, interpolated over flight, Gaussian sigma "
                       f"{ground_sigma_s} s (dropbear_wbc.settle.ground.ground_correction)")
    else:
        dz = np.full(len(low), -float(np.percentile(low.min(axis=1), 5.0)))
        contacts = None
        ground_desc = "clip-wide: 5th percentile lowest plant sole -> z=0 (no human foot positions in the GMR npz)"
    root_pos[:, 2] += dz
    pelvis_pos[:, 2] += dz
    low += dz[:, None]
    cen[..., 2] += dz[:, None]
    if contacts is None:
        plant_contacts, _ = detect_contacts(low, cen, clip.fps, contact, ground=0.0)
        contacts = plant_contacts
        contact_info["source"] = "plant-FK sole height + horizontal speed (fallback: no human foot positions)"
    else:
        # same (height-only) rule as the source feet, so the comparison measures geometry, not the speed gate. After the
        # foot-contact stage the plant stance soles are ON the ground and swing soles clear it by only ~1 cm, so the
        # plant side uses a 5 mm band there (a 3 cm band would count every low swing frame as contact).
        plant_rule = HUMAN_CONTACT if stage_report is None else STAGE_PLANT_CONTACT
        plant_contacts, _ = detect_contacts(low, cen, clip.fps, plant_rule, ground=0.0)
        dis = contacts != plant_contacts
        contact_info = {
            "source": ("human source feet: toe/ankle sole proxy, hysteresis 3 cm on / 5 cm off, min stance 0.1 s "
                       "(foot-contact stage phases)" if stage_report is not None else
                       "human source feet: toe/ankle sole proxy within 3 cm of the ground (raw LAFAN Foot/Toe)"
                       if clip.human_feet_raw is not None else
                       "human source feet (GMR LeftFootMod/RightFootMod height above rolling ground)"),
            "no_foot_in_contact_frac_source": float((~contacts.any(axis=1)).mean()),
            "no_foot_in_contact_frac_plant": float((~plant_contacts.any(axis=1)).mean()),
            "disagreement_frac": {"left": float(dis[:, 0].mean()), "right": float(dis[:, 1].mean()),
                                  "any": float(dis.any(axis=1).mean())},
            "disagreement_limit": 0.10,
            "plant_rule": {"height_thresh_m": plant_rule.height_thresh, "speed": "not used"},
        }
        contact_info["flag"] = bool(max(contact_info["disagreement_frac"]["left"],
                                        contact_info["disagreement_frac"]["right"]) > 0.10)
    slip, pen, flt = [], [], []
    for s in range(2):
        c = contacts[:, s]
        if c.any():
            v = horizontal_speed(cen[:, s], clip.fps)
            slip.append(float(np.percentile(v[c], 95)))
            pen.append(float(max(0.0, -low[c, s].min())))
            flt.append(float(np.percentile(low[c, s], 95)))
    ik = {b: {"pos_err_p50_m": float(np.median(clip.task_pos_err[:, k])),
              "pos_err_p95_m": float(np.percentile(clip.task_pos_err[:, k], 95)),
              "rot_err_p50_deg": float(np.degrees(np.median(clip.task_rot_err[:, k]))),
              "rot_err_p95_deg": float(np.degrees(np.percentile(clip.task_rot_err[:, k], 95)))}
          for k, b in enumerate(clip.task_bodies)}
    dq_sem = np.abs(np.diff(q_used, axis=0)) * clip.fps
    dq_mot = np.abs(np.diff(q_motor, axis=0)) * clip.fps
    vlim = motor_velocity_limits()
    speed = {
        "semantic_max_rad_s": {n: float(dq_sem[:, i].max()) for i, n in enumerate(SEMANTIC_NAMES)},
        "frames_semantic_over_10rad_s": int((dq_sem.max(axis=1) > 10.0).sum()),
        "motor_max_rad_s": {n: float(dq_mot[:, i].max()) for i, n in enumerate(MOTOR_NAMES)},
        "frames_motor_over_velocity_limit": int((dq_mot > vlim).any(axis=1).sum()),
        "note": "finite differences at the clip rate; spikes come from the shoulder YXZ singularity near 90 deg "
                "abduction (pitch/yaw swing) and from fast source motion; settle/tracking must absorb them",
    }
    metrics = {
        "joint_speed": speed,
        "stance_foot_slip_p95_mps": max(slip) if slip else None,
        "stance_penetration_max_m": max(pen) if pen else None,
        "stance_float_p95_m": max(flt) if flt else None,
        "min_sole_height_m": float(low.min()),
        "ground_shift_m": {"min": float(dz.min()), "max": float(dz.max()), "median": float(np.median(dz))},
        "ground_method": ground_desc,
        "contact_agreement": contact_info,
        "contact_frac": {"left": float(contacts[:, 0].mean()), "right": float(contacts[:, 1].mean())},
        "feet_model": "plant forward model (MeasuredFK from the calibration sweep) + sole hulls",
        "gmr_ik_task_errors": ik,
        "root_path_length_m": float(np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1).sum()),
    }
    if stage_report is not None:
        metrics["foot_contact_stage"] = stage_report
    saturation = {
        "per_joint": sat,
        "frames_with_any_saturation_frac": frames_sat,
        "worst_joint": max(sat, key=lambda k: sat[k]["max_excess_deg"]),
        "usd_motor_limit_violation_deg": lim_viol,
        "ik_at_limit": {n: float(at_lim[:, i].mean()) for i, n in enumerate(SEMANTIC_NAMES) if at_lim[:, i].any()},
        "motor_step_projection": step_proj,
        "note": "requested = GMR IK output (already inside the serial model's joint limits); used = what the motors "
                "realise after SemanticMap clipping (serial3 motor limits, feasible ankle set)",
    }
    return GmrResult(fps=clip.fps, root_pos=root_pos, root_quat_wxyz=quat_continuous(matrix_to_quat(root_rot)),
                     motor_q=q_motor, pelvis_pos=pelvis_pos, pelvis_quat_wxyz=clip.pelvis_quat_wxyz, q_used=q_used,
                     contacts=contacts, saturation=saturation, metrics=metrics)


def joint_range_summary(q: np.ndarray) -> dict[str, list[float]]:
    return {n: [float(np.degrees(q[:, i].min())), float(np.degrees(q[:, i].max()))] for i, n in enumerate(SEMANTIC_NAMES)}


def serial_provenance(xml: str | Path) -> dict[str, Any]:
    xml = resolve_path(xml)
    meta = json.loads(xml.with_suffix(".json").read_text())
    return {"serial_model": str(xml).replace("\\", "/"), "serial_model_xml_sha256": sha256_file(xml),
            "serial_model_calibration_sha256": meta["inputs"]["calibration_sha256"], "status": meta["status"]}
