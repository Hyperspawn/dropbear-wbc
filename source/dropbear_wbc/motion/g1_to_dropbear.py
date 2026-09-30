"""Map a :class:`G1Motion` to Dropbear: semantic trajectory -> motor trajectory + root pose.

Pipeline (all vectorised over frames)
-------------------------------------
1. **G1 FK** (``g1_model``) gives pelvis/torso orientations, sole points and hip centre.
2. **Joint mapping** into ``dropbear-semantic-v1`` (CONTRACTS.md s2). G1 wrist pitch/yaw are dropped.

   * ``anatomical`` (default): semantic angles are computed from G1 *segment frames* with the same
     definitions the calibration uses for Dropbear (``kinematics/calib_fit.py``): hip / shoulder =
     intrinsic YXZ Euler of the zero-referenced thigh / upper-arm frame relative to the (Dropbear) body;
     knee, ankle pitch/roll and wrist roll equal the G1 joint values (exact for these G1 links); elbow =
     G1 elbow + ``pi/2 - e_axes`` where ``e_axes`` is the G1 elbow value at which the shoulder-yaw and
     wrist-roll axes are parallel (the calibration's notion of a straight arm) -- exactly pi/2 for G1,
     so the elbow maps by value. (By joint *centres* G1 is straight at ~1.385 rad; reported only.)
     This removes the G1 link-frame quirks that a by-name copy carries over (hip roll axis tilted
     10 deg, shoulder pitch axis tilted 16 deg; logs/motion_pipeline/probe_g1_anatomical_offsets.log).
     Euler angles are fitted by seeded Gauss-Newton from the by-name values, so they stay on the G1
     branch. Dropbear's shoulder (untilted pitch-roll-yaw) is singular with the arm horizontal to the
     side, where G1's tilted shoulder is not; there a conditioning-dependent prior keeps pitch/yaw near
     the G1 values and accepts a small (mostly twist) orientation residual instead of whirling angles.
   * ``by_name``: copy the G1 joint value for every semantic name (CONTRACTS.md wording). The sidecar
     reports ``joint_mapping_delta_vs_by_name_deg`` so the two can be compared per clip.
3. **Waist** (G1 ``waist_yaw/roll/pitch``; Dropbear has no waist, pelvis and chest are one body):

   * ``fold`` (default): the Dropbear body takes the G1 *torso* orientation
     ``R_body = R_pelvis @ R_waist`` about the hip centre, and the hip angles are recomputed so every
     thigh keeps its G1 world orientation (``R_thigh' = R_waist^T R_thigh`` decomposed in the exact
     G1 hip chain ``Ry(p) Ry(-10deg) Rx(r) Rz(y)``). Arms are attached to the torso, so the upper body
     orientation is preserved exactly and the legs are preserved up to joint-range saturation.
   * ``drop``: waist angles are discarded (body = G1 pelvis orientation); upper body is then off by the
     full waist rotation. Reported for comparison (``upper_body_error_deg``).
   * ``auto`` (default): per frame, fold the largest fraction ``alpha`` in {0, 0.1, .., 1} of the waist
     rotation (slerp from identity) that adds no hip saturation (> 0.5 deg summed over the 6 hip DoFs)
     beyond what the raw G1 hip angles already have; ``alpha`` is smoothed with a moving min followed by
     a moving average (0.2 s), which never exceeds the feasible value. Torso-twist clips (throws,
     dances) exceed Dropbear's +-30 deg hip yaw under a full fold, so a full fold would trade upper-body
     accuracy for twisted stance feet; ``auto`` keeps the legs feasible and the upper body as close as
     the hips allow. Upper-body error = ``(1 - alpha) * |waist rotation|``.
4. **Root**: the semantic pelvis point (hip centre) is scaled by ``s = H_dropbear / H_g1`` (standing
   hip heights, sole to hip-pitch axis), horizontally re-centred so the clip starts at x = y = 0.
   Optional ground fix: a smooth per-frame height offset puts the lowest contact sole of the
   *Dropbear semantic leg model* on z = 0 in contact frames (interpolated through flight phases).
   The articulation root ``world`` pose is then ``T_world_pelvis @ pelvis_T_root`` (calibration).
5. **Contacts** from G1 sole height + speed (``contacts.detect_contacts``). Semantic angles are then
   clipped to the calibration's valid range (round trip through the SemanticMap) and the root is
   corrected on the *achieved* pose: ``foot_lock_xy`` shifts the root horizontally so stance feet of the
   Dropbear semantic leg model stay put (angle copying onto different leg proportions otherwise makes
   them slide), then ``ground_fix`` sets the height.
6. **semantic -> motor** via the calibration ``SemanticMap`` (contract API), with per-joint saturation
   statistics measured by round trip (``motor_to_semantic(semantic_to_motor(q))`` vs ``q``) and a
   motor-limit check against the USD limits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .calibration_view import CalibrationView
from .contacts import ContactParams, HysteresisParams, detect_contacts, detect_contacts_hysteresis, horizontal_speed
from .foot_contact import FootContactParams
from .g1_model import G1_JOINT_INDEX, G1Kinematics, load_g1_kinematics
from .g1_sources import G1Motion
from .names import MOTOR_NAMES, SEMANTIC_INDEX, SEMANTIC_NAMES
from .rotations import (
    fit_euler_intrinsic,
    matrix_to_quat,
    quat_continuous,
    quat_slerp,
    quat_to_matrix,
    rot_x,
    rot_y,
    rot_z,
    rotation_angle,
    yaw_of_matrix,
)
from .semantic_skeleton import LegGeometry, leg_fk

__all__ = [
    "RetargetOptions",
    "SemanticTrajectory",
    "DropbearMotion",
    "g1_to_semantic",
    "semantic_to_dropbear",
    "retarget_g1_motion",
    "saturation_stats",
    "usd_motor_limits",
    "apply_foot_stage",
]

REPO = Path(__file__).resolve().parents[3]
USD_TREE_JSON = REPO / "docs/usd_tree_45586414.json"

#: semantic name -> G1 joint name (by-name mapping, CONTRACTS.md s2)
SEMANTIC_FROM_G1: dict[str, str] = {n: f"{n}_joint" for n in SEMANTIC_NAMES}
DROPPED_G1_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                     "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                     "right_wrist_pitch_joint", "right_wrist_yaw_joint")
#: G1 MJCF: hip_roll_link has a fixed -10 deg pitch (quat 0.996179, 0, -0.0873386, 0).
G1_HIP_ROLL_LINK_PITCH = 2.0 * np.arcsin(-0.0873386)
#: Near the YXZ singularity (roll ~ +-90 deg, e.g. arm horizontal sideways in jumping jacks) keep
#: pitch/yaw near the G1 values instead of whirling; residual reported as euler_fit_max_residual_rad.
SINGULAR_PRIOR = 1e-2


@dataclass(frozen=True)
class RetargetOptions:
    waist_mode: str = "auto"  # 'auto' | 'fold' | 'drop'
    joint_mapping: str = "anatomical"  # 'anatomical' | 'by_name'
    ground_fix: bool = True
    ground_fix_smooth_s: float = 0.15
    recenter_xy: bool = True
    foot_lock_xy: bool = True
    contact: ContactParams = ContactParams()
    output_fps: float | None = None  # None = keep source rate
    # serial3 (hips, shoulders) motor solutions continuous in time (review fix 2026-09-24): each frame's inverse is seeded
    # with the previous frame's motors (removes Euler branch flips: Take_102 max motor step 3.15 -> 0.35 rad, zero
    # orientation error). max_serial_speed_rad_s additionally rate-limits the serial motors (e.g. 8.0: the kimodo wave's
    # passage through the YXZ singularity at +-90 deg abduction lags <= 12.6 deg instead of jumping 1.8 rad), but on
    # genuinely fast clips the lag grows to 60-97 deg (logs/review_fixes/compare_library_motor_dynamics_rate8.log), so
    # it is OFF by default; clips that still exceed the motor cap are rejected by the settle/validator dynamics gate.
    serial_continuity: bool = True
    max_serial_speed_rad_s: float | None = None
    # foot-contact stage (foot_contact track 2026-09-24, dropbear_wbc.motion.foot_contact): None = off (legacy). When
    # set, stance feet are pinned flat on the ground at world-locked poses with the PLANT leg FK (MeasuredFK), the
    # pelvis height/position is re-solved and the contacts come from ``contact_hysteresis`` (default HysteresisParams).
    foot_contact: FootContactParams | None = None
    contact_hysteresis: HysteresisParams | None = None
    # minimal-change projection of ALL motor trajectories to a max frame-to-frame step (foot_contact track): None = off.
    # 0.18 rad at 50 Hz = 9 rad/s keeps the reference under the validator's 10 rad/s / 0.3 rad-per-frame gate; on
    # Take_102 it changes 5 arm motors on <= 57 frames by <= 0.34 rad (foot_contact.project_motor_steps).
    max_motor_step_rad: float | None = None
    # time stretch of the source clip (1 = as recorded). The hip-height scaling makes G1 motion 1.42x larger in space
    # but keeps its timing, which is dynamically too fast for the bigger body: the Kimodo walk becomes 1.57 m/s and the
    # deployable trackers settle at 82 % of it (docs/ISSUES.md #27). Froude similarity (same v^2 / (g L)) says time
    # should stretch by sqrt(length scale) = sqrt(1.42) ~ 1.19; the policies' own speed matches that (1.29 m/s).
    time_scale: float = 1.0


@dataclass
class SemanticTrajectory:
    """Dropbear semantic-space clip.

    ``pelvis_pos`` (T,3) [m] and ``pelvis_quat_wxyz`` (T,4): semantic pelvis frame in world (ground z=0).
    ``q`` (T,22) [rad] in :data:`SEMANTIC_NAMES` order, *achieved* (clipped to the calibration's valid
    range, consistent with the motor CSV); ``q_requested`` is the unclipped mapping of the source clip.
    ``contacts`` (T,2) bool [left, right].
    """

    fps: float
    pelvis_pos: np.ndarray
    pelvis_quat_wxyz: np.ndarray
    q: np.ndarray
    contacts: np.ndarray | None
    meta: dict[str, Any] = field(default_factory=dict)
    q_requested: np.ndarray | None = None  # before clipping to the calibration range (None = same as q)
    source_foot_yaw: np.ndarray | None = None  # (T, 2) heading of the source feet [rad] (G1 ankle_roll_link x axis)
    leg_motors: np.ndarray | None = None  # (T, 22) motors whose leg columns realise q exactly (foot stage), else None
    # (T, 22) achieved pose BEFORE the foot stage (foot stage only): the saturation statistics compare the request with
    # this for the legs, so they keep meaning "clipped by the valid range"; the stage's intentional leg changes are in
    # meta.foot_contact_stage.leg_semantic_change_deg
    q_pre_stage: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.q.shape[0])


@dataclass
class DropbearMotion:
    """Contract ``dropbear-motion-csv-v1`` content: root ``world`` pose + 22 motor angles (T,22) [rad]."""

    fps: float
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    motor_q: np.ndarray
    semantic: SemanticTrajectory
    saturation: dict[str, Any]
    metrics: dict[str, Any]


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


def usd_motor_limits(path: Path = USD_TREE_JSON) -> dict[str, tuple[float, float]]:
    """Motor joint limits [rad] from the USD tree dump (authored in degrees)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for j in d["joints"]:
        if j["name"] in MOTOR_NAMES:
            out[j["name"]] = (np.deg2rad(j["lo"]), np.deg2rad(j["hi"]))
    missing = set(MOTOR_NAMES) - set(out)
    if missing:
        raise KeyError(f"USD tree lacks motors {sorted(missing)}")
    return out


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """Angles wrapped to (-pi, pi]."""
    w = np.mod(np.asarray(a, dtype=np.float64) + np.pi, 2.0 * np.pi) - np.pi
    return np.where(w == -np.pi, np.pi, w)


def _moving_min(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = win | 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(xp, k).min(axis=1)


def _moving_max(x: np.ndarray, win: int) -> np.ndarray:
    return -_moving_min(-x, win)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = win | 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


# --------------------------------------------------------------------------------------------------
# G1 -> semantic
# --------------------------------------------------------------------------------------------------


def g1_to_semantic(
    motion: G1Motion,
    cal: CalibrationView,
    opts: RetargetOptions = RetargetOptions(),
    g1: G1Kinematics | None = None,
) -> SemanticTrajectory:
    """Joint + root mapping into the Dropbear semantic space (see module docstring)."""
    if opts.waist_mode not in ("auto", "fold", "drop"):
        raise ValueError(f"waist_mode must be 'auto', 'fold' or 'drop', got {opts.waist_mode!r}")
    if opts.joint_mapping not in ("anatomical", "by_name"):
        raise ValueError(f"joint_mapping must be 'anatomical' or 'by_name', got {opts.joint_mapping!r}")
    if opts.output_fps is not None:
        motion = motion.resample(opts.output_fps)
    g1 = g1 or load_g1_kinematics()
    fk = g1.forward(motion.root_pos, motion.root_quat_wxyz, motion.dof)
    t = motion.num_frames

    # ---- joints by name -----------------------------------------------------------------------------
    q = np.zeros((t, 22))
    for name, g1_name in SEMANTIC_FROM_G1.items():
        q[:, SEMANTIC_INDEX[name]] = motion.dof[:, G1_JOINT_INDEX[g1_name]]

    r_pel = fk.body_rot("pelvis")
    r_torso = fk.body_rot("torso_link")
    r_waist = np.swapaxes(r_pel, -1, -2) @ r_torso  # pelvis_R_torso
    waist_angle = rotation_angle(r_waist)

    hip_idx = {
        side: [SEMANTIC_INDEX[f"{side}_hip_{a}"] for a in ("pitch", "roll", "yaw")] for side in ("left", "right")
    }
    sh_idx = {
        side: [SEMANTIC_INDEX[f"{side}_shoulder_{a}"] for a in ("pitch", "roll", "yaw")] for side in ("left", "right")
    }
    q_byname = q.copy()
    anatomical = opts.joint_mapping == "anatomical"
    # Zero-referenced segment frames in world (identity w.r.t. pelvis / torso at the G1 zero pose):
    #   thigh = hip_yaw_link @ Ry(+10 deg) (undo the fixed hip_roll_link tilt), upper arm = shoulder_yaw_link.
    tilt_fix = rot_y(np.full(t, -G1_HIP_ROLL_LINK_PITCH))
    thigh_w = {s: fk.body_rot(f"{s}_hip_yaw_link") @ tilt_fix for s in ("left", "right")}
    upper_w = {s: fk.body_rot(f"{s}_shoulder_yaw_link") for s in ("left", "right")}
    if anatomical:
        for side in ("left", "right"):
            q[:, SEMANTIC_INDEX[f"{side}_elbow"]] += g1.elbow_semantic_offset
    q_waist = matrix_to_quat(r_waist)
    identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (t, 1))
    fit_residual = [0.0]

    def fold(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fold fraction ``alpha`` (T,) of the waist rotation into the body; re-express the limbs."""
        r_part = quat_to_matrix(quat_slerp(identity, q_waist, alpha))
        r_body = r_pel @ r_part
        qa = q.copy()
        rbt = np.swapaxes(r_body, -1, -2)
        for side in ("left", "right"):
            hi = hip_idx[side]
            if anatomical:
                # Segment-frame semantics (calib_fit definition): YXZ Euler of the body-relative thigh frame.
                ang, res = fit_euler_intrinsic("YXZ", rbt @ thigh_w[side], q_byname[:, hi],
                                               singular_prior=SINGULAR_PRIOR)
            elif not np.any(alpha > 0.0):
                continue  # by_name + nothing folded: the G1 values are used unchanged
            else:
                # Exact G1 hip chain Ry(p) Ry(tilt) Rx(r) Rz(y), re-expressed relative to the folded body.
                seed = q_byname[:, hi] + np.array([G1_HIP_ROLL_LINK_PITCH, 0.0, 0.0])
                ang, res = fit_euler_intrinsic("YXZ", rbt @ thigh_w[side] @ np.swapaxes(tilt_fix, -1, -2), seed,
                                               singular_prior=SINGULAR_PRIOR)
                ang[:, 0] -= G1_HIP_ROLL_LINK_PITCH
            # the fit stays on the seed's 2*pi wrap; joints live in (-pi, pi] (review fix 2026-09-24: a G1 yaw seed of
            # 114 deg once produced a requested 290 deg that SemanticMap clipped to 180 instead of -70)
            qa[:, hi] = wrap_to_pi(ang)
            fit_residual[0] = max(fit_residual[0], float(res.max()))
            if anatomical:
                # Arms keep their G1 world orientation even when the waist is only partly folded.
                si = sh_idx[side]
                ang, res = fit_euler_intrinsic("YXZ", rbt @ upper_w[side], q_byname[:, si],
                                               singular_prior=SINGULAR_PRIOR)
                qa[:, si] = wrap_to_pi(ang)
                fit_residual[0] = max(fit_residual[0], float(res.max()))
        return qa, r_body

    comp = hip_idx["left"] + hip_idx["right"] + (sh_idx["left"] + sh_idx["right"] if anatomical else [])

    def comp_excess(qa: np.ndarray) -> np.ndarray:
        q_m, _ = cal.semantic_to_motor(qa)
        back = cal.motor_to_semantic(q_m)
        return np.abs(qa[:, comp] - back[:, comp]).sum(axis=1)

    auto_info: dict[str, Any] = {}
    if opts.waist_mode == "fold":
        alpha = np.ones(t)
    elif opts.waist_mode == "drop":
        alpha = np.zeros(t)
    else:  # auto: largest fold fraction that adds no saturation to the compensating joints
        grid = np.linspace(0.0, 1.0, 11)
        exc = np.stack([comp_excess(fold(np.full(t, a))[0]) for a in grid])  # (11, T)
        feasible = exc <= exc[0] + np.deg2rad(0.5)
        alpha_star = np.array([grid[np.flatnonzero(feasible[:, i])].max() for i in range(t)])
        win = max(1, int(round(0.2 * motion.fps)))
        alpha = _moving_average(_moving_min(alpha_star, win), win)
        auto_info = {
            "alpha_mean": float(alpha.mean()),
            "alpha_min": float(alpha.min()),
            "fold_mode_comp_excess_deg_max": float(np.rad2deg(exc[-1].max())),
            "drop_mode_comp_excess_deg_max": float(np.rad2deg(exc[0].max())),
        }
    q, r_body = fold(alpha)
    mapping_delta = {
        n: float(np.rad2deg(np.abs(q[:, i] - q_byname[:, i]).max()))
        for n, i in SEMANTIC_INDEX.items()
        if np.abs(q[:, i] - q_byname[:, i]).max() > 1e-9
    }
    chest_err = (1.0 - alpha) * waist_angle  # Dropbear chest vs G1 torso orientation
    # by_name: the arms ride on the chest, so they inherit its error; anatomical: arms are re-expressed.
    upper_err = chest_err

    # ---- contacts at G1 scale ------------------------------------------------------------------------
    sole_h = np.stack([fk.sole_height("left"), fk.sole_height("right")], axis=1)
    foot_c = np.stack([fk.foot_center("left"), fk.foot_center("right")], axis=1)
    hyst = opts.contact_hysteresis or (HysteresisParams() if opts.foot_contact is not None else None)
    if hyst is not None:
        contacts, ground, _ = detect_contacts_hysteresis(sole_h, foot_c, motion.fps, hyst)
    else:
        contacts, ground = detect_contacts(sole_h, foot_c, motion.fps, opts.contact)
    foot_yaw = np.stack([yaw_of_matrix(fk.body_rot(f"{s}_ankle_roll_link")) for s in ("left", "right")], axis=1)

    # ---- root (semantic pelvis point = hip centre) ---------------------------------------------------
    scale = cal.standing_hip_height / g1.standing_hip_height
    hip_c = motion.root_pos + np.einsum("tij,j->ti", r_pel, g1.hip_center_in_pelvis)
    pelvis_pos = scale * hip_c
    pelvis_pos[:, 2] = scale * (hip_c[:, 2] - ground)
    if opts.recenter_xy:
        pelvis_pos[:, :2] -= pelvis_pos[0, :2]

    # ---- clip to the calibration's valid range: everything below uses the *achieved* pose ------------
    q_motor, _ = cal.semantic_to_motor(q)
    q_ach = cal.motor_to_semantic(q_motor)

    geom = LegGeometry.from_calibration(cal)
    lock_corr = np.zeros((t, 2))
    if opts.foot_lock_xy:
        feet = leg_fk(pelvis_pos, r_body, q_ach, geom).foot_center[..., :2]  # (T, 2, 2)
        lock_corr = _foot_lock_correction(feet, contacts)
        pelvis_pos[:, :2] += lock_corr
    ground_offset = np.zeros(t)
    if opts.ground_fix:
        sole_db = leg_fk(pelvis_pos, r_body, q_ach, geom).sole_height  # (T, 2)
        any_c = contacts.any(axis=1)
        if any_c.any():
            lowest = np.where(contacts, sole_db, np.inf).min(axis=1)
            idx = np.flatnonzero(any_c)
            raw = np.interp(np.arange(t), idx, -lowest[idx])
            # dilate before smoothing: avg(movmax(raw)) >= raw, so smoothing never re-introduces penetration
            win = int(round(opts.ground_fix_smooth_s * motion.fps))
            ground_offset = _moving_average(_moving_max(raw, win), win)
            pelvis_pos[:, 2] += ground_offset

    meta = {
        "g1_standing_hip_height_m": g1.standing_hip_height,
        "dropbear_standing_hip_height_m": cal.standing_hip_height,
        "scale": scale,
        "waist_mode": opts.waist_mode,
        "joint_mapping": opts.joint_mapping,
        "joint_mapping_delta_vs_by_name_deg": mapping_delta,
        "euler_fit_max_residual_rad": fit_residual[0],
        "g1_elbow_semantic_offset_rad": g1.elbow_semantic_offset if anatomical else 0.0,
        "g1_elbow_straight_by_joint_centres_rad": g1.elbow_straight_value,
        "waist_fold_fraction": {"mean": float(alpha.mean()), "min": float(alpha.min()), "max": float(alpha.max())},
        "waist_auto": auto_info,
        "g1_ground_z_m": ground,
        "ground_offset_m": {"max_abs": float(np.abs(ground_offset).max()), "mean": float(ground_offset.mean())},
        "foot_lock_xy_correction_m": {
            "max_abs": float(np.linalg.norm(lock_corr, axis=1).max()),
            "final": [float(v) for v in lock_corr[-1]],
        },
        "waist_angle_deg": {"max": float(np.rad2deg(waist_angle.max())), "mean": float(np.rad2deg(waist_angle.mean()))},
        "upper_body_error_deg": {
            "note": "chest (torso) orientation error; arm segments are re-expressed and keep their G1 world "
            "orientation when joint_mapping='anatomical' (up to saturation)",
            "chosen_mode_max": float(np.rad2deg(upper_err.max())),
            "chosen_mode_mean": float(np.rad2deg(upper_err.mean())),
            "drop_mode_max": float(np.rad2deg(waist_angle.max())),
            "drop_mode_mean": float(np.rad2deg(waist_angle.mean())),
            "fold_mode_max": 0.0,
        },
        "dropped_g1_joints": list(DROPPED_G1_JOINTS),
        "g1_sole_height_raw": {
            "p5": float(np.percentile(sole_h.min(axis=1), 5)),
            "median": float(np.median(sole_h.min(axis=1))),
        },
        "g1_contact_fraction": {"left": float(contacts[:, 0].mean()), "right": float(contacts[:, 1].mean())},
        "contact_detection": ({"method": "hysteresis", **vars(hyst)} if hyst is not None
                              else {"method": "threshold", **vars(opts.contact)}),
        "g1_stance_foot_speed_p95_mps_scaled": _stance_speed_p95(foot_c, contacts, motion.fps, scale),
        "g1_root_speed_mps": _root_speed_stats(hip_c, motion.fps),
        "g1_yaw_change_deg": float(np.rad2deg(np.abs(np.unwrap(yaw_of_matrix(r_pel))[-1] - yaw_of_matrix(r_pel)[0]))),
        "g1_min_hip_height_ratio": float((hip_c[:, 2] - ground).min() / g1.standing_hip_height),
        "g1_max_knee_rad": float(motion.dof[:, [3, 9]].max()),
        "g1_wrist_pitch_yaw_max_abs_deg": float(np.rad2deg(np.abs(motion.dof[:, [20, 21, 27, 28]]).max())),
    }
    return SemanticTrajectory(
        fps=motion.fps,
        pelvis_pos=pelvis_pos,
        pelvis_quat_wxyz=quat_continuous(matrix_to_quat(r_body)),
        q=q_ach,
        contacts=contacts,
        meta=meta,
        q_requested=q,
        source_foot_yaw=foot_yaw,
    )


def _foot_lock_correction(feet_xy: np.ndarray, contacts: np.ndarray) -> np.ndarray:
    """Root xy correction (T, 2) [m] that cancels the contact-weighted stance-foot displacement.

    ``feet_xy`` (T, 2 feet, 2) world foot centres computed with the uncorrected root. Increment
    ``d[t] - d[t-1] = -sum_s w_s (f_s[t] - f_s[t-1])`` with ``w_s`` = contact share of foot s at t and t-1;
    zero increment when no foot is in contact (flight / missing contact keeps the correction).
    Single support: the stance foot is exactly stationary. Double support: the mean of both feet is.
    """
    t = feet_xy.shape[0]
    both = contacts[1:] & contacts[:-1]  # stance at both ends of the step
    w = both.astype(np.float64)
    n = w.sum(axis=1, keepdims=True)
    w = np.divide(w, n, out=np.zeros_like(w), where=n > 0)
    step = feet_xy[1:] - feet_xy[:-1]  # (T-1, 2, 2)
    inc = -np.einsum("ts,tsk->tk", w, step)
    out = np.zeros((t, 2))
    out[1:] = np.cumsum(inc, axis=0)
    return out


def _stance_speed_p95(foot_c: np.ndarray, contacts: np.ndarray, fps: float, scale: float) -> float | None:
    """Reference: p95 horizontal speed of the *G1* stance feet, times the size scale [m/s]."""
    vals = []
    for s in range(2):
        c = contacts[:, s]
        if c.any():
            vals.append(float(np.percentile(horizontal_speed(foot_c[:, s], fps)[c], 95)) * scale)
    return max(vals) if vals else None


def _root_speed_stats(hip_c: np.ndarray, fps: float) -> dict[str, float]:
    v = horizontal_speed(hip_c, fps)
    disp = float(np.linalg.norm(hip_c[-1, :2] - hip_c[0, :2]))
    path = float(np.sum(np.linalg.norm(np.diff(hip_c[:, :2], axis=0), axis=1)))
    vz = np.gradient(hip_c[:, 2], 1.0 / fps)
    return {
        "mean": float(v.mean()),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
        "displacement_m": disp,
        "path_length_m": path,
        "max_up_velocity": float(vz.max()),
    }


# --------------------------------------------------------------------------------------------------
# semantic -> Dropbear motors + root
# --------------------------------------------------------------------------------------------------


def saturation_stats(
    q_requested: np.ndarray, q_achieved: np.ndarray, tol: float = 1e-4
) -> dict[str, dict[str, float]]:
    """Per semantic joint: fraction of frames clipped, max / mean |excess| [deg], requested range [deg]."""
    excess = q_requested - q_achieved
    out: dict[str, dict[str, float]] = {}
    for i, n in enumerate(SEMANTIC_NAMES):
        e = np.abs(excess[:, i])
        sat = e > tol
        out[n] = {
            "frac_frames": float(sat.mean()),
            "max_excess_deg": float(np.rad2deg(e.max())),
            "mean_excess_deg_when_saturated": float(np.rad2deg(e[sat].mean())) if sat.any() else 0.0,
            "requested_min_deg": float(np.rad2deg(q_requested[:, i].min())),
            "requested_max_deg": float(np.rad2deg(q_requested[:, i].max())),
        }
    return out


def semantic_to_dropbear(sem: SemanticTrajectory, cal: CalibrationView, serial_continuity: bool = True,
                         max_serial_speed_rad_s: float | None = None,
                         max_motor_step_rad: float | None = None) -> DropbearMotion:
    """Semantic trajectory -> motor angles (contract SemanticMap) + articulation-root pose.

    With ``serial_continuity`` (and a real SemanticMap) the serial3 groups use
    ``SemanticMap.semantic_to_motor_sequence`` (continuity-seeded, rate-limited to ``max_serial_speed_rad_s``); the
    achieved semantics (``q_back``) are what those motors realise, so the sidecar saturation also reports the lag."""
    q_motor, info = cal.semantic_to_motor(sem.q)
    seq_info = None
    leg_cols = None
    if sem.leg_motors is not None:  # foot stage: its leg motors are exact; keep them (no inverse round trip)
        from .foot_contact import leg_motor_columns

        leg_cols = leg_motor_columns()
    smap = getattr(cal, "semantic_map", None)
    if serial_continuity and hasattr(smap, "semantic_to_motor_sequence") and len(sem.q) > 1:
        step = None if max_serial_speed_rad_s is None else float(max_serial_speed_rad_s) / float(sem.fps)
        q_seq, seq_info = smap.semantic_to_motor_sequence(sem.q, max_serial_step=step)
        seq_info = {**seq_info, "max_serial_speed_rad_s": max_serial_speed_rad_s,
                    "max_motor_step_rad_per_frame_before": float(np.abs(np.diff(q_motor, axis=0)).max())}
        q_motor = q_seq
    if leg_cols is not None:
        roundtrip = float(np.abs(q_motor[:, leg_cols] - sem.leg_motors[:, leg_cols]).max())
        q_motor = q_motor.copy()
        q_motor[:, leg_cols] = sem.leg_motors[:, leg_cols]
        seq_info = {**(seq_info or {}), "foot_stage_leg_motor_inverse_roundtrip_max_rad": roundtrip}
    step_proj = None
    q_motor_pre_proj = q_motor
    if max_motor_step_rad is not None:
        from .foot_contact import project_motor_steps

        q_motor, step_proj = project_motor_steps(q_motor, float(max_motor_step_rad))
        step_proj["columns"] = {MOTOR_NAMES[c]: v for c, v in step_proj["columns"].items()}
    q_back = cal.motor_to_semantic(q_motor)
    q_back_pre = q_back
    if step_proj is not None and step_proj["columns"]:  # the semantic trajectory follows the projected motors
        sem = SemanticTrajectory(**{**sem.__dict__, "q": q_back.copy()})
        # saturation = clipping by the valid range, measured BEFORE the rate projection; the projection's own effect is
        # reported here (Euler-angle difference; ill-conditioned near the shoulder's YXZ singularity at 90 deg abduction,
        # where a small motor change reads as a large pitch/yaw change -- see max_change_rad per motor)
        q_back_pre = cal.motor_to_semantic(q_motor_pre_proj)
        dq = np.degrees(np.abs(q_back - q_back_pre).max(axis=0))
        step_proj["semantic_change_deg"] = {n: float(dq[i]) for i, n in enumerate(SEMANTIC_NAMES) if dq[i] > 1e-6}
        step_proj["note"] = ("Euler-angle difference; ill-conditioned near the shoulder YXZ singularity (90 deg "
                             "abduction), where a small motor change reads as a large pitch/yaw change")
    q_req = sem.q if sem.q_requested is None else sem.q_requested
    q_sat = q_back_pre
    if sem.q_pre_stage is not None:  # legs: clipping of the request (pre-stage), not the stage's own corrections
        from .foot_contact import LEG_SEM

        legs = [SEMANTIC_INDEX[f"{s}_{j}"] for s in ("left", "right") for j in LEG_SEM]
        q_sat = q_back_pre.copy()
        q_sat[:, legs] = sem.q_pre_stage[:, legs]
    sat = saturation_stats(q_req, q_sat)
    roundtrip_err = float(np.rad2deg(np.abs(sem.q - q_back).max()))

    limits = usd_motor_limits()
    lim_viol: dict[str, float] = {}
    for i, n in enumerate(MOTOR_NAMES):
        lo, hi = limits[n]
        over = np.maximum(q_motor[:, i] - hi, 0.0) + np.maximum(lo - q_motor[:, i], 0.0)
        if over.max() > 1e-6:
            lim_viol[n] = float(np.rad2deg(over.max()))

    r_body = quat_to_matrix(sem.pelvis_quat_wxyz)
    p_tr = cal.pelvis_T_root
    root_pos = sem.pelvis_pos + np.einsum("tij,j->ti", r_body, p_tr[:3, 3])
    root_rot = r_body @ p_tr[:3, :3]
    root_quat = quat_continuous(matrix_to_quat(root_rot))

    # Leg-model metrics on the *achieved* semantic pose.
    geom = LegGeometry.from_calibration(cal)
    lf = leg_fk(sem.pelvis_pos, r_body, q_back, geom)
    metrics: dict[str, Any] = {}
    if sem.contacts is not None:
        sole = lf.sole_height
        fc = lf.foot_center
        slip, pen, flt = [], [], []
        for s in range(2):
            c = sem.contacts[:, s]
            if c.any():
                v = horizontal_speed(fc[:, s], sem.fps)
                slip.append(float(np.percentile(v[c], 95)))
                pen.append(float(max(0.0, -sole[c, s].min())))
                flt.append(float(np.percentile(sole[c, s], 95)))
        metrics["stance_foot_slip_p95_mps"] = max(slip) if slip else None
        metrics["stance_penetration_max_m"] = max(pen) if pen else None
        metrics["stance_float_p95_m"] = max(flt) if flt else None
    metrics["min_sole_height_m"] = float(lf.sole_height.min())
    total_sat_frac = float(np.mean(np.abs(q_req - q_sat).max(axis=1) > 1e-4))
    metrics["achieved_semantic_roundtrip_err_deg"] = roundtrip_err
    saturation = {
        "per_joint": sat,
        "frames_with_any_saturation_frac": total_sat_frac,
        "worst_joint": max(sat, key=lambda k: sat[k]["max_excess_deg"]),
        "usd_motor_limit_violation_deg": lim_viol,
        "semantic_map_info_keys": sorted(info.keys()),
        "serial_continuity": seq_info,
        "motor_step_projection": step_proj,
        # cross-check: the SemanticMap's own report on the achieved pose (should be ~0: q is pre-clipped)
        "semantic_map_report_clip_frac_on_achieved": (
            {n: float(v) for n, v in zip(SEMANTIC_NAMES, np.asarray(info["clipped"]).reshape(-1, 22).mean(axis=0))
             if v > 0}
            if "clipped" in info
            else None
        ),
    }
    return DropbearMotion(
        fps=sem.fps,
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        motor_q=q_motor,
        semantic=sem,
        saturation=saturation,
        metrics=metrics,
    )


def apply_foot_stage(sem: SemanticTrajectory, cal: CalibrationView, params: FootContactParams,
                     cfk: Any = None) -> SemanticTrajectory:
    """Run :func:`dropbear_wbc.motion.foot_contact.foot_contact_stage` on a semantic trajectory (G1 route)."""
    from .foot_contact import foot_contact_stage, get_calibration_fk

    cfk = cfk or get_calibration_fk(cal.path)
    res = foot_contact_stage(sem.fps, sem.pelvis_pos, sem.pelvis_quat_wxyz, sem.q, sem.contacts, cfk,
                             cal.pelvis_T_root, params, source_yaw=sem.source_foot_yaw)
    meta = dict(sem.meta)
    meta["foot_contact_stage"] = res.report
    meta["contacts_before_foot_stage_fraction"] = {"left": float(sem.contacts[:, 0].mean()),
                                                   "right": float(sem.contacts[:, 1].mean())}
    return SemanticTrajectory(fps=sem.fps, pelvis_pos=res.pelvis_pos, pelvis_quat_wxyz=res.pelvis_quat_wxyz,
                              q=res.q_sem, contacts=res.contacts, meta=meta, q_requested=sem.q_requested,
                              source_foot_yaw=sem.source_foot_yaw, leg_motors=res.motor_q, q_pre_stage=sem.q.copy())


def retarget_g1_motion(
    motion: G1Motion, cal: CalibrationView, opts: RetargetOptions = RetargetOptions()
) -> DropbearMotion:
    """Full G1 -> Dropbear retarget (semantic mapping, root mapping, contacts, [foot stage], motors)."""
    sem = g1_to_semantic(motion, cal, opts)
    if opts.foot_contact is not None:
        sem = apply_foot_stage(sem, cal, opts.foot_contact)
    return semantic_to_dropbear(sem, cal, serial_continuity=opts.serial_continuity,
                                max_serial_speed_rad_s=opts.max_serial_speed_rad_s,
                                max_motor_step_rad=opts.max_motor_step_rad)
