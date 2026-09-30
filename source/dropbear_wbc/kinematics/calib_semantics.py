"""Semantic angle extraction from segment orientations (definitions of ``dropbear-semantic-v1``).

All rotations are root-relative (root frame = pelvis frame: x forward, y left, z up). For each segment X a
reference rotation ``R_X(ref)`` is stored in the calibration (legs: at the semantic zero pose; arms: at the
authored rest pose, upper arm hanging, arm straight). With ``F_X = R_X R_X(ref)^T``:

* hip / shoulder (pitch, roll, yaw) = intrinsic YXZ Euler angles of ``F_thigh`` / ``F_upper_arm``
  (``R = Ry(pitch) Rx(roll) Rz(yaw)``, the G1 joint order);
* knee = rotation angle of ``F_thigh^T F_shank``, signed by its axis' component on +y (flexion > 0);
* ankle (pitch, roll) = YXZ Euler of ``F_shank^T F_foot`` (the third angle is a residual, reported);
* elbow = pi/2 + rotation angle of ``F_upper^T F_forearm`` signed about +y (G1 ``*_elbow_joint``
  convention: straight arm = pi/2, flexion decreases the value, 0 = forearm pointing forward);
* wrist roll = rotation angle of ``F_forearm^T F_hand`` signed about the elbow->hand direction (-z in the
  rest-aligned forearm frame), i.e. the G1 wrist-roll axis (+x of the G1 elbow link).

Units: radians.
"""
from __future__ import annotations

import numpy as np

from dropbear_wbc.kinematics.rigid import euler_yxz, signed_angle_about
from dropbear_wbc.kinematics.semantic import SEMANTIC_NAMES

Y = np.array([0.0, 1.0, 0.0])
DOWN = np.array([0.0, 0.0, -1.0])
SIDES = ("left", "right")
SEGMENTS = ("thigh", "shank", "foot", "upper_arm", "forearm", "hand")


def _t(m: np.ndarray) -> np.ndarray:
    return np.swapaxes(m, -1, -2)


def leg_semantics(r_t: np.ndarray, r_s: np.ndarray, r_f: np.ndarray, ref: dict) -> dict[str, np.ndarray]:
    """Leg angles (+ diagnostics prefixed '_') from thigh/shank/foot rotations (..., 3, 3)."""
    ft, fs, ff = r_t @ _t(ref["thigh"]), r_s @ _t(ref["shank"]), r_f @ _t(ref["foot"])
    hip = euler_yxz(ft)
    knee, knee_axis = signed_angle_about(_t(ft) @ fs, Y)
    ank = euler_yxz(_t(fs) @ ff)
    return {"hip_pitch": hip[..., 0], "hip_roll": hip[..., 1], "hip_yaw": hip[..., 2], "knee": knee,
            "ankle_pitch": ank[..., 0], "ankle_roll": ank[..., 1],
            "_ankle_yaw_residual": ank[..., 2], "_knee_axis": knee_axis}


def arm_semantics(r_u: np.ndarray, r_e: np.ndarray, r_h: np.ndarray, ref: dict) -> dict[str, np.ndarray]:
    """Arm angles (+ diagnostics prefixed '_') from upper-arm/forearm/hand rotations (..., 3, 3)."""
    fu, fe, fh = r_u @ _t(ref["upper_arm"]), r_e @ _t(ref["forearm"]), r_h @ _t(ref["hand"])
    sh = euler_yxz(fu)
    el, el_axis = signed_angle_about(_t(fu) @ fe, Y)
    wr, wr_axis = signed_angle_about(_t(fe) @ fh, DOWN)
    return {"shoulder_pitch": sh[..., 0], "shoulder_roll": sh[..., 1], "shoulder_yaw": sh[..., 2],
            "elbow": np.pi / 2 + el, "wrist_roll": wr, "_elbow_axis": el_axis, "_wrist_axis": wr_axis}


def semantics_from_rotations(get_r, refs: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(..., 22) semantic vector and diagnostics from ``get_r(side, segment) -> (..., 3, 3)``.

    Args:
        get_r: callable returning root-relative rotations of a segment.
        refs: ``{side: {segment: (3, 3) reference rotation}}``.
    """
    out: dict[str, np.ndarray] = {}
    diag: dict[str, np.ndarray] = {}
    for side in SIDES:
        leg = leg_semantics(get_r(side, "thigh"), get_r(side, "shank"), get_r(side, "foot"), refs[side])
        arm = arm_semantics(get_r(side, "upper_arm"), get_r(side, "forearm"), get_r(side, "hand"), refs[side])
        for k, v in {**leg, **arm}.items():
            if k.startswith("_"):
                diag[f"{side}{k}"] = v
            else:
                out[f"{side}_{k}"] = v
    return np.stack([out[n] for n in SEMANTIC_NAMES], axis=-1), diag
