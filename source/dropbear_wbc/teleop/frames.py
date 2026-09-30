"""Frame conventions for XR teleop (televuer / xr_teleoperate) and the operator -> robot mapping.

The constants and the head-relative transform are adapted from ``unitreerobotics/televuer``
(``src/televuer/tv_wrapper.py`` @ 766de45e, MIT License, Copyright (c) 2025 Unitree Robotics; see
``third_party/televuer_NOTICE.md``).

Bases
-----
* OpenXR / WebXR (what Vuer streams): y up, z back, x right; 4x4 matrices arrive column-major (``order="F"``).
* Robot (this repo): x forward, y left, z up. ``Brobot = T_ROBOT_OPENXR @ Bxr @ T_OPENXR_ROBOT``.

Initial-pose conventions (televuer docstring)
---------------------------------------------
* Hand tracking: the WebXR wrist joint frame (x = wrist -> middle finger ... in the OpenXR arm convention) is
  turned into the **Unitree humanoid arm convention** by right-multiplying ``T_TO_UNITREE_HUMANOID_{LEFT,RIGHT}_ARM``.
* Controller tracking: the controller pose already follows the Unitree arm convention.
* Unitree arm convention = the Dropbear IK end-effector convention (:mod:`.arm_ik`): x along the forearm toward the
  fingers, z from pinky to index; identity with the forearm pointing forward, palm facing inward, thumb up.

Head-relative reference (``arm_reference_mode``)
------------------------------------------------
``head_yaw`` (xr_teleoperate default): wrist poses are expressed relative to the head position and the head's
**yaw only** (pitch/roll ignored), so looking around does not move the robot's hands. ``head_position``:
translation only. xr_teleoperate then shifts the origin from the head to the G1 waist by (+0.15, 0, +0.45) m;
Dropbear instead applies :class:`OperatorMapping` (scale + per-hand offset, set by a calibration step) into the
IK torso frame (pelvis origin, root axes; see :mod:`.arm_ik`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

T_TO_UNITREE_HUMANOID_LEFT_ARM = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
T_TO_UNITREE_HUMANOID_RIGHT_ARM = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
T_ROBOT_OPENXR = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
T_OPENXR_ROBOT = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
# televuer defaults when no valid XR data has arrived yet (OpenXR basis)
CONST_HEAD_POSE = np.array([[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, -0.2], [0, 0, 0, 1]], dtype=float)
CONST_RIGHT_ARM_POSE = np.array([[1, 0, 0, 0.15], [0, 1, 0, 1.13], [0, 0, 1, -0.3], [0, 0, 0, 1]], dtype=float)
CONST_LEFT_ARM_POSE = np.array([[1, 0, 0, -0.15], [0, 1, 0, 1.13], [0, 0, 1, -0.3], [0, 0, 0, 1]], dtype=float)
XR_TELEOP_WAIST_OFFSET = np.array([0.15, 0.0, 0.45])
"""xr_teleoperate's head -> G1 waist origin shift (informational; Dropbear uses :class:`OperatorMapping`)."""


def fast_mat_inv(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


def is_valid_pose(m: np.ndarray) -> bool:
    """televuer ``safe_mat_update`` criterion: finite and non-singular."""
    m = np.asarray(m, dtype=float)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        return False
    d = np.linalg.det(m)
    return bool(np.isfinite(d) and abs(d) > 1e-6)


def xr_matrix(flat16) -> np.ndarray:
    """Vuer/WebXR 16-float column-major matrix -> 4x4."""
    return np.asarray(flat16, dtype=float).reshape(4, 4, order="F")


def xr_flat(m: np.ndarray) -> list[float]:
    """4x4 -> Vuer/WebXR 16-float column-major list (inverse of :func:`xr_matrix`)."""
    return np.asarray(m, dtype=float).reshape(-1, order="F").tolist()


def to_robot_basis(m_xr: np.ndarray) -> np.ndarray:
    return T_ROBOT_OPENXR @ m_xr @ T_OPENXR_ROBOT


def to_xr_basis(m_robot: np.ndarray) -> np.ndarray:
    return T_OPENXR_ROBOT @ m_robot @ T_ROBOT_OPENXR


def head_yaw_rot(r_head_robot: np.ndarray) -> np.ndarray:
    """Yaw-only rotation of a head orientation in the robot basis (televuer ``get_Brobot_world_head_yaw_rot``)."""
    x = np.array(r_head_robot[:, 0], dtype=float, copy=True)
    x[2] = 0.0
    n = np.linalg.norm(x)
    if not np.isfinite(n) or n < 1e-6:
        return np.eye(3)
    x /= n
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, z])


def arm_relative_to_head(arm_world_robot: np.ndarray, head_world_robot: np.ndarray, mode: str = "head_yaw"
                         ) -> np.ndarray:
    """Wrist pose (robot basis, Unitree arm convention, XR world frame) -> relative to the head.

    ``head_yaw``: rotate by the head yaw and translate by the head position; ``head_position``: translate only.
    (televuer ``transform_IPunitree_Brobot_world_arm_to_head_then_waist`` without the waist shift.)"""
    out = np.array(arm_world_robot, dtype=float, copy=True)
    if mode == "head_yaw":
        ry = head_yaw_rot(head_world_robot[:3, :3])
        out[:3, :3] = ry.T @ arm_world_robot[:3, :3]
        out[:3, 3] = ry.T @ (arm_world_robot[:3, 3] - head_world_robot[:3, 3])
    elif mode == "head_position":
        out[:3, 3] = arm_world_robot[:3, 3] - head_world_robot[:3, 3]
    else:
        raise ValueError(f"unknown arm_reference_mode {mode!r}")
    return out


def xr_to_head_relative(head_xr: np.ndarray, left_xr: np.ndarray, right_xr: np.ndarray, hand_tracking: bool,
                        mode: str = "head_yaw") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw XR matrices (OpenXR basis, XR world) -> (head in robot basis, left, right wrist poses relative to the
    head in the robot basis and Unitree arm convention). Invalid matrices fall back to televuer's constants."""
    head = head_xr if is_valid_pose(head_xr) else CONST_HEAD_POSE
    lv, rv = is_valid_pose(left_xr), is_valid_pose(right_xr)
    left = left_xr if lv else CONST_LEFT_ARM_POSE
    right = right_xr if rv else CONST_RIGHT_ARM_POSE
    head_r = to_robot_basis(head)
    left_r = to_robot_basis(left)
    right_r = to_robot_basis(right)
    if hand_tracking:
        left_r = left_r @ (T_TO_UNITREE_HUMANOID_LEFT_ARM if lv else np.eye(4))
        right_r = right_r @ (T_TO_UNITREE_HUMANOID_RIGHT_ARM if rv else np.eye(4))
    return head_r, arm_relative_to_head(left_r, head_r, mode), arm_relative_to_head(right_r, head_r, mode)


def head_relative_to_xr(head_robot: np.ndarray, left_rel: np.ndarray, right_rel: np.ndarray, hand_tracking: bool,
                        mode: str = "head_yaw") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`xr_to_head_relative` (for simulated XR clients / tests): head pose in the robot basis and
    head-relative wrist poses -> raw XR matrices (OpenXR basis)."""
    ry = head_yaw_rot(head_robot[:3, :3]) if mode == "head_yaw" else np.eye(3)
    out = []
    for rel, t_conv in ((left_rel, T_TO_UNITREE_HUMANOID_LEFT_ARM), (right_rel, T_TO_UNITREE_HUMANOID_RIGHT_ARM)):
        w = np.eye(4)
        w[:3, :3] = ry @ rel[:3, :3]
        w[:3, 3] = ry @ rel[:3, 3] + head_robot[:3, 3]
        if hand_tracking:
            w = w @ fast_mat_inv(t_conv)
        out.append(to_xr_basis(w))
    return to_xr_basis(head_robot), out[0], out[1]


@dataclass
class OperatorMapping:
    """Head-relative operator wrist poses -> robot torso-frame wrist targets.

    ``p_torso = offset[side] + scale * p_head_rel``;  ``R_torso = R_offset[side] @ R_head_rel``.

    Defaults: ``offset`` = the nominal robot eye point in the torso frame for both hands (so an uncalibrated
    operator's hands land where a head-mounted view would put them), ``scale`` = robot arm length / human arm
    length, ``R_offset`` = identity (xr_teleoperate passes orientation through unchanged).
    :meth:`calibrate` replaces the offsets so that the operator's current (reference) wrist positions map exactly
    onto given robot wrist positions (xr_teleoperate has no such step; its ``scale_arms`` is commented out).
    """

    scale: float = 0.70
    eye_torso: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.0, 0.71]))
    offset: dict = field(default_factory=dict)
    r_offset: dict = field(default_factory=dict)
    calibrated: bool = False

    def __post_init__(self):
        for s in ("left", "right"):
            self.offset.setdefault(s, np.array(self.eye_torso, dtype=float))
            self.r_offset.setdefault(s, np.eye(3))

    def apply(self, side: str, rel: np.ndarray) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = self.r_offset[side] @ rel[:3, :3]
        out[:3, 3] = self.offset[side] + self.scale * rel[:3, 3]
        return out

    def calibrate(self, rel: dict, robot_ref: dict, orientation: bool = False) -> dict:
        """Set per-hand offsets from one reference pose. ``rel``/``robot_ref``: side -> 4x4 (head-relative operator
        pose, robot torso-frame reference pose). Returns a summary for the record."""
        out = {}
        for s in ("left", "right"):
            self.offset[s] = robot_ref[s][:3, 3] - self.scale * rel[s][:3, 3]
            if orientation:
                self.r_offset[s] = robot_ref[s][:3, :3] @ rel[s][:3, :3].T
            out[s] = {"offset": self.offset[s].tolist(), "r_offset": self.r_offset[s].tolist()}
        self.calibrated = True
        return {"scale": self.scale, "orientation": orientation, **out}

    def to_dict(self) -> dict:
        return {"scale": self.scale, "eye_torso": np.asarray(self.eye_torso).tolist(), "calibrated": self.calibrated,
                "offset": {s: np.asarray(v).tolist() for s, v in self.offset.items()},
                "r_offset": {s: np.asarray(v).tolist() for s, v in self.r_offset.items()}}
