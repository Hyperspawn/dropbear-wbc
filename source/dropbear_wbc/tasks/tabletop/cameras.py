"""Camera placement for the tabletop task (pure numpy).

All cameras use Isaac Lab's ``"world"`` offset convention (camera forward = +x, up = +z of the camera frame).

* ``head`` (ego view, 640 x 480): rigidly attached to the anchor body ``head_5mm_ujoint_base__5__1`` (fixed to the
  torso; CONTRACTS 5.1), 1.2 cm in front of the visor's front face at eye height on the torso midline, pitched 68 deg
  down, where a real head camera would sit. The head (visor) body ``head_u_joint_center__8__1`` spans root x -0.112 ..
  0.123, z 1.774 .. 1.992 (USD visual bbox, ``logs/tabletop/body_visual_bbox_rest.json``); the first pose (0.10, 1.78)
  was INSIDE it (uniform frames, ``logs/tabletop/smoke_v2.json``). Candidates compared in
  ``logs/tabletop/camera_probe_v1`` / ``media/camera_probe_v1_head_candidates.png``: the chest handle covers the lower
  centre of the image, the lateral work areas, block, zone and pushing hand are visible for both arms. The head itself
  sits on the 6-DoF neck Stewart platform (PD-held), so the anchor gives a steadier ego view than the head link; the
  pose is expressed in the anchor LINK frame.
* ``left_wrist`` / ``right_wrist`` (optional; rendered 640 x 480 like the head, stored 320 x 240): on the hand body
  ``LH/RH_shoulder_ex_al_interface_1``, 6 cm off the hand axis, looking along the hand toward its end face (the hand
  extends along body -y). Rendering them at 320 x 240 coincided with intermittent uniform frames on every TiledCamera
  (smoke_v2); the probe at one resolution had none.
* ``scene`` (optional, demo only, not part of the dataset): fixed third-person view in the env frame.
"""
from __future__ import annotations

import math

import numpy as np

ANCHOR_BODY = "head_5mm_ujoint_base__5__1"
ANCHOR_REST_POS_ROOT = np.array([0.101, -0.0447, 1.6388])     # logs/tabletop/geometry_probe.log (authored rest)
ANCHOR_REST_QUAT_ROOT = np.array([0.5, 0.5, 0.5, 0.5])        # wxyz; rigid with the root (fixed joint)
HAND_BODY = {"left": "LH_shoulder_ex_al_interface_1", "right": "RH_shoulder_ex_al_interface_1"}

HEAD_CAM_POS_ROOT = np.array([0.135, -0.0697, 1.88])
HEAD_CAM_PITCH_DOWN_DEG = 68.0
HEAD_CAM_FOCAL_MM = 10.0            # with the 20.955 mm aperture: 92 deg horizontal FOV
HEAD_CAM_RES = (640, 480)
WRIST_CAM_RES = (640, 480)          # render resolution
WRIST_CAM_STORE_RES = (320, 240)    # recorded resolution (downscaled in the collector)
WRIST_CAM_FOCAL_MM = 9.0
SCENE_CAM_POS_ENV = np.array([1.45, 0.95, 1.95])   # env frame (world units), third-person
SCENE_CAM_TARGET_ENV = np.array([0.25, -0.07, 1.05])


def quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z)."""
    r = np.asarray(r, float)
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def look_rotation(forward: np.ndarray, up_hint=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World-convention camera rotation (columns = camera x forward, y left, z up) looking along ``forward``."""
    f = np.asarray(forward, float)
    f /= np.linalg.norm(f)
    left = np.cross(np.asarray(up_hint, float), f)
    left /= np.linalg.norm(left)
    up = np.cross(f, left)
    return np.stack([f, left, up], axis=1)


def head_cam_offset() -> tuple[tuple, tuple]:
    """(pos, quat wxyz) of the head camera in the anchor LINK frame (world convention)."""
    p = math.radians(HEAD_CAM_PITCH_DOWN_DEG)
    r_cam_root = look_rotation(np.array([math.cos(p), 0.0, -math.sin(p)]))
    r_anchor = matrix_from_quat(ANCHOR_REST_QUAT_ROOT)
    pos = r_anchor.T @ (HEAD_CAM_POS_ROOT - ANCHOR_REST_POS_ROOT)
    q = quat_from_matrix(r_anchor.T @ r_cam_root)
    return tuple(float(v) for v in pos), tuple(float(v) for v in q)


def wrist_cam_offset(side: str) -> tuple[tuple, tuple]:
    """(pos, quat wxyz) of a wrist camera in the hand BODY frame: 6 cm off-axis along body +x, 2 cm down the hand,
    looking along body -y (toward the hand's end face), tilted 12 deg toward the axis."""
    t = math.radians(12.0)
    fwd = np.array([-math.sin(t), -math.cos(t), 0.0])
    r = look_rotation(fwd, up_hint=(1.0, 0.0, 0.0))
    q = quat_from_matrix(r)
    return (0.06, -0.02, 0.0), tuple(float(v) for v in q)


def scene_cam_pose(root_pos_w) -> tuple[tuple, tuple]:
    """(pos, quat wxyz) of the third-person camera in the env frame."""
    pos = SCENE_CAM_POS_ENV
    r = look_rotation(SCENE_CAM_TARGET_ENV - pos)
    return tuple(float(v) for v in pos), tuple(float(v) for v in quat_from_matrix(r))


def head_cam_pose_root() -> tuple[np.ndarray, np.ndarray]:
    """(position (3,), rotation (3, 3), columns = camera x forward, y left, z up) of the head camera in the ROOT frame
    (the anchor is rigid with the root, so this is the rest pose exactly)."""
    p = math.radians(HEAD_CAM_PITCH_DOWN_DEG)
    return HEAD_CAM_POS_ROOT.copy(), look_rotation(np.array([math.cos(p), 0.0, -math.sin(p)]))


def project_head(p_root, res=HEAD_CAM_RES, focal_mm: float = HEAD_CAM_FOCAL_MM, aperture_mm: float = 20.955):
    """Pixel (u right, v down) of a root-frame point in the head image (square pixels), or None behind the camera."""
    pos, r = head_cam_pose_root()
    c = r.T @ (np.asarray(p_root, float) - pos)  # camera frame: x forward, y left, z up
    if c[0] <= 1e-6:
        return None
    f = focal_mm / aperture_mm * res[0]
    return np.array([0.5 * res[0] - f * c[1] / c[0], 0.5 * res[1] - f * c[2] / c[0]])


def green_centroid(rgb: np.ndarray, min_pixels: int = 30):
    """(centroid (u, v), pixel count) of the target zone's green in an RGB frame; centroid None below ``min_pixels``."""
    a = np.asarray(rgb[..., :3], dtype=np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (g > 120) & (g > r + 40) & (g > b + 20)
    n = int(m.sum())
    if n < min_pixels:
        return None, n
    vs, us = np.nonzero(m)
    return np.array([us.mean(), vs.mean()]), n
