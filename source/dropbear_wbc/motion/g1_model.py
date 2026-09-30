"""Unitree G1 (29-DoF) joint orders and a vectorised, dependency-free forward kinematics.

The kinematic tree is parsed from the upstream MuJoCo MJCF
(``unitree_mujoco/unitree_robots/g1/g1_29dof.xml``, read-only reference clone). Only body
``pos``/``quat`` and hinge ``axis`` attributes are used; meshes are not loaded. The result
was cross-checked against ``mujoco.mj_kinematics`` in ``tests/test_g1_model.py``.

Frames / units
--------------
* World: x forward, y left, z up, metres. The floating base is the ``pelvis`` body.
* Joint angles: radians, MuJoCo/G1-SDK order (:data:`G1_JOINT_NAMES`).
* Foot sole points: the four contact spheres (radius 5 mm) of each ``*_ankle_roll_link``;
  the sole height is ``sphere_centre_z - 0.005``.
"""

from __future__ import annotations

import functools
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dropbear_wbc import paths as _paths

from .rotations import quat_to_matrix

__all__ = [
    "G1_JOINT_NAMES",
    "G1_ISAACLAB_JOINT_NAMES",
    "ISAACLAB_TO_MUJOCO",
    "ASAP_23DOF_JOINT_NAMES",
    "DEFAULT_G1_MJCF",
    "G1Kinematics",
    "G1FKResult",
    "load_g1_kinematics",
]

#: G1 SDK / MuJoCo joint order (29), as in unitree_rl_lab ``joint_sdk_names``.
G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
G1_JOINT_INDEX: dict[str, int] = {n: i for i, n in enumerate(G1_JOINT_NAMES)}

#: GR00T gear_sonic_deploy ``policy_parameters.hpp`` table ``isaaclab_to_mujoco``. Despite the name it is
#: indexed by the MuJoCo joint index and returns the Isaac-Lab index: ``fk.cpp`` reads
#: ``joint_angles_isaac[isaaclab_to_mujoco[mujoco_joint]]``. So ``q_mujoco[m] = q_isaac[ISAACLAB_TO_MUJOCO[m]]``.
ISAACLAB_TO_MUJOCO: tuple[int, ...] = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
)
G1_ISAACLAB_JOINT_NAMES: tuple[str, ...] = tuple(
    G1_JOINT_NAMES[ISAACLAB_TO_MUJOCO.index(i)] for i in range(len(ISAACLAB_TO_MUJOCO))
)

#: ASAP ``g1_29dof_anneal_23dof`` dof order (humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml).
ASAP_23DOF_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
)

DEFAULT_G1_MJCF = Path(os.environ.get("G1_MJCF") or _paths.setting(
    "G1_MJCF", str(_paths.upstream_dir() / "unitree_mujoco" / "unitree_robots" / "g1" / "g1_29dof.xml")))

_SOLE_SPHERE_RADIUS = 0.005


@dataclass(frozen=True)
class _Body:
    name: str
    parent: int  # -1 for the floating base
    pos: np.ndarray  # (3,) in parent frame
    rot: np.ndarray  # (3, 3) parent_R_body
    joint: int  # index into G1_JOINT_NAMES or -1 (fixed / floating)
    axis: np.ndarray  # (3,) hinge axis in body frame
    sole_points: np.ndarray  # (k, 3) contact sphere centres in body frame


@dataclass
class G1FKResult:
    """Forward-kinematics output for T frames.

    ``pos``: (T, B, 3) body origins in world [m]; ``rot``: (T, B, 3, 3) world_R_body.
    ``names``: body names (B).
    """

    names: tuple[str, ...]
    pos: np.ndarray
    rot: np.ndarray
    _sole_local: dict[str, np.ndarray]

    def index(self, name: str) -> int:
        return self.names.index(name)

    def body_pos(self, name: str) -> np.ndarray:
        return self.pos[:, self.index(name)]

    def body_rot(self, name: str) -> np.ndarray:
        return self.rot[:, self.index(name)]

    def sole_points(self, side: str) -> np.ndarray:
        """World positions (T, 4, 3) of the lowest points of the 4 contact spheres of ``side`` ('left'/'right'):
        sphere centre minus the radius along world z."""
        body = f"{side}_ankle_roll_link"
        b = self.index(body)
        centres = self.pos[:, b, None, :] + np.einsum("tij,kj->tki", self.rot[:, b], self._sole_local[body])
        return centres - np.array([0.0, 0.0, _SOLE_SPHERE_RADIUS])

    def sole_height(self, side: str) -> np.ndarray:
        """Lowest sole point z (T,) [m] for ``side``."""
        return self.sole_points(side)[..., 2].min(axis=-1)

    def foot_center(self, side: str) -> np.ndarray:
        """Mean of the sole points (T, 3) [m] for ``side``."""
        return self.sole_points(side).mean(axis=1)


class G1Kinematics:
    """Pure-numpy G1 forward kinematics parsed from an MJCF file."""

    def __init__(self, mjcf_path: Path | str = DEFAULT_G1_MJCF) -> None:
        self.mjcf_path = Path(mjcf_path)
        if not self.mjcf_path.is_file():
            raise FileNotFoundError(
                f"G1 MJCF not found at {self.mjcf_path}; set G1_MJCF to unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
            )
        self.bodies = self._parse(self.mjcf_path)
        self.names = tuple(b.name for b in self.bodies)

    @staticmethod
    def _floats(raw: str | None, n: int, default: tuple[float, ...]) -> np.ndarray:
        if raw is None:
            return np.array(default, dtype=np.float64)
        vals = [float(v) for v in raw.split()]
        if len(vals) != n:
            raise ValueError(f"expected {n} numbers, got {raw!r}")
        return np.array(vals, dtype=np.float64)

    def _parse(self, path: Path) -> list[_Body]:
        root = ET.parse(path).getroot()
        world = root.find("worldbody")
        if world is None:
            raise ValueError(f"{path}: no <worldbody>")
        pelvis = world.find("body")
        if pelvis is None or pelvis.get("name") != "pelvis":
            raise ValueError(f"{path}: first body must be 'pelvis'")
        bodies: list[_Body] = []

        def visit(el: ET.Element, parent: int) -> None:
            name = el.get("name", "")
            pos = self._floats(el.get("pos"), 3, (0.0, 0.0, 0.0))
            quat = self._floats(el.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
            if parent < 0:
                pos = np.zeros(3)  # floating base pose comes from the motion
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            joint_idx, axis = -1, np.zeros(3)
            for j in el.findall("joint"):
                jtype = j.get("type", "hinge")
                jname = j.get("name", "")
                if jtype == "hinge":
                    if jname not in G1_JOINT_INDEX:
                        raise ValueError(f"unknown G1 hinge {jname!r}")
                    joint_idx = G1_JOINT_INDEX[jname]
                    axis = self._floats(j.get("axis"), 3, (0.0, 0.0, 1.0))
                    jpos = self._floats(j.get("pos"), 3, (0.0, 0.0, 0.0))
                    if np.linalg.norm(jpos) > 1e-9:
                        raise ValueError(f"joint {jname} has non-zero pos; unsupported")
            sole = []
            for g in el.findall("geom"):
                gtype = g.get("type", "sphere")
                if gtype == "sphere" and name.endswith("ankle_roll_link"):
                    sole.append(self._floats(g.get("pos"), 3, (0.0, 0.0, 0.0)))
            bodies.append(
                _Body(
                    name=name,
                    parent=parent,
                    pos=pos,
                    rot=quat_to_matrix(quat),
                    joint=joint_idx,
                    axis=axis / max(np.linalg.norm(axis), 1e-12),
                    sole_points=np.array(sole).reshape(-1, 3),
                )
            )
            me = len(bodies) - 1
            for child in el.findall("body"):
                visit(child, me)

        visit(pelvis, -1)
        found = {b.joint for b in bodies if b.joint >= 0}
        if found != set(range(len(G1_JOINT_NAMES))):
            raise ValueError(f"{path}: expected 29 G1 hinges, found {len(found)}")
        return bodies

    def forward(self, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, dof: np.ndarray) -> G1FKResult:
        """Vectorised FK. Inputs (T,3), (T,4 wxyz), (T,29 rad, MuJoCo order)."""
        root_pos = np.atleast_2d(np.asarray(root_pos, dtype=np.float64))
        dof = np.atleast_2d(np.asarray(dof, dtype=np.float64))
        root_rot = quat_to_matrix(np.atleast_2d(root_quat_wxyz))
        t = root_pos.shape[0]
        if dof.shape != (t, len(G1_JOINT_NAMES)):
            raise ValueError(f"dof must be (T, 29), got {dof.shape}")
        n = len(self.bodies)
        pos = np.empty((t, n, 3))
        rot = np.empty((t, n, 3, 3))
        for i, b in enumerate(self.bodies):
            if b.parent < 0:
                pos[:, i] = root_pos
                r = root_rot
            else:
                pr = rot[:, b.parent]
                pos[:, i] = pos[:, b.parent] + pr @ b.pos
                r = pr @ b.rot
            if b.joint >= 0:
                r = r @ _axis_angle_matrix(b.axis, dof[:, b.joint])
            rot[:, i] = r
        sole = {b.name: b.sole_points for b in self.bodies if b.sole_points.size}
        return G1FKResult(names=self.names, pos=pos, rot=rot, _sole_local=sole)

    # ---- static reference quantities (zero pose, pelvis at origin) ---------------------------------
    @functools.cached_property
    def zero_pose(self) -> G1FKResult:
        return self.forward(np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]]), np.zeros((1, 29)))

    @property
    def hip_center_in_pelvis(self) -> np.ndarray:
        """Midpoint of the two hip-pitch joint origins in the pelvis frame [m] (the 'semantic pelvis point')."""
        z = self.zero_pose
        return 0.5 * (z.body_pos("left_hip_pitch_link")[0] + z.body_pos("right_hip_pitch_link")[0])

    @property
    def standing_hip_height(self) -> float:
        """Hip-pitch joint height above the sole with all joints at zero (straight legs) [m]."""
        z = self.zero_pose
        sole = min(z.sole_height("left")[0], z.sole_height("right")[0])
        return float(self.hip_center_in_pelvis[2] - sole)

    @property
    def standing_pelvis_height(self) -> float:
        """Pelvis origin height above the sole at the zero pose [m] (~0.79)."""
        z = self.zero_pose
        return float(-min(z.sole_height("left")[0], z.sole_height("right")[0]))

    @functools.cached_property
    def elbow_straight_value(self) -> float:
        """G1 elbow joint value [rad] at which shoulder (shoulder_roll_link origin), elbow and wrist
        (wrist_roll_link origin) are collinear (left arm; the right arm is mirror-symmetric). ~1.385."""

        def signed_flex(e: float) -> float:
            dof = np.zeros((1, 29))
            dof[0, G1_JOINT_INDEX["left_elbow_joint"]] = e
            f = self.forward(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0, 0.0]]), dof)
            s = f.body_pos("left_shoulder_roll_link")[0]
            el = f.body_pos("left_elbow_link")[0]
            w = f.body_pos("left_wrist_roll_link")[0]
            u, v = el - s, w - el
            axis = f.body_rot("left_elbow_link")[0][:, 1]
            return float(np.arctan2(np.dot(np.cross(u, v), axis), np.dot(u, v)))

        lo, hi = 0.5, 2.5  # signed_flex changes sign once in this bracket
        f_lo = signed_flex(lo)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = signed_flex(mid)
            if np.sign(f_mid) == np.sign(f_lo):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @functools.cached_property
    def elbow_axes_collinear_value(self) -> float:
        """G1 elbow value [rad] at which the shoulder-yaw axis (upper-arm axis) and the wrist-roll axis
        (forearm axis) are parallel -- the joint-axis notion of a straight arm used by the Dropbear
        calibration (``calib_build``: upper_forearm_angle from the same two axes). Exactly pi/2 for G1."""
        lo, hi = 0.5, 2.5
        for _ in range(3):  # coarse-to-fine batched search
            grid = np.linspace(lo, hi, 2001)
            dof = np.zeros((len(grid), 29))
            dof[:, G1_JOINT_INDEX["left_elbow_joint"]] = grid
            f = self.forward(np.zeros((len(grid), 3)), np.tile([1.0, 0.0, 0.0, 0.0], (len(grid), 1)), dof)
            up = f.body_rot("left_shoulder_yaw_link")[:, :, 2]
            fa = f.body_rot("left_wrist_roll_link")[:, :, 0]
            k = int(np.argmin(1.0 - np.abs(np.einsum("ti,ti->t", up, fa))))
            step = grid[1] - grid[0]
            lo, hi = grid[k] - step, grid[k] + step
        return float(grid[k])

    @property
    def elbow_semantic_offset(self) -> float:
        """Add to a G1 elbow value to get the semantic elbow (straight arm = pi/2) [rad].

        Uses the joint-axis definition of 'straight' (as the calibration does), which for G1 is exactly
        pi/2, so the offset is ~0: the G1 elbow maps by value. The joint-centre definition
        (:attr:`elbow_straight_value`, ~1.385) differs by ~10.6 deg because G1's elbow centre sits 1.6 cm
        in front of the upper-arm axis; it is reported as a diagnostic only."""
        return float(np.pi / 2 - self.elbow_axes_collinear_value)

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Joint ranges [rad] from the MJCF."""
        root = ET.parse(self.mjcf_path).getroot()
        out: dict[str, tuple[float, float]] = {}
        for j in root.iter("joint"):
            name = j.get("name", "")
            if name in G1_JOINT_INDEX and j.get("range"):
                lo, hi = (float(v) for v in j.get("range", "").split())
                out[name] = (lo, hi)
        return out


def _axis_angle_matrix(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrices for a fixed unit ``axis`` and angles (T,) -> (T, 3, 3)."""
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.stack(
        [
            c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C,
        ],
        axis=-1,
    ).reshape(angle.shape + (3, 3))


@functools.lru_cache(maxsize=4)
def load_g1_kinematics(mjcf_path: str | None = None) -> G1Kinematics:
    """Cached :class:`G1Kinematics` (default MJCF path or ``$G1_MJCF``)."""
    return G1Kinematics(mjcf_path or DEFAULT_G1_MJCF)
