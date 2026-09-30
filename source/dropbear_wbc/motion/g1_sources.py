"""Loaders for every Unitree-G1 motion-clip format found in the G1 ecosystem.

All loaders return a :class:`G1Motion`:

* ``root_pos``        (T, 3)  pelvis origin in the motion's world frame [m], z up, ground ~ z=0
* ``root_quat_wxyz``  (T, 4)  pelvis orientation, world_R_pelvis, unit, wxyz
* ``dof``             (T, 29) joint angles [rad], G1 SDK / MuJoCo order (:data:`G1_JOINT_NAMES`)
* ``fps``             source frame rate [Hz]

Supported formats (conventions verified against upstream readers, see each loader docstring):

========================  ===============================================================  ======
format id                 layout                                                           fps
========================  ===============================================================  ======
``qpos_xyzw``             36 cols, no header: pos(3) m, quat xyzw(4), 29 dof rad          given
``qpos_wxyz``             36 cols, no header: pos(3) m, quat wxyz(4), 29 dof rad          given
``bones_seed_csv``        header ``Frame,root_translateX..``; cm, extrinsic-xyz Euler deg,  120
                          29 dof deg (soma-retargeter / BONES-SEED G1 retarget)
``asap_pkl``              joblib {key: {root_trans_offset, root_rot xyzw, dof (T,23), fps}} file
``sonic_reference_dir``   GR00T gear_sonic_deploy reference folder (joint_pos.csv Isaac-Lab  50*
                          order + body_pos.csv/body_quat.csv, body 0 = pelvis, wxyz)
========================  ===============================================================  ======

``*`` = not verified on real data (the on-disk files are git-lfs pointers).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .g1_model import (
    ASAP_23DOF_JOINT_NAMES,
    G1_JOINT_INDEX,
    G1_JOINT_NAMES,
    ISAACLAB_TO_MUJOCO,
)
from .rotations import (
    euler_xyz_extrinsic_to_quat,
    quat_canonical,
    quat_continuous,
    quat_normalize,
    quat_slerp,
    xyzw_to_wxyz,
)

__all__ = [
    "G1Motion",
    "SourceSpec",
    "SOURCES",
    "LicenseInfo",
    "load_g1_motion",
    "detect_source",
    "discover_source_files",
    "load_qpos_csv",
    "load_bones_seed_csv",
    "load_asap_pkl",
    "load_sonic_reference_dir",
    "is_git_lfs_pointer",
]

from dropbear_wbc import paths as _paths  # noqa: E402

UPSTREAM = _paths.upstream_dir()  # $DROPBEAR_UPSTREAM / .dropbear.env, else ../upstream


@dataclass(frozen=True)
class LicenseInfo:
    """License / provenance record written into every sidecar."""

    spdx_or_name: str
    summary: str
    redistributable: bool
    url: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "license": self.spdx_or_name,
            "summary": self.summary,
            "redistributable": self.redistributable,
            "url": self.url,
        }


@dataclass
class G1Motion:
    """A G1 motion clip in the canonical G1 convention (see module docstring)."""

    fps: float
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    dof: np.ndarray
    source: str
    license: LicenseInfo
    source_file: str
    clip: str
    fmt: str
    dof_present: np.ndarray = field(default_factory=lambda: np.ones(29, dtype=bool))
    notes: list[str] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def duration(self) -> float:
        """Clip duration [s] = (T - 1) / fps."""
        return (self.num_frames - 1) / float(self.fps)

    def validate(self) -> None:
        """Raise ``ValueError`` on shape / finiteness / normalisation problems."""
        t = self.num_frames
        if t < 2:
            raise ValueError(f"{self.clip}: need >= 2 frames, got {t}")
        if self.root_pos.shape != (t, 3) or self.root_quat_wxyz.shape != (t, 4) or self.dof.shape != (t, 29):
            raise ValueError(
                f"{self.clip}: bad shapes pos{self.root_pos.shape} quat{self.root_quat_wxyz.shape} dof{self.dof.shape}"
            )
        for name, arr in (("root_pos", self.root_pos), ("root_quat", self.root_quat_wxyz), ("dof", self.dof)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{self.clip}: non-finite values in {name}")
        dev = np.abs(np.linalg.norm(self.root_quat_wxyz, axis=1) - 1.0).max()
        if dev > 1e-3:
            raise ValueError(f"{self.clip}: root quaternion not unit (max |q|-1 = {dev:.2e})")
        if not (1.0 <= self.fps <= 1000.0):
            raise ValueError(f"{self.clip}: implausible fps {self.fps}")

    def resample(self, fps: float) -> "G1Motion":
        """Resample to ``fps`` (linear pos/dof, slerp quaternions). Keeps the first frame, drops the tail
        beyond the last full output sample (same rule as BeyondMimic csv_to_npz)."""
        if abs(fps - self.fps) < 1e-9:
            return self
        times = np.arange(0.0, self.duration + 1e-9, 1.0 / fps)
        pos, quat, dof = _interp_frames(self.root_pos, self.root_quat_wxyz, self.dof, self.fps, times)
        notes = self.notes + [f"resampled {self.fps:g} Hz -> {fps:g} Hz (lerp/slerp)"]
        return replace(self, fps=float(fps), root_pos=pos, root_quat_wxyz=quat, dof=dof, notes=notes)


def _interp_frames(
    pos: np.ndarray, quat: np.ndarray, dof: np.ndarray, fps: float, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_src = times * fps
    i0 = np.clip(np.floor(t_src).astype(int), 0, len(pos) - 1)
    i1 = np.clip(i0 + 1, 0, len(pos) - 1)
    a = (t_src - i0)[:, None]
    q = quat_continuous(quat)
    return (
        (1 - a) * pos[i0] + a * pos[i1],
        quat_canonical(quat_slerp(q[i0], q[i1], a[:, 0])),
        (1 - a) * dof[i0] + a * dof[i1],
    )


# --------------------------------------------------------------------------------------------------
# Source registry
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """Where a G1 motion source lives, how to parse it and under which license."""

    key: str
    fmt: str
    fps: float | None  # None = read from file
    license: LicenseInfo
    patterns: tuple[str, ...]  # globs relative to UPSTREAM (or absolute)
    description: str
    path_markers: tuple[str, ...] = ()  # substrings used by detect_source()


SOURCES: dict[str, SourceSpec] = {
    "unitree_rl_lab_mimic": SourceSpec(
        key="unitree_rl_lab_mimic",
        fmt="qpos_xyzw",
        fps=60.0,  # unitree_rl_lab scripts/mimic/csv_to_npz.py --input_fps default 60; files named *_60hz.csv
        license=LicenseInfo(
            "Apache-2.0",
            "unitree_rl_lab repository license (Apache-2.0). Upstream mocap provenance of the dance clips is not "
            "documented in the repository; treat as research use until confirmed.",
            redistributable=True,
            url="https://github.com/unitreerobotics/unitree_rl_lab",
        ),
        patterns=("unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/*/*.csv",),
        description="unitree_rl_lab mimic task reference motions (BeyondMimic CSV, 60 Hz)",
        path_markers=("unitree_rl_lab",),
    ),
    "soma_retargeter_g1": SourceSpec(
        key="soma_retargeter_g1",
        fmt="bones_seed_csv",
        fps=120.0,  # BVH Frame Time 0.008333 s; soma_retargeter.io.csv.load_csv default fps=120
        license=LicenseInfo(
            "NVIDIA-Sample-Data-Evaluation-License",
            "BONES-SEED sample motions bundled with soma-retargeter: internal evaluation/testing of NVIDIA "
            "technologies only; no redistribution of the data or derivative works.",
            redistributable=False,
            url="https://github.com/NVIDIA/soma-retargeter/blob/main/assets/motions/LICENSE.txt",
        ),
        patterns=("soma-retargeter/assets/motions/csv/unitree_g1/*.csv",),
        description="soma-retargeter BONES-SEED sample clips retargeted to G1 (cm / Euler deg / deg, 120 Hz)",
        path_markers=("soma-retargeter", "unitree_g1"),
    ),
    "kimodo_g1": SourceSpec(
        key="kimodo_g1",
        fmt="qpos_wxyz",
        fps=30.0,  # ProtoMotions docs kimodo_preparation.rst: Kimodo G1 CSV at 30 fps, wxyz, m, rad
        license=LicenseInfo(
            "NVIDIA-Open-Model-License",
            "Kimodo-G1 generated motions (outputs of a model under the NVIDIA Open Model License), shipped in "
            "ProtoMotions (Apache-2.0 code).",
            redistributable=True,
            url="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
        ),
        patterns=("ProtoMotions/data/g1-kimodo-generated/*.csv",),
        description="Kimodo text-to-motion G1 generations (MuJoCo qpos CSV, wxyz, 30 Hz)",
        path_markers=("kimodo",),
    ),
    "asap_g1": SourceSpec(
        key="asap_g1",
        fmt="asap_pkl",
        fps=None,  # per-file 'fps' (30)
        license=LicenseInfo(
            "MIT",
            "ASAP repository (MIT). Motions are SMPL fits of human videos (TairanTestbed; filenames reference "
            "public figures) retargeted to G1-23dof; rights in the underlying videos are not covered by MIT.",
            redistributable=True,
            url="https://github.com/LeCAR-Lab/ASAP",
        ),
        patterns=("ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/**/*.pkl",),
        description="ASAP G1 23-dof (anneal) motion pkl (root_trans_offset, root_rot xyzw, dof 23, fps 30)",
        path_markers=("ASAP", "g1_29dof_anneal_23dof"),
    ),
    "lafan1_g1": SourceSpec(
        key="lafan1_g1",
        fmt="qpos_xyzw",
        fps=30.0,  # whole_body_tracking README: csv_to_npz.py --input_fps 30 for LAFAN1 G1 retargets
        license=LicenseInfo(
            "CC-BY-NC-ND-4.0",
            "LAFAN1 (Ubisoft La Forge) retargeted to G1 by lvhaidong/LAFAN1_Retargeting_Dataset: "
            "non-commercial, no derivatives.",
            redistributable=False,
            url="https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset",
        ),
        patterns=(
            "LAFAN1_Retargeting_Dataset/g1/*.csv",
            str(_paths.REPO_ROOT.parent / "data" / "LAFAN1_Retargeting_Dataset" / "g1" / "*.csv").replace("\\", "/"),
        ),
        description="LAFAN1 retargeted to G1 (BeyondMimic CSV: pos, quat xyzw, 29 dof, 30 Hz); not on disk yet",
        path_markers=("LAFAN",),
    ),
    "sonic_reference_g1": SourceSpec(
        key="sonic_reference_g1",
        fmt="sonic_reference_dir",
        fps=50.0,  # UNVERIFIED: BeyondMimic-style NPZ export at the 50 Hz policy rate
        license=LicenseInfo(
            "NVIDIA-GR00T-WholeBodyControl (see repo LICENSE dual notice)",
            "GR00T gear_sonic_deploy reference example motions; check the repository dual-license notice "
            "before redistribution.",
            redistributable=False,
            url="https://github.com/NVlabs/GR00T-WholeBodyControl",
        ),
        patterns=("GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/*",),
        description="GR00T SONIC deploy reference folders (joint_pos Isaac-Lab order + body pos/quat)",
        path_markers=("gear_sonic_deploy",),
    ),
}


def _resolve_pattern(pattern: str) -> list[str]:
    p = pattern if os.path.isabs(pattern) else str(UPSTREAM / pattern)
    return sorted(glob.glob(p, recursive=True))


def discover_source_files(key: str) -> list[Path]:
    """All on-disk files (or folders for ``sonic_reference_dir``) of source ``key``."""
    spec = SOURCES[key]
    out: list[Path] = []
    for pattern in spec.patterns:
        for f in _resolve_pattern(pattern):
            path = Path(f)
            if spec.fmt == "sonic_reference_dir":
                if path.is_dir() and (path / "joint_pos.csv").exists():
                    out.append(path)
            elif path.is_file():
                out.append(path)
    return out


def detect_source(path: Path | str) -> str:
    """Guess the source key from a path (used when the CLI gets no ``--source``)."""
    s = str(path).replace("\\", "/")
    if Path(path).is_dir():
        return "sonic_reference_g1"
    if s.endswith(".pkl"):
        return "asap_g1"
    for key in ("soma_retargeter_g1", "kimodo_g1", "unitree_rl_lab_mimic", "lafan1_g1", "sonic_reference_g1"):
        if any(m.lower() in s.lower() for m in SOURCES[key].path_markers):
            return key
    # Fall back on content: a BONES-SEED header, otherwise the BeyondMimic xyzw layout.
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    if first.startswith("Frame,root_translateX"):
        return "soma_retargeter_g1"
    raise ValueError(
        f"cannot infer source for {path}; pass --source (one of {sorted(SOURCES)}) -- 36-column qpos CSVs are "
        "ambiguous between xyzw (BeyondMimic/LAFAN1/unitree_rl_lab) and wxyz (Kimodo)"
    )


def is_git_lfs_pointer(path: Path | str) -> bool:
    """True if ``path`` is a git-lfs pointer stub rather than real data."""
    try:
        with open(path, "rb") as f:
            head = f.read(64)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec")


# --------------------------------------------------------------------------------------------------
# Format loaders (return raw arrays in canonical convention)
# --------------------------------------------------------------------------------------------------

RawMotion = tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, list[str]]


def load_qpos_csv(path: Path | str, quat_order: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """36-column MuJoCo-qpos CSV without header.

    * ``quat_order='xyzw'``: BeyondMimic / unitree_rl_lab / LAFAN1-G1 layout. Verified against
      ``unitree_rl_lab/scripts/mimic/csv_to_npz.py`` (``motion[:, 3:7][:, [3, 0, 1, 2]]  # convert to wxyz``).
    * ``quat_order='wxyz'``: Kimodo / ARDY G1 layout (ProtoMotions ``convert_g1_csv_to_proto.py
      --rot-format quat_wxyz``; kimodo ``MujocoQposConverter`` writes MuJoCo qpos which is wxyz).
    """
    if is_git_lfs_pointer(path):
        raise FileNotFoundError(f"{path} is a git-lfs pointer, not motion data")
    data = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if data.shape[1] != 36:
        raise ValueError(f"{path}: expected 36 columns, got {data.shape[1]}")
    pos = data[:, 0:3]
    if quat_order == "xyzw":
        quat = xyzw_to_wxyz(data[:, 3:7])
    elif quat_order == "wxyz":
        quat = data[:, 3:7].copy()
    else:
        raise ValueError(quat_order)
    return pos, quat_canonical(quat_normalize(quat)), data[:, 7:36]


def load_bones_seed_csv(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """soma-retargeter / BONES-SEED G1 CSV.

    Header ``Frame, root_translateX/Y/Z, root_rotateX/Y/Z, <29 joint names>``. Units: translation cm,
    root rotation = extrinsic x-y-z Euler degrees (``soma_retargeter/io/csv.py`` writes
    ``R.from_quat(q).as_euler('xyz', degrees=True)`` and reads back with ``wp.quat_rpy``; ProtoMotions
    docs: "extrinsic XYZ Euler angles in degrees"), joints degrees. Joint columns are matched by name.
    """
    with open(path, "r", encoding="utf-8") as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
    if header[:7] != [
        "Frame", "root_translateX", "root_translateY", "root_translateZ", "root_rotateX", "root_rotateY", "root_rotateZ"
    ]:
        raise ValueError(f"{path}: not a BONES-SEED G1 CSV header: {header[:7]}")
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    pos = data[:, 1:4] * 0.01
    quat = euler_xyz_extrinsic_to_quat(np.deg2rad(data[:, 4:7]))
    dof = np.zeros((data.shape[0], 29))
    names = header[7:]
    missing = [n for n in G1_JOINT_NAMES if n not in names]
    if missing:
        raise ValueError(f"{path}: missing G1 joints {missing}")
    for n in G1_JOINT_NAMES:
        dof[:, G1_JOINT_INDEX[n]] = np.deg2rad(data[:, 7 + names.index(n)])
    return pos, quat, dof


def load_asap_pkl(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, str]:
    """ASAP ``g1_29dof_anneal_23dof`` joblib pickle.

    Structure ``{motion_key: {root_trans_offset (T,3), root_rot (T,4) xyzw, dof (T,23), pose_aa, fps}}``.
    Conventions from ``scripts/vis/vis_q_mj.py``: ``qpos[:3] = root_trans_offset``,
    ``qpos[3:7] = root_rot[[3, 0, 1, 2]]`` (xyzw -> wxyz), ``qpos[7:] = dof`` with the 23-dof order of
    ``g1_29dof_anneal_23dof.yaml`` (no wrist roll/pitch/yaw). ``fit_smpl_motion.py`` writes
    ``root_rot = sRot.from_rotvec(...).as_quat()`` (scipy -> xyzw) and ``fps = 30``.
    The 6 wrist DoFs are filled with 0 and marked absent in ``dof_present``.
    """
    import joblib  # local import: only needed for this format

    blob = joblib.load(path)
    if not isinstance(blob, dict) or not blob:
        raise ValueError(f"{path}: expected a non-empty dict")
    if len(blob) != 1:
        raise ValueError(f"{path}: expected exactly one motion key, got {len(blob)}")
    key, m = next(iter(blob.items()))
    dof23 = np.asarray(m["dof"], dtype=np.float64)
    if dof23.ndim != 2 or dof23.shape[1] != len(ASAP_23DOF_JOINT_NAMES):
        raise ValueError(f"{path}: dof shape {dof23.shape}, expected (T, 23)")
    dof = np.zeros((dof23.shape[0], 29))
    present = np.zeros(29, dtype=bool)
    for j, n in enumerate(ASAP_23DOF_JOINT_NAMES):
        dof[:, G1_JOINT_INDEX[n]] = dof23[:, j]
        present[G1_JOINT_INDEX[n]] = True
    pos = np.asarray(m["root_trans_offset"], dtype=np.float64)
    quat = quat_canonical(quat_normalize(xyzw_to_wxyz(np.asarray(m["root_rot"], dtype=np.float64))))
    return pos, quat, dof, float(m["fps"]), present, str(key)


def load_sonic_reference_dir(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GR00T ``gear_sonic_deploy/reference/<motion>/`` folder (UNVERIFIED: files on disk are lfs pointers).

    ``joint_pos.csv`` (header, 29 cols, Isaac-Lab order -> remapped with ``isaaclab_to_mujoco`` from
    policy_parameters.hpp, used as in ``fk.cpp``: ``q_mujoco[m] = q_isaac[isaaclab_to_mujoco[m]]``), ``body_pos.csv`` / ``body_quat.csv`` (header, 14 bodies x 3 / x 4 wxyz; body 0 is
    the pelvis per ``metadata.txt`` body indexes ``[0, ...]``).
    """
    path = Path(path)
    for f in ("joint_pos.csv", "body_pos.csv", "body_quat.csv"):
        if is_git_lfs_pointer(path / f):
            raise FileNotFoundError(f"{path / f} is a git-lfs pointer (run git lfs pull in the upstream clone)")
    jp = np.loadtxt(path / "joint_pos.csv", delimiter=",", skiprows=1, ndmin=2)
    bp = np.loadtxt(path / "body_pos.csv", delimiter=",", skiprows=1, ndmin=2)
    bq = np.loadtxt(path / "body_quat.csv", delimiter=",", skiprows=1, ndmin=2)
    if jp.shape[1] != 29:
        raise ValueError(f"{path}: joint_pos has {jp.shape[1]} columns")
    dof = jp[:, list(ISAACLAB_TO_MUJOCO)]  # q_mujoco[m] = q_isaac[isaaclab_to_mujoco[m]]
    return bp[:, 0:3], quat_canonical(quat_normalize(bq[:, 0:4])), dof


# --------------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------------


def _clip_name(path: Path, source: str) -> str:
    name = path.name if path.is_dir() else path.stem
    if source == "unitree_rl_lab_mimic":
        name = name.replace(".bvh_60hz", "")
    if source == "asap_g1":
        name = name.removeprefix("0-")
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def load_g1_motion(path: Path | str, source: str | None = None, fps: float | None = None) -> G1Motion:
    """Load any supported G1 clip into a :class:`G1Motion`.

    ``source`` selects the :data:`SOURCES` entry (format, fps, license); inferred from the path if None.
    ``fps`` overrides the source default (needed only for unusual files).
    """
    path = Path(path)
    source = source or detect_source(path)
    if source not in SOURCES:
        raise KeyError(f"unknown source {source!r}; choose from {sorted(SOURCES)}")
    spec = SOURCES[source]
    present = np.ones(29, dtype=bool)
    notes: list[str] = []
    if spec.fmt == "qpos_xyzw":
        pos, quat, dof = load_qpos_csv(path, "xyzw")
        src_fps = spec.fps
    elif spec.fmt == "qpos_wxyz":
        pos, quat, dof = load_qpos_csv(path, "wxyz")
        src_fps = spec.fps
    elif spec.fmt == "bones_seed_csv":
        pos, quat, dof = load_bones_seed_csv(path)
        src_fps = spec.fps
    elif spec.fmt == "asap_pkl":
        pos, quat, dof, src_fps, present, key = load_asap_pkl(path)
        notes.append(f"asap motion key: {key}; wrist roll/pitch/yaw absent in 23-dof source (set to 0)")
    elif spec.fmt == "sonic_reference_dir":
        pos, quat, dof = load_sonic_reference_dir(path)
        src_fps = spec.fps
        notes.append("sonic reference loader and 50 Hz rate are UNVERIFIED (no real files on disk)")
    else:  # pragma: no cover - registry is static
        raise ValueError(spec.fmt)
    if fps is not None:
        src_fps = float(fps)
    if src_fps is None:
        raise ValueError(f"{path}: fps unknown; pass fps=")
    motion = G1Motion(
        fps=float(src_fps),
        root_pos=np.ascontiguousarray(pos, dtype=np.float64),
        root_quat_wxyz=np.ascontiguousarray(quat_continuous(quat), dtype=np.float64),
        dof=np.ascontiguousarray(dof, dtype=np.float64),
        source=source,
        license=spec.license,
        source_file=str(path).replace("\\", "/"),
        clip=_clip_name(path, source),
        fmt=spec.fmt,
        dof_present=present,
        notes=notes,
    )
    motion.validate()
    return motion


LOADERS: dict[str, Callable[..., object]] = {
    "qpos_xyzw": lambda p: load_qpos_csv(p, "xyzw"),
    "qpos_wxyz": lambda p: load_qpos_csv(p, "wxyz"),
    "bones_seed_csv": load_bones_seed_csv,
    "asap_pkl": load_asap_pkl,
    "sonic_reference_dir": load_sonic_reference_dir,
}


def source_summary() -> str:
    """Human-readable table of sources and on-disk file counts (for logs)."""
    lines = []
    for key, spec in SOURCES.items():
        files = discover_source_files(key)
        lfs = sum(1 for f in files if f.is_dir() and is_git_lfs_pointer(f / "joint_pos.csv"))
        lines.append(
            f"{key:22s} fmt={spec.fmt:20s} fps={spec.fps!s:6s} files={len(files):3d} lfs_pointers={lfs:3d} "
            f"license={spec.license.spdx_or_name}"
        )
    return "\n".join(lines)


def dump_sources_json() -> str:
    return json.dumps(
        {k: {"fmt": s.fmt, "fps": s.fps, "license": s.license.as_dict(), "patterns": s.patterns} for k, s in SOURCES.items()},
        indent=2,
    )
