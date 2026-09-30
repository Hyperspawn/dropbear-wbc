"""``dropbear-motion-csv-v1`` I/O and 50 Hz resampling (docs/CONTRACTS.md section 3).

CSV columns: ``root_x, root_y, root_z, root_qx, root_qy, root_qz, root_qw`` (pose of the root body
``world`` in the world frame; metres, xyzw quaternion) followed by the 22 motor angles [rad] named by
their USD joint names in motor-contract order. A header row is required. The sidecar ``<clip>.json``
holds at least ``fps``; ``contact_hint`` (optional) is per-frame ``[[left, right], ...]`` booleans or a
dict ``{"left": [...], "right": [...]}``.

Resampling follows BeyondMimic ``csv_to_npz.py``: linear interpolation of root position and joints,
slerp of the root quaternion, uniform output grid ``t = k / fps_out``. Unlike BeyondMimic, the final
input time is included when it falls on the output grid (so a 50 Hz input is passed through unchanged).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dropbear_wbc.motion.rotations import quat_continuous, quat_normalize, quat_slerp
from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES

ROOT_COLUMNS: tuple[str, ...] = ("root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw")
CSV_COLUMNS: tuple[str, ...] = ROOT_COLUMNS + tuple(MOTOR_NAMES)


@dataclass
class MotionClip:
    """A motion in motor space.

    Attributes:
        fps: frame rate [Hz].
        root_pos: (T, 3) root ``world`` body position [m].
        root_quat: (T, 4) root orientation, **wxyz** (converted from the CSV's xyzw).
        motor_pos: (T, 22) motor angles [rad], motor-contract order.
        contact: (T, 2) bool foot contact hint (left, right) or None.
        sidecar: the sidecar dict as read (may be empty).
    """

    fps: float
    root_pos: np.ndarray
    root_quat: np.ndarray
    motor_pos: np.ndarray
    contact: np.ndarray | None = None
    sidecar: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.motor_pos.shape[0])

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.fps


def sidecar_path(csv_path: Path) -> Path:
    return Path(csv_path).with_suffix(".json")


def _parse_contact(raw, num_frames: int) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        arr = np.stack([np.asarray(raw["left"], dtype=bool), np.asarray(raw["right"], dtype=bool)], axis=-1)
    else:
        arr = np.asarray(raw, dtype=bool)
    if arr.shape != (num_frames, 2):
        raise ValueError(f"contact_hint has shape {arr.shape}, expected ({num_frames}, 2)")
    return arr


def read_motion_csv(csv_path: str | Path, sidecar: str | Path | None = None) -> MotionClip:
    """Read a contract CSV and its sidecar JSON. Raises on header mismatch or missing ``fps``."""
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as f:
        header = next(csv.reader(f))
    header = [h.strip() for h in header]
    if tuple(header) != CSV_COLUMNS:
        raise ValueError(f"{csv_path}: header does not match dropbear-motion-csv-v1 "
                         f"(first mismatch at column {next(i for i, (a, b) in enumerate(zip(header, CSV_COLUMNS)) if a != b) if len(header) == len(CSV_COLUMNS) else 'count'})")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2, dtype=np.float64)
    side_path = Path(sidecar) if sidecar else sidecar_path(csv_path)
    side = json.loads(side_path.read_text()) if side_path.exists() else {}
    if "fps" not in side:
        raise ValueError(f"sidecar {side_path} missing or has no 'fps'")
    root_q = quat_continuous(quat_normalize(data[:, [6, 3, 4, 5]]))  # xyzw -> wxyz
    return MotionClip(fps=float(side["fps"]), root_pos=data[:, 0:3], root_quat=root_q, motor_pos=data[:, 7:],
                      contact=_parse_contact(side.get("contact_hint"), data.shape[0]), sidecar=side)


def write_motion_csv(csv_path: str | Path, clip: MotionClip, sidecar_extra: dict | None = None) -> None:
    """Write a contract CSV (+ sidecar with ``fps`` and optional ``contact_hint``)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    q = clip.root_quat
    rows = np.concatenate([clip.root_pos, q[:, 1:4], q[:, 0:1], clip.motor_pos], axis=1)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            w.writerow([f"{v:.9g}" for v in r])
    side = dict(clip.sidecar)
    side["fps"] = clip.fps
    if clip.contact is not None:
        side["contact_hint"] = clip.contact.astype(bool).tolist()
    if sidecar_extra:
        side.update(sidecar_extra)
    sidecar_path(csv_path).write_text(json.dumps(side, indent=1))


def resample(clip: MotionClip, fps_out: float = 50.0) -> MotionClip:
    """Resample to ``fps_out``: lerp root position/motors, slerp root quaternion, nearest contact hint."""
    n_in = clip.num_frames
    if n_in < 2:
        return MotionClip(fps_out, clip.root_pos.copy(), clip.root_quat.copy(), clip.motor_pos.copy(),
                          None if clip.contact is None else clip.contact.copy(), dict(clip.sidecar))
    duration = clip.duration
    dt = 1.0 / fps_out
    times = np.arange(0.0, duration + 1e-9, dt)
    phase = times * clip.fps
    i0 = np.clip(np.floor(phase).astype(np.int64), 0, n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w = np.clip(phase - i0, 0.0, 1.0)
    lerp = lambda a: a[i0] * (1.0 - w[:, None]) + a[i1] * w[:, None]  # noqa: E731
    root_q = quat_slerp(clip.root_quat[i0], clip.root_quat[i1], w)
    contact = None
    if clip.contact is not None:
        contact = clip.contact[np.clip(np.rint(phase).astype(np.int64), 0, n_in - 1)]
    return MotionClip(fps_out, lerp(clip.root_pos), quat_continuous(quat_normalize(root_q)), lerp(clip.motor_pos),
                      contact, dict(clip.sidecar))
