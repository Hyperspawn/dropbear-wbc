"""Writer, reader and validator for ``dropbear-motion-csv-v1`` (CONTRACTS.md section 3).

Files per clip (``data/motions/<source>/``):

* ``<clip>.csv``   header ``root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw,<22 USD motor names>``;
                    root = articulation root body ``world`` in the world frame [m, quaternion xyzw];
                    motors [rad] in motor-contract order.
* ``<clip>.json``  sidecar: ``fps, source, source_file, source_license, retarget_method, semantic_names,
                    semantic_trajectory, contact_hint{left,right}, notes`` (+ extra diagnostics).
* ``semantic/<clip>.semantic.csv``  optional semantic trajectory: pelvis pose (xyzw) + 22 semantic
                    angles [rad] + contact_left/right (0/1). Kept in a sub-folder so globbing
                    ``<source>/*.csv`` only yields motor CSVs.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .names import CSV_COLUMNS, MOTOR_NAMES, SEMANTIC_NAMES
from .rotations import quat_continuous, wxyz_to_xyzw, xyzw_to_wxyz

__all__ = [
    "SCHEMA_ID",
    "SIDECAR_REQUIRED",
    "write_motion",
    "read_motion_csv",
    "read_sidecar",
    "validate_motion_files",
    "MotionFileError",
    "LoadedMotion",
]

SCHEMA_ID = "dropbear-motion-csv-v1"
SIDECAR_REQUIRED = (
    "fps",
    "source",
    "source_file",
    "source_license",
    "retarget_method",
    "semantic_names",
    "contact_hint",
    "notes",
)
SEMANTIC_COLUMNS = (
    ("pelvis_x", "pelvis_y", "pelvis_z", "pelvis_qx", "pelvis_qy", "pelvis_qz", "pelvis_qw")
    + SEMANTIC_NAMES
    + ("contact_left", "contact_right")
)


class MotionFileError(ValueError):
    pass


@dataclass
class LoadedMotion:
    fps: float
    root_pos: np.ndarray  # (T, 3)
    root_quat_wxyz: np.ndarray  # (T, 4)
    motor_q: np.ndarray  # (T, 22)
    sidecar: dict[str, Any]


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o).replace("\\", "/")
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_motion(
    out_dir: Path | str,
    clip: str,
    *,
    fps: float,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    motor_q: np.ndarray,
    sidecar: dict[str, Any],
    semantic: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None = None,
) -> tuple[Path, Path]:
    """Write ``<clip>.csv`` + ``<clip>.json`` (+ ``semantic/<clip>.semantic.csv``). Returns (csv, json).

    ``semantic`` = (pelvis_pos (T,3), pelvis_quat_wxyz (T,4), q_sem (T,22), contacts (T,2) or None).
    ``sidecar`` must contain the contract keys except ``fps``/``semantic_names``/``semantic_trajectory``
    which are filled here.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = int(root_pos.shape[0])
    if root_quat_wxyz.shape != (t, 4) or motor_q.shape != (t, len(MOTOR_NAMES)):
        raise MotionFileError(f"bad shapes: quat {root_quat_wxyz.shape}, motors {motor_q.shape}, T={t}")
    data = np.concatenate([root_pos, wxyz_to_xyzw(quat_continuous(root_quat_wxyz)), motor_q], axis=1)
    if not np.all(np.isfinite(data)):
        raise MotionFileError(f"{clip}: non-finite values")
    csv_path = out_dir / f"{clip}.csv"
    np.savetxt(csv_path, data, delimiter=",", header=",".join(CSV_COLUMNS), comments="", fmt="%.7f")

    side = dict(sidecar)
    side["schema"] = SCHEMA_ID
    side["clip"] = clip
    side["fps"] = float(fps)
    side["num_frames"] = t
    side["duration_s"] = (t - 1) / float(fps)
    side["semantic_names"] = list(SEMANTIC_NAMES)
    side["motor_names"] = list(MOTOR_NAMES)
    side.setdefault("created_utc", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    if semantic is not None:
        ppos, pquat, qsem, contacts = semantic
        sem_dir = out_dir / "semantic"
        sem_dir.mkdir(exist_ok=True)
        c = np.zeros((t, 2)) if contacts is None else contacts.astype(np.float64)
        sem = np.concatenate([ppos, wxyz_to_xyzw(quat_continuous(pquat)), qsem, c], axis=1)
        sem_path = sem_dir / f"{clip}.semantic.csv"
        np.savetxt(sem_path, sem, delimiter=",", header=",".join(SEMANTIC_COLUMNS), comments="", fmt="%.7f")
        side["semantic_trajectory"] = f"semantic/{clip}.semantic.csv"
    else:
        side.setdefault("semantic_trajectory", None)
    missing = [k for k in SIDECAR_REQUIRED if k not in side]
    if missing:
        raise MotionFileError(f"sidecar for {clip} lacks {missing}")
    json_path = out_dir / f"{clip}.json"
    json_path.write_text(json.dumps(side, indent=1, default=_json_default), encoding="utf-8")
    return csv_path, json_path


def read_sidecar(csv_path: Path | str) -> dict[str, Any]:
    p = Path(csv_path).with_suffix(".json")
    if not p.is_file():
        raise MotionFileError(f"missing sidecar {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def read_motion_csv(csv_path: Path | str) -> LoadedMotion:
    """Read a motion CSV (header checked) and its sidecar."""
    csv_path = Path(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        header = tuple(h.strip() for h in f.readline().strip().split(","))
    if header != CSV_COLUMNS:
        raise MotionFileError(f"{csv_path}: header mismatch (expected {len(CSV_COLUMNS)} contract columns)")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    side = read_sidecar(csv_path)
    return LoadedMotion(
        fps=float(side["fps"]),
        root_pos=data[:, 0:3],
        root_quat_wxyz=xyzw_to_wxyz(data[:, 3:7]),
        motor_q=data[:, 7:],
        sidecar=side,
    )


def validate_motion_files(csv_path: Path | str) -> list[str]:
    """Return a list of problems (empty = valid) for a clip's CSV + sidecar (+ semantic file)."""
    problems: list[str] = []
    try:
        m = read_motion_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        return [f"read error: {exc}"]
    t = m.motor_q.shape[0]
    side = m.sidecar
    for k in SIDECAR_REQUIRED:
        if k not in side:
            problems.append(f"sidecar missing '{k}'")
    if side.get("schema") != SCHEMA_ID:
        problems.append(f"sidecar schema {side.get('schema')!r} != {SCHEMA_ID}")
    if not (isinstance(side.get("fps"), (int, float)) and 1 <= side["fps"] <= 1000):
        problems.append(f"bad fps {side.get('fps')!r}")
    if side.get("num_frames") != t:
        problems.append(f"num_frames {side.get('num_frames')} != csv rows {t}")
    if tuple(side.get("semantic_names", ())) != SEMANTIC_NAMES:
        problems.append("semantic_names differ from contract")
    if t < 2:
        problems.append("fewer than 2 frames")
    if m.motor_q.shape[1] != 22:
        problems.append(f"{m.motor_q.shape[1]} motor columns")
    if not np.all(np.isfinite(m.motor_q)) or not np.all(np.isfinite(m.root_pos)):
        problems.append("non-finite values")
    qn = np.linalg.norm(m.root_quat_wxyz, axis=1)
    if np.abs(qn - 1).max() > 1e-5:
        problems.append(f"root quaternion not unit (max dev {np.abs(qn - 1).max():.2e})")
    ch = side.get("contact_hint")
    if ch is not None:
        if not isinstance(ch, dict) or set(ch) - {"left", "right", "method"} or "left" not in ch or "right" not in ch:
            problems.append("contact_hint must be {left: [...], right: [...]} (+ optional method)")
        elif len(ch["left"]) != t or len(ch["right"]) != t:
            problems.append("contact_hint length != num_frames")
        elif not all(isinstance(v, bool) for v in ch["left"] + ch["right"]):
            problems.append("contact_hint values must be booleans")
    lic = side.get("source_license")
    if not isinstance(lic, dict) or "license" not in lic:
        problems.append("source_license must be a dict with 'license'")
    sem_rel = side.get("semantic_trajectory")
    if sem_rel:
        sp = Path(csv_path).parent / sem_rel
        if not sp.is_file():
            problems.append(f"semantic_trajectory {sem_rel} missing")
        else:
            with open(sp, "r", encoding="utf-8") as f:
                hdr = tuple(h.strip() for h in f.readline().strip().split(","))
            if hdr != SEMANTIC_COLUMNS:
                problems.append("semantic trajectory header mismatch")
            else:
                sd = np.loadtxt(sp, delimiter=",", skiprows=1, ndmin=2)
                if sd.shape[0] != t:
                    problems.append("semantic trajectory length != num_frames")
    return problems
