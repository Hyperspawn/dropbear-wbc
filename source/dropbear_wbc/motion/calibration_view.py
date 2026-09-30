"""Read-only view of ``dropbear-semantic-calibration-v1`` plus the SemanticMap used for semantic->motor.

The motion pipeline needs only a handful of calibration quantities:

* ``standing_hip_height`` [m] - hip-pitch axis height above the sole, legs straight (scales G1 motion),
* ``root_T_pelvis`` (4x4)    - pose of the semantic pelvis frame (hip centre, x fwd / y left / z up)
                               expressed in the articulation root body ``world`` frame (rigid, both are
                               on the same Dropbear body),
* leg / foot segment lengths - used by the approximate "semantic skeleton" FK (root height, contacts,
                               synthetic IK),
* ``standing_semantic_pos``   - semantic pose of ``standing_motor_pos``.

Key names in the real calibration JSON are not fixed by CONTRACTS.md, so each field is looked up
under several candidate paths (:data:`FIELD_PATHS`). A missing field raises with the list of paths
tried; extend the candidate lists rather than editing callers.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .names import MOTOR_NAMES, SEMANTIC_NAMES
from .rotations import quat_to_matrix

__all__ = [
    "DEFAULT_CALIBRATION",
    "MOCK_CALIBRATION",
    "SemanticMapLike",
    "CalibrationView",
    "load_calibration",
]

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CALIBRATION = Path(
    os.environ.get("DROPBEAR_CALIBRATION", str(REPO / "data/calibration/dropbear_semantic_calibration.json"))
)
MOCK_CALIBRATION = REPO / "tests/fixtures/mock_semantic_calibration.json"

FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "standing_hip_height": (
        "segment_lengths.standing_hip_height",
        "segment_lengths.standing_hip_height_m",
        "segments.standing_hip_height",
        "segments.standing_hip_height_m",
        "standing_hip_height",
        "standing_hip_height_m",
    ),
    "root_T_pelvis": (
        "rest_transforms.pelvis_in_root",
        "rest_transforms.root_T_pelvis",
        "rest_transforms.world_T_pelvis",
        "rest_transforms.pelvis",
    ),
    "pelvis_T_root": (
        "rest_transforms.root_in_pelvis",
        "rest_transforms.pelvis_T_root",
        "rest_transforms.pelvis_T_world",
    ),
    "hip_width": ("segment_lengths.hip_width", "segments.hip_width"),
    "hip_to_knee": ("segment_lengths.hip_to_knee", "segments.hip_to_knee", "segment_lengths.thigh"),
    "knee_to_ankle": ("segment_lengths.knee_to_ankle", "segments.knee_to_ankle", "segment_lengths.shank"),
    "ankle_to_sole": ("segment_lengths.ankle_to_sole", "segments.ankle_to_sole", "segment_lengths.ankle_height"),
    "ankle_forward_of_hip": ("segment_lengths.ankle_forward_of_hip",),
    "foot_front": ("segment_lengths.foot_front", "segments.foot_front"),
    "foot_back": ("segment_lengths.foot_back", "segments.foot_back"),
    "foot_half_width": ("segment_lengths.foot_half_width", "segments.foot_half_width"),
    "standing_semantic_pos": ("standing_semantic_pos", "standing_semantic"),
    "standing_motor_pos": ("standing_motor_pos",),
    "usd_sha256": ("usd_sha256", "provenance.usd_sha256", "usd.sha256"),
}

# Defaults used ONLY for optional foot-shape fields (clearly reported in `defaults_used`).
OPTIONAL_DEFAULTS: dict[str, float] = {
    "foot_front": 0.18, "foot_back": 0.11, "foot_half_width": 0.04, "ankle_forward_of_hip": 0.0,
}


class SemanticMapLike(Protocol):
    def semantic_to_motor(self, q_sem: np.ndarray) -> Any: ...

    def motor_to_semantic(self, q_motor: np.ndarray) -> Any: ...


def _get(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def _pose_to_matrix(v: Any) -> np.ndarray:
    """Accept {pos, quat_wxyz} | {pos, quat} | {translation, rotation_wxyz} | 4x4 list."""
    if isinstance(v, (list, tuple)) and np.asarray(v).shape == (4, 4):
        return np.asarray(v, dtype=np.float64)
    if isinstance(v, dict):
        pos = next((v[k] for k in ("pos", "position", "translation", "p") if k in v), None)
        quat = next((v[k] for k in ("quat_wxyz", "quat", "rotation_wxyz", "q_wxyz") if k in v), None)
        if pos is not None and quat is not None:
            m = np.eye(4)
            m[:3, :3] = quat_to_matrix(np.asarray(quat, dtype=np.float64))
            m[:3, 3] = pos
            return m
        if "matrix" in v:
            return np.asarray(v["matrix"], dtype=np.float64)
    raise ValueError(f"cannot parse pose {v!r}")


@dataclass
class CalibrationView:
    path: Path
    raw: dict[str, Any]
    semantic_map: SemanticMapLike
    semantic_map_impl: str
    is_mock: bool
    standing_hip_height: float
    root_T_pelvis: np.ndarray
    hip_width: float
    hip_to_knee: float
    knee_to_ankle: float
    ankle_to_sole: float
    ankle_forward_of_hip: float
    foot_front: float
    foot_back: float
    foot_half_width: float
    standing_semantic_pos: np.ndarray
    standing_motor_pos: np.ndarray
    usd_sha256: str
    defaults_used: list[str] = field(default_factory=list)

    @property
    def pelvis_T_root(self) -> np.ndarray:
        return np.linalg.inv(self.root_T_pelvis)

    def semantic_to_motor(self, q_sem: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Contract call; normalises the return to ``(q_motor, info_dict)``."""
        q = np.asarray(q_sem, dtype=np.float64)
        try:  # real SemanticMap: returns (q_motor, SaturationReport) only when asked
            out = self.semantic_map.semantic_to_motor(q, return_report=True)
        except TypeError:
            out = self.semantic_map.semantic_to_motor(q)
        if isinstance(out, tuple):
            q_motor, info = out[0], (out[1] if len(out) > 1 else {})
        else:
            q_motor, info = out, {}
        return np.asarray(q_motor, dtype=np.float64), _info_as_dict(info)

    def motor_to_semantic(self, q_motor: np.ndarray) -> np.ndarray:
        out = self.semantic_map.motor_to_semantic(np.asarray(q_motor, dtype=np.float64))
        return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float64)

    def provenance(self) -> dict[str, Any]:
        import hashlib

        try:
            sha = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        except OSError:
            sha = None
        return {
            "calibration_path": str(self.path).replace("\\", "/"),
            # review fix 2026-09-24: which calibration (bytes) and which ankle plant produced the motor columns;
            # tools/settle_motion.py refuses a sidecar from the other ankle variant
            "calibration_sha256": sha,
            "calibration_created": self.raw.get("created"),
            "authored_ankle_tierods": bool(self.raw.get("authored_ankle_tierods", True)),
            "plant_variant": self.raw.get("plant_variant"),
            "calibration_schema": self.raw.get("schema"),
            "calibration_is_mock": self.is_mock,
            "semantic_map_impl": self.semantic_map_impl,
            "usd_sha256": self.usd_sha256,
            "calibration_defaults_used": self.defaults_used,
        }


def _info_as_dict(info: Any) -> dict[str, Any]:
    if isinstance(info, dict):
        return info
    if hasattr(info, "__dict__"):
        return dict(vars(info))
    return {"raw": info}


def _load_semantic_map(path: Path, raw: dict[str, Any]) -> tuple[SemanticMapLike, str]:
    is_mock = bool(raw.get("MOCK", False))
    real_err: Exception | None = None
    try:
        mod = importlib.import_module("dropbear_wbc.kinematics.semantic")
        return mod.SemanticMap.load(path), "dropbear_wbc.kinematics.semantic.SemanticMap"
    except Exception as exc:  # noqa: BLE001
        real_err = exc
    if is_mock:
        from .mock_semantic import MockSemanticMap

        return MockSemanticMap.load(path), f"MockSemanticMap (real SemanticMap unavailable: {type(real_err).__name__}: {real_err})"
    raise RuntimeError(
        f"dropbear_wbc.kinematics.semantic.SemanticMap could not load {path}: {real_err!r}"
    ) from real_err


def load_calibration(path: Path | str | None = None, *, allow_mock: bool = False) -> CalibrationView:
    """Load the calibration view. ``allow_mock`` must be True to accept a JSON flagged ``"MOCK": true``."""
    path = Path(path) if path is not None else DEFAULT_CALIBRATION
    if not path.is_file():
        raise FileNotFoundError(
            f"calibration {path} not found (the calibrate_settle builder writes it); for development pass "
            f"--calibration {MOCK_CALIBRATION} --allow-mock"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    is_mock = bool(raw.get("MOCK", False))
    if is_mock and not allow_mock:
        raise ValueError(f"{path} is a MOCK calibration; pass allow_mock=True / --allow-mock explicitly")
    for key, expected in (("semantic_names", SEMANTIC_NAMES), ("motor_names", MOTOR_NAMES)):
        if key in raw and tuple(raw[key]) != expected:
            raise ValueError(f"{path}: {key} differs from the contract order")
    smap, impl = _load_semantic_map(path, raw)
    defaults: list[str] = []

    def find(name: str, required: bool = True) -> Any:
        for p in FIELD_PATHS[name]:
            try:
                return _get(raw, p)
            except KeyError:
                continue
        if name in OPTIONAL_DEFAULTS:
            defaults.append(f"{name}={OPTIONAL_DEFAULTS[name]}")
            return OPTIONAL_DEFAULTS[name]
        if required:
            raise KeyError(f"{path}: none of {FIELD_PATHS[name]} present (needed for '{name}')")
        return None

    root_T_pelvis_raw = find("root_T_pelvis", required=False)
    if root_T_pelvis_raw is not None:
        root_T_pelvis = _pose_to_matrix(root_T_pelvis_raw)
    else:
        root_T_pelvis = np.linalg.inv(_pose_to_matrix(find("pelvis_T_root")))

    standing_motor = np.asarray(find("standing_motor_pos"), dtype=np.float64)
    standing_sem_raw = find("standing_semantic_pos", required=False)
    if standing_sem_raw is None:
        q = smap.motor_to_semantic(standing_motor)
        standing_sem = np.asarray(q[0] if isinstance(q, tuple) else q, dtype=np.float64)
    else:
        standing_sem = np.asarray(standing_sem_raw, dtype=np.float64)

    return CalibrationView(
        path=path,
        raw=raw,
        semantic_map=smap,
        semantic_map_impl=impl,
        is_mock=is_mock,
        standing_hip_height=float(find("standing_hip_height")),
        root_T_pelvis=root_T_pelvis,
        hip_width=float(find("hip_width")),
        hip_to_knee=float(find("hip_to_knee")),
        knee_to_ankle=float(find("knee_to_ankle")),
        ankle_to_sole=float(find("ankle_to_sole")),
        ankle_forward_of_hip=float(find("ankle_forward_of_hip")),
        foot_front=float(find("foot_front")),
        foot_back=float(find("foot_back")),
        foot_half_width=float(find("foot_half_width")),
        standing_semantic_pos=standing_sem,
        standing_motor_pos=standing_motor,
        usd_sha256=str(find("usd_sha256", required=False) or ""),
        defaults_used=defaults,
    )
