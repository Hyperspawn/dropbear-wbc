"""Default (standing) motor pose resolution. Pure Python: no Isaac/torch imports.

Contract section 5: the action offset / default pose is ``standing_motor_pos`` from the semantic
calibration JSON (``data/calibration/dropbear_semantic_calibration.json``, schema
``dropbear-semantic-calibration-v1``) when available; otherwise the legacy ``DROPBEAR_CFG`` pose.
Angles in radians, motor-contract order.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from .dropbear_names import (
    LEGACY_DEFAULT_MOTOR_POS,
    MOTOR_HARD_LIMITS_DEG,
    MOTOR_NAMES,
    REPO_ROOT,
    USD_SHA256,
    authored_ankle_requested,
)

CALIBRATION_SCHEMA: str = "dropbear-semantic-calibration-v1"
DEFAULT_CALIBRATION_PATH: Path = REPO_ROOT / "data" / "calibration" / "dropbear_semantic_calibration.json"


class CalibrationError(ValueError):
    """Raised when a calibration JSON exists but its standing pose is unusable (fail closed)."""


@dataclass(frozen=True)
class DefaultPose:
    """Resolved default motor pose.

    Attributes:
        motor_pos: Motor name -> angle [rad] for all 22 motors.
        source: ``"calibration:<path>"`` or ``"legacy_DROPBEAR_CFG"``.
        info: Calibration provenance (``created``, ``usd_sha256``, ``plant_variant``, ``sha256`` of the JSON);
            empty for the legacy pose.
    """

    motor_pos: dict[str, float]
    source: str
    info: dict = field(default_factory=dict)

    def as_tuple(self) -> tuple[float, ...]:
        """Angles in motor-contract order."""
        return tuple(self.motor_pos[name] for name in MOTOR_NAMES)


def calibration_path() -> Path:
    """``$DROPBEAR_CALIBRATION_JSON`` if set, else :data:`DEFAULT_CALIBRATION_PATH`."""
    env = os.environ.get("DROPBEAR_CALIBRATION_JSON")
    return Path(env) if env else DEFAULT_CALIBRATION_PATH


def _validate_pose(pose: dict[str, float], origin: str) -> dict[str, float]:
    missing = [n for n in MOTOR_NAMES if n not in pose]
    extra = [n for n in pose if n not in MOTOR_NAMES]
    if missing or extra:
        raise CalibrationError(f"{origin}: standing_motor_pos missing={missing} unexpected={extra}")
    out: dict[str, float] = {}
    for name in MOTOR_NAMES:
        value = float(pose[name])
        if not math.isfinite(value):
            raise CalibrationError(f"{origin}: {name} is not finite ({value})")
        lo, hi = (math.radians(v) for v in MOTOR_HARD_LIMITS_DEG[name])
        if not (lo - 1e-6 <= value <= hi + 1e-6):
            raise CalibrationError(
                f"{origin}: {name}={value:.4f} rad outside hard limits [{lo:.4f}, {hi:.4f}]"
            )
        out[name] = value
    return out


def _read_calibration(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"{p}: invalid JSON: {exc}") from exc


def calibration_info(path: str | Path | None = None) -> dict:
    """Provenance of a calibration JSON (empty dict if it does not exist)."""
    p = Path(path) if path is not None else calibration_path()
    if not p.is_file():
        return {}
    data = _read_calibration(p)
    return {
        "path": p.as_posix(),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "created": data.get("created"),
        "usd_sha256": data.get("usd_sha256"),
        "plant_variant": data.get("plant_variant"),
        "authored_ankle_tierods": calibration_authored_ankle(data),
        "schema": data.get("schema"),
    }


def calibration_authored_ankle(data: dict) -> bool:
    """Ankle variant a calibration was measured on (docs/CONTRACTS.md 0.2). Calibrations without the key predate the
    0.2 adoption and were measured on the authored (revolute) tie rods (same rule as ``motion_npz.npz_authored_ankle``)."""
    return bool(data.get("authored_ankle_tierods", True))


def load_standing_motor_pos(path: str | Path | None = None,
                            expected_authored_ankle: bool | None = None) -> dict[str, float] | None:
    """Read ``standing_motor_pos`` from a calibration JSON.

    Accepts a 22-list in motor-contract order or a ``{motor_name: rad}`` dict.

    Returns:
        The validated pose, or ``None`` if the file does not exist.

    Raises:
        CalibrationError: if the file exists but is malformed, has the wrong schema, lists motors in a
            different order (``motor_names``), was built on a different USD (``usd_sha256``, checked unless
            ``$DROPBEAR_USD`` overrides the plant), was measured on the other ankle variant than
            ``expected_authored_ankle`` (default: the plant ``$DROPBEAR_AUTHORED_ANKLE`` selects, CONTRACTS 0.2),
            is a DIAGNOSTIC plant variant, or the pose is incomplete, non-finite or outside the authored hard limits.
    """
    p = Path(path) if path is not None else calibration_path()
    if not p.is_file():
        return None
    data = _read_calibration(p)
    schema = data.get("schema")
    if schema is not None and schema != CALIBRATION_SCHEMA:
        raise CalibrationError(f"{p}: schema {schema!r} != {CALIBRATION_SCHEMA!r}")
    names = data.get("motor_names")
    if names is not None and list(names) != list(MOTOR_NAMES):
        raise CalibrationError(f"{p}: motor_names differ from the motor contract (order matters): {names}")
    sha = data.get("usd_sha256")
    if sha and not os.environ.get("DROPBEAR_USD") and sha != USD_SHA256:
        raise CalibrationError(f"{p}: calibrated on USD sha {sha[:12]}..., contract plant is {USD_SHA256[:12]}...")
    # plant variant (review fix 2026-09-24): the same fail-closed rule as for motion NPZs (CONTRACTS 5.1)
    variant = str(data.get("plant_variant") or "")
    if "DIAGNOSTIC" in variant.upper():
        raise CalibrationError(f"{p}: DIAGNOSTIC plant variant ({variant[:80]}...) is not a contract calibration")
    want = authored_ankle_requested() if expected_authored_ankle is None else bool(expected_authored_ankle)
    have = calibration_authored_ankle(data)
    if have != want:
        name = lambda a: "authored revolute" if a else "spherical (CONTRACTS 0.2 default)"  # noqa: E731
        raise CalibrationError(f"{p}: measured on the {name(have)} ankle tie rods, but the plant uses the {name(want)} "
                               "ones (docs/CONTRACTS.md 0.2): use the matching calibration or set $DROPBEAR_AUTHORED_ANKLE")
    raw = data.get("standing_motor_pos")
    if raw is None:
        raise CalibrationError(f"{p}: no 'standing_motor_pos'")
    if isinstance(raw, dict):
        pose = {str(k): float(v) for k, v in raw.items()}
    elif isinstance(raw, (list, tuple)):
        if len(raw) != len(MOTOR_NAMES):
            raise CalibrationError(f"{p}: standing_motor_pos has {len(raw)} entries, expected {len(MOTOR_NAMES)}")
        pose = {name: float(v) for name, v in zip(MOTOR_NAMES, raw)}
    else:
        raise CalibrationError(f"{p}: standing_motor_pos must be a list or dict, got {type(raw).__name__}")
    return _validate_pose(pose, str(p))


def resolve_default_pose(path: str | Path | None = None, use_calibration: bool = True) -> DefaultPose:
    """Return the calibration standing pose if present (and ``use_calibration``), else the legacy pose.

    ``$DROPBEAR_CALIBRATION_JSON=none`` forces the legacy pose (used to replay a policy trained without one).
    """
    if path is None and os.environ.get("DROPBEAR_CALIBRATION_JSON", "").strip().lower() == "none":
        use_calibration = False
    if use_calibration:
        p = Path(path) if path is not None else calibration_path()
        pose = load_standing_motor_pos(p)
        if pose is not None:
            return DefaultPose(motor_pos=pose, source=f"calibration:{p.as_posix()}", info=calibration_info(p))
    return DefaultPose(motor_pos=dict(LEGACY_DEFAULT_MOTOR_POS), source="legacy_DROPBEAR_CFG")
