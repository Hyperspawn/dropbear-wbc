"""Contract ``dropbear-motion-npz-v1`` reader/writer/validator (numpy only; no Isaac imports).

Layout (``docs/CONTRACTS.md`` section 4), all frames at the policy rate::

    fps                                   ()        50
    joint_pos, joint_vel                  (T, J)    ALL articulation joints, Isaac order [rad, m, rad/s, m/s]
    body_pos_w                            (T, B, 3) world frame, ground z = 0, env origin (0, 0) [m]
    body_quat_w                           (T, B, 4) wxyz
    body_lin_vel_w, body_ang_vel_w        (T, B, 3) world frame [m/s, rad/s]
    joint_names (J,), body_names (B,)     Isaac articulation order
    motor_names                           (22,)     motor contract order
    closure_residual_m                    (T,)      worst loop-closure anchor gap [m]
    meta                                  ()        JSON string

Interpretation used by the tracking env (Isaac Lab 2.x semantics): ``body_pos_w``/``body_quat_w`` are
*link frame* poses and ``body_lin_vel_w`` is the *centre-of-mass* linear velocity of each body (what
``Articulation.data.body_lin_vel_w`` returns and what ``write_root_state_to_sim`` expects for the root).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA = "dropbear-motion-npz-v1"
REQUIRED_KEYS: tuple[str, ...] = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
    "motor_names",
    "closure_residual_m",
    "meta",
)


class MotionFormatError(ValueError):
    """The NPZ violates the contract or does not match the live articulation (fail closed)."""


@dataclass
class MotionArrays:
    """In-memory contract motion. Arrays are float32 except names (lists of str)."""

    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    joint_names: list[str]
    body_names: list[str]
    motor_names: list[str]
    closure_residual_m: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return (self.num_frames - 1) / self.fps


def _names(arr: np.ndarray, key: str) -> list[str]:
    if arr.ndim != 1:
        raise MotionFormatError(f"{key} must be 1-D, got shape {arr.shape}")
    return [str(x) for x in arr.tolist()]


def check_arrays(m: MotionArrays) -> None:
    """Internal consistency checks (shapes, finiteness, unit quaternions, fps, name uniqueness)."""
    t = m.joint_pos.shape[0]
    j, b = len(m.joint_names), len(m.body_names)
    expect = {
        "joint_pos": (t, j),
        "joint_vel": (t, j),
        "body_pos_w": (t, b, 3),
        "body_quat_w": (t, b, 4),
        "body_lin_vel_w": (t, b, 3),
        "body_ang_vel_w": (t, b, 3),
        "closure_residual_m": (t,),
    }
    if t < 2:
        raise MotionFormatError(f"need at least 2 frames, got {t}")
    for key, shape in expect.items():
        arr = getattr(m, key)
        if arr.shape != shape:
            raise MotionFormatError(f"{key} has shape {arr.shape}, expected {shape}")
        if not np.all(np.isfinite(arr)):
            raise MotionFormatError(f"{key} contains non-finite values")
    norms = np.linalg.norm(m.body_quat_w, axis=-1)
    if np.max(np.abs(norms - 1.0)) > 1e-3:
        raise MotionFormatError(f"body_quat_w not unit (max |norm-1| = {np.max(np.abs(norms - 1.0)):.2e})")
    if not (np.isfinite(m.fps) and m.fps > 0):
        raise MotionFormatError(f"invalid fps {m.fps}")
    for key in ("joint_names", "body_names", "motor_names"):
        names = getattr(m, key)
        if len(set(names)) != len(names):
            raise MotionFormatError(f"{key} has duplicates")
    missing = [n for n in m.motor_names if n not in m.joint_names]
    if missing:
        raise MotionFormatError(f"motor_names not in joint_names: {missing}")


def load_motion_npz(path: str | Path) -> MotionArrays:
    """Load and self-check a contract NPZ (does not compare against an articulation)."""
    p = Path(path)
    if not p.is_file():
        raise MotionFormatError(f"motion file not found: {p}")
    with np.load(p, allow_pickle=False) as data:
        missing = [k for k in REQUIRED_KEYS if k not in data.files]
        if missing:
            raise MotionFormatError(f"{p}: missing keys {missing}")
        meta_raw = str(data["meta"])
        try:
            meta = json.loads(meta_raw) if meta_raw else {}
        except json.JSONDecodeError as exc:
            raise MotionFormatError(f"{p}: meta is not JSON: {exc}") from exc
        m = MotionArrays(
            fps=float(data["fps"]),
            joint_pos=np.asarray(data["joint_pos"], dtype=np.float32),
            joint_vel=np.asarray(data["joint_vel"], dtype=np.float32),
            body_pos_w=np.asarray(data["body_pos_w"], dtype=np.float32),
            body_quat_w=np.asarray(data["body_quat_w"], dtype=np.float32),
            body_lin_vel_w=np.asarray(data["body_lin_vel_w"], dtype=np.float32),
            body_ang_vel_w=np.asarray(data["body_ang_vel_w"], dtype=np.float32),
            joint_names=_names(data["joint_names"], "joint_names"),
            body_names=_names(data["body_names"], "body_names"),
            motor_names=_names(data["motor_names"], "motor_names"),
            closure_residual_m=np.asarray(data["closure_residual_m"], dtype=np.float32),
            meta=meta,
        )
    check_arrays(m)
    return m


def validate_against_articulation(
    m: MotionArrays,
    joint_names: Sequence[str],
    body_names: Sequence[str],
    motor_names: Sequence[str],
    expected_fps: float | None = None,
    fps_tol: float = 1e-6,
) -> None:
    """Fail closed unless the NPZ names/order match the live articulation exactly.

    Raises:
        MotionFormatError: on any mismatch (order matters: rows are written straight into the sim).
    """
    if list(m.joint_names) != list(joint_names):
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(m.joint_names, joint_names)) if a != b][:5]
        raise MotionFormatError(
            f"joint_names mismatch: npz J={len(m.joint_names)} vs articulation J={len(joint_names)}; first diffs {diff}"
        )
    if list(m.body_names) != list(body_names):
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(m.body_names, body_names)) if a != b][:5]
        raise MotionFormatError(
            f"body_names mismatch: npz B={len(m.body_names)} vs articulation B={len(body_names)}; first diffs {diff}"
        )
    if list(m.motor_names) != list(motor_names):
        raise MotionFormatError(f"motor_names {m.motor_names} != contract {list(motor_names)}")
    if expected_fps is not None and abs(m.fps - expected_fps) > fps_tol:
        raise MotionFormatError(f"fps {m.fps} != policy rate {expected_fps}")


def expected_usd_sha256() -> str | None:
    """SHA-256 the motion must have been produced on: the contract plant's, or ``None`` (skip the check) when
    ``$DROPBEAR_USD`` overrides the USD (the NPZ then only has to match by names)."""
    import os

    if os.environ.get("DROPBEAR_USD"):
        return None
    from dropbear_wbc.robots.dropbear_names import USD_SHA256

    return USD_SHA256


def npz_authored_ankle(meta: dict) -> bool:
    """Ankle plant variant an NPZ was produced on (docs/CONTRACTS.md 0.2).

    ``meta.authored_ankle_tierods`` (written by ``tools/settle_motion.py`` and ``tools/make_static_npz.py`` since the
    0.2 adoption); NPZs without the key predate it and were produced on the authored (revolute) ankle.
    """
    return bool(meta.get("authored_ankle_tierods", True))


VALIDATION_SCHEMA = "dropbear-motion-validation-v1"
"""Per-clip verdict file ``<clip>.validation.json`` written next to the NPZ by ``tools/validate_motion_npz.py``."""


def validation_sidecar_path(npz_path: str | Path) -> Path:
    """``data/motions/<src>/<clip>.npz`` -> ``data/motions/<src>/<clip>.validation.json``."""
    p = Path(npz_path)
    return p.with_name(p.stem + ".validation.json")


def load_validation_verdict(npz_path: str | Path) -> dict[str, Any] | None:
    """Read the validator's verdict for ``npz_path`` (``None`` when no ``<clip>.validation.json`` exists).

    Returns ``{path, verdict, reasons, created, npz_sha256, stale}``. ``stale`` is ``True`` when the verdict was
    written for other NPZ bytes (``npz_sha256`` recorded and different) or, for verdict files that predate the
    ``npz_sha256`` key, when the NPZ is newer than the verdict file; ``None`` when it cannot be decided.

    Raises:
        MotionFormatError: if the file exists but is not a readable ``dropbear-motion-validation-v1`` JSON
            (fail closed: an unreadable verdict must not silently count as "no verdict").
    """
    import hashlib
    import os

    npz = Path(npz_path)
    vpath = validation_sidecar_path(npz)
    if not vpath.is_file():
        return None
    try:
        data = json.loads(vpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MotionFormatError(f"{vpath}: unreadable motion validation verdict: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != VALIDATION_SCHEMA or "verdict" not in data:
        raise MotionFormatError(f"{vpath}: not a {VALIDATION_SCHEMA} verdict file")
    stale: bool | None = None
    recorded = data.get("npz_sha256")
    if recorded and npz.is_file():
        stale = hashlib.sha256(npz.read_bytes()).hexdigest() != recorded
    elif npz.is_file():
        stale = os.path.getmtime(npz) > os.path.getmtime(vpath) + 1.0
    return {"path": str(vpath).replace("\\", "/"), "verdict": data.get("verdict"),
            "reasons": list(data.get("reasons") or []), "created": data.get("created"),
            "npz_sha256": recorded, "stale": stale}


def validate_provenance(
    m: MotionArrays, expected_usd_sha256: str | None = None, allow_rejected: bool = False,
    expected_authored_ankle: bool | None = None, validation: dict[str, Any] | None = None,
) -> None:
    """Fail closed on NPZ metadata that says the motion must not be used.

    * ``meta.schema``, if present, must be :data:`SCHEMA`;
    * ``meta.status == "rejected"`` (``tools/settle_motion.py`` quality gate: too many frames with closure
      residual above threshold / not converged / reference jumps) is refused unless ``allow_rejected``;
    * a ``validation`` verdict (:func:`load_validation_verdict`, the sibling ``<clip>.validation.json`` of
      ``tools/validate_motion_npz.py``) of ``"rejected"`` is refused unless ``allow_rejected`` -- also when the
      verdict is stale (re-run the validator instead of trusting an old verdict either way);
    * ``meta.usd_sha256``, if present, must equal ``expected_usd_sha256`` (when that is given);
    * the ankle plant variant (:func:`npz_authored_ankle`) must equal ``expected_authored_ankle`` (when given):
      the settled passive ankle joints of one variant violate the loop closures of the other.

    Raises:
        MotionFormatError: on any violation.
    """
    meta = m.meta or {}
    schema = meta.get("schema")
    if schema is not None and schema != SCHEMA:
        raise MotionFormatError(f"meta.schema {schema!r} != {SCHEMA!r}")
    if meta.get("status") == "rejected" and not allow_rejected:
        closure = meta.get("closure", {})
        raise MotionFormatError(
            "motion NPZ was REJECTED by its producer (meta.status='rejected'; "
            f"flagged_fraction={closure.get('flagged_fraction')}, max closure residual={closure.get('max_m')} m; "
            f"reasons={meta.get('status_reasons')}); set allow_rejected_motion=True only for debugging"
        )
    if validation and validation.get("verdict") == "rejected" and not allow_rejected:
        raise MotionFormatError(
            f"motion NPZ was REJECTED by tools/validate_motion_npz.py ({validation.get('path')}"
            f"{', verdict possibly STALE' if validation.get('stale') else ''}): {validation.get('reasons')}; "
            "an exploratory run on it needs allow_rejected_motion=True (scripts/train.py --allow_rejected_motion), "
            "which is recorded in run_info.json and in the export sidecar"
        )
    sha = meta.get("usd_sha256")
    if expected_usd_sha256 and sha and sha != expected_usd_sha256:
        raise MotionFormatError(f"motion was settled on USD sha {sha[:12]}..., expected {expected_usd_sha256[:12]}...")
    if expected_authored_ankle is not None and npz_authored_ankle(meta) != bool(expected_authored_ankle):
        variant = lambda a: "authored revolute" if a else "spherical (CONTRACTS 0.2 default)"  # noqa: E731
        raise MotionFormatError(
            f"motion was produced on the {variant(npz_authored_ankle(meta))} ankle tie rods, but the plant uses the "
            f"{variant(expected_authored_ankle)} ones (docs/CONTRACTS.md 0.2): re-settle the clip "
            "(tools/settle_motion.py / tools/make_static_npz.py) or set $DROPBEAR_AUTHORED_ANKLE to match"
        )


def save_motion_npz(path: str | Path, m: MotionArrays) -> Path:
    """Write a contract NPZ (after :func:`check_arrays`). Returns the path."""
    check_arrays(m)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(m.meta)
    meta.setdefault("schema", SCHEMA)
    np.savez(
        p,
        fps=np.array(m.fps, dtype=np.float64),
        joint_pos=m.joint_pos.astype(np.float32),
        joint_vel=m.joint_vel.astype(np.float32),
        body_pos_w=m.body_pos_w.astype(np.float32),
        body_quat_w=m.body_quat_w.astype(np.float32),
        body_lin_vel_w=m.body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=m.body_ang_vel_w.astype(np.float32),
        joint_names=np.array(m.joint_names),
        body_names=np.array(m.body_names),
        motor_names=np.array(m.motor_names),
        closure_residual_m=m.closure_residual_m.astype(np.float32),
        meta=np.array(json.dumps(meta)),
    )
    return p
