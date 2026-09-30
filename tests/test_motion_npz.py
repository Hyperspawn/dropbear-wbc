"""CPU tests for the contract NPZ reader/validator (dropbear_wbc.tasks.tracking.motion_npz)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES  # noqa: E402
from dropbear_wbc.tasks.tracking.motion_npz import (  # noqa: E402
    MotionArrays,
    MotionFormatError,
    load_motion_npz,
    save_motion_npz,
    validate_against_articulation,
)

JOINTS = list(MOTOR_NAMES) + ["head_LeadScrew1", "LL_Revolute28", "LL_Revolute115:0"]
BODIES = ["world", "head_5mm_ujoint_base__5__1", "LL_skateboard_bearing_left_2"]


def make(t: int = 5) -> MotionArrays:
    rng = np.random.default_rng(0)
    quat = np.zeros((t, len(BODIES), 4), dtype=np.float32)
    quat[..., 0] = 1.0
    return MotionArrays(
        fps=50.0,
        joint_pos=rng.normal(size=(t, len(JOINTS))).astype(np.float32),
        joint_vel=np.zeros((t, len(JOINTS)), dtype=np.float32),
        body_pos_w=rng.normal(size=(t, len(BODIES), 3)).astype(np.float32),
        body_quat_w=quat,
        body_lin_vel_w=np.zeros((t, len(BODIES), 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((t, len(BODIES), 3), dtype=np.float32),
        joint_names=list(JOINTS),
        body_names=list(BODIES),
        motor_names=list(MOTOR_NAMES),
        closure_residual_m=np.full((t,), 1e-4, dtype=np.float32),
        meta={"note": "unit test"},
    )


def test_roundtrip(tmp_path):
    p = save_motion_npz(tmp_path / "m.npz", make())
    m = load_motion_npz(p)
    assert m.num_frames == 5 and m.fps == 50.0
    assert m.joint_names == JOINTS and m.body_names == BODIES and m.motor_names == list(MOTOR_NAMES)
    assert m.meta["schema"] == "dropbear-motion-npz-v1"
    validate_against_articulation(m, JOINTS, BODIES, MOTOR_NAMES, expected_fps=50.0)


def test_fail_closed_on_name_order(tmp_path):
    m = load_motion_npz(save_motion_npz(tmp_path / "m.npz", make()))
    with pytest.raises(MotionFormatError, match="joint_names"):
        validate_against_articulation(m, JOINTS[::-1], BODIES, MOTOR_NAMES)
    with pytest.raises(MotionFormatError, match="body_names"):
        validate_against_articulation(m, JOINTS, BODIES[:-1], MOTOR_NAMES)
    with pytest.raises(MotionFormatError, match="motor_names"):
        validate_against_articulation(m, JOINTS, BODIES, list(MOTOR_NAMES)[::-1])
    with pytest.raises(MotionFormatError, match="fps"):
        validate_against_articulation(m, JOINTS, BODIES, MOTOR_NAMES, expected_fps=30.0)


def test_rejects_bad_arrays(tmp_path):
    bad = make()
    bad.body_quat_w[0, 0] = [2.0, 0, 0, 0]
    with pytest.raises(MotionFormatError, match="unit"):
        save_motion_npz(tmp_path / "q.npz", bad)
    bad = make()
    bad.joint_pos[1, 2] = np.nan
    with pytest.raises(MotionFormatError, match="non-finite"):
        save_motion_npz(tmp_path / "n.npz", bad)
    bad = make()
    bad.joint_vel = bad.joint_vel[:, :-1]
    with pytest.raises(MotionFormatError, match="shape"):
        save_motion_npz(tmp_path / "s.npz", bad)


def test_missing_key(tmp_path):
    p = tmp_path / "partial.npz"
    np.savez(p, fps=50.0, joint_pos=np.zeros((2, 1)))
    with pytest.raises(MotionFormatError, match="missing keys"):
        load_motion_npz(p)


# ---------------------------------------------------------------------------------------------------------
# NPZs written the way tools/settle_motion.py writes them (np.savez_compressed, 0-d fps, extra keys, meta
# with status/usd_sha256), using the LIVE articulation names recorded in the smoke NPZ when it exists.
# ---------------------------------------------------------------------------------------------------------
from dropbear_wbc.robots.dropbear_names import USD_SHA256  # noqa: E402
from dropbear_wbc.tasks.tracking.motion_npz import expected_usd_sha256, validate_provenance  # noqa: E402

SMOKE_NPZ = REPO / "data" / "motions" / "smoke" / "dropbear_static_stand.npz"


def _live_names() -> tuple[list[str], list[str]]:
    if SMOKE_NPZ.is_file():
        m = load_motion_npz(SMOKE_NPZ)
        return m.joint_names, m.body_names
    return list(JOINTS), list(BODIES)


def write_settle_style(path: Path, status: str = "ok", usd_sha: str = USD_SHA256, t: int = 6) -> Path:
    import json

    joints, bodies = _live_names()
    quat = np.zeros((t, len(bodies), 4), dtype=np.float32)
    quat[..., 0] = 1.0
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    meta = {"schema": "dropbear-motion-npz-v1", "tool": "settle_motion 1.0", "status": status, "usd_sha256": usd_sha,
            "closure": {"flagged_fraction": 0.5 if status == "rejected" else 0.0, "max_m": 0.01}}
    np.savez_compressed(
        path, fps=np.asarray(50.0),
        joint_pos=f32(np.zeros((t, len(joints)))), joint_vel=f32(np.zeros((t, len(joints)))),
        body_pos_w=f32(np.zeros((t, len(bodies), 3))), body_quat_w=quat,
        body_lin_vel_w=f32(np.zeros((t, len(bodies), 3))), body_ang_vel_w=f32(np.zeros((t, len(bodies), 3))),
        joint_names=np.array(joints), body_names=np.array(bodies), motor_names=np.array(MOTOR_NAMES),
        closure_residual_m=f32(np.full(t, 1e-3)), meta=np.array(json.dumps(meta)),
        closure_residual_rad=f32(np.zeros(t)), frame_flags=np.zeros(t, dtype=bool), ground_dz=f32(np.zeros(t)),
        motor_target=f32(np.zeros((t, 22))), contact=np.ones((t, 2), dtype=bool),
    )
    return path


def test_settle_motion_style_npz_loads_and_matches_live_names(tmp_path, monkeypatch):
    monkeypatch.delenv("DROPBEAR_USD", raising=False)
    p = write_settle_style(tmp_path / "clip.npz")
    m = load_motion_npz(p)
    joints, bodies = _live_names()
    validate_against_articulation(m, joints, bodies, MOTOR_NAMES, expected_fps=50.0, fps_tol=1e-3)
    validate_provenance(m, expected_usd_sha256())
    assert m.fps == 50.0 and m.num_frames == 6
    if SMOKE_NPZ.is_file():  # the live articulation: 91 joints, 90 bodies (contract 0.1)
        assert len(joints) == 91 and len(bodies) == 90 and bodies[0] == "world"


def test_provenance_fail_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("DROPBEAR_USD", raising=False)
    rejected = load_motion_npz(write_settle_style(tmp_path / "r.npz", status="rejected"))
    with pytest.raises(MotionFormatError, match="REJECTED"):
        validate_provenance(rejected, expected_usd_sha256())
    validate_provenance(rejected, expected_usd_sha256(), allow_rejected=True)
    other = load_motion_npz(write_settle_style(tmp_path / "o.npz", usd_sha="0" * 64))
    with pytest.raises(MotionFormatError, match="USD sha"):
        validate_provenance(other, expected_usd_sha256())
    monkeypatch.setenv("DROPBEAR_USD", "X:/some/other.usd")  # plant override: names must match, sha not checked
    assert expected_usd_sha256() is None
    validate_provenance(other, expected_usd_sha256())
    wrong_schema = make()
    wrong_schema.meta = {"schema": "beyondmimic-npz"}
    with pytest.raises(MotionFormatError, match="schema"):
        validate_provenance(wrong_schema)


# ---- validation verdict (<clip>.validation.json) fail-closed (review fix 2026-09-24) -------------------------------
def _write_verdict(npz_path, verdict, **extra):
    import json

    from dropbear_wbc.tasks.tracking.motion_npz import VALIDATION_SCHEMA, validation_sidecar_path

    data = {"schema": VALIDATION_SCHEMA, "verdict": verdict, "reasons": ["test reason"], "created": "t", **extra}
    validation_sidecar_path(npz_path).write_text(json.dumps(data))


def test_validation_verdict_rejected_fails_closed(tmp_path):
    import hashlib

    from dropbear_wbc.tasks.tracking.motion_npz import load_validation_verdict, validate_provenance

    p = save_motion_npz(tmp_path / "clip.npz", make())
    assert load_validation_verdict(p) is None
    _write_verdict(p, "rejected", npz_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    v = load_validation_verdict(p)
    assert v["verdict"] == "rejected" and v["stale"] is False and v["reasons"] == ["test reason"]
    m = load_motion_npz(p)
    with pytest.raises(MotionFormatError, match="REJECTED by tools/validate_motion_npz.py"):
        validate_provenance(m, validation=v)
    validate_provenance(m, validation=v, allow_rejected=True)  # explicit exploratory acceptance
    _write_verdict(p, "accepted")
    validate_provenance(m, validation=load_validation_verdict(p))


def test_validation_verdict_stale_and_malformed(tmp_path):
    from dropbear_wbc.tasks.tracking.motion_npz import (
        load_validation_verdict,
        validate_provenance,
        validation_sidecar_path,
    )

    p = save_motion_npz(tmp_path / "clip.npz", make())
    _write_verdict(p, "rejected", npz_sha256="0" * 64)  # verdict for other bytes
    v = load_validation_verdict(p)
    assert v["stale"] is True
    with pytest.raises(MotionFormatError, match="STALE"):  # a stale 'rejected' still refuses
        validate_provenance(load_motion_npz(p), validation=v)
    validation_sidecar_path(p).write_text("{not json")
    with pytest.raises(MotionFormatError, match="unreadable"):
        load_validation_verdict(p)
    validation_sidecar_path(p).write_text('{"schema": "other", "verdict": "accepted"}')
    with pytest.raises(MotionFormatError, match="not a dropbear-motion-validation-v1"):
        load_validation_verdict(p)


def test_real_take102_verdict_is_rejected():
    from dropbear_wbc.tasks.tracking.motion_npz import load_validation_verdict

    p = REPO / "data/motions/unitree_rl_lab_mimic/G1_Take_102.npz"
    if not p.is_file():
        pytest.skip("Take_102 NPZ not present")
    assert load_validation_verdict(p)["verdict"] == "rejected"
