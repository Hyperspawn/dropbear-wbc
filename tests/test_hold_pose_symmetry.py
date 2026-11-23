"""Item 376 (prepared part of item 370): hold-target left/right symmetry check, ``--default-pose calibration``, and
the loaded-stand calibration sha check when no calibration is given (legacy default pose / policy mode).

Defaults are unchanged: ``--default-pose legacy`` and ``--hold-pose loaded``. The legacy pose (sdk
``motors.DEFAULT_POS``) has both hip rolls at -0.1 rad motor = semantic left -5.86 / right -5.59 deg (mirror error
11.46 deg); it is the hold target only with ``--hold-pose default``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "tools"))

import policy_runner as pr  # noqa: E402
from dropbear_wbc.deploy import config as dc  # noqa: E402
from dropbear_wbc.kinematics.semantic import SemanticMap  # noqa: E402
from dropbear_wbc.sdk import motors  # noqa: E402

CALIB = ROOT / "data/calibration/dropbear_semantic_calibration.json"
LOADED = ROOT / "data/calibration/dropbear_loaded_stand.json"
C = json.loads(CALIB.read_text(encoding="utf-8"))
HIPS = {"left_hip_roll", "right_hip_roll"}


def _build(*argv):
    cfg, _ = pr.build(pr.parse_args(["--mode", "hold", *argv]))
    return cfg


def test_defaults_unchanged():
    a = pr.parse_args([])
    assert a.default_pose == "legacy" and a.hold_pose == "loaded"
    assert pr.default_pose_calibration(a) is None
    assert np.array_equal(_build().default_pose_sdk, np.asarray(motors.DEFAULT_POS, float))


def test_legacy_pose_report_is_the_known_hip_roll():
    r = dc.pose_mirror_report(motors.DEFAULT_POS, CALIB)
    assert r["asymmetric"] and r["worst"] in HIPS
    assert set(r["over_threshold_deg"]) == HIPS
    assert r["max_err_deg"] == pytest.approx(11.46, abs=0.05)
    assert r["hip_roll_deg"]["left_hip_roll"] == pytest.approx(-5.86, abs=0.05)
    assert r["hip_roll_deg"]["right_hip_roll"] == pytest.approx(-5.59, abs=0.05)
    assert r["threshold_deg"] == dc.POSE_MIRROR_WARN_DEG == 3.0


@pytest.mark.parametrize("pose", ["standing", "loaded"])
def test_calibrated_stands_are_symmetric(pose):
    q = C["standing_motor_pos"] if pose == "standing" else json.loads(LOADED.read_text())["loaded_standing_motor_pos"]
    r = dc.pose_mirror_report(q, CALIB)
    assert not r["asymmetric"] and r["max_err_deg"] < 0.1 and r["over_threshold_deg"] == {}


def test_mirror_convention_matches_training_symmetry():
    from dropbear_wbc.tasks.locomotion.symmetry import semantic_mirror

    sm, m, names = SemanticMap(C), semantic_mirror(CALIB), C["semantic_names"]
    rng = np.random.default_rng(376)
    for _ in range(5):
        q = np.asarray(C["standing_motor_pos"]) + rng.uniform(-0.15, 0.15, motors.NUM_MOTORS)
        s = np.asarray(sm.motor_to_semantic(q, clip=False))
        ref = np.degrees(np.abs(m @ s - s))
        r = dc.pose_mirror_report(q, CALIB)
        assert r["max_err_deg"] == pytest.approx(ref.max(), abs=1e-9) and r["worst"] == names[int(ref.argmax())]
        assert set(r["over_threshold_deg"]) == {n for n, e in zip(names, ref) if e > 3.0}


def test_threshold_is_strict_and_hip_sign():
    """A pose with only the right hip roll moved to mirror the left one is symmetric; pushing it 3.5 deg off is not."""
    q = np.asarray(motors.DEFAULT_POS, float).copy()
    i = motors.MOTOR_NAMES.index("PG_right_leg_pitch")
    q[i] = +0.1
    assert dc.pose_mirror_report(q, CALIB)["max_err_deg"] < 3.0
    q[i] += np.radians(3.5)
    assert dc.pose_mirror_report(q, CALIB)["asymmetric"]


def test_report_input_checks(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        dc.pose_mirror_report(np.zeros(21), CALIB)
    bad = dict(C, motor_names=list(reversed(C["motor_names"])))
    p = tmp_path / "c.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="motor order"):
        dc.pose_mirror_report(motors.DEFAULT_POS, p)
    assert Path(dc.pose_mirror_report(motors.DEFAULT_POS)["calibration"]).resolve() == CALIB.resolve()


def test_default_pose_calibration_keyword():
    cfg = _build("--default-pose", "calibration", "--hold-pose", "default")
    assert np.array_equal(cfg.default_pose_sdk, np.asarray(C["standing_motor_pos"], float))
    assert np.array_equal(cfg.default_pose_sdk, _build("--default-pose", str(CALIB), "--hold-pose", "default").default_pose_sdk)
    assert pr.default_pose_calibration(pr.parse_args(["--default-pose", "calibration"])) == pr.CALIBRATION_JSON == CALIB
    assert cfg.meta["hold_target_symmetry"]["asymmetric"] is False


def test_hold_target_symmetry_warns_only_on_legacy_target(capsys):
    cfg = _build("--hold-pose", "default")
    rep = cfg.meta["hold_target_symmetry"]
    assert rep["asymmetric"] and set(rep["over_threshold_deg"]) == HIPS
    assert "WARNING: hold target is not left/right symmetric" in capsys.readouterr().out
    cfg = _build()  # all defaults: hold target = loaded stand
    assert cfg.meta["hold_target_symmetry"]["asymmetric"] is False
    assert "WARNING: hold target" not in capsys.readouterr().out


def test_summary_records_symmetry():
    src = (ROOT / "tools/policy_runner.py").read_text(encoding="utf-8")
    assert '"hold_target_symmetry": cfg.meta.get("hold_target_symmetry")' in src


def _stale_loaded(tmp_path):
    rec = json.loads(LOADED.read_text(encoding="utf-8"))
    rec["calibration_sha256"] = "0" * 64
    p = tmp_path / "stale_loaded.json"
    p.write_text(json.dumps(rec))
    return p


@pytest.mark.parametrize("mode", ["hold", "policy"])
def test_loaded_stand_sha_checked_without_calibration_arg(tmp_path, mode):
    p = _stale_loaded(tmp_path)
    with pytest.raises(ValueError, match="derived from calibration"):
        pr.resolve_hold_pose(pr.parse_args(["--mode", mode, "--hold-pose", str(p)]))
    q = pr.resolve_hold_pose(pr.parse_args(["--mode", mode]))  # repo loaded stand matches the repo calibration
    assert np.array_equal(q, dc.load_loaded_stand_pose(LOADED, CALIB))


def test_explicit_calibration_still_checked(tmp_path):
    other = tmp_path / "other_calib.json"
    other.write_text(CALIB.read_text(encoding="utf-8") + "\n")
    with pytest.raises(ValueError, match="derived from calibration"):
        pr.resolve_hold_pose(pr.parse_args(["--mode", "hold", "--default-pose", str(other)]))


def test_report_is_unclipped():
    """A target beyond the semantic joint limits is judged as commanded, not after clipping to the limits."""
    sm, names = SemanticMap(C), C["semantic_names"]
    q = np.asarray(C["standing_motor_pos"], float).copy()
    q[motors.MOTOR_NAMES.index("PG_left_leg_pitch")] += 1.5
    s_raw = np.asarray(sm.motor_to_semantic(q, clip=False))
    s_clip = np.asarray(sm.motor_to_semantic(q, clip=True))
    assert not np.allclose(s_raw, s_clip)  # the case really exceeds a limit
    r = dc.pose_mirror_report(q, CALIB)
    assert r["hip_roll_deg"]["left_hip_roll"] == pytest.approx(np.degrees(s_raw[names.index("left_hip_roll")]), abs=1e-3)
