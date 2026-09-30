"""Calibration view: mock gating, alias lookup, contract names; real calibration if present."""
from __future__ import annotations

import json

import numpy as np
import pytest

import motion_test_paths  # noqa: F401
from motion_test_paths import MOCK_CAL, REAL_CAL
from dropbear_wbc.motion.calibration_view import load_calibration


def test_mock_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="MOCK"):
        load_calibration(MOCK_CAL)
    cal = load_calibration(MOCK_CAL, allow_mock=True)
    assert cal.is_mock and "Mock" in cal.semantic_map_impl or "SemanticMap" in cal.semantic_map_impl


def test_mock_semantic_roundtrip_and_clipping():
    cal = load_calibration(MOCK_CAL, allow_mock=True)
    rng = np.random.default_rng(2)
    q = np.tile(cal.standing_semantic_pos, (50, 1)) + rng.normal(scale=0.05, size=(50, 22))
    m, info = cal.semantic_to_motor(q)
    np.testing.assert_allclose(cal.motor_to_semantic(m), q, atol=1e-9)
    q[:, 3] = 3.0  # left knee far beyond range
    m, info = cal.semantic_to_motor(q)
    assert np.all(cal.motor_to_semantic(m)[:, 3] < 3.0)


def test_alias_pelvis_T_root_and_missing_field(tmp_path):
    raw = json.loads(MOCK_CAL.read_text())
    pose = raw["rest_transforms"].pop("pelvis_in_root")
    raw["rest_transforms"]["root_in_pelvis"] = {"pos": [-v for v in pose["pos"]], "quat_wxyz": [1, 0, 0, 0]}
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(raw))
    cal = load_calibration(p, allow_mock=True)
    np.testing.assert_allclose(cal.root_T_pelvis[:3, 3], pose["pos"])
    del raw["segment_lengths"]["standing_hip_height"]
    p.write_text(json.dumps(raw))
    with pytest.raises(KeyError, match="standing_hip_height"):
        load_calibration(p, allow_mock=True)


@pytest.mark.skipif(not REAL_CAL.is_file(), reason="real calibration not written yet")
def test_real_calibration_loads():
    cal = load_calibration(REAL_CAL)
    assert not cal.is_mock
    assert 0.8 < cal.standing_hip_height < 1.2
    m, _ = cal.semantic_to_motor(cal.standing_semantic_pos[None])
    np.testing.assert_allclose(m[0], cal.standing_motor_pos, atol=1e-3)
