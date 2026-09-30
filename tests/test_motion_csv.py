"""dropbear-motion-csv-v1 writer/reader/validator + end-to-end file pipeline (MOCK calibration)."""
from __future__ import annotations

import json

import numpy as np
import pytest

import motion_test_paths  # noqa: F401
from motion_test_paths import MOCK_CAL
from dropbear_wbc.motion.calibration_view import load_calibration
from dropbear_wbc.motion.g1_sources import discover_source_files
from dropbear_wbc.motion.motion_csv import (
    SIDECAR_REQUIRED,
    read_motion_csv,
    validate_motion_files,
    write_motion,
)
from dropbear_wbc.motion.names import CSV_COLUMNS, MOTOR_NAMES, check_against_package
from dropbear_wbc.motion.pipeline import retarget_file


def _sidecar(t):
    return {
        "source": "unit_test",
        "source_file": "<none>",
        "source_license": {"license": "test", "summary": "", "redistributable": True, "url": ""},
        "retarget_method": "test",
        "contact_hint": {"left": [True] * t, "right": [False] * t},
        "notes": [],
    }


def _write(tmp_path, t=12, **over):
    rng = np.random.default_rng(0)
    q = rng.normal(size=(t, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    args = dict(fps=50.0, root_pos=rng.normal(size=(t, 3)), root_quat_wxyz=q, motor_q=rng.normal(size=(t, 22)),
                sidecar=_sidecar(t))
    args.update(over)
    return write_motion(tmp_path, "clip", **args), args


def test_roundtrip_and_header(tmp_path):
    (csv_path, json_path), args = _write(tmp_path)
    with open(csv_path, encoding="utf-8") as f:
        assert tuple(f.readline().strip().split(",")) == CSV_COLUMNS
    assert CSV_COLUMNS[7:] == MOTOR_NAMES
    m = read_motion_csv(csv_path)
    np.testing.assert_allclose(m.motor_q, args["motor_q"], atol=1e-6)
    np.testing.assert_allclose(m.root_pos, args["root_pos"], atol=1e-6)
    # quaternion written as xyzw; compare rotations (sign may flip for continuity)
    dots = np.abs(np.sum(m.root_quat_wxyz * args["root_quat_wxyz"], axis=1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-6)
    side = json.loads(json_path.read_text())
    for k in SIDECAR_REQUIRED:
        assert k in side
    assert validate_motion_files(csv_path) == []


def test_validator_catches_problems(tmp_path):
    (csv_path, json_path), _ = _write(tmp_path)
    side = json.loads(json_path.read_text())
    side["contact_hint"]["left"] = side["contact_hint"]["left"][:-1]
    json_path.write_text(json.dumps(side))
    assert any("contact_hint length" in p for p in validate_motion_files(csv_path))
    del side["notes"]
    json_path.write_text(json.dumps(side))
    assert any("notes" in p for p in validate_motion_files(csv_path))
    lines = csv_path.read_text().splitlines()
    lines[0] = lines[0].replace("root_qw", "root_w")
    csv_path.write_text("\n".join(lines))
    assert validate_motion_files(csv_path)[0].startswith("read error")


def test_writer_rejects_bad_shapes(tmp_path):
    with pytest.raises(ValueError):
        _write(tmp_path, motor_q=np.zeros((12, 21)))


def test_names_agree_with_other_modules():
    status = check_against_package()
    assert status.get("dropbear_wbc.sdk.motors.MOTOR_NAMES") in ("ok", "absent")
    assert "ok" in status.values()  # at least one other module defines the tables and agrees


def test_end_to_end_file_pipeline_every_source(tmp_path):
    cal = load_calibration(MOCK_CAL, allow_mock=True)
    done = 0
    for key in ("unitree_rl_lab_mimic", "soma_retargeter_g1", "kimodo_g1", "asap_g1"):
        files = discover_source_files(key)
        if not files:
            continue
        entry = retarget_file(files[-1], tmp_path, source=key, cal=cal)
        csv_path = tmp_path / key / f"{entry['clip']}.csv"
        assert validate_motion_files(csv_path) == []
        side = json.loads(csv_path.with_suffix(".json").read_text())
        assert side["source"] == key and side["calibration"]["calibration_is_mock"] is True
        assert side["retarget_method"].startswith("MOCK-CALIBRATION")
        assert set(side["saturation"]["per_joint"]) and "suitability" in side
        assert (tmp_path / key / "semantic" / f"{entry['clip']}.semantic.csv").is_file()
        done += 1
    if not done:
        pytest.skip("no G1 sources on disk")
