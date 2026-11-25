"""Item 377: scripts/examples/dropbear_low_level_example.py ``--default-pose {legacy,calibration,<path>}``.

Before item 377 Stage 1 always ramped all 22 motors to sdk ``motors.DEFAULT_POS`` (both hip rolls -0.1 rad, semantic
mirror error 11.5 deg, item 370) with no report. The default stays 'legacy' (choice of default = item 370 decision);
the target is mirror-checked before any transport is opened (warn only) and recorded as ``summary.default_pose``.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from dropbear_wbc.sdk.types import LowState, MotorStateBlock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/examples/dropbear_low_level_example.py"
spec = importlib.util.spec_from_file_location("ll_example_377", SCRIPT)
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

CALIB = ROOT / "data/calibration/dropbear_semantic_calibration.json"
REPO = json.loads(CALIB.read_text(encoding="utf-8"))
STAND = np.asarray(REPO["standing_motor_pos"], float)
LEGACY = np.asarray(ex.motors.DEFAULT_POS, float)
WRIST_SHIFT = 0.3


def _calib_x(d: Path, stand=None, shift=WRIST_SHIFT) -> Path:
    c = json.loads(CALIB.read_text(encoding="utf-8"))
    c["dofs"]["left_wrist_roll"]["offset"] = float(c["dofs"]["left_wrist_roll"]["offset"]) + shift
    if stand is not None:
        c["standing_motor_pos"] = list(stand)
    p = d / "calib_x.json"
    p.write_text(json.dumps(c), encoding="utf-8")
    return p


class _Pub:
    def __init__(self):
        self.sent = []

    def Write(self, cmd):
        self.sent.append(cmd.motor.q.astype(float).copy())


def _state(q):
    st = LowState(tick=1, stamp_ns=0)
    st.motor = MotorStateBlock.zeros(ex.NUM_MOTOR)
    st.motor.q[:] = np.asarray(q, np.float32)
    return st


def test_parse_args_default_is_legacy():
    assert ex.parse_args([]).default_pose == "legacy"
    assert ex.parse_args(["--default-pose", "calibration"]).default_pose == "calibration"


def test_resolve_legacy_is_sdk_default_and_a_copy():
    pose, calib, source = ex.resolve_default_pose("legacy")
    assert source == "legacy" and calib == ex.CALIBRATION_JSON and np.array_equal(pose, LEGACY)
    pose[0] += 1.0
    assert np.array_equal(np.asarray(ex.motors.DEFAULT_POS, float), LEGACY)
    assert ex.resolve_default_pose(None)[2] == "legacy"


def test_resolve_calibration_is_repo_standing_pose():
    pose, calib, source = ex.resolve_default_pose("calibration")
    assert source == "calibration" and calib == CALIB and np.allclose(pose, STAND)
    assert not np.allclose(pose, LEGACY)


def test_resolve_path_uses_that_file(tmp_path):
    stand = STAND.copy()
    stand[2] += 0.05
    x = _calib_x(tmp_path, stand)
    pose, calib, source = ex.resolve_default_pose(str(x))
    assert source == "path" and Path(calib) == x and np.allclose(pose, stand)


def test_resolve_bad_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ex.resolve_default_pose(str(tmp_path / "nope.json"))
    bad = STAND.copy()
    bad[4] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        ex.resolve_default_pose(str(_calib_x(tmp_path, bad)))
    with pytest.raises(ValueError, match="shape"):
        ex.resolve_default_pose(str(_calib_x(tmp_path, STAND[:20])))


def test_legacy_pose_warns_asymmetric(capsys):
    rep = ex.default_pose_symmetry(LEGACY, CALIB, "legacy")
    assert rep["asymmetric"] and rep["max_err_deg"] == pytest.approx(11.5, abs=0.3)
    assert rep["default_pose_source"] == "legacy" and rep["calibration"] == CALIB.as_posix()
    assert "WARNING: default pose (legacy) is not left/right symmetric" in capsys.readouterr().out


def test_calibration_pose_symmetric_no_warning(capsys):
    rep = ex.default_pose_symmetry(STAND, CALIB, "calibration")
    assert not rep["asymmetric"] and rep["max_err_deg"] < 0.5
    assert "WARNING" not in capsys.readouterr().out


def test_path_pose_checked_against_its_calibration(tmp_path):
    x = _calib_x(tmp_path, STAND)  # repo-symmetric pose, map X: left wrist offset moved 0.3 rad
    rep = ex.default_pose_symmetry(STAND, x, "path")
    assert rep["asymmetric"] and rep["worst"].endswith("wrist_roll") and rep["calibration"] == x.as_posix()
    assert rep["max_err_deg"] == pytest.approx(np.degrees(WRIST_SHIFT), abs=0.5)


def test_missing_calibration_note_and_none(tmp_path, capsys):
    assert ex.default_pose_symmetry(LEGACY, tmp_path / "gone.json", "legacy") is None
    assert "NOTE: default pose mirror check skipped" in capsys.readouterr().out


def test_record_block():
    pose, rec = ex.default_pose_record("calibration")
    assert np.allclose(pose, STAND)
    assert rec["choice"] == "calibration" and rec["source"] == "calibration" and rec["calibration"] == CALIB.as_posix()
    assert rec["mirror"]["asymmetric"] is False and np.allclose(rec["default_pose_sdk"], STAND)
    _, rec = ex.default_pose_record("legacy")
    assert rec["source"] == "legacy" and rec["mirror"]["asymmetric"] is True and np.allclose(rec["default_pose_sdk"], LEGACY)


def test_custom_default_none_is_legacy():
    assert np.array_equal(ex.Custom(1.0).default, LEGACY)


def test_custom_rejects_bad_pose():
    with pytest.raises(ValueError):
        ex.Custom(1.0, default_pose=np.zeros(5))
    bad = STAND.copy()
    bad[0] = np.inf
    with pytest.raises(ValueError):
        ex.Custom(1.0, default_pose=bad)


def test_stage1_ends_at_and_stage2_swings_about_given_pose():
    c = ex.Custom(4.0, default_pose=STAND)
    c.lowcmd_publisher_ = _Pub()
    c.LowStateHandler(_state(np.zeros(ex.NUM_MOTOR)))
    c.time_ = c.duration_ - c.control_dt_ - 1e-9
    c.LowCmdWrite()
    assert np.allclose(c.lowcmd_publisher_.sent[-1], STAND, atol=2e-3)
    c.time_ = c.duration_ + 0.5 - c.control_dt_
    c.LowCmdWrite()
    q = c.lowcmd_publisher_.sent[-1]
    swing = {getattr(ex.DropbearJointIndex, n) for n in dir(ex.DropbearJointIndex) if not n.startswith("_")}
    rest = [i for i in range(ex.NUM_MOTOR) if i not in swing]
    assert np.allclose(q[rest], STAND[rest], atol=2e-3)
    i = ex.DropbearJointIndex.LeftWristRoll
    assert q[i] - STAND[i] == pytest.approx(np.deg2rad(30.0) * np.sin(2 * np.pi * 0.5 * 0.5), abs=2e-3)
    i = ex.DropbearJointIndex.LeftCalfMotorA  # calibration -0.368 vs legacy -0.2 rad
    assert q[i] - STAND[i] == pytest.approx(np.deg2rad(15.0) * np.sin(2 * np.pi * 0.5 * 0.5), abs=2e-3)


def _run(tmp_path, port, *extra):
    summ = tmp_path / f"s{port}.json"
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, "-u", str(SCRIPT), "--duration", "1", "--connect-timeout-s", "0.3",
                        "--cmd-port", str(port), "--state-port", str(port + 1), "--summary", str(summ), *extra],
                       capture_output=True, text=True, timeout=60)
    return r, summ, time.monotonic() - t0


def test_main_records_calibration_pose(tmp_path):
    r, summ, _ = _run(tmp_path, 27841, "--default-pose", "calibration")
    assert r.returncode == 1 and "StateLost" in r.stderr  # no bridge: nothing commanded
    s = json.loads(summ.read_text())
    assert s["default_pose"]["source"] == "calibration" and s["default_pose"]["mirror"]["asymmetric"] is False
    assert "WARNING: default pose" not in r.stdout
    assert np.allclose(s["default_pose"]["commanded_sdk"], STAND)  # the pose Custom actually ramps to


def test_main_default_legacy_warns(tmp_path):
    r, summ, _ = _run(tmp_path, 27843)
    s = json.loads(summ.read_text())
    assert s["default_pose"]["choice"] == "legacy" and s["default_pose"]["mirror"]["asymmetric"] is True
    assert "WARNING: default pose (legacy)" in r.stdout


def test_main_bad_path_fails_before_transport(tmp_path):
    r, summ, dt = _run(tmp_path, 27845, "--default-pose", str(tmp_path / "missing.json"))
    assert r.returncode != 0 and "FileNotFoundError" in r.stderr and not summ.exists()
    assert "ChannelFactoryInitialize" not in r.stderr and dt < 30


def test_main_resolves_pose_before_opening_transport():
    import ast
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test))
    calls = [(c.lineno, ast.unparse(c.func)) for c in ast.walk(main) if isinstance(c, ast.Call)]
    first = {name: min(l for l, f in calls if f == name) for name in ("default_pose_record", "ChannelFactoryInitialize")}
    assert first["default_pose_record"] < first["ChannelFactoryInitialize"]
    assert not any(f.endswith(".Init") and l < first["default_pose_record"] for l, f in calls)
