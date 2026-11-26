"""Item 393: tools/probe_newton_plant.py ``--default-pose {legacy,calibration,<path>}``.

Before item 393 ``--target default`` always ramped the Newton plant to sdk ``motors.DEFAULT_POS`` (both hip rolls
-0.1 rad, semantic mirror error 11.5 deg, item 370) and the report did not say which pose was used. The default stays
'legacy' (same as the low-level example, item 377, and policy_runner); the pose is resolved with the shared
``deploy.config`` helpers before the GPU lock / plant build, mirror-checked (warn only) and recorded as
``default_pose`` in the report and the printed summary. The plant is a numpy fake (no Newton on CI).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from dropbear_wbc.deploy import config as deploy_config
from dropbear_wbc.sdk import motors

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/probe_newton_plant.py"
spec = importlib.util.spec_from_file_location("probe_newton_plant_393", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

CALIB = ROOT / "data/calibration/dropbear_semantic_calibration.json"
STAND = np.asarray(json.loads(CALIB.read_text(encoding="utf-8"))["standing_motor_pos"], float)
LEGACY = np.asarray(motors.DEFAULT_POS, float)
Q_INIT = np.linspace(-0.05, 0.05, motors.NUM_MOTORS)


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a, float)

    def numpy(self):
        return self.a


class _Readout:
    def __init__(self, q, tick=0):
        self.motor_q = np.asarray(q, float).copy()
        self.motor_tau = np.zeros(motors.NUM_MOTORS)
        self.root_pos_w = np.array([0.0, 0.0, 1.0])
        self.root_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self.tick, self.time_s = tick, tick * 0.002


class _FakePlant:
    built = 0

    def __init__(self, cfg):
        type(self).built += 1
        self.cfg = types.SimpleNamespace(tick_dt=0.002)
        self.report = {"fake": True}
        self.motor_qd_idx = np.arange(motors.NUM_MOTORS)
        self.model = types.SimpleNamespace(joint_limit_lower=_Arr(-np.ones(motors.NUM_MOTORS) * 3),
                                           joint_limit_upper=_Arr(np.ones(motors.NUM_MOTORS) * 3))
        self.solver = None
        self.commands: list[np.ndarray] = []
        _FakePlant.last = self

    def initial_readout(self):
        return _Readout(Q_INIT)

    def set_motor_command(self, q, dq, tau, kp, kd, mode):
        self.commands.append(np.asarray(q, float).copy())

    def step(self):
        pass

    def readout(self):
        return _Readout(self.commands[-1], len(self.commands))

    def closure_residuals_m(self):
        return np.zeros(2)


class _FakeLock:
    acquired = 0

    def __init__(self, owner):
        self.waited_s = 0.0

    def acquire(self):
        type(self).acquired += 1

    def release(self):
        pass


@pytest.fixture
def run(monkeypatch, tmp_path, capsys):
    fake = types.ModuleType("dropbear_wbc.newton_sim.plant")
    fake.DropbearNewtonPlant = _FakePlant
    fake.PlantConfig = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "dropbear_wbc.newton_sim.plant", fake)
    monkeypatch.setattr(probe, "GpuLock", _FakeLock)
    _FakePlant.built, _FakeLock.acquired = 0, 0

    def _run(*extra, ticks=1200):
        out = tmp_path / "probe.json"
        rc = probe.main(["--ticks", str(ticks), "--ramp-s", "1.0", "--report", str(out), *extra])
        stdout = capsys.readouterr().out
        return rc, json.loads(out.read_text(encoding="utf-8")), stdout
    return _run


def _calib_x(d: Path, stand) -> Path:
    c = json.loads(CALIB.read_text(encoding="utf-8"))
    c["standing_motor_pos"] = list(map(float, stand))
    p = d / "calib_x.json"
    p.write_text(json.dumps(c), encoding="utf-8")
    return p


def _final_q_des(rep):
    return np.array([rep["final_per_motor"][n]["q_des"] for n in motors.MOTOR_NAMES])


def _printed(stdout: str) -> dict:
    return json.loads(stdout[stdout.index("{\n"):])


def test_cli_default_is_legacy(run):
    rc, rep, _ = run("--target", "default")
    assert rc == 0 and rep["status"] == "executed"
    assert rep["args"]["default_pose"] == "legacy"
    assert rep["default_pose"]["source"] == "legacy" and rep["default_pose"]["used"] is True
    assert np.allclose(_final_q_des(rep), LEGACY)


def test_legacy_target_mirror_warns(run):
    _, rep, out = run("--target", "default", "--default-pose", "legacy")
    m = rep["default_pose"]["mirror"]
    assert m["asymmetric"] is True and m["max_err_deg"] == pytest.approx(11.456, abs=0.01)
    assert set(m["hip_roll_deg"]) == {"left_hip_roll", "right_hip_roll"}
    assert "[probe] WARNING: default pose (legacy) is not left/right symmetric" in out
    assert rep["default_pose"]["calibration"] == CALIB.as_posix()


def test_calibration_target_drives_plant_to_standing_pose(run):
    rc, rep, out = run("--target", "default", "--default-pose", "calibration")
    assert rc == 0
    assert np.allclose(_final_q_des(rep), STAND)
    assert np.allclose(_FakePlant.last.commands[-1], STAND)
    assert rep["default_pose"]["source"] == "calibration"
    assert rep["default_pose"]["mirror"]["asymmetric"] is False
    assert rep["default_pose"]["mirror"]["default_pose_source"] == "calibration"
    assert rep["default_pose"]["mirror"]["max_err_deg"] < 0.1
    assert np.allclose(rep["default_pose"]["default_pose_sdk"], STAND)
    assert "WARNING: default pose" not in out


def test_path_target(run, tmp_path):
    stand = STAND.copy()
    stand[motors.motor_index("LH_wrist_roll")] += 0.2
    p = _calib_x(tmp_path, stand)
    _, rep, _ = run("--target", "default", "--default-pose", str(p))
    assert rep["default_pose"]["source"] == "path"
    assert rep["default_pose"]["calibration"] == p.as_posix()
    assert rep["default_pose"]["mirror"]["calibration"] == p.as_posix()
    assert np.allclose(_final_q_des(rep), stand)


def test_ramp_is_linear_to_resolved_pose(run):
    run("--target", "default", "--default-pose", "calibration", ticks=300)
    c = _FakePlant.last.commands  # ramp 1.0 s / 0.002 s = 500 ticks; tick 250 = halfway
    assert np.allclose(c[0], Q_INIT)
    assert np.allclose(c[250], Q_INIT + 0.5 * (STAND - Q_INIT))


def test_target_initial_records_pose_unused_and_holds_initial(run):
    _, rep, out = run("--default-pose", "calibration")
    assert rep["default_pose"]["used"] is False and rep["default_pose"]["source"] == "calibration"
    assert np.allclose(_final_q_des(rep), Q_INIT)


def test_missing_path_fails_before_lock_and_plant(run, tmp_path):
    rc, rep, _ = run("--target", "default", "--default-pose", str(tmp_path / "nope.json"))
    assert rc == 1 and rep["status"] == "failed"
    assert "nope.json" in rep["error"]
    assert _FakePlant.built == 0 and _FakeLock.acquired == 0


def test_printed_summary_has_default_pose(run):
    _, rep, out = run("--target", "default")
    s = _printed(out)["default_pose"]
    assert s["source"] == "legacy" and s["used"] is True and s["asymmetric"] is True
    assert s["mirror_max_err_deg"] == pytest.approx(11.456, abs=0.01)
    assert np.allclose(s["default_pose_sdk"], LEGACY)
    _, _, out = run("--target", "default", "--default-pose", "calibration")
    s = _printed(out)["default_pose"]
    assert s["asymmetric"] is False and s["mirror_max_err_deg"] < 0.1


def test_missing_calibration_note_mirror_none(monkeypatch, run, tmp_path):
    import dropbear_wbc.kinematics.semantic as sem
    monkeypatch.setattr(sem, "DEFAULT_CALIBRATION", tmp_path / "gone.json")
    _, rep, out = run("--target", "default")
    assert rep["status"] == "executed" and rep["default_pose"]["mirror"] is None
    assert "[probe] NOTE: default pose mirror check skipped" in out
    assert _printed(out)["default_pose"]["mirror_max_err_deg"] is None


def test_probe_reuses_shared_helpers_no_copy():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not defs & {"resolve_default_pose", "default_pose_symmetry", "default_pose_record", "pose_mirror_report"}
    assert probe.default_pose_record is deploy_config.default_pose_record
    assert not re.search(r"\bDEFAULT_POS\b", SCRIPT.read_text(encoding="utf-8").split("def main", 1)[1])


def test_low_level_example_delegates_to_shared_helpers():
    spec2 = importlib.util.spec_from_file_location("ll_example_393", ROOT / "scripts/examples/dropbear_low_level_example.py")
    ex = importlib.util.module_from_spec(spec2)
    try:
        spec2.loader.exec_module(ex)
    except ModuleNotFoundError as e:  # transport deps (zmq) not installed
        pytest.skip(str(e))
    assert ex._resolve_default_pose is deploy_config.resolve_default_pose
    assert ex._default_pose_symmetry is deploy_config.default_pose_symmetry
    assert ex.CALIBRATION_JSON.resolve() == Path(__import__("dropbear_wbc.kinematics.semantic",
                                                            fromlist=["x"]).DEFAULT_CALIBRATION).resolve()


def test_shared_resolve_default_pose(tmp_path):
    pose, calib, src = deploy_config.resolve_default_pose("legacy")
    assert src == "legacy" and np.array_equal(pose, LEGACY) and calib.resolve() == CALIB.resolve()
    pose[0] += 1.0
    assert np.array_equal(np.asarray(motors.DEFAULT_POS, float), LEGACY)  # copy, not the sdk constant
    pose, calib, src = deploy_config.resolve_default_pose("calibration", CALIB)
    assert src == "calibration" and np.allclose(pose, STAND)
    bad = STAND.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        deploy_config.resolve_default_pose(str(_calib_x(tmp_path, bad)))


def test_help_mentions_default_pose(capsys):
    with pytest.raises(SystemExit):
        probe.main(["--help"])
    h = " ".join(capsys.readouterr().out.split())
    assert "--default-pose" in h and "NOT mirror-symmetric" in h
