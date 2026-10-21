"""Item 241: FSM MoveToDefault / Hold target = calibration-derived loaded stand (knee crank ~3 deg).

The calibration's standing_motor_pos holds ~80 N*m static on each knee crank (1.3x the 60 N*m rating of CEM-60 /
X10-200, notes/219b); the trained walkers stand at crank 2-4 deg. The static-ID gate keeps the HOLD knee below 0.5x
rated at the heaviest studied mass (65.2 kg, hw_v4_arms) for both CoP conventions.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "tools"))

from dropbear_wbc.deploy.config import hold_config, load_loaded_stand_pose, load_standing_pose  # noqa: E402
from dropbear_wbc.deploy.fsm import DeployController, FsmState  # noqa: E402
from dropbear_wbc.sdk import motors  # noqa: E402

CALIB = ROOT / "data/calibration/dropbear_semantic_calibration.json"
LOADED = ROOT / "data/calibration/dropbear_loaded_stand.json"
XML = ROOT / "data/robot/dropbear_serial.xml"
KNEES = ("LL_knee_actuator_joint", "RL_knee_actuator_joint")
HEAVIEST_KG = 65.2


def test_committed_record_matches_calibration():
    rec = json.loads(LOADED.read_text(encoding="utf-8"))
    assert rec["schema"] == "dropbear-loaded-stand-v1"
    assert rec["calibration_sha256"] == hashlib.sha256(CALIB.read_bytes()).hexdigest(), \
        "calibration changed: re-run tools/derive_loaded_stand.py"
    pose = load_loaded_stand_pose(LOADED, CALIB)
    ref = load_standing_pose(CALIB)
    lim = np.asarray(json.loads(CALIB.read_text())["motor_limits_rad"], float).reshape(-1, 2)
    assert np.all(pose >= lim[:, 0] - 1e-9) and np.all(pose <= lim[:, 1] + 1e-9)
    for k in KNEES:
        i = motors.MOTOR_NAMES.index(k)
        assert math.degrees(pose[i]) == pytest.approx(3.0, abs=0.05)
        assert math.degrees(ref[i]) > 15.0
    legs = {i for i, n in enumerate(motors.MOTOR_NAMES) if n.startswith(("LL_", "RL_", "PG_"))}
    for i in set(range(motors.NUM_MOTORS)) - legs:  # arms stay at the reference stand
        assert pose[i] == pytest.approx(ref[i], abs=1e-6), motors.MOTOR_NAMES[i]
    # semantic side: only hip pitch, knee and ankle pitch move (the serial hip chain moves the PG_* motors with them)
    from dropbear_wbc.kinematics.semantic import SEMANTIC_NAMES

    q_new = np.asarray(rec["loaded_standing_semantic_pos"], float)
    q_ref = np.asarray(json.loads(CALIB.read_text())["standing_semantic_pos"], float)
    moved = {i for i, n in enumerate(SEMANTIC_NAMES) if n.endswith(("_hip_pitch", "_knee", "_ankle_pitch"))}
    for i in set(range(len(SEMANTIC_NAMES))) - moved:
        assert q_new[i] == pytest.approx(q_ref[i], abs=1e-6), SEMANTIC_NAMES[i]


def test_wrong_calibration_is_refused(tmp_path):
    other = tmp_path / "calib.json"
    other.write_text(CALIB.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="derive_loaded_stand"):
        load_loaded_stand_pose(LOADED, other)



def test_fsm_hold_targets_loaded_stand_and_keeps_policy_offset():
    ref = load_standing_pose(CALIB)
    loaded = load_loaded_stand_pose(LOADED)
    cfg_old = hold_config(ref)
    cfg_new = hold_config(ref, loaded)
    assert np.array_equal(cfg_old.hold_target_sdk, ref)
    assert np.array_equal(cfg_new.hold_target_sdk, loaded)
    assert np.array_equal(cfg_new.default_pose_sdk, ref) and np.array_equal(cfg_new.action_offset, ref)
    import inspect

    src = inspect.getsource(DeployController)
    assert "cfg.hold_target_sdk - self._q_start" in src and "q_des = cfg.hold_target_sdk.copy()" in src
    assert FsmState.HOLD.value == "hold"


def test_runner_hold_pose_flag():
    import policy_runner as pr

    args = pr.parse_args(["--mode", "hold", "--default-pose", str(CALIB)])
    assert args.hold_pose == "loaded"
    assert np.array_equal(pr.resolve_hold_pose(args), load_loaded_stand_pose(LOADED))
    assert pr.resolve_hold_pose(pr.parse_args(["--hold-pose", "default"])) is None


mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def plant():
    from dropbear_wbc.kinematics.loaded_stand import StandPlant

    return StandPlant(json.loads(CALIB.read_text(encoding="utf-8")), XML)


@pytest.mark.parametrize("build", ["cem6_2026_10_16", "v4_2026_10_16", "v4_arms_2026_10_20"])
@pytest.mark.parametrize("cop_y", ["sole", "com"])
def test_static_id_hold_knee_below_half_rated(plant, build, cop_y):
    from dropbear_wbc.kinematics.loaded_stand import static_stand_torques
    from dropbear_wbc.kinematics.semantic import MOTOR_NAMES
    from dropbear_wbc.robots.hw_motor_specs import JOINT_MOTOR_MAPS, MOTOR_MODELS

    rec = json.loads(LOADED.read_text(encoding="utf-8"))
    scale = HEAVIEST_KG / plant.mass
    q_new = np.asarray(rec["loaded_standing_semantic_pos"], float)
    q_old = np.asarray(plant.calib["standing_semantic_pos"], float)
    t_new = static_stand_torques(plant, q_new, cop_y=cop_y) * scale
    t_old = static_stand_torques(plant, q_old, cop_y=cop_y) * scale
    for k in KNEES:
        rated = MOTOR_MODELS[JOINT_MOTOR_MAPS[build][k]].rated_torque
        i = MOTOR_NAMES.index(k)
        assert abs(t_new[i]) < 0.5 * rated, (k, t_new[i], rated)
        assert abs(t_old[i]) > 1.2 * rated  # the pre-241 HOLD pose; documents why the default moved


def test_derive_reproduces_committed_record(plant, tmp_path):
    import derive_loaded_stand as dls

    out = tmp_path / "ls.json"
    assert dls.main(["--out", str(out)]) == 0
    a, b = json.loads(out.read_text()), json.loads(LOADED.read_text())
    assert np.allclose(a["loaded_standing_motor_pos"], b["loaded_standing_motor_pos"], atol=1e-7)
