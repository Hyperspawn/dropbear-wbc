"""``tools/motion_duty_screen.py``: pure helpers always; the MJCF screen when mujoco is installed (tests-cpu)."""
from __future__ import annotations

import sys

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/)

sys.path.insert(0, str(sdk_test_paths.ROOT / "tools"))
import motion_duty_screen as mds  # noqa: E402

STAND = sdk_test_paths.ROOT / "data/motions/synthetic/stand.npz"


def test_window_rms_finds_the_held_segment():
    x = np.zeros((500, 2))
    x[200:300, 0] = 3.0  # 2 s hold at 50 Hz
    w = mds.window_rms_max(x, 100)
    assert w[0] == pytest.approx(3.0) and w[1] == 0.0
    assert mds.window_rms_max(x, 1000)[0] == pytest.approx(np.sqrt(0.2) * 3.0)  # shorter than window: whole RMS


def test_longest_run_and_duty_metrics():
    tau = np.zeros((100, 1))
    tau[10:40, 0] = 12.0
    tau[60:65, 0] = -30.0
    m = mds.duty_metrics(tau, 0.02, np.array([10.0]), np.array([25.0]), 0.2)
    assert m["over_s"][0] == pytest.approx(30 * 0.02)
    assert m["over_frac"][0] == pytest.approx(0.35)
    assert m["peak"][0] == pytest.approx(1.2)
    assert mds.severity("peak", m["peak"][0]) == "HIGH"
    assert mds.severity("sus", 1.1) == "warn" and mds.severity("sus", 0.9) == ""


def test_gaussian_lp_is_zero_phase_and_minus_3db_at_fc():
    dt, fc = 0.02, 6.0
    t = np.arange(2000) * dt
    y = mds.gaussian_lp(np.sin(2 * np.pi * fc * t), dt, fc)
    mid = slice(200, 1800)
    amp = np.sqrt(2 * (y[mid] ** 2).mean())
    assert amp == pytest.approx(1 / np.sqrt(2), abs=0.02)
    lag = np.argmax(np.correlate(y[mid], np.sin(2 * np.pi * fc * t[mid]), "full")) - (len(t[mid]) - 1)
    assert lag == 0
    assert np.array_equal(mds.gaussian_lp(t, dt, None), t)


mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def plant():
    return mds.ArmDutyPlant()


def test_stand_clip_is_clean_and_fk_matches(plant):
    c = mds.load_clip(STAND)
    err = plant.fk_check(c, STAND)
    arm = {b: e for b, e in err.items() if b.startswith(("LH_", "RH_", "world"))}
    assert max(arm.values()) < 0.01, arm  # root convention + semantic map: arm bodies within 1 cm
    res = mds.screen([STAND], plant=plant)
    rec = res["clips"][0]
    assert rec["worst"] == "ok", rec["findings"]
    el = rec["motors"]["RH_elbow_joint"]
    assert el["dynamic"]["sus"] == pytest.approx(el["static"]["sus"], abs=0.02)  # still clip: dynamics ~ gravity


def _held_pose_clip(plant, tmp_path, yaw):
    """Stand clip with the right shoulder motor moved to ``yaw`` and held: the arm raised, elbow at its LUT end."""
    z = dict(np.load(STAND, allow_pickle=True))
    q = z["motor_target"].copy()
    mn = [str(n) for n in z["motor_names"]]
    q[:, mn.index("RH_yaw")] = yaw
    q[:, mn.index("RH_elbow_joint")] = 0.069
    z["motor_target"] = q
    p = tmp_path / "held.npz"
    np.savez(p, **z)
    return p


def test_raised_straight_arm_hold_is_flagged_static(plant, tmp_path):
    """Forearm horizontal near the straight-arm end of the LUT: ~1.9 N*m at the joint (notes/09), ~0.9x X8 Pro rated
    at the motor with zero payload -> static_sus warn on the right elbow only."""
    res = mds.screen([_held_pose_clip(plant, tmp_path, -1.4)], plant=plant)
    rec = res["clips"][0]
    el = rec["motors"]["RH_elbow_joint"]["static"]
    assert 0.85 < el["sus"] < 1.0, el
    assert {(f["motor"], f["metric"], f["severity"]) for f in rec["findings"]} >= {("RH_elbow_joint", "static_sus", "warn")}
    assert rec["motors"]["LH_elbow_joint"]["static"]["sus"] < 0.5


def test_cli_fail_on(tmp_path, plant):
    p = _held_pose_clip(plant, tmp_path, -1.4)
    assert mds.main([str(p), "--json", str(tmp_path / "o.json")]) == 0
    assert mds.main([str(p), "--fail-on", "warn"]) == 1
    assert mds.main([str(STAND), "--fail-on", "warn"]) == 0
