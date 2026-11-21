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


def test_sus_thresholds_follow_the_thermal_model():
    """Item 363: warn = max continuous ratio at ambient, HIGH = same trip time as 1.3x from a 25 C soak."""
    from dropbear_wbc.robots.thermal import DESIGN_AMBIENT_C, ThermalModel
    assert mds.thresholds(25.0)["sus"] == (1.0905, 1.3)  # HIGH parity with the old fixed 1.3x at 25 C
    assert mds.thresholds(35.0)["sus"] == (1.0152, 1.2064)
    assert mds.thresholds(45.0)["sus"] == (0.9339, 1.1064)
    assert DESIGN_AMBIENT_C == 45.0 and mds.thresholds()["sus"] == mds.thresholds(45.0)["sus"]
    for amb in (25.0, 45.0):  # peak / static unchanged at any ambient
        assert mds.thresholds(amb)["peak"] == (1.0, 1.0) and mds.thresholds(amb)["static_sus"] == (0.8, 1.0)
    assert mds.sus_high_trip_s() == pytest.approx(1041.0, abs=0.5)  # 17.35 min (notes/129, notes/362)
    m45 = ThermalModel(45.0)
    assert m45.time_to_trip(mds.thresholds(45.0)["sus"][1], interpolate=True) == pytest.approx(1041.0, abs=1.0)
    assert m45.time_to_trip(0.93, t_max=7200) == float("inf") and m45.time_to_trip(0.94, t_max=7200) < 7200


def test_sus_severity_at_25C_matches_old_high_and_flips_at_45C():
    old = {"sus": (1.0, 1.3)}
    t25, t45 = mds.thresholds(25.0), mds.thresholds(45.0)
    for v in np.linspace(0.5, 2.0, 301):  # 25 C: HIGH exactly where the old fixed 1.3 put it
        assert (mds.severity("sus", v, t25) == "HIGH") == (mds.severity("sus", v, old) == "HIGH"), v
    # warn moves 1.0 -> 1.09 at 25 C (rated is continuous below 36.9 C ambient)
    assert mds.severity("sus", 1.05, old) == "warn" and mds.severity("sus", 1.05, t25) == ""
    # 45 C default: 0.95 and 1.15 get worse
    assert mds.severity("sus", 0.95, old) == "" and mds.severity("sus", 0.95) == "warn"
    assert mds.severity("sus", 1.15, t25) == "warn" and mds.severity("sus", 1.15) == "HIGH"
    assert mds.severity("sus", 0.93) == "" and mds.severity("sus", 1.10) == "warn"


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


def test_held_arm_sus_flips_with_ambient(plant, tmp_path):
    """The raised straight-arm hold (RH elbow dynamic sus ~0.93x rated): no sus finding at 25 or 45 C (just under
    the 0.934 continuous limit), sus warn at 47 C (0.917) and 50 C (0.890, trips within the hour); ambient and schema
    recorded."""
    p = _held_pose_clip(plant, tmp_path, -1.4)
    out = {}
    for amb in (25.0, 45.0, 47.0, 50.0):
        res = mds.screen([p], plant=plant, ambient_C=amb)
        assert res["ambient_C"] == amb and res["schema"] == "dropbear-motion-duty-screen-v2"
        assert res["thresh"]["sus"] == mds.thresholds(amb)["sus"]
        rec = res["clips"][0]
        out[amb] = rec
    sus = out[25.0]["motors"]["RH_elbow_joint"]["dynamic"]["sus"]
    assert 0.9167 < sus < 0.9339, sus
    f = {a: {(x["motor"], x["metric"], x["severity"]) for x in r["findings"]} for a, r in out.items()}
    assert ("RH_elbow_joint", "sus", "warn") not in f[25.0] | f[45.0]
    assert ("RH_elbow_joint", "sus", "warn") in f[47.0] & f[50.0]
    th = {a: r["motors"]["RH_elbow_joint"]["thermal"] for a, r in out.items()}
    assert th[45.0]["sus_trip_s"] is None and 0 < th[50.0]["sus_trip_s"] < mds.SUS_TRIP_MAX_S
    assert th[25.0]["rms_steady_winding_C"] < th[45.0]["rms_steady_winding_C"] < 100.0


def test_cli_ambient_option(tmp_path, plant):
    import json
    for amb, warn in (("25", 1.0905), (None, 0.9339)):
        o = tmp_path / f"o{amb}.json"
        assert mds.main([str(STAND), "--json", str(o)] + (["--ambient_C", amb] if amb else [])) == 0
        r = json.loads(o.read_text(encoding="utf-8"))
        assert r["ambient_C"] == (float(amb) if amb else 45.0) and r["thresh"]["sus"][0] == warn
