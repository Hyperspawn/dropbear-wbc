"""Two-node ambient-aware thermal model (``dropbear_wbc.robots.thermal``) and its use in the telemetry scan and the
scoreboard. Reference numbers: notes/361 (max continuous 1.09 / 1.02 / 0.93 and rated steady winding 85.2 / 97.6 /
110.0 C at 25 / 35 / 45 C ambient) and notes/129 (1.3x rated from a 25 C soak trips after 17.35 min)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import telemetry_issue_scan as tis  # noqa: E402
from dropbear_wbc.robots import thermal as th  # noqa: E402
from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES  # noqa: E402
from tools import scoreboard as sb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _reference_trace(ratio, dt, ambient, T0=None):
    """Independent copy of the notes/36 twin_v2 ThermalState law (explicit Euler)."""
    a = 0.00393
    P_ref = 60.0 / (1.0 + a * (24.0 + 60.0 - 25.0))
    R_wh, R_ha = 0.3, 0.7
    C_w, C_h = 40.0 / R_wh, 900.0 / R_ha
    n = ratio.shape[1]
    Tw = np.full(n, ambient if T0 is None else T0, float)
    Th = Tw.copy()
    out = []
    for r in ratio:
        P = P_ref * r * r * (1.0 + a * (Tw - 25.0))
        dTw = (P - (Tw - Th) / R_wh) / C_w
        dTh = ((Tw - Th) / R_wh - (Th - ambient) / R_ha) / C_h
        Tw, Th = Tw + dt * dTw, Th + dt * dTh
        out.append(Tw.copy())
    return np.array(out)


@pytest.mark.parametrize("amb,cont,tw,th_", [(25.0, 1.0905, 85.24, 67.17), (35.0, 1.0152, 97.60, 78.82),
                                             (45.0, 0.9339, 109.97, 90.48)])
def test_notes361_parity(amb, cont, tw, th_):
    m = th.ThermalModel(amb)
    assert abs(m.max_continuous_ratio() - cont) < 5e-4
    assert abs(float(m.steady_winding_C(1.0)) - tw) < 0.01
    assert abs(float(m.steady_housing_C(1.0)) - th_) < 0.01
    # the closed-form continuous limit sits exactly on the trip
    assert abs(float(m.steady_winding_C(m.max_continuous_ratio())) - m.t_trip_C) < 1e-6


def test_rated_not_continuous_above_36_9_C():
    m = th.ThermalModel(25.0)
    assert abs(m.ambient_for_continuous(1.0) - 36.937) < 0.01
    assert th.ThermalModel(36.8).max_continuous_ratio() > 1.0 > th.ThermalModel(37.0).max_continuous_ratio()
    assert th.ThermalModel(100.0).max_continuous_ratio() == 0.0


def test_notes129_trip_times():
    m = th.ThermalModel(25.0)
    assert abs(m.time_to_trip(1.3, dt=0.05) / 60.0 - 17.35) < 0.02
    assert abs(m.time_to_trip(1.1, dt=0.05, t_max=7200.0) / 60.0 - 73.35) < 0.05
    assert math.isinf(m.time_to_trip(1.05, dt=0.05, t_max=7200.0))
    # warmer room trips the same duty sooner (notes/361 F3 direction), from a soak at that ambient
    assert th.ThermalModel(45.0).time_to_trip(1.3, dt=0.05) < th.ThermalModel(35.0).time_to_trip(1.3, dt=0.05) < 17.35 * 60


def test_ambient_is_required_and_finite():
    with pytest.raises(TypeError):
        th.ThermalModel()  # noqa
    with pytest.raises(ValueError):
        th.ThermalModel(float("nan"))
    with pytest.raises(TypeError):
        th.thermal_trace_check(np.ones((10, 1)), np.ones(1), 0.02)  # noqa


def test_trace_matches_reference_law_and_soaks_at_ambient():
    rng = np.random.default_rng(3)
    r = rng.uniform(0.0, 1.8, (2000, 3))
    for amb in (25.0, 45.0):
        got = th.ThermalModel(amb).simulate(r, 0.02)["tw"]
        assert np.abs(got - _reference_trace(r, 0.02, amb)).max() < 1e-9
    idle = th.ThermalModel(45.0).simulate(np.zeros((50, 2)), 0.02)
    assert np.allclose(idle["tw_end"], 45.0) and np.allclose(idle["th_end"], 45.0)


def test_long_run_converges_to_closed_form_steady_state():
    m = th.ThermalModel(35.0)
    s = m.simulate(np.full((1, 1), 0.9), 1.0, loop_s=20000.0, record=False)
    assert abs(float(s["tw_end"][0]) - float(m.steady_winding_C(0.9))) < 0.05
    assert abs(float(s["th_end"][0]) - float(m.steady_housing_C(0.9))) < 0.05


def test_trace_check_value_and_looped_trip():
    dt = 0.02
    tau = np.zeros((500, 2))
    tau[:, 0] = 0.97 * 10.0   # 0.97x rated: continuous at 25 C, not at 45 C
    tau[:, 1] = 1.3 * 10.0
    rated = np.array([10.0, 10.0])
    cool = th.thermal_trace_check(tau, rated, dt, 25.0)
    warm = th.thermal_trace_check(tau, rated, dt, 45.0)
    assert cool["model"]["ambient_C"] == 25.0 and warm["model"]["ambient_C"] == 45.0
    assert cool["value"][0] < 1.0 < warm["value"][0]
    assert math.isinf(cool["trip_s"][0])
    assert abs(cool["trip_s"][1] / 60.0 - 17.35) < 0.02           # 10 s trace looped = constant 1.3x
    assert warm["trip_s"][1] < cool["trip_s"][1]
    assert math.isfinite(float(warm["steady_winding_C"][0])) and float(warm["steady_winding_C"][0]) > 100.0


# ---- telemetry_issue_scan ----
def _telemetry(path, knee_ratio, T=1000, dt=0.02):
    N = len(MOTOR_NAMES)
    knee = MOTOR_NAMES.index("RL_knee_actuator_joint")
    peak = np.full(N, 25.0)
    rated = peak / 2.0
    tau = np.zeros((T, N))
    tau[:, knee] = knee_ratio * rated[knee]
    q = np.zeros((T, N))
    np.savez_compressed(path, dt=dt, motor_names=np.array(MOTOR_NAMES), peak_torque=peak, rated_torque=rated,
                        q=q, qd=q.copy(), q_target=q.copy(), tau=tau, tau_pd=tau.copy(),
                        anchor_z=np.full(T, 1.4), meta=np.array(json.dumps({"actuator_profile": "hw_v1"})),
                        joint_limits=np.tile(np.radians([-60.0, 60.0]), (N, 1)))
    return path


def _thermal(r, sev=None):
    return [f for f in r["findings"] if f["check"] == "thermal" and (sev is None or f["severity"] == sev)]


def test_scan_thermal_is_ambient_aware(tmp_path):
    p = _telemetry(tmp_path / "t.npz", 0.97)
    cool = tis.scan(p, ambient_C=25.0)
    warm = tis.scan(p)  # default = design ambient 45 C
    assert tis.SUSTAIN_S == 5.0
    assert th.DESIGN_AMBIENT_C == 45.0 and warm["thermal"]["ambient_C"] == 45.0 and cool["thermal"]["ambient_C"] == 25.0
    assert not _thermal(cool)
    f = _thermal(warm)
    assert len(f) == 1 and f[0]["motor"] == "RL_knee_actuator_joint" and f[0]["severity"] == "warn"
    assert abs(f[0]["value"] - 0.97 / 0.9339) < 1e-3 and "45 C ambient" in f[0]["detail"]
    pm = warm["thermal"]["per_motor"]["RL_knee_actuator_joint"]
    assert pm["steady_winding_C"] > 100.0 and abs(pm["rms_over_rated"] - 0.97) < 1e-9
    assert cool["thermal"]["per_motor"]["RL_knee_actuator_joint"]["time_to_trip_s"] is None
    assert warm["hw_gate"]["pass"] and cool["hw_gate"]["pass"]


def test_scan_thermal_high_fails_gate_with_trip_time(tmp_path):
    p = _telemetry(tmp_path / "t.npz", 1.3)
    r = tis.scan(p, ambient_C=25.0)
    assert not _thermal(r, "HIGH")  # 1.3 / 1.09 = 1.19 < 1.3 at 25 C: warn only
    pm = r["thermal"]["per_motor"]["RL_knee_actuator_joint"]
    assert abs(pm["time_to_trip_s"] / 60.0 - 17.35) < 0.02
    hot = tis.scan(_telemetry(tmp_path / "h.npz", 1.25), ambient_C=45.0)
    assert [f["motor"] for f in _thermal(hot, "HIGH")] == ["RL_knee_actuator_joint"]  # 1.25 / 0.934 = 1.34
    assert "thermal:RL_knee_actuator_joint" in hot["hw_gate"]["fails"]


def test_scan_cli_records_ambient(tmp_path):
    p = _telemetry(tmp_path / "t.npz", 0.5, T=200)
    out = tmp_path / "o.json"
    subprocess.run([sys.executable, str(ROOT / "tools" / "telemetry_issue_scan.py"), str(p), "--json", str(out),
                    "--ambient_C", "35", "--limits_json", str(tmp_path / "none.json")], check=True, capture_output=True)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d[0]["thermal"]["ambient_C"] == 35.0
    assert abs(d[0]["thermal"]["max_continuous_ratio"] - 1.0152) < 1e-3


# ---- scoreboard ----
def _summary(ratio, motor="LL_knee_actuator_joint", profile="hw_v1"):
    specs = sb.motor_specs(profile)
    per = {n: {"peak_limit_Nm": sp["peak"], "max_Nm": 0.5 * sp["peak"], "rms_Nm": 0.3 * sp["rated"], "clip_frac": 0.0,
               "max_speed_rad_s": 0.5 * sp["no_load"]} for n, sp in specs.items()}
    per[motor]["rms_Nm"] = ratio * specs[motor]["rated"]
    return {"finite": True, "falls_total": 0, "actuator_profile": profile, "eval_design": {"perturbed": True},
            "by_scenario": {"fwd": {"envs": 2, "motor_torque": {"per_motor": per}}}, "motor_torque": {"per_motor": per}}


def test_scoreboard_rms_gate_uses_ambient_continuous_limit():
    s = _summary(0.97)
    assert sb.evaluate("locomotion", s, ambient_C=25.0)["pass"]
    r = sb.evaluate("locomotion", s)  # default 45 C
    assert not r["pass"] and r["thermal"]["ambient_C"] == 45.0 and abs(r["thermal"]["rms_limit"] - 0.9339) < 1e-3
    assert any("rms 0.97x rated (LL_knee_actuator_joint" in w and "continuous at 45 C ambient" in w for w in r["why"])
    assert sb.evaluate("locomotion", _summary(1.05), ambient_C=25.0)["pass"]  # below 1.09 at 25 C
    assert not sb.evaluate("locomotion", _summary(1.10), ambient_C=25.0)["pass"]
    assert abs(sb.rms_limit(35.0) - 1.0152) < 1e-3
