"""Item 356: explicit DatasheetMotor PD must be discretely stable at the 5 ms step before a group is built.

kd_side="motor" multiplies the field kd by the gear ratio; on light motor-space inertia the explicit damping then
over-shoots (c*h/I > 2) and diverges. Every published hw profile (kd_side "output") passes; under "motor" every
profile is refused, armature_dr included (the knee crank stays explicit: kd_sim 150 on 0.063 + 0.24 kg*m^2)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from dropbear_wbc.robots import hw_motor_specs as S

PROFILES = sorted(S.HW_PROFILE_MAPS)


def _opts(profile):
    o = S.HW_PROFILE_OPTIONS.get(profile, {})
    return {"gain_overrides": o.get("gain_overrides"), "armature_dr": o.get("armature_dr", False)}


def _numpy_rho(kp, kd, inertia, dt=0.005):
    K, C = kp / inertia, kd / inertia
    a = np.array([[1 - K * dt * dt, dt * (1 - C * dt)], [-K * dt, 1 - C * dt]])
    return float(max(abs(np.linalg.eigvals(a))))


@pytest.mark.parametrize("profile", PROFILES)
def test_published_profiles_are_pd_stable(profile):
    tab = S.explicit_pd_stability(S.HW_PROFILE_MAPS[profile], **_opts(profile))
    assert tab and max(v["rho"] for v in tab.values()) < S.PD_RHO_MAX
    S.check_explicit_pd_stability(S.HW_PROFILE_MAPS[profile], **_opts(profile))


@pytest.mark.parametrize("profile", PROFILES)
def test_kd_side_motor_is_refused_on_every_profile(profile):
    with pytest.raises(ValueError, match="item 356"):
        S.check_explicit_pd_stability(S.HW_PROFILE_MAPS[profile], kd_side="motor", **_opts(profile))


def test_calf_and_knee_numbers():
    tab = S.explicit_pd_stability("user_2026_09_25", kd_side="motor")
    calf = tab["LL_Revolute67"]
    assert calf["kd_sim"] == pytest.approx(36.0) and calf["inertia"] == pytest.approx(0.02754, abs=1e-6)
    assert calf["kd_sim"] * 0.005 / calf["inertia"] == pytest.approx(6.55, abs=0.05)
    assert calf["rho"] == pytest.approx(5.598, abs=1e-3)
    armdr = S.explicit_pd_stability("user_2026_09_25", kd_side="motor", armature_dr=True)
    assert not set(S.ARMATURE_DR_IMPLICIT_JOINTS) & set(armdr)
    knee = armdr["LL_knee_actuator_joint"]
    assert knee["link"] == pytest.approx(0.24) and knee["rho"] > 1.4  # link inertia included, still unstable
    assert S.explicit_pd_stability("user_2026_09_25")["LL_knee_actuator_joint"]["rho"] < 1.0


def test_rho_matches_eigenvalues_and_link_default():
    for side in ("output", "motor"):
        for j, v in S.explicit_pd_stability("cem6_2026_10_16", kd_side=side).items():
            assert v["rho"] == pytest.approx(_numpy_rho(v["kp"], v["kd_sim"], v["inertia"]), rel=1e-9), (side, j)
            assert v["link"] == S.STOP_LINK_INERTIA_MIN.get(j, 0.0)
            assert v["inertia"] == pytest.approx(v["armature"] + v["link"])


def test_unknown_override_rejected():
    with pytest.raises(ValueError, match="unknown motor"):
        S.explicit_pd_stability(gain_overrides={"nope": (1.0, 1.0)})


def test_group_builder_refuses_and_allow_flag(monkeypatch):
    pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        compat = importlib.import_module("test_datasheet_motor_lab_compat")
    finally:
        sys.path.pop(0)
    for name, mod in compat._make_lab("2.2").items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "dropbear_wbc.robots.hw_actuators", raising=False)
    try:
        H = importlib.import_module("dropbear_wbc.robots.hw_actuators")
        assert H.make_hw_profile_groups("hw_cem6")
        with pytest.raises(ValueError, match="LL_Revolute67 rho 5.60"):
            H.make_hw_motor_groups("user_2026_09_25", kd_side="motor")
        assert H.make_hw_motor_groups("user_2026_09_25", kd_side="motor", allow_unstable_pd=True)
        with pytest.raises(ValueError, match="knee_actuator"):
            H.make_hw_motor_groups("user_2026_09_25", kd_side="motor", armature_dr=True)
    finally:
        sys.modules.pop("dropbear_wbc.robots.hw_actuators", None)


def test_gain_overrides_enter_the_check():
    tab = S.explicit_pd_stability(S.HW_PROFILE_MAPS["hw_v1ie"], **_opts("hw_v1ie"))
    assert tab["LH_elbow_joint"]["kp"] == pytest.approx(200.0)
    hot = S.explicit_pd_stability(gain_overrides={"LH_wrist_roll": (50.0, 36.0)})["LH_wrist_roll"]
    assert hot["kd_sim"] == pytest.approx(36.0) and hot["rho"] > 5.0
    with pytest.raises(ValueError, match="LH_wrist_roll"):
        S.check_explicit_pd_stability(gain_overrides={"LH_wrist_roll": (50.0, 36.0)})
