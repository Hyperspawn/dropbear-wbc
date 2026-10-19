"""The vendor CEM-60 knee (twin v2.1, 2026-10-14) and the profiles built on it (docs/archive/ACTUATORS_HISTORY.md section 14)."""
import math

import numpy as np

import sdk_test_paths  # noqa: F401
from dropbear_wbc.robots.hw_motor_specs import (
    HW_PROFILE_MAPS,
    HW_PROFILE_OPTIONS,
    MOTOR_MODELS,
    hw_motor_torque,
    joint_hw_params,
)

RPM = 2.0 * math.pi / 60.0
KNEES = ("LL_knee_actuator_joint", "RL_knee_actuator_joint")


def test_vendor_cem60_matches_the_documents():
    m = MOTOR_MODELS["EPS-CEM-60-V2"]
    assert (m.gear_ratio, m.peak_torque, m.rated_torque, m.mass_kg, m.status) == (30, 75.0, 60.0, 0.93, "vendor")
    # the torque-speed line passes the rated point (60 N*m at 60 rpm), keeps the full peak to 55 rpm, ends at 80 rpm
    top = lambda w: float(hw_motor_torque(1e4, w, m.peak_torque, m.saturation_effort, m.no_load_speed, 0.0, 0.0)[1])
    assert math.isclose(top(60 * RPM), 60.0, rel_tol=1e-9)
    assert math.isclose(top(55 * RPM), 75.0, rel_tol=1e-9)
    assert math.isclose(top(80 * RPM), 0.0, abs_tol=1e-9)
    assert top(0.0) == 75.0
    assert math.isclose(m.armature, 0.7e-4 * 30**2)


def test_v2_profiles_change_only_the_knees():
    for v1, v2 in (("hw_v1", "hw_v2"), ("hw_v1i", "hw_v2i"), ("hw_v1ie", "hw_v2ie")):
        a, b = joint_hw_params(HW_PROFILE_MAPS[v1]), joint_hw_params(HW_PROFILE_MAPS[v2])
        assert {n for n in a if a[n] != b[n]} == set(KNEES)
        assert all(b[k]["model"] == "EPS-CEM-60-V2" for k in KNEES)
        assert HW_PROFILE_OPTIONS.get(v1) == HW_PROFILE_OPTIONS.get(v2)
    r50, v2 = joint_hw_params(HW_PROFILE_MAPS["hw_v2ie_r50"]), joint_hw_params(HW_PROFILE_MAPS["hw_v2ie"])
    for k in KNEES:
        diff = {f for f in r50[k] if r50[k][f] != v2[k][f]}
        assert diff == {"name", "model", "gear_ratio", "armature"}
        assert np.isclose(r50[k]["armature"], 0.7e-4 * 50**2)


def test_published_profiles_keep_the_provisional_knee():
    # every policy published on 2026-09-30 was trained on hw_v1*; they must replay with the knee they were trained on
    for prof in ("hw_v1", "hw_v1i", "hw_v1ie"):
        p = joint_hw_params(HW_PROFILE_MAPS[prof])
        assert all(p[k]["model"] == "EPS-CEM-60" and p[k]["peak_torque"] == 60.0 for k in KNEES)


HEAVY = ("PG_left_leg_pitch", "PG_right_leg_pitch", "LL_hip_joint", "RL_hip_joint") + KNEES


def test_v4_candidates_match_the_datasheets():
    x8 = MOTOR_MODELS["RMD-X8-P20-120-V4"]
    assert (x8.gear_ratio, x8.peak_torque, x8.rated_torque, x8.mass_kg) == (19.61, 120.0, 43.0, 1.40)
    x10 = MOTOR_MODELS["RMD-X10-P20-200-V4.1"]
    assert (x10.gear_ratio, x10.peak_torque, x10.rated_torque, x10.mass_kg) == (20, 170.0, 60.0, 1.64)
    top = lambda m, w: float(hw_motor_torque(1e4, w, m.peak_torque, m.saturation_effort, m.no_load_speed, 0, 0)[1])
    assert abs(top(x8, 127 * RPM) - 43.0) < 0.5  # rated point on the torque-speed line
    # the 48 V sweep holds ~73 rpm up to 84 N*m: the line gives the rated 60 N*m up to 72.7 rpm (rated 74)
    assert top(x10, 72.5 * RPM) >= 60.0
    assert abs(top(x10, 65 * RPM) - 124.0) < 1.0


def test_twin_v3_builds_change_only_the_six_heavy_joints():
    base = joint_hw_params(HW_PROFILE_MAPS["hw_v1ie"])
    cem6 = joint_hw_params(HW_PROFILE_MAPS["hw_cem6"])
    v4 = joint_hw_params(HW_PROFILE_MAPS["hw_v4"])
    assert {n for n in base if base[n]["model"] != cem6[n]["model"]} == set(HEAVY)
    assert all(cem6[n]["model"] == "EPS-CEM-60-V2" for n in HEAVY)
    assert {n for n in base if base[n]["model"] != v4[n]["model"]} == set(HEAVY)
    assert all(v4[n]["model"] == "RMD-X8-P20-120-V4" for n in HEAVY[:4])
    assert all(v4[n]["model"] == "RMD-X10-P20-200-V4.1" for n in KNEES)
    for p in ("hw_cem6", "hw_v4"):
        assert HW_PROFILE_OPTIONS[p] == HW_PROFILE_OPTIONS["hw_v2ie"]


def test_motor_mass_offsets_swap_the_cad_x10s2():
    from dropbear_wbc.robots.hw_motor_specs import motor_mass_offsets

    cem6, v4 = motor_mass_offsets("hw_cem6"), motor_mass_offsets("hw_v4")
    assert abs(sum(cem6.values()) - 6 * (0.93 - 1.70)) < 1e-6
    assert abs(sum(v4.values()) - (4 * (1.40 - 1.70) + 2 * (1.64 - 1.70))) < 1e-6
    assert set(cem6) == {"world", "LL_RMD_X10_S2_MIR4__3_Stator_1", "RL_RMD_X10_S2_MIR4__3_Stator_1"}
    for old in ("hw_v1ie", "hw_v2ie", "hw_v2ie_r50"):  # published policies keep their plant
        assert motor_mass_offsets(old) == {}
