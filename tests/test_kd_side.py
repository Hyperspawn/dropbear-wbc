"""kd_side flag: motion-mode field kd vs IdealPD output damping (notes/13 T5, item 60). CPU, Isaac-free."""
from __future__ import annotations

import math

import pytest

import sdk_test_paths  # noqa: F401

from dropbear_wbc.robots.hw_motor_specs import (  # noqa: E402
    DEFAULT_KD_SIDE,
    HW_GAINS,
    HW_PROFILE_OPTIONS,
    JOINT_ROLES,
    KD_SIDES,
    MOTION_MODE_LIMITS,
    MOTOR_MODELS,
    armature_dr_group_plan,
    joint_hw_params,
    kd_sim_from_field,
    normalize_kd_side,
)


def test_default_kd_side_is_output_and_preserves_field_damping():
    assert DEFAULT_KD_SIDE == "output"
    assert KD_SIDES == ("output", "motor")
    assert normalize_kd_side(None) == "output"
    params = joint_hw_params()
    for name, p in params.items():
        role = JOINT_ROLES[name]
        kp, kd = HW_GAINS[role]
        assert p["kp"] == kp
        assert p["kd"] == kd  # field
        assert p["kd_sim"] == kd
        assert p["kd_output"] == kd
        assert p["kd_side"] == "output"


def test_kd_sim_from_field_conversion_known_ratios():
    # notes/13 T5: tau_out = kd_field * omega_m * N = kd_field * N * omega_out when kd_side=motor
    for N in (7, 9, 35, 30):
        assert kd_sim_from_field(5.0, N, "output") == 5.0
        assert kd_sim_from_field(5.0, N, "motor") == 5.0 * N
        assert kd_sim_from_field(2.0, N, "motor") == 2.0 * N
        assert kd_sim_from_field(4.0, N, "motor") == 4.0 * N
    with pytest.raises(ValueError, match="kd_side"):
        normalize_kd_side("rotor")
    with pytest.raises(ValueError, match="kd_side"):
        kd_sim_from_field(1.0, 7, "both")


def test_motor_side_multiplies_by_gear_ratio_per_model():
    params = joint_hw_params(kd_side="motor")
    # hip roll / yaw: X10 1:7, field kd 5 -> 35
    assert params["PG_left_leg_pitch"]["model"] == "RMD-X10-V3-1:7"
    assert params["PG_left_leg_pitch"]["kd"] == 5.0
    assert params["PG_left_leg_pitch"]["kd_sim"] == pytest.approx(35.0)
    assert params["PG_left_leg_roll"]["kd_sim"] == pytest.approx(35.0)
    # hip pitch: X10-S2 1:35, field 5 -> 175
    assert params["LL_hip_joint"]["kd_sim"] == pytest.approx(175.0)
    # knee CEM-60 1:30, field 5 -> 150
    assert params["LL_knee_actuator_joint"]["kd_sim"] == pytest.approx(150.0)
    # calf / elbow X8 Pro 1:9: calf field 4 -> 36; elbow field 2 -> 18
    assert params["LL_Revolute67"]["kd_sim"] == pytest.approx(36.0)
    assert params["LH_elbow_joint"]["kd_sim"] == pytest.approx(18.0)
    assert params["LH_elbow_joint"]["kd_side"] == "motor"
    # field values unchanged vs output side
    out = joint_hw_params(kd_side="output")
    for name in params:
        assert params[name]["kd"] == out[name]["kd"]
        assert params[name]["kp"] == out[name]["kp"]
        assert params[name]["kd_sim"] == pytest.approx(out[name]["kd"] * params[name]["gear_ratio"])


def test_gain_overrides_recompute_kd_sim_and_stay_field_capped():
    plan = armature_dr_group_plan(
        gain_overrides={"LH_elbow_joint": (200.0, 4.0)}, kd_side="motor"
    )
    # elbow is NOT in implicit set; check via joint_hw_params path used by the plan's override logic
    params = joint_hw_params(kd_side="motor")
    # direct override through plan for an implicit joint (wrist): field 2 * 9 = 18 default; override to 4 -> 36
    plan2 = armature_dr_group_plan(
        gain_overrides={"LH_wrist_roll": (50.0, 4.0)}, kd_side="motor"
    )
    assert plan2["implicit"]["LH_wrist_roll"]["kd_field"] == 4.0
    assert plan2["implicit"]["LH_wrist_roll"]["kd"] == pytest.approx(36.0)  # kd_sim
    assert plan2["kd_side"] == "motor"
    # default armdr plan: field kd on calves (4) -> kd_sim == 4 under output
    plan0 = armature_dr_group_plan()
    assert plan0["implicit"]["LL_Revolute67"]["kd"] == pytest.approx(4.0)
    assert plan0["kd_side"] == "output"
    # field kd > 5 rejected even under motor (do not put kd*N into HW_GAINS/overrides)
    with pytest.raises(ValueError, match="FIELD|motion-mode|outside"):
        armature_dr_group_plan(gain_overrides={"LH_wrist_roll": (50.0, 36.0)}, kd_side="motor")
    # HW_GAINS themselves are legal field values
    joint_hw_params()
    from dropbear_wbc.robots.hw_motor_specs import _check_motion_mode_gains

    with pytest.raises(ValueError, match="FIELD"):
        _check_motion_mode_gains("LH_elbow_joint", 50.0, 18.0, kd_side="motor")
    assert MOTION_MODE_LIMITS["kd_max"] == 5.0
    for role, (kp, kd) in HW_GAINS.items():
        assert kd <= MOTION_MODE_LIMITS["kd_max"]
        assert kp <= MOTION_MODE_LIMITS["kp_max"]
    # profile options may carry kd_side; none do yet (default path)
    for opts in HW_PROFILE_OPTIONS.values():
        if "kd_side" in opts:
            assert opts["kd_side"] in KD_SIDES


def test_hw_v1ie_elbow_override_kd_sim_under_motor():
    """hw_v1ie gain_overrides keep field kd=4; motor side -> 4*9=36 at the elbow joint."""
    from dropbear_wbc.robots.hw_motor_specs import _with_gains

    p = joint_hw_params(kd_side="motor")["LH_elbow_joint"]
    over = _with_gains(p, 200.0, 4.0, "motor")
    assert over["kd"] == 4.0
    assert over["kd_sim"] == pytest.approx(36.0)
    assert over["kp"] == 200.0


def test_damping_dict_helper_matches_pre_patch_under_default():
    """Isaac-free stand-in for make_hw_motor_groups damping=per_joint('kd_sim'): default == HW_GAINS kd."""
    params = joint_hw_params()
    damping = {n: float(params[n]["kd_sim"]) for n in params}
    field = {n: float(HW_GAINS[JOINT_ROLES[n]][1]) for n in params}
    assert damping == field
    motor_d = {n: float(p["kd_sim"]) for n, p in joint_hw_params(kd_side="motor").items()}
    for n, p in params.items():
        assert motor_d[n] == pytest.approx(field[n] * p["gear_ratio"])
    # known N from MOTOR_MODELS
    assert MOTOR_MODELS["RMD-X10-V3-1:7"].gear_ratio == 7
    assert MOTOR_MODELS["RMD-X8-Pro-V2-1:9"].gear_ratio == 9
    assert MOTOR_MODELS["RMD-X10-S2-V3-1:35"].gear_ratio == 35
    assert MOTOR_MODELS["EPS-CEM-60"].gear_ratio == 30
