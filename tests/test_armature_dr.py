"""Armature DR between alt/datasheet reads + T3 stability gate (notes/13 T2-T3, item 58). CPU, Isaac-free."""
from __future__ import annotations

import math

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401

from dropbear_wbc.robots.hw_motor_specs import (  # noqa: E402
    ARMATURE_DR_CRITICAL_JOINTS,
    ARMATURE_DR_IMPLICIT_JOINTS,
    ARMATURE_DR_PHYSICS_DT,
    ARMATURE_DR_SAFE_DT,
    HW_PROFILE_MAPS,
    HW_PROFILE_OPTIONS,
    MOTOR_MODELS,
    armature_dr_group_plan,
    armature_dr_ranges,
    explicit_pd_spectral_radius,
    joint_hw_params,
    joints_unstable_at_alt,
    link_inertia_at_rest,
    sample_armature_log_uniform,
)


def test_armature_alt_matches_notes_13_catalog_read():
    """alt = rotor x N (catalog as already-reflected); datasheet = rotor x N^2."""
    x10s2 = MOTOR_MODELS["RMD-X10-S2-V3-1:35"]
    x10 = MOTOR_MODELS["RMD-X10-V3-1:7"]
    x8 = MOTOR_MODELS["RMD-X8-Pro-V2-1:9"]
    assert x10s2.armature_alt == pytest.approx(0.0198625)
    assert x10.armature_alt == pytest.approx(0.0039725)
    assert x8.armature_alt == pytest.approx(0.00306)
    assert x10s2.armature / x10s2.armature_alt == pytest.approx(x10s2.gear_ratio)
    assert MOTOR_MODELS["EPS-CEM-60"].armature_alt is None
    params = joint_hw_params("user_2026_09_25")
    assert params["LL_Revolute67"]["armature_alt"] == pytest.approx(0.00306)
    assert params["LL_hip_joint"]["armature_alt"] == pytest.approx(0.0198625)


def test_critical_set_and_ranges_are_alt_to_datasheet():
    assert ARMATURE_DR_CRITICAL_JOINTS == (
        "LL_Revolute67", "LL_Revolute81", "RL_Revolute67", "RL_Revolute81",
        "LH_wrist_roll", "RH_wrist_roll", "LH_roll", "RH_roll",
    )
    ranges = armature_dr_ranges()
    assert set(ranges) == set(ARMATURE_DR_CRITICAL_JOINTS)
    params = joint_hw_params()
    for j, (lo, hi) in ranges.items():
        assert lo == pytest.approx(params[j]["armature_alt"])
        assert hi == pytest.approx(params[j]["armature"])
        assert 0.0 < lo < hi
    # CEM knee has no alt -> not in a full-map DR
    full = armature_dr_ranges(joints=list(params))
    assert "LL_knee_actuator_joint" not in full


def test_t3_link_criterion_and_explicit_5ms_instability():
    """notes/13 T3: link < alt*3 on the critical set; calves+wrist unstable at alt/5ms, stable at 2.5ms."""
    for j in ARMATURE_DR_CRITICAL_JOINTS:
        alt = joint_hw_params()[j]["armature_alt"]
        assert link_inertia_at_rest(j) < alt * 3.0
    unstable = joints_unstable_at_alt(dt=ARMATURE_DR_PHYSICS_DT)
    assert unstable == list(ARMATURE_DR_IMPLICIT_JOINTS)
    assert joints_unstable_at_alt(dt=ARMATURE_DR_SAFE_DT) == []
    # calf A at alt: rho_5 > 2, rho_2.5 < 1 (notes/13 B)
    p = joint_hw_params()["LL_Revolute67"]
    J_alt = link_inertia_at_rest("LL_Revolute67") + p["armature_alt"]
    assert explicit_pd_spectral_radius(p["kp"], p["kd"], J_alt, 0.005) == pytest.approx(2.580, abs=0.02)
    assert explicit_pd_spectral_radius(p["kp"], p["kd"], J_alt, 0.0025) < 1.0
    # datasheet armature stays stable at 5 ms
    J_ds = link_inertia_at_rest("LL_Revolute67") + p["armature"]
    assert explicit_pd_spectral_radius(p["kp"], p["kd"], J_ds, 0.005) < 1.0
    # upper-arm roll: critical by T3 link criterion but rho_alt < 1 at 5 ms
    assert "LH_roll" not in unstable
    pr = joint_hw_params()["LH_roll"]
    Jr = link_inertia_at_rest("LH_roll") + pr["armature_alt"]
    assert explicit_pd_spectral_radius(pr["kp"], pr["kd"], Jr, 0.005) < 1.0


def test_armature_dr_group_plan_splits_implicit_calves_wrists():
    plan = armature_dr_group_plan()
    assert set(plan["implicit"]) == set(ARMATURE_DR_IMPLICIT_JOINTS)
    assert set(plan["ranges"]) == set(ARMATURE_DR_CRITICAL_JOINTS)
    # X8 Pro remaining on DatasheetMotor: pitch x2, roll x2, elbow x2 (6); calves+wrists moved out
    assert len(plan["explicit"]["RMD-X8-Pro-V2-1:9"]) == 6
    assert "LH_roll" in plan["explicit"]["RMD-X8-Pro-V2-1:9"]
    assert "LL_Revolute67" not in plan["explicit"]["RMD-X8-Pro-V2-1:9"]
    for j, cfg in plan["implicit"].items():
        lo, hi = cfg["armature_range"]
        assert cfg["armature"] == pytest.approx(math.exp(0.5 * (math.log(lo) + math.log(hi))))
        assert cfg["effort"] == pytest.approx(25.0)  # X8 Pro peak


def test_sample_armature_log_uniform_stays_in_range():
    ranges = armature_dr_ranges()
    samples, names = sample_armature_log_uniform(ranges, 500, rng=np.random.default_rng(0))
    assert names == list(ranges)
    assert samples.shape == (500, 8)
    for i, n in enumerate(names):
        lo, hi = ranges[n]
        assert samples[:, i].min() >= lo * 0.999
        assert samples[:, i].max() <= hi * 1.001
    # log-space mean near geometric mean of the range (coarse)
    log_mean = np.log(samples).mean(axis=0)
    for i, n in enumerate(names):
        lo, hi = ranges[n]
        assert log_mean[i] == pytest.approx(0.5 * (math.log(lo) + math.log(hi)), abs=0.15)


def test_armdr_profiles_registered():
    for p in ("hw_v1_armdr", "hw_v1i_armdr", "hw_v1ie_armdr"):
        assert p in HW_PROFILE_MAPS
        assert HW_PROFILE_OPTIONS[p].get("armature_dr") is True
    assert HW_PROFILE_OPTIONS["hw_v1ie_armdr"]["gain_overrides"]["LH_elbow_joint"] == (200.0, 4.0)
    assert "armature_dr" not in HW_PROFILE_OPTIONS.get("hw_v1", {})
    assert "armature_dr" not in HW_PROFILE_OPTIONS.get("hw_v1ie", {})
