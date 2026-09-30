"""The SDK's stdlib-only motor constants must equal the shared contract constants (no silent divergence).

``dropbear_wbc.sdk.motors`` is duplicated on purpose (dependency-free for the SDK wire layer); this test
pins it to ``dropbear_wbc.robots.dropbear_names`` (the CONTRACTS section 1 constant module).
"""
from __future__ import annotations

import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk import motors

names = pytest.importorskip("dropbear_wbc.robots.dropbear_names")


def test_motor_and_neck_names_identical():
    assert motors.MOTOR_NAMES == names.MOTOR_NAMES
    assert motors.NECK_NAMES == names.NECK_NAMES


def test_actuator_parameters_identical():
    for i, m in enumerate(motors.MOTOR_NAMES):
        effort, kp, kd, arm = names.ACTUATOR_PARAMS[names.motor_group(m)]
        assert (motors.EFFORT_LIMIT[i], motors.DEFAULT_KP[i], motors.DEFAULT_KD[i], motors.ARMATURE[i]) == \
            (effort, kp, kd, arm), m
    effort, kp, kd, arm = names.ACTUATOR_PARAMS["neck"]
    assert (motors.NECK_EFFORT_LIMIT, motors.NECK_KP, motors.NECK_KD, motors.NECK_ARMATURE) == (effort, kp, kd, arm)
    _, _, pd, parm = names.ACTUATOR_PARAMS["passive"]
    assert (motors.LEGACY_PASSIVE_DAMPING, motors.PASSIVE_ARMATURE) == (pd, parm)
    # the value the simulators use is the one Isaac actually applies (CONTRACTS 0.3; logs/review_fixes/passive_damping)
    assert motors.PASSIVE_DAMPING == getattr(names, "EFFECTIVE_PASSIVE_DAMPING", 0.0) == 0.0


def test_legacy_default_pose_identical():
    assert motors.DEFAULT_POS == tuple(names.LEGACY_DEFAULT_MOTOR_POS[m] for m in motors.MOTOR_NAMES)


def test_passive_joint_count_matches_isaac_dof_count():
    spherical = {"LL_Revolute115", "LL_Revolute117", "RL_Revolute115", "RL_Revolute117"}
    dofs = sum(3 if j in spherical else 1 for j in motors.PASSIVE_JOINTS)
    assert dofs == names.EXPECTED_NUM_PASSIVE == 63
