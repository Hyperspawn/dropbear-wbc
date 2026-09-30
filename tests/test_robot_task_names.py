"""CPU tests for dropbear_wbc.robots.dropbear_names / defaults (no Isaac). Run: python -m pytest tests -q"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.robots import dropbear_names as N  # noqa: E402
from dropbear_wbc.robots.defaults import (  # noqa: E402
    CalibrationError,
    load_standing_motor_pos,
    resolve_default_pose,
)

USD_TREE = json.loads((REPO / "docs" / "usd_tree_45586414.json").read_text(encoding="utf-8"))
USD_BODIES = [b[0] for b in USD_TREE["bodies"]]

CONTRACT_MOTORS = [
    "PG_left_leg_pitch", "PG_left_leg_roll", "LL_hip_joint", "LL_knee_actuator_joint", "LL_Revolute67", "LL_Revolute81",
    "PG_right_leg_pitch", "PG_right_leg_roll", "RL_hip_joint", "RL_knee_actuator_joint", "RL_Revolute67", "RL_Revolute81",
    "LH_yaw", "LH_pitch", "LH_roll", "LH_elbow_joint", "LH_wrist_roll",
    "RH_yaw", "RH_pitch", "RH_roll", "RH_elbow_joint", "RH_wrist_roll",
]


def isaac_dof_names() -> list[str]:
    """Articulation DOF names implied by the USD tree: non-excluded revolute/prismatic joints (1 DOF) and
    spherical joints (3 DOFs named ``<joint>:0..2``); fixed joints contribute none."""
    names = []
    for j in USD_TREE["joints"]:
        if j["excl"]:
            continue
        if j["type"] in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            names.append(j["name"])
        elif j["type"] == "PhysicsSphericalJoint":
            names += [f"{j['name']}:{k}" for k in range(3)]
    return names


def test_motor_names_match_contract():
    assert list(N.MOTOR_NAMES) == CONTRACT_MOTORS
    assert len(set(N.MOTOR_NAMES)) == 22


def test_motor_names_match_sdk_module_if_present():
    try:
        from dropbear_wbc.sdk import motors as sdk_motors
    except Exception:  # noqa: BLE001
        pytest.skip("sdk.motors not importable")
    assert tuple(sdk_motors.MOTOR_NAMES) == N.MOTOR_NAMES
    assert tuple(sdk_motors.NECK_NAMES) == N.NECK_NAMES


def test_dof_partition_motor_neck_passive():
    dofs = isaac_dof_names()
    assert len(dofs) == N.EXPECTED_NUM_JOINTS == 91
    passive_res = [N.PASSIVE_LEG_REGEX, N.PASSIVE_ARM_REGEX, N.PASSIVE_HEAD_REGEX]
    counts = {"motor": 0, "neck": 0, "passive": 0}
    for name in dofs:
        hits = [name in N.MOTOR_NAMES, name in N.NECK_NAMES] + [bool(re.fullmatch(r, name)) for r in passive_res]
        assert sum(hits) == 1, (name, hits)
        counts["motor" if hits[0] else "neck" if hits[1] else "passive"] += 1
    assert counts == {"motor": 22, "neck": 6, "passive": N.EXPECTED_NUM_PASSIVE}


def test_bodies_exist_and_are_not_orphans():
    joint_bodies = {b for j in USD_TREE["joints"] for b in j["b0"] + j["b1"]}
    orphans = [b for b in USD_BODIES if b not in joint_bodies]
    assert sorted(orphans) == sorted(N.ORPHAN_BODIES)
    assert len(USD_BODIES) - len(orphans) == N.EXPECTED_NUM_BODIES
    for b in (N.ROOT_BODY, N.ANCHOR_BODY, *N.TRACKED_BODIES, *N.FOOT_BODIES, *N.HAND_BODIES, *N.KEY_BODIES.values()):
        assert b in USD_BODIES and b not in N.ORPHAN_BODIES, b


def test_anchor_fixed_to_root():
    fixed = [j for j in USD_TREE["joints"] if j["type"] == "PhysicsFixedJoint" and not j["excl"]]
    assert any(j["b0"] == [N.ROOT_BODY] and j["b1"] == [N.ANCHOR_BODY] for j in fixed)


def test_tracked_bodies_and_drivers():
    assert len(N.TRACKED_BODIES) == 14 and len(set(N.TRACKED_BODIES)) == 14
    assert N.TRACKED_BODIES[0] == N.ANCHOR_BODY
    assert set(N.EXPECTED_BODY_DRIVERS) == set(N.TRACKED_BODIES)
    for motors in N.EXPECTED_BODY_DRIVERS.values():
        assert set(motors) <= set(N.MOTOR_NAMES)


def test_undesired_contact_regex_excludes_feet_and_hands_only():
    rx = N.undesired_contact_body_regex()
    excluded = [b for b in USD_BODIES if not re.fullmatch(rx, b)]
    assert sorted(excluded) == sorted(N.FOOT_BODIES + N.HAND_BODIES)


def test_anchor_offset_is_unit_and_inverts_rest_rotation():
    w, x, y, z = N.ANCHOR_FRAME_OFFSET_WXYZ
    assert math.isclose(w * w + x * x + y * y + z * z, 1.0, abs_tol=1e-12)
    # rest orientation (0.5, 0.5, 0.5, 0.5) * offset == identity
    a = (0.5, 0.5, 0.5, 0.5)
    b = N.ANCHOR_FRAME_OFFSET_WXYZ
    prod = (
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )
    assert all(math.isclose(p, q, abs_tol=1e-12) for p, q in zip(prod, (1.0, 0.0, 0.0, 0.0)))


def test_action_scale_rule():
    scale = {n: 0.25 * N.motor_param(n, 0) / N.motor_param(n, 1) for n in N.MOTOR_NAMES}
    assert math.isclose(scale["LH_yaw"], 0.2)
    assert math.isclose(scale["PG_left_leg_pitch"], 0.25 * 200 / 150)
    assert math.isclose(scale["LL_knee_actuator_joint"], 0.375)
    assert math.isclose(scale["LL_Revolute67"], 0.25)


def test_legacy_pose_within_hard_limits():
    for name, value in N.LEGACY_DEFAULT_MOTOR_POS.items():
        lo, hi = (math.radians(v) for v in N.MOTOR_HARD_LIMITS_DEG[name])
        assert lo <= value <= hi, name


def test_default_pose_fallback_and_calibration(tmp_path):
    missing = tmp_path / "none.json"
    assert load_standing_motor_pos(missing) is None
    pose = resolve_default_pose(missing)
    assert pose.source == "legacy_DROPBEAR_CFG" and len(pose.as_tuple()) == 22

    good = tmp_path / "cal.json"
    values = [0.0] * 22
    values[0], values[12] = -0.05, 0.1  # PG_left_leg_pitch, LH_yaw
    values[3] = 0.2  # knee within [0, 30 deg]
    values[9] = 0.2
    values[15] = 0.2  # elbows within [0, 30 deg]
    values[20] = 0.2
    good.write_text(json.dumps({"schema": "dropbear-semantic-calibration-v1", "standing_motor_pos": values,
                                "authored_ankle_tierods": False}))
    pose = resolve_default_pose(good)
    assert pose.source.startswith("calibration:") and pose.as_tuple()[3] == pytest.approx(0.2)

    as_dict = tmp_path / "cal_dict.json"
    as_dict.write_text(json.dumps({"standing_motor_pos": dict(zip(N.MOTOR_NAMES, values)),
                                   "authored_ankle_tierods": False}))
    assert load_standing_motor_pos(as_dict)["RH_elbow_joint"] == pytest.approx(0.2)

    # plant variant fail-closed (review fix 2026-09-24): a calibration without the key predates CONTRACTS 0.2 (authored)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"standing_motor_pos": values}))
    with pytest.raises(CalibrationError, match="ankle tie rods"):
        load_standing_motor_pos(legacy)
    assert load_standing_motor_pos(legacy, expected_authored_ankle=True)["RH_elbow_joint"] == pytest.approx(0.2)
    diag = tmp_path / "diag.json"
    diag.write_text(json.dumps({"standing_motor_pos": values, "authored_ankle_tierods": False,
                                "plant_variant": "DIAGNOSTIC: spherical tie rods sweep"}))
    with pytest.raises(CalibrationError, match="DIAGNOSTIC"):
        load_standing_motor_pos(diag)

    bad = tmp_path / "bad.json"
    values_bad = list(values)
    values_bad[3] = -0.5  # knee below 0
    bad.write_text(json.dumps({"standing_motor_pos": values_bad, "authored_ankle_tierods": False}))
    with pytest.raises(CalibrationError):
        load_standing_motor_pos(bad)
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"standing_motor_pos": values[:21], "authored_ankle_tierods": False}))
    with pytest.raises(CalibrationError):
        load_standing_motor_pos(short)
    wrong_schema = tmp_path / "schema.json"
    wrong_schema.write_text(json.dumps({"schema": "other", "standing_motor_pos": values}))
    with pytest.raises(CalibrationError):
        load_standing_motor_pos(wrong_schema)


def test_closure_motors_subset_and_serial_complement():
    assert set(N.CLOSURE_MOTORS) <= set(N.MOTOR_NAMES) and len(N.CLOSURE_MOTORS) == 8
    serial = [m for m in N.MOTOR_NAMES if m not in N.CLOSURE_MOTORS]
    assert len(serial) == 14
    # serial motors must not drive any excluded (loop-closing) joint's bodies directly: their child bodies
    # appear in no excluded joint of the USD tree
    excluded_bodies = {b for j in USD_TREE["joints"] if j["excl"] for b in j["b0"] + j["b1"]}
    children = {j["name"]: j["b1"][0] for j in USD_TREE["joints"] if j["b1"]}
    for m in serial:
        assert children[m] not in excluded_bodies, m


def test_usd_defect_constants_match_tree():
    joints = {j["name"]: j for j in USD_TREE["joints"]}
    assert joints["LL_Revolute121"]["axis"] == "X" and joints["RL_Revolute121"]["axis"] == "Z"
    assert N.JOINT_AXIS_FIXES == {"LL_Revolute121": "Z"}
    assert set(N.USD_JOINT_FRICTION) <= set(joints)


MOCK_CALIBRATION = REPO / "tests" / "fixtures" / "mock_semantic_calibration.json"


@pytest.mark.skipif(not MOCK_CALIBRATION.is_file(), reason="mock calibration fixture not generated")
def test_calibration_json_schema_as_written_by_calib_build(tmp_path, monkeypatch):
    """A calibration shaped like calib_build.fit_calibration's output (list standing_motor_pos, motor_names,
    usd_sha256) becomes the default pose; env var override; fail closed on order / plant mismatches."""
    from dropbear_wbc.robots.defaults import calibration_info, calibration_path

    monkeypatch.delenv("DROPBEAR_USD", raising=False)
    data = json.loads(MOCK_CALIBRATION.read_text(encoding="utf-8"))
    pose = resolve_default_pose(MOCK_CALIBRATION)
    assert pose.source.startswith("calibration:")
    assert pose.as_tuple() == pytest.approx(tuple(data["standing_motor_pos"]))
    assert pose.info["usd_sha256"] == N.USD_SHA256 and len(pose.info["sha256"]) == 64

    monkeypatch.setenv("DROPBEAR_CALIBRATION_JSON", str(MOCK_CALIBRATION))
    assert calibration_path() == MOCK_CALIBRATION
    assert resolve_default_pose().source.startswith("calibration:")
    assert calibration_info()["path"].endswith("mock_semantic_calibration.json")
    monkeypatch.delenv("DROPBEAR_CALIBRATION_JSON")

    swapped = dict(data, motor_names=list(reversed(data["motor_names"])))
    p = tmp_path / "swapped.json"
    p.write_text(json.dumps(swapped))
    with pytest.raises(CalibrationError, match="motor_names"):
        load_standing_motor_pos(p)
    other = dict(data, usd_sha256="0" * 64)
    p = tmp_path / "other.json"
    p.write_text(json.dumps(other))
    with pytest.raises(CalibrationError, match="USD sha"):
        load_standing_motor_pos(p)
    monkeypatch.setenv("DROPBEAR_USD", "X:/other.usd")  # explicit plant override: sha not enforced
    assert load_standing_motor_pos(p) is not None


def test_calibration_env_none_forces_legacy(monkeypatch):
    monkeypatch.setenv("DROPBEAR_CALIBRATION_JSON", "none")
    assert resolve_default_pose().source == "legacy_DROPBEAR_CFG"
