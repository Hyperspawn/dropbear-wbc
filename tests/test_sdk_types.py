"""Unit tests for the dropbear_hg-v1 message types, CRC and motor contract (CPU only)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk import motors
from dropbear_wbc.sdk.crc import crc32_bytes, crc32_core, crc32_core_bitwise
from dropbear_wbc.sdk.types import (CrcError, IMUState, LowCmd, LowState, MotorCmdBlock, MotorMode, MotorStateBlock,
                                    SimState, quat_wxyz_to_rpy)

ROOT = Path(__file__).resolve().parents[1]


def test_motor_order_matches_contracts_table():
    """MOTOR_NAMES must equal the CONTRACTS.md section 1 table (idx -> USD joint)."""
    text = (ROOT / "docs" / "CONTRACTS.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\d+) \| (\w+) \|", text, flags=re.M)
    table = {int(i): name for i, name in rows}
    right_arm = re.search(r"\| 17–21 \| ([^|]+) \|", text).group(1)
    for k, name in enumerate(n.strip() for n in right_arm.split(",")):
        table[17 + k] = name
    assert [table[i] for i in range(22)] == list(motors.MOTOR_NAMES)


def test_motor_contract_matches_usd_dump():
    """Every motor/neck/passive name exists in the USD dump; motors/neck carry drives; coverage is complete."""
    dump = json.loads((ROOT / "docs" / "usd_tree_45586414.json").read_text(encoding="utf-8"))
    joints = {j["name"]: j for j in dump["joints"] if not j["excl"]}
    movable = {n for n, j in joints.items() if j["type"] != "PhysicsFixedJoint"}
    for name in motors.MOTOR_NAMES + motors.NECK_NAMES:
        assert joints[name]["drive"], name
    assert set(motors.MOTOR_NAMES) | set(motors.NECK_NAMES) | set(motors.PASSIVE_JOINTS) == movable
    assert len(set(motors.MOTOR_NAMES)) == 22 and len(motors.NECK_NAMES) == 6


def test_per_motor_parameters():
    assert motors.EFFORT_LIMIT[motors.motor_index("LL_knee_actuator_joint")] == 300.0
    assert motors.DEFAULT_KP[motors.motor_index("LH_yaw")] == 50.0
    assert motors.DEFAULT_KD[motors.motor_index("RL_Revolute81")] == 4.0
    assert motors.DEFAULT_POS[motors.motor_index("RL_knee_actuator_joint")] == 0.3
    assert len(motors.EFFORT_LIMIT) == len(motors.DEFAULT_KP) == len(motors.DEFAULT_POS) == 22


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_crc_table_equals_unitree_bitwise(seed):
    words = np.random.default_rng(seed).integers(0, 2**32, size=37, dtype=np.uint64).astype(np.uint32)
    assert crc32_core(words) == crc32_core_bitwise(words)


def test_crc_known_vector():
    # Unitree's algorithm is CRC-32/MPEG-2 over the big-endian bytes of each word. Its standard check
    # value is CRC("123456789") = 0x0376E6E7; our word API needs whole words, so verify the byte table
    # through the private helper and the word path against the bitwise transcription.
    from dropbear_wbc.sdk import crc as crc_mod
    c = 0xFFFFFFFF
    for byte in b"123456789":
        c = ((c << 8) & 0xFFFFFFFF) ^ crc_mod._TABLE[((c >> 24) ^ byte) & 0xFF]
    assert c == 0x0376E6E7
    words = np.frombuffer(b"12345678", dtype=">u4").astype(np.uint32)
    assert crc32_core(words) == crc32_core_bitwise(words)
    assert crc32_bytes(b"") == 0xFFFFFFFF


def _random_cmd(rng: np.random.Generator, with_neck: bool) -> LowCmd:
    cmd = LowCmd(tick=int(rng.integers(0, 2**32)), stamp_ns=int(rng.integers(0, 2**62)))
    cmd.motor = MotorCmdBlock.from_arrays(22, mode=rng.integers(0, 2, 22), q=rng.normal(size=22),
                                          dq=rng.normal(size=22), tau=rng.normal(size=22),
                                          kp=rng.uniform(0, 300, 22), kd=rng.uniform(0, 20, 22))
    if with_neck:
        cmd.neck = MotorCmdBlock.from_arrays(6, q=rng.normal(size=6) * 0.01)
    return cmd


@pytest.mark.parametrize("with_neck", [False, True])
def test_lowcmd_roundtrip(with_neck):
    cmd = _random_cmd(np.random.default_rng(3), with_neck)
    back = LowCmd.from_bytes(cmd.to_bytes())
    assert back.tick == cmd.tick and back.stamp_ns == cmd.stamp_ns and back.crc == cmd.crc != 0
    for f in MotorCmdBlock.FIELDS:
        np.testing.assert_array_equal(getattr(back.motor, f), getattr(cmd.motor, f))
    assert (back.neck is None) == (not with_neck)
    if with_neck:
        np.testing.assert_array_equal(back.neck.q, cmd.neck.q)


def test_unitree_style_accessors():
    cmd = LowCmd()
    cmd.motor_cmd[3].q = 0.25
    cmd.motor_cmd[3].kp = 200.0
    cmd.motor_cmd[3].mode = MotorMode.DISABLE
    assert cmd.motor.q[3] == np.float32(0.25) and cmd.motor.kp[3] == 200.0 and cmd.motor.mode[3] == 0
    assert cmd.motor_cmd[-1].q == 0.0
    with pytest.raises(IndexError):
        cmd.motor_cmd[22]
    state = LowState()
    state.motor.q[5] = 1.5
    assert state.motor_state[5].q == 1.5
    assert state.imu_state.quaternion[0] == 1.0


def test_lowstate_roundtrip_with_sim_block():
    rng = np.random.default_rng(7)
    st = LowState(tick=12345, stamp_ns=99)
    st.imu = IMUState(quat_wxyz=np.array([0.9, 0.1, 0.2, 0.3], np.float32), gyro=rng.normal(size=3).astype(np.float32),
                      accel=np.array([0, 0, 9.81], np.float32), rpy=np.array([0.1, 0.2, 0.3], np.float32))
    st.motor = MotorStateBlock(mode=np.ones(22, np.uint8), q=rng.normal(size=22).astype(np.float32),
                               dq=rng.normal(size=22).astype(np.float32), ddq=np.zeros(22, np.float32),
                               tau_est=rng.normal(size=22).astype(np.float32), temperature=np.full(22, 25, np.float32))
    st.neck = MotorStateBlock.zeros(6)
    st.sim = SimState(time_s=1.25, root_pos_w=np.array([1, 2, 3], np.float32), body_names=("world", "anchor"),
                      body_pos_w=np.arange(6, dtype=np.float32).reshape(2, 3),
                      body_quat_w=np.tile(np.array([1, 0, 0, 0], np.float32), (2, 1)))
    back = LowState.from_bytes(st.to_bytes())
    np.testing.assert_array_equal(back.motor.q, st.motor.q)
    np.testing.assert_array_equal(back.imu.accel, st.imu.accel)
    assert back.sim.body_names == ("world", "anchor") and back.sim.time_s == 1.25
    np.testing.assert_array_equal(back.sim.body_pose("anchor")[0], [3, 4, 5])


def test_crc_mismatch_detected():
    cmd = _random_cmd(np.random.default_rng(11), False)
    import msgpack
    d = msgpack.unpackb(cmd.to_bytes(), raw=False)
    q = np.frombuffer(d["motor"]["q"], np.float32).copy()
    q[0] += 1.0
    d["motor"]["q"] = q.tobytes()
    with pytest.raises(CrcError):
        LowCmd.from_bytes(msgpack.packb(d, use_bin_type=True))
    LowCmd.from_bytes(msgpack.packb(d, use_bin_type=True), verify_crc=False)


def test_malformed_rejected():
    import msgpack
    d = msgpack.unpackb(LowCmd().to_bytes(), raw=False)
    d["motor"]["q"] = np.zeros(21, np.float32).tobytes()
    with pytest.raises(ValueError):
        LowCmd.from_bytes(msgpack.packb(d, use_bin_type=True), verify_crc=False)
    with pytest.raises(ValueError):
        LowState.from_bytes(LowCmd().to_bytes())
    with pytest.raises(ValueError):
        MotorCmdBlock.from_arrays(22, q=np.zeros(3))


def test_rpy():
    half = np.deg2rad(30) / 2
    q = np.array([np.cos(half), 0, 0, np.sin(half)])  # yaw 30 deg
    np.testing.assert_allclose(quat_wxyz_to_rpy(q), [0, 0, np.deg2rad(30)], atol=1e-6)
