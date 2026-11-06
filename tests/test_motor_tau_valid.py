"""Item 85: per-motor tau_valid / age_us on dropbear_hg-v1 MotorStateBlock."""
from __future__ import annotations

import msgpack
import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk.types import (
    MOTOR_AGE_UNKNOWN,
    MOTION_MODE_KP500_RAIL_ERR_RAD,
    MOTION_MODE_TAU_RAIL_NM,
    CrcError,
    IMUState,
    LowState,
    MotorStateBlock,
    motion_mode_tau_valid,
    tau_est_with_iq_fallback,
)


def _state(tau=None, tau_valid=None, age_us=None) -> LowState:
    st = LowState(tick=7, stamp_ns=100)
    st.imu = IMUState(valid=1, age_us=0)
    n = 22
    kw = dict(
        mode=np.ones(n, np.uint8),
        q=np.zeros(n, np.float32),
        dq=np.zeros(n, np.float32),
        ddq=np.zeros(n, np.float32),
        tau_est=np.zeros(n, np.float32) if tau is None else np.asarray(tau, np.float32),
        temperature=np.full(n, 25, np.float32),
    )
    if tau_valid is not None:
        kw["tau_valid"] = np.asarray(tau_valid, np.uint8)
    if age_us is not None:
        kw["age_us"] = np.asarray(age_us, np.uint32)
    st.motor = MotorStateBlock(**kw)
    return st


def test_rail_constant_and_knee_48_mrad():
    assert MOTION_MODE_TAU_RAIL_NM == 24.0
    assert MOTION_MODE_KP500_RAIL_ERR_RAD == pytest.approx(0.048)
    assert MOTION_MODE_KP500_RAIL_ERR_RAD * 500.0 == pytest.approx(24.0)


def test_motion_mode_tau_valid_mask():
    tau = np.array([-30.0, -24.0, -23.9, 0.0, 23.9, 24.0, 50.0], np.float64)
    got = motion_mode_tau_valid(tau)
    np.testing.assert_array_equal(got, [0, 0, 1, 1, 1, 0, 0])


def test_iq_fallback_replaces_railed_slots():
    tau_est = np.array([10.0, 24.0, -24.0, 5.0], np.float32)
    valid = motion_mode_tau_valid(tau_est)  # [1, 0, 0, 1]
    iq = np.array([1.0, 2.0, -3.0, 0.5], np.float64)
    kt = 8.0  # N*m/A
    out = tau_est_with_iq_fallback(tau_est, valid, iq, kt)
    np.testing.assert_allclose(out, [10.0, 16.0, -24.0, 5.0])


def test_roundtrip_preserves_health():
    tau = np.linspace(-30, 30, 22, dtype=np.float32)
    valid = motion_mode_tau_valid(tau)
    age = np.arange(22, dtype=np.uint32) * 100
    st = _state(tau=tau, tau_valid=valid, age_us=age)
    back = LowState.from_bytes(st.to_bytes())
    np.testing.assert_array_equal(back.motor.tau_est, tau)
    np.testing.assert_array_equal(back.motor.tau_valid, valid)
    np.testing.assert_array_equal(back.motor.age_us, age)
    assert back.motor[0].tau_valid == int(valid[0])
    assert back.motor[3].age_us == 300


def test_legacy_sender_fail_open_without_motor_health():
    """Pre-item-85 frame: no tau_valid/age_us keys -> tau_valid=1, age unknown; CRC over legacy bytes."""
    st = _state(tau=np.full(22, 3.0, np.float32), tau_valid=np.ones(22, np.uint8), age_us=np.zeros(22, np.uint32))
    st.validate()
    d = msgpack.unpackb(st.to_bytes(), raw=False)
    del d["motor"]["tau_valid"], d["motor"]["age_us"]
    d["crc"] = st.compute_crc(motor_health=False)  # what a pre-85 sender computes (imu health still on)
    back = LowState.from_bytes(msgpack.packb(d, use_bin_type=True))
    assert int(back.motor.tau_valid.sum()) == 22
    assert int(back.motor.age_us[0]) == MOTOR_AGE_UNKNOWN
    np.testing.assert_array_equal(back.motor.tau_est, st.motor.tau_est)


def test_partial_health_keys_rejected():
    st = _state()
    d = msgpack.unpackb(st.to_bytes(), raw=False)
    del d["motor"]["age_us"]
    with pytest.raises(ValueError, match="both"):
        LowState.from_bytes(msgpack.packb(d, use_bin_type=True), verify_crc=False)


def test_tampered_tau_valid_fails_crc():
    st = _state(tau_valid=np.ones(22, np.uint8), age_us=np.zeros(22, np.uint32))
    d = msgpack.unpackb(st.to_bytes(), raw=False)
    tv = np.frombuffer(d["motor"]["tau_valid"], np.uint8).copy()
    tv[0] = 0
    d["motor"]["tau_valid"] = tv.tobytes()
    with pytest.raises(CrcError):
        LowState.from_bytes(msgpack.packb(d, use_bin_type=True))


def test_default_zeros_trust_tau_est():
    b = MotorStateBlock.zeros(22)
    assert int(b.tau_valid.sum()) == 22
    assert int(b.age_us[0]) == MOTOR_AGE_UNKNOWN


def test_invalid_tau_valid_rejected_on_encode():
    st = _state()
    st.motor.tau_valid[0] = 2
    with pytest.raises(ValueError, match="tau_valid"):
        st.to_bytes()


def test_docs_mention_rail_and_sample_tick():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    contracts = (root / "docs" / "CONTRACTS.md").read_text(encoding="utf-8")
    sdk = (root / "docs" / "SDK.md").read_text(encoding="utf-8")
    assert "tau_valid" in contracts and "+-24" in contracts.replace("±", "+-").replace("+-24", "+-24")
    assert "48 mrad" in contracts or "0.048" in contracts
    assert "sample" in contracts.lower() and "publish-counted" in contracts
    assert "tau_valid" in sdk and "age_us" in sdk
