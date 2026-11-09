"""Item 325: LowState motor.tau_src tells a 0x500 ``t`` torque apart from a producer-substituted ``iq * Kt``."""

from __future__ import annotations

import msgpack
import numpy as np
import pytest

import sdk_test_paths  # noqa: F401

from dropbear_wbc.sdk import TAU_SRC_DIRECT, TAU_SRC_IQ_KT, LowState, MotorStateBlock
from dropbear_wbc.sdk.motors import NUM_MOTORS, NUM_NECK
from dropbear_wbc.sdk.types import CrcError, crc32_bytes

KNEES = (3, 9)


def _state(neck: bool = False) -> LowState:
    st = LowState(tick=7, stamp_ns=123)
    st.motor.tau_est[:] = np.linspace(-5, 5, NUM_MOTORS, dtype=np.float32)
    for k in KNEES:
        st.motor.tau_est[k] = 79.0
        st.motor.tau_src[k] = TAU_SRC_IQ_KT
    st.motor.age_us[:] = 900
    if neck:
        st.neck = MotorStateBlock.zeros(NUM_NECK)
    return st


def test_default_is_direct():
    blk = MotorStateBlock.zeros(NUM_MOTORS)
    assert blk.tau_src.dtype == np.uint8 and (blk.tau_src == TAU_SRC_DIRECT).all()


@pytest.mark.parametrize("neck", [False, True])
def test_roundtrip(neck):
    st = _state(neck)
    out = LowState.from_bytes(st.to_bytes())
    assert out.motor.tau_src.tolist() == st.motor.tau_src.tolist()
    assert out.motor[3].tau_est == pytest.approx(79.0)
    if neck:
        assert (out.neck.tau_src == TAU_SRC_DIRECT).all()


def test_crc_covers_tau_src():
    d = msgpack.unpackb(_state().to_bytes(), raw=False)
    src = bytearray(d["motor"]["tau_src"])
    src[3] = TAU_SRC_DIRECT  # flip one knee back to "0x500 t" in transit
    d["motor"]["tau_src"] = bytes(src)
    with pytest.raises(CrcError):
        LowState.from_bytes(msgpack.packb(d, use_bin_type=True))


def test_pre_item_325_sender_decodes_direct_and_crc_ok():
    st = _state()
    st.motor.tau_src[:] = TAU_SRC_DIRECT
    st.validate()
    crc = crc32_bytes(st._canonical(motor_src=False, neck_src=False))
    d = msgpack.unpackb(st.to_bytes(), raw=False)
    del d["motor"]["tau_src"]
    d["crc"] = crc
    out = LowState.from_bytes(msgpack.packb(d, use_bin_type=True))
    assert (out.motor.tau_src == TAU_SRC_DIRECT).all()


def test_pre_item_85_sender_still_decodes():
    st = LowState(tick=1, stamp_ns=2)
    crc = crc32_bytes(st._canonical(motor_health=False))
    d = msgpack.unpackb(st.to_bytes(), raw=False)
    for k in ("tau_valid", "age_us", "tau_src"):
        del d["motor"][k]
    d["crc"] = crc
    out = LowState.from_bytes(msgpack.packb(d, use_bin_type=True))
    assert (out.motor.tau_valid == 1).all() and (out.motor.tau_src == TAU_SRC_DIRECT).all()


def test_tau_src_without_health_rejected():
    d = msgpack.unpackb(_state().to_bytes(), raw=False)
    del d["motor"]["tau_valid"], d["motor"]["age_us"]
    with pytest.raises(ValueError, match="tau_src"):
        LowState.from_bytes(msgpack.packb(d, use_bin_type=True), verify_crc=False)


def test_iq_kt_requires_tau_valid():
    st = _state()
    st.motor.tau_valid[3] = 0
    with pytest.raises(ValueError, match=r"slots \[3\]"):
        st.to_bytes()


def test_unknown_source_rejected():
    st = _state()
    st.motor.tau_src[0] = 2
    with pytest.raises(ValueError, match="tau_src"):
        st.to_bytes()


def test_recorder_declares_column():
    from dropbear_wbc.teleop.recorder import EXTRA_COLUMNS
    assert EXTRA_COLUMNS["observation.motor_tau_src"][0] == 22
