"""Item 137 (local patch 0004): IMUState valid/age_us contract and the deploy FSM IMU-health gate (CPU only)."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

import deploy_fixtures as fx
import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy.config import ObsTermCfg, _fsm_from, hold_config
from dropbear_wbc.deploy.fsm import DeployController, FsmState
from dropbear_wbc.newton_sim.imu import LowStateAssembler
from dropbear_wbc.newton_sim.sensor_gate import PolicyGatedSensorModel, make_sensor_model
from dropbear_wbc.sdk import motors
from dropbear_wbc.sdk.types import IMU_AGE_UNKNOWN, CrcError, IMUState, LowState

ROOT = sdk_test_paths.ROOT
DEF = np.asarray(motors.DEFAULT_POS)


class CountingPolicy:
    """Stand-in for OnnxPolicy: zero actions, counts evaluations."""

    has_reference = False

    def __init__(self):
        self.calls = 0

    def act(self, obs, time_step=None):
        self.calls += 1
        return np.zeros(motors.NUM_MOTORS, np.float32)


def _ctrl(**fsm):
    cfg = hold_config()
    cfg.observations = [ObsTermCfg("g", "projected_gravity")]
    for k, v in fsm.items():
        setattr(cfg.fsm, k, v)
    pol = CountingPolicy()
    return DeployController(cfg, pol), pol


def _to_hold(ctrl, t0=0.0, tick0=0):
    st = fx.low_state(tick0, DEF)
    ctrl.request(FsmState.MOVE_TO_DEFAULT, st, t0)
    ctrl.step(st, t0 + 2.0)
    assert ctrl.state == FsmState.HOLD
    return t0 + 2.0


# ----------------------------------------------------------------------------- SDK contract

def test_default_imustate_is_invalid_not_upright_forever():
    imu = IMUState()
    assert imu.valid == 0 and imu.age_us == IMU_AGE_UNKNOWN and imu.age_s == float("inf")
    assert LowState().imu.valid == 0
    np.testing.assert_array_equal(imu.quat_wxyz, [1, 0, 0, 0])  # placeholder, flagged invalid


def test_wire_roundtrip_and_crc_cover_health():
    st = fx.low_state(7, DEF, imu_valid=1, imu_age_us=12345)
    back = LowState.from_bytes(st.to_bytes())
    assert back.imu.valid == 1 and back.imu.age_us == 12345
    d = msgpack.unpackb(st.to_bytes(), raw=False)
    assert d["imu"]["valid"] == 1 and d["imu"]["age_us"] == 12345
    d["imu"]["valid"] = 0  # flip health without updating the CRC
    with pytest.raises(CrcError):
        LowState.from_bytes(msgpack.packb(d, use_bin_type=True))
    d["imu"]["valid"], d["imu"]["age_us"] = 1, 999
    with pytest.raises(CrcError):
        LowState.from_bytes(msgpack.packb(d, use_bin_type=True))


def test_legacy_sender_decodes_invalid_with_legacy_crc():
    st = fx.low_state(3, DEF, imu_valid=1, imu_age_us=0)
    st.validate()
    st.crc = st.compute_crc(imu_health=False)  # what a pre-0004 sender computes
    d = msgpack.unpackb(st.to_bytes(), raw=False)
    d["crc"] = st.compute_crc(imu_health=False)
    del d["imu"]["valid"], d["imu"]["age_us"]
    back = LowState.from_bytes(msgpack.packb(d, use_bin_type=True))
    assert back.imu.valid == 0 and back.imu.age_us == IMU_AGE_UNKNOWN  # fails closed
    d2 = msgpack.unpackb(st.to_bytes(), raw=False)
    del d2["imu"]["age_us"]
    with pytest.raises(ValueError, match="both"):
        LowState.from_bytes(msgpack.packb(d2, use_bin_type=True), verify_crc=False)


@pytest.mark.parametrize("valid,age", [(2, 0), (1, -1), (1, 2**32)])
def test_health_fields_validated(valid, age):
    st = fx.low_state(1, DEF, imu_valid=valid, imu_age_us=age)
    with pytest.raises(ValueError):
        st.to_bytes()


# ----------------------------------------------------------------------------- FSM gate

def test_dead_imu_refuses_policy():
    ctrl, pol = _ctrl()
    t = _to_hold(ctrl)
    dead = fx.low_state(10, DEF, imu_valid=0, imu_age_us=IMU_AGE_UNKNOWN)
    assert ctrl.request(FsmState.POLICY, dead, t) is False
    _, info = ctrl.step(dead, t)
    assert ctrl.state == FsmState.HOLD and info.transition == "hold->policy refused(imu_invalid)"
    assert ctrl.request(FsmState.POLICY, fx.low_state(20, DEF, imu_valid=0), t + 0.02) is False
    _, info = ctrl.step(fx.low_state(20, DEF, imu_valid=0), t + 0.02)
    assert info.transition is None  # refusal reported once per distinct fault
    assert pol.calls == 0


def test_default_lowstate_cannot_enter_or_keep_policy():
    """The notes/38 F1 bug: a never-filled LowState (identity quat) used to read as upright forever."""
    ctrl, pol = _ctrl()
    t = _to_hold(ctrl)
    blank = LowState(tick=11)
    blank.motor.q[:] = DEF.astype(np.float32)
    assert ctrl.request(FsmState.POLICY, blank, t) is False
    assert ctrl.request(FsmState.POLICY, fx.low_state(12, DEF), t) is True
    ctrl.step(fx.low_state(12, DEF), t)
    assert ctrl.state == FsmState.POLICY and pol.calls == 1
    blank2 = LowState(tick=13)
    cmd, info = ctrl.step(blank2, t + 0.02)
    assert info.transition == "policy->passive(imu_invalid)" and ctrl.state == FsmState.PASSIVE
    assert pol.calls == 1  # policy NOT evaluated on the dead sample
    np.testing.assert_allclose(cmd.motor.kp, 0.0)  # passive command in the same step


def test_valid_fresh_imu_allows_and_keeps_policy():
    ctrl, pol = _ctrl()
    t = _to_hold(ctrl)
    assert ctrl.request(FsmState.POLICY, fx.low_state(10, DEF, imu_age_us=25_000), t)
    for k in range(50):
        _, info = ctrl.step(fx.low_state(10 + 10 * k, DEF, imu_age_us=33_000), t + 0.02 * k)
        assert info.state == FsmState.POLICY and info.extra["imu_fault"] is None
    assert pol.calls == 50 and ctrl.state == FsmState.POLICY


def test_stale_age_refuses_and_drops_to_passive():
    ctrl, pol = _ctrl()
    t = _to_hold(ctrl)
    assert not ctrl.request(FsmState.POLICY, fx.low_state(10, DEF, imu_age_us=61_000), t)
    assert ctrl.request(FsmState.POLICY, fx.low_state(20, DEF, imu_age_us=60_000), t)  # boundary: <= max passes
    ctrl.step(fx.low_state(20, DEF, imu_age_us=60_000), t)
    _, info = ctrl.step(fx.low_state(30, DEF, imu_age_us=80_000), t + 0.02)
    assert info.transition == "policy->passive(imu_stale)" and ctrl.state == FsmState.PASSIVE


def test_unknown_age_is_stale():
    ctrl, _ = _ctrl()
    t = _to_hold(ctrl)
    assert not ctrl.request(FsmState.POLICY, fx.low_state(10, DEF, imu_age_us=IMU_AGE_UNKNOWN), t)


def test_reused_lowstate_tick_ages_out():
    """Wall-clock runner re-uses the last LowState when none arrives: its age must grow with controller time.
    With the state freshness gate on, re-use is attributed to state_stale (item 279, tests/test_fsm_stale_attribution.py);
    the IMU-only path is what still runs with the state gate off."""
    ctrl, pol = _ctrl(state_gate=False)
    t = _to_hold(ctrl)
    st = fx.low_state(10, DEF, imu_age_us=10_000)
    assert ctrl.request(FsmState.POLICY, st, t)
    for k in range(3):  # 10 + 0/20/40 ms <= 60 ms
        _, info = ctrl.step(st, t + 0.02 * k)
        assert ctrl.state == FsmState.POLICY
    _, info = ctrl.step(st, t + 0.06)  # 10 + 60 = 70 ms
    assert info.transition == "policy->passive(imu_stale)"
    assert info.extra["imu_age_s"] == pytest.approx(0.07)


def test_frozen_detector_optional():
    ctrl, pol = _ctrl(imu_frozen_steps=5)
    t = _to_hold(ctrl)
    assert ctrl.request(FsmState.POLICY, fx.low_state(10, DEF, gyro=(0.01, 0, 0)), t)
    for k in range(1, 6):
        _, info = ctrl.step(fx.low_state(10 + 10 * k, DEF, gyro=(0.01, 0, 0)), t + 0.02 * k)
    assert info.transition == "policy->passive(imu_frozen)"
    off, _ = _ctrl()  # default 0 = off: constant noise-free sim data is legal
    t = _to_hold(off)
    off.request(FsmState.POLICY, fx.low_state(10, DEF), t)
    for k in range(1, 40):
        off.step(fx.low_state(10 + 10 * k, DEF), t + 0.02 * k)
    assert off.state == FsmState.POLICY


def test_gate_disable_restores_legacy_behaviour():
    ctrl, _ = _ctrl(imu_gate=False)
    t = _to_hold(ctrl)
    assert ctrl.request(FsmState.POLICY, LowState(tick=10), t)


def test_fsm_yaml_keys():
    f = _fsm_from({"imu_gate": False, "imu_max_age_ms": 35, "imu_frozen_steps": 25})
    assert f.imu_gate is False and f.imu_max_age_s == pytest.approx(0.035) and f.imu_frozen_steps == 25
    assert _fsm_from({"imu_max_age_ms": None}).imu_max_age_s is None
    d = _fsm_from(None)
    assert d.imu_gate and d.imu_max_age_s == pytest.approx(0.060) and d.imu_frozen_steps == 0
    with pytest.raises(ValueError):
        _fsm_from({"imu_frozen_steps": -1})


def test_runner_flags():
    spec = importlib.util.spec_from_file_location("policy_runner", ROOT / "tools" / "policy_runner.py")
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    cfg, _ = pr.build(pr.parse_args(["--imu-max-age-ms", "35", "--imu-frozen-steps", "25"]))
    assert cfg.fsm.imu_max_age_s == pytest.approx(0.035) and cfg.fsm.imu_frozen_steps == 25 and cfg.fsm.imu_gate
    cfg, _ = pr.build(pr.parse_args(["--imu-max-age-ms", "0", "--no-imu-gate"]))
    assert cfg.fsm.imu_max_age_s is None and not cfg.fsm.imu_gate


# ----------------------------------------------------------------------------- sim side

def _readout(tick):
    return SimpleNamespace(tick=tick, time_s=tick * 0.002, motor_q=np.zeros(22), motor_dq=np.zeros(22),
                           motor_tau=np.zeros(22), neck_q=np.zeros(6), neck_dq=np.zeros(6),
                           root_pos_w=np.array([0, 0, 1.0]), root_quat_wxyz=np.array([1.0, 0, 0, 0]),
                           root_lin_vel_w=np.zeros(3), root_ang_vel_w=np.zeros(3), body_pos_w=np.zeros((0, 3)),
                           body_quat_wxyz=np.zeros((0, 4)))


def test_sim_assembler_sets_valid_and_zero_age():
    st = LowStateAssembler(0.002).build(_readout(1), np.ones(22))
    assert st.imu.valid == 1 and st.imu.age_us == 0
    assert LowState.from_bytes(st.to_bytes()).imu.valid == 1


def test_sim_assembler_reports_modelled_age_with_gate_hold():
    gate = PolicyGatedSensorModel(make_sensor_model("motion_mode", seed=3, policy_hz=50.0), 10)
    asm = LowStateAssembler(0.002, sensor_model=gate)
    lag_s = gate.ep.imu_lag * 0.02
    ages = [asm.build(_readout(k), np.ones(22)).imu.age_us for k in range(25)]
    for k, a in enumerate(ages):
        assert a == pytest.approx(1e6 * (lag_s + (k % 10) * 0.002), abs=1)


def test_default_gate_does_not_false_trip_sim_profiles():
    worst = 0
    for name in ("motion_mode", "center_0010", "a1_reply", "deployed_054"):
        for seed in range(40):
            gate = PolicyGatedSensorModel(make_sensor_model(name, seed=seed, policy_hz=50.0), 10)
            asm = LowStateAssembler(0.002, sensor_model=gate)
            worst = max(worst, max(asm.build(_readout(k), np.ones(22)).imu.age_us for k in range(10)))
    assert worst <= 60_000, worst  # default imu_max_age_s
    assert worst > 20_000  # why 20 ms would false-trip
