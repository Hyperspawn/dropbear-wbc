"""CPU-only: LowStateAssembler refuses a sensor model whose tick does not match the build tick (patch 0081)."""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.newton_sim.imu import LowStateAssembler, sensor_tick_mismatch
from dropbear_wbc.newton_sim.sensor_gate import PolicyGatedSensorModel, make_sensor_model, policy_steps_per_bridge
from dropbear_wbc.newton_sim.sensor_model import SensorErrorCfg, SensorErrorModel


def _readout(tick, q=0.0, dt=0.002):
    return SimpleNamespace(tick=tick, time_s=tick * dt, motor_q=np.full(22, q), motor_dq=np.zeros(22),
                           motor_tau=np.zeros(22), neck_q=np.zeros(6), neck_dq=np.zeros(6),
                           root_pos_w=np.array([0, 0, 1.0]), root_quat_wxyz=np.array([1.0, 0, 0, 0]),
                           root_lin_vel_w=np.zeros(3), root_ang_vel_w=np.zeros(3),
                           body_pos_w=np.zeros(0), body_quat_wxyz=np.zeros(0))


def _clean(lag_ticks: float, tick_dt: float = 0.02) -> SensorErrorModel:
    c = SensorErrorCfg(tilt_rad=0, gyro_bias=0, accel_bias=0, gyro_noise=0, accel_noise=0, gyro_bias_walk=0,
                       yaw_drift_rate=0, joint_q_lsb=0, joint_dq_lsb=0, imu_age_ticks=(lag_ticks, lag_ticks),
                       joint_age_ticks=(lag_ticks, lag_ticks), joint_update_prob=1.0)
    return SensorErrorModel(c, tick_dt=tick_dt, seed=0)


def _measured_joint_delay_s(asm, dt, n=200, t_step=0.1):
    """Time from a joint step in truth to the first build whose measured q crosses half the step."""
    k_step = int(round(t_step / dt))
    for k in range(n):
        ls = asm.build(_readout(k, q=1.0 if k >= k_step else 0.0, dt=dt), np.ones(22))
        if ls.motor.q[0] >= 0.5:
            return (k - k_step) * dt, ls.imu.age_us * 1e-6
    raise AssertionError("step never arrived")


def test_raw_model_at_bridge_rate_is_refused():
    with pytest.raises(ValueError, match="PolicyGatedSensorModel"):
        LowStateAssembler(0.002, sensor_model=make_sensor_model("motion_mode", seed=0, policy_hz=50.0))


def test_matching_raw_and_gated_models_are_accepted():
    LowStateAssembler(0.02, sensor_model=make_sensor_model("motion_mode", seed=0, policy_hz=50.0))
    for dt in (0.002, 0.004, 0.005, 0.01):
        gate = PolicyGatedSensorModel(make_sensor_model("motion_mode", seed=0), policy_steps_per_bridge(dt, 50.0))
        LowStateAssembler(dt, sensor_model=gate)
    LowStateAssembler(0.002, sensor_model=SimpleNamespace(imu=lambda *a: a, joints=lambda *a: a))  # unchecked


def test_gated_model_with_non_dividing_bridge_tick_is_refused():
    dt = 0.003  # round(20/3) = 7 -> 21 ms per advance, held-age step 2.857 ms
    gate = PolicyGatedSensorModel(make_sensor_model("motion_mode", seed=0), policy_steps_per_bridge(dt, 50.0))
    assert gate.steps_per_policy == 7
    msg = sensor_tick_mismatch(dt, gate)
    assert msg is not None and "7 builds" in msg
    with pytest.raises(ValueError):
        LowStateAssembler(dt, sensor_model=gate)


def test_tolerance_is_one_percent():
    assert sensor_tick_mismatch(0.02, _clean(1.0, tick_dt=0.0201)) is None
    assert sensor_tick_mismatch(0.02, _clean(1.0, tick_dt=0.0203)) is not None


def test_override_warns_and_shows_the_age_shrink():
    raw = _clean(2.0)  # 2 policy ticks = 40 ms designed
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        asm = LowStateAssembler(0.002, sensor_model=raw, allow_sensor_tick_mismatch=True)
    assert any(issubclass(x.category, RuntimeWarning) and "10.00x" in str(x.message) for x in w)
    delay, reported = _measured_joint_delay_s(asm, 0.002)
    assert reported == pytest.approx(0.040)
    assert delay == pytest.approx(0.004, abs=0.0021)  # 2 bridge ticks, not 40 ms: 10x shrink


def test_gated_model_delivers_designed_delay():
    gate = PolicyGatedSensorModel(_clean(2.0), 10)
    delay, _ = _measured_joint_delay_s(LowStateAssembler(0.002, sensor_model=gate), 0.002)
    assert 0.030 <= delay <= 0.060  # 2 policy ticks of 20 ms + up to one hold period
