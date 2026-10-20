"""CPU-only tests for the measured-sensor model (newton_sim.sensor_model) and its LowStateAssembler hook."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

import sdk_test_paths  # noqa: F401
from dropbear_wbc.newton_sim.imu import LowStateAssembler
from dropbear_wbc.newton_sim.sensor_model import PROFILES, FracDelay, SensorErrorCfg, SensorErrorModel, quantize


def _readout(tick, quat=(1, 0, 0, 0), ang_v=(0, 0, 0), q=None):
    return SimpleNamespace(tick=tick, time_s=tick * 0.02, motor_q=np.zeros(22) if q is None else np.asarray(q, float),
                           motor_dq=np.zeros(22), motor_tau=np.zeros(22), neck_q=np.zeros(6), neck_dq=np.zeros(6),
                           root_pos_w=np.array([0, 0, 1.0]), root_quat_wxyz=np.asarray(quat, float),
                           root_lin_vel_w=np.zeros(3), root_ang_vel_w=np.asarray(ang_v, float),
                           body_pos_w=np.zeros(0), body_quat_wxyz=np.zeros(0))


def test_ideal_profile_is_identity():
    m = SensorErrorModel(PROFILES["ideal"], seed=0)
    q0 = np.array([math.cos(0.2), math.sin(0.2), 0, 0])
    q, g, a = m.imu(q0, [0.1, -0.2, 0.3], [0, 0, 9.81])
    assert np.allclose(q, q0) and np.allclose(g, [0.1, -0.2, 0.3]) and np.allclose(a, [0, 0, 9.81])
    jq, jdq = m.joints(np.full(22, 0.123), np.full(22, -1.0))
    assert np.allclose(jq, 0.123) and np.allclose(jdq, -1.0)


def test_assembler_default_unchanged_and_hook_applies():
    plain = LowStateAssembler(0.02).build(_readout(0, ang_v=(0, 0, 0.5)), np.ones(22))
    assert np.allclose(plain.imu.gyro, [0, 0, 0.5])
    cfg = SensorErrorCfg(gyro_noise=0, gyro_bias_walk=0, imu_age_ticks=(0, 0), joint_age_ticks=(0, 0))
    asm = LowStateAssembler(0.02, sensor_model=SensorErrorModel(cfg, seed=1))
    ls = asm.build(_readout(0, ang_v=(0, 0, 0.5), q=np.full(22, 0.0101)), np.ones(22))
    assert not np.allclose(ls.imu.gyro, [0, 0, 0.5])  # bias + misalignment
    assert np.all(np.abs(ls.imu.gyro - [0, 0, 0.5]) < 0.02 + 0.5 * math.sin(math.radians(3.0)) * 2 + 1e-3)
    assert np.allclose(ls.motor.q, np.round(0.0101 / 0.38e-3) * 0.38e-3, atol=1e-6)
    assert ls.sim is not None and np.allclose(ls.sim.root_ang_vel_w, [0, 0, 0.5])  # privileged block stays truth


def test_a1_reply_quantum_is_one_degree():
    m = SensorErrorModel(PROFILES["a1_reply"], seed=0)
    m.ep.joint_lag = 0.0
    jq, _ = m.joints(np.full(22, math.radians(0.6)), np.zeros(22))
    assert np.allclose(jq, math.radians(1.0))


def test_frac_delay_interpolates():
    d = FracDelay(2.0)
    outs = [d.push(np.array([float(k)]), 1.5)[0] for k in range(6)]
    assert outs[0] == 0.0 and math.isclose(outs[5], 5 - 1.5)


def test_yaw_drift_accumulates_and_tilt_bounded():
    cfg = replace_cfg(tilt_rad=math.radians(3.0), yaw_drift_rate=0.02, gyro_noise=0, gyro_bias_walk=0,
                      imu_age_ticks=(0, 0))
    m = SensorErrorModel(cfg, tick_dt=0.02, seed=3)
    for _ in range(500):  # 10 s
        q, _, _ = m.imu([1, 0, 0, 0], [0, 0, 0], [0, 0, 9.81])
    ang = 2 * math.acos(min(1.0, abs(q[0])))
    assert ang <= abs(m.ep.yaw_rate) * 10 + 2 * math.radians(3.0) + 1e-6
    assert ang >= abs(m.ep.yaw_rate) * 10 - 2 * math.radians(3.0) - 1e-6


def test_stale_hold_matches_poll_rate():
    m = SensorErrorModel(PROFILES["deployed_054"], seed=0)
    m.ep.joint_lag = 0.0
    changes = 0
    prev = m.joints(np.zeros(22), np.zeros(22))[0]
    for k in range(1, 1001):
        jq, _ = m.joints(np.full(22, math.radians(2.0) * k), np.zeros(22))
        changes += int(np.sum(jq != prev))
        prev = jq
    rate_hz = changes / 22 / (1000 * 0.02)
    assert 1.2 < rate_hz < 2.2, rate_hz


def test_gyro_saturates_at_default_full_scale():
    m = SensorErrorModel(PROFILES["motion_mode"], seed=0)
    m.ep.imu_lag = 0.0
    _, g, _ = m.imu([1, 0, 0, 0], [0, 0, 10.0], [0, 0, 9.81])
    assert g[2] <= math.radians(250.0) + 1e-9


def test_quantize():
    assert quantize(np.array([0.26]), 0.1)[0] == 0.30000000000000004 or math.isclose(quantize(np.array([0.26]), 0.1)[0], 0.3)


def replace_cfg(**kw):
    from dataclasses import replace
    return replace(SensorErrorCfg(), **kw)
