"""CPU-only tests for --sensor-profile wiring and policy-rate gating (no Isaac/Newton plant)."""
from __future__ import annotations

import math

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.newton_sim.imu import LowStateAssembler
from dropbear_wbc.newton_sim.sensor_gate import (
    PolicyGatedSensorModel,
    make_sensor_model,
    policy_steps_per_bridge,
)
from dropbear_wbc.newton_sim.sensor_model import PROFILES, SensorErrorCfg, SensorErrorModel
from types import SimpleNamespace

# Import after path setup (tools/ is not on sys.path by default).
import importlib.util
from pathlib import Path

_BRIDGE = Path(__file__).resolve().parents[1] / "tools" / "newton_bridge.py"
_spec = importlib.util.spec_from_file_location("newton_bridge", _BRIDGE)
_bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bridge)
parse_args = _bridge.parse_args


def _readout(tick, dt=0.002, ang_v=(0, 0, 0.5), q=None):
    return SimpleNamespace(
        tick=tick, time_s=tick * dt,
        motor_q=np.zeros(22) if q is None else np.asarray(q, float),
        motor_dq=np.zeros(22), motor_tau=np.zeros(22),
        neck_q=np.zeros(6), neck_dq=np.zeros(6),
        root_pos_w=np.array([0, 0, 1.0]), root_quat_wxyz=np.array([1.0, 0, 0, 0]),
        root_lin_vel_w=np.zeros(3), root_ang_vel_w=np.asarray(ang_v, float),
        body_pos_w=np.zeros(0), body_quat_wxyz=np.zeros(0),
    )


def test_parse_args_accepts_profile_seed_policy_hz():
    a = parse_args(["--sensor-profile", "motion_mode", "--sensor-seed", "7", "--sensor-policy-hz", "50"])
    assert a.sensor_profile == "motion_mode" and a.sensor_seed == 7 and a.sensor_policy_hz == 50.0
    b = parse_args(["--sensor-profile", "center_0010"])
    assert b.sensor_profile == "center_0010" and b.sensor_seed is None and b.sensor_policy_hz == 50.0
    c = parse_args([])
    assert c.sensor_profile is None


def test_parse_args_rejects_unknown_profile():
    with pytest.raises(SystemExit):
        parse_args(["--sensor-profile", "not_a_real_profile"])


def test_parse_args_choices_cover_all_profiles():
    for name in sorted(PROFILES):
        assert parse_args(["--sensor-profile", name]).sensor_profile == name


def test_policy_steps_per_bridge():
    assert policy_steps_per_bridge(0.002, 50.0) == 10
    assert policy_steps_per_bridge(0.002, 100.0) == 5
    assert policy_steps_per_bridge(0.02, 50.0) == 1


def test_make_sensor_model_tick_dt():
    m = make_sensor_model("motion_mode", seed=0, policy_hz=50.0)
    assert math.isclose(m.tick_dt, 0.02)
    assert 0.35 <= m.ep.imu_lag <= 1.7
    with pytest.raises(KeyError):
        make_sensor_model("nope")


def test_policy_rate_age_vs_wrong_bridge_rate():
    """Advancing N times at 50 Hz with imu_age in (0.35,1.7) -> ~7-34 ms effective lag;
    calling the same model at 500 Hz incorrectly shrinks ages ~10x."""
    cfg = SensorErrorCfg(
        tilt_rad=0, gyro_bias=0, gyro_bias_walk=0, gyro_noise=0, yaw_drift_rate=0,
        accel_bias=0, accel_noise=0, imu_age_ticks=(1.0, 1.0), joint_age_ticks=(0, 0),
        gyro_lsb=0, accel_lsb=0, joint_q_lsb=0, joint_dq_lsb=0,
    )
    # Correct: policy rate, lag = 1.0 * 0.02 s = 20 ms
    m50 = SensorErrorModel(cfg, tick_dt=0.02, seed=0)
    # Ramp a scalar "signal" through gyro z so lag is measurable
    outs50 = []
    for k in range(20):
        _, g, _ = m50.imu([1, 0, 0, 0], [0, 0, float(k)], [0, 0, 9.81])
        outs50.append(g[2])
    # After buffer fills, output at step k equals input at k - 1 (lag=1.0 tick)
    assert math.isclose(outs50[5], 4.0), outs50[5]  # k=5 -> lag1 -> 4
    lag_s_50 = 1.0 * 0.02  # 20 ms

    # Wrong: call at 500 Hz with same lag-in-ticks -> 1.0 * 0.002 = 2 ms
    m500 = SensorErrorModel(cfg, tick_dt=0.002, seed=0)
    outs500 = []
    for k in range(20):
        _, g, _ = m500.imu([1, 0, 0, 0], [0, 0, float(k)], [0, 0, 9.81])
        outs500.append(g[2])
    assert math.isclose(outs500[5], 4.0), outs500[5]
    lag_s_500 = 1.0 * 0.002  # 2 ms
    assert math.isclose(lag_s_50 / lag_s_500, 10.0)
    assert 7e-3 <= lag_s_50 <= 34e-3
    assert lag_s_500 < 4e-3  # the 10x-shrunk pitfall


def test_hold_between_policy_steps_does_not_shrink_lag():
    """Bridge-rate publish without advancing the model must not shrink lag."""
    cfg = SensorErrorCfg(
        tilt_rad=0, gyro_bias=0, gyro_bias_walk=0, gyro_noise=0, yaw_drift_rate=0,
        accel_bias=0, accel_noise=0, imu_age_ticks=(1.0, 1.0), joint_age_ticks=(0, 0),
        gyro_lsb=0, accel_lsb=0, joint_q_lsb=0, joint_dq_lsb=0,
    )
    raw = SensorErrorModel(cfg, tick_dt=0.02, seed=0)
    gate = PolicyGatedSensorModel(raw, steps_per_policy=10)  # 500 Hz bridge / 50 Hz policy
    # Drive a step in gyro z on the first (advance) call, then hold for 9 bridge ticks
    # with a different truth value; held output must stay at the advanced measurement.
    _, g0, _ = gate.imu([1, 0, 0, 0], [0, 0, 1.0], [0, 0, 9.81])
    held = []
    for _ in range(9):
        _, g, _ = gate.imu([1, 0, 0, 0], [0, 0, 99.0], [0, 0, 9.81])
        held.append(g[2])
    assert all(math.isclose(h, g0[2]) for h in held), (g0[2], held)
    # Next policy boundary advances again (with lag=1, sees previous push)
    _, g10, _ = gate.imu([1, 0, 0, 0], [0, 0, 2.0], [0, 0, 9.81])
    # model has been advanced twice: first push 1.0, second push 2.0; lag=1 -> output is 1.0
    assert math.isclose(g10[2], 1.0), g10[2]
    # And only 2 model.imu calls happened (t advanced 2 * 0.02)
    assert math.isclose(raw.t, 0.04), raw.t


def test_assembler_with_gate_privileged_stays_truth():
    raw = make_sensor_model("motion_mode", seed=1, policy_hz=50.0)
    # zero ages/noise for a clean hook check: replace episode lags
    raw.ep.imu_lag = 0.0
    raw.ep.joint_lag = 0.0
    gate = PolicyGatedSensorModel(raw, steps_per_policy=10)
    asm = LowStateAssembler(0.002, sensor_model=gate)
    plain = LowStateAssembler(0.002).build(_readout(0, ang_v=(0, 0, 0.5)), np.ones(22))
    ls = asm.build(_readout(0, ang_v=(0, 0, 0.5), q=np.full(22, 0.0101)), np.ones(22))
    assert not np.allclose(ls.imu.gyro, plain.imu.gyro)  # bias/misalignment applied
    assert ls.sim is not None and np.allclose(ls.sim.root_ang_vel_w, [0, 0, 0.5])


def test_gate_survives_assembler_recreate():
    """Recreating LowStateAssembler with the same gate must not reset the episode."""
    raw = make_sensor_model("ideal", seed=0, policy_hz=50.0)
    gate = PolicyGatedSensorModel(raw, steps_per_policy=1)
    lag0 = (raw.ep.imu_lag, raw.ep.joint_lag)
    asm1 = LowStateAssembler(0.02, sensor_model=gate)
    asm1.build(_readout(0, dt=0.02), np.ones(22))
    t_after = raw.t
    asm2 = LowStateAssembler(0.02, sensor_model=gate)
    asm2.build(_readout(1, dt=0.02), np.ones(22))
    assert (raw.ep.imu_lag, raw.ep.joint_lag) == lag0
    assert math.isclose(raw.t, t_after + 0.02)
