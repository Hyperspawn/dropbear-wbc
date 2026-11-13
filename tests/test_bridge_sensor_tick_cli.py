"""newton_bridge rejects a sensor tick that does not divide the policy period at argparse time (item 339)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.newton_sim.imu import sensor_tick_mismatch
from dropbear_wbc.newton_sim.sensor_gate import (
    PolicyGatedSensorModel,
    bridge_sensor_tick_error,
    make_sensor_model,
    policy_steps_per_bridge,
)

_BRIDGE = Path(__file__).resolve().parents[1] / "tools" / "newton_bridge.py"
_spec = importlib.util.spec_from_file_location("newton_bridge_cli339", _BRIDGE)
_bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bridge)
parse_args = _bridge.parse_args


@pytest.mark.parametrize("sim_dt,sub,hz", [(0.002, 1, 50.0), (0.001, 2, 50.0), (0.004, 1, 50.0),
                                            (0.005, 1, 50.0), (0.0025, 1, 100.0), (0.02, 1, 50.0)])
def test_dividing_ticks_accepted(sim_dt, sub, hz):
    assert bridge_sensor_tick_error(sim_dt, sub, hz) is None
    a = parse_args(["--sensor-profile", "motion_mode", "--sim-dt", str(sim_dt), "--substeps", str(sub),
                    "--sensor-policy-hz", str(hz)])
    assert a.sim_dt == sim_dt


@pytest.mark.parametrize("sim_dt,sub,hz", [(0.003, 1, 50.0), (0.0015, 2, 50.0), (0.007, 1, 50.0),
                                            (0.002, 1, 600.0), (0.002, 1, 45.0)])
def test_non_dividing_ticks_rejected_at_argparse(sim_dt, sub, hz, capsys):
    assert bridge_sensor_tick_error(sim_dt, sub, hz) is not None
    with pytest.raises(SystemExit) as e:
        parse_args(["--sensor-profile", "motion_mode", "--sim-dt", str(sim_dt), "--substeps", str(sub),
                    "--sensor-policy-hz", str(hz)])
    assert e.value.code == 2
    assert "does not divide" in capsys.readouterr().err


def test_no_sensor_profile_is_not_checked():
    # perfect sensors: no model, any tick is fine (0081 only checks when a model is attached)
    assert parse_args(["--sim-dt", "0.003"]).sim_dt == 0.003


@pytest.mark.parametrize("bad", [(0.0, 1, 50.0), (-0.002, 1, 50.0), (0.002, 0, 50.0), (0.002, 1, 0.0),
                                 (float("nan"), 1, 50.0), (float("inf"), 1, 50.0),
                                 (0.002, 1, float("nan")), (0.002, 1, float("inf"))])
def test_degenerate_values_rejected(bad):
    assert bridge_sensor_tick_error(*bad) is not None


def test_cli_check_agrees_with_assembler_check():
    """Same verdict as imu.sensor_tick_mismatch on the model the bridge would build, over a grid."""
    n = 0
    for us in range(500, 10001, 250):
        for sub in (1, 2, 4):
            for hz in (25.0, 50.0, 100.0, 200.0):
                sim_dt = us * 1e-6
                tick = sim_dt * sub
                gated = PolicyGatedSensorModel(make_sensor_model("motion_mode", seed=0, policy_hz=hz),
                                               policy_steps_per_bridge(tick, hz))
                assert (bridge_sensor_tick_error(sim_dt, sub, hz) is None) == (sensor_tick_mismatch(tick, gated) is None), \
                    (sim_dt, sub, hz)
                n += 1
    assert n == 39 * 3 * 4
