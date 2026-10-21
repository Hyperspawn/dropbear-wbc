"""CEM-60 torque-speed line vs its 330 W rating (notes/213)."""
import math

import pytest

import sdk_test_paths  # noqa: F401

from dropbear_wbc.robots.hw_motor_specs import (
    HW_PROFILE_MAPS, HW_PROFILE_OPTIONS, JOINT_MOTOR_MAPS, MOTOR_MODELS, RATED_POWER_W, envelope_torque,
    joint_hw_params, motor_mass_offsets, rated_point_shaft_power, rated_point_speed,
)

RPM = 2.0 * math.pi / 60.0


def test_v2_rated_point_breaks_330w():
    """Pins the known defect of the published V2 line so a silent 'fix' of it (which would change hw_cem6) is caught."""
    m = MOTOR_MODELS["EPS-CEM-60-V2"]
    assert rated_point_speed(m) / RPM == pytest.approx(60.0)
    assert rated_point_shaft_power(m) == pytest.approx(377.0, abs=0.5)
    assert rated_point_shaft_power(m) > RATED_POWER_W[m.name]


def test_p330_line():
    m = MOTOR_MODELS["EPS-CEM-60-V2-P330"]
    assert rated_point_speed(m) / RPM == pytest.approx(40.0)
    assert rated_point_shaft_power(m) == pytest.approx(251.3, abs=0.1)
    assert rated_point_shaft_power(m) <= RATED_POWER_W[m.name]
    assert envelope_torque(m, 30.0 * RPM) == pytest.approx(75.0)
    assert envelope_torque(m, 55.0 * RPM) == pytest.approx(37.5)
    assert envelope_torque(MOTOR_MODELS["EPS-CEM-60-V2"], 55.0 * RPM) == pytest.approx(75.0)
    # identical to V2 except the line
    v2 = MOTOR_MODELS["EPS-CEM-60-V2"]
    for f in ("gear_ratio", "peak_torque", "rated_torque", "no_load_speed", "rotor_inertia", "coulomb_friction",
              "viscous_friction", "mass_kg"):
        assert getattr(m, f) == getattr(v2, f)


@pytest.mark.parametrize("name", sorted(RATED_POWER_W))
def test_saturation_ge_peak(name):
    m = MOTOR_MODELS[name]
    assert m.saturation_effort >= m.peak_torque


@pytest.mark.parametrize("p330,base", [("hw_cem6_p330", "hw_cem6"), ("hw_v2ie_p330", "hw_v2ie")])
def test_p330_profiles_differ_only_in_cem_line(p330, base):
    a, b = JOINT_MOTOR_MAPS[HW_PROFILE_MAPS[p330]], JOINT_MOTOR_MAPS[HW_PROFILE_MAPS[base]]
    assert a.keys() == b.keys()
    changed = {j for j in a if a[j] != b[j]}
    assert changed and all(b[j] == "EPS-CEM-60-V2" and a[j] == "EPS-CEM-60-V2-P330" for j in changed)
    assert HW_PROFILE_OPTIONS[p330] == HW_PROFILE_OPTIONS[base]
    pa, pb = joint_hw_params(HW_PROFILE_MAPS[p330]), joint_hw_params(HW_PROFILE_MAPS[base])
    for j in pa:
        for k in ("kp", "kd", "armature", "peak_torque", "rated_torque", "no_load_speed"):
            assert pa[j][k] == pb[j][k], (j, k)
    assert motor_mass_offsets(p330) == motor_mass_offsets(base)
