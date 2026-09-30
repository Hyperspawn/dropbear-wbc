"""Arm gravity feed-forward (``dropbear_wbc.teleop.gravity``): exactness of the lumped model, signs, finite-difference
consistency with the potential energy, left/right symmetry and the motor-space mapping.

The comparison against simulated holding torques is a run artefact, not a unit test:
``logs/teleop/probe_elbow_pd0.5_analysis.log`` (Newton CPU bridge, measured vs model at 8 elbow poses).
"""
from __future__ import annotations

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.kinematics.semantic import DEFAULT_CALIBRATION
from dropbear_wbc.teleop.arm_ik import MIRROR_SIGN, SIDES, DropbearArmIK
from dropbear_wbc.teleop.gravity import DEFAULT_INERTIA, ArmGravity, _partials

pytestmark = pytest.mark.skipif(not (DEFAULT_CALIBRATION.exists() and DEFAULT_INERTIA.exists()),
                                reason="needs the calibration and data/teleop/arm_inertia.json")


@pytest.fixture(scope="module")
def grav() -> ArmGravity:
    return ArmGravity(DropbearArmIK())


def _potential(grav: ArmGravity, side: str, q5: np.ndarray) -> float:
    """U of the model in use: bodies beyond the elbow follow the measured wrist path when ``forearm_model='table'``."""
    ch = grav.ik.chains[side]
    _, _, tfs = _partials(ch, q5)
    off = grav._forearm_offset(ch, np.asarray(q5, dtype=float), tfs)
    u = 0.0
    for k, m, c0 in grav.bodies[side]:
        r, t = tfs[k]
        u += m * grav.g * (r @ c0 + t + (off if k >= 4 else 0.0))[2]
    return u


def test_mass_and_segments(grav):
    assert abs(grav.arm_mass["left"] - grav.arm_mass["right"]) < 1e-9
    assert 4.5 < grav.arm_mass["left"] < 5.5
    segs = {k for k, _, _ in grav.bodies["left"]}
    assert segs == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("forearm_model", ["table", "pivot"])
def test_torque_is_gradient_of_potential(grav, forearm_model):
    grav = ArmGravity(grav.ik, forearm_model=forearm_model)
    rng = np.random.default_rng(0)
    for side in SIDES:
        ch = grav.ik.chains[side]
        for q in rng.uniform(ch.lower, ch.upper, size=(20, 5)):
            tau = grav.semantic_torque(side, q)
            num = np.zeros(5)
            for j in range(5):
                dq = np.zeros(5)
                dq[j] = 1e-6
                num[j] = (_potential(grav, side, q + dq) - _potential(grav, side, q - dq)) / 2e-6
            assert np.abs(tau - num).max() < 1e-5, (tau, num)


def test_signs_and_symmetry(grav):
    # forearm pointing forward (elbow 0): gravity extends the elbow, so the holding torque is negative (flexion)
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    assert grav.semantic_torque("left", q)[3] < -0.5
    # arm raised forward (pitch < 0): gravity drives pitch back toward 0, so the holding torque (+dU/dq) is
    # negative (keeps pitching forward); about 5 kg at ~0.2 m -> several N*m
    q = np.array([-1.2, 0.05, 0.0, 1.2, 0.0])
    assert grav.semantic_torque("left", q)[0] < -2.0
    # hanging straight arm: small torques only (the forearm COM hangs ~1.8 cm behind the elbow pivot -> ~0.3 N*m)
    q = np.array([0.0, 0.0, 0.0, grav.ik.chains["left"].info["elbow_rest_rad"], 0.0])
    assert np.abs(grav.semantic_torque("left", q)[[0, 3]]).max() < 0.5
    # mirror symmetry: mirrored configuration -> mirrored torques (roll / yaw / wrist roll flip sign)
    rng = np.random.default_rng(1)
    for q in rng.uniform(grav.ik.chains["left"].lower, grav.ik.chains["left"].upper, size=(20, 5)):
        tl = grav.semantic_torque("left", q)
        tr = grav.semantic_torque("right", q * MIRROR_SIGN)
        assert np.abs(tr - MIRROR_SIGN * tl).max() < 0.05 * max(1.0, np.abs(tl).max())


def test_motor_torque_uses_map_derivative(grav):
    """Motor torque = J^T tau_sem; shoulder / wrist maps are +-1, the elbow four-bar ratio is 3.5..5.5."""
    q = np.concatenate([np.array([-0.4, 0.2, 0.1, 0.5, 0.0]), np.array([-0.4, 0.2, 0.1, 0.5, 0.0]) * MIRROR_SIGN])
    m10, _ = grav.ik.to_motor_fast(q)
    jac = grav.sem_motor_jacobian(m10)
    d = np.abs(np.diag(jac))
    assert np.allclose(d[[0, 1, 2, 4, 5, 6, 7, 9]], 1.0, atol=1e-3)
    assert 3.5 < d[3] < 5.5 and 3.5 < d[8] < 5.5
    tau_m = grav.motor_torque(q, m10)
    assert np.allclose(tau_m, jac.T @ grav.semantic_torque_both(q))
