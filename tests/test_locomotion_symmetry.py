"""Mirror map for rsl-rl symmetry augmentation (tasks/locomotion/symmetry.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.kinematics.semantic import DEFAULT_CALIBRATION, SemanticMap
from dropbear_wbc.robots.dropbear_names import ARM_MOTORS, MOTOR_NAMES, motor_param
from dropbear_wbc.tasks.locomotion import symmetry as sym

pytestmark = pytest.mark.skipif(not DEFAULT_CALIBRATION.exists(), reason="semantic calibration not generated")


def _pair(n: str) -> str:
    for a, b in (("left", "right"), ("LL_", "RL_"), ("LH_", "RH_")):
        if a in n:
            return n.replace(a, b)
        if b in n:
            return n.replace(b, a)
    raise AssertionError(n)


def test_motor_mirror_is_involution():
    m = sym.motor_mirror_matrix()
    assert np.abs(m @ m - np.eye(sym.NUM_MOTORS)).max() < 1e-9


def test_full_observation_mirror_is_involution():
    o = sym.observation_mirror_matrix()
    assert o.shape == (78, 78)
    assert np.abs(o @ o - np.eye(78)).max() < 1e-9


def test_partners_and_signs_match_semantic_roles():
    expected_sign = {  # from the calibration's semantic role of each motor, not from its name
        "PG_left_leg_pitch": -1, "PG_left_leg_roll": -1, "LL_hip_joint": +1, "LL_knee_actuator_joint": +1,
        "LL_Revolute67": +1, "LL_Revolute81": +1,
        "LH_yaw": -1, "LH_pitch": +1, "LH_roll": +1, "LH_elbow_joint": +1, "LH_wrist_roll": -1,
    }
    pairs = {n: (src, g) for n, src, g in sym.motor_mirror_pairs()}
    for n in MOTOR_NAMES:
        src, g = pairs[n]
        assert src == _pair(n), (n, src)
        left = n if n in expected_sign else _pair(n)
        assert np.sign(g) == expected_sign[left], (n, g)
        assert 0.94 < abs(g) < 1.06, (n, g)
    m = sym.motor_mirror_matrix()
    off = np.abs(m)
    off[np.arange(22), np.abs(m).argmax(axis=1)] = 0
    assert off.max() < 0.01  # ankle pair / hip serial cross terms only


def test_agrees_with_nonlinear_semantic_mirror():
    """Small motor deltas around the stand: M dq equals mirroring through the full semantic map and back."""
    calib = json.loads(DEFAULT_CALIBRATION.read_text())
    sm = SemanticMap(calib)
    q0 = np.asarray(calib["standing_motor_pos"])
    s = sym.semantic_mirror()
    m = sym.motor_mirror_matrix()
    rng = np.random.default_rng(0)
    base = np.asarray(sm.semantic_to_motor(s @ np.asarray(sm.motor_to_semantic(q0, clip=False)), clip=False))
    assert np.abs(base - q0).max() < 2e-3  # default stand is mirror-symmetric to ~1 mrad
    for _ in range(20):
        dq = rng.uniform(-0.02, 0.02, 22)
        qm = np.asarray(sm.semantic_to_motor(s @ np.asarray(sm.motor_to_semantic(q0 + dq, clip=False)), clip=False))
        assert np.abs((qm - base) - m @ dq).max() < 2e-3


def test_root_terms():
    o = sym.observation_mirror_matrix()
    x = np.zeros(78)
    x[:12] = [1, 2, 3, 4, 5, 6, 0.1, 0.2, -0.97, 0.5, 0.3, 0.4]
    y = o @ x
    np.testing.assert_allclose(y[:12], [1, -2, 3, -4, 5, -6, 0.1, -0.2, -0.97, 0.5, -0.3, -0.4])


def test_action_scales_symmetric_so_action_mirror_equals_motor_mirror():
    scale = np.array([0.1 if n in ARM_MOTORS else 0.25 * motor_param(n, 0) / motor_param(n, 1) for n in MOTOR_NAMES])
    np.testing.assert_allclose(sym.action_mirror_matrix(scale), sym.motor_mirror_matrix(), atol=1e-12)


def test_unknown_term_raises():
    with pytest.raises(KeyError):
        sym.observation_mirror_matrix(layout=(("base_lin_vel", 3), ("gait_phase", 2)))


def test_compute_symmetric_states_torch():
    torch = pytest.importorskip("torch")
    obs = torch.randn(5, 78)
    act = torch.randn(5, 22)
    o2, a2 = sym.compute_symmetric_states(obs=obs, actions=act, env=None, obs_type="policy")
    assert o2.shape == (10, 78) and a2.shape == (10, 22)
    torch.testing.assert_close(o2[:5], obs)
    o4, a4 = sym.compute_symmetric_states(obs=o2[5:], actions=a2[5:])
    torch.testing.assert_close(o4[5:], obs, atol=1e-5, rtol=0)
    torch.testing.assert_close(a4[5:], act, atol=1e-5, rtol=0)
    _, none = sym.compute_symmetric_states(obs=obs, actions=None)
    assert none is None


def test_mirror_respects_the_observation_normalizer():
    """rsl-rl stores normalized observations: the augmentation must un-normalize, mirror and re-normalize."""
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")

    from dropbear_wbc.tasks.locomotion.symmetry import (DEFAULT_POLICY_LAYOUT, compute_symmetric_states,
                                                        observation_mirror_matrix, set_observation_normalizers)

    g = torch.Generator().manual_seed(0)
    n = sum(d for _, d in DEFAULT_POLICY_LAYOUT)
    norm = SimpleNamespace(_mean=torch.randn(1, n, generator=g), _std=torch.rand(1, n, generator=g) + 0.5, eps=1e-2)
    m = torch.tensor(observation_mirror_matrix(DEFAULT_POLICY_LAYOUT), dtype=torch.float32)
    raw = torch.randn(5, n, generator=g)
    x = (raw - norm._mean) / (norm._std + norm.eps)
    try:
        set_observation_normalizers(policy=norm)
        aug, _ = compute_symmetric_states(obs=x)
        expected = (raw @ m.T - norm._mean) / (norm._std + norm.eps)
        torch.testing.assert_close(aug[5:], expected, atol=1e-5, rtol=1e-5)
        set_observation_normalizers()  # without a normalizer: the plain mirror of what it is given
        aug, _ = compute_symmetric_states(obs=x)
        torch.testing.assert_close(aug[5:], x @ m.T, atol=1e-5, rtol=1e-5)
    finally:
        set_observation_normalizers()


def _stand_root_positions():
    from pathlib import Path

    npz = Path(sym.__file__).resolve().parents[4] / "data" / "motions" / "smoke" / "dropbear_static_stand.npz"
    if not npz.exists():
        pytest.skip("standing reference not generated")
    z = np.load(npz, allow_pickle=True)
    names = [str(b) for b in z["body_names"]]
    pos = np.asarray(z["body_pos_w"][0], dtype=float)
    w, x, y, zq = np.asarray(z["body_quat_w"][0, 0], dtype=float)
    rot = np.array([[1 - 2 * (y * y + zq * zq), 2 * (x * y - w * zq), 2 * (x * zq + w * y)],
                    [2 * (x * y + w * zq), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - w * x)],
                    [2 * (x * zq - w * y), 2 * (y * zq + w * x), 1 - 2 * (x * x + y * y)]])
    return names, (pos - pos[0]) @ rot


def test_sagittal_plane_reflects_every_left_right_body_pair():
    names, p = _stand_root_positions()
    pairs = [(i, names.index("R" + n[1:])) for i, n in enumerate(names)
             if n[:2] in ("LL", "LH") and "R" + n[1:] in names]
    assert len(pairs) >= 20
    li, ri = (np.array(v) for v in zip(*pairs))
    mirrored = sym.reflect_root_positions(p[li])
    assert np.abs(mirrored[:, 1] - p[ri, 1]).max() < 1e-3  # 0.34 mm max over 24 pairs (midpoints within 0.17 mm)
    assert np.abs(sym.reflect_root_positions(p[li], 0.0)[:, 1] - p[ri, 1]).min() > 0.1  # y = 0 is the wrong plane


def test_sagittal_plane_matches_amp_feature_mirror():
    pytest.importorskip("torch")
    from dropbear_wbc.tasks.locomotion.amp import FeatureMirror

    assert abs(FeatureMirror.midline_y() - sym.SAGITTAL_PLANE_Y) < 5e-4


def test_reflect_root_positions_is_involution_and_torch_ok():
    p = np.array([[0.1, 0.03, -0.9], [0.0, -0.17, 0.2]])
    assert np.allclose(sym.reflect_root_positions(sym.reflect_root_positions(p)), p)
    torch = pytest.importorskip("torch")
    t = torch.tensor(p)
    assert torch.allclose(sym.reflect_root_positions(t), torch.tensor(sym.reflect_root_positions(p)))
    assert torch.equal(t, torch.tensor(p))  # input untouched


def test_positional_term_is_refused_with_plane_hint():
    with pytest.raises(KeyError, match="SAGITTAL_PLANE_Y"):
        sym.observation_mirror_matrix(layout=(("base_lin_vel", 3), ("feet_pos_b", 6)))
