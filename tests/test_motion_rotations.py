"""Rotation helpers vs scipy (CPU)."""
from __future__ import annotations

import numpy as np
import pytest

import motion_test_paths  # noqa: F401
from dropbear_wbc.motion import rotations as R

scipy_rot = pytest.importorskip("scipy.spatial.transform").Rotation
rng = np.random.default_rng(0)


def test_quat_matrix_roundtrip_and_scipy():
    q = R.quat_canonical(R.quat_normalize(rng.normal(size=(64, 4))))
    m = R.quat_to_matrix(q)
    np.testing.assert_allclose(m, scipy_rot.from_quat(R.wxyz_to_xyzw(q)).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(R.matrix_to_quat(m), q, atol=1e-10)


def test_quat_mul_matches_matrix_product():
    a = R.quat_normalize(rng.normal(size=(32, 4)))
    b = R.quat_normalize(rng.normal(size=(32, 4)))
    np.testing.assert_allclose(R.quat_to_matrix(R.quat_mul(a, b)), R.quat_to_matrix(a) @ R.quat_to_matrix(b), atol=1e-12)


def test_euler_extrinsic_xyz_matches_scipy_lowercase():
    ang = rng.uniform(-np.pi, np.pi, size=(50, 3))
    ours = R.quat_to_matrix(R.euler_xyz_extrinsic_to_quat(ang))
    ref = scipy_rot.from_euler("xyz", ang).as_matrix()
    np.testing.assert_allclose(ours, ref, atol=1e-12)


@pytest.mark.parametrize("order", ["YXZ", "XYZ", "ZYX", "ZXY"])
def test_intrinsic_decomposition_roundtrip(order):
    ang = rng.uniform(-1.4, 1.4, size=(100, 3))
    m = R.euler_intrinsic_to_matrix(order, ang)
    np.testing.assert_allclose(m, scipy_rot.from_euler(order, ang).as_matrix(), atol=1e-12)
    back = R.matrix_to_euler_intrinsic(order, m)
    np.testing.assert_allclose(R.euler_intrinsic_to_matrix(order, back), m, atol=1e-10)


def test_slerp_endpoints_and_midpoint_angle():
    a = R.quat_normalize(rng.normal(size=(10, 4)))
    b = R.quat_normalize(rng.normal(size=(10, 4)))
    np.testing.assert_allclose(R.quat_to_matrix(R.quat_slerp(a, b, np.zeros(10))), R.quat_to_matrix(a), atol=1e-9)
    np.testing.assert_allclose(R.quat_to_matrix(R.quat_slerp(a, b, np.ones(10))), R.quat_to_matrix(b), atol=1e-9)
    ma, mb = R.quat_to_matrix(a), R.quat_to_matrix(b)
    mid = R.quat_to_matrix(R.quat_slerp(a, b, np.full(10, 0.5)))
    full = R.rotation_angle(np.swapaxes(ma, -1, -2) @ mb)
    half = R.rotation_angle(np.swapaxes(ma, -1, -2) @ mid)
    np.testing.assert_allclose(half, full / 2, atol=1e-9)
