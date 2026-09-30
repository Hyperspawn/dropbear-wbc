"""Pure-numpy G1 FK vs MuJoCo (reference) on the upstream MJCF."""
from __future__ import annotations

import numpy as np
import pytest

import motion_test_paths  # noqa: F401
from dropbear_wbc.motion.g1_model import (
    DEFAULT_G1_MJCF,
    G1_ISAACLAB_JOINT_NAMES,
    G1_JOINT_NAMES,
    load_g1_kinematics,
)

pytestmark = pytest.mark.skipif(not DEFAULT_G1_MJCF.is_file(), reason="G1 MJCF not on disk")


def test_joint_orders():
    assert len(G1_JOINT_NAMES) == 29 and len(set(G1_JOINT_NAMES)) == 29
    assert sorted(G1_ISAACLAB_JOINT_NAMES) == sorted(G1_JOINT_NAMES)
    # Isaac Lab order interleaves left/right: first three are the two hip pitches and waist yaw.
    assert G1_ISAACLAB_JOINT_NAMES[:3] == ("left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint")


def test_fk_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(str(DEFAULT_G1_MJCF))
    d = mujoco.MjData(m)
    kin = load_g1_kinematics()
    rng = np.random.default_rng(1)
    t = 20
    pos = rng.normal(scale=0.3, size=(t, 3))
    quat = rng.normal(size=(t, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    dof = rng.uniform(-1.0, 1.0, size=(t, 29))
    fk = kin.forward(pos, quat, dof)
    for i in range(t):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = pos[i], quat[i], dof[i]
        mujoco.mj_kinematics(m, d)
        for name in ("pelvis", "left_knee_link", "right_ankle_roll_link", "torso_link", "left_wrist_yaw_link"):
            np.testing.assert_allclose(fk.body_pos(name)[i], d.body(name).xpos, atol=1e-9)
            np.testing.assert_allclose(fk.body_rot(name)[i], d.body(name).xmat.reshape(3, 3), atol=1e-9)
        # sole points = sphere centres minus radius in world z
        foot = m.body("left_ankle_roll_link").id
        sph = [g for g in range(m.ngeom) if m.geom_bodyid[g] == foot and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        np.testing.assert_allclose(
            np.sort(fk.sole_points("left")[i][:, 2]), np.sort(d.geom_xpos[sph][:, 2] - 0.005), atol=1e-9
        )


def test_reference_heights():
    kin = load_g1_kinematics()
    assert 0.78 < kin.standing_pelvis_height < 0.80  # MJCF spawns the pelvis at 0.793 with feet on the ground
    assert 0.67 < kin.standing_hip_height < 0.70
    np.testing.assert_allclose(kin.hip_center_in_pelvis, [0.0, 0.0, -0.1027], atol=1e-6)
