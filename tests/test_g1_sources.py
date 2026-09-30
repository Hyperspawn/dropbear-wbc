"""G1 motion loaders on the real upstream files (+ synthetic files for formats not on disk)."""
from __future__ import annotations

import json

import numpy as np
import pytest

import motion_test_paths  # noqa: F401
from dropbear_wbc.motion.g1_model import G1_ISAACLAB_JOINT_NAMES, G1_JOINT_INDEX, load_g1_kinematics
from dropbear_wbc.motion.g1_sources import (
    SOURCES,
    detect_source,
    discover_source_files,
    is_git_lfs_pointer,
    load_g1_motion,
)
from dropbear_wbc.motion.rotations import quat_to_matrix, wxyz_to_xyzw

EXPECTED_FPS = {"unitree_rl_lab_mimic": 60.0, "soma_retargeter_g1": 120.0, "kimodo_g1": 30.0, "asap_g1": 30.0}
EXPECTED_COUNT = {"unitree_rl_lab_mimic": 2, "soma_retargeter_g1": 10, "kimodo_g1": 5, "asap_g1": 52}


def _files(key):
    files = discover_source_files(key)
    if not files:
        pytest.skip(f"{key}: no files on disk")
    return files


def _up_z(quat_wxyz):
    return quat_to_matrix(quat_wxyz)[:, 2, 2]


@pytest.mark.parametrize("key", sorted(EXPECTED_FPS))
def test_real_files_shapes_fps_units(key):
    files = _files(key)
    assert len(files) == EXPECTED_COUNT[key]
    kin = load_g1_kinematics()
    for f in files:
        m = load_g1_motion(f, key)
        assert m.fps == EXPECTED_FPS[key]
        assert m.dof.shape == (m.num_frames, 29) and m.root_pos.shape == (m.num_frames, 3)
        np.testing.assert_allclose(np.linalg.norm(m.root_quat_wxyz, axis=1), 1.0, atol=1e-6)
        # ASAP single_foot_balance_level4 leans the pelvis ~55 deg (arabesque-like), squat_level3 ~34 deg
        up_min = 0.5 if key == "asap_g1" else 0.9
        assert np.median(_up_z(m.root_quat_wxyz)) > up_min, "pelvis should be upright most of the clip"
        assert np.all(np.abs(m.dof) < 3.2), "joint angles must be radians"
        fk = kin.forward(m.root_pos, m.root_quat_wxyz, m.dof)
        lowest = np.minimum(fk.sole_height("left"), fk.sole_height("right"))
        if key == "asap_g1":
            # ASAP clips are ground-aligned at their lowest frame (fit_smpl_motion.py height fix) but video-derived
            # and float in many frames (logs/motion_pipeline/probe_asap_ground.log).
            assert -0.06 < lowest.min() < 0.06
            assert 0.70 < m.root_pos[0, 2] < 1.0
        else:
            tol = 0.04 if key == "kimodo_g1" else 0.03  # generated clips penetrate a little (dance p5 -3.4 cm)
            assert abs(np.percentile(lowest, 5)) < tol, f"{f.name}: soles not on the ground"
            assert 0.74 < m.root_pos[0, 2] < 0.81, f"{f.name}: standing pelvis height {m.root_pos[0, 2]}"


def _stance_foot_speed(m):
    """Median horizontal speed [m/s] of feet whose soles are within 3 cm of the ground."""
    fk = load_g1_kinematics().forward(m.root_pos, m.root_quat_wxyz, m.dof)
    out = []
    for side in ("left", "right"):
        c = fk.foot_center(side)
        v = np.linalg.norm(np.gradient(c[:, :2], 1.0 / m.fps, axis=0), axis=1)
        low = fk.sole_height(side) < 0.03
        out.append(v[low])
    return float(np.median(np.concatenate(out)))


@pytest.mark.parametrize("idx", [0, 1])
def test_beyondmimic_csv_is_xyzw_not_wxyz(idx):
    """Upright-ness cannot tell xyzw from wxyz here (a ~180 deg yaw flip is also upright), but planted feet
    can: with the correct convention stance feet are (nearly) static."""
    f = _files("unitree_rl_lab_mimic")[idx]
    good = load_g1_motion(f, "unitree_rl_lab_mimic")
    wrong = load_g1_motion(f, "kimodo_g1")  # same 36 columns read as wxyz
    assert np.median(_up_z(good.root_quat_wxyz)) > 0.99
    v_good, v_wrong = _stance_foot_speed(good), _stance_foot_speed(wrong)
    assert v_good < 0.15 and v_wrong > 2.5 * v_good, (v_good, v_wrong)


def test_kimodo_csv_is_wxyz():
    f = _files("kimodo_g1")[0]
    good = load_g1_motion(f, "kimodo_g1")
    assert np.median(_up_z(good.root_quat_wxyz)) > 0.99


def test_soma_euler_is_extrinsic_xyz_and_units():
    scipy_rot = pytest.importorskip("scipy.spatial.transform").Rotation
    f = _files("soma_retargeter_g1")[0]
    m = load_g1_motion(f, "soma_retargeter_g1")
    raw = np.loadtxt(f, delimiter=",", skiprows=1)
    np.testing.assert_allclose(m.root_pos, raw[:, 1:4] * 0.01)
    ref = scipy_rot.from_euler("xyz", raw[:, 4:7], degrees=True).as_matrix()
    np.testing.assert_allclose(quat_to_matrix(m.root_quat_wxyz), ref, atol=1e-9)
    with open(f, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    j = header.index("left_knee_joint")
    np.testing.assert_allclose(m.dof[:, G1_JOINT_INDEX["left_knee_joint"]], np.deg2rad(raw[:, j]))


def test_asap_23dof_mapping():
    f = _files("asap_g1")[0]
    m = load_g1_motion(f, "asap_g1")
    wrists = [G1_JOINT_INDEX[f"{s}_wrist_{a}_joint"] for s in ("left", "right") for a in ("roll", "pitch", "yaw")]
    assert not m.dof_present[wrists].any() and m.dof_present.sum() == 23
    assert np.all(m.dof[:, wrists] == 0.0)
    import joblib

    raw = next(iter(joblib.load(f).values()))
    np.testing.assert_allclose(m.dof[:, G1_JOINT_INDEX["right_elbow_joint"]], raw["dof"][:, 22])  # last ASAP dof
    np.testing.assert_allclose(m.dof[:, G1_JOINT_INDEX["waist_pitch_joint"]], raw["dof"][:, 14])
    np.testing.assert_allclose(wxyz_to_xyzw(m.root_quat_wxyz) * np.sign(raw["root_rot"][:, 3:4]),
                               raw["root_rot"] * np.sign(raw["root_rot"][:, 3:4]), atol=1e-9)


def test_lafan1_loader_on_synthetic_file(tmp_path):
    """LAFAN1-G1 files are not on disk; check the loader on a file written in that layout."""
    t = 40
    rng = np.random.default_rng(3)
    pos = np.c_[np.linspace(0, 1, t), np.zeros(t), np.full(t, 0.78)]
    yaw = np.linspace(0, 1.0, t)
    quat_xyzw = np.c_[np.zeros(t), np.zeros(t), np.sin(yaw / 2), np.cos(yaw / 2)]
    dof = rng.uniform(-0.5, 0.5, size=(t, 29))
    d = tmp_path / "LAFAN1_Retargeting_Dataset" / "g1"
    d.mkdir(parents=True)
    f = d / "walk1_subject1.csv"
    np.savetxt(f, np.c_[pos, quat_xyzw, dof], delimiter=",")
    assert detect_source(f) == "lafan1_g1"
    m = load_g1_motion(f)
    assert m.fps == 30.0 and m.license.spdx_or_name == "CC-BY-NC-ND-4.0" and not m.license.redistributable
    np.testing.assert_allclose(m.dof, dof)
    np.testing.assert_allclose(m.root_quat_wxyz[:, 0], np.cos(yaw / 2), atol=1e-12)


def test_sonic_reference_loader_on_synthetic_folder(tmp_path):
    t = 10
    rng = np.random.default_rng(4)
    dof_mj = rng.uniform(-1, 1, size=(t, 29))
    isaac = np.stack([dof_mj[:, G1_JOINT_INDEX[n]] for n in G1_ISAACLAB_JOINT_NAMES], axis=1)
    folder = tmp_path / "gear_sonic_deploy" / "clip"
    folder.mkdir(parents=True)
    np.savetxt(folder / "joint_pos.csv", isaac, delimiter=",", header=",".join(f"joint_{i}" for i in range(29)), comments="")
    bp = np.zeros((t, 42))
    bp[:, 2] = 0.79
    bq = np.zeros((t, 56))
    bq[:, 0] = 1.0
    np.savetxt(folder / "body_pos.csv", bp, delimiter=",", header=",".join(f"c{i}" for i in range(42)), comments="")
    np.savetxt(folder / "body_quat.csv", bq, delimiter=",", header=",".join(f"c{i}" for i in range(56)), comments="")
    m = load_g1_motion(folder, "sonic_reference_g1")
    np.testing.assert_allclose(m.dof, dof_mj)
    assert m.root_pos[0, 2] == pytest.approx(0.79)


def test_sonic_reference_files_on_disk_are_lfs_pointers():
    files = discover_source_files("sonic_reference_g1")
    if not files:
        pytest.skip("GR00T reference not cloned")
    assert all(is_git_lfs_pointer(f / "joint_pos.csv") for f in files)
    with pytest.raises(FileNotFoundError, match="git-lfs"):
        load_g1_motion(files[0], "sonic_reference_g1")


def test_resample_preserves_endpoints():
    f = _files("kimodo_g1")[0]
    m = load_g1_motion(f, "kimodo_g1")
    r = m.resample(50.0)
    assert r.fps == 50.0
    np.testing.assert_allclose(r.dof[0], m.dof[0])
    assert abs(r.duration - m.duration) < 1.0 / 50.0 + 1e-9


def test_source_registry_licenses_complete():
    for key, spec in SOURCES.items():
        d = spec.license.as_dict()
        assert d["license"] and d["summary"], key
    assert SOURCES["soma_retargeter_g1"].license.redistributable is False
    assert SOURCES["lafan1_g1"].license.spdx_or_name == "CC-BY-NC-ND-4.0"
    json.dumps({k: v.license.as_dict() for k, v in SOURCES.items()})
