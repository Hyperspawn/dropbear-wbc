"""Synthetic Dropbear clips: validity, smoothness, planted feet, ranges (MOCK calibration)."""
from __future__ import annotations

import numpy as np
import pytest

import motion_test_paths  # noqa: F401
from motion_test_paths import MOCK_CAL, ROOT
from dropbear_wbc.motion.calibration_view import load_calibration
from dropbear_wbc.motion.g1_to_dropbear import semantic_to_dropbear
from dropbear_wbc.motion.motion_csv import validate_motion_files, write_motion
from dropbear_wbc.motion.names import SEMANTIC_INDEX, SEMANTIC_NAMES
from dropbear_wbc.motion.rotations import quat_to_matrix
from dropbear_wbc.motion.semantic_skeleton import LegGeometry, leg_fk, planar_leg_ik
from dropbear_wbc.motion.synthetic import SYNTHETIC_CLIPS, make_clip, min_jerk, probe_semantic_ranges


@pytest.fixture(scope="module")
def cal():
    return load_calibration(MOCK_CAL, allow_mock=True)


def test_min_jerk_boundary_conditions():
    tau = np.linspace(0, 1, 1001)
    s = min_jerk(tau)
    assert s[0] == 0 and s[-1] == pytest.approx(1.0)
    ds = np.gradient(s, tau)
    assert abs(ds[0]) < 1e-2 and abs(ds[-1]) < 1e-2
    assert np.all(np.diff(s) >= 0)


def test_planar_ik_matches_leg_fk(cal):
    geom = LegGeometry.from_calibration(cal)
    x = np.array([0.05, 0.02, 0.0])
    z = np.array([-0.80, -0.70, -0.85])
    hp, kn, ap = planar_leg_ik(x, z, geom)
    q = np.zeros((3, 22))
    q[:, SEMANTIC_INDEX["left_hip_pitch"]], q[:, SEMANTIC_INDEX["left_knee"]], q[:, SEMANTIC_INDEX["left_ankle_pitch"]] = hp, kn, ap
    lf = leg_fk(np.zeros((3, 3)), np.tile(np.eye(3), (3, 1, 1)), q, geom)
    rel = lf.ankle[:, 0] - lf.hip[:, 0]
    np.testing.assert_allclose(rel[:, 0], x, atol=1e-9)
    np.testing.assert_allclose(rel[:, 2], z, atol=1e-9)
    np.testing.assert_allclose(hp + kn + ap, 0.0, atol=1e-12)  # flat foot


@pytest.mark.parametrize("name", list(SYNTHETIC_CLIPS))
def test_clip_valid_planted_smooth(name, cal, tmp_path):
    clip = make_clip(name, cal)
    sem = clip.semantic
    assert sem.fps == 50.0
    if name == "stand":
        assert sem.num_frames == 501  # 10 s at 50 Hz, inclusive
    geom = LegGeometry.from_calibration(cal)
    lf = leg_fk(sem.pelvis_pos, quat_to_matrix(sem.pelvis_quat_wxyz), sem.q, geom)
    # feet planted: soles on z=0 and foot centres stationary
    np.testing.assert_allclose(lf.sole_height, 0.0, atol=1e-6)
    np.testing.assert_allclose(lf.foot_center - lf.foot_center[:1], 0.0, atol=1e-6)
    # smooth: bounded joint accelerations [rad/s^2]
    acc = np.diff(sem.q, 2, axis=0) * sem.fps**2
    assert np.abs(acc).max() < 60.0, f"max joint accel {np.abs(acc).max():.1f}"
    # no saturation, valid files
    res = semantic_to_dropbear(sem, cal)
    assert res.saturation["frames_with_any_saturation_frac"] == 0.0
    side = {"source": "synthetic", "source_file": name, "source_license": {"license": "gen", "summary": "",
            "redistributable": True, "url": ""}, "retarget_method": "synthetic", "notes": [],
            "contact_hint": {"left": [True] * sem.num_frames, "right": [True] * sem.num_frames}}
    csv_path, _ = write_motion(tmp_path, name, fps=sem.fps, root_pos=res.root_pos, root_quat_wxyz=res.root_quat_wxyz,
                               motor_q=res.motor_q, sidecar=side,
                               semantic=(sem.pelvis_pos, sem.pelvis_quat_wxyz, sem.q, sem.contacts))
    assert validate_motion_files(csv_path) == []


def test_clip_specific_behaviour(cal):
    ranges = probe_semantic_ranges(cal)
    wave = make_clip("wave_right", cal).semantic.q
    legs = [SEMANTIC_INDEX[n] for n in SEMANTIC_NAMES if "hip" in n or "knee" in n or "ankle" in n]
    assert np.ptp(wave[:, legs], axis=0).max() == 0.0  # legs still
    assert np.ptp(wave[:, SEMANTIC_INDEX["left_shoulder_roll"]]) == 0.0  # left arm still
    assert wave[:, SEMANTIC_INDEX["right_shoulder_roll"]].min() < -1.0  # right arm raised sideways (G1 sign)
    shift = make_clip("weight_shift", cal).semantic
    assert np.ptp(shift.pelvis_pos[:, 1]) > 0.05
    squat = make_clip("squat_lite", cal).semantic
    k = squat.q[:, SEMANTIC_INDEX["left_knee"]]
    lo, hi = ranges["left_knee"]
    assert k.max() > k.min() + 0.3 and lo <= k.min() and k.max() <= hi
    assert np.ptp(squat.pelvis_pos[:, 2]) > 0.05


def test_semantic_ranges_probe_on_mock(cal):
    r = probe_semantic_ranges(cal)
    assert r["left_knee"][0] == pytest.approx(0.0, abs=0.01) and r["left_knee"][1] == pytest.approx(np.pi / 2, abs=0.01)
    assert r["left_hip_pitch"][0] == pytest.approx(-np.deg2rad(50), abs=0.01)


def test_wave_right_v2_expressive(cal):
    """wave_right_v2 (demo_eval): right upper arm raised above horizontal, forearm near vertical, +-30 deg yaw wave at
    1.5 Hz, legs and left arm still, returns to the standing pose."""
    clip = make_clip("wave_right_v2", cal).semantic
    q, si = clip.q, SEMANTIC_INDEX
    legs = [si[n] for n in SEMANTIC_NAMES if "hip" in n or "knee" in n or "ankle" in n]
    assert np.ptp(q[:, legs], axis=0).max() == 0.0
    assert np.ptp(q[:, [si[n] for n in SEMANTIC_NAMES if n.startswith("left_")]], axis=0).max() == 0.0
    t = np.arange(len(q)) / clip.fps
    hold = (t > 3.3) & (t < 7.7)
    assert q[hold, si["right_shoulder_pitch"]].max() < -1.5  # raised: upper arm above horizontal (pitch < -86 deg)
    yaw = q[hold, si["right_shoulder_yaw"]]
    assert 0.5 * np.ptp(yaw) >= np.deg2rad(25.0)  # visible side-to-side wave
    # dominant wave frequency ~1.5 Hz
    spec = np.abs(np.fft.rfft(yaw - yaw.mean()))
    freqs = np.fft.rfftfreq(len(yaw), 1.0 / clip.fps)
    assert abs(freqs[spec.argmax()] - 1.5) < 0.25
    np.testing.assert_allclose(q[-1], q[0], atol=1e-9)  # returns to the start pose
