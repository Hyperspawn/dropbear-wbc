"""G1 -> Dropbear mapping: sign sanity, left/right, waist fold, root, contacts (MOCK calibration)."""
from __future__ import annotations

import numpy as np
import pytest

import motion_test_paths  # noqa: F401
from motion_test_paths import MOCK_CAL
from dropbear_wbc.motion.calibration_view import load_calibration
from dropbear_wbc.motion.g1_model import DEFAULT_G1_MJCF, G1_JOINT_INDEX, load_g1_kinematics
from dropbear_wbc.motion.g1_sources import SOURCES, G1Motion, discover_source_files, load_g1_motion
from dropbear_wbc.motion.g1_to_dropbear import (
    G1_HIP_ROLL_LINK_PITCH,
    RetargetOptions,
    g1_to_semantic,
    retarget_g1_motion,
)
from dropbear_wbc.motion.names import SEMANTIC_INDEX
from dropbear_wbc.motion.rotations import quat_to_matrix, rot_x, rot_y, rot_z
from dropbear_wbc.motion.semantic_skeleton import LegGeometry, leg_fk

if not DEFAULT_G1_MJCF.is_file():  # Unitree G1 model: git clone https://github.com/unitreerobotics/unitree_mujoco ../upstream/unitree_mujoco
    pytest.skip(f"G1 MJCF not found at {DEFAULT_G1_MJCF} (set G1_MJCF or clone unitree_mujoco)", allow_module_level=True)

J = G1_JOINT_INDEX
S = SEMANTIC_INDEX


@pytest.fixture(scope="module")
def cal():
    return load_calibration(MOCK_CAL, allow_mock=True)


def standing_motion(t: int = 30, fps: float = 30.0, **dof_values: float) -> G1Motion:
    """G1 zero pose with the pelvis at its standing height, optional constant joint overrides."""
    kin = load_g1_kinematics()
    dof = np.zeros((t, 29))
    for k, v in dof_values.items():
        dof[:, J[k]] = v
    return G1Motion(
        fps=fps,
        root_pos=np.tile([0.3, -0.2, kin.standing_pelvis_height], (t, 1)),
        root_quat_wxyz=np.tile([1.0, 0, 0, 0], (t, 1)),
        dof=dof,
        source="synthetic_test",
        license=SOURCES["kimodo_g1"].license,
        source_file="<test>",
        clip="test",
        fmt="test",
    )


BY_NAME = RetargetOptions(waist_mode="drop", joint_mapping="by_name")


def test_knee_flexion_maps_positive_and_sides_not_swapped(cal):
    sem = g1_to_semantic(standing_motion(left_knee_joint=0.8), cal, RetargetOptions(waist_mode="drop"))
    assert sem.q_requested[0, S["left_knee"]] == pytest.approx(0.8)
    assert sem.q_requested[0, S["right_knee"]] == pytest.approx(0.0)
    # G1 FK confirms positive knee = flexion (foot behind the knee).
    kin = load_g1_kinematics()
    m = standing_motion(left_knee_joint=0.8)
    fk = kin.forward(m.root_pos, m.root_quat_wxyz, m.dof)
    assert fk.body_pos("left_ankle_roll_link")[0, 0] < fk.body_pos("left_knee_link")[0, 0] - 0.1


def test_shoulder_roll_positive_is_left_abduction(cal):
    kin = load_g1_kinematics()
    base = standing_motion()
    m = standing_motion(left_shoulder_roll_joint=0.5)
    y0 = kin.forward(base.root_pos, base.root_quat_wxyz, base.dof).body_pos("left_elbow_link")[0, 1]
    y1 = kin.forward(m.root_pos, m.root_quat_wxyz, m.dof).body_pos("left_elbow_link")[0, 1]
    assert y1 > y0 + 0.03  # elbow moves to +y (left): abduction
    sem = g1_to_semantic(m, cal, BY_NAME)
    assert sem.q[0, S["left_shoulder_roll"]] == pytest.approx(0.5)
    sem_a = g1_to_semantic(m, cal, RetargetOptions(waist_mode="drop"))  # anatomical: same sign, small change
    assert 0.4 < sem_a.q[0, S["left_shoulder_roll"]] < 0.6
    assert sem.q[0, S["right_shoulder_roll"]] == pytest.approx(0.0)
    # right arm abduction is negative in G1 (and therefore in the semantic space)
    mr = standing_motion(right_shoulder_roll_joint=-0.5)
    yr = kin.forward(mr.root_pos, mr.root_quat_wxyz, mr.dof).body_pos("right_elbow_link")[0, 1]
    assert yr < -0.18
    assert g1_to_semantic(mr, cal, BY_NAME).q[0, S["right_shoulder_roll"]] == pytest.approx(-0.5)
    assert -0.6 < g1_to_semantic(mr, cal, RetargetOptions(waist_mode="drop")).q[0, S["right_shoulder_roll"]] < -0.4


def test_all_named_joints_copied_and_dropped_ones_ignored(cal):
    rng = np.random.default_rng(5)
    vals = {n: float(rng.uniform(-0.2, 0.2)) for n in J}
    sem = g1_to_semantic(standing_motion(**vals), cal, BY_NAME)
    for s in S:
        assert sem.q_requested[0, S[s]] == pytest.approx(vals[f"{s}_joint"]), s


def _thigh(q, side):
    p, r, y = (q[..., S[f"{side}_hip_{a}"]] for a in ("pitch", "roll", "yaw"))
    return rot_y(p + G1_HIP_ROLL_LINK_PITCH) @ rot_x(r) @ rot_z(y)


def test_waist_fold_preserves_thigh_and_torso_orientation(cal):
    rng = np.random.default_rng(6)
    t = 25
    m = standing_motion(t)
    for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        m.dof[:, J[n]] = rng.uniform(-0.4, 0.4, t)
    for n in ("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "right_hip_pitch_joint"):
        m.dof[:, J[n]] = rng.uniform(-0.3, 0.3, t)
    kin = load_g1_kinematics()
    fk = kin.forward(m.root_pos, m.root_quat_wxyz, m.dof)
    sem = g1_to_semantic(m, cal, RetargetOptions(waist_mode="fold", joint_mapping="by_name", ground_fix=False,
                                                 foot_lock_xy=False))
    r_body = quat_to_matrix(sem.pelvis_quat_wxyz)
    np.testing.assert_allclose(r_body, fk.body_rot("torso_link"), atol=1e-9)
    q_raw = np.zeros((t, 22))
    for s in S:
        q_raw[:, S[s]] = m.dof[:, J[f"{s}_joint"]]
    for side in ("left", "right"):
        world_g1 = fk.body_rot("pelvis") @ _thigh(q_raw, side)
        world_db = r_body @ _thigh(sem.q_requested, side)
        np.testing.assert_allclose(world_db, world_g1, atol=1e-5)
        # the G1 thigh link frame from FK is the same object (MJCF link quat has 6 significant digits)
        np.testing.assert_allclose(world_g1, fk.body_rot(f"{side}_hip_yaw_link"), atol=1e-6)


def test_auto_waist_never_adds_hip_saturation(cal):
    files = discover_source_files("soma_retargeter_g1")
    if not files:
        pytest.skip("soma clips not on disk")
    m = load_g1_motion(files[0], "soma_retargeter_g1")  # throw_ball: strong torso twist
    hips = [S[f"{s}_hip_{a}"] for s in ("left", "right") for a in ("pitch", "roll", "yaw")]
    exc = {}
    for mode in ("auto", "drop", "fold"):
        sem = g1_to_semantic(m, cal, RetargetOptions(waist_mode=mode))
        exc[mode] = np.abs(sem.q_requested[:, hips] - sem.q[:, hips]).sum(axis=1)
        if mode == "auto":
            alpha = sem.meta["waist_fold_fraction"]
            assert 0.0 <= alpha["min"] <= alpha["max"] <= 1.0
            upper_auto = sem.meta["upper_body_error_deg"]["chosen_mode_mean"]
            upper_drop = sem.meta["upper_body_error_deg"]["drop_mode_mean"]
            assert upper_auto <= upper_drop + 1e-9
    assert np.all(exc["auto"] <= exc["drop"] + np.deg2rad(0.5) + 1e-9)
    assert exc["fold"].max() > exc["auto"].max()  # full fold would saturate the hips more on this clip


def test_root_standing_straight_legs(cal):
    """Straight-leg G1 stand -> Dropbear pelvis at its standing hip height, root 'world' below the soles."""
    sem_res = retarget_g1_motion(standing_motion(), cal, RetargetOptions(waist_mode="drop"))
    sem = sem_res.semantic
    np.testing.assert_allclose(sem.pelvis_pos[:, :2], 0.0, atol=1e-9)  # re-centred
    geom = LegGeometry.from_calibration(cal)
    lf = leg_fk(sem.pelvis_pos, quat_to_matrix(sem.pelvis_quat_wxyz), sem.q, geom)
    np.testing.assert_allclose(lf.sole_height, 0.0, atol=2e-3)  # soles on the ground after ground fix
    assert sem.pelvis_pos[0, 2] == pytest.approx(cal.standing_hip_height, abs=5e-3)
    sole_in_root = cal.raw["segment_lengths"]["sole_z_in_root"]
    assert sem_res.root_pos[0, 2] == pytest.approx(-sole_in_root, abs=5e-3)  # 'world' origin ~12.5 cm below soles
    np.testing.assert_allclose(sem_res.root_quat_wxyz[:, 0], 1.0, atol=1e-9)
    assert sem.contacts.all()


def test_scale_is_hip_height_ratio(cal):
    sem = g1_to_semantic(standing_motion(), cal)
    assert sem.meta["scale"] == pytest.approx(cal.standing_hip_height / load_g1_kinematics().standing_hip_height)


def test_contacts_flight_phase(cal):
    t, fps = 60, 30.0
    m = standing_motion(t, fps)
    lift = np.zeros(t)
    lift[25:35] = 0.25  # 1/3 s jump
    m.root_pos[:, 2] += lift
    sem = g1_to_semantic(m, cal)
    assert sem.contacts[:20].all() and sem.contacts[45:].all()
    assert not sem.contacts[27:33].any()


def test_foot_lock_reduces_slip_on_real_clip(cal):
    files = discover_source_files("kimodo_g1")
    if not files:
        pytest.skip("kimodo clips not on disk")
    m = load_g1_motion([f for f in files if "walk" in f.name][0], "kimodo_g1")
    off = retarget_g1_motion(m, cal, RetargetOptions(foot_lock_xy=False)).metrics["stance_foot_slip_p95_mps"]
    on = retarget_g1_motion(m, cal, RetargetOptions()).metrics["stance_foot_slip_p95_mps"]
    assert on < off


def test_motor_output_shapes_and_saturation_report(cal):
    files = discover_source_files("unitree_rl_lab_mimic")
    if not files:
        pytest.skip("rl_lab clips not on disk")
    res = retarget_g1_motion(load_g1_motion(files[0], "unitree_rl_lab_mimic"), cal)
    assert res.motor_q.shape == (res.semantic.num_frames, 22)
    assert np.all(np.isfinite(res.motor_q))
    per = res.saturation["per_joint"]
    assert set(per) == set(S)
    # deep-knee dance on the MOCK ranges must report saturation instead of hiding it
    assert res.saturation["frames_with_any_saturation_frac"] > 0.0
    assert res.saturation["usd_motor_limit_violation_deg"] == {}


# ---------------------------------------------------------------------------------------------------
# anatomical (segment-frame) mapping
# ---------------------------------------------------------------------------------------------------


def test_anatomical_zero_pose_and_straight_elbow(cal):
    kin = load_g1_kinematics()
    assert 1.3 < kin.elbow_straight_value < 1.45  # by joint centres the G1 arm is straight at ~1.385 ...
    assert kin.elbow_axes_collinear_value == pytest.approx(np.pi / 2, abs=1e-3)  # ... by joint axes at pi/2
    assert abs(kin.elbow_semantic_offset) < 1e-3
    sem = g1_to_semantic(standing_motion(), cal, RetargetOptions(waist_mode="drop"))
    for n in ("left_hip_pitch", "left_hip_roll", "left_hip_yaw", "right_shoulder_pitch", "left_shoulder_yaw"):
        assert abs(sem.q_requested[0, S[n]]) < 1e-3, n  # MJCF 6-digit link quats: ~6e-5 rad inconsistency
    assert sem.q_requested[0, S["left_elbow"]] == pytest.approx(kin.elbow_semantic_offset)
    e = kin.elbow_straight_value
    straight = standing_motion(left_elbow_joint=e, right_elbow_joint=e)
    sem = g1_to_semantic(straight, cal, RetargetOptions(waist_mode="drop"))
    assert sem.q_requested[0, S["left_elbow"]] == pytest.approx(e + kin.elbow_semantic_offset, abs=1e-9)
    # the right arm is mirror-symmetric: it is straight at the same value
    fk = kin.forward(straight.root_pos[:1], straight.root_quat_wxyz[:1], straight.dof[:1])
    s_, e_, w_ = (fk.body_pos(f"right_{b}")[0] for b in ("shoulder_roll_link", "elbow_link", "wrist_roll_link"))
    u, v = e_ - s_, w_ - e_
    axis = fk.body_rot("right_elbow_link")[0][:, 1]
    signed_flex = np.arctan2(np.cross(u, v) @ axis, u @ v)  # in-plane flexion (lateral offsets excluded)
    assert abs(np.degrees(signed_flex)) < 0.1


@pytest.mark.parametrize("mode", ["auto", "drop", "fold"])
def test_anatomical_limbs_keep_g1_world_orientation(cal, mode):
    rng = np.random.default_rng(8)
    t = 40
    m = standing_motion(t)
    for n in J:
        if "wrist" not in n:
            m.dof[:, J[n]] = rng.uniform(-0.5, 0.5, t)
    m.dof[:, J["left_shoulder_roll_joint"]] = rng.uniform(0.0, 0.9, t)  # away from the YXZ singularity
    kin = load_g1_kinematics()
    fk = kin.forward(m.root_pos, m.root_quat_wxyz, m.dof)
    sem = g1_to_semantic(m, cal, RetargetOptions(waist_mode=mode))
    assert sem.meta["euler_fit_max_residual_rad"] < 1e-5
    r_body = quat_to_matrix(sem.pelvis_quat_wxyz)
    tilt_fix = rot_y(np.full(t, -G1_HIP_ROLL_LINK_PITCH))
    for side in ("left", "right"):
        qh = sem.q_requested[:, [S[f"{side}_hip_{a}"] for a in ("pitch", "roll", "yaw")]]
        np.testing.assert_allclose(r_body @ rot_y(qh[:, 0]) @ rot_x(qh[:, 1]) @ rot_z(qh[:, 2]),
                                   fk.body_rot(f"{side}_hip_yaw_link") @ tilt_fix, atol=1e-5)
        qs = sem.q_requested[:, [S[f"{side}_shoulder_{a}"] for a in ("pitch", "roll", "yaw")]]
        np.testing.assert_allclose(r_body @ rot_y(qs[:, 0]) @ rot_x(qs[:, 1]) @ rot_z(qs[:, 2]),
                                   fk.body_rot(f"{side}_shoulder_yaw_link"), atol=1e-5)


def test_anatomical_mapping_is_continuous_on_jumping_jacks(cal):
    files = [f for f in discover_source_files("kimodo_g1") if "jumpjack" in f.name]
    if not files:
        pytest.skip("kimodo jumpjack not on disk")
    m = load_g1_motion(files[0], "kimodo_g1")
    sem = g1_to_semantic(m, cal)
    names = [f"{s}_shoulder_{a}" for s in ("left", "right") for a in ("pitch", "roll", "yaw")]
    step_sem = np.abs(np.diff(sem.q_requested[:, [S[n] for n in names]], axis=0)).max()
    step_g1 = np.abs(np.diff(m.dof[:, [J[n + "_joint"] for n in names]], axis=0)).max()
    assert step_sem < 2.0 * step_g1 + 0.05, (step_sem, step_g1)
    # the continuity is bought with a bounded orientation residual near the singularity
    assert sem.meta["euler_fit_max_residual_rad"] < np.deg2rad(20.0)


def test_contacts_rolling_ground_floaty_and_single_leg(cal):
    t, fps = 180, 30.0
    # video-style float: the whole robot drifts 10 cm up in the second half, feet static -> still contact
    m = standing_motion(t, fps)
    m.root_pos[90:, 2] += 0.10
    sem = g1_to_semantic(m, cal)
    # centred 2 s rolling window: an abrupt step is absorbed after half a window (1 s = 30 frames)
    assert sem.contacts[5:85].all() and sem.contacts[125:175].all()
    # single-leg balance: left foot held ~20+ cm up for 3 s -> left not in contact, right in contact
    m = standing_motion(t, fps, left_hip_pitch_joint=-0.8, left_knee_joint=1.2)
    sem = g1_to_semantic(m, cal)
    assert not sem.contacts[:, 0].any() and sem.contacts[:, 1].all()
