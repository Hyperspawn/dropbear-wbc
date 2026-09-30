"""CPU tests of the foot-contact stage (``dropbear_wbc.motion.foot_contact``, foot_contact track 2026-09-24).

Pure helpers (contact cleaning, hysteresis with external speeds, motor-step projection, human-foot contact rule) run
everywhere. The stage itself needs the REAL calibration and its raw sweep (``CalibrationFK``); those tests skip without it.
"""
from __future__ import annotations

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ to sys.path)

from dropbear_wbc.motion.contacts import HysteresisParams, contact_segments, detect_contacts_hysteresis  # noqa: E402
from dropbear_wbc.motion.foot_contact import (  # noqa: E402
    FootContactParams,
    clean_contacts,
    leg_motor_columns,
    project_motor_steps,
)

# ------------------------------------------------------------------------------------------------ pure helpers


def test_clean_contacts_fills_short_swing_and_drops_short_stance():
    fps = 50.0
    c = np.ones((100, 2), dtype=bool)
    c[40:42, 0] = False  # 2-frame lift (< min_swing 0.08 s = 4 frames): filled
    c[:, 1] = False
    c[10:13, 1] = True  # 3-frame touch (< min_stance 0.1 s = 5 frames): dropped
    c[50:80, 1] = True  # real stance: kept
    out = clean_contacts(c, fps, min_stance_s=0.10, min_swing_s=0.08)
    assert out[:, 0].all()
    assert not out[10:13, 1].any()
    assert out[50:80, 1].all() and out[:, 1].sum() == 30


def test_hysteresis_uses_external_speeds():
    """Regression: detect_contacts_hysteresis(speeds=...) must use the given speeds (a name clash once discarded them)."""
    fps, t = 50.0, 100
    h = np.zeros((t, 2))  # both feet on the ground all the time
    pos = np.zeros((t, 2, 3))  # and not moving
    speeds = np.zeros((t, 2))
    speeds[30:60, 1] = 1.0  # ... but the right foot's external speed says it slides at 1 m/s
    p = HysteresisParams(h_on=0.03, h_off=0.05, v_on=0.3, v_off=0.6, min_stance_s=0.1, min_swing_s=0.08)
    c, _, sig = detect_contacts_hysteresis(h, pos, fps, p, ground=0.0, speeds=speeds)
    assert c[:, 0].all()
    assert not c[30:60, 1].any() and c[:30, 1].all() and c[60:, 1].all()
    np.testing.assert_allclose(sig["speed"], speeds)
    c2, _, _ = detect_contacts_hysteresis(h, pos, fps, p, ground=0.0)  # without: the positions say "planted"
    assert c2.all()


def test_project_motor_steps_is_minimal_and_feasible():
    t = 40
    m = np.zeros((t, 3))
    m[20:, 0] = 1.0  # a 1 rad jump in one frame
    m[:, 1] = np.linspace(0, 0.5, t)  # already feasible: untouched
    m[:, 2] = 0.05 * np.sin(np.arange(t))  # feasible
    out, info = project_motor_steps(m, 0.18)
    assert np.abs(np.diff(out, axis=0)).max() <= 0.18 + 1e-6
    np.testing.assert_array_equal(out[:, 1:], m[:, 1:])
    assert list(info["columns"].keys()) == [0]
    # the L2 projection of a symmetric step is a symmetric ramp centred on the jump (6 steps of <= 0.18 rad)
    np.testing.assert_allclose(out[:, 0] + out[::-1, 0], 1.0, atol=1e-5)
    assert (np.abs(out[:, 0] - m[:, 0]) > 1e-6).sum() <= 6
    assert info["columns"][0]["max_step_after"] <= 0.18 + 1e-6


def test_leg_motor_columns_are_the_12_leg_motors():
    from dropbear_wbc.motion.names import MOTOR_NAMES

    cols = leg_motor_columns()
    assert len(set(cols.tolist())) == 12
    assert sorted(MOTOR_NAMES[c] for c in cols) == sorted(MOTOR_NAMES[:12])


def test_human_contact_rule_rejects_sliding_feet():
    """The stage's human-foot rule: a foot sliding along the ground (walk steps that barely clear it) is NOT stance."""
    from dropbear_wbc.motion.gmr_serial import HUMAN_HYSTERESIS, GmrClip, human_foot_contacts, human_foot_speeds

    fps, t = 50.0, 150
    raw = np.zeros((t, 2, 2, 3))  # (T, foot, [ankle, toe], xyz)
    raw[:, :, 0, 2] = 0.08  # ankle height at rest
    raw[:, :, 1, 0] = 0.15  # toe 15 cm in front of the ankle, on the ground
    raw[:, 1, :, 1] = -0.2
    # right foot: planted 0-49, slides 1 m/s 1 cm above the ground 50-99 (a low swing), planted 100-149
    x = np.concatenate([np.zeros(50), np.linspace(0, 1.0, 50), np.full(50, 1.0)])
    raw[:, 1, :, 0] += x[:, None]
    raw[50:100, 1, :, 2] += 0.01
    clip = GmrClip(fps=fps, pelvis_pos=np.zeros((t, 3)), pelvis_quat_wxyz=np.tile([1.0, 0, 0, 0], (t, 1)),
                   q_sem=np.zeros((t, 22)), meta={}, task_bodies=[], task_pos_err=np.zeros((t, 0)),
                   task_rot_err=np.zeros((t, 0)), human_feet_raw=raw)
    sp = human_foot_speeds(clip)
    assert sp[60:90, 1].min() > 0.9 and sp[:, 0].max() < 1e-9
    c = human_foot_contacts(clip, hysteresis=HUMAN_HYSTERESIS)
    assert c[:, 0].all()
    assert c[:45, 1].all() and c[105:, 1].all() and not c[52:98, 1].any()


# ------------------------------------------------------------------------------------------------ the stage


@pytest.fixture(scope="module")
def plant():
    try:
        from dropbear_wbc.motion.calibration_view import load_calibration
        from dropbear_wbc.motion.foot_contact import PlantLegModel, get_calibration_fk

        cal = load_calibration()
        cfk = get_calibration_fk(cal.path)
    except (FileNotFoundError, OSError, KeyError) as e:  # pragma: no cover - needs the real calibration + raw sweep
        pytest.skip(f"real calibration / raw sweep not available: {e}")
    return cal, cfk, PlantLegModel(cfk)


def _standing_clip(cal, cfk, t=60):
    q = np.tile(cfk.smap.motor_to_semantic(np.asarray(cfk.cal["standing_motor_pos"], dtype=np.float64)[None]), (t, 1))
    pel = np.tile([0.0, 0.0, 1.0], (t, 1))
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (t, 1))
    return pel, quat, q


def _world_feet(model, cfk, cal, res):
    from dropbear_wbc.motion.foot_contact import SIDES
    from dropbear_wbc.motion.rotations import quat_to_matrix

    tr = cal.pelvis_T_root
    r_pel = quat_to_matrix(res.pelvis_quat_wxyz)
    r_root, p_root = r_pel @ tr[:3, :3], res.pelvis_pos + r_pel @ tr[:3, 3]
    out = {}
    for s in SIDES:
        r_rel, p_rel = cfk.fk.leg(s, res.motor_q)["foot"]
        rw = r_root @ r_rel
        pw = p_root + np.einsum("tij,tj->ti", r_root, p_rel)
        f = rw @ model.r_ref[s].T
        out[s] = {"low": model.lowest_z(s, rw, pw), "centre": model.sole_centre(s, rw, pw),
                  "tilt_deg": np.degrees(np.arccos(np.clip(f[:, 2, 2], -1, 1)))}
    return out


def test_stage_pins_perturbed_stance_feet_flat_and_locked(plant):
    """Retargeted legs that tilt / lift a stance foot (knee and ankle errors, pelvis bobbing) come out with both soles
    flat on z = 0 at world-locked positions, within the motor box and the step limit."""
    from dropbear_wbc.motion.foot_contact import foot_contact_stage
    from dropbear_wbc.motion.names import SEMANTIC_INDEX

    cal, cfk, model = plant
    t = 60
    pel, quat, q = _standing_clip(cal, cfk, t)
    ph = np.sin(np.linspace(0, 2 * np.pi, t))
    q[:, SEMANTIC_INDEX["left_knee"]] += 0.12 * ph  # knee error -> the foot moves up/down
    q[:, SEMANTIC_INDEX["right_ankle_pitch"]] += 0.10 * ph  # ankle error -> the sole tilts
    pel[:, 2] += 0.03 * ph
    pel[:, 1] += 0.02 * ph  # lateral sway
    contacts = np.ones((t, 2), dtype=bool)
    res = foot_contact_stage(50.0, pel, quat, q, contacts, cfk, cal.pelvis_T_root, FootContactParams(), model=model)
    rep = res.report
    assert rep["before"]["stance_tilt_deg"]["max"] > 3.0  # the perturbation is real
    assert rep["after"]["stance_pos_mm"]["max"] < 3.0
    assert rep["after"]["stance_tilt_deg"]["max"] < 1.0
    feet = _world_feet(model, cfk, cal, res)
    for s in ("left", "right"):
        assert np.abs(feet[s]["low"]).max() < 0.003
        assert np.linalg.norm(feet[s]["centre"][:, :2] - feet[s]["centre"][0, :2], axis=1).max() < 0.003  # no slip
        assert feet[s]["tilt_deg"].max() < 1.0
    m12 = res.leg_motors
    lo = np.concatenate([model.lo["left"], model.lo["right"]])
    hi = np.concatenate([model.hi["left"], model.hi["right"]])
    assert (m12 >= lo - 1e-9).all() and (m12 <= hi + 1e-9).all()
    assert np.abs(np.diff(m12, axis=0)).max() <= FootContactParams().max_motor_step_rad + 0.01
    # arms untouched, legs replaced consistently (semantic output = motor_to_semantic of the leg motors)
    arm = [SEMANTIC_INDEX[n] for n in SEMANTIC_INDEX if "shoulder" in n or "elbow" in n or "wrist" in n]
    np.testing.assert_allclose(res.q_sem[:, arm], q[:, arm], atol=1e-9)
    np.testing.assert_allclose(cfk.smap.motor_to_semantic(res.motor_q)[:, model.sem_idx["left"]],
                               res.q_sem[:, model.sem_idx["left"]], atol=1e-6)


def test_stage_swing_foot_clears_ground_and_relands_on_its_target(plant):
    """A swing phase whose retargeted foot stays on (or in) the ground is lifted to the clearance, and the foot lands
    where the next stance target is (the swing offsets blend between the two stance phases)."""
    from dropbear_wbc.motion.foot_contact import foot_contact_stage
    from dropbear_wbc.motion.names import SEMANTIC_INDEX

    cal, cfk, model = plant
    t = 80
    pel, quat, q = _standing_clip(cal, cfk, t)
    q[:, SEMANTIC_INDEX["right_knee"]] -= 0.05  # a straighter right leg pushes its sole into the ground
    contacts = np.ones((t, 2), dtype=bool)
    contacts[30:50, 1] = False  # right swing 0.4 s
    p = FootContactParams()
    res = foot_contact_stage(50.0, pel, quat, q, contacts, cfk, cal.pelvis_T_root, p, model=model)
    feet = _world_feet(model, cfk, cal, res)
    low_r = feet["right"]["low"]
    assert low_r[35:45].min() >= p.clearance_m - 0.002  # mid-swing: clear by ~1 cm
    assert low_r[30:50].min() > -0.001  # never below the ground during the swing
    st = contacts[:, 1]
    assert np.abs(low_r[st]).max() < 0.003 and np.abs(feet["left"]["low"]).max() < 0.003
    segs = contact_segments(st)
    c = feet["right"]["centre"]
    for a, b in segs:
        assert np.linalg.norm(c[a:b, :2] - c[a, :2], axis=1).max() < 0.003


def test_plant_gate_prediction_passes_after_stage(plant):
    from dropbear_wbc.motion.foot_contact import foot_contact_stage, plant_gate_metrics
    from dropbear_wbc.motion.names import SEMANTIC_INDEX
    from dropbear_wbc.motion.rotations import matrix_to_quat, quat_to_matrix

    cal, cfk, model = plant
    t = 50
    pel, quat, q = _standing_clip(cal, cfk, t)
    q[:, SEMANTIC_INDEX["left_ankle_roll"]] += 0.08
    pel[:, 2] += 0.05  # floating by 5 cm
    contacts = np.ones((t, 2), dtype=bool)
    res = foot_contact_stage(50.0, pel, quat, q, contacts, cfk, cal.pelvis_T_root, FootContactParams(), model=model)
    tr = cal.pelvis_T_root
    r_pel = quat_to_matrix(res.pelvis_quat_wxyz)
    root_pos = res.pelvis_pos + r_pel @ tr[:3, 3]
    root_quat = matrix_to_quat(r_pel @ tr[:3, :3])
    g = plant_gate_metrics(cfk, res.motor_q, root_pos, root_quat, res.contacts, 50.0)
    assert g["gates_pass_predicted"]
    assert max(abs(g["contact_sole_z_mm"]["min"]), abs(g["contact_sole_z_mm"]["max"])) < 3.0
