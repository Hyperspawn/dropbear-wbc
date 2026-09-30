"""CPU tests of the GMR -> Dropbear (gmr-serial-v1) path.

* the generated GMR IK configs match the current serial model (bodies exist, unit quaternions, provenance SHA);
* the converter (``dropbear_wbc.motion.gmr_serial``) maps GMR joints BY NAME, reproduces the calibration's
  semantic-zero motors at the zero pose, converts the root to the USD ``world`` body and writes a valid contract CSV;
* every clip in ``data/motions/gmr_lafan1`` is a valid contract clip with the LAFAN1 license and small saturation;
* ``DropbearMotorLimit`` (needs ``mink``; skipped otherwise) produces consistent motor-space inequalities.

Run: ``python -m pytest tests/test_gmr_serial.py -q`` (system Python, .venv-newton or .venv-gmr).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ to sys.path)

mujoco = pytest.importorskip("mujoco")

from dropbear_wbc.kinematics import serial_model as sm  # noqa: E402
from dropbear_wbc.kinematics.semantic import SEMANTIC_NAMES  # noqa: E402
from dropbear_wbc.motion.calibration_view import load_calibration  # noqa: E402
from dropbear_wbc.motion.gmr_serial import gmr_to_dropbear, load_gmr_npz  # noqa: E402
from dropbear_wbc.motion.motion_csv import read_motion_csv, validate_motion_files, write_motion  # noqa: E402
from dropbear_wbc.motion.rotations import quat_to_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "data/calibration/dropbear_semantic_calibration.json"
XML = ROOT / "data/robot/dropbear_serial.xml"
CFG = ROOT / "data/robot/gmr"
CLIPS = ROOT / "data/motions/gmr_lafan1"

pytestmark = pytest.mark.skipif(not (XML.is_file() and CAL.is_file()), reason="serial model not built")


@pytest.fixture(scope="module")
def meta():
    return json.loads(XML.with_suffix(".json").read_text())


@pytest.fixture(scope="module")
def cfk():
    return sm.CalibrationFK(CAL)


@pytest.mark.parametrize("name", ["bvh_lafan1_to_dropbear.json", "smplx_to_dropbear.json"])
def test_gmr_configs(name, meta):
    path = CFG / name
    if not path.is_file():
        pytest.skip("GMR configs not generated")
    cfg = json.loads(path.read_text())
    assert cfg["_dropbear"]["serial_model_calibration_sha256"] == meta["inputs"]["calibration_sha256"], \
        "stale GMR config: rerun tools/make_gmr_configs.py"
    m = mujoco.MjModel.from_xml_path(str(XML))
    bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
    assert cfg["robot_root_name"] == "pelvis"
    for table in ("ik_match_table1", "ik_match_table2"):
        for body, entry in cfg[table].items():
            assert body in bodies, body
            human, wp, wr, poff, q = entry
            assert human in cfg["human_scale_table"], human
            assert wp >= 0 and wr >= 0 and len(poff) == 3
            assert abs(np.linalg.norm(q) - 1.0) < 1e-6
    assert all(v > 0 for v in cfg["human_scale_table"].values())


def _fake_gmr(tmp_path: Path, meta: dict, q_sem: np.ndarray, shuffle: bool) -> Path:
    t = len(q_sem)
    names = list(SEMANTIC_NAMES)
    order = np.random.default_rng(0).permutation(22) if shuffle else np.arange(22)
    qpos = np.zeros((t, 29))
    qpos[:, 2] = 0.98
    qpos[:, 3] = 1.0
    qpos[:, 7:] = q_sem[:, order]
    path = tmp_path / "fake.npz"
    np.savez(path, fps=np.float64(30.0), qpos=qpos, qpos_joint_names=np.array([names[i] for i in order]),
             robot="dropbear", task_bodies=np.array(["Hips"]), task_robot_bodies=np.array(["pelvis"]),
             task_pos_err=np.zeros((t, 1)), task_rot_err=np.zeros((t, 1)), human_pos=np.zeros((t, 1, 3)),
             solve_ms=np.zeros(t), meta=json.dumps({"bvh": "fake.bvh", "format": "lafan1", "frame_range": [0, t],
                                                     "ik_config": "none", "xml": str(XML)}))
    return path


@pytest.mark.parametrize("shuffle", [False, True])
def test_converter_zero_pose(tmp_path, meta, cfk, shuffle):
    cal = load_calibration(CAL)
    q = np.zeros((6, 22))
    clip = load_gmr_npz(_fake_gmr(tmp_path, meta, q, shuffle))
    assert np.allclose(clip.q_sem, q)
    res = gmr_to_dropbear(clip, cal, cfk, meta)
    assert np.allclose(res.motor_q, cfk.qz, atol=2e-6), np.abs(res.motor_q - cfk.qz).max()
    assert res.saturation["frames_with_any_saturation_frac"] == 0.0
    # root = pelvis * inv(pelvis_in_root): with an identity pelvis the root sits pelvis_in_root below the pelvis
    pel = np.asarray(meta["frames"]["pelvis_in_root"]["pos"])
    assert np.allclose(res.root_pos - res.pelvis_pos, -pel, atol=1e-9)
    assert np.allclose(quat_to_matrix(res.root_quat_wxyz), np.eye(3), atol=1e-9)
    # ground fix: standing still at the zero pose -> the lowest sole is on the ground, both feet in contact
    assert abs(res.metrics["min_sole_height_m"]) < 1e-9
    assert res.contacts.all()
    csv, _ = write_motion(tmp_path / "out", "zero", fps=30.0, root_pos=res.root_pos, root_quat_wxyz=res.root_quat_wxyz,
                          motor_q=res.motor_q, sidecar={"source": "test", "source_file": "fake.bvh",
                                                        "source_license": {"license": "test"},
                                                        "retarget_method": "gmr-serial-v1", "contact_hint": None,
                                                        "notes": []},
                          semantic=(res.pelvis_pos, res.pelvis_quat_wxyz, res.q_used, res.contacts))
    assert validate_motion_files(csv) == []


def test_converter_rejects_other_robots(tmp_path, meta):
    path = _fake_gmr(tmp_path, meta, np.zeros((3, 22)), False)
    d = dict(np.load(path))
    d["robot"] = np.array("unitree_g1")
    np.savez(tmp_path / "g1.npz", **d)
    with pytest.raises(ValueError):
        load_gmr_npz(tmp_path / "g1.npz")


def test_gmr_lafan1_clips(cfk):
    csvs = sorted(CLIPS.glob("*.csv"))
    if not csvs:
        pytest.skip("no gmr_lafan1 clips")
    for csv in csvs:
        assert validate_motion_files(csv) == [], csv
        mo = read_motion_csv(csv)
        side = mo.sidecar
        assert side["retarget_method"].startswith("gmr-serial-v1"), csv
        assert side["source_license"]["license"] == "CC-BY-NC-ND-4.0" and side["source_license"]["redistributable"] is False
        assert side["serial_model"]["serial_model_calibration_sha256"] == cfk.sha256, f"{csv.name}: stale (rerun batch)"
        worst = max(v["max_excess_deg"] for v in side["saturation"]["per_joint"].values())
        assert worst < 3.0, (csv.name, worst)
        sem = np.loadtxt(csv.parent / side["semantic_trajectory"], delimiter=",", skiprows=1)[:, 7:29]
        # semantic file = what the motors realise (forward map)
        assert np.abs(cfk.smap.motor_to_semantic(mo.motor_q) - sem).max() < 1e-4, csv.name
        # ... and the inverse returns the motors wherever it does not clip. The foot-contact stage (library v4) solves
        # in the full motor box and reaches calf-motor pairs at the edge of the ankle's feasible set, where the lut2d
        # inverse (whole-cell validity) snaps to the nearest valid node (foot_contact track, 2026-09-24)
        m, rep = cfk.smap.semantic_to_motor(sem, return_report=True)
        ok = ~np.asarray(rep.clipped).reshape(len(sem), -1).any(axis=1)
        if ok.any():
            assert np.abs(m[ok] - mo.motor_q[ok]).max() < 1e-4, csv.name


def test_motor_limit_constraint(meta):
    pytest.importorskip("mink")
    from dropbear_wbc.kinematics.semantic import SemanticMap
    from dropbear_wbc.motion.gmr_limits import DropbearMotorLimit

    class _Cfg:
        def __init__(self, q):
            self.q = q

    m = mujoco.MjModel.from_xml_path(str(XML))
    smap = SemanticMap.load(CAL)
    lim = DropbearMotorLimit(m, smap)
    q = np.zeros(m.nq)
    q[3] = 1.0
    c = lim.compute_qp_inequalities(_Cfg(q), 0.005)
    assert c.G.shape[1] == m.nv and c.G.shape[0] == c.h.shape[0] == 2 * (4 * 3 + 2 * 2)
    assert (c.h >= -1e-9).all()  # the zero pose is feasible
    # an infeasible hip combination (inside the semantic box, outside the motor limits) violates a row
    si = {n: i for i, n in enumerate(SEMANTIC_NAMES)}
    s = np.zeros(22)
    s[[si["left_hip_pitch"], si["left_hip_roll"], si["left_hip_yaw"]]] = smap.semantic_limits[
        [si["left_hip_pitch"], si["left_hip_roll"], si["left_hip_yaw"]], 1]
    q[lim.qadr] = s
    c = lim.compute_qp_inequalities(_Cfg(q), 0.005)
    assert (c.h < 0).any()


def test_gmr_lafan1_contacts_follow_the_source():
    """Contact hints come from the human source feet (review fix 2026-09-24): the no-contact fraction of the hint equals
    the source's, the jump clip keeps its flight phases, and the sidecar records the source-vs-plant agreement."""
    # clip sidecars only (not the settle's <clip>_v4.report.json / the validator's <clip>_v4.validation.json)
    sides = {p.stem: json.loads(p.read_text()) for p in sorted(CLIPS.glob("lafan1_*.json")) if "." not in p.stem}
    if not sides:
        pytest.skip("no gmr_lafan1 clips")
    for name, side in sides.items():
        ca = side["metrics"].get("contact_agreement") or {}
        assert ca.get("source", "").startswith("human source feet"), name
        hint = np.stack([side["contact_hint"]["left"], side["contact_hint"]["right"]], 1)
        assert abs(float((~hint.any(1)).mean()) - ca["no_foot_in_contact_frac_source"]) < 1e-9, name
        # per-frame grounding (v3) or the foot-contact stage, which pins stance soles to z = 0 (library v4, foot_contact)
        gm = side["metrics"]["ground_method"]
        assert "per-frame" in gm or gm.startswith("foot-contact stage"), name
        assert "flag" in ca and "disagreement_frac" in ca, name
    if "lafan1_jumps1_subject1" in sides:
        assert sides["lafan1_jumps1_subject1"]["metrics"]["contact_agreement"]["no_foot_in_contact_frac_source"] > 0.3
    for name in ("lafan1_dance2_subject1", "lafan1_run1_subject2"):
        if name in sides:  # the IK-foot hints had 49 % / 86 % "no foot in contact"; the source has 3 % / 11 %
            assert sides[name]["metrics"]["contact_agreement"]["no_foot_in_contact_frac_source"] < 0.2, name


def test_gmr_lafan1_arm_direction_and_root_policy():
    """Arm targets anchored at the robot shoulder and the blend root policy (review fixes 2026-09-24)."""
    gdir = ROOT / "logs/review_fixes/gmr_v2/gmr"
    npzs = sorted(gdir.glob("*_dropbear.npz"))
    if not npzs:
        pytest.skip("no review_fixes GMR solves")
    for p in npzs:
        d = np.load(p, allow_pickle=False)
        ad = np.degrees(d["arm_dir_err"][:, :, 0])  # upper arm, left/right
        assert np.median(ad) < 6.0, (p.name, np.median(ad))  # was 7.4-14.2 deg with the human-shoulder anchor
        tb = [str(x) for x in d["task_bodies"]]
        rot50 = np.degrees(np.median(d["task_rot_err"], axis=0))
        assert rot50[tb.index("Hips")] < 12.0, (p.name, rot50[tb.index("Hips")])  # was 13-20 deg (chest-following)
