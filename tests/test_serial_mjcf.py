"""FK parity of the DERIVED serial MJCF (``data/robot/dropbear_serial.xml``) against the calibration.

The serial model is a surrogate (docs/SERIAL_MODEL_AND_GMR.md): joint values are dropbear-semantic-v1 angles,
orientations are exact, positions are least-squares fits with two known approximations (hip chain order,
polycentric knee/elbow four-bars). These tests

* check the model matches the CURRENT calibration (else: regenerate with ``tools/build_serial_mjcf.py``),
* check structure (22 semantic hinges, ranges = semantic valid ranges, mass, sites),
* compare MuJoCo FK of the key bodies (feet, hands, head, anchor, thighs, shanks, arms) with
  (a) the calibration's own forward model on >= 256 uniform random semantic poses,
  (b) frames of the motion library (realistic joint distribution), if present,
  (c) the verify-stage PHYSICS poses (the calibration's verify NPZ, else the legacy one),
  and write ``logs/serial_mjcf_gmr/fk_parity_report.json``.

The asserted bounds are regression bounds on the documented accuracy (orientation < 1 deg everywhere; exact
bodies < 2 mm; feet p95 < 30 mm). The < 1 cm target is REPORTED per body (``frac_below_10mm``, ``meets_1cm_*``),
not asserted, because the hip chain-order and four-bar approximations exceed it by design at large angles.

Run: ``.venv-newton/Scripts/python.exe -m pytest tests/test_serial_mjcf.py -q``
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ to sys.path)

mujoco = pytest.importorskip("mujoco")

from dropbear_wbc.kinematics import serial_model as sm  # noqa: E402
from dropbear_wbc.kinematics.calib_fit import Raw, seg_bodies  # noqa: E402
from dropbear_wbc.kinematics.calib_semantics import semantics_from_rotations  # noqa: E402
from dropbear_wbc.kinematics.semantic import SEMANTIC_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "data/robot/dropbear_serial.xml"
META = ROOT / "data/robot/dropbear_serial.json"
CAL = ROOT / "data/calibration/dropbear_semantic_calibration.json"
REPORT = ROOT / "logs/serial_mjcf_gmr/fk_parity_report.json"
LEGACY_VERIFY = ROOT / "logs/calibrate_settle/raw/semantic_verify.npz"
N_RANDOM = 256

pytestmark = pytest.mark.skipif(not (XML.is_file() and META.is_file() and CAL.is_file()),
                                reason="serial model or calibration not built")

_REPORT: dict = {}


def _repo_path(p) -> Path:
    """The serial-model meta records absolute build paths; resolve them inside this checkout when they do not exist."""
    p = Path(p)
    if p.exists():
        return p
    s = str(p).replace("\\", "/")
    return ROOT / s.split("dropbear-wbc/", 1)[1] if "dropbear-wbc/" in s else p


@pytest.fixture(scope="module")
def smod():
    return sm.SerialModel(XML, META)


@pytest.fixture(scope="module")
def cfk():
    return sm.CalibrationFK(CAL)


def _write_report():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(REPORT.read_text()) if REPORT.is_file() else {}
    old.update(_REPORT)
    REPORT.write_text(json.dumps(old, indent=1))


def test_matches_current_calibration(smod):
    inp = smod.meta["inputs"]
    assert inp["calibration_sha256"] == sm.sha256_file(CAL), \
        "data/robot/dropbear_serial.* is stale: run tools/build_serial_mjcf.py"
    assert inp["raw_sweep_sha256"] == sm.sha256_file(_repo_path(inp["raw_sweep"]))
    assert inp["usd_sha256"] == json.loads(CAL.read_text())["usd_sha256"]
    assert "DERIVED, NOT CANONICAL" in smod.meta["status"] and "DERIVED, NOT CANONICAL" in XML.read_text()


def test_structure(smod, cfk):
    m = smod.model
    assert m.nq == 7 + 22 and m.nv == 6 + 22 and m.nu == 22
    assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    lim = cfk.smap.semantic_limits
    for i, n in enumerate(SEMANTIC_NAMES):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        assert j >= 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, n
        assert np.allclose(m.jnt_range[j], lim[i], atol=2e-6), (n, m.jnt_range[j], lim[i])
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        assert a >= 0, f"actuator {n}"
    props = json.loads(_repo_path(smod.meta["inputs"]["body_properties"]).read_text())
    usd_total = sum(b["mass"] for n, b in props["bodies"].items() if n not in sm.ORPHAN_BODIES)
    total = float(m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")])
    assert abs(total - usd_total) < 1e-3, (total, usd_total)
    assert (m.body_mass[1:] >= 0).all()
    tracked = [b for b, lab in sm.KEY_SITE_BODIES.items()]
    for b in tracked:
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"usd_{b}") >= 0, b
    _REPORT["structure"] = {"nq": int(m.nq), "nu": int(m.nu), "total_mass_kg": total, "usd_total_mass_kg": usd_total}


def test_zero_pose_exact(smod, cfk):
    """At the semantic zero every body is exact except the four-bar subtrees, which sit at their least-squares
    zero position (shifted by the recorded ``zero_pose_shift``)."""
    segs, used, _ = cfk.segments(np.zeros((1, 22)))
    par = sm.evaluate_parity(smod, segs, used)
    fit = smod.meta["fit"]
    for lab, r in par.items():
        side = lab.split("_", 1)[0]
        expect = 0.0
        if lab.endswith(("shank", "foot")):
            expect = fit[f"{side}_knee"]["zero_pose_shift_mm"]
        elif lab.endswith(("forearm", "hand")):
            expect = fit[f"{side}_elbow"]["zero_pose_shift_mm"]
        assert abs(r["position"]["max_mm"] - expect) < 0.5, (lab, r["position"], expect)
        assert expect < 12.0, (lab, expect)
        assert r["orientation_deg"]["max"] < 0.05, (lab, r["orientation_deg"])
    _REPORT["zero_pose"] = {k: {"pos_mm": v["position"]["max_mm"], "ori_deg": v["orientation_deg"]["max"]}
                            for k, v in par.items()}
    _write_report()


def _summary(par: dict) -> dict:
    return {k: {"pos_mm": {kk: round(vv, 3) for kk, vv in v["position"].items() if kk != "n"},
                "ori_deg": {kk: round(vv, 4) for kk, vv in v["orientation_deg"].items()},
                "frac_below_10mm": round(v["frac_below_10mm"], 4), "n": v["position"]["n"]} for k, v in par.items()}


EXACT = ("pelvis", "torso", "head", "left_upper_arm", "right_upper_arm")


def _assert_bounds(par: dict, feet_p95_mm: float, feet_max_mm: float):
    for lab, r in par.items():
        assert r["orientation_deg"]["max"] < 1.0, (lab, r["orientation_deg"])
        if lab in EXACT:
            assert r["position"]["max_mm"] < 2.0, (lab, r["position"])
        elif lab.endswith(("forearm", "hand")):
            assert r["position"]["p95_mm"] < 10.0 and r["position"]["max_mm"] < 15.0, (lab, r["position"])
        else:
            assert r["position"]["p95_mm"] < feet_p95_mm and r["position"]["max_mm"] < feet_max_mm, (lab, r["position"])


def test_fk_parity_random_vs_calibration_model(smod, cfk):
    rng = np.random.default_rng(2026)
    lim = cfk.smap.semantic_limits
    q = rng.uniform(lim[:, 0], lim[:, 1], (N_RANDOM, 22))
    segs, used, _ = cfk.segments(q)
    par = sm.evaluate_parity(smod, segs, used)
    _REPORT["random_uniform_vs_calibration_fk"] = {
        "reference": "CalibrationFK (SemanticMap.semantic_to_motor + MeasuredFK from the raw sweep)",
        "poses": f"{N_RANDOM} uniform random semantic poses inside the valid ranges (clipped to reachable)",
        "bodies": _summary(par),
        "meets_1cm_target_p95": {k: v["position"]["p95_mm"] < 10.0 for k, v in par.items()},
    }
    _write_report()
    _assert_bounds(par, feet_p95_mm=30.0, feet_max_mm=50.0)


def test_fk_parity_motion_library_frames(smod, cfk):
    files = sorted(glob.glob(str(ROOT / "data/motions/*/semantic/*.semantic.csv")))
    if not files:
        pytest.skip("no motion library semantic trajectories")
    rows = []
    for f in files:
        a = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)[:, 7:29]
        rows.append(a[:: max(1, len(a) // 40)])
    q = np.concatenate(rows)
    segs, used, _ = cfk.segments(q)
    par = sm.evaluate_parity(smod, segs, used)
    _REPORT["motion_library_frames_vs_calibration_fk"] = {
        "poses": f"{len(q)} frames sampled from {len(files)} data/motions/*/semantic trajectories (clipped)",
        "bodies": _summary(par),
        "meets_1cm_target_p95": {k: v["position"]["p95_mm"] < 10.0 for k, v in par.items()},
    }
    _write_report()
    _assert_bounds(par, feet_p95_mm=30.0, feet_max_mm=50.0)


def test_fk_parity_verify_physics_poses(smod, cfk):
    ver = cfk.cal.get("verification", {}).get("verify_npz")
    path = sm.resolve_path(ver) if ver else None
    note = "calibration's own verify NPZ"
    if path is None or not path.is_file():
        path, note = LEGACY_VERIFY, ("legacy verify NPZ (plant with revolute ankle tie rods, calibration 1a7161d3); "
                                     "the serial kinematic tree is identical, only the ankle closure type differs")
    if not path.is_file():
        pytest.skip("no verify NPZ")
    raw = Raw(path)
    if raw.usd_sha() != cfk.cal["usd_sha256"]:
        pytest.skip("verify NPZ from another USD")
    idx = np.arange(len(raw.program_names))
    sem, diag = semantics_from_rotations(lambda s, k: raw.R(seg_bodies(s)[k], idx), cfk.refs)
    targets = {}
    for s in sm.SIDES:
        for k in ("thigh", "shank", "foot", "upper_arm", "forearm", "hand"):
            b = seg_bodies(s)[k]
            targets[b] = (raw.R(b, idx), raw.P(b, idx))
    for b in (sm.ROOT_BODY, sm.ANCHOR_BODY, sm.HEAD_BODY):
        targets[b] = (raw.R(b, idx), raw.P(b, idx))
    par = sm.evaluate_parity(smod, targets, sem)
    _REPORT["verify_physics_poses"] = {
        "verify_npz": str(path).replace("\\", "/"), "note": note,
        "poses": f"{len(idx)} settled physics poses; semantic angles measured from the physics link rotations",
        "gap_max_mm": float(1e3 * raw.gaps.max()),
        "bodies": _summary(par),
        "meets_1cm_target_p95": {k: v["position"]["p95_mm"] < 10.0 for k, v in par.items()},
    }
    _write_report()
    _assert_bounds(par, feet_p95_mm=30.0, feet_max_mm=50.0)


def test_root_conversion_roundtrip(smod):
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = rng.normal(size=3)
        qv = rng.normal(size=4)
        qv /= np.linalg.norm(qv)
        smod.fk(smod.qpos(p, qv, np.zeros(22)))
        r_site, p_site = smod.site_pose("usd_world")
        pos, quat = smod.root_from_pelvis(p, qv)
        assert np.allclose(pos, p_site, atol=1e-7)
        assert np.allclose(sm.quat_mat(quat), r_site, atol=1e-7)


def _xml_joints(path: Path):
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    wb = root.find("worldbody")
    return root, wb


def test_integer_axes_and_motionlib_variant(smod):
    """Every hinge axis is an integer unit vector (PHC / SONIC motion_lib / ProtoMotions parse axes with int());
    the motionlib variant has exactly one hinge per non-root body and the same FK as the full model."""
    ml = _repo_path(smod.meta["files"]["motionlib_xml"])
    assert ml.is_file()
    for path in (XML, ml):
        root, wb = _xml_joints(path)
        joints = wb.findall(".//joint")
        assert len(joints) == 22 and "type" not in joints[0].attrib
        for j in joints:
            ax = [int(v) for v in j.attrib["axis"].split(" ")]  # raises on non-integers, like Humanoid_Batch
            assert sorted(abs(v) for v in ax) == [0, 0, 1], (path.name, j.attrib["name"], ax)
            assert j.attrib.get("range") is not None
        assert len(list(root.find("actuator"))) == 22
    root, wb = _xml_joints(ml)
    pelvis = wb.find("body")
    for body in pelvis.iter("body"):
        if body is pelvis:
            continue
        assert len(body.findall("joint")) == 1, body.attrib["name"]
    m_ml = mujoco.MjModel.from_xml_path(str(ml))
    d_ml = mujoco.MjData(m_ml)
    rng = np.random.default_rng(7)
    lim = np.array([smod.meta["joints"][n]["range"] for n in SEMANTIC_NAMES])
    qadr_ml = [m_ml.jnt_qposadr[mujoco.mj_name2id(m_ml, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in SEMANTIC_NAMES]
    for _ in range(20):
        q = rng.uniform(lim[:, 0], lim[:, 1])
        smod.fk(smod.qpos(np.array([0.1, -0.2, 0.9]), np.array([0.9, 0.1, 0.2, 0.3]) / np.linalg.norm([0.9, 0.1, 0.2, 0.3]), q))
        d_ml.qpos[:] = 0
        d_ml.qpos[:7] = smod.data.qpos[:7]
        d_ml.qpos[qadr_ml] = q
        mujoco.mj_kinematics(m_ml, d_ml)
        for sn in smod.meta["sites"]:
            i = mujoco.mj_name2id(m_ml, mujoco.mjtObj.mjOBJ_SITE, sn)
            _, p_full = smod.site_pose(sn)
            assert np.allclose(d_ml.site_xpos[i], p_full, atol=1e-9), sn
    assert abs(float(m_ml.body_subtreemass[1]) - float(smod.model.body_subtreemass[1])) < 1e-9


def test_actuator_block_has_no_comments_and_meshes_exist(smod):
    """SONIC's Humanoid_Batch iterates <actuator> children with lxml (comments included) and needs <asset> meshes."""
    import xml.etree.ElementTree as ET

    for path in (XML, _repo_path(smod.meta["files"]["motionlib_xml"])):
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        root = ET.parse(path, parser=parser).getroot()
        act = root.find("actuator")
        assert all(isinstance(c.tag, str) and "name" in c.attrib for c in act), path.name
        meshdir = path.parent / root.find("compiler").attrib["meshdir"]
        for mesh in root.find("asset").findall("mesh"):
            assert (meshdir / mesh.attrib["file"]).is_file(), mesh.attrib["file"]


def test_sonic_humanoid_batch_fk(smod):
    """Real GR00T SONIC Humanoid_Batch (read-only clone, imported by path) vs MuJoCo on random poses."""
    import sys

    sonic = Path("$DROPBEAR_UPSTREAM/GR00T-WholeBodyControl")
    if not sonic.is_dir():
        pytest.skip("SONIC reference clone missing")
    for mod in ("torch", "easydict", "loguru", "lxml", "omegaconf", "hydra", "open3d"):
        pytest.importorskip(mod)
    sys.path.insert(0, str(sonic))
    import torch
    from easydict import EasyDict
    from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

    ml = _repo_path(smod.meta["files"]["motionlib_xml"])
    hb = Humanoid_Batch(EasyDict(asset=EasyDict(assetRoot=str(ml.parent), assetFileName=ml.name), extend_config=[]))
    m = mujoco.MjModel.from_xml_path(str(ml))
    d = mujoco.MjData(m)
    assert hb.num_dof == 22 and hb.num_bodies == m.nbody - 1
    axes = hb.dof_axis.double().numpy()
    rng = np.random.default_rng(11)
    lim = np.array([m.jnt_range[j] for j in range(1, m.njnt)])
    dof = rng.uniform(lim[:, 0], lim[:, 1], (8, 22))
    root = rng.normal(size=(8, 3))
    rv = rng.normal(size=(8, 3)) * 0.5
    pose = np.zeros((1, 8, hb.num_bodies, 3))
    pose[0, :, 0] = rv
    pose[0, :, 1:] = axes[None] * dof[:, :, None]
    res = hb.fk_batch(torch.from_numpy(pose).float(), torch.from_numpy(root[None]).float())
    gt = res.global_translation[0].double().numpy()
    from scipy.spatial.transform import Rotation as R

    for k in range(8):
        d.qpos[:3] = root[k]
        q = R.from_rotvec(rv[k]).as_quat()
        d.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        d.qpos[7:] = dof[k]
        mujoco.mj_kinematics(m, d)
        assert np.abs(d.xpos[1:] - gt[k]).max() < 1e-5
