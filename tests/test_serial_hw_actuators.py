"""hw-map surrogates for the DERIVED serial model + URDF export (item 62, dropbear-backlog notes/62)."""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ to sys.path)
from dropbear_wbc.kinematics import serial_hw_actuators as sh
from dropbear_wbc.robots import hw_motor_specs as hw


@pytest.fixture(scope="module")
def jac_si():
    return sh.standing_jacobian(), sh.semantic_index()


def test_legacy_reproduces_shipped_json(jac_si):
    jac, si = jac_si
    shipped = json.loads(sh.SERIAL_META.read_text())["surrogate_actuators"]["joints"]
    leg = sh.legacy_surrogates(jac, si)
    for jn, s in shipped.items():
        for k in ("effort", "kp", "kd", "armature"):
            assert leg[jn][k] == pytest.approx(s[k], rel=1e-6), (jn, k)


# v4_arms_2026_10_20 is a sim study with X8-120 shoulders (120 N*m peak), deliberately stronger than the legacy 40 N*m arms
@pytest.mark.parametrize("map_name", sorted(m for m in hw.JOINT_MOTOR_MAPS if m != "v4_arms_2026_10_20"))
def test_hw_surrogate_never_stronger_than_legacy_and_heavier_geared(jac_si, map_name):
    jac, si = jac_si
    leg, sur = sh.legacy_surrogates(jac, si), sh.hw_surrogates(jac, si, map_name)
    for jn in si:
        assert 0 < sur[jn]["effort_rated"] < sur[jn]["effort_peak"] <= leg[jn]["effort"] + 1e-6, jn
        assert sur[jn]["armature"] > leg[jn]["armature"], jn
        assert sur[jn]["velocity"] > 0 and sur[jn]["motors"], jn


def test_hw_v1_legs_are_much_weaker_than_shipped(jac_si):
    """Pins the notes/62 finding: shipped serial legs are 2-5x the user's motor map at the stand."""
    jac, si = jac_si
    leg, sur = sh.legacy_surrogates(jac, si), sh.hw_surrogates(jac, si, "user_2026_09_25")
    ratio = {jn: leg[jn]["effort"] / sur[jn]["effort_peak"] for jn in si}
    for s in ("left", "right"):
        assert ratio[f"{s}_knee"] > 4.5 and ratio[f"{s}_hip_roll"] > 4.5 and ratio[f"{s}_hip_pitch"] > 2.0
        assert sur[f"{s}_hip_pitch"]["armature"] / leg[f"{s}_hip_pitch"]["armature"] > 50


def test_single_motor_joint_matches_motor_model(jac_si):
    jac, si = jac_si
    sur = sh.hw_surrogates(jac, si, "user_2026_09_25")
    s = sur["left_shoulder_pitch"]  # one X10 1:7 (LH_yaw), |J| ~ 1
    assert list(s["motors"].values()) == ["RMD-X10-V3-1:7"]
    assert s["effort_peak"] == pytest.approx(40.0, rel=1e-3)
    assert s["armature"] == pytest.approx(hw.MOTOR_MODELS["RMD-X10-V3-1:7"].armature, rel=1e-3)


def test_hw_mjcf_and_urdf_roundtrip(tmp_path, jac_si):
    mujoco = pytest.importorskip("mujoco")
    jac, si = jac_si
    sur = sh.hw_surrogates(jac, si, "user_2026_09_25")
    src = mujoco.MjModel.from_xml_path(str(sh.SERIAL_XML))
    shutil.copytree(sh.SERIAL_XML.parent / "meshes", tmp_path / "meshes")
    (tmp_path / "hw.xml").write_text(sh.hw_mjcf_text(sh.SERIAL_XML.read_text(), sur, "user_2026_09_25"))
    mh = mujoco.MjModel.from_xml_path(str(tmp_path / "hw.xml"))
    assert mh.body_mass.sum() == pytest.approx(src.body_mass.sum())
    for jn, s in sur.items():
        j = mujoco.mj_name2id(mh, mujoco.mjtObj.mjOBJ_JOINT, jn)
        a = mujoco.mj_name2id(mh, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
        assert mh.dof_armature[mh.jnt_dofadr[j]] == pytest.approx(s["armature"], abs=1e-5)
        assert mh.actuator_forcerange[a, 1] == pytest.approx(s["effort_peak"], abs=0.01)
        assert mh.actuator_gainprm[a, 0] == pytest.approx(s["kp"], abs=1e-3)
    txt = sh.urdf_text(src, sur, "user_2026_09_25").replace(
        '<robot name="dropbear_serial_user_2026_09_25">',
        '<robot name="rt"><mujoco><compiler meshdir="meshes" discardvisual="false" fusestatic="false" strippath="true"/></mujoco>')
    (tmp_path / "rt.urdf").write_text(txt)
    mu = mujoco.MjModel.from_xml_path(str(tmp_path / "rt.urdf"))
    assert mu.body_mass.sum() == pytest.approx(src.body_mass.sum(), rel=1e-9)
    ds, du = mujoco.MjData(src), mujoco.MjData(mu)
    rng = np.random.default_rng(0)
    hinges = [j for j in range(src.njnt) if src.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    for _ in range(50):
        for j in hinges:
            v = rng.uniform(*src.jnt_range[j])
            jn = mujoco.mj_id2name(src, mujoco.mjtObj.mjOBJ_JOINT, j)
            ds.qpos[src.jnt_qposadr[j]] = v
            du.qpos[mu.jnt_qposadr[mujoco.mj_name2id(mu, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = v
        mujoco.mj_kinematics(src, ds)
        mujoco.mj_kinematics(mu, du)
        ps, pu = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis") for m in (src, mu))
        rs, ru = ds.xmat[ps].reshape(3, 3), du.xmat[pu].reshape(3, 3)
        for b in range(1, src.nbody):
            bu = mujoco.mj_name2id(mu, mujoco.mjtObj.mjOBJ_BODY, mujoco.mj_id2name(src, mujoco.mjtObj.mjOBJ_BODY, b))
            assert bu >= 0
            np.testing.assert_allclose(rs.T @ (ds.xpos[b] - ds.xpos[ps]), ru.T @ (du.xpos[bu] - du.xpos[pu]), atol=1e-7)
            np.testing.assert_allclose(rs.T @ (ds.xipos[b] - ds.xpos[ps]), ru.T @ (du.xipos[bu] - du.xpos[pu]), atol=1e-7)


COMMITTED_HW_JSON = sorted(sh.SERIAL_XML.parent.glob("dropbear_serial_hw_*.json"))


@pytest.mark.parametrize("path", COMMITTED_HW_JSON, ids=lambda p: p.stem)
def test_committed_surrogates_are_up_to_date(jac_si, path):
    """Every committed hw surrogate (JSON + MJCF) matches what tools/export_serial_hw.py computes from hw_motor_specs
    today, field by field; a motor-map or motor-model edit without a re-export fails here."""
    jac, si = jac_si
    meta = json.loads(path.read_text(encoding="utf-8"))
    profile, map_name = path.stem.removeprefix("dropbear_serial_"), meta["map"]
    assert meta["profile"] == profile
    assert hw.HW_PROFILE_MAPS.get(profile, map_name) == map_name, (profile, map_name)
    assert meta["source_xml_sha256"] == hashlib.sha256(sh.SERIAL_XML.read_bytes()).hexdigest()
    fresh = sh.hw_surrogates(jac, si, map_name)
    assert set(meta["joints"]) == set(fresh)
    for j, rec in fresh.items():
        assert set(meta["joints"][j]) == set(rec), j
        for k, v in rec.items():
            if isinstance(v, float):
                assert meta["joints"][j][k] == pytest.approx(v, rel=1e-9, abs=1e-12), (j, k)
            else:
                assert meta["joints"][j][k] == v, (j, k)
    xml = path.with_suffix(".xml")
    assert xml.read_text(encoding="utf-8") == sh.hw_mjcf_text(sh.SERIAL_XML.read_text(), fresh, map_name)


def test_current_builds_have_committed_surrogates():
    """Build A / Build B and their power-consistent CEM-60 replays (notes/213) all ship a surrogate."""
    have = {p.stem.removeprefix("dropbear_serial_") for p in COMMITTED_HW_JSON}
    assert {"hw_cem6", "hw_v4", "hw_cem6_p330", "hw_v2ie_p330"} <= have


def test_p330_differs_only_in_the_torque_speed_line(jac_si):
    """P330 halves the CEM-60 saturation (240 -> 120 N*m): same peak/velocity/armature, so identical MJCF, but the
    full 75 N*m knee torque ends at 30 rpm motor-equivalent instead of 55 rpm."""
    jac, si = jac_si
    a, b = sh.hw_surrogates(jac, si, "cem6_2026_10_16"), sh.hw_surrogates(jac, si, "cem6_p330_2026_10_20")
    for j in a:
        for k in ("effort_peak", "effort_rated", "velocity", "armature", "kp", "kd"):
            assert a[j][k] == b[j][k], (j, k)
    knee = "left_knee"
    assert b[knee]["effort_saturation"] == pytest.approx(0.5 * a[knee]["effort_saturation"], rel=1e-9)
    w = 0.5 * b[knee]["velocity"]  # half no-load speed: the P330 line is below peak, the old one is not
    assert sh.envelope_effort(a[knee], w) == pytest.approx(a[knee]["effort_peak"])
    assert sh.envelope_effort(b[knee], w) < 0.85 * a[knee]["effort_peak"]
    xml = sh.SERIAL_XML.read_text()
    assert sh.hw_mjcf_text(xml, a, "m") == sh.hw_mjcf_text(xml, b, "m")


def test_envelope_effort_single_motor_matches_motor_model(jac_si):
    jac, si = jac_si
    rec = sh.hw_surrogates(jac, si, "cem6_p330_2026_10_20")["left_knee"]
    m = hw.MOTOR_MODELS["EPS-CEM-60-V2-P330"]
    g = m.no_load_speed / rec["velocity"]  # |J| of the single knee motor (velocity = no-load / |J|)
    for frac in (0.0, 0.3, 0.6, 0.9, 1.0, 1.2):
        w = frac * rec["velocity"]
        assert sh.envelope_effort(rec, w) == pytest.approx(g * hw.envelope_torque(m, g * w), rel=1e-9, abs=1e-9)

def _load_export_tool():
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_serial_hw", sh.SERIAL_XML.parents[2] / "tools" / "export_serial_hw.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lf_bytes(data: bytes) -> bytes:
    """Normalize CRLF/CR to LF so a Windows autocrlf (or editor) working tree still matches a fresh LF export."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


@pytest.mark.parametrize("path", COMMITTED_HW_JSON, ids=lambda p: p.stem)
def test_committed_surrogate_files_equal_a_fresh_export(tmp_path, path):
    """Byte-for-byte staleness gate over all three committed artefacts (JSON incl. legacy/method/sha, MJCF, URDF):
    re-run tools/export_serial_hw.py for the file's map into tmp and compare. Catches edits the field-level test
    above does not see (URDF inertials/limits, legacy block, schema/method text, hand edits).

    Newlines are normalized (CRLF/CR -> LF) on both sides before equality so a core.autocrlf=true checkout does
    not fail spuriously; a real content change still fails. data/robot/*.{json,xml,urdf} are also eol=lf in
    .gitattributes so clean checkouts stay LF."""
    tool = _load_export_tool()
    meta = json.loads(path.read_text(encoding="utf-8"))
    profile = path.stem.removeprefix("dropbear_serial_")
    assert tool.PROFILE_FOR_MAP[meta["map"]] == profile, (profile, meta["map"])
    assert tool.main(["--map", meta["map"], "--out", str(tmp_path)]) == 0
    for suffix in (".json", ".xml", ".urdf"):
        committed, fresh = path.with_suffix(suffix), tmp_path / f"{path.stem}{suffix}"
        assert committed.exists(), f"{committed.name} missing: run tools/export_serial_hw.py --map {meta['map']}"
        assert _lf_bytes(committed.read_bytes()) == _lf_bytes(fresh.read_bytes()), (
            f"{committed.name} is stale: run tools/export_serial_hw.py --map {meta['map']}")
