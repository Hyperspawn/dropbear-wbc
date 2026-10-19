"""hw-map surrogates for the DERIVED serial model + URDF export (item 62, dropbear-backlog notes/62)."""
from __future__ import annotations

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


@pytest.mark.parametrize("map_name", sorted(hw.JOINT_MOTOR_MAPS))
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


@pytest.mark.parametrize("profile,map_name", [("hw_cem6", "cem6_2026_10_16"), ("hw_v4", "v4_2026_10_16")])
def test_current_build_surrogates_are_up_to_date(jac_si, profile, map_name):
    """The committed hw_cem6 / hw_v4 surrogate JSON matches what the exporter computes from hw_motor_specs today."""
    jac, si = jac_si
    path = sh.SERIAL_XML.parent / f"dropbear_serial_{profile}.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["map"] == map_name
    fresh = sh.hw_surrogates(jac, si, map_name)
    for j, rec in fresh.items():
        assert abs(meta["joints"][j]["effort_peak"] - rec["effort_peak"]) < 1e-6, j
