"""Hand-written sign / role constants vs their source of truth (item 342). CPU, Isaac-free.

Several modules carry per-joint signs or motor-role name lists that are only correct for the current semantic
calibration (``data/calibration/dropbear_semantic_calibration.json``) and the USD joint axes it was fitted on:

* the left/right semantic mirror convention (roll / yaw flip, pitch / knee / elbow keep) in ``calib_build.MIRROR_SIGN``,
  ``teleop.arm_ik.MIRROR_SIGN`` and the name rule of ``tasks.locomotion.symmetry.semantic_mirror`` (which feeds the
  calibration-derived motor mirror of PPO symmetry augmentation and AMP);
* the root-frame reflection signs ``symmetry.BASE_TERM_SIGNS`` and ``amp.FeatureMirror.BASE_SIGNS``;
* the arm-chain joint axes ``arm_ik._AXIS_IDS`` (teleop IK and gravity compensation);
* motor-role lists: ``arm_ik.ARM_MOTOR_NAMES``, ``velocity_env_cfg.HIP_ROLL_YAW_MOTORS``, the shoulder-roll motors
  of ``flat_env_cfg.enable_arm_swing`` (abduction term), ``dropbear_names.KNEE_MOTORS`` (+crank = flexion in the
  knee-flexion rewards), ``calib_fit.seg_motors``.

A re-measured calibration (tools/calibrate_semantics.py) or a re-authored USD that changes any of them silently
mirrors augmentation samples with a wrong sign, mirrors teleop arms wrongly, or points a reward at the wrong motor.
The arm-swing reward constants are covered by tests/test_arm_swing_sign.py.

Physical sign source: the serial MJCF (``dropbear_serial_hw_v1.xml``, generated from the calibration; joints are the
semantic DOFs, qpos 0 = semantic zero). A left DOF at +a and the right partner at sign*a must give mirror-image link
poses about the sagittal plane; the correct sign does so to ~2.6 mm, the wrong one is >= 59 mm off.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "data/calibration/dropbear_semantic_calibration.json"
SERIAL_XML = ROOT / "data/robot/dropbear_serial_hw_v1.xml"
VEL_CFG = ROOT / "source/dropbear_wbc/tasks/locomotion/velocity_env_cfg.py"
FLAT_CFG = ROOT / "source/dropbear_wbc/tasks/locomotion/config/dropbear/flat_env_cfg.py"

BASES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
         "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
ARM_BASES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
MIRROR_OK_M = 0.006  # correct sign: 2.6 mm (zero-pose asymmetry of the fitted model)
MIRROR_BAD_M = 0.03  # wrong sign: >= 59 mm

pytestmark = pytest.mark.skipif(not CALIB.exists(), reason="semantic calibration not generated")


def _calib() -> dict:
    return json.loads(CALIB.read_text())


def _motor(calib: dict, sem: str) -> str:
    motors = calib["dofs"][sem]["motors"]
    assert len(motors) == 1, (sem, motors)
    return motors[0]


# ---------------------------------------------------------------------------------------------------- physical signs
_FK_CACHE: dict = {}


def _fk_mirror_error(q: dict[str, float]) -> float:
    """Max distance between each left link's body-frame probe points, reflected y -> 2 y_mid - y, and the right
    partner's mirrored probes (local y negated: both sides' link frames share the semantic orientation)."""
    mujoco = pytest.importorskip("mujoco")
    if "m" not in _FK_CACHE:
        _FK_CACHE["m"] = mujoco.MjModel.from_xml_path(str(SERIAL_XML))
    m = _FK_CACHE["m"]
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    for n, v in q.items():
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    mujoco.mj_forward(m, d)
    probes = np.array([[0.10, 0.07, 0.05], [-0.05, 0.03, -0.10], [0.02, -0.06, 0.08]])
    flip = np.array([1.0, -1.0, 1.0])
    pairs = []
    for i in range(m.nbody):
        b = m.body(i).name
        if b.startswith("left_"):
            r = b.replace("left_", "right_")
            pl = d.body(b).xpos + (d.body(b).xmat.reshape(3, 3) @ probes.T).T
            pr = d.body(r).xpos + (d.body(r).xmat.reshape(3, 3) @ (probes * flip).T).T
            pairs.append((pl, pr))
    assert len(pairs) >= 10
    y_mid = float(np.mean([0.5 * (pl[:, 1] + pr[:, 1]).mean() for pl, pr in pairs]))
    if "y_mid" not in _FK_CACHE and not q:
        _FK_CACHE["y_mid"] = y_mid
    y_mid = _FK_CACHE.get("y_mid", y_mid)
    return max(float(np.abs(pl * flip + [0.0, 2.0 * y_mid, 0.0] - pr).max()) for pl, pr in pairs)


def _physical_signs() -> dict[str, int]:
    """Per semantic base DOF: the partner sign that makes left/right FK mirror images (serial MJCF)."""
    if "signs" in _FK_CACHE:
        return _FK_CACHE["signs"]
    mujoco = pytest.importorskip("mujoco")
    zero = _fk_mirror_error({})
    assert zero < MIRROR_OK_M, zero
    m = mujoco.MjModel.from_xml_path(str(SERIAL_XML))
    out = {}
    for base in BASES:
        hi = float(m.jnt_range[m.joint(f"left_{base}").id][1])
        a = min(0.3, 0.8 * hi)
        assert a > 0.1, (base, hi)
        err = {s: _fk_mirror_error({f"left_{base}": a, f"right_{base}": s * a}) for s in (1, -1)}
        good = min(err, key=err.get)
        assert err[good] < MIRROR_OK_M and err[-good] > MIRROR_BAD_M, (base, err)
        out[base] = good
    _FK_CACHE["signs"] = out
    return out


def test_serial_model_has_the_g1_mirror_convention():
    """Pitch / knee / elbow keep sign, roll / yaw flip (the convention every table below assumes)."""
    s = _physical_signs()
    assert s == {b: (-1 if ("roll" in b or "yaw" in b) else 1) for b in BASES}, s


def test_calib_build_mirror_sign_matches_serial_model():
    from dropbear_wbc.kinematics.calib_build import MIRROR_SIGN

    assert set(MIRROR_SIGN) == set(BASES)
    phys = _physical_signs()
    for base, sgn in MIRROR_SIGN.items():
        assert sgn == phys[base], (base, sgn, phys[base])


def test_teleop_mirror_sign_matches_serial_model():
    from dropbear_wbc.teleop.arm_ik import ARM_JOINTS, MIRROR_SIGN

    assert tuple(ARM_JOINTS) == ARM_BASES
    phys = _physical_signs()
    assert MIRROR_SIGN.tolist() == [float(phys[j]) for j in ARM_JOINTS]


def test_symmetry_semantic_mirror_matches_serial_model():
    """``semantic_mirror`` (the root of the PPO / AMP motor mirror) uses the physical sign for every DOF."""
    from dropbear_wbc.tasks.locomotion.symmetry import semantic_mirror

    calib = _calib()
    sem = list(calib["semantic_names"])
    s = semantic_mirror(CALIB)
    phys = _physical_signs()
    for i, n in enumerate(sem):
        side, base = n.split("_", 1)
        partner = sem.index(f"{'right' if side == 'left' else 'left'}_{base}")
        assert s[i, partner] == phys[base], (n, s[i, partner], phys[base])
        assert np.count_nonzero(s[i]) == 1, n


def test_mirror_tables_agree_on_random_whole_body_poses():
    """All 11 DOFs at once (random, inside both sides' joint ranges): the table mirror stays within the model asymmetry."""
    mujoco = pytest.importorskip("mujoco")
    from dropbear_wbc.kinematics.calib_build import MIRROR_SIGN

    m = mujoco.MjModel.from_xml_path(str(SERIAL_XML))
    rng = np.random.default_rng(342)
    _fk_mirror_error({})
    for _ in range(10):
        q = {}
        for base, sgn in MIRROR_SIGN.items():
            lo_l, hi_l = m.jnt_range[m.joint(f"left_{base}").id]
            lo_r, hi_r = sorted(sgn * m.jnt_range[m.joint(f"right_{base}").id])
            lo, hi = max(lo_l, lo_r), min(hi_l, hi_r)
            a = float(rng.uniform(lo + 0.3 * (hi - lo), hi - 0.3 * (hi - lo)))
            q[f"left_{base}"], q[f"right_{base}"] = a, sgn * a
        assert _fk_mirror_error(q) < 0.01, q


# ---------------------------------------------------------------------------------------------------- root-frame signs
def _reflect_signs() -> dict[str, tuple[float, ...]]:
    """x-z plane reflection R = diag(1, -1, 1): vectors -> R v, pseudo-vectors -> det(R) R v."""
    r = np.diag([1.0, -1.0, 1.0])
    vec = tuple(np.diag(r).tolist())
    pseudo = tuple((np.linalg.det(r) * np.diag(r)).tolist())
    return {"base_lin_vel": vec, "base_ang_vel": pseudo, "projected_gravity": vec,
            "velocity_commands": (vec[0], vec[1], pseudo[2])}  # (vx, vy, wz)


def test_base_term_signs_are_the_reflection():
    from dropbear_wbc.tasks.locomotion.symmetry import BASE_TERM_SIGNS

    assert {k: tuple(v) for k, v in BASE_TERM_SIGNS.items()} == _reflect_signs()


def test_amp_base_signs_are_the_reflection():
    pytest.importorskip("torch")
    from dropbear_wbc.tasks.locomotion.amp import FeatureMirror

    ref = _reflect_signs()
    # amp_features: height (scalar), projected gravity, lin vel, ang vel (root frame)
    expected = (1.0,) + ref["projected_gravity"] + ref["base_lin_vel"] + ref["base_ang_vel"]
    assert tuple(FeatureMirror.BASE_SIGNS) == expected


# ---------------------------------------------------------------------------------------------------- arm chain axes
def test_arm_ik_axis_ids_match_serial_model():
    mujoco = pytest.importorskip("mujoco")
    from dropbear_wbc.teleop.arm_ik import _AXIS_IDS, ARM_JOINTS

    m = mujoco.MjModel.from_xml_path(str(SERIAL_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for side in ("left", "right"):
        for j, a in zip(ARM_JOINTS, _AXIS_IDS):
            ax = d.xaxis[m.joint(f"{side}_{j}").id]
            assert int(np.argmax(np.abs(ax))) == a and abs(ax[a]) > 0.99, (side, j, ax)
            assert ax[a] > 0, (side, j, ax)  # _rot(axis, +q) is a positive rotation about +axis


# ---------------------------------------------------------------------------------------------------- motor roles
def test_teleop_arm_motor_names_match_calibration():
    from dropbear_wbc.teleop.arm_ik import ARM_JOINTS, ARM_MOTOR_NAMES

    c = _calib()
    for side in ("left", "right"):
        assert ARM_MOTOR_NAMES[side] == tuple(_motor(c, f"{side}_{j}") for j in ARM_JOINTS), side


def test_calib_fit_segment_motors_match_calibration():
    from dropbear_wbc.kinematics.calib_fit import seg_motors

    c = _calib()
    for side in ("left", "right"):
        seg = seg_motors(side)
        for base in BASES:
            if base in ("ankle_pitch", "ankle_roll"):
                assert c["dofs"][f"{side}_{base}"]["motors"] == [seg["calf_a"], seg["calf_b"]], (side, base)
            else:
                assert _motor(c, f"{side}_{base}") == seg[base], (side, base)


def _module_list(path: Path, name: str) -> list:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(tgt, ast.Name) and tgt.id == name:
                return list(ast.literal_eval(node.value))
    raise AssertionError(f"{name} not found in {path}")


def test_hip_roll_yaw_motors_match_calibration():
    c = _calib()
    expected = {_motor(c, f"{s}_{b}") for s in ("left", "right") for b in ("hip_roll", "hip_yaw")}
    got = _module_list(VEL_CFG, "HIP_ROLL_YAW_MOTORS")
    assert len(got) == 4 and set(got) == expected, got


def test_arm_abduction_term_uses_the_shoulder_roll_motors():
    tree = ast.parse(FLAT_CFG.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "enable_arm_swing")
    roll = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "roll_cfg")
    names = ast.literal_eval(next(k.value for k in roll.keywords if k.arg == "joint_names"))
    c = _calib()
    assert names == [_motor(c, "left_shoulder_roll"), _motor(c, "right_shoulder_roll")], names


def test_knee_motors_flex_positive_in_left_right_order():
    """knee_flexion_in_swing / _in_stance read +crank (from the stand) as flexion, feet in (L, R) order."""
    from dropbear_wbc.robots.dropbear_names import KNEE_MOTORS

    c = _calib()
    assert tuple(KNEE_MOTORS) == (_motor(c, "left_knee"), _motor(c, "right_knee"))
    for side in ("left", "right"):
        k = c["dofs"][f"{side}_knee"]
        assert k["type"] == "lut1d", (side, k["type"])
        assert np.all(np.diff(k["semantic_values"]) > 0) and np.all(np.diff(k["motor_grid"]) > 0), side


# ---------------------------------------------------------------------------------------------------- default poses
def _semantic_mirror_error_deg(q_motor: np.ndarray) -> dict[str, float]:
    from dropbear_wbc.kinematics.semantic import SemanticMap
    from dropbear_wbc.tasks.locomotion.symmetry import semantic_mirror

    c = _calib()
    s = np.asarray(SemanticMap(c).motor_to_semantic(np.asarray(q_motor, float), clip=False))
    e = semantic_mirror(CALIB) @ s - s
    return {n: float(np.degrees(abs(v))) for n, v in zip(c["semantic_names"], e)}


def test_calibration_standing_pose_is_mirror_symmetric():
    err = _semantic_mirror_error_deg(_calib()["standing_motor_pos"])
    assert max(err.values()) < 0.5, {k: v for k, v in err.items() if v >= 0.5}


def test_legacy_default_pose_asymmetry_is_only_the_known_hip_roll():
    """``LEGACY_DEFAULT_MOTOR_POS`` (= sdk ``motors.DEFAULT_POS``, the deploy ``"legacy"`` default / hold pose) puts BOTH
    hip-roll motors at -0.1 rad. The two motors map to semantic hip roll with the same sign, so the legs roll the
    same way (left -5.9 deg, right -5.6 deg; mirror error 11.5 deg) instead of mirror-symmetrically. Pinned here
    (item 370); any other asymmetry > 3 deg in the legacy pose fails."""
    from dropbear_wbc.robots.dropbear_names import LEGACY_DEFAULT_MOTOR_POS, MOTOR_NAMES

    q = np.array([LEGACY_DEFAULT_MOTOR_POS[n] for n in MOTOR_NAMES])
    err = _semantic_mirror_error_deg(q)
    bad = {k for k, v in err.items() if v > 3.0}
    assert bad <= {"left_hip_roll", "right_hip_roll"}, {k: err[k] for k in bad}
