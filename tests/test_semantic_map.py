"""CPU unit tests of ``dropbear_wbc.kinematics.semantic.SemanticMap``.

Part 1 uses a synthetic calibration (known analytic maps) to test the machinery: linear / lut1d / lut2d
forward and inverse, clipping and saturation reports, batching.
Part 2 checks the real measured calibration ``data/calibration/dropbear_semantic_calibration.json``
(skipped if it has not been generated): round trip, monotonic tables, zero/standing consistency, and
left/right mirror symmetry (asymmetries beyond tolerance must be listed in the calibration findings).

Run: python -m pytest tests/test_semantic_map.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.kinematics.semantic import MOTOR_NAMES, SCHEMA, SEMANTIC_NAMES, SemanticMap

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/calibration/dropbear_semantic_calibration.json"
DEG = np.pi / 180


# ----------------------------------------------------------------------------------------------------
def _synthetic() -> dict:
    lim = np.tile([-1.0, 1.0], (22, 1))
    dofs = {}
    ank = {}
    for side, pre, pg, arm in (("left", "LL", "PG_left_leg", "LH"), ("right", "RL", "PG_right_leg", "RH")):
        lin = {"hip_roll": f"{pg}_pitch", "hip_yaw": f"{pg}_roll", "hip_pitch": f"{pre}_hip_joint",
               "shoulder_pitch": f"{arm}_yaw", "shoulder_roll": f"{arm}_pitch", "shoulder_yaw": f"{arm}_roll",
               "wrist_roll": f"{arm}_wrist_roll"}
        for k, (role, motor) in enumerate(lin.items()):
            scale, off = (-1.0 if k % 2 else 0.98), 0.05 * k
            vr = sorted([scale * (-1 - off), scale * (1 - off)])
            dofs[f"{side}_{role}"] = {"type": "linear", "motors": [motor], "scale": scale, "offset": off,
                                      "valid_range": vr}
        mg = np.linspace(0, 0.5, 26)
        dofs[f"{side}_knee"] = {"type": "lut1d", "motors": [f"{pre}_knee_actuator_joint"], "motor_grid": mg.tolist(),
                                "semantic_values": (2.0 * mg + 0.8 * mg ** 2).tolist(), "valid_range": [0, 1.2]}
        dofs[f"{side}_elbow"] = {"type": "lut1d", "motors": [f"{arm}_elbow_joint"], "motor_grid": mg.tolist(),
                                 "semantic_values": (np.pi / 2 - 1.5 * mg).tolist(), "valid_range": [0.82, 1.57]}
        # ankle: pitch = -(a+b)/2 + 0.1 a^2, roll = s*(a-b)/2 on a 21x21 grid; one infeasible corner
        ag = bg = np.linspace(-0.8, 0.8, 21)
        A, B = np.meshgrid(ag, bg, indexing="ij")
        s = 1.0 if side == "left" else -1.0
        pitch = -(A + B) / 2 + 0.1 * A ** 2
        roll = s * (A - B) / 2
        valid = ~((A > 0.6) & (B < -0.6))
        from dropbear_wbc.kinematics.calib_pair import build_pair

        pair = build_pair(side, (f"{pre}_Revolute67", f"{pre}_Revolute81"), ag, bg, pitch, roll, valid,
                          (-1.0, 1.0), (-1.0, 1.0), [])
        ank[side] = json.loads(json.dumps(pair, default=float).replace("NaN", "null"))
        for nm in ("pitch", "roll"):
            dofs[f"{side}_ankle_{nm}"] = {"type": "lut2d", "motors": pair["motors"], "pair": side,
                                          "valid_range": pair[f"{nm}_range"]}
    return {"schema": SCHEMA, "semantic_names": list(SEMANTIC_NAMES), "motor_names": list(MOTOR_NAMES),
            "motor_limits_rad": lim.tolist(), "dofs": dofs, "ankle_pairs": ank,
            "semantic_zero_motor_pos": [0.0] * 22, "standing_motor_pos": [0.0] * 22}


@pytest.fixture(scope="module")
def synth() -> SemanticMap:
    return SemanticMap(_synthetic())


def test_names_exported():
    assert len(SEMANTIC_NAMES) == 22 and len(MOTOR_NAMES) == 22
    from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES as ROBOT_MOTORS

    assert tuple(ROBOT_MOTORS) == MOTOR_NAMES


def test_synthetic_roundtrip_inside_range(synth):
    rng = np.random.default_rng(0)
    lo, hi = synth.semantic_limits[:, 0], synth.semantic_limits[:, 1]
    q = lo + (hi - lo) * (0.1 + 0.8 * rng.random((500, 22)))
    m, rep = synth.semantic_to_motor(q, return_report=True)
    back = synth.motor_to_semantic(m)
    # ankle pairs: requests inside the box but outside the feasible set are projected; compare 'used'
    err = np.abs(back - rep.used)
    lin_lut = [i for i, n in enumerate(SEMANTIC_NAMES) if "ankle" not in n]
    ankle = [i for i, n in enumerate(SEMANTIC_NAMES) if "ankle" in n]
    assert err[:, lin_lut].max() < 1e-9, err[:, lin_lut].max()
    # projected ankle requests land on inverse-grid nodes solved to < 0.05 deg
    assert err[:, ankle].max() < 0.05 * DEG, err[:, ankle].max()
    assert not rep.clipped[:, lin_lut].any()


def test_synthetic_clipping_reports(synth):
    q = np.zeros(22)
    q[SEMANTIC_NAMES.index("left_knee")] = 5.0
    q[SEMANTIC_NAMES.index("right_elbow")] = -1.0
    m, rep = synth.semantic_to_motor(q, return_report=True)
    s = rep.summary()
    assert "left_knee" in s and "right_elbow" in s
    assert m[MOTOR_NAMES.index("LL_knee_actuator_joint")] == pytest.approx(0.5)
    assert m[MOTOR_NAMES.index("RH_elbow_joint")] == pytest.approx(0.5)
    assert synth.last_report is rep
    # motor side clipping
    mm = np.zeros(22)
    mm[0] = 3.0
    _, rep2 = synth.motor_to_semantic(mm, return_report=True)
    assert rep2.clipped[0] and rep2.used[0] == pytest.approx(1.0)


def test_synthetic_pair_projection_and_accuracy(synth):
    m = np.zeros((200, 22))
    rng = np.random.default_rng(3)
    ia, ib = MOTOR_NAMES.index("LL_Revolute67"), MOTOR_NAMES.index("LL_Revolute81")
    m[:, ia] = rng.uniform(-0.75, 0.55, 200)
    m[:, ib] = rng.uniform(-0.55, 0.75, 200)
    q = synth.motor_to_semantic(m)
    m2 = synth.semantic_to_motor(q)
    np.testing.assert_allclose(m2[:, [ia, ib]], m[:, [ia, ib]], atol=2e-3)
    # an infeasible request (inside the box, in the cut corner) is projected and reported
    q_bad = np.zeros(22)
    jp, jr = SEMANTIC_NAMES.index("left_ankle_pitch"), SEMANTIC_NAMES.index("left_ankle_roll")
    q_bad[jp], q_bad[jr] = -0.07, 0.78  # a~0.85, b~-0.7 region (invalid)
    _, rep = synth.semantic_to_motor(q_bad, return_report=True)
    assert rep.clipped[jr] or rep.clipped[jp]


def test_batch_shapes(synth):
    q = np.zeros((3, 4, 22))
    m = synth.semantic_to_motor(q)
    assert m.shape == (3, 4, 22)
    assert synth.motor_to_semantic(m).shape == (3, 4, 22)
    with pytest.raises(ValueError):
        synth.semantic_to_motor(np.zeros(21))


def test_lut1d_rejects_non_monotonic():
    cal = _synthetic()
    cal["dofs"]["left_knee"]["semantic_values"][5] = 10.0
    with pytest.raises(ValueError):
        SemanticMap(cal)


# ----------------------------------------------------------------------------------------------------
real_only = pytest.mark.skipif(not REAL.exists(), reason="measured calibration not generated yet")


@pytest.fixture(scope="module")
def real() -> SemanticMap:
    return SemanticMap.load(REAL)


@real_only
def test_real_schema_and_provenance(real):
    c = real.calib
    assert c["schema"] == SCHEMA and not c.get("MOCK", False)
    p = c["provenance"]
    assert p["usd_sha256"] == "45586414b065cd982d487cbd868fe982108b3b8ccec64d3dfcf629652ed8db0f"
    assert p["script"] and p["raw_sweep"]
    for k in ("semantic_zero_motor_pos", "standing_motor_pos", "key_bodies", "segment_lengths", "rest_transforms"):
        assert k in c


@real_only
def test_real_roundtrip(real):
    rng = np.random.default_rng(0)
    lo, hi = real.semantic_limits[:, 0], real.semantic_limits[:, 1]
    q = lo + (hi - lo) * (0.05 + 0.9 * rng.random((2000, 22)))
    m, rep = real.semantic_to_motor(q, return_report=True)
    back = real.motor_to_semantic(m)
    err = np.abs(back - rep.used)
    non_ankle = [i for i, n in enumerate(SEMANTIC_NAMES) if "ankle" not in n]
    assert err[:, non_ankle].max() < 1e-6
    ankle = [i for i, n in enumerate(SEMANTIC_NAMES) if "ankle" in n]
    assert err[:, ankle].max() < 0.2 * DEG, np.degrees(err[:, ankle].max())
    # motor -> semantic -> motor inside the motor limits (non-ankle: exact). lut1d motors are sampled inside
    # their measured motor grid (the sweep stops 0.25 deg short of each authored limit; beyond it the map clips).
    lim = real.motor_limits.copy()
    for d in real.calib["dofs"].values():
        if d["type"] == "lut1d":
            k = MOTOR_NAMES.index(d["motors"][0])
            lim[k] = [min(d["motor_grid"]), max(d["motor_grid"])]
    mm = lim[:, 0] + (lim[:, 1] - lim[:, 0]) * rng.random((500, 22))
    mm2 = real.semantic_to_motor(real.motor_to_semantic(mm))
    motor_non_ankle = [i for i, n in enumerate(MOTOR_NAMES) if "Revolute67" not in n and "Revolute81" not in n]
    np.testing.assert_allclose(mm2[:, motor_non_ankle], mm[:, motor_non_ankle], atol=1e-6)


@real_only
def test_real_tables_monotonic(real):
    for name, d in real.calib["dofs"].items():
        if d["type"] == "lut1d":
            s = np.diff(d["semantic_values"])
            assert np.all(s > 0) or np.all(s < 0), name
    findings = " ".join(real.calib.get("findings", []))
    for side in ("left", "right"):
        knee = real.calib["dofs"][f"{side}_knee"]
        if knee["type"] == "fixed":  # a locked mechanism must be reported loudly
            assert f"{side}_knee: LOCKED" in findings
            continue
        assert np.all(np.diff(knee["semantic_values"]) > 0), "knee flexion must increase with the crank"


@real_only
def test_real_zero_and_standing(real):
    z = real.motor_to_semantic(real.semantic_zero_motor_pos)
    legs = [i for i, n in enumerate(SEMANTIC_NAMES) if any(k in n for k in ("hip", "knee", "ankle"))]
    assert np.abs(z[legs]).max() < 0.5 * DEG, dict(zip(np.array(SEMANTIC_NAMES)[legs], np.degrees(z[legs])))
    shoulders = [i for i, n in enumerate(SEMANTIC_NAMES) if "shoulder" in n or "wrist" in n]
    assert np.abs(z[shoulders]).max() < 0.5 * DEG
    s = real.motor_to_semantic(real.standing_motor_pos)
    np.testing.assert_allclose(s, real.calib["standing_semantic_pos"], atol=1e-9)
    lim = real.motor_limits
    assert np.all(real.standing_motor_pos >= lim[:, 0] - 1e-9) and np.all(real.standing_motor_pos <= lim[:, 1] + 1e-9)
    if all(real.calib["dofs"][f"{k}_knee"]["type"] == "lut1d" for k in ("left", "right")):
        assert s[SEMANTIC_NAMES.index("left_knee")] > 0.05 and s[SEMANTIC_NAMES.index("right_knee")] > 0.05


MIRROR_SIGN = {"hip_yaw": -1, "hip_roll": -1, "hip_pitch": 1, "knee": 1, "ankle_pitch": 1, "ankle_roll": -1,
               "shoulder_pitch": 1, "shoulder_roll": -1, "shoulder_yaw": -1, "elbow": 1, "wrist_roll": -1}


@real_only
def test_real_left_right_symmetry(real):
    """Mirror symmetry of valid ranges; every violation > 2 deg must be reported in the findings."""
    findings = " ".join(real.calib.get("findings", []))
    asym = {}
    for base, sgn in MIRROR_SIGN.items():
        l = np.array(real.calib["dofs"][f"left_{base}"]["valid_range"])
        r = np.sort(sgn * np.array(real.calib["dofs"][f"right_{base}"]["valid_range"]))
        d = float(np.degrees(np.abs(l - r).max()))
        if d > 2.0:
            asym[base] = d
    for base in asym:
        assert f"asymmetr" in findings and base in findings, (base, asym)
    # linear scales mirror exactly up to the fitted slope
    for base in ("hip_roll", "hip_yaw", "hip_pitch", "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "wrist_roll"):
        l, r = real.calib["dofs"][f"left_{base}"], real.calib["dofs"][f"right_{base}"]
        assert abs(abs(l["scale"]) - abs(r["scale"])) < 0.02, base


# ----------------------------------------------------------------------------------------------------
def _serial3_calib(tilt_deg: float = 10.0) -> dict:
    """Synthetic calibration whose left/right hips are serial3 chains roll(X) -> yaw(Z') -> pitch(Y') with the
    yaw/pitch axes tilted about x (Dropbear-like); everything else linear."""
    c = _synthetic()
    t = np.radians(tilt_deg)
    groups = {}
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        pg, pre = ("PG_left_leg", "LL") if side == "left" else ("PG_right_leg", "RL")
        chain = [f"{pg}_pitch", f"{pg}_roll", f"{pre}_hip_joint"]
        sem = [f"{side}_hip_pitch", f"{side}_hip_roll", f"{side}_hip_yaw"]
        role = [f"{pre}_hip_joint", f"{pg}_pitch", f"{pg}_roll"]
        axes = [[1.0, 0.0, 0.0], [0.0, -sgn * np.sin(t), -np.cos(t)], [0.0, np.cos(t), -sgn * np.sin(t)]]
        for n, m in zip(sem, role):
            c["dofs"][n] = {"type": "serial3", "motors": [m], "scale": 1.0 if "pitch" in n or "roll" in n else -1.0,
                            "offset": 0.0, "valid_range": [-0.6, 0.6]}
        groups[f"{side}_hip"] = {"chain_motors": chain, "semantic": sem, "role_motors": role,
                                 "linear_scale": [1.0, 1.0, -1.0], "linear_offset": [0.0, 0.0, 0.0],
                                 "axes_root": axes, "slopes": [1.0, 1.0, 1.0], "R_rest": np.eye(3).tolist(),
                                 "R_ref": np.eye(3).tolist(), "semantic_box": [[-0.6, 0.6]] * 3}
    c["serial_groups"] = groups
    return c


def _rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def test_serial3_forward_matches_chain_and_inverse_roundtrips():
    from dropbear_wbc.kinematics.semantic import euler_yxz

    c = _serial3_calib()
    smap = SemanticMap(c)
    g = c["serial_groups"]["left_hip"]
    rng = np.random.default_rng(3)
    m = np.zeros((200, 22))
    idx = [MOTOR_NAMES.index(n) for n in g["chain_motors"]]
    m[:, idx] = rng.uniform(-0.5, 0.5, (200, 3))
    s = smap.motor_to_semantic(m)
    for k in range(0, 200, 37):  # independent chain product
        r = _rot(g["axes_root"][0], m[k, idx[0]]) @ _rot(g["axes_root"][1], m[k, idx[1]]) @ _rot(g["axes_root"][2], m[k, idx[2]])
        e = euler_yxz(r)
        np.testing.assert_allclose(s[k, [SEMANTIC_NAMES.index(n) for n in g["semantic"]]], e, atol=1e-9)
    m2 = smap.semantic_to_motor(s)
    np.testing.assert_allclose(m2[:, idx], m[:, idx], atol=1e-8)
    # the per-DOF linear map is NOT the same thing for combined motion (why serial3 exists)
    lin_pitch = m[:, MOTOR_NAMES.index("LL_hip_joint")]
    assert np.abs(lin_pitch - s[:, SEMANTIC_NAMES.index("left_hip_pitch")]).max() > 0.02


def test_serial3_motor_clipping_reports_achieved_orientation():
    smap = SemanticMap(_serial3_calib())
    q = np.zeros((1, 22))
    q[0, SEMANTIC_NAMES.index("left_hip_pitch")] = 0.59
    q[0, SEMANTIC_NAMES.index("left_hip_yaw")] = 0.59
    chain = [MOTOR_NAMES.index(n) for n in ("PG_left_leg_pitch", "PG_left_leg_roll", "LL_hip_joint")]
    smap.motor_limits[chain] = [-0.3, 0.3]
    m, rep = smap.semantic_to_motor(q, return_report=True)
    assert np.all(np.abs(m[0, chain]) <= 0.3 + 1e-12)
    back = smap.motor_to_semantic(m)
    hip = [SEMANTIC_NAMES.index(f"left_hip_{k}") for k in ("pitch", "roll", "yaw")]
    np.testing.assert_allclose(back[0, hip], rep.used[0, hip], atol=1e-9)  # used = orientation actually reached
    assert rep.clipped[0, SEMANTIC_NAMES.index("left_hip_pitch")] and rep.clipped[0, SEMANTIC_NAMES.index("left_hip_yaw")]
