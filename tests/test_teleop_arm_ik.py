"""Dropbear arm IK (``dropbear_wbc.teleop.arm_ik``): model vs calibration, reachability, joint limits, IK error
on FK-generated targets, left/right mirror symmetry, continuity, calibration reload, motor mapping.

CPU only; runs with the system Python, ``.venv-teleop`` or ``.venv-newton``.
"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (sys.path)
from dropbear_wbc.teleop.arm_ik import (
    MIRROR_SIGN, SIDES, DropbearArmIK, IKWeights, analytic_seed, mirror_pose, mirrored_chain, pose, so3_log,
    solve_chain, solve_chain_robust,
)
from dropbear_wbc.kinematics.semantic import DEFAULT_CALIBRATION

pytestmark = pytest.mark.skipif(not DEFAULT_CALIBRATION.exists(), reason="no semantic calibration JSON")


@pytest.fixture(scope="module")
def ik() -> DropbearArmIK:
    return DropbearArmIK()


def _random_q(chain, rng, n):
    return rng.uniform(chain.lower, chain.upper, size=(n, 5))


# ------------------------------------------------------------------------------------------------ model
def test_chain_matches_calibration_geometry(ik):
    """The serial model reproduces the calibration's measured wrist positions at the semantic zero and the
    standing pose (the four-bar elbow pivot residual is ~5 mm rms in the calibration itself)."""
    for side in SIDES:
        checks = ik.chains[side].info["model_vs_calibration"]
        assert set(checks) == {"semantic_zero", "standing"}
        for name, c in checks.items():
            assert c["error_m"] < 0.005, (side, name, c)
        info = ik.chains[side].info
        assert 0.29 < info["upper_arm_m"] < 0.31 and 0.10 < info["forearm_m"] < 0.11, info


def test_elbow_table_model(ik):
    """Default elbow model = the calibration's measured wrist path (falls back to the pivot without the raw sweep).
    It reproduces the calibrated wrist positions exactly and differs from the pivot model by the pivot residual."""
    if ik.elbow_model_info["used"] != "table":
        pytest.skip(f"raw sweep unavailable: {ik.elbow_model_info.get('fallback_reason')}")
    piv = DropbearArmIK(elbow_model="pivot")
    rng = np.random.default_rng(12)
    for side in SIDES:
        for name, c in ik.chains[side].info["model_vs_calibration"].items():
            assert c["error_m"] < 2e-4, (side, name, c)
        q = rng.uniform(ik.chains[side].lower, ik.chains[side].upper, size=(200, 5))
        d = np.linalg.norm(ik.chains[side].fk_batch(q)[0] - piv.chains[side].fk_batch(q)[0], axis=-1)
        assert 0.001 < d.max() < 0.02  # the pivot model's wrist error (~5 mm typical, 13 mm at full flexion)


def test_fk_batch_and_jacobian(ik):
    rng = np.random.default_rng(0)
    for side in SIDES:
        ch = ik.chains[side]
        qs = _random_q(ch, rng, 20)
        pb, rb = ch.fk_batch(qs)
        for q, p_b, r_b in zip(qs, pb, rb):
            p, r, j = ch.fk(q, jacobian=True)
            assert np.allclose(p, p_b, atol=1e-12) and np.allclose(r, r_b, atol=1e-12)
            jn = np.zeros((6, 5))
            for i in range(5):
                dq = np.zeros(5)
                dq[i] = 1e-6
                p2, r2 = ch.fk(q + dq)
                jn[:3, i] = (p2 - p) / 1e-6
                jn[3:, i] = so3_log(r2 @ r.T) / 1e-6
            assert np.abs(jn - j).max() < 1e-5


def test_ee_frame_convention(ik):
    """EE rotation is the identity at the semantic zero (forearm forward, Unitree arm convention) and
    Ry(pi/2) (x pointing down) with the straight hanging arm."""
    for side in SIDES:
        ch = ik.chains[side]
        _, r0 = ch.fk(np.zeros(5))
        assert np.allclose(r0, np.eye(3), atol=1e-12)
        q = np.array([0.0, 0.0, 0.0, np.pi / 2, 0.0])
        p, r = ch.fk(q)
        assert np.allclose(r[:, 0], [0, 0, -1], atol=1e-12)
        # straight arm: wrist below the elbow, about one forearm length
        assert p[2] < ch.points[3][2] - 0.09


# ------------------------------------------------------------------------------------------------ IK accuracy
def test_ik_error_on_fk_targets_warm(ik):
    """FK-generated (hence reachable, orientation-consistent) targets, warm start 0.1 rad away: error < 5 mm
    for every sample (measured well below 0.1 mm)."""
    rng = np.random.default_rng(1)
    for side in SIDES:
        ch = ik.chains[side]
        errs = []
        for q in _random_q(ch, rng, 80):
            t = pose(*ch.fk(q))
            q0 = ch.clip(q + rng.normal(0.0, 0.1, 5))
            r = solve_chain_robust(ch, t, q0, q0)
            errs.append(r.pos_err_m)
        errs = np.array(errs)
        assert errs.max() < 0.005, (side, errs.max())
        assert np.median(errs) < 1e-6


def test_reachability_cold_start(ik):
    """Cold start from the standing pose (multi-start seeds): >= 97 % of workspace targets within 5 mm."""
    rng = np.random.default_rng(2)
    for side in SIDES:
        ch = ik.chains[side]
        errs = np.array([solve_chain_robust(ch, pose(*ch.fk(q)), ch.q_rest).pos_err_m
                         for q in _random_q(ch, rng, 60)])
        assert (errs < 0.005).mean() >= 0.97, (side, np.sort(errs)[-5:])
        assert np.median(errs) < 1e-6


def test_unreachable_target_points_at_it(ik):
    """A target beyond the reach: the arm stretches toward it; the residual is the distance beyond reach."""
    for side in SIDES:
        ch = ik.chains[side]
        s = ch.points[0]
        direction = np.array([0.6, 0.3 if side == "left" else -0.3, -0.2])
        direction /= np.linalg.norm(direction)
        target = s + 2.0 * ch.reach * direction
        r = solve_chain_robust(ch, pose(target), ch.q_rest, weights=IKWeights(rotation=0.0))
        w, _ = ch.fk(r.q)
        u = (w - s) / np.linalg.norm(w - s)
        assert u @ direction > 0.99, (side, u, direction)
        assert abs(r.pos_err_m - np.linalg.norm(target - w)) < 1e-9
        assert np.linalg.norm(w - s) > 0.9 * np.linalg.norm(ch.wrist0 - s) - 0.05


def test_priority_vs_weighted_mode(ik):
    """Documents the design choice: the single weighted R1_A5 cost lets orientation residuals move the wrist;
    the priority mode does not."""
    rng = np.random.default_rng(4)
    ch = ik.chains["left"]
    e_pri, e_w = [], []
    for q in _random_q(ch, rng, 60):
        t = pose(*ch.fk(q))
        q0 = ch.clip(q + rng.normal(0.0, 0.1, 5))
        e_pri.append(solve_chain(ch, t, q0, q0, IKWeights(mode="priority")).pos_err_m)
        e_w.append(solve_chain(ch, t, q0, q0, IKWeights(mode="weighted")).pos_err_m)
    assert max(e_pri) < 5e-4  # singularity-robust damping slows the last 0.1 mm near a stretched arm
    assert max(e_w) > 1e-3  # the weighted cost trades mm of position for orientation / posture


# ------------------------------------------------------------------------------------------------ limits
def test_joint_limits_respected(ik):
    """Arbitrary targets (reachable or not, any orientation): semantic solutions stay inside the IK limits and
    the motor targets inside the motor limits."""
    rng = np.random.default_rng(5)
    for k in range(80):
        targets = {}
        for side in SIDES:
            ch = ik.chains[side]
            p = ch.points[0] + rng.normal(0.0, 0.35, 3) - ik.torso_origin_root
            rv = rng.normal(0.0, 1.5, 3)
            from dropbear_wbc.motion.rotations import quat_from_axis_angle, quat_to_matrix
            ang = np.linalg.norm(rv)
            rot = quat_to_matrix(quat_from_axis_angle(rv / max(ang, 1e-9), np.array(ang)))
            targets[side] = pose(p, rot)
        res = ik.solve(targets["left"], targets["right"])
        for i, side in enumerate(SIDES):
            ch = ik.chains[side]
            q = res.q_sem[5 * i:5 * i + 5]
            assert np.all(q >= ch.lower - 1e-12) and np.all(q <= ch.upper + 1e-12), (side, q)
        m10, _ = ik.to_motor(res.q_sem)
        lim = ik.smap.motor_limits[12:22]
        assert np.all(m10 >= lim[:, 0] - 1e-9) and np.all(m10 <= lim[:, 1] + 1e-9)


# ------------------------------------------------------------------------------------------------ symmetry
def _mirror_about(t: np.ndarray, y_mid: float) -> np.ndarray:
    """Mirror a root-frame pose through the plane y = y_mid."""
    out = mirror_pose(t)
    out[1, 3] = 2.0 * y_mid - t[1, 3]
    return out


def test_mirror_symmetry_exact_solver():
    """With a perfectly mirrored chain pair the solver is exactly symmetric: mirrored targets -> mirrored q."""
    ik = DropbearArmIK()
    left = ik.chains["left"]
    y_mid = ik.torso_origin_root[1]
    right = mirrored_chain(left, "right", y_mid)
    rng = np.random.default_rng(6)
    for q in _random_q(left, rng, 40):
        t_l = pose(*left.fk(q))
        t_r = _mirror_about(t_l, y_mid)
        q0 = left.clip(q + rng.normal(0.0, 0.1, 5))
        r_l = solve_chain_robust(left, t_l, q0, q0)
        r_r = solve_chain_robust(right, t_r, q0 * MIRROR_SIGN, q0 * MIRROR_SIGN)
        assert np.abs(r_r.q - MIRROR_SIGN * r_l.q).max() < 1e-6, (r_l.q, r_r.q)


def test_mirror_symmetry_calibrated(ik):
    """The real right arm vs the mirrored left arm (torso frame y -> -y): mirrored configurations give mirrored
    wrists within the calibration's left/right asymmetry (a few mm), and mirrored targets mirrored solutions."""
    rng = np.random.default_rng(7)
    left, right = ik.chains["left"], ik.chains["right"]
    dmax, qd, n = 0.0, [], 0
    for q in _random_q(left, rng, 80):
        qr = MIRROR_SIGN * q
        if np.any(qr < right.lower) or np.any(qr > right.upper):
            continue
        n += 1
        t_l = ik.fk("left", q)
        t_r = ik.fk("right", qr)
        dmax = max(dmax, float(np.linalg.norm(mirror_pose(t_l)[:3, 3] - t_r[:3, 3])))
        q0 = np.concatenate([q, qr]) + rng.normal(0.0, 0.05, 10)
        ik.reset(q0)
        res = ik.solve(t_l, mirror_pose(t_l), q_init=q0)
        qd.append(float(np.abs(res.q_sem[5:] - MIRROR_SIGN * res.q_sem[:5]).max()))
        assert max(a.pos_err_m for a in res.arms.values()) < 0.001
    assert n > 40
    assert dmax < 0.006, dmax
    # the redundant DoF (swivel, wrist roll) amplify the mm-level calibration asymmetry a little
    assert np.median(qd) < 0.04 and max(qd) < 0.15, (np.median(qd), max(qd))


# ------------------------------------------------------------------------------------------------ continuity
def test_continuity_along_figure8(ik):
    """A 10 s figure-8 (+-7 cm, +-5 cm) at 100 Hz inside the workspace, fixed target orientation, warm-started
    with the control-loop iteration cap: per-step joint changes stay small and the wrist stays on target."""
    ik.reset()
    dt, n = 0.01, 1000
    qb = np.array([-0.5, 0.2, 0.0, 0.1, 0.0])
    base = {s: ik.fk(s, qb * (1.0 if s == "left" else MIRROR_SIGN)) for s in SIDES}
    q_last, max_dq, max_err = None, 0.0, 0.0
    for k in range(n):
        t = k * dt
        tg = {}
        for s in SIDES:
            m = base[s].copy()
            m[:3, 3] += np.array([0.0, 0.07 * np.sin(2 * np.pi * t / 5.0), 0.05 * np.sin(4 * np.pi * t / 5.0)])
            d = np.linalg.norm(ik.torso_to_root(m)[:3, 3] - ik.chains[s].points[0])
            assert d < 0.95 * ik.chains[s].reach  # the path stays inside the workspace
            tg[s] = m
        res = ik.solve(tg["left"], tg["right"], max_iters=6)
        if q_last is not None:
            max_dq = max(max_dq, float(np.abs(res.q_sem - q_last).max()))
        q_last = res.q_sem
        max_err = max(max_err, max(a.pos_err_m for a in res.arms.values()))
    assert max_dq < 0.03, max_dq   # target moves <= 4.4 mm per step
    assert max_err < 0.001, max_err


def test_straight_arm_trap_escaped_by_restarts():
    """Regression (logs/teleop/probe_elbow_contract_final_analysis.log): moving the wrist from a nearly straight
    hanging arm (elbow 1.4) to elbow 1.0 drove the warm-started IK onto the straight-elbow limit, where
    d|wrist - shoulder| / d elbow ~ 0, and it stayed ~3.9 mm off for seconds with the 10 mm restart threshold.
    The control-loop settings (2 mm threshold, light seeds) keep the residual below 2 mm."""
    from dropbear_wbc.teleop.devices import ScriptedSource

    # the trap exists with the best-fit pivot elbow (where |wrist - shoulder| peaks at the elbow limit); the measured
    # wrist-path table (default) does not have it, and must also stay below 2 mm with the loop settings
    ik = DropbearArmIK(elbow_model="pivot")
    qs = [(0.0, 0.05, 0.0, 1.4, 0.0), (0.0, 0.05, 0.0, 1.0, 0.0)]
    start = {"left": ik.fk("left", ik.rest_q()[:5]), "right": ik.fk("right", ik.rest_q()[5:])}
    poses = {"left": [ik.fk("left", np.array(q)) for q in qs],
             "right": [ik.fk("right", np.array(q) * MIRROR_SIGN) for q in qs]}
    worst = {}
    for thr in (10.0, 2.0):
        src = ScriptedSource(start, kind="poses", poses=poses, hold_s=2.0, move_s=1.0)
        src.start()
        ik.reset()
        w_max = 0.0
        for k in range(300):
            w = src.get(k * 0.02)
            r = ik.solve(w.left, w.right, max_iters=5, restarts="light", restart_above_m=thr * 1e-3)
            w_max = max(w_max, r.arms["left"].pos_err_m, r.arms["right"].pos_err_m)
        worst[thr] = w_max
    assert worst[10.0] > 0.003   # the trap exists
    assert worst[2.0] < 0.002    # and the loop settings escape it
    ik_t = DropbearArmIK()       # measured table model
    if ik_t.elbow_model_info["used"] == "table":
        src = ScriptedSource(start, kind="poses", poses=poses, hold_s=2.0, move_s=1.0)
        src.start()
        ik_t.reset()
        w_max = 0.0
        for k in range(300):
            w = src.get(k * 0.02)
            r = ik_t.solve(w.left, w.right, max_iters=5, restarts="light", restart_above_m=2e-3)
            w_max = max(w_max, r.arms["left"].pos_err_m, r.arms["right"].pos_err_m)
        assert w_max < 0.002


def test_reachable_shell_bounds(ik):
    """The exact shell (min, max shoulder-centre -> wrist distance) bounds every FK sample."""
    rng = np.random.default_rng(11)
    for side in SIDES:
        ch = ik.chains[side]
        lo, hi = ch.shell
        w, _ = ch.fk_batch(rng.uniform(ch.lower, ch.upper, size=(2000, 5)))
        d = np.linalg.norm(w - ch.points[0], axis=-1)
        assert d.min() >= lo - 1e-3 and d.max() <= hi + 1e-3
        assert 0.25 < lo < 0.30 and 0.39 < hi < 0.42


# ------------------------------------------------------------------------------------------------ reload / motor
def test_reload_when_calibration_changes(tmp_path):
    src = DEFAULT_CALIBRATION
    dst = tmp_path / "calib.json"
    shutil.copy(src, dst)
    ik = DropbearArmIK(dst, elbow_model="pivot")  # every chain parameter then comes from the JSON being edited
    q = np.array([-0.4, 0.2, 0.1, 0.8, 0.0])
    before = ik.fk("left", q)[:3, 3].copy()
    assert not ik.reload_if_changed()
    c = json.loads(dst.read_text())
    for key in ("shoulder_center", "elbow_center", "wrist"):
        c["geometry"]["per_side"]["left"]["arm_points_rest_root"][key][2] += 0.01
    dst.write_text(json.dumps(c))
    os.utime(dst, (time.time() + 5, time.time() + 5))
    assert ik.reload_if_changed()
    after = ik.fk("left", q)[:3, 3]
    assert abs((after - before)[2] - 0.01) < 1e-9 and np.abs((after - before)[:2]).max() < 1e-9
    assert not ik.reload_if_changed()


def test_to_motor_fast_equals_contract_path(ik):
    rng = np.random.default_rng(8)
    lo = np.concatenate([ik.chains[s].lower for s in SIDES])
    hi = np.concatenate([ik.chains[s].upper for s in SIDES])
    for _ in range(50):
        q = rng.uniform(lo - 0.2, hi + 0.2)
        a, sa = ik.to_motor(q)
        b, sb = ik.to_motor_fast(q)
        assert np.array_equal(a, b) and set(sa) == set(sb)


def test_semantic_motor_roundtrip_of_solutions(ik):
    rng = np.random.default_rng(9)
    for _ in range(30):
        q = np.concatenate([_random_q(ik.chains[s], rng, 1)[0] for s in SIDES])
        m10, sat = ik.to_motor(q)
        if sat:
            continue
        back = ik.motor_to_semantic_arms(ik.motor22(m10))
        assert np.abs(back - q).max() < 2e-3, (q, back)


def test_analytic_seed_distance(ik):
    """The seed puts the wrist at the right distance from the shoulder (elbow from the distance table)."""
    rng = np.random.default_rng(10)
    ch = ik.chains["left"]
    for q in _random_q(ch, rng, 30):
        p, _ = ch.fk(q)
        seed = analytic_seed(ch, p)
        ps, _ = ch.fk(seed)
        d_t, d_s = np.linalg.norm(p - ch.points[0]), np.linalg.norm(ps - ch.points[0])
        assert abs(d_t - d_s) < 0.03
