"""CPU tests of ``dropbear_wbc.kinematics.ankle_bind``: the semantic ankle-roll clamp for the pinned tie-rod ankle
(exact numpy path used by deploy, tabulated torch path used in training, sidecar wiring through the deploy FSM).

Run: python -m pytest tests/test_ankle_roll_bind.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.kinematics.semantic import MOTOR_NAMES, SEMANTIC_NAMES

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/calibration/dropbear_semantic_calibration.json"
DEG = np.pi / 180
pytestmark = pytest.mark.skipif(not REAL.is_file(), reason="no measured calibration")


@pytest.fixture(scope="module")
def bind():
    from dropbear_wbc.kinematics.ankle_bind import AnkleRollBind

    return AnkleRollBind(roll_limit_deg=8.0, margin_deg=1.0)


@pytest.fixture(scope="module")
def box_targets(bind):
    lim = bind.smap.motor_limits
    return np.random.default_rng(0).uniform(lim[:, 0], lim[:, 1], (5000, 22))


def test_clamped_roll_within_limit_and_inside_untouched(bind, box_targets):
    q = box_targets
    qc, hit = bind.clamp(q, return_mask=True)
    assert 0.5 < hit.mean() < 0.85  # most of the per-motor box is beyond the pinned-rod bind (notes: 64-71 %)
    assert np.abs(bind.roll(qc)).max() <= 7.0 * DEG + 1e-6
    inside = ~hit.any(-1)
    assert np.array_equal(qc[inside], q[inside])  # bit for bit
    ank = [i for ab in bind.motor_indices for i in ab]
    other = [i for i in range(22) if i not in ank]
    assert np.array_equal(qc[:, other], q[:, other])
    lim = bind.smap.motor_limits
    assert np.all(qc >= lim[:, 0] - 1e-9) and np.all(qc <= lim[:, 1] + 1e-9)
    assert np.abs(bind.clamp(qc) - qc).max() < 1e-9  # idempotent


def test_pitch_kept_where_feasible(bind, box_targets):
    q = box_targets
    qc, hit = bind.clamp(q, return_mask=True)
    s0, s1 = bind.smap.motor_to_semantic(q), bind.smap.motor_to_semantic(qc)
    for k, side in enumerate(("left", "right")):
        i = SEMANTIC_NAMES.index(f"{side}_ankle_pitch")
        d = np.abs(s1[hit[:, k], i] - s0[hit[:, k], i])
        assert np.percentile(d, 95) < 0.05 * DEG


def test_left_r81_on_its_lower_bound_at_stand(bind):
    """notes/248 F3: LL_Revolute81 at its -50 deg clamp with R67 at the stand is ~12 deg of roll past the bind."""
    q = bind.smap.standing_motor_pos.copy()
    q[MOTOR_NAMES.index("LL_Revolute81")] = -50 * DEG
    assert abs(bind.roll(q)[0]) > 10 * DEG
    qc, hit = bind.clamp(q, return_mask=True)
    assert hit[0] and not hit[1]
    assert abs(abs(bind.roll(qc)[0]) - 7 * DEG) < 0.01 * DEG


def test_torch_lut_matches_limit(bind, box_targets):
    torch = pytest.importorskip("torch")
    from dropbear_wbc.kinematics.ankle_bind import apply_lut_torch

    q = box_targets
    qt = apply_lut_torch(torch.tensor(q), bind.lut(), bind.motor_indices).numpy()
    assert np.abs(bind.roll(qt)).max() <= 7.2 * DEG  # bilinear table: within 0.2 deg of the exact clamp
    _, hit = bind.clamp(q, return_mask=True)
    inside = ~hit.any(-1)
    near = np.abs(bind.roll(q)).max(-1) < 6.0 * DEG  # cells fully inside the limit carry a zero correction
    assert np.abs(qt[inside & near] - q[inside & near]).max() < 1e-9


def test_deploy_sidecar_applies_ankle_roll_bind(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import deploy_fixtures as fx
    from dropbear_wbc.deploy.config import load_sidecar
    from dropbear_wbc.deploy.fsm import DeployController, FsmState
    from dropbear_wbc.deploy.motion import OnnxMotion
    from dropbear_wbc.deploy.policy import OnnxPolicy
    from dropbear_wbc.kinematics.ankle_bind import AnkleRollBind
    from dropbear_wbc.sdk import motors

    fx.write_beyondmimic_onnx(tmp_path / "policy.onnx", weight_scale=50.0)  # large actions: targets far out
    side = fx.write_sidecar(tmp_path / "policy.json", "policy.onnx")
    side["action_scale"] = [1.0] * len(side["joint_names"])
    side["ankle_roll_bind"] = {"roll_limit_deg": 8.0, "margin_deg": 1.0}
    (tmp_path / "policy.json").write_text(json.dumps(side), encoding="utf-8")
    policy = OnnxPolicy(tmp_path / "policy.onnx")
    cfg = load_sidecar(tmp_path / "policy.json", policy.metadata)
    assert cfg.ankle_roll_bind == {"roll_limit_deg": 8.0, "margin_deg": 1.0}
    motion = OnnxMotion(cfg.motion, cfg.joint_names, cfg.step_dt, policy.reference, list(fx.REF_JOINTS),
                        list(fx.BODIES), fx.T_FRAMES)
    ctrl = DeployController(cfg, policy, motion)
    st = fx.low_state(0, np.asarray(motors.DEFAULT_POS))
    ctrl.request(FsmState.MOVE_TO_DEFAULT, st, 0.0)
    ctrl.step(st, 2.0)
    ctrl.request(FsmState.POLICY, st, 2.0)
    b = AnkleRollBind()
    rolls, raw = [], []
    for k in range(5):
        cmd, info = ctrl.step(fx.low_state(10 + k, np.asarray(motors.DEFAULT_POS)), 2.02 + 0.02 * k)
        q = cmd.motor.q.astype(float)
        rolls.append(np.abs(b.roll(q)).max())
        unclamped = np.asarray(motors.DEFAULT_POS, dtype=float).copy()
        unclamped[cfg.motor_ids] = cfg.action_offset + cfg.action_scale * info.action
        raw.append(np.abs(b.roll(unclamped)).max())
    assert max(raw) > 8.0 * DEG  # the clamp is exercised
    assert max(rolls) <= 7.0 * DEG + 1e-3  # float32 LowCmd
    bad = dict(side, ankle_roll_bind={"roll_limit_deg": 1.0, "margin_deg": 2.0})
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="ankle_roll_bind"):
        load_sidecar(tmp_path / "bad.json", policy.metadata)
