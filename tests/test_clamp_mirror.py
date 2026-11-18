"""Left/right mirror consistency of the locomotion target clamp under --symmetry (item 250; CPU, no Isaac)."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ to sys.path)

from dropbear_wbc.kinematics.semantic import DEFAULT_CALIBRATION  # noqa: E402
from dropbear_wbc.robots.dropbear_names import MOTOR_HARD_LIMITS_DEG, MOTOR_NAMES  # noqa: E402
from dropbear_wbc.tasks.locomotion.clamp_mirror import (  # noqa: E402
    MIRROR_TOL_DEG,
    asymmetric_motors,
    check_symmetry_clamp,
    clamp_clip_rad,
    clamp_mirror_record,
    clamp_mirror_report,
    mirrored_limits_deg,
    symmetric_limits_deg,
)
from dropbear_wbc.tasks.locomotion.run_continuity import (  # noqa: E402
    SETTING_FLAGS,
    UNRECORDED,
    check_settings_continuity,
    effective_settings,
)
from dropbear_wbc.tasks.locomotion.symmetry import motor_mirror_matrix  # noqa: E402
from test_chunked_locomotion_resume import _drive, _record, _train_settings  # noqa: E402
from test_locomotion_run_continuity import _settings  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
R81 = ("LL_Revolute81", "RL_Revolute81")


def _q0_deg() -> np.ndarray:
    return np.degrees(np.asarray(json.loads(Path(DEFAULT_CALIBRATION).read_text())["standing_motor_pos"], dtype=float))


def _box(limits):
    return (np.array([limits[n][0] for n in MOTOR_NAMES], float), np.array([limits[n][1] for n in MOTOR_NAMES], float))


def _mirror_escape(limits, n=20000, seed=0) -> np.ndarray:
    """Per sample: how far [deg] the mirror q0 + M (q - q0) of a uniform point of the box lands outside the box."""
    lo, hi = _box(limits)
    q = np.random.default_rng(seed).uniform(lo, hi, size=(n, len(MOTOR_NAMES)))
    q0, m = _q0_deg(), np.asarray(motor_mirror_matrix())
    qm = q0 + (q - q0) @ m.T
    return np.maximum(np.maximum(lo - qm, qm - hi), 0.0).max(axis=1)


def test_usd_clamp_is_mirror_inconsistent_only_at_revolute81():
    bad = asymmetric_motors()
    assert set(bad) == set(R81)
    assert all(9.5 < v < 11.0 for v in bad.values())
    rep = clamp_mirror_report()
    assert max(r["max_abs_deg"] for n, r in rep.items() if n not in R81) < 1.6  # elbows 1.55 (calibration gain 1.052)
    assert MOTOR_HARD_LIMITS_DEG["RL_Revolute81"] == (-60.0, 50.0) and MOTOR_HARD_LIMITS_DEG["LL_Revolute81"] == (-50.0, 60.0)
    mir = mirrored_limits_deg()
    assert mir["RL_Revolute81"] == pytest.approx((-50.0, 60.0), abs=0.2)  # mirror of the left box on the right
    assert mir["LL_Revolute81"] == pytest.approx((-60.0, 50.0), abs=0.7)


def test_symmetric_limits_are_a_mirror_invariant_sub_box():
    sym = symmetric_limits_deg()
    for n in MOTOR_NAMES:
        assert MOTOR_HARD_LIMITS_DEG[n][0] - 1e-9 <= sym[n][0] <= sym[n][1] <= MOTOR_HARD_LIMITS_DEG[n][1] + 1e-9
    assert asymmetric_motors(sym) == {}
    assert max(r["max_abs_deg"] for r in clamp_mirror_report(sym).values()) < 0.7
    for n in R81:
        assert sym[n] == pytest.approx((-50.0, 50.0), abs=0.6)
    shrink = {n: max(sym[n][0] - MOTOR_HARD_LIMITS_DEG[n][0], MOTOR_HARD_LIMITS_DEG[n][1] - sym[n][1]) for n in MOTOR_NAMES}
    assert all(v < 1.5 for n, v in shrink.items() if n not in R81)  # only R81 loses real range (RH elbow 1.47)
    assert all(9.5 < shrink[n] < 10.5 for n in R81)


def test_sampled_mirrors_escape_the_usd_box_but_not_the_symmetric_one():
    usd = _mirror_escape(MOTOR_HARD_LIMITS_DEG)
    sym = _mirror_escape(symmetric_limits_deg())
    assert 0.12 < float(np.mean(usd > MIRROR_TOL_DEG)) < 0.25  # R81 on either side beyond the partner's bound
    assert usd.max() > 9.0
    assert sym.max() < 0.7


def test_legacy_clip_dict_is_unchanged_without_symmetric():
    for margin in (0.0, 3.0):
        m = math.radians(margin)
        legacy = {n: (math.radians(lo) - m, math.radians(hi) + m) for n, (lo, hi) in MOTOR_HARD_LIMITS_DEG.items()}
        got = clamp_clip_rad(margin)
        assert got == legacy and list(got) == list(legacy)
    sym = clamp_clip_rad(3.0, symmetric=True)
    assert math.degrees(sym["RL_Revolute81"][0]) == pytest.approx(-53.0, abs=0.1)
    assert math.degrees(sym["LL_Revolute81"][1]) == pytest.approx(53.0, abs=0.6)


def test_gate_refuses_fresh_symmetry_runs_on_the_usd_clamp():
    kw = dict(symmetry=True, clamp_active=True, symmetric_clamp=False)
    with pytest.raises(ValueError, match="--symmetric_clamp.*--allow_asymmetric_clamp"):
        check_symmetry_clamp(continue_run=False, **kw)
    with pytest.raises(ValueError, match="RL_Revolute81"):
        check_symmetry_clamp(continue_run=False, **kw)
    assert check_symmetry_clamp(continue_run=False, symmetry=True, clamp_active=True, symmetric_clamp=True) == []
    warn = check_symmetry_clamp(continue_run=True, **kw)
    assert len(warn) == 1 and warn[0].startswith("WARNING") and "--continue_run" in warn[0]
    warn = check_symmetry_clamp(continue_run=False, allow_asymmetric=True, **kw)
    assert len(warn) == 1 and "--allow_asymmetric_clamp" in warn[0]
    assert check_symmetry_clamp(continue_run=False, symmetry=False, clamp_active=True, symmetric_clamp=False) == []
    assert check_symmetry_clamp(continue_run=False, symmetry=True, clamp_active=False, symmetric_clamp=False) == []
    fixed = dict(MOTOR_HARD_LIMITS_DEG, RL_Revolute81=(-50.0, 60.0))  # if CAD says the right side is (-50, 60)
    assert check_symmetry_clamp(continue_run=False, limits=fixed, **kw) == []


def test_run_info_record_of_the_built_clamp():
    assert clamp_mirror_record(None, False) is None
    rec = clamp_mirror_record(clamp_clip_rad(0.0), False)
    assert rec["worst_motor"] == "LL_Revolute81" and rec["max_mismatch_deg"] > 9.5
    assert set(rec["asymmetric_motors"]) == set(R81) and not rec["symmetric_clamp"]
    rec = clamp_mirror_record(clamp_clip_rad(3.0, symmetric=True), True)
    assert rec["symmetric_clamp"] and rec["asymmetric_motors"] == {} and rec["max_mismatch_deg"] < 0.7
    assert rec["clip_deg"]["RL_Revolute81"][0] == pytest.approx(-53.0, abs=0.1)
    json.dumps(rec)


def test_symmetric_clamp_is_a_continuity_setting():
    assert SETTING_FLAGS["symmetric_clamp"] == "--symmetric_clamp"
    base = ["--gait_v3", "--symmetry"]
    rec = json.loads(json.dumps(_settings(base)))
    assert rec["symmetric_clamp"] is False and _settings(base + ["--symmetric_clamp"])["symmetric_clamp"] is True
    with pytest.raises(ValueError, match="--symmetric_clamp"):
        check_settings_continuity(rec, _settings(base + ["--symmetric_clamp"]), continue_run=True)
    notes = check_settings_continuity(rec, _settings(base + ["--symmetric_clamp"]), continue_run=True, allow_change=True)
    assert any("CHANGED" in n and "symmetric_clamp" in n for n in notes)
    # without gait shaping there is no clamp: the flag is not a task change
    assert effective_settings(_settings(["--symmetric_clamp"]))["symmetric_clamp"] is None
    # a run_info from before item 250 lacks the key: noted, not refused
    old = {k: v for k, v in rec.items() if k != "symmetric_clamp"}
    notes = check_settings_continuity(old, _settings(base + ["--symmetric_clamp"]), continue_run=True)
    assert any(UNRECORDED in n and "symmetric_clamp" in n for n in notes)


def test_chunk_driver_passes_and_inherits_the_clamp_flags(monkeypatch, tmp_path):
    info = _record(["--gait_v3", "--symmetry", "--symmetric_clamp"])
    assert info["symmetric_clamp"] is True
    rc, cmds, _ = _drive(monkeypatch, tmp_path, ["--chunks", "1"], info)
    a, cur = _train_settings(cmds[0], "hw_cem6")
    assert rc == 0 and a.symmetric_clamp and "--symmetric_clamp" in cmds[0]
    assert check_settings_continuity(info, cur, continue_run=True) == []
    _, cmds, _ = _drive(monkeypatch, tmp_path / "b", ["--chunks", "1", "--allow_asymmetric_clamp"],
                        _record(["--gait_v3", "--symmetry"]))
    a, _ = _train_settings(cmds[0], "hw_cem6")
    assert a.allow_asymmetric_clamp and not a.symmetric_clamp


def _src(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


def test_train_locomotion_wiring():
    s = _src("scripts/train_locomotion.py")
    i_sym = s.index("clamp_targets_to_limits(args.target_clamp_margin_deg, symmetric=True)")
    assert s.index("enable_gait_shaping(knee_stance=-1.0 if args.gait_v2") < i_sym  # after gait shaping sets the clamp
    assert i_sym < s.index("env_cfg.enable_clip_excess_penalty(") < s.index("env_cfg.enable_ankle_roll_bind(")
    i_gate = s.index("check_symmetry_clamp(symmetry=args.symmetry")
    assert i_gate < s.index("env = gym.make(")
    assert "allow_asymmetric=args.allow_asymmetric_clamp" in s and "continue_run=args.continue_run" in s[i_gate:i_gate + 300]
    assert '"clamp_mirror": clamp_mirror' in s
    tree = ast.parse(s)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"]
    flags = {a.value for c in calls for a in c.args if isinstance(a, ast.Constant)}
    assert {"--symmetric_clamp", "--allow_asymmetric_clamp"} <= flags


def test_env_cfg_and_replay_scripts_use_the_symmetric_box():
    cfg = _src("source/dropbear_wbc/tasks/locomotion/config/dropbear/flat_env_cfg.py")
    fn = cfg[cfg.index("    def clamp_targets_to_limits"):cfg.index("    def set_stand_npz")]
    assert "clamp_clip_rad(margin_deg, symmetric=symmetric)" in fn and "self.symmetric_clamp = bool(symmetric)" in fn
    assert "MOTOR_HARD_LIMITS_DEG.items()" not in fn  # one source of the clip dict
    for rel in ("scripts/play_locomotion.py", "scripts/robustness_bench.py"):
        s = _src(rel)
        assert "clamp_targets_to_limits(margin, symmetric=symmetric)" in s and '"symmetric_clamp"' in s, rel
