"""tools/run_chunked_locomotion.py --resume_run inherits the run's training flags; chunk loop index intact;
run_settings audit covers the locomotion key set (item 343; CPU, no Isaac)."""
from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest

import sdk_test_paths  # noqa: F401  (adds source/ to sys.path)

from dropbear_wbc.tasks.locomotion.run_continuity import (  # noqa: E402
    check_settings_continuity,
    locomotion_settings,
)
from dropbear_wbc.tasks.tracking.run_settings import audit_run_dir  # noqa: E402
from test_locomotion_run_continuity import _RUN, _parser  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("run_chunked_locomotion_t", _REPO / "tools" / "run_chunked_locomotion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(argv, profile="hw_cem6", task="Dropbear-Velocity-Flat-v0"):
    a = _parser().parse_args(argv)
    info = {"task": task, **locomotion_settings(a, actuator_profile=profile),
            "pushes": bool(a.pushes or a.push_velocity is not None or a.push_interval is not None),
            "push_velocity": a.push_velocity, "push_interval": a.push_interval}
    return json.loads(json.dumps(info))


def _drive(monkeypatch, tmp_path, argv, info=None, rcs=None, ckpt=True):
    """Run the driver's main() with subprocess stubbed; returns (train argv lists, chunks.json records)."""
    tool = _tool()
    exp = tmp_path / "exp"
    run = exp / "R"
    run.mkdir(parents=True)
    if info is not None:
        (run / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    if ckpt:
        (run / "model_0.pt").write_bytes(b"")
    cmds, rcs = [], list(rcs or [])

    def fake_run(cmd, **kw):
        cmds.append(cmd[cmd.index("scripts/train_locomotion.py") + 1:])
        return types.SimpleNamespace(returncode=rcs.pop(0) if rcs else 0)

    monkeypatch.setattr(tool, "EXP", exp)
    monkeypatch.setattr(tool, "pin_calibration", lambda src: None)
    monkeypatch.setattr(tool, "clear_pre_boot_lock", lambda: None)
    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    monkeypatch.setattr(tool.time, "sleep", lambda s: None)
    rc = tool.main(["--run_name", "R", "--resume_run", "R", "--log_dir", str(tmp_path / "logs"), *argv])
    recs = json.loads((run / "chunks.json").read_text(encoding="utf-8")) if (run / "chunks.json").is_file() else []
    return rc, cmds, recs


def _train_settings(cmd, profile):
    a = _parser().parse_args([x for x in cmd if x != "--headless"])
    return a, locomotion_settings(a, actuator_profile=a.actuator_profile or profile)


_CASES = [
    _RUN + ["--arm_gate_hold_s=2", "--amp_grad_penalty=5", "--ankle_roll_bind_margin_deg=2", "--symmetry_mirror_loss=1",
            "--pushes", "--push_interval", "4", "8", "--push_velocity=0.6", "--stair_max_m=0.15", "--slope_max=0.35"],
    [],
    ["--gait_v2", "--knee_band", "200,5", "--stop_load_penalty=-5", "--impact_penalty=-0.005", "--knee_stop_penalty=-1"],
    ["--gait_shaping", "--target_clamp_margin_deg=0", "--action_rate_weight=-0.05", "--arm_action_scale=0.3",
     "--arm_deviation_weight=-0.02", "--clip_excess_arms=-1.0"],
]


@pytest.mark.parametrize("argv", _CASES, ids=["full", "bare", "v2_stop", "shaping"])
def test_resume_without_flags_reproduces_every_recorded_setting(monkeypatch, tmp_path, argv):
    info = _record(argv)
    rc, cmds, _ = _drive(monkeypatch, tmp_path, ["--chunks", "1"], info)
    assert rc == 0 and len(cmds) == 1
    a, cur = _train_settings(cmds[0], "hw_cem6")
    assert check_settings_continuity(info, cur, continue_run=True) == []  # train_locomotion would accept the chunk
    assert {k: info[k] for k in locomotion_settings(a, actuator_profile="x")} == json.loads(json.dumps(cur))
    assert a.task == info["task"] and a.continue_run and a.load_run == "R"
    pushes = bool(a.pushes or a.push_velocity is not None or a.push_interval is not None)
    assert (pushes, a.push_velocity, a.push_interval) == (info["pushes"], info["push_velocity"], info["push_interval"])


def test_given_flag_wins_and_differs_loudly(monkeypatch, tmp_path):
    info = _record(_RUN)
    _, cmds, _ = _drive(monkeypatch, tmp_path, ["--chunks", "1", "--thermal_penalty=-1", "--task", "Other-v0"], info)
    a, cur = _train_settings(cmds[0], "hw_cem6")
    assert a.thermal_penalty == -1 and a.task == "Other-v0" and a.gait_v3  # rest inherited
    with pytest.raises(ValueError, match="--thermal_penalty"):
        check_settings_continuity(info, cur, continue_run=True)
    _, cmds, _ = _drive(monkeypatch, tmp_path / "b", ["--chunks", "1", "--allow_settings_change"], info)
    assert "--allow_settings_change" in cmds[0]


def test_run_without_run_info_inherits_nothing(monkeypatch, tmp_path):
    _, cmds, _ = _drive(monkeypatch, tmp_path, ["--chunks", "1"], None, ckpt=False)
    a, _ = _train_settings(cmds[0], None)
    assert not a.gait_v3 and a.task == "Dropbear-Velocity-Flat-v0" and not a.continue_run


def test_chunk_index_survives_clip_excess_flags_and_crash_retry(monkeypatch, tmp_path):
    # before item 343 the flag loop reused `k` (the chunk index): str seed on retry, TypeError after chunk 1
    rc, cmds, recs = _drive(monkeypatch, tmp_path, ["--chunks", "3", "--seed", "10", "--clip_excess_arms=-1.0",
                                                    "--clip_excess_legs=-0.5"], _record([]), rcs=[0, 1, 0, 0])
    assert rc == 0 and [r["chunk"] for r in recs] == [1, 2, 3]
    seeds = [int(c[c.index("--seed") + 1]) for c in cmds]
    assert seeds == [10, 11, 111, 12]
    assert recs[1]["crash_retries_used"] == 1 and recs[1]["log"].endswith("chunk2_retry1.log")


def _write_run(d: Path, infos):
    d.mkdir(parents=True)
    (d / "run_info.json").write_text(json.dumps(infos[0]), encoding="utf-8")
    for i, info in enumerate(infos[1:]):
        (d / f"run_info_resume_2026-11-0{i + 1}_00-00-00.json").write_text(json.dumps(info), encoding="utf-8")


def test_audit_reports_locomotion_drift(tmp_path):
    _write_run(tmp_path / "drift", [_record(_RUN), _record([])])  # pre-0085 silent drop of the flags
    keys = {r["key"] for r in audit_run_dir(tmp_path / "drift")}
    assert {"gait_v3", "terrain", "amp_weight", "symmetry", "arm_swing_weight", "clip_excess_legs",
            "ankle_roll_bind", "thermal_penalty"} <= keys
    _write_run(tmp_path / "same", [_record(_RUN), _record(_RUN)])
    assert audit_run_dir(tmp_path / "same") == []
    _write_run(tmp_path / "off", [_record([]), _record(["--arm_swing_k=0.6", "--amp_grad_penalty=3"])])
    assert audit_run_dir(tmp_path / "off") == []  # parameters of disabled features are not drift
    flagged = _record([])
    flagged["run_settings_notes"] = ["training settings CHANGED gait_v3 (--allow_settings_change)"]
    _write_run(tmp_path / "flag", [_record(_RUN), flagged])
    assert all(r["flagged"] for r in audit_run_dir(tmp_path / "flag"))


def test_no_nested_for_loop_reuses_the_outer_loop_variable():
    import ast

    bad = []
    for f in sorted([*(_REPO / "tools").rglob("*.py"), *(_REPO / "scripts").rglob("*.py")]):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.For) and isinstance(n.target, ast.Name):
                bad += [f"{f.relative_to(_REPO)}:{m.lineno} reuses {n.target.id!r} of line {n.lineno}"
                        for m in ast.walk(n) if m is not n and isinstance(m, ast.For)
                        and isinstance(m.target, ast.Name) and m.target.id == n.target.id]
    assert not bad, bad
