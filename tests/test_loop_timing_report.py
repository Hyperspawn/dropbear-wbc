"""Item 308: post-run loop timing check over steps.jsonl (tools/loop_timing_report.py)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("loop_timing_report", _ROOT / "tools" / "loop_timing_report.py")
ltr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ltr)


def _rows(n=200, late=0.3, period=0.02, fsm="policy"):
    return [{"step": i, "t": i * period + late * 1e-3, "slot": i, "late_ms": late, "fsm": fsm} for i in range(n)]


def test_clean_run_passes():
    rep = ltr.check(_rows())
    assert rep["status"] == "pass" and rep["skipped_slots"] == 0 and rep["max_gap_slots"] == 1 and rep["dup_slots"] == 0
    assert rep["late_ms"]["p99"] == pytest.approx(0.3)


def test_stall_counts_skipped_slots_and_overrun():
    rows = _rows(100)
    # a 105 ms stall at step 50: that step starts 86 ms late in slot 54, later steps shift by 4 slots (WallSchedule)
    for r in rows[50:]:
        r["slot"] += 4
        r["t"] += 0.08
    rows[50]["late_ms"] = 86.0
    rep = ltr.check(rows)
    assert rep["status"] == "fail" and rep["skipped_slots"] == 4 and rep["max_gap_slots"] == 5
    assert rep["overrun_steps"] == 1 and rep["late_streak"] == 1 and rep["late_ms"]["max"] == 86.0
    assert any(f.startswith("skipped_slots=") for f in rep["failures"]) and any("late_ms=86" in f for f in rep["failures"])


def test_burst_is_dup_slot():
    rows = _rows(20)
    rows[11]["slot"] = rows[10]["slot"]  # two steps in one slot: catch-up burst
    rep = ltr.check(rows)
    assert rep["dup_slots"] == 1 and rep["status"] == "fail" and any(f.startswith("dup_slots") for f in rep["failures"])


def test_p99_limit_and_override():
    rows = _rows(100)
    for r in rows[:3]:
        r["late_ms"] = 3.0  # 3 % of steps 3 ms late: under overrun (5 ms) but p99 > 2 ms
    rep = ltr.check(rows)
    assert rep["overrun_steps"] == 0 and rep["failures"] == ["p99_late_ms=3 > 2.0"]
    assert ltr.check(rows, limits={"max_p99_late_ms": 4.0})["status"] == "pass"


def test_by_fsm_split_and_sim_clock_skipped(tmp_path):
    rows = _rows(10, fsm="passive") + [dict(r, slot=r["slot"] + 10, t=r["t"] + 0.2) for r in _rows(10, fsm="policy")]
    rows[15]["late_ms"] = 9.0
    rep = ltr.check(rows)
    assert rep["by_fsm"]["policy"]["overrun_steps"] == 1 and rep["by_fsm"]["passive"]["overrun_steps"] == 0
    sim = [{"step": i, "t": i * 0.02, "slot": None, "late_ms": None, "fsm": "policy"} for i in range(5)]
    p = tmp_path / "steps.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in sim))
    assert ltr.main([str(p)]) == 0 and ltr.check(sim)["status"] == "skipped"


def test_cli_exit_codes(tmp_path):
    p = tmp_path / "steps.jsonl"
    rows = _rows(50)
    rows[20]["late_ms"] = 30.0
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert ltr.main([str(p), "--json", str(tmp_path / "r.json")]) == 1
    assert json.loads((tmp_path / "r.json").read_text())["status"] == "fail"
    assert ltr.main([str(p), "--max-late-ms", "50", "--max-overrun-steps", "1", "--max-p99-late-ms", "50"]) == 0
    (tmp_path / "bad.jsonl").write_text("{not json\n")
    assert ltr.main([str(tmp_path / "bad.jsonl")]) == 2


def test_real_runner_log_agrees_with_summary(tmp_path):
    """End to end on a real wall-clock runner log with injected 105 ms stalls: the report re-derives the runner's
    own loop_timing counts from steps.jsonl."""
    from test_wall_schedule_rt import _run_runner_with_stall

    _, s, steps = _run_runner_with_stall(tmp_path, (50, 75), 0.105)
    rep = ltr.check(steps)
    lt = s["loop_timing"]
    # WallSchedule also counts the final wait() whose LowState ends the --duration run (no LowCmd, no log row), so
    # loop_timing.steps is one more than the logged/commanded steps; that last wait can add at most one overrun.
    assert rep["steps"] == s["steps"] == lt["steps"] - 1
    assert rep["overrun_steps"] <= lt["overrun_steps"] <= rep["overrun_steps"] + 1 and rep["overrun_steps"] >= 2
    assert rep["skipped_slots"] <= lt["skipped_slots"] and rep["skipped_slots"] >= 8 and rep["dup_slots"] == 0
    assert rep["late_ms"]["max"] == pytest.approx(lt["late_ms"]["max"], abs=1e-6) and rep["status"] == "fail"
