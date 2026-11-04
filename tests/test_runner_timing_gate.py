"""Item 312: policy_runner --timing-gate applies the loop_timing_report check in-process (first-power gate)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from test_wall_schedule_rt import _run_runner_with_stall

_ROOT = Path(__file__).resolve().parents[1]


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "tools" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Loose limits so a loaded CI host cannot fail the "clean" run; the defaults are exercised by the stall case.
_LOOSE = ("--timing-limit", "max_p99_late_ms=1000", "--timing-limit", "max_late_ms=1000",
          "--timing-limit", "max_skipped_slots=1000", "--timing-limit", "max_overrun_steps=1000")


def test_gate_matches_post_run_report(tmp_path):
    rc, s, steps = _run_runner_with_stall(tmp_path, (), 0.0, extra=("--timing-gate", *_LOOSE))
    gate = s["timing_gate"]
    assert rc == 0 and s["status"] == "executed" and gate["status"] == "pass", s.get("error")
    post = _mod("loop_timing_report").check(steps, limits=gate["limits"])
    for k in ("steps", "overrun_steps", "skipped_slots", "dup_slots", "max_gap_slots", "late_streak", "late_ms"):
        assert gate[k] == post[k], k
    assert gate["steps"] == s["loop_timing"]["steps"] == len(steps)


def test_stall_fails_gate_with_default_limits(tmp_path):
    rc, s, _ = _run_runner_with_stall(tmp_path, (50,), 0.105, extra=("--timing-gate",))
    gate = s["timing_gate"]
    assert rc == 1 and s["status"] == "failed" and gate["status"] == "fail"
    assert gate["skipped_slots"] >= 4 and any(f.startswith("skipped_slots=") for f in gate["failures"])
    assert s["error"].startswith("timing gate:")


def test_no_gate_keeps_status(tmp_path):
    rc, s, _ = _run_runner_with_stall(tmp_path, (50,), 0.105)
    assert "timing_gate" not in s and s["status"] == "executed" and rc == 0


@pytest.mark.parametrize("argv", [["--timing-gate"], ["--clock", "wall", "--timing-limit", "max_p99=2"],
                                  ["--clock", "wall", "--timing-limit", "max_p99_late_ms"]])
def test_bad_args_rejected(argv):
    with pytest.raises(SystemExit) as e:
        _mod("policy_runner").parse_args(argv)
    assert e.value.code == 2


def test_limit_implies_gate_and_casts():
    a = _mod("policy_runner").parse_args(["--clock", "wall", "--timing-limit", "max_skipped_slots=3"])
    assert a.timing_gate and a.timing_limits["max_skipped_slots"] == 3 and isinstance(a.timing_limits["max_skipped_slots"], int)
