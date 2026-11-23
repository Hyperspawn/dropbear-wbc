"""tools/calibrate_semantics.py points at the post-recalibration gate (item 373, dropbear-backlog notes/371, notes/373).

A run that ends with a successful ``fit`` prints the tools/check_calibration_consistency.py commands (check only and
``--write``); ``--run-gate`` (``--stage fit`` only) runs the check-only gate and returns its exit code. The stages are
replaced by no-ops, so nothing here needs Isaac, a GPU or the raw sweep.
"""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import sdk_test_paths  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/calibrate_semantics.py"
GATE = ROOT / "tools/check_calibration_consistency.py"
DEFAULT_OUT = ROOT / "data/calibration/dropbear_semantic_calibration.json"


def _load(argv=()):
    """Import the tool with a controlled argv (it parses its CLI at import)."""
    old = sys.argv
    sys.argv = [str(TOOL), *argv]
    try:
        spec = importlib.util.spec_from_file_location("calibrate_semantics_373", TOOL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


cs = _load()


def _args(**kw):
    a = SimpleNamespace(run_gate=False, gate_no_tests=False, out=DEFAULT_OUT)
    a.__dict__.update(kw)
    return a


# ---------------------------------------------------------------- message / command

def test_gate_tool_path_is_the_real_gate():
    assert cs.GATE_TOOL == GATE and GATE.is_file()
    assert cs.DEFAULT_OUT == DEFAULT_OUT
    from dropbear_wbc.kinematics import serial_provenance as sp

    assert cs.DEFAULT_OUT == ROOT / sp.CALIBRATION_REL  # the file the gate checks is the default --out
    assert cs.GATE_REL == "tools/check_calibration_consistency.py"


def test_message_names_check_and_write_commands_for_default_out():
    msg = cs.gate_message(DEFAULT_OUT)
    lines = msg.splitlines()
    assert "python tools/check_calibration_consistency.py            # check only" in msg
    assert "python tools/check_calibration_consistency.py --write" in msg
    assert all(ln.startswith("[calibrate]") for ln in lines)
    assert "NOT covered" not in msg
    assert "notes/371" in msg


def test_message_flags_out_not_covered(tmp_path):
    other = tmp_path / "cal_v9.json"
    msg = cs.gate_message(other)
    assert "NOT covered" in msg and str(other) in msg
    assert "data/calibration/dropbear_semantic_calibration.json" in msg
    # still tells the user the gate commands
    assert "--write" in msg


def test_relative_default_out_counts_as_covered(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert cs._gate_covers(Path("data/calibration/dropbear_semantic_calibration.json"))
    assert not cs._gate_covers(Path("data/calibration/other.json"))


def test_same_file_name_elsewhere_is_not_covered(tmp_path):
    twin = tmp_path / "data/calibration/dropbear_semantic_calibration.json"
    assert not cs._gate_covers(twin)
    assert "NOT covered" in cs.gate_message(twin)
    assert "--out" in cs.gate_usage_error(_args(run_gate=True, out=twin), ["fit"])


def test_gate_command():
    assert cs.gate_command() == [sys.executable, str(GATE)]
    assert cs.gate_command(no_tests=True) == [sys.executable, str(GATE), "--no-tests"]


def test_gate_cli_has_the_flags_the_message_uses():
    tree = ast.parse(GATE.read_text(encoding="utf-8"))
    flags = {n.args[0].value for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
             and n.args and isinstance(n.args[0], ast.Constant)}
    assert {"--write", "--no-tests"} <= flags


@pytest.mark.parametrize("rc", [0, 1, 2])
def test_run_gate_passes_command_and_exit_code(rc, capsys):
    seen = []
    assert cs.run_gate(no_tests=rc == 1, runner=lambda cmd: seen.append(cmd) or rc) == rc
    assert seen == [cs.gate_command(no_tests=rc == 1)]
    out = capsys.readouterr().out
    assert f"gate exit {rc}" in out
    assert ("--write" in out.splitlines()[-1]) == (rc != 0)


# ---------------------------------------------------------------- usage

def test_usage_errors(tmp_path):
    assert cs.gate_usage_error(_args(), ["fit"]) is None
    assert cs.gate_usage_error(_args(), ["sweep", "fit", "verify", "fit"]) is None
    assert cs.gate_usage_error(_args(run_gate=True), ["fit"]) is None
    assert cs.gate_usage_error(_args(run_gate=True, gate_no_tests=True), ["fit"]) is None
    assert "--run-gate" in cs.gate_usage_error(_args(gate_no_tests=True), ["fit"])
    assert "--stage fit" in cs.gate_usage_error(_args(run_gate=True), ["sweep", "fit", "verify", "fit"])
    assert "--stage fit" in cs.gate_usage_error(_args(run_gate=True), ["verify"])
    assert "--out" in cs.gate_usage_error(_args(run_gate=True, out=tmp_path / "x.json"), ["fit"])
    assert cs.gate_usage_error(_args(out=tmp_path / "x.json"), ["fit"]) is None  # print-only is fine


def test_cli_refuses_run_gate_with_other_out(tmp_path):
    r = subprocess.run([sys.executable, str(TOOL), "--stage", "fit", "--run-gate", "--out", str(tmp_path / "x.json")],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode != 0 and "gate checks" in r.stderr
    r = subprocess.run([sys.executable, str(TOOL), "--stage", "all", "--run-gate", "--raw", str(tmp_path / "a.npz"),
                        "--verify-raw", str(tmp_path / "b.npz")], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode != 0 and "--stage fit" in r.stderr
    assert not (tmp_path / "a.npz").exists()


def test_help_lists_gate_flags():
    r = subprocess.run([sys.executable, str(TOOL), "--help"], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0
    assert "--run-gate" in r.stdout and "--gate-no-tests" in r.stdout
    assert "check_calibration_consistency.py" in r.stdout


# ---------------------------------------------------------------- main() with no-op stages

@pytest.fixture
def run_main(monkeypatch, tmp_path):
    def _run(stage, fail=None, gate_rc=0, **kw):
        calls = []
        args = SimpleNamespace(stage=stage, raw=tmp_path / "s.npz", verify_raw=tmp_path / "v.npz", force=False,
                               gpu_lock_held=True, out=DEFAULT_OUT, run_gate=False, gate_no_tests=False)
        args.__dict__.update(kw)
        monkeypatch.setattr(cs, "ARGS", args)
        monkeypatch.setattr(cs, "_resolve_raw_paths", lambda: None)
        monkeypatch.setattr(cs, "_launch_app", lambda: None)
        lock = tmp_path / "repo/.locks/gpu.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("x")
        monkeypatch.setattr(cs, "REPO", tmp_path / "repo")

        def stage_fn(name):
            def f():
                calls.append(name)
                print(f"<{name}>")
                if fail == name:
                    raise RuntimeError(f"{name} broke")
            return f

        for n in ("sweep", "fit", "verify"):
            monkeypatch.setattr(cs, f"_stage_{n}", stage_fn(n))
        gate_calls = []
        monkeypatch.setattr(cs, "run_gate", lambda no_tests=False: gate_calls.append(no_tests) or gate_rc)
        return cs.main(), calls, gate_calls

    return _run


def test_fit_prints_gate_after_fit(run_main, capsys):
    rc, calls, gate = run_main("fit")
    out = capsys.readouterr().out
    assert rc == 0 and calls == ["fit"] and gate == []
    assert out.index("<fit>") < out.index("check_calibration_consistency.py --write")
    assert out.count("check_calibration_consistency.py --write") == 1


def test_all_prints_gate_once_after_final_fit(run_main, capsys):
    rc, calls, gate = run_main("all")
    out = capsys.readouterr().out
    assert rc == 0 and calls == ["sweep", "fit", "verify", "fit"]
    assert out.count("check_calibration_consistency.py --write") == 1
    assert out.rindex("<fit>") < out.index("check_calibration_consistency.py --write")


@pytest.mark.parametrize("stage", ["sweep", "verify"])
def test_non_fit_stage_does_not_print_gate(run_main, capsys, stage):
    rc, calls, _ = run_main(stage)
    assert rc == 0 and calls == [stage]
    assert "check_calibration_consistency" not in capsys.readouterr().out


def test_failed_fit_does_not_print_gate(run_main, capsys):
    rc, calls, gate = run_main("fit", fail="fit", run_gate=True)
    assert rc == 1 and gate == []
    assert "check_calibration_consistency" not in capsys.readouterr().out


def test_failed_verify_in_all_does_not_print_gate(run_main, capsys):
    rc, calls, _ = run_main("all", fail="verify")
    assert rc == 1 and calls == ["sweep", "fit", "verify"]
    assert "check_calibration_consistency" not in capsys.readouterr().out


@pytest.mark.parametrize("gate_rc", [0, 1])
@pytest.mark.parametrize("no_tests", [False, True])
def test_run_gate_after_fit_returns_gate_exit(run_main, capsys, gate_rc, no_tests):
    rc, calls, gate = run_main("fit", run_gate=True, gate_no_tests=no_tests, gate_rc=gate_rc)
    assert calls == ["fit"] and gate == [no_tests] and rc == gate_rc
    assert "check_calibration_consistency.py --write" in capsys.readouterr().out


def test_print_only_by_default_never_runs_gate(run_main):
    _, _, gate = run_main("fit")
    assert gate == []
