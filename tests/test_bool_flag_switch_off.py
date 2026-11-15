"""An inherited boolean run setting can be switched off on a resumed chunk (item 181; CPU, no Isaac).

Before: scripts/train.py --hw_regularizers / --stop_model were store_true(default=None), so a --continue_run chunk could
inherit True but never pass False; run_chunked_training.py / run_chunked_locomotion.py had no --no-* form either (and
run_chunked_training.py no --allow_settings_change pass-through), so a recorded True was permanent for the run.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import types
from pathlib import Path

import pytest

import sdk_test_paths  # noqa: F401  (adds source/ to sys.path)

from dropbear_wbc.tasks.tracking.run_settings import CLI_DEFAULTS, RUN_SETTING_KEYS, resolve_run_settings  # noqa: E402
from test_chunked_locomotion_resume import _drive, _record  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
BOOL_KEYS = [k for k in RUN_SETTING_KEYS if isinstance(CLI_DEFAULTS[k], bool)]


def _actions(path: Path) -> dict[str, str]:
    """``{flag: action source}`` for every add_argument('--flag', action=...) in ``path``."""
    out = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            flag = getattr(node.args[0], "value", "")
            kw = {k.arg: k.value for k in node.keywords}
            if isinstance(flag, str) and flag.startswith("--") and "action" in kw:
                out[flag[2:]] = ast.unparse(kw["action"])
    return out


def test_bool_run_settings_have_a_no_form_in_train_py():
    assert BOOL_KEYS == ["hw_regularizers", "stop_model"]
    acts = _actions(_REPO / "scripts/train.py")
    for k in BOOL_KEYS:
        assert acts[k] == "argparse.BooleanOptionalAction", (k, acts[k])


def test_given_false_vs_recorded_true_refused_then_allowed():
    rec = {k: (True if k in BOOL_KEYS else CLI_DEFAULTS[k]) for k in RUN_SETTING_KEYS}
    rec["actuator_profile"] = "hw_v1ie"
    unset = {k: None for k in RUN_SETTING_KEYS}
    rs, _ = resolve_run_settings(unset, rec, continue_run=True)
    assert rs["hw_regularizers"] is True and rs["stop_model"] is True
    for k in BOOL_KEYS:
        with pytest.raises(ValueError, match=k):
            resolve_run_settings({**unset, k: False}, rec, continue_run=True)
        rs, notes = resolve_run_settings({**unset, k: False}, rec, continue_run=True, allow_change=True)
        assert rs[k] is False and any(n.startswith(f"{k}: CHANGED True -> False") for n in notes)


def _tracking_tool():
    spec = importlib.util.spec_from_file_location("run_chunked_training_t", _REPO / "tools" / "run_chunked_training.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _drive_tracking(monkeypatch, tmp_path, argv):
    tool = _tracking_tool()
    run = tmp_path / "exp" / "R"
    run.mkdir(parents=True)
    (run / "chunks.json").write_text(json.dumps([{"chunk": 1, "rc": 0, "calibration": None}]), encoding="utf-8")
    cmds = []

    def fake_run(cmd, **kw):
        cmds.append(cmd[cmd.index("scripts/train.py") + 1:])
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool, "EXP", tmp_path / "exp")
    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    monkeypatch.setattr(tool.time, "sleep", lambda s: None)
    assert tool.main(["--run_name", "R", "--resume_run", "R", "--chunks", "1", *argv]) == 0
    return cmds[0]


@pytest.mark.parametrize("argv,want,absent", [
    ([], [], ["--hw_regularizers", "--no-hw_regularizers", "--stop_model", "--no-stop_model", "--allow_settings_change"]),
    (["--hw_regularizers", "--stop_model"], ["--hw_regularizers", "--stop_model"], ["--no-hw_regularizers"]),
    (["--no-hw_regularizers", "--no-stop_model", "--allow_settings_change"],
     ["--no-hw_regularizers", "--no-stop_model", "--allow_settings_change"], ["--hw_regularizers", "--stop_model"]),
], ids=["unset_inherits", "on", "off"])
def test_run_chunked_training_forwards_tristate(monkeypatch, tmp_path, argv, want, absent):
    cmd = _drive_tracking(monkeypatch, tmp_path, argv)
    assert "--continue_run" in cmd
    for w in want:
        assert w in cmd, (w, cmd)
    for a in absent:
        assert a not in cmd, (a, cmd)


LOCO_BOOLS = ["pushes", "symmetry", "stop_model", "gait_shaping", "gait_v2", "gait_v3"]


def test_locomotion_driver_bools_have_a_no_form():
    acts = _actions(_REPO / "tools/run_chunked_locomotion.py")
    for k in LOCO_BOOLS:
        assert acts[k] == "argparse.BooleanOptionalAction", (k, acts[k])
    assert not [k for k, a in acts.items() if a == "'store_true'" and k not in ("allow_settings_change",)], acts


def test_locomotion_resume_can_switch_symmetry_off(monkeypatch, tmp_path):
    info = _record(["--symmetry", "--gait_v3", "--pushes"])
    _, cmds, _ = _drive(monkeypatch, tmp_path, ["--chunks", "1"], info=info)
    assert "--symmetry" in cmds[0] and "--gait_v3" in cmds[0] and "--pushes" in cmds[0]  # inherited
    _, cmds, _ = _drive(monkeypatch, tmp_path / "b", ["--chunks", "1", "--no-symmetry", "--no-pushes",
                                                       "--allow_settings_change"], info=info)
    assert "--symmetry" not in cmds[0] and "--pushes" not in cmds[0] and "--gait_v3" in cmds[0]
    assert "--allow_settings_change" in cmds[0]
