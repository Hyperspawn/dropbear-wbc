"""Item 387: ``tools/hw_stand_test.py --motor_pose_json`` sha-checks a loaded stand against the repo calibration.

Before item 387 the tool read ``--motor_pose_key`` (default ``loaded_standing_motor_pos``) of the JSON with no check,
so a loaded stand derived from another calibration silently set the sim init pose. Now that key goes through
``deploy.config.load_motor_pose_json`` -> ``load_loaded_stand_pose(path, calib)``: mismatch / no sha / missing
calibration refuse unless ``--allow_loaded_stand_mismatch`` (WARNING). Other keys are read unchecked, as before. The tool
itself needs Isaac Lab; it is imported here with a stub ``isaaclab.app`` (parser + ``read_hold_pose``) and its
``main`` wiring is checked by AST. CPU only."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from dropbear_wbc.deploy import config as dc  # noqa: E402
from dropbear_wbc.sdk import motors  # noqa: E402

TOOL = ROOT / "tools/hw_stand_test.py"
CALIB = ROOT / "data/calibration/dropbear_semantic_calibration.json"
LOADED = ROOT / "data/calibration/dropbear_loaded_stand.json"
REC = json.loads(LOADED.read_text(encoding="utf-8"))
Q_LOADED = [float(x) for x in REC["loaded_standing_motor_pos"]]
KEY = "loaded_standing_motor_pos"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write(path: Path, sha: str | None = "keep", shift: float = 0.0, extra: dict | None = None) -> Path:
    rec = dict(REC)
    if sha is None:
        rec.pop("calibration_sha256")
    elif sha != "keep":
        rec["calibration_sha256"] = sha
    rec[KEY] = [q + shift for q in Q_LOADED]
    rec.update(extra or {})
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


def _other_calib(d: Path) -> Path:
    p = d / "other_calib.json"
    p.write_text(CALIB.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    return p


def _tool(monkeypatch, argv: list[str]):
    """Import tools/hw_stand_test.py with a stub Isaac Lab app launcher and the given argv."""
    app_mod = types.ModuleType("isaaclab.app")

    class AppLauncher:
        def __init__(self, a):
            self.app = None

        @staticmethod
        def add_app_launcher_args(p):
            p.add_argument("--headless", action="store_true")

    app_mod.AppLauncher = AppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_mod)
    monkeypatch.setattr(sys, "argv", ["hw_stand_test.py", "--out", "x.json", *argv])
    spec = importlib.util.spec_from_file_location("hw_stand_test_387", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- deploy.config.load_motor_pose_json -------------------------------------------------------------------------

def test_repo_loaded_stand_accepted():
    pose, chk = dc.load_motor_pose_json(LOADED)
    assert pose == Q_LOADED
    assert chk == {"motor_pose_json": LOADED.as_posix(), "key": KEY, "checked": True,
                   "calibration": Path(dc.hold_pose_calibration(None, LOADED.parent)).as_posix(),
                   "sha_match": True, "allowed_mismatch": False}
    assert chk["calibration"].endswith("data/calibration/dropbear_semantic_calibration.json")


def test_default_key_is_loaded_stand(tmp_path):
    p = _write(tmp_path / "ls.json", sha="0" * 64)
    with pytest.raises(dc.LoadedStandCalibrationMismatch):
        dc.load_motor_pose_json(p)


def test_stale_sha_refused(tmp_path):
    p = _write(tmp_path / "ls.json", sha="0" * 64)
    with pytest.raises(dc.LoadedStandCalibrationMismatch, match="--allow_loaded_stand_mismatch"):
        dc.load_motor_pose_json(p, KEY)


def test_record_without_sha_refused(tmp_path):
    p = _write(tmp_path / "ls.json", sha=None)
    with pytest.raises(dc.LoadedStandCalibrationMismatch):
        dc.load_motor_pose_json(p, KEY)


def test_checked_against_given_calibration(tmp_path):
    other = _other_calib(tmp_path)
    with pytest.raises(dc.LoadedStandCalibrationMismatch):
        dc.load_motor_pose_json(LOADED, KEY, calibration_path=other)
    p = _write(tmp_path / "ls.json", sha=_sha(other), shift=0.01)
    pose, chk = dc.load_motor_pose_json(p, KEY, calibration_path=other)
    assert np.allclose(pose, np.asarray(Q_LOADED) + 0.01)
    assert chk["sha_match"] is True and chk["calibration"] == other.as_posix()


def test_missing_calibration_refused_unless_allowed(tmp_path, capsys):
    gone = tmp_path / "nope.json"
    with pytest.raises(dc.LoadedStandCalibrationMismatch, match="not found"):
        dc.load_motor_pose_json(LOADED, KEY, calibration_path=gone)
    pose, chk = dc.load_motor_pose_json(LOADED, KEY, calibration_path=gone, allow_mismatch=True)
    assert pose == Q_LOADED and chk["sha_match"] is False and chk["allowed_mismatch"] is True
    assert "WARNING" in capsys.readouterr().out


@pytest.mark.parametrize("sha", ["0" * 64, None])
def test_allow_loads_with_warning(tmp_path, capsys, sha):
    p = _write(tmp_path / "ls.json", sha=sha, shift=0.02)
    pose, chk = dc.load_motor_pose_json(p, KEY, allow_mismatch=True)
    assert np.allclose(pose, np.asarray(Q_LOADED) + 0.02)
    assert chk["checked"] is True and chk["sha_match"] is False and chk["allowed_mismatch"] is True
    assert "WARNING" in capsys.readouterr().out


def test_allow_is_noop_on_matching_input(capsys):
    a = dc.load_motor_pose_json(LOADED, KEY)
    b = dc.load_motor_pose_json(LOADED, KEY, allow_mismatch=True)
    assert a == b
    assert "WARNING" not in capsys.readouterr().out


def test_other_key_read_unchecked_as_before(tmp_path):
    q = [0.1 * i for i in range(motors.NUM_MOTORS)]
    p = _write(tmp_path / "ls.json", sha="0" * 64, extra={"my_pose": q})
    pose, chk = dc.load_motor_pose_json(p, "my_pose")
    assert pose == q
    assert chk == {"motor_pose_json": p.as_posix(), "key": "my_pose", "checked": False, "calibration": None,
                   "sha_match": None, "allowed_mismatch": False}
    plain = tmp_path / "plain.json"  # no schema, no sha: other keys never needed one
    plain.write_text(json.dumps({"pose": q}), encoding="utf-8")
    assert dc.load_motor_pose_json(plain, "pose")[0] == q


def test_other_key_wrong_length_rejected(tmp_path):
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"pose": [0.0] * 21}), encoding="utf-8")
    with pytest.raises(ValueError, match="not 22"):
        dc.load_motor_pose_json(plain, "pose")


def test_loaded_key_needs_loaded_stand_schema(tmp_path):
    p = tmp_path / "bare.json"
    p.write_text(json.dumps({KEY: Q_LOADED}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        dc.load_motor_pose_json(p, KEY, allow_mismatch=True)


def test_mismatch_is_subclass_not_other_errors(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": "x", KEY: Q_LOADED}), encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        dc.load_motor_pose_json(p, KEY)
    assert not isinstance(ei.value, dc.LoadedStandCalibrationMismatch)


# ---- tools/hw_stand_test.py ---------------------------------------------------------------------------------------

def test_tool_flag_default_off(monkeypatch):
    mod = _tool(monkeypatch, [])
    assert mod.args.allow_loaded_stand_mismatch is False
    assert mod.args.motor_pose_key == KEY and mod.args.motor_pose_json is None


@pytest.mark.parametrize("flag", ["--allow_loaded_stand_mismatch", "--allow-loaded-stand-mismatch"])
def test_tool_flag_spellings(monkeypatch, flag):
    assert _tool(monkeypatch, [flag]).args.allow_loaded_stand_mismatch is True


def test_tool_read_hold_pose_refuses_stale(monkeypatch, tmp_path):
    p = _write(tmp_path / "ls.json", sha="0" * 64)
    mod = _tool(monkeypatch, ["--motor_pose_json", str(p)])
    with pytest.raises(dc.LoadedStandCalibrationMismatch):
        mod.read_hold_pose(mod.args)


def test_tool_read_hold_pose_flag_allows(monkeypatch, tmp_path, capsys):
    p = _write(tmp_path / "ls.json", sha="0" * 64)
    mod = _tool(monkeypatch, ["--motor_pose_json", str(p), "--allow_loaded_stand_mismatch"])
    pose, chk = mod.read_hold_pose(mod.args)
    assert pose == Q_LOADED and chk["sha_match"] is False and chk["allowed_mismatch"] is True
    assert "WARNING" in capsys.readouterr().out


def test_tool_read_hold_pose_repo_and_other_key(monkeypatch, tmp_path):
    mod = _tool(monkeypatch, ["--motor_pose_json", str(LOADED)])
    pose, chk = mod.read_hold_pose(mod.args)
    assert pose == Q_LOADED and chk["checked"] is True and chk["sha_match"] is True
    q = [0.0] * motors.NUM_MOTORS
    p = _write(tmp_path / "ls.json", sha="0" * 64, extra={"my_pose": q})
    mod = _tool(monkeypatch, ["--motor_pose_json", str(p), "--motor_pose_key", "my_pose"])
    pose, chk = mod.read_hold_pose(mod.args)
    assert pose == q and chk["checked"] is False


def _main_ast() -> ast.FunctionDef:
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def test_tool_main_uses_checked_reader():
    main = _main_ast()
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    assert any(isinstance(c.func, ast.Name) and c.func.id == "read_hold_pose" for c in calls)
    loads = [c for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == "loads"]
    assert not loads, "main must not read --motor_pose_json itself (unchecked)"


def test_tool_output_records_check():
    main = _main_ast()
    for n in ast.walk(main):
        if isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == "tool" for k in n.keys):
            entries = {k.value: v for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant)}
            assert isinstance(entries.get("motor_pose_check"), ast.Name)
            assert entries["motor_pose_check"].id == "pose_check"
            return
    pytest.fail("output dict not found")
