"""Item 394: tools/policy_runner.py ``--default-pose`` goes through the shared ``deploy.config`` parser.

Before item 394 ``policy_runner.default_pose_calibration`` re-parsed legacy / calibration / <path> itself and the hold
mode read ``standing_motor_pos`` with its own ``load_standing_pose`` call, while the low-level example (item 377) and
tools/probe_newton_plant.py (item 393) used ``deploy.config.resolve_default_pose``. Now one parser
(``deploy.config.default_pose_calibration_path``, no file I/O) feeds ``resolve_default_pose`` and the runner.

Runner-visible behaviour is unchanged (same paths, same poses, same errors for a missing file / wrong shape / missing
key, ``CALIBRATION_JSON`` still the repo file and still patchable) except one intentional difference: a non-finite
``standing_motor_pos`` now refuses (ValueError) instead of becoming a NaN MoveToDefault/Hold target.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "tools"))

import policy_runner as pr  # noqa: E402
from dropbear_wbc.deploy import config as dc  # noqa: E402
from dropbear_wbc.kinematics.semantic import DEFAULT_CALIBRATION  # noqa: E402
from dropbear_wbc.sdk import motors  # noqa: E402

CALIB = ROOT / "data/calibration/dropbear_semantic_calibration.json"
C = json.loads(CALIB.read_text(encoding="utf-8"))
STAND = np.asarray(C["standing_motor_pos"], float)
LEGACY = np.asarray(motors.DEFAULT_POS, float)


def _calib_x(tmp_path: Path, pose, name: str = "calib_x.json") -> Path:
    data = dict(C)
    data["standing_motor_pos"] = [float(v) for v in pose]
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _build(*argv):
    cfg, _ = pr.build(pr.parse_args(["--mode", "hold", "--hold-pose", "default", *argv]))
    return cfg


def _example():
    spec = importlib.util.spec_from_file_location("example_394", ROOT / "scripts/examples/dropbear_low_level_example.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the shared parser -----------------------------------------------------------------------------------------------

def test_shared_parser_contract(tmp_path):
    repo = tmp_path / "repo.json"  # never read
    assert dc.default_pose_calibration_path("legacy", repo) == (repo, "legacy")
    assert dc.default_pose_calibration_path(None, repo) == (repo, "legacy")
    assert dc.default_pose_calibration_path("calibration", repo) == (repo, "calibration")
    other = tmp_path / "nope" / "other.json"
    assert dc.default_pose_calibration_path(str(other), repo) == (other, "path")
    assert dc.default_pose_calibration_path("calibration") == (DEFAULT_CALIBRATION, "calibration")
    assert dc.default_pose_calibration_path() == (DEFAULT_CALIBRATION, "legacy")
    assert isinstance(dc.default_pose_calibration_path("calibration", str(repo))[0], Path)


def test_resolve_default_pose_uses_parser(monkeypatch, tmp_path):
    x = _calib_x(tmp_path, STAND + 0.01)
    seen = []
    real = dc.default_pose_calibration_path

    def spy(choice, repo=None):
        seen.append(choice)
        return real(choice, repo)

    monkeypatch.setattr(dc, "default_pose_calibration_path", spy)
    pose, calib, src = dc.resolve_default_pose(str(x), CALIB)
    assert seen == [str(x)] and calib == x and src == "path" and np.allclose(pose, STAND + 0.01)


# --- runner routes through it ----------------------------------------------------------------------------------------

def test_runner_imports_shared_helpers():
    assert pr.default_pose_calibration_path is dc.default_pose_calibration_path
    assert pr.resolve_default_pose is dc.resolve_default_pose
    assert pr.DEFAULT_POSE_CHOICES_HELP is dc.DEFAULT_POSE_CHOICES_HELP


def test_runner_default_pose_calibration_calls_shared_parser(monkeypatch):
    calls = []

    def spy(choice, repo=None):
        calls.append((choice, repo))
        return Path("/x/spy.json"), "path"

    monkeypatch.setattr(pr, "default_pose_calibration_path", spy)
    assert pr.default_pose_calibration(pr.parse_args(["--default-pose", "calibration"])) == Path("/x/spy.json")
    assert calls == [("calibration", pr.CALIBRATION_JSON)]


def test_build_hold_calls_shared_resolver(monkeypatch):
    calls = []
    fake = STAND + 0.02

    def spy(choice, repo=None):
        calls.append((choice, repo))
        return fake.copy(), CALIB, "calibration"

    monkeypatch.setattr(pr, "resolve_default_pose", spy)
    cfg = _build("--default-pose", "calibration")
    assert calls == [("calibration", pr.CALIBRATION_JSON)]
    assert np.array_equal(cfg.default_pose_sdk, fake)


def test_no_local_parser_left():
    tree = ast.parse((ROOT / "tools/policy_runner.py").read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "load_standing_pose" not in names
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "default_pose_calibration")
    consts = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant)}
    assert "calibration" not in consts  # the keyword is parsed in deploy.config only


def test_help_text_shared(capsys):
    with pytest.raises(SystemExit):
        pr.parse_args(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert " ".join(("hold mode: " + dc.DEFAULT_POSE_CHOICES_HELP).split()) in out


# --- behaviour preserved ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("choice", ["legacy", "calibration", "PATH"])
def test_default_pose_calibration_values(tmp_path, choice):
    x = tmp_path / "missing_is_fine.json"  # parsing does not read the file
    arg = str(x) if choice == "PATH" else choice
    got = pr.default_pose_calibration(pr.parse_args(["--default-pose", arg]))
    assert got == {"legacy": None, "calibration": pr.CALIBRATION_JSON, "PATH": x}[choice]
    assert pr.CALIBRATION_JSON == CALIB == DEFAULT_CALIBRATION
    assert pr.default_pose_calibration(object()) is None  # no attribute -> legacy


def test_hold_calibration_unread_path(tmp_path):
    x = tmp_path / "nope.json"
    a = pr.parse_args(["--mode", "hold", "--default-pose", str(x)])
    assert pr.hold_calibration(a) == (x, "--default-pose")
    assert pr.hold_calibration(pr.parse_args(["--mode", "hold"])) == (pr.CALIBRATION_JSON, "repo")


@pytest.mark.parametrize("choice", ["legacy", "calibration", "PATH"])
def test_build_pose_parity(tmp_path, choice):
    x = _calib_x(tmp_path, STAND + 0.03)
    arg = str(x) if choice == "PATH" else choice
    cfg = _build("--default-pose", arg)
    expect = {"legacy": LEGACY, "calibration": STAND, "PATH": STAND + 0.03}[choice]
    assert np.array_equal(cfg.default_pose_sdk, expect)
    assert np.array_equal(cfg.default_joint_pos, expect) and np.array_equal(cfg.action_offset, expect)
    assert np.array_equal(cfg.hold_target_sdk, expect)
    assert np.array_equal(cfg.default_pose_sdk, dc.resolve_default_pose(arg, CALIB)[0])


def test_legacy_equals_hold_config_none():
    cfg, ref = _build(), dc.hold_config(None)
    for k in ("default_pose_sdk", "default_joint_pos", "action_offset", "kp", "kd"):
        assert np.array_equal(getattr(cfg, k), getattr(ref, k))


def test_three_clis_agree(tmp_path):
    ex = _example()
    x = _calib_x(tmp_path, STAND - 0.01)
    for arg in ("legacy", "calibration", str(x)):
        runner = _build("--default-pose", arg).default_pose_sdk
        assert np.array_equal(runner, ex.resolve_default_pose(arg)[0])
        assert np.array_equal(runner, dc.default_pose_record(arg, "[t]", CALIB)[0])


def test_patched_repo_calibration_followed(monkeypatch, tmp_path):
    x = _calib_x(tmp_path, STAND + 0.04, "patched_repo.json")
    monkeypatch.setattr(pr, "CALIBRATION_JSON", x)
    assert pr.default_pose_calibration(pr.parse_args(["--default-pose", "calibration"])) == x
    assert np.allclose(_build("--default-pose", "calibration").default_pose_sdk, STAND + 0.04)
    assert np.array_equal(_build().default_pose_sdk, LEGACY)


def test_errors_preserved(tmp_path):
    with pytest.raises(FileNotFoundError):
        _build("--default-pose", str(tmp_path / "nope.json"))
    with pytest.raises(ValueError, match="has shape"):
        _build("--default-pose", str(_calib_x(tmp_path, STAND[:20])))
    bad = tmp_path / "nokey.json"
    bad.write_text(json.dumps({"schema": "x"}), encoding="utf-8")
    with pytest.raises(KeyError):
        _build("--default-pose", str(bad))


def test_nonfinite_default_pose_refused(tmp_path):
    """Intentional difference: before item 394 a NaN standing_motor_pos became the NaN hold target."""
    nan = STAND.copy()
    nan[3] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        _build("--default-pose", str(_calib_x(tmp_path, nan)))
    inf = STAND.copy()
    inf[0] = np.inf
    with pytest.raises(ValueError, match="not finite"):
        _build("--default-pose", str(_calib_x(tmp_path, inf, "inf.json")))
