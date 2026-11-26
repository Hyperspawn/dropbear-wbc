"""Item 395: tools/policy_runner.py records a ``default_pose`` block in ``summary.config``.

Before item 395 the runner logged only ``default_pose_sdk``; a run summary did not say whether hold mode used the
legacy pose, the repo calibration or which calibration path (``--hold-pose loaded`` still uses the ``--default-pose``
choice for the sha / mirror calibration). The low-level example (item 377) and tools/probe_newton_plant.py (item 393)
record ``deploy.config.default_pose_record``; the runner now does the same in hold mode (``used`` true) and records
``source "config default_joint_pos"`` / ``used`` false in policy mode, where ``--default-pose`` is not used (a
non-legacy choice prints a NOTE). Warn only: nothing new refuses.
"""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "tools"))

import policy_runner as pr  # noqa: E402
from dropbear_wbc.deploy import config as dc  # noqa: E402
from dropbear_wbc.sdk import motors  # noqa: E402

CALIB = ROOT / "data/calibration/dropbear_semantic_calibration.json"
C = json.loads(CALIB.read_text(encoding="utf-8"))
STAND = np.asarray(C["standing_motor_pos"], float)
LEGACY = np.asarray(motors.DEFAULT_POS, float)
JOINTS = list(motors.MOTOR_NAMES[:6])
HIPS = {"left_hip_roll", "right_hip_roll"}


def _calib_x(tmp_path: Path, pose, name: str = "calib_x.json") -> Path:
    data = dict(C)
    data["standing_motor_pos"] = [float(v) for v in pose]
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _hold(*argv):
    cfg, _ = pr.build(pr.parse_args(["--mode", "hold", *argv]))
    return cfg


# --- hold mode: the block -----------------------------------------------------------------------------------------

def test_hold_legacy_block():
    rec = _hold("--hold-pose", "default").meta["default_pose"]
    assert rec["choice"] == "legacy" and rec["source"] == "legacy" and rec["used"] is True
    assert rec["calibration"] == CALIB.as_posix()
    assert np.array_equal(rec["default_pose_sdk"], LEGACY)
    m = rec["mirror"]
    assert m["asymmetric"] is True and set(m["over_threshold_deg"]) == HIPS
    assert m["max_err_deg"] == pytest.approx(11.456, abs=1e-3) and m["default_pose_source"] == "legacy"


def test_hold_calibration_block():
    rec = _hold("--default-pose", "calibration", "--hold-pose", "default").meta["default_pose"]
    assert (rec["choice"], rec["source"], rec["used"]) == ("calibration", "calibration", True)
    assert rec["calibration"] == CALIB.as_posix()
    assert np.allclose(rec["default_pose_sdk"], STAND)
    assert rec["mirror"]["asymmetric"] is False and rec["mirror"]["max_err_deg"] == pytest.approx(0.045, abs=1e-3)


def test_hold_path_block(tmp_path):
    x = _calib_x(tmp_path, STAND)
    rec = _hold("--default-pose", str(x), "--hold-pose", "default").meta["default_pose"]
    assert (rec["choice"], rec["source"]) == (str(x), "path")
    assert rec["calibration"] == x.as_posix() and rec["mirror"]["calibration"] == x.as_posix()


def test_hold_block_pose_is_the_config_pose(tmp_path):
    for argv in ([], ["--default-pose", "calibration"], ["--default-pose", str(_calib_x(tmp_path, STAND + 0.01))]):
        cfg = _hold(*argv, "--hold-pose", "default")
        assert np.array_equal(cfg.meta["default_pose"]["default_pose_sdk"], cfg.default_pose_sdk)


def test_hold_block_with_loaded_stand_still_names_the_choice():
    """--hold-pose loaded replaces the hold target, not the default pose: the block still says which one is used."""
    cfg = _hold("--default-pose", "calibration", "--hold-pose", "loaded")
    assert cfg.hold_pose_sdk is not None
    assert cfg.meta["default_pose"]["source"] == "calibration"
    assert cfg.meta["hold_pose_check"]["calibration"] == cfg.meta["default_pose"]["calibration"]


def test_hold_block_is_json_safe():
    rec = _hold("--hold-pose", "default").meta["default_pose"]
    assert json.loads(json.dumps(rec)) == rec


def test_hold_mirror_warns_only(capsys):
    _hold("--hold-pose", "default")
    out = capsys.readouterr().out
    assert "[runner] WARNING: default pose (legacy) is not left/right symmetric" in out
    assert "[runner] default pose: legacy (sdk motors.DEFAULT_POS)" in out
    _hold("--default-pose", "calibration", "--hold-pose", "default")
    assert "WARNING: default pose" not in capsys.readouterr().out


def test_hold_missing_repo_calibration_is_a_note(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pr, "CALIBRATION_JSON", tmp_path / "missing.json")
    cfg = _hold("--hold-pose", "default")
    assert cfg.meta["default_pose"]["mirror"] is None and cfg.meta["default_pose"]["source"] == "legacy"
    assert "NOTE: default pose mirror check skipped" in capsys.readouterr().out


def test_hold_uses_shared_record(monkeypatch):
    seen = []
    real = pr.default_pose_record

    def spy(choice, tag="[deploy]", repo=None):
        seen.append((choice, tag, repo))
        return real(choice, tag, repo)

    monkeypatch.setattr(pr, "default_pose_record", spy)
    _hold("--default-pose", "calibration", "--hold-pose", "default")
    assert seen == [("calibration", "[runner]", pr.CALIBRATION_JSON)]


def test_hold_errors_unchanged(tmp_path):
    with pytest.raises(FileNotFoundError):
        _hold("--default-pose", str(tmp_path / "nope.json"), "--hold-pose", "default")
    with pytest.raises(ValueError, match="not finite"):
        _hold("--default-pose", str(_calib_x(tmp_path, np.full(22, np.nan))), "--hold-pose", "default")


# --- policy mode --------------------------------------------------------------------------------------------------

class _FakePolicy:
    action_dim = None
    metadata: dict = {}
    has_reference = False

    def __init__(self, *a, **kw):
        pass


@pytest.fixture
def fake_policy(monkeypatch):
    from dropbear_wbc.deploy import policy as dpol

    monkeypatch.setattr(dpol, "OnnxPolicy", _FakePolicy)


def _yaml(d: Path, default=None) -> Path:
    doc = {"joint_names": JOINTS, "policy": "p.onnx"}
    if default is not None:
        doc["default_joint_pos"] = default
    p = d / "deploy.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def test_policy_block_says_config(tmp_path, fake_policy, capsys):
    cfg, _ = pr.build(pr.parse_args(["--mode", "policy", "--deploy-yaml", str(_yaml(tmp_path))]))
    rec = cfg.meta["default_pose"]
    assert rec["source"] == "config default_joint_pos" and rec["used"] is False and rec["choice"] == "legacy"
    assert rec["calibration"] is None and rec["mirror"] is None
    assert np.array_equal(rec["default_pose_sdk"], cfg.default_pose_sdk)
    assert "ignored in policy mode" not in capsys.readouterr().out


def test_policy_block_names_config_calibration(tmp_path, fake_policy):
    other = tmp_path / "other_calib.json"
    other.write_text(CALIB.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    import hashlib
    rec = json.loads((ROOT / "data/calibration/dropbear_loaded_stand.json").read_text(encoding="utf-8"))
    rec["calibration_sha256"] = hashlib.sha256(other.read_bytes()).hexdigest()
    stand = tmp_path / "stand.json"
    stand.write_text(json.dumps(rec), encoding="utf-8")
    cfg, _ = pr.build(pr.parse_args(["--mode", "policy", "--deploy-yaml",
                                     str(_yaml(tmp_path, {"calibration": other.name})), "--hold-pose", str(stand)]))
    assert cfg.meta["default_pose"]["calibration"] == cfg.stand_calibration.as_posix()


def test_policy_ignored_choice_is_a_note(tmp_path, fake_policy, capsys):
    cfg, _ = pr.build(pr.parse_args(["--mode", "policy", "--deploy-yaml", str(_yaml(tmp_path)),
                                     "--default-pose", "calibration"]))
    assert cfg.meta["default_pose"]["choice"] == "calibration" and cfg.meta["default_pose"]["used"] is False
    assert "NOTE: --default-pose calibration ignored in policy mode" in capsys.readouterr().out


# --- summary.config -------------------------------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.parametrize("choice,source", [("legacy", "legacy"), ("calibration", "calibration")])
def test_summary_config_has_default_pose(tmp_path, choice, source):
    """main() writes summary.config.default_pose even when no LowState arrives (failed run, summary still written)."""
    out = tmp_path / "summary.json"
    pr.main(["--mode", "hold", "--hold-pose", "default", "--default-pose", choice, "--host", "127.0.0.1",
             "--cmd-port", str(_free_port()), "--state-port", str(_free_port()), "--connect-timeout-s", "0.2",
             "--summary", str(out)])
    summary = json.loads(out.read_text(encoding="utf-8"))
    rec = summary["config"]["default_pose"]
    assert rec["source"] == source and rec["used"] is True
    assert rec["default_pose_sdk"] == summary["config"]["default_pose_sdk"]
    assert rec["mirror"]["asymmetric"] is (choice == "legacy")
