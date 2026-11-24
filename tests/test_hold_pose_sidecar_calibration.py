"""Item 378: a yaml/sidecar ``hold_pose: {"loaded_stand": <path>}`` is sha-checked against the config's calibration.

Before item 378, ``deploy.config._resolve_hold_pose`` loaded the loaded stand with no check (the runner ``--hold-pose``
path is checked since item 376). The check uses ``default_joint_pos: {"calibration": <path>}`` of the same config when
given, else the repo calibration. A mismatch (or a missing calibration file) refuses unless
``allow_loaded_stand_mismatch`` / ``--allow-loaded-stand-mismatch`` is set; then it warns and loads.
"""
from __future__ import annotations

import hashlib
import json
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
LOADED = ROOT / "data/calibration/dropbear_loaded_stand.json"
REC = json.loads(LOADED.read_text(encoding="utf-8"))
Q_LOADED = np.asarray(REC["loaded_standing_motor_pos"], float)
JOINTS = list(motors.MOTOR_NAMES[:6])


def _write_loaded(path: Path, sha: str | None, shift: float = 0.0) -> Path:
    rec = dict(REC)
    if sha is None:
        rec.pop("calibration_sha256")
    else:
        rec["calibration_sha256"] = sha
    rec["loaded_standing_motor_pos"] = (Q_LOADED + shift).tolist()
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


def _other_calib(d: Path) -> Path:
    """Same content as the repo calibration, different bytes -> different sha."""
    p = d / "other_calib.json"
    p.write_text(CALIB.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    return p


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _yaml(d: Path, hold_pose, default=None) -> Path:
    doc = {"joint_names": JOINTS, "hold_pose": hold_pose}
    if default is not None:
        doc["default_joint_pos"] = default
    p = d / "deploy.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def _sidecar(d: Path, hold_pose, default=None) -> Path:
    n = len(JOINTS)
    doc = {"schema": dc.SIDECAR_SCHEMA, "joint_names": JOINTS, "joint_stiffness": [20.0] * n,
           "joint_damping": [1.0] * n, "action_scale": [0.25] * n, "hold_pose": hold_pose}
    if default is not None:
        doc["default_joint_pos"] = default
    p = d / "policy.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _load(kind, d, hold_pose, default=None, **kw):
    if kind == "yaml":
        return dc.load_deploy_yaml(_yaml(d, hold_pose, default), **kw)
    return dc.load_sidecar(_sidecar(d, hold_pose, default), **kw)


KINDS = ["yaml", "sidecar"]


@pytest.mark.parametrize("kind", KINDS)
def test_repo_loaded_stand_accepted_against_repo_calibration(tmp_path, kind):
    cfg = _load(kind, tmp_path, {"loaded_stand": str(LOADED)})
    assert np.array_equal(cfg.hold_pose_sdk, Q_LOADED)
    chk = cfg.meta["hold_pose_check"]
    assert chk == {"loaded_stand": LOADED.resolve().as_posix(), "calibration": CALIB.resolve().as_posix(),
                   "sha_match": True}


@pytest.mark.parametrize("kind", KINDS)
def test_stale_loaded_stand_refused(tmp_path, kind):
    p = _write_loaded(tmp_path / "stale.json", "0" * 64)
    with pytest.raises(dc.LoadedStandCalibrationMismatch, match="derive_loaded_stand") as ei:
        _load(kind, tmp_path, {"loaded_stand": p.name})
    assert "allow_loaded_stand_mismatch" in str(ei.value) and "hold_pose" in str(ei.value)
    assert isinstance(ei.value, ValueError)  # existing ValueError handlers still catch it


@pytest.mark.parametrize("kind", KINDS)
def test_record_without_sha_refused(tmp_path, kind):
    p = _write_loaded(tmp_path / "nosha.json", None)
    with pytest.raises(dc.LoadedStandCalibrationMismatch):
        _load(kind, tmp_path, {"loaded_stand": p.name})


@pytest.mark.parametrize("kind", KINDS)
def test_checked_against_config_calibration_not_repo(tmp_path, kind):
    """default_joint_pos {"calibration": other}: the repo loaded stand (repo sha) is refused ..."""
    other = _other_calib(tmp_path)
    with pytest.raises(dc.LoadedStandCalibrationMismatch, match="other_calib"):
        _load(kind, tmp_path, {"loaded_stand": str(LOADED)}, {"calibration": other.name})


@pytest.mark.parametrize("kind", KINDS)
def test_loaded_stand_of_config_calibration_accepted(tmp_path, kind):
    """... and a loaded stand derived from that calibration is accepted although it does not match the repo one."""
    other = _other_calib(tmp_path)
    p = _write_loaded(tmp_path / "other_loaded.json", _sha(other), shift=0.01)
    cfg = _load(kind, tmp_path, {"loaded_stand": p.name}, {"calibration": other.name})
    assert np.allclose(cfg.hold_pose_sdk, Q_LOADED + 0.01)
    assert cfg.meta["hold_pose_check"]["calibration"] == other.resolve().as_posix()
    assert cfg.meta["hold_pose_check"]["sha_match"] is True
    with pytest.raises(dc.LoadedStandCalibrationMismatch):  # the same file without the calibration -> repo -> refused
        _load(kind, tmp_path, {"loaded_stand": p.name})


@pytest.mark.parametrize("kind", KINDS)
def test_calibration_path_relative_to_config(tmp_path, kind):
    sub = tmp_path / "export"
    sub.mkdir()
    other = _other_calib(sub)
    p = _write_loaded(sub / "l.json", _sha(other))
    cfg = _load(kind, sub, {"loaded_stand": "l.json"}, {"calibration": "other_calib.json"})
    assert cfg.meta["hold_pose_check"]["calibration"] == other.resolve().as_posix()
    assert np.array_equal(cfg.hold_pose_sdk, Q_LOADED)
    assert p.exists()


@pytest.mark.parametrize("kind", KINDS)
def test_allow_flag_loads_with_warning(tmp_path, kind, capsys):
    p = _write_loaded(tmp_path / "stale.json", "0" * 64, shift=0.02)
    cfg = _load(kind, tmp_path, {"loaded_stand": p.name}, allow_loaded_stand_mismatch=True)
    assert np.allclose(cfg.hold_pose_sdk, Q_LOADED + 0.02)
    assert cfg.meta["hold_pose_check"]["sha_match"] is False
    assert "WARNING" in capsys.readouterr().out


@pytest.mark.parametrize("kind", KINDS)
def test_allow_flag_does_not_change_matching_input(tmp_path, kind, capsys):
    cfg = _load(kind, tmp_path, {"loaded_stand": str(LOADED)}, allow_loaded_stand_mismatch=True)
    assert np.array_equal(cfg.hold_pose_sdk, Q_LOADED) and cfg.meta["hold_pose_check"]["sha_match"] is True
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("kind", KINDS)
def test_missing_calibration_refused_unless_allowed(tmp_path, kind):
    hp, dflt = {"loaded_stand": str(LOADED)}, {"calibration": "missing_calib.json"}
    with pytest.raises(dc.LoadedStandCalibrationMismatch, match="not found"):
        dc._resolve_hold_pose(hp, tmp_path, tmp_path / "missing_calib.json")
    pose, chk = dc._resolve_hold_pose(hp, tmp_path, tmp_path / "missing_calib.json", allow_mismatch=True)
    assert np.array_equal(pose, Q_LOADED) and chk["sha_match"] is False
    with pytest.raises(FileNotFoundError):  # the default pose itself needs the file: unchanged behaviour
        _load(kind, tmp_path, hp, dflt)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("hp", [None, "default"])
def test_absent_or_default_hold_pose_unchanged(tmp_path, kind, hp):
    cfg = _load(kind, tmp_path, hp)
    assert cfg.hold_pose_sdk is None and "hold_pose_check" not in cfg.meta


@pytest.mark.parametrize("kind", KINDS)
def test_bad_hold_pose_value_still_rejected(tmp_path, kind):
    with pytest.raises(ValueError, match="hold_pose: expected"):
        _load(kind, tmp_path, {"loaded": str(LOADED)})


def test_hold_pose_calibration_source():
    base = Path("/x/y")
    assert dc.hold_pose_calibration({"calibration": "c.json"}, base) == Path("/x/y/c.json")
    for v in (None, "legacy", [0.0] * 6, "0,0,0"):
        assert dc.hold_pose_calibration(v, base) == CALIB.resolve()


def test_sidecar_onnx_metadata_default_uses_repo_calibration(tmp_path):
    """default_joint_pos from ONNX metadata (a list) names no calibration -> repo calibration."""
    p = _sidecar(tmp_path, {"loaded_stand": str(LOADED)})
    cfg = dc.load_sidecar(p, {"default_joint_pos": ",".join(["0.0"] * len(JOINTS))})
    assert cfg.meta["hold_pose_check"]["calibration"] == CALIB.resolve().as_posix()
    stale = _write_loaded(tmp_path / "stale.json", "0" * 64)
    with pytest.raises(dc.LoadedStandCalibrationMismatch):
        dc.load_sidecar(_sidecar(tmp_path, {"loaded_stand": stale.name}), {"default_joint_pos": "0,0,0,0,0,0"})


def test_sidecar_meta_keys_kept(tmp_path):
    p = _sidecar(tmp_path, {"loaded_stand": str(LOADED)})
    d = json.loads(p.read_text())
    d.update(task="t", run_path="r")
    p.write_text(json.dumps(d))
    cfg = dc.load_sidecar(p)
    assert cfg.meta["task"] == "t" and cfg.meta["run_path"] == "r" and "hold_pose_check" in cfg.meta


# ---- runner -------------------------------------------------------------------------------------------------------
def test_runner_flag_default_off_and_resolve(tmp_path, capsys):
    assert pr.parse_args([]).allow_loaded_stand_mismatch is False
    p = _write_loaded(tmp_path / "stale.json", "0" * 64, shift=0.03)
    for mode in ("hold", "policy"):
        with pytest.raises(dc.LoadedStandCalibrationMismatch):
            pr.resolve_hold_pose(pr.parse_args(["--mode", mode, "--hold-pose", str(p)]))
        q = pr.resolve_hold_pose(pr.parse_args(["--mode", mode, "--hold-pose", str(p),
                                                "--allow-loaded-stand-mismatch"]))
        assert np.allclose(q, Q_LOADED + 0.03)
    assert "WARNING" in capsys.readouterr().out


def test_runner_resolve_other_errors_not_swallowed(tmp_path, capsys):
    """The override covers the calibration mismatch only: a bad schema still fails at once, with no override WARNING."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "nope"}))
    with pytest.raises(ValueError, match="schema"):
        pr.resolve_hold_pose(pr.parse_args(["--hold-pose", str(bad), "--allow-loaded-stand-mismatch"]))
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("flag", [False, True])
def test_runner_passes_flag_on_onnx_metadata_reload(tmp_path, monkeypatch, flag):
    """With ONNX metadata the runner re-parses the sidecar; the second parse gets the flag too."""
    import types

    from dropbear_wbc.deploy import policy as dpol

    calls = []

    class Stop(Exception):
        pass

    def fake(*a, **kw):
        calls.append((a, kw))
        if len(calls) == 2:
            raise Stop
        return types.SimpleNamespace(policy_path=tmp_path / "p.onnx", onnx_obs_input=None, onnx_time_input=None)

    class FakePolicy:
        def __init__(self, *a, **kw):
            self.metadata = {"joint_names": "a"}

    monkeypatch.setattr(pr, "load_sidecar", fake)
    monkeypatch.setattr(dpol, "OnnxPolicy", FakePolicy)
    argv = ["--mode", "policy", "--sidecar", str(tmp_path / "x")] + (["--allow-loaded-stand-mismatch"] if flag else [])
    with pytest.raises(Stop):
        pr.build(pr.parse_args(argv))
    assert [kw for _, kw in calls] == [{"allow_loaded_stand_mismatch": flag}] * 2
    assert calls[1][0][1] == {"joint_names": "a"}


@pytest.mark.parametrize("src,flag", [("sidecar", False), ("sidecar", True), ("deploy-yaml", False),
                                      ("deploy-yaml", True)])
def test_runner_passes_flag_to_config_loaders(tmp_path, monkeypatch, src, flag):
    seen = {}

    class Stop(Exception):
        pass

    def fake(*a, **kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(pr, "load_sidecar" if src == "sidecar" else "load_deploy_yaml", fake)
    argv = ["--mode", "policy", f"--{src}", str(tmp_path / "x")] + (["--allow-loaded-stand-mismatch"] if flag else [])
    with pytest.raises(Stop):
        pr.build(pr.parse_args(argv))
    assert seen == {"allow_loaded_stand_mismatch": flag}


def test_runner_summary_carries_hold_pose_check():
    src = (ROOT / "tools/policy_runner.py").read_text(encoding="utf-8")
    assert '"hold_pose_check": cfg.meta.get("hold_pose_check")' in src
    assert 'cfg.meta["hold_pose_check"]["replaced_by"]' in src
