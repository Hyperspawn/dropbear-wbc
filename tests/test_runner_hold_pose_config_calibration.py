"""Item 389: in policy mode the runner ``--hold-pose`` loaded stand is sha-checked against the loaded config's
calibration (``default_joint_pos: {"calibration": <path>}``), not always the repo calibration.

Before item 389, ``policy_runner.resolve_hold_pose`` used the repo calibration in policy mode, so a yaml/sidecar on
calibration X got its action offset / default pose from X and its MoveToDefault/Hold target from a loaded stand of
the repo calibration. Now ``DeployConfig.stand_calibration`` (the config-named calibration, else None -> repo) is passed
in; a mismatch refuses unless ``--allow-loaded-stand-mismatch`` (WARNING, ``sha_match`` false). Hold mode is unchanged
(``--default-pose``, else repo). The check is recorded in ``summary.config.hold_pose_check``.
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


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _other_calib(d: Path) -> Path:
    """Same content as the repo calibration, different bytes -> different sha."""
    p = d / "other_calib.json"
    p.write_text(CALIB.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    return p


def _stand(path: Path, sha: str | None, shift: float = 0.0) -> Path:
    rec = dict(REC)
    if sha is None:
        rec.pop("calibration_sha256")
    else:
        rec["calibration_sha256"] = sha
    rec["loaded_standing_motor_pos"] = (Q_LOADED + shift).tolist()
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


def _yaml(d: Path, default=None, hold_pose=None) -> Path:
    doc = {"joint_names": JOINTS, "policy": "p.onnx"}
    if default is not None:
        doc["default_joint_pos"] = default
    if hold_pose is not None:
        doc["hold_pose"] = hold_pose
    p = d / "deploy.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def _sidecar(d: Path, default=None) -> Path:
    n = len(JOINTS)
    doc = {"schema": dc.SIDECAR_SCHEMA, "joint_names": JOINTS, "joint_stiffness": [20.0] * n,
           "joint_damping": [1.0] * n, "action_scale": [0.25] * n, "policy_onnx": "p.onnx"}
    if default is not None:
        doc["default_joint_pos"] = default
    p = d / "policy.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _args(*argv):
    return pr.parse_args(["--mode", "policy", *argv])


# ---- DeployConfig.stand_calibration ------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["yaml", "sidecar"])
def test_stand_calibration_is_config_named_calibration(tmp_path, kind):
    other = _other_calib(tmp_path)
    cfg = dc.load_deploy_yaml(_yaml(tmp_path, {"calibration": other.name})) if kind == "yaml" \
        else dc.load_sidecar(_sidecar(tmp_path, {"calibration": other.name}))
    assert cfg.stand_calibration == other.resolve()


@pytest.mark.parametrize("kind", ["yaml", "sidecar"])
@pytest.mark.parametrize("default", [None, "legacy", [0.0] * 6])
def test_stand_calibration_none_without_named_calibration(tmp_path, kind, default):
    cfg = dc.load_deploy_yaml(_yaml(tmp_path, default)) if kind == "yaml" else dc.load_sidecar(_sidecar(tmp_path, default))
    assert cfg.stand_calibration is None


def test_stand_calibration_none_for_onnx_metadata_default(tmp_path):
    cfg = dc.load_sidecar(_sidecar(tmp_path), {"default_joint_pos": ",".join(["0.0"] * 6)})
    assert cfg.stand_calibration is None


def test_hold_config_has_no_stand_calibration():
    assert dc.hold_config().stand_calibration is None


def test_config_calibration_helper(tmp_path):
    assert dc.config_calibration({"calibration": "a/b.json"}, tmp_path) == (tmp_path / "a/b.json").resolve()
    for v in (None, "legacy", [0.0], "1,2"):
        assert dc.config_calibration(v, tmp_path) is None
    assert dc.hold_pose_calibration(None, tmp_path) == Path(dc.hold_pose_calibration("legacy", tmp_path))


# ---- resolve_hold_pose_checked (policy mode) --------------------------------------------------------------------
def test_policy_mode_checks_against_given_calibration(tmp_path):
    other = _other_calib(tmp_path)
    with pytest.raises(dc.LoadedStandCalibrationMismatch, match="config default_joint_pos"):
        pr.resolve_hold_pose_checked(_args(), other)  # repo loaded stand vs calibration X
    p = _stand(tmp_path / "x_stand.json", _sha(other), shift=0.02)
    pose, chk = pr.resolve_hold_pose_checked(_args("--hold-pose", str(p)), other)
    assert np.allclose(pose, Q_LOADED + 0.02)
    assert chk == {"hold_pose": f"--hold-pose {p}", "loaded_stand": p.as_posix(), "calibration": other.as_posix(),
                   "calibration_source": "config default_joint_pos", "sha_match": True, "allowed_mismatch": False}


def test_policy_mode_repo_stand_of_repo_calibration_refused_against_x(tmp_path):
    """The default --hold-pose loaded (repo stand) is refused for a config on another calibration."""
    other = _other_calib(tmp_path)
    with pytest.raises(dc.LoadedStandCalibrationMismatch, match="--allow-loaded-stand-mismatch"):
        pr.resolve_hold_pose(_args("--hold-pose", "loaded"), other)


def test_policy_mode_without_calibration_uses_repo(tmp_path):
    pose, chk = pr.resolve_hold_pose_checked(_args())
    assert np.array_equal(pose, Q_LOADED)
    assert chk["calibration"] == pr.CALIBRATION_JSON.as_posix() and chk["calibration_source"] == "repo"
    assert chk["loaded_stand"] == pr.LOADED_STAND_JSON.as_posix() and chk["sha_match"] is True
    p = _stand(tmp_path / "x_stand.json", _sha(_other_calib(tmp_path)))
    with pytest.raises(dc.LoadedStandCalibrationMismatch):
        pr.resolve_hold_pose_checked(_args("--hold-pose", str(p)))


def test_policy_mode_allow_flag_loads_with_warning(tmp_path, capsys):
    other = _other_calib(tmp_path)
    pose, chk = pr.resolve_hold_pose_checked(_args("--allow-loaded-stand-mismatch"), other)
    assert np.array_equal(pose, Q_LOADED)
    assert chk["sha_match"] is False and chk["allowed_mismatch"] is True
    assert "WARNING" in capsys.readouterr().out


def test_policy_mode_allow_flag_noop_on_match(tmp_path, capsys):
    other = _other_calib(tmp_path)
    p = _stand(tmp_path / "x_stand.json", _sha(other))
    _, chk = pr.resolve_hold_pose_checked(_args("--hold-pose", str(p), "--allow-loaded-stand-mismatch"), other)
    assert chk["sha_match"] is True and chk["allowed_mismatch"] is False
    assert "WARNING" not in capsys.readouterr().out


def test_policy_mode_no_sha_refused(tmp_path):
    other = _other_calib(tmp_path)
    p = _stand(tmp_path / "nosha.json", None)
    with pytest.raises(dc.LoadedStandCalibrationMismatch):
        pr.resolve_hold_pose_checked(_args("--hold-pose", str(p)), other)


def test_missing_calibration_refused_unless_allowed(tmp_path, capsys):
    missing = tmp_path / "missing.json"
    with pytest.raises(dc.LoadedStandCalibrationMismatch, match="not found"):
        pr.resolve_hold_pose_checked(_args(), missing)
    pose, chk = pr.resolve_hold_pose_checked(_args("--allow-loaded-stand-mismatch"), missing)
    assert np.array_equal(pose, Q_LOADED) and chk["sha_match"] is False


def test_hold_pose_default_returns_none(tmp_path):
    assert pr.resolve_hold_pose_checked(_args("--hold-pose", "default"), _other_calib(tmp_path)) == (None, None)


def test_hold_mode_ignores_calibration_path(tmp_path):
    """Hold mode keeps --default-pose / repo; a calibration_path argument is not used there."""
    other = _other_calib(tmp_path)
    pose, chk = pr.resolve_hold_pose_checked(pr.parse_args(["--mode", "hold"]), other)
    assert np.array_equal(pose, Q_LOADED) and chk["calibration_source"] == "repo"
    with pytest.raises(dc.LoadedStandCalibrationMismatch, match="--default-pose"):
        pr.resolve_hold_pose_checked(pr.parse_args(["--mode", "hold", "--default-pose", str(other)]))


def test_other_errors_not_swallowed(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "nope"}))
    with pytest.raises(ValueError, match="schema") as ei:
        pr.resolve_hold_pose_checked(_args("--hold-pose", str(bad), "--allow-loaded-stand-mismatch"), CALIB)
    assert not isinstance(ei.value, dc.LoadedStandCalibrationMismatch)
    assert "WARNING" not in capsys.readouterr().out


# ---- build(): the config's calibration reaches the check ---------------------------------------------------------
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


@pytest.mark.parametrize("src", ["deploy-yaml", "sidecar"])
def test_build_refuses_repo_stand_for_config_on_other_calibration(tmp_path, fake_policy, src):
    other = _other_calib(tmp_path)
    path = _yaml(tmp_path, {"calibration": other.name}) if src == "deploy-yaml" \
        else _sidecar(tmp_path, {"calibration": other.name})
    with pytest.raises(dc.LoadedStandCalibrationMismatch, match="config default_joint_pos"):
        pr.build(_args(f"--{src}", str(path)))


@pytest.mark.parametrize("src", ["deploy-yaml", "sidecar"])
def test_build_accepts_stand_of_config_calibration(tmp_path, fake_policy, src):
    other = _other_calib(tmp_path)
    path = _yaml(tmp_path, {"calibration": other.name}) if src == "deploy-yaml" \
        else _sidecar(tmp_path, {"calibration": other.name})
    p = _stand(tmp_path / "x_stand.json", _sha(other), shift=0.01)
    cfg, _ = pr.build(_args(f"--{src}", str(path), "--hold-pose", str(p)))
    assert np.allclose(cfg.hold_pose_sdk, Q_LOADED + 0.01)
    chk = cfg.meta["hold_pose_check"]
    assert chk["calibration"] == other.resolve().as_posix() and chk["sha_match"] is True
    assert chk["calibration_source"] == "config default_joint_pos" and chk["replaces"] is None


def test_build_allow_flag_records_mismatch(tmp_path, fake_policy, capsys):
    other = _other_calib(tmp_path)
    cfg, _ = pr.build(_args("--deploy-yaml", str(_yaml(tmp_path, {"calibration": other.name})),
                            "--allow-loaded-stand-mismatch"))
    assert np.array_equal(cfg.hold_pose_sdk, Q_LOADED)
    assert cfg.meta["hold_pose_check"]["sha_match"] is False and cfg.meta["hold_pose_check"]["allowed_mismatch"]
    assert "WARNING" in capsys.readouterr().out


def test_build_repo_config_unchanged(tmp_path, fake_policy):
    """A config without a named calibration still holds the repo loaded stand, checked vs the repo calibration."""
    cfg, _ = pr.build(_args("--deploy-yaml", str(_yaml(tmp_path))))
    assert np.array_equal(cfg.hold_pose_sdk, Q_LOADED)
    assert cfg.meta["hold_pose_check"]["calibration_source"] == "repo"


def test_build_hold_pose_default_keeps_config_check(tmp_path, fake_policy):
    """--hold-pose default: no runner check; a config hold_pose check stays, marked replaced (item 388 unchanged)."""
    other = _other_calib(tmp_path)
    p = _stand(tmp_path / "x_stand.json", _sha(other))
    cfg, _ = pr.build(_args("--deploy-yaml", str(_yaml(tmp_path, {"calibration": other.name},
                                                         {"loaded_stand": p.name})), "--hold-pose", "default"))
    assert cfg.hold_pose_sdk is None
    assert cfg.meta["hold_pose_check"]["replaced_by"] == "--hold-pose default"
    assert "calibration_source" not in cfg.meta["hold_pose_check"]


def test_build_runner_check_nests_replaced_config_check(tmp_path, fake_policy):
    other = _other_calib(tmp_path)
    p = _stand(tmp_path / "x_stand.json", _sha(other), shift=0.01)
    cfg, _ = pr.build(_args("--deploy-yaml", str(_yaml(tmp_path, {"calibration": other.name},
                                                         {"loaded_stand": p.name})), "--hold-pose", str(p)))
    chk = cfg.meta["hold_pose_check"]
    assert chk["hold_pose"] == f"--hold-pose {p}" and chk["sha_match"] is True
    assert chk["replaces"]["replaced_by"] == f"--hold-pose {p}"
    assert chk["replaces"]["calibration"] == other.resolve().as_posix()


def test_build_hold_mode_records_check():
    cfg, _ = pr.build(pr.parse_args(["--mode", "hold"]))
    assert cfg.meta["hold_pose_check"]["calibration_source"] == "repo"
    assert np.array_equal(cfg.hold_pose_sdk, Q_LOADED)


def test_summary_carries_hold_pose_check():
    src = (ROOT / "tools/policy_runner.py").read_text(encoding="utf-8")
    assert '"hold_pose_check": cfg.meta.get("hold_pose_check")' in src
