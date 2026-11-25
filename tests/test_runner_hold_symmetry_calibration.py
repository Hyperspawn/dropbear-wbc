"""Item 391: the hold-target mirror check (``policy_runner.hold_target_symmetry``, item 376) uses the same calibration
as the ``--hold-pose`` sha check (``hold_calibration``: hold mode -> ``--default-pose``; policy mode -> the loaded
config's ``default_joint_pos {"calibration"}``; else the repo calibration).

Before item 391 the mirror check always used the repo calibration, so for a config on calibration X the (warn-only)
report used the wrong semantic map. The report now carries ``calibration`` + ``calibration_source``, and
``summary.config.hold_pose_check`` (when present) records ``mirror_calibration``.
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
WRIST_SHIFT = 0.3  # rad on left_wrist_roll's offset: ~17 deg mirror error under X for any repo-symmetric pose


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _calib_x(d: Path) -> Path:
    """Calibration X: the repo file with the left wrist roll offset moved -> a different semantic map + sha."""
    c = json.loads(CALIB.read_text(encoding="utf-8"))
    c["dofs"]["left_wrist_roll"]["offset"] = float(c["dofs"]["left_wrist_roll"]["offset"]) + WRIST_SHIFT
    p = d / "calib_x.json"
    p.write_text(json.dumps(c), encoding="utf-8")
    return p


def _stand(d: Path, calib: Path) -> Path:
    rec = dict(REC, calibration_sha256=_sha(calib))
    p = d / "x_stand.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


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


def _policy(*argv):
    return pr.parse_args(["--mode", "policy", *argv])


# ---- the test calibration really changes the mirror verdict -------------------------------------------------------
def test_calib_x_changes_mirror_verdict(tmp_path):
    x = _calib_x(tmp_path)
    assert dc.pose_mirror_report(Q_LOADED, CALIB)["asymmetric"] is False
    rep = dc.pose_mirror_report(Q_LOADED, x)
    assert rep["asymmetric"] is True and rep["worst"] in ("left_wrist_roll", "right_wrist_roll")
    assert rep["max_err_deg"] == pytest.approx(np.degrees(WRIST_SHIFT), abs=0.5)


# ---- hold_calibration ---------------------------------------------------------------------------------------------
def test_hold_calibration_table(tmp_path):
    x = _calib_x(tmp_path)
    assert pr.hold_calibration(_policy(), x) == (x, "config default_joint_pos")
    assert pr.hold_calibration(_policy(), None) == (pr.CALIBRATION_JSON, "repo")
    hold = lambda *a: pr.parse_args(["--mode", "hold", *a])  # noqa: E731
    assert pr.hold_calibration(hold("--default-pose", str(x)), CALIB) == (x, "--default-pose")
    assert pr.hold_calibration(hold("--default-pose", "calibration")) == (pr.CALIBRATION_JSON, "--default-pose")
    assert pr.hold_calibration(hold(), x) == (pr.CALIBRATION_JSON, "repo")  # legacy pose; calibration_path ignored
    assert isinstance(pr.hold_calibration(_policy(), str(x))[0], Path)


def test_sha_check_and_mirror_check_share_the_calibration(tmp_path):
    x = _calib_x(tmp_path)
    _, chk = pr.resolve_hold_pose_checked(_policy("--hold-pose", str(_stand(tmp_path, x))), x)
    assert (Path(chk["calibration"]), chk["calibration_source"]) == pr.hold_calibration(_policy(), x)


# ---- hold_target_symmetry ------------------------------------------------------------------------------------------
def _cfg_holding(q):
    cfg = dc.hold_config(None, np.asarray(q, float))
    return cfg


def test_symmetry_uses_given_calibration(tmp_path, capsys):
    x = _calib_x(tmp_path)
    rep = pr.hold_target_symmetry(_cfg_holding(Q_LOADED), x, "config default_joint_pos")
    assert rep["asymmetric"] is True and rep["calibration"] == x.as_posix()
    assert rep["calibration_source"] == "config default_joint_pos"
    assert "WARNING: hold target is not left/right symmetric" in capsys.readouterr().out


def test_symmetry_default_is_repo(capsys):
    rep = pr.hold_target_symmetry(_cfg_holding(Q_LOADED))
    assert rep["asymmetric"] is False and rep["calibration_source"] == "repo"
    assert Path(rep["calibration"]) == pr.CALIBRATION_JSON
    rep = pr.hold_target_symmetry(_cfg_holding(Q_LOADED), None, "config default_joint_pos")
    assert rep["calibration_source"] == "repo"  # no path -> repo, whatever the label
    assert "WARNING" not in capsys.readouterr().out


def test_symmetry_missing_calibration_skips_with_note(tmp_path, capsys):
    assert pr.hold_target_symmetry(_cfg_holding(Q_LOADED), tmp_path / "nope.json", "--default-pose") is None
    out = capsys.readouterr().out
    assert "mirror check skipped" in out and "nope.json" in out and "--default-pose" in out


# ---- build(): policy mode -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("src", ["deploy-yaml", "sidecar"])
def test_build_policy_mirror_uses_config_calibration(tmp_path, fake_policy, src):
    x = _calib_x(tmp_path)
    path = _yaml(tmp_path, {"calibration": x.name}) if src == "deploy-yaml" else _sidecar(tmp_path, {"calibration": x.name})
    cfg, _ = pr.build(_policy(f"--{src}", str(path), "--hold-pose", str(_stand(tmp_path, x))))
    rep = cfg.meta["hold_target_symmetry"]
    assert rep["calibration"] == x.resolve().as_posix() and rep["calibration_source"] == "config default_joint_pos"
    assert rep["asymmetric"] is True  # repo map: symmetric (pre-391 verdict)
    chk = cfg.meta["hold_pose_check"]
    assert chk["mirror_calibration"] == chk["calibration"] == x.resolve().as_posix()


def test_build_policy_repo_config_unchanged(tmp_path, fake_policy):
    cfg, _ = pr.build(_policy("--deploy-yaml", str(_yaml(tmp_path))))
    rep = cfg.meta["hold_target_symmetry"]
    assert rep["calibration_source"] == "repo" and Path(rep["calibration"]) == pr.CALIBRATION_JSON
    assert rep["asymmetric"] is False
    assert cfg.meta["hold_pose_check"]["mirror_calibration"] == pr.CALIBRATION_JSON.as_posix()


def test_build_policy_hold_pose_default_uses_config_calibration(tmp_path, fake_policy):
    """--hold-pose default: no runner sha check, but the mirror check still uses the config's calibration."""
    x = _calib_x(tmp_path)
    cfg, _ = pr.build(_policy("--deploy-yaml", str(_yaml(tmp_path, {"calibration": x.name})), "--hold-pose", "default"))
    assert cfg.meta["hold_target_symmetry"]["calibration"] == x.resolve().as_posix()
    assert cfg.meta.get("hold_pose_check") is None


def test_build_policy_hold_pose_default_marks_config_check(tmp_path, fake_policy):
    """Item 388 behaviour kept (config check replaced); the replaced config check gets mirror_calibration."""
    x = _calib_x(tmp_path)
    p = _stand(tmp_path, x)
    cfg, _ = pr.build(_policy("--deploy-yaml", str(_yaml(tmp_path, {"calibration": x.name}, {"loaded_stand": p.name})),
                              "--hold-pose", "default"))
    assert cfg.hold_pose_sdk is None
    chk = cfg.meta["hold_pose_check"]
    assert chk["replaced_by"] == "--hold-pose default" and chk["mirror_calibration"] == x.resolve().as_posix()


# ---- build(): hold mode ---------------------------------------------------------------------------------------------
def test_build_hold_mode_default_pose_calibration(tmp_path):
    x = _calib_x(tmp_path)
    cfg, _ = pr.build(pr.parse_args(["--mode", "hold", "--default-pose", str(x), "--hold-pose", "default"]))
    rep = cfg.meta["hold_target_symmetry"]
    assert rep["calibration"] == x.as_posix() and rep["calibration_source"] == "--default-pose"
    assert rep["asymmetric"] is True


def test_build_hold_mode_stand_of_default_pose_calibration(tmp_path):
    x = _calib_x(tmp_path)
    cfg, _ = pr.build(pr.parse_args(["--mode", "hold", "--default-pose", str(x), "--hold-pose",
                                     str(_stand(tmp_path, x))]))
    assert cfg.meta["hold_pose_check"]["mirror_calibration"] == x.as_posix()
    assert cfg.meta["hold_target_symmetry"]["calibration_source"] == "--default-pose"


def test_build_hold_mode_legacy_uses_repo():
    cfg, _ = pr.build(pr.parse_args(["--mode", "hold"]))
    assert cfg.meta["hold_target_symmetry"]["calibration_source"] == "repo"
    assert cfg.meta["hold_pose_check"]["mirror_calibration"] == pr.CALIBRATION_JSON.as_posix()


def test_summary_keys():
    src = (ROOT / "tools/policy_runner.py").read_text(encoding="utf-8")
    assert '"hold_target_symmetry": cfg.meta.get("hold_target_symmetry")' in src
    assert '"hold_pose_check": cfg.meta.get("hold_pose_check")' in src
