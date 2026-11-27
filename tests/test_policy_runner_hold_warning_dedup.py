"""Item 396: tools/policy_runner.py prints one WARNING, not two, for the default pose asymmetry in hold mode.

Since item 395 hold mode records ``deploy.config.default_pose_record`` (prints ``WARNING: default pose (legacy) is not
left/right symmetric``). With ``--hold-pose default`` the hold target is that same pose, so ``hold_target_symmetry``
printed a second WARNING for the same 11.456 deg hip-roll asymmetry. Now the second one is a NOTE when the hold target
is the default pose (no ``--hold-pose`` override), judged against the same calibration, and that pose was already
warned about. Both reports stay in the summary; ``hold_target_symmetry`` gets ``warning_duplicate_of``. Any other hold
target (loaded stand, another calibration, policy mode where the default pose is not mirror-checked) still warns.
"""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

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
HOLD_WARN = "WARNING: hold target is not left/right symmetric"
POSE_WARN = "WARNING: default pose"
DUP_NOTE = "NOTE: hold target = default pose"


def _calib_x(tmp_path: Path, pose, name: str = "calib_x.json") -> Path:
    data = dict(C)
    data["standing_motor_pos"] = [float(v) for v in pose]
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _hold(*argv):
    cfg, _ = pr.build(pr.parse_args(["--mode", "hold", *argv]))
    return cfg


def _mirror_warnings(out: str) -> int:
    return sum(1 for ln in out.splitlines() if "WARNING" in ln and "not left/right symmetric" in ln)


# --- build(): hold mode ----------------------------------------------------------------------------------------------

def test_legacy_default_hold_warns_once(capsys):
    cfg = _hold("--default-pose", "legacy", "--hold-pose", "default")
    out = capsys.readouterr().out
    assert _mirror_warnings(out) == 1
    assert "WARNING: default pose (legacy) is not left/right symmetric" in out and HOLD_WARN not in out
    assert DUP_NOTE in out and "11.5 deg" in out.split(DUP_NOTE)[1].splitlines()[0]
    assert cfg.meta["hold_target_symmetry"]["warning_duplicate_of"] == "default_pose"


def test_legacy_default_hold_keeps_both_reports():
    cfg = _hold("--hold-pose", "default")
    pose_m, hold_m = cfg.meta["default_pose"]["mirror"], cfg.meta["hold_target_symmetry"]
    assert pose_m["asymmetric"] is True and hold_m["asymmetric"] is True
    assert pose_m["max_err_deg"] == pytest.approx(11.456, abs=1e-3) == hold_m["max_err_deg"]
    assert hold_m["over_threshold_deg"] == pose_m["over_threshold_deg"]
    assert hold_m["calibration"] == pose_m["calibration"] == CALIB.as_posix()
    assert hold_m["warning_duplicate_of"] == "default_pose" and "warning_duplicate_of" not in pose_m


def test_asymmetric_path_default_hold_warns_once(tmp_path, capsys):
    x = _calib_x(tmp_path, STAND + 0.03)
    cfg = _hold("--default-pose", str(x), "--hold-pose", "default")
    out = capsys.readouterr().out
    assert _mirror_warnings(out) == 1 and "WARNING: default pose (path)" in out and DUP_NOTE + " (path)" in out
    rep = cfg.meta["hold_target_symmetry"]
    assert rep["asymmetric"] is True and rep["calibration"] == x.as_posix()
    assert rep["max_err_deg"] == pytest.approx(3.658, abs=1e-3)
    assert rep["warning_duplicate_of"] == "default_pose"


def test_calibration_default_hold_no_warning_no_note(capsys):
    cfg = _hold("--default-pose", "calibration", "--hold-pose", "default")
    out = capsys.readouterr().out
    assert _mirror_warnings(out) == 0 and DUP_NOTE not in out
    assert "warning_duplicate_of" not in cfg.meta["hold_target_symmetry"]


def test_loaded_stand_hold_unchanged(capsys):
    cfg = _hold("--hold-pose", "loaded")  # legacy default pose, symmetric loaded stand as the hold target
    out = capsys.readouterr().out
    assert _mirror_warnings(out) == 1 and POSE_WARN in out and HOLD_WARN not in out and DUP_NOTE not in out
    assert cfg.meta["hold_target_symmetry"]["asymmetric"] is False
    assert "warning_duplicate_of" not in cfg.meta["hold_target_symmetry"]


def test_missing_calibration_still_notes(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pr, "CALIBRATION_JSON", tmp_path / "missing.json")
    cfg = _hold("--hold-pose", "default")
    out = capsys.readouterr().out
    assert cfg.meta["hold_target_symmetry"] is None and _mirror_warnings(out) == 0 and DUP_NOTE not in out


# --- hold_target_symmetry(): when the WARNING stays ------------------------------------------------------------------

def _legacy_block():
    _, rec = dc.default_pose_record("legacy", "[t]", CALIB)
    return rec


def test_symmetry_suppresses_only_with_matching_block(capsys):
    cfg = dc.hold_config(LEGACY, None)
    rep = pr.hold_target_symmetry(cfg, CALIB, "--default-pose", _legacy_block())
    out = capsys.readouterr().out
    assert HOLD_WARN not in out and DUP_NOTE in out and rep["warning_duplicate_of"] == "default_pose"
    rep = pr.hold_target_symmetry(cfg, CALIB, "--default-pose")  # no block -> old behaviour
    out = capsys.readouterr().out
    assert HOLD_WARN in out and DUP_NOTE not in out and "warning_duplicate_of" not in rep


def test_symmetry_warns_for_override_hold_target(capsys):
    """--hold-pose override equal to the legacy pose: the hold target is not 'the default pose' -> still warns."""
    cfg = dc.hold_config(LEGACY, LEGACY.copy())
    rep = pr.hold_target_symmetry(cfg, CALIB, "--default-pose", _legacy_block())
    assert HOLD_WARN in capsys.readouterr().out and "warning_duplicate_of" not in rep


def test_symmetry_warns_for_other_calibration(tmp_path, capsys):
    x = _calib_x(tmp_path, STAND)
    cfg = dc.hold_config(LEGACY, None)
    rep = pr.hold_target_symmetry(cfg, x, "--default-pose", _legacy_block())
    assert HOLD_WARN in capsys.readouterr().out and "warning_duplicate_of" not in rep


def test_symmetry_warns_for_other_pose(capsys):
    block = _legacy_block()
    cfg = dc.hold_config(LEGACY + 0.01, None)
    rep = pr.hold_target_symmetry(cfg, CALIB, "--default-pose", block)
    assert HOLD_WARN in capsys.readouterr().out and "warning_duplicate_of" not in rep


@pytest.mark.parametrize("mirror", [None, "symmetric"])
def test_symmetry_warns_when_block_did_not_warn(capsys, mirror):
    """Policy-mode block (mirror None) or a block whose mirror was symmetric: nothing was warned -> warn here."""
    block = _legacy_block()
    block["mirror"] = None if mirror is None else {**block["mirror"], "asymmetric": False}
    cfg = dc.hold_config(LEGACY, None)
    rep = pr.hold_target_symmetry(cfg, CALIB, "config default_joint_pos", block)
    assert HOLD_WARN in capsys.readouterr().out and "warning_duplicate_of" not in rep


def test_symmetric_hold_target_never_notes(capsys):
    """Nothing to report for a symmetric hold target, whatever the block claims: no WARNING and no NOTE."""
    _, block = dc.default_pose_record("calibration", "[t]", CALIB)
    block["mirror"] = {**block["mirror"], "asymmetric": True}
    rep = pr.hold_target_symmetry(dc.hold_config(STAND, None), CALIB, "--default-pose", block)
    out = capsys.readouterr().out
    assert rep["asymmetric"] is False and HOLD_WARN not in out and DUP_NOTE not in out
    assert "warning_duplicate_of" not in rep


def test_already_warned_predicate():
    block = _legacy_block()
    cfg = dc.hold_config(LEGACY, None)
    assert pr.default_pose_already_warned(cfg, CALIB, block) is True
    assert pr.default_pose_already_warned(cfg, CALIB, None) is False
    assert pr.default_pose_already_warned(cfg, CALIB, {**block, "calibration": None}) is False
    assert pr.default_pose_already_warned(dc.hold_config(LEGACY, STAND), CALIB, block) is False


# --- summary ---------------------------------------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_summary_keeps_both_reports(tmp_path, capsys):
    out = tmp_path / "summary.json"
    pr.main(["--mode", "hold", "--hold-pose", "default", "--host", "127.0.0.1", "--cmd-port", str(_free_port()),
             "--state-port", str(_free_port()), "--connect-timeout-s", "0.2", "--summary", str(out)])
    assert _mirror_warnings(capsys.readouterr().out) == 1
    c = json.loads(out.read_text(encoding="utf-8"))["config"]
    assert c["default_pose"]["mirror"]["asymmetric"] is True
    assert c["hold_target_symmetry"]["asymmetric"] is True
    assert c["hold_target_symmetry"]["warning_duplicate_of"] == "default_pose"
    assert c["hold_target_symmetry"]["max_err_deg"] == c["default_pose"]["mirror"]["max_err_deg"]
