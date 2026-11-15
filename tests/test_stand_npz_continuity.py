"""--continue_run keeps the run's stand (reset) NPZ; --resume_run inherits it by sha256 (item 346; CPU, no Isaac)."""
from __future__ import annotations

import pytest

import sdk_test_paths  # noqa: F401  (adds source/ to sys.path)

from dropbear_wbc.tasks.locomotion.run_continuity import (  # noqa: E402
    UNRECORDED,
    check_stand_continuity,
    file_sha256,
    resolve_run_stand_npz,
)
from test_chunked_locomotion_resume import _drive, _record, _train_settings  # noqa: E402


def _npz(p, payload=b"stand-a"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload)
    return p


def _info(path, sha=None):
    return {"stand_npz": str(path), "stand_npz_sha256": sha or file_sha256(path)}


def test_same_content_other_path_passes_different_content_refused(tmp_path):
    a = _npz(tmp_path / "a" / "stand.npz")
    b = _npz(tmp_path / "b" / "copy.npz")  # same bytes, other machine / path
    c = _npz(tmp_path / "c" / "stand.npz", b"regenerated")
    info = _info(a)
    assert check_stand_continuity(info, b, continue_run=True) == []
    with pytest.raises(ValueError, match="stand NPZ .* differs"):
        check_stand_continuity(info, c, continue_run=True)
    notes = check_stand_continuity(info, c, continue_run=True, allow_change=True)
    assert len(notes) == 1 and "CHANGED" in notes[0]


def test_not_continued_or_unrecorded_is_not_refused(tmp_path):
    c = _npz(tmp_path / "c.npz", b"x")
    assert check_stand_continuity(_info(c, "0" * 64), c, continue_run=False) == []
    assert check_stand_continuity(None, c, continue_run=True) == []
    notes = check_stand_continuity({"task": "t"}, c, continue_run=True)
    assert len(notes) == 1 and UNRECORDED in notes[0]


def test_resolve_recorded_path_then_repo_data_part(tmp_path):
    rec = _npz(tmp_path / "pc" / "data" / "motions" / "smoke" / "s.npz")
    assert resolve_run_stand_npz(_info(rec), tmp_path / "repo") == (str(rec), "matched")
    # run dir copied from Windows: recorded path absent here, same file under this repo's data/
    win = {"stand_npz": r"H:\dropbear\dropbear-wbc\data\motions\smoke\s.npz", "stand_npz_sha256": file_sha256(rec)}
    local = _npz(tmp_path / "repo" / "data" / "motions" / "smoke" / "s.npz")
    assert resolve_run_stand_npz(win, tmp_path / "repo") == (str(local), "matched")
    local.write_bytes(b"regenerated")  # same name, other content: never returned
    path, note = resolve_run_stand_npz(win, tmp_path / "repo")
    assert path is None and "no local file matches" in note
    assert resolve_run_stand_npz({"task": "t"}, tmp_path)[0] is None


def test_driver_inherits_stand_npz_and_given_flag_wins(monkeypatch, tmp_path):
    s = _npz(tmp_path / "data" / "motions" / "custom_stand.npz")
    info = {**_record([]), **_info(s)}
    _, cmds, _ = _drive(monkeypatch, tmp_path / "r1", ["--chunks", "1"], info)
    a, _ = _train_settings(cmds[0], "hw_cem6")
    assert str(a.stand_npz) == str(s)
    assert check_stand_continuity(info, a.stand_npz, continue_run=True) == []
    other = _npz(tmp_path / "other.npz", b"other")
    _, cmds, _ = _drive(monkeypatch, tmp_path / "r2", ["--chunks", "1", "--stand_npz", str(other)], info)
    a, _ = _train_settings(cmds[0], "hw_cem6")
    assert str(a.stand_npz) == str(other)
    with pytest.raises(ValueError, match="--stand_npz"):
        check_stand_continuity(info, a.stand_npz, continue_run=True)


def test_driver_without_matching_file_passes_no_stand_flag(monkeypatch, tmp_path, capsys):
    info = {**_record([]), "stand_npz": "/nowhere/s.npz", "stand_npz_sha256": "f" * 64}
    _, cmds, _ = _drive(monkeypatch, tmp_path, ["--chunks", "1"], info)
    assert "--stand_npz" not in cmds[0]
    assert "stand_npz_not_inherited" in capsys.readouterr().out


def test_train_locomotion_calls_stand_check_before_env_build():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "train_locomotion.py").read_text(encoding="utf-8")
    i_check = src.index("check_stand_continuity(prev_info")
    assert i_check < src.index("gym.make(args.task") and "allow_change=args.allow_settings_change)  # item 346" in src
