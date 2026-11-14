"""Locomotion --continue_run continuity of the training flags and reward weights (item 182; CPU, no Isaac)."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pytest

import sdk_test_paths  # noqa: F401  (adds source/ to sys.path)

from dropbear_wbc.tasks.locomotion.run_continuity import (  # noqa: E402
    SETTING_FLAGS,
    UNRECORDED,
    check_reward_continuity,
    check_settings_continuity,
    effective_settings,
    locomotion_settings,
)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "train_locomotion.py"


def _parser() -> argparse.ArgumentParser:
    """The real train_locomotion argparse (its parser.add_argument calls, without the Isaac AppLauncher args)."""
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "add_argument"
             and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == "parser"]
    assert len(calls) > 40
    ns = {"parser": argparse.ArgumentParser(), "Path": Path, "_REPO": _REPO}
    exec(compile(ast.Module(body=calls, type_ignores=[]), str(_SCRIPT), "exec"), ns)
    return ns["parser"]


def _settings(argv, profile="hw_cem6"):
    return locomotion_settings(_parser().parse_args(argv), actuator_profile=profile)


_RUN = ["--gait_v3", "--terrain", "rough", "--thermal_penalty=-0.5", "--amp_weight=0.5", "--symmetry",
        "--arm_swing_weight=0.3", "--arm_abduction_weight=-0.3", "--clip_excess_legs=-1.0", "--ankle_roll_bind_deg=20"]


def _recorded(argv=_RUN, profile="hw_cem6") -> dict:
    """What run_info.json holds after a JSON round trip."""
    return json.loads(json.dumps({**_settings(argv, profile), "reward_weights": {"track": 1.0, "thermal": -0.5}}))


def test_every_setting_has_a_flag_and_the_record_covers_them():
    rec = _settings(_RUN)
    assert set(rec) == set(SETTING_FLAGS)
    flags = {a for act in _parser()._actions for a in act.option_strings}
    assert set(SETTING_FLAGS.values()) <= flags


def test_same_flags_continue_cleanly_after_json_round_trip():
    assert check_settings_continuity(_recorded(), _settings(_RUN), continue_run=True) == []


def test_chunk_resumed_without_the_flags_is_refused_naming_each_dropped_term():
    with pytest.raises(ValueError) as e:
        check_settings_continuity(_recorded(), _settings([]), continue_run=True)
    msg = str(e.value)
    for flag in ("--gait_v3", "--terrain", "--thermal_penalty", "--amp_weight", "--symmetry", "--arm_swing_weight",
                 "--clip_excess_legs", "--ankle_roll_bind_deg"):
        assert flag in msg, flag


def test_profile_change_is_refused():
    with pytest.raises(ValueError, match="--actuator_profile"):
        check_settings_continuity(_recorded(), _settings(_RUN, profile="hw_v4"), continue_run=True)


def test_allow_change_notes_changed_and_fresh_run_is_unchecked():
    notes = check_settings_continuity(_recorded(), _settings([]), continue_run=True, allow_change=True)
    assert any("CHANGED" in n and "--gait_v3" in n for n in notes)
    assert check_settings_continuity(_recorded(), _settings([]), continue_run=False) == []
    assert check_settings_continuity(None, _settings([]), continue_run=True) == []


def test_parameters_of_disabled_features_do_not_trip():
    base = _settings([])
    rec = json.loads(json.dumps(base))
    for argv in (["--arm_swing_k=0.6"], ["--amp_grad_penalty=5"], ["--symmetry_mirror_loss=0.2"],
                 ["--stair_max_m=0.2"], ["--target_clamp_margin_deg=5"]):
        assert check_settings_continuity(rec, _settings(argv), continue_run=True) == [], argv
    with pytest.raises(ValueError, match="--arm_swing_k"):  # ... but they do when the feature is on
        check_settings_continuity(_recorded(), _settings(_RUN + ["--arm_swing_k=0.6"]), continue_run=True)


def test_old_run_info_without_keys_is_noted_not_refused():
    old = {"actuator_profile": "hw_cem6", "thermal_penalty": None}
    notes = check_settings_continuity(old, _settings([], profile="hw_cem6"), continue_run=True)
    assert len(notes) == 1 and UNRECORDED in notes[0] and "gait_v3" in notes[0]
    assert effective_settings(old)["gait_v3"] == UNRECORDED


def test_reward_weights_continuity():
    prev = _recorded()
    assert check_reward_continuity(prev, {"track": 1.0, "thermal": -0.5}, continue_run=True) == []
    for cur, tag in (({"track": 1.0}, "-thermal"), ({"track": 1.0, "thermal": -0.5, "amp": 0.5}, "+amp"),
                     ({"track": 2.0, "thermal": -0.5}, "track: 1.0 -> 2.0")):
        with pytest.raises(ValueError) as e:
            check_reward_continuity(prev, cur, continue_run=True)
        assert tag in str(e.value)
    assert "CHANGED" in check_reward_continuity(prev, {"track": 1.0}, continue_run=True, allow_change=True)[0]
    assert UNRECORDED in check_reward_continuity({}, {"track": 1.0}, continue_run=True)[0]
    assert check_reward_continuity(prev, {}, continue_run=False) == []


def test_train_locomotion_wiring():
    src = _SCRIPT.read_text(encoding="utf-8")
    i_set, i_make = src.index("check_settings_continuity(prev_info"), src.index("gym.make(args.task")
    i_rew, i_load = src.index("check_reward_continuity(prev_info"), src.index("runner.load(")
    assert i_set < i_make < i_rew < i_load  # settings refused before the env build, rewards before the checkpoint
    tree = ast.parse(src)
    run_info = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "run_info" for t in n.targets)
                    and isinstance(n.value, ast.Dict))
    literal = {k.value for k in run_info.keys if isinstance(k, ast.Constant)}
    assert any(k is None and isinstance(v, ast.Name) and v.id == "settings" for k, v in zip(run_info.keys, run_info.values))
    assert not (literal & set(SETTING_FLAGS)), literal & set(SETTING_FLAGS)  # one source for the recorded settings
    assert "reward_weights" in literal and '"reward_weights": reward_weights' in src
