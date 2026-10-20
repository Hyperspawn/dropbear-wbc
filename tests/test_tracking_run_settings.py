"""Tracking run settings: --continue_run inheritance, drift audit, legacy_hwarm armature, NoState default (CPU)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import sdk_test_paths  # noqa: F401  (adds source/ to sys.path)

from dropbear_wbc.robots.dropbear_names import ACTUATOR_GROUP_MOTORS, ACTUATOR_PARAMS, MOTOR_NAMES  # noqa: E402
from dropbear_wbc.robots.hw_motor_specs import (  # noqa: E402
    HW_PROFILE_MAPS,
    LEGACY_HW_ARMATURE_PROFILES,
    joint_hw_params,
    legacy_group_armature,
)
from dropbear_wbc.tasks.tracking.run_settings import (  # noqa: E402
    RUN_SETTING_KEYS,
    audit_run_dir,
    latest_run_info,
    main,
    resolve_run_settings,
)

_REPO = Path(__file__).resolve().parents[1]
HW_RUN = {"actuator_profile": "hw_v1ie", "hw_regularizers": True, "knee_stop_penalty": -10.0,
          "feet_slide_penalty": -1.0, "target_clamp_margin_deg": 0.0, "action_rate_weight": None}
UNSET = {k: None for k in RUN_SETTING_KEYS}


def test_fresh_run_uses_cli_and_defaults():
    rs, notes = resolve_run_settings({**UNSET, "actuator_profile": "hw_v1"}, None, continue_run=False)
    assert {k: rs[k] for k in HW_RUN} == {"actuator_profile": "hw_v1", "hw_regularizers": False,
                                          "knee_stop_penalty": 0.0, "feet_slide_penalty": 0.0,
                                          "target_clamp_margin_deg": None, "action_rate_weight": None}
    assert rs["stop_model"] is False and rs["knee_band"] is None and rs["thermal_penalty"] is None
    assert notes == []
    # a warm start (--resume without --continue_run) may switch the plant: the previous run is ignored
    rs, _ = resolve_run_settings(dict(UNSET), HW_RUN, continue_run=False)
    assert rs["actuator_profile"] is None and rs["hw_regularizers"] is False


def test_continue_run_inherits_everything_unset():
    """The pre-patch failure: a resumed chunk without flags trained on legacy with no hw terms and no clamp."""
    rs, notes = resolve_run_settings(dict(UNSET), HW_RUN, continue_run=True)
    assert {k: rs[k] for k in HW_RUN} == HW_RUN
    assert len(notes) == len(RUN_SETTING_KEYS) and all("inherited" in n for n in notes)


def test_continue_run_refuses_a_different_setting_unless_allowed():
    with pytest.raises(ValueError, match="actuator_profile"):
        resolve_run_settings({**UNSET, "actuator_profile": "legacy"}, HW_RUN, continue_run=True)
    with pytest.raises(ValueError, match="knee_stop_penalty"):
        resolve_run_settings({**UNSET, "knee_stop_penalty": -5.0}, HW_RUN, continue_run=True)
    rs, notes = resolve_run_settings({**UNSET, "actuator_profile": "hw_v1"}, HW_RUN, continue_run=True,
                                     allow_change=True)
    assert rs["actuator_profile"] == "hw_v1" and rs["hw_regularizers"] is True
    assert any("CHANGED 'hw_v1ie' -> 'hw_v1'" in n for n in notes)
    # repeating the recorded value is fine
    rs, _ = resolve_run_settings({**UNSET, "actuator_profile": "hw_v1ie", "hw_regularizers": True}, HW_RUN,
                                 continue_run=True)
    assert {k: rs[k] for k in HW_RUN} == HW_RUN


def test_old_run_info_without_keys_means_legacy():
    rs, _ = resolve_run_settings(dict(UNSET), {"task": "Dropbear-Tracking-Flat-v0"}, continue_run=True)
    assert rs["actuator_profile"] == "legacy" and rs["hw_regularizers"] is False and rs["knee_stop_penalty"] == 0.0


def _write(d: Path, name: str, info: dict) -> None:
    (d / name).write_text(json.dumps(info), encoding="utf-8")


def test_latest_run_info_and_drift_audit(tmp_path, capsys):
    run = tmp_path / "2026-09-26_01-00-00_kimodo_nostate"
    run.mkdir()
    _write(run, "run_info.json", HW_RUN)
    _write(run, "run_info_resume_2026-09-26_02-00-00.json", HW_RUN)
    # pre-patch resumed chunk without flags (what train.py recorded)
    legacy_chunk = {**HW_RUN, "actuator_profile": "legacy", "hw_regularizers": False, "knee_stop_penalty": 0.0,
                    "feet_slide_penalty": 0.0, "target_clamp_margin_deg": None}
    _write(run, "run_info_resume_2026-09-26_03-00-00.json", legacy_chunk)
    assert latest_run_info(run)["actuator_profile"] == "legacy"
    drift = audit_run_dir(run)
    assert {r["key"] for r in drift} == {"actuator_profile", "hw_regularizers", "knee_stop_penalty",
                                         "feet_slide_penalty", "target_clamp_margin_deg"}
    assert all(r["file"] == "run_info_resume_2026-09-26_03-00-00.json" and not r["flagged"] for r in drift)
    assert main([str(run)]) == 1
    ok = tmp_path / "ok"
    ok.mkdir()
    _write(ok, "run_info.json", HW_RUN)
    _write(ok, "run_info_resume_2026-09-26_02-00-00.json", {**HW_RUN, "run_settings_notes": ["x: inherited"]})
    assert audit_run_dir(ok) == [] and main([str(ok)]) == 0
    flagged = tmp_path / "flagged"
    flagged.mkdir()
    _write(flagged, "run_info.json", HW_RUN)
    _write(flagged, "run_info_resume_2026-09-26_02-00-00.json",
           {**HW_RUN, "actuator_profile": "hw_v1", "run_settings_notes": ["actuator_profile: CHANGED ..."]})
    assert [r["flagged"] for r in audit_run_dir(flagged)] == [True] and main([str(flagged)]) == 0
    capsys.readouterr()


def test_legacy_hwarm_armature_is_the_datasheet_armature():
    for name, map_name in LEGACY_HW_ARMATURE_PROFILES.items():
        assert name not in HW_PROFILE_MAPS  # not a hardware twin: the scoreboard must not grade it
        arm = legacy_group_armature(map_name)
        params = joint_hw_params(map_name)
        assert set(arm) == set(ACTUATOR_GROUP_MOTORS)
        flat = {m: a for g in arm.values() for m, a in g.items()}
        assert sorted(flat) == sorted(MOTOR_NAMES)
        assert all(flat[m] == pytest.approx(params[m]["armature"]) for m in MOTOR_NAMES)
        assert min(flat.values()) > ACTUATOR_PARAMS["arms"][3]  # every motor is heavier than the legacy 0.01


def _calls_in_post_init(path: Path, cls: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
    return [ast.unparse(c) for c in ast.walk(init) if isinstance(c, ast.Call)]


def test_nostate_train_tasks_default_to_a_hw_profile_and_play_tasks_do_not():
    lib = _REPO / "source/dropbear_wbc/tasks/tracking/config/dropbear/library_env_cfg.py"
    flat = _REPO / "source/dropbear_wbc/tasks/tracking/config/dropbear/flat_env_cfg.py"
    for cls in ("DropbearLibraryNoStateEnvCfg", "DropbearFlatNoStateEnvCfg"):
        assert "self.set_actuator_profile(DEPLOYABLE_TRACKING_PROFILE)" in _calls_in_post_init(lib, cls)
    for cls in ("DropbearLibraryNoStatePlayEnvCfg", "DropbearFlatNoStatePlayEnvCfg"):  # play takes run_info's profile
        assert not any("set_actuator_profile" in c for c in _calls_in_post_init(lib, cls))
    tree = ast.parse(flat.read_text(encoding="utf-8"))
    val = next(n.value.value for n in tree.body if isinstance(n, ast.AnnAssign)
               and getattr(n.target, "id", "") == "DEPLOYABLE_TRACKING_PROFILE")
    assert val in HW_PROFILE_MAPS


def test_train_py_reads_run_settings_only_through_the_resolver():
    src = (_REPO / "scripts/train.py").read_text(encoding="utf-8")
    for key in RUN_SETTING_KEYS:
        assert f"args.{key}" not in src, key
    tree = ast.parse(src)
    defaults = {}
    for c in ast.walk(tree):
        if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "add_argument" and c.args:
            flag = getattr(c.args[0], "value", "")
            for kw in c.keywords:
                if kw.arg == "default":
                    defaults[flag] = ast.unparse(kw.value)
    for key in RUN_SETTING_KEYS:  # unset must be distinguishable from an explicit value
        assert defaults.get(f"--{key}") == "None", key


S7_RUN = {**HW_RUN, "actuator_profile": "hw_v4", "thermal_penalty": -0.002, "stop_model": True,
          "stop_load_penalty": -5.0, "impact_penalty": -0.005, "knee_band": "150,15", "body_pos_std": 0.2,
          "joint_pos_weight": 1.0, "end_threshold_m": 0.2, "anchor_height_weight": 1.0, "feet_height_weight": 1.0}


def test_continue_run_inherits_the_later_training_flags():
    """A chunk resumed without its flags keeps the stop model, knee band, safety penalties, strictness and height terms."""
    rs, notes = resolve_run_settings(dict(UNSET), S7_RUN, continue_run=True)
    assert {k: rs[k] for k in S7_RUN} == S7_RUN
    for k in ("stop_model", "knee_band", "thermal_penalty", "anchor_height_weight"):
        assert any(n.startswith(f"{k}: inherited") for n in notes), (k, notes)


def test_continue_run_refuses_switching_off_the_stop_model_or_band():
    with pytest.raises(ValueError, match="stop_load_penalty"):
        resolve_run_settings({**UNSET, "stop_load_penalty": 0.0}, S7_RUN, continue_run=True)
    with pytest.raises(ValueError, match="knee_band"):
        resolve_run_settings({**UNSET, "knee_band": "100,10"}, S7_RUN, continue_run=True)
    rs, notes = resolve_run_settings({**UNSET, "knee_band": "100,10"}, S7_RUN, continue_run=True, allow_change=True)
    assert rs["knee_band"] == "100,10" and any("CHANGED" in n for n in notes)


def test_old_run_info_without_the_later_keys_means_off():
    old = dict(HW_RUN)  # a run_info from before these flags existed
    rs, _ = resolve_run_settings(dict(UNSET), old, continue_run=True)
    assert rs["stop_model"] is False and rs["knee_band"] is None and rs["stop_load_penalty"] == 0.0
    assert rs["thermal_penalty"] is None and rs["feet_height_weight"] == 0.0


def test_train_py_leaves_every_run_setting_flag_unset_by_default():
    """resolve_run_settings treats None as 'not given'; a non-None argparse default would defeat inheritance."""
    tree = ast.parse((_REPO / "scripts/train.py").read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            flag = getattr(node.args[0], "value", "")
            if isinstance(flag, str) and flag.startswith("--"):
                kw = {k.arg: k.value for k in node.keywords}
                defaults[flag[2:]] = ("<expr>" if "default" in kw and not isinstance(kw["default"], ast.Constant) else ast.literal_eval(kw["default"]) if "default" in kw else (
                    False if getattr(kw.get("action"), "value", "") == "store_true" else None))
    for key in RUN_SETTING_KEYS:
        assert key in defaults, key
        assert defaults[key] is None, (key, defaults[key])
