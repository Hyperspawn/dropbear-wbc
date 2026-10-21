"""``--robot_mass_kg``: opt-in re-centring of the plant mass and the root mass DR on a measured mass. CPU, Isaac-free.

Without the option every profile keeps ``motor_mass_offsets`` and the ROOT_MASS_ADD_* ranges (published policies
replay unchanged). With it, the gap to CAD + the profile's motor swap goes on ``world`` and the root DR is a band.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import sdk_test_paths  # noqa: F401

from dropbear_wbc.robots.dropbear_names import ROOT_BODY
from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS, motor_mass_offsets
from dropbear_wbc.robots.mass_dr import (
    ROOT_MASS_ADD_LOCOMOTION, ROOT_MASS_ADD_TRACKING, ROOT_MASS_BAND_LOCOMOTION, ROOT_MASS_BAND_TRACKING,
    cad_total_mass_kg, check_robot_mass_kg, plant_mass_offsets, predicted_mass_kg, root_mass_add_range,
    run_robot_mass_kg,
)
from dropbear_wbc.tasks.tracking.run_settings import RUN_SETTING_KEYS, resolve_run_settings

_REPO = Path(__file__).resolve().parents[1]
PROFILES = ("legacy", *sorted(HW_PROFILE_MAPS))


@pytest.mark.parametrize("profile", PROFILES)
def test_default_is_unchanged(profile):
    assert plant_mass_offsets(profile) == motor_mass_offsets(profile)
    assert predicted_mass_kg(profile) == pytest.approx(cad_total_mass_kg() + sum(motor_mass_offsets(profile).values()))


def test_default_ranges_are_unchanged():
    assert root_mass_add_range("tracking") == ROOT_MASS_ADD_TRACKING == (0.0, 6.0)
    assert root_mass_add_range("locomotion") == ROOT_MASS_ADD_LOCOMOTION == (-1.0, 6.0)
    with pytest.raises(ValueError):
        root_mass_add_range("other")


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("kg", (55.0, 60.0, 63.3))
def test_measured_mass_recentres_the_plant(profile, kg):
    off = plant_mass_offsets(profile, kg)
    assert predicted_mass_kg(profile, kg) == pytest.approx(kg, abs=1e-3)
    # only world changes vs the motor swap
    base = motor_mass_offsets(profile)
    assert {k: v for k, v in off.items() if k != ROOT_BODY} == {k: v for k, v in base.items() if k != ROOT_BODY}


def test_hw_cem6_at_60kg_puts_8p45kg_on_world():
    off = plant_mass_offsets("hw_cem6", 60.0)
    assert off[ROOT_BODY] - motor_mass_offsets("hw_cem6").get(ROOT_BODY, 0.0) == pytest.approx(8.45, abs=0.02)


def test_band_with_measured_mass():
    assert root_mass_add_range("tracking", 60.0) == ROOT_MASS_BAND_TRACKING
    assert root_mass_add_range("locomotion", 60.0) == ROOT_MASS_BAND_LOCOMOTION


@pytest.mark.parametrize("bad", (0.0, 132.0, float("nan"), -60.0, 600.0))
def test_implausible_mass_fails(bad):
    with pytest.raises(ValueError):
        check_robot_mass_kg(bad)
    with pytest.raises(ValueError):
        plant_mass_offsets("hw_cem6", bad)


def test_run_mass_is_the_newest_non_null_chunk():
    assert run_robot_mass_kg([]) is None
    assert run_robot_mass_kg([{"actuator_profile": "hw_v4"}]) is None
    assert run_robot_mass_kg([{"robot_mass_kg": 60.0}, {"robot_mass_kg": None}]) == 60.0
    assert run_robot_mass_kg([{"robot_mass_kg": 60.0}, {"robot_mass_kg": 61.5}]) == 61.5


def test_continue_run_inherits_and_guards_the_mass():
    unset = {k: None for k in RUN_SETTING_KEYS}
    assert "robot_mass_kg" in RUN_SETTING_KEYS
    rs, _ = resolve_run_settings(dict(unset), {"actuator_profile": "hw_v4", "robot_mass_kg": 61.0}, continue_run=True)
    assert rs["robot_mass_kg"] == 61.0
    rs, _ = resolve_run_settings(dict(unset), {"actuator_profile": "hw_v4"}, continue_run=True)
    assert rs["robot_mass_kg"] is None  # old run: default plant
    with pytest.raises(ValueError, match="robot_mass_kg"):
        resolve_run_settings({**unset, "robot_mass_kg": 60.0}, {"robot_mass_kg": 61.0}, continue_run=True)


def _flags(path: str) -> set[str]:
    tree = ast.parse((_REPO / path).read_text(encoding="utf-8"))
    return {c.args[0].value for c in ast.walk(tree) if isinstance(c, ast.Call)
            and getattr(c.func, "attr", "") == "add_argument" and c.args and isinstance(c.args[0], ast.Constant)}


@pytest.mark.parametrize("script", ("scripts/train.py", "scripts/train_locomotion.py", "scripts/play.py",
                                    "scripts/play_locomotion.py", "scripts/robustness_bench.py"))
def test_scripts_take_the_option(script):
    assert "--robot_mass_kg" in _flags(script)
    assert "set_robot_mass(" in (_REPO / script).read_text(encoding="utf-8")


@pytest.mark.parametrize("script", ("scripts/play.py", "scripts/play_locomotion.py"))
def test_perturbed_play_reapplies_the_runs_mass_band(script):
    """Perturbed play copies add_base_mass from the TRAIN config (default plant); the run's band must be re-applied
    on a copy, after the copy (otherwise a --robot_mass_kg run is evaluated on the default range)."""
    src = (_REPO / script).read_text(encoding="utf-8")
    block = src[src.index("if args.perturbed:"):]
    assert "copy.deepcopy(" in block.split("set_robot_mass(env_cfg.robot_mass_kg)")[0]
    assert "set_robot_mass(env_cfg.robot_mass_kg)" in block


@pytest.mark.parametrize("cfg", ("source/dropbear_wbc/tasks/tracking/config/dropbear/flat_env_cfg.py",
                                 "source/dropbear_wbc/tasks/locomotion/config/dropbear/flat_env_cfg.py"))
def test_env_cfgs_default_to_no_measured_mass(cfg):
    tree = ast.parse((_REPO / cfg).read_text(encoding="utf-8"))
    ann = [n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "robot_mass_kg"]
    assert len(ann) == 1 and isinstance(ann[0].value, ast.Constant) and ann[0].value.value is None
