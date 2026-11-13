"""Event / DR record in run_info (item 190): snapshot, --continue_run continuity, drift audit (CPU, no Isaac)."""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import sdk_test_paths  # noqa: F401  (adds source/ to sys.path)

from dropbear_wbc.tasks.dr_record import (  # noqa: E402
    UNRECORDED,
    check_dr_continuity,
    dr_drift,
    dr_event_record,
    dr_record_diff,
)
from dropbear_wbc.tasks.tracking.run_settings import audit_run_dir, main  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


@dataclass
class _Entity:  # SceneEntityCfg stand-in
    name: str
    joint_names: object = None
    body_names: object = None
    joint_ids: object = field(default_factory=lambda: slice(None))


@dataclass
class _Term:  # EventTermCfg stand-in
    func: object
    mode: str
    params: dict = field(default_factory=dict)
    interval_range_s: object = None


def randomize_rigid_body_mass(*a, **k):
    pass


def randomize_limb_mass_scale(*a, **k):
    pass


def push_by_setting_velocity(*a, **k):
    pass


class _Events:
    def __init__(self, base_add=(0.0, 6.0), motor_first=True, limb=(0.9, 1.1), with_limb=True):
        motor = _Term(randomize_rigid_body_mass, "startup", {"asset_cfg": _Entity("robot", body_names=["thigh"]),
                                                             "mass_distribution_params": (-0.1, 0.1)})
        if motor_first:
            self.motor_mass = motor
        self.add_base_mass = _Term(randomize_rigid_body_mass, "startup",
                                   {"asset_cfg": _Entity("robot", body_names="world"),
                                    "mass_distribution_params": base_add, "operation": "add"})
        self.scale_limb_mass = (_Term(randomize_limb_mass_scale, "startup",
                                      {"asset_cfg": _Entity("robot"), "scale_range": limb}) if with_limb else None)
        if not motor_first:
            self.motor_mass = motor
        self.push_robot = _Term(push_by_setting_velocity, "interval", {"velocity_range": {"x": (-0.5, 0.5)}},
                                interval_range_s=(1.0, 3.0))


def test_record_is_json_ordered_and_skips_none():
    rec = dr_event_record(_Events(with_limb=False))
    assert json.loads(json.dumps(rec)) == rec
    assert [t["name"] for t in rec["terms"]] == ["motor_mass", "add_base_mass", "push_robot"]
    base = rec["terms"][1]
    assert base["func"].endswith(":randomize_rigid_body_mass") and base["mode"] == "startup"
    assert base["params"]["mass_distribution_params"] == [0.0, 6.0]
    assert base["params"]["asset_cfg"] == {"entity": {"name": "robot", "joint_names": None, "body_names": "world"}}
    assert rec["terms"][2]["interval_range_s"] == [1.0, 3.0]
    assert dr_event_record(None) == {"version": 1, "terms": []}


def test_diff_names_param_add_remove_and_order_changes():
    a = dr_event_record(_Events())
    assert dr_record_diff(a, dr_event_record(_Events())) == []
    assert dr_record_diff(a, dr_event_record(_Events(base_add=(-1.0, 6.0)))) == [
        "add_base_mass.params.mass_distribution_params: [0.0, 6.0] -> [-1.0, 6.0]"]
    assert dr_record_diff(a, dr_event_record(_Events(with_limb=False))) == ["scale_limb_mass: removed"]
    assert any(d.startswith("scale_limb_mass: added") for d in dr_record_diff(dr_event_record(_Events(with_limb=False)), a))
    order = dr_record_diff(a, dr_event_record(_Events(motor_first=False)))
    assert len(order) == 1 and order[0].startswith("order: ['motor_mass', 'add_base_mass'")


def test_continue_run_refuses_changed_dr_unless_allowed():
    cur = dr_event_record(_Events())
    prev = {"dr_events": dr_event_record(_Events(limb=(0.8, 1.2)))}
    assert check_dr_continuity(prev, cur, continue_run=False) == []
    assert check_dr_continuity({"dr_events": cur}, cur, continue_run=True) == []
    with pytest.raises(ValueError, match=r"scale_limb_mass\.params\.scale_range: \[0\.8, 1\.2\] -> \[0\.9, 1\.1\]"):
        check_dr_continuity(prev, cur, continue_run=True)
    notes = check_dr_continuity(prev, cur, continue_run=True, allow_change=True)
    assert len(notes) == 1 and "CHANGED" in notes[0]


def test_pre_record_run_is_reported_unrecorded_not_silently_accepted():
    cur = dr_event_record(_Events())
    notes = check_dr_continuity({"actuator_profile": "hw_v1ie"}, cur, continue_run=True)
    assert len(notes) == 1 and UNRECORDED in notes[0] and "CHANGED" not in notes[0]
    d = dr_drift({}, {"dr_events": cur})
    assert d["before"] == UNRECORDED and d["unverifiable"] is True
    assert dr_drift({}, {}) is None


def _write(run: Path, infos):
    run.mkdir()
    for i, info in enumerate(infos):
        name = "run_info.json" if i == 0 else f"run_info_resume_2026-11-1{i}_00-00-00.json"
        (run / name).write_text(json.dumps(info), encoding="utf-8")


def test_drift_audit_sees_dr_changes(tmp_path, capsys):
    a, b = dr_event_record(_Events()), dr_event_record(_Events(base_add=(0.0, 8.0)))
    _write(tmp_path / "same", [{"dr_events": a}, {"dr_events": a}])
    assert audit_run_dir(tmp_path / "same") == [] and main([str(tmp_path / "same")]) == 0
    _write(tmp_path / "changed", [{"dr_events": a}, {"dr_events": b}])
    drift = audit_run_dir(tmp_path / "changed")
    assert [r["key"] for r in drift] == ["dr_events"] and not drift[0]["flagged"]
    assert main([str(tmp_path / "changed")]) == 1
    _write(tmp_path / "ack", [{"dr_events": a}, {"dr_events": b, "run_settings_notes": ["dr_events: CHANGED x"]}])
    assert main([str(tmp_path / "ack")]) == 0
    _write(tmp_path / "legacy", [{}, {"dr_events": a}])  # pre-0010 / pre-0080 run continued now: fail closed
    drift = audit_run_dir(tmp_path / "legacy")
    assert drift[0]["unverifiable"] and main([str(tmp_path / "legacy")]) == 1
    _write(tmp_path / "old", [{}, {}])
    assert audit_run_dir(tmp_path / "old") == []
    capsys.readouterr()


@pytest.mark.parametrize("script", ["scripts/train.py", "scripts/train_locomotion.py"])
def test_train_scripts_record_and_check_dr_before_gym_make(script):
    src = (_REPO / script).read_text(encoding="utf-8")
    tree = ast.parse(src)
    keys = {k.value for n in ast.walk(tree) if isinstance(n, ast.Dict) for k in n.keys if isinstance(k, ast.Constant)}
    assert "dr_events" in keys
    calls = {n.func.id: n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    make = min(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "make")
    assert calls["dr_event_record"] < make and calls["check_dr_continuity"] < make
