"""Item 367: the hw-library curation gate checks the ambient of the duty screen the curation was made with.

A pre-0101 fixed-rated screen (schema v1, no ambient_C) or a screen at another ambient than the 45 C design ambient
WARNS by default (older *_hw.json on the PC / Brev keep working) and is refused with require_curation_ambient
(train.py --require_curation_ambient); the outcome is recorded in run_info hw_library_curation.ambient_status.
"""
from __future__ import annotations

import ast
import copy
import json

import pytest

import sdk_test_paths  # noqa: F401  (adds source/)

from dropbear_wbc.robots.thermal import DESIGN_AMBIENT_C
from dropbear_wbc.tasks.tracking.library_curation import (
    CURATION_SCHEMA, SCREEN_SCHEMA_V2, CurationAmbientError, UncuratedLibraryError, check_hw_library,
    curation_problems, screen_ambient_status,
)

ROOT = sdk_test_paths.ROOT
LIBS = ROOT / "data/motions_v6ts/libraries"

V1_SCREEN = {"schema": "dropbear-motion-duty-screen-v1", "map": "user_2026_09_25", "window_s": 2.0, "lp_hz": 6.0,
             "basis": "dynamic", "payload_kg": 0.0,
             "thresh": {"sus": [1.0, 1.3], "peak": [1.0, 1.0], "static_sus": [0.8, 1.0]}}


def _v2(amb=45.0, thermal_amb="same"):
    # a screen as tools/motion_duty_screen.py writes it now (current thermal constants / sus thresholds, item 368)
    from dropbear_wbc.robots.thermal import duty_screen_thermal, duty_sus_thresholds

    th = duty_screen_thermal(amb)
    th["ambient_C"] = amb if thermal_amb == "same" else thermal_amb
    thresh = {**V1_SCREEN["thresh"], "sus": list(duty_sus_thresholds(amb))}
    return {**{k: v for k, v in V1_SCREEN.items() if k != "schema"}, "schema": SCREEN_SCHEMA_V2, "ambient_C": amb,
            "thresh": thresh, "thermal": th}


def _manifest(screen):
    dec = [{"name": "stand", "severity": "ok", "action": "keep", "weight_in": 1.0, "weight_out": 1.0}]
    cur = {"schema": CURATION_SCHEMA, "source_manifest": "src", "policy": {"HIGH": "exclude"},
           "counts": {"HIGH": 0, "warn": 0, "ok": 1, "unscreened": 0}, "decisions": dec}
    if screen is not None:
        cur["screen"] = screen
    return {"schema": "dropbear-motion-library-v1", "name": "src_hw",
            "clips": [{"name": "stand", "npz": "stand.npz", "weight": 1.0}], "curation": cur}


def _write(tmp_path, m):
    p = tmp_path / "lib_hw.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    return p


CASES = [  # (screen, expected status, detail needle)
    (_v2(45.0), "ok", ""),
    (V1_SCREEN, "no_ambient", "fixed-rated"),
    (None, "no_ambient", "no 'screen' block"),
    ({**_v2(45.0), "ambient_C": None}, "no_ambient", "fixed-rated"),
    ({**_v2(45.0), "schema": "dropbear-motion-duty-screen-v1"}, "no_ambient", "fixed-rated"),
    ({**_v2(45.0), "ambient_C": "hot"}, "no_ambient", "not a finite number"),
    (_v2(25.0), "mismatch", "screened at 25 C"),
    (_v2(35.0), "mismatch", "screened at 35 C"),
    (_v2(45.0, thermal_amb=25.0), "inconsistent", "screen.thermal.ambient_C 25 C"),
]


def test_design_ambient_is_45():
    assert DESIGN_AMBIENT_C == 45.0


@pytest.mark.parametrize("screen,status,needle", CASES)
def test_status_classification(screen, status, needle):
    st = screen_ambient_status(_manifest(copy.deepcopy(screen)))
    assert st["status"] == status, st
    assert needle in st["detail"], st
    assert st["design_ambient_C"] == 45.0
    assert (st["detail"] == "") is (status == "ok")


def test_no_curation_has_no_status():
    m = _manifest(_v2())
    m.pop("curation")
    assert screen_ambient_status(m) is None


def test_explicit_design_ambient():
    assert screen_ambient_status(_manifest(_v2(35.0)), 35.0)["status"] == "ok"
    assert screen_ambient_status(_manifest(_v2(45.0)), 35.0)["status"] == "mismatch"


@pytest.mark.parametrize("screen,status,_n", CASES)
def test_default_only_warns_and_records(tmp_path, screen, status, _n):
    p = _write(tmp_path, _manifest(copy.deepcopy(screen)))
    rec, notes = check_hw_library(p, "hw_v1ie")
    assert rec["curated"] is True and rec["allowed"] is False and rec["problems"] == []
    assert rec["ambient_status"]["status"] == status and rec["ambient_status"]["required"] is False
    assert "OK" in notes[0]
    warn = [n for n in notes if n.startswith("WARNING")]
    assert (len(warn) == 1) is (status != "ok")
    if warn:
        assert status in warn[0] and "curate_hw_library" in warn[0]


@pytest.mark.parametrize("screen,status,_n", [c for c in CASES if c[1] != "ok"])
def test_require_refuses_fresh_run(tmp_path, screen, status, _n):
    p = _write(tmp_path, _manifest(copy.deepcopy(screen)))
    with pytest.raises(CurationAmbientError, match=status):
        check_hw_library(p, "hw_v1ie", require_curation_ambient=True)
    assert issubclass(CurationAmbientError, UncuratedLibraryError)


def test_require_passes_ok(tmp_path):
    p = _write(tmp_path, _manifest(_v2(45.0)))
    rec, notes = check_hw_library(p, "hw_cem6", require_curation_ambient=True)
    assert rec["ambient_status"]["status"] == "ok" and rec["ambient_status"]["required"] is True
    assert not any(n.startswith("WARNING") for n in notes)


@pytest.mark.parametrize("kw,needle", [({"allow_uncurated": True}, "--allow_uncurated"),
                                        ({"continue_run": True}, "--continue_run")])
def test_require_downgraded_by_allow_or_continue(tmp_path, kw, needle):
    p = _write(tmp_path, _manifest(V1_SCREEN))
    rec, notes = check_hw_library(p, "hw_v1ie", require_curation_ambient=True, **kw)
    assert rec["allowed"] is True and rec["ambient_status"]["status"] == "no_ambient"
    assert needle in notes[-1] and notes[-1].startswith("WARNING")


def test_uncurated_still_refused_with_status_none(tmp_path):
    m = _manifest(_v2())
    m.pop("curation")
    with pytest.raises(UncuratedLibraryError):
        check_hw_library(_write(tmp_path, m), "hw_v1ie")
    rec, _ = check_hw_library(_write(tmp_path, m), "hw_v1ie", allow_uncurated=True)
    assert rec["ambient_status"] is None


def test_sim_profile_unchecked(tmp_path):
    p = _write(tmp_path, _manifest(V1_SCREEN))
    rec, notes = check_hw_library(p, "legacy", require_curation_ambient=True)
    assert rec["checked"] is False and rec["ambient_status"] is None and notes == []


def test_shipped_hw_manifests_pass_strict():
    hw = sorted(LIBS.glob("*_hw.json"))
    assert [p.name for p in hw] == ["accepted_v7gen_hw.json", "lowbody_v1_hw.json"]
    for p in hw:
        m = json.loads(p.read_text(encoding="utf-8"))
        assert curation_problems(m) == []
        rec, notes = check_hw_library(p, "hw_v1ie", require_curation_ambient=True)
        assert rec["ambient_status"]["status"] == "ok" and rec["ambient_status"]["ambient_C"] == 45.0
        assert not any(n.startswith("WARNING") for n in notes)


def test_train_py_and_chunked_driver_wire_the_flag():
    src = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    main = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = ast.get_source_segment(src, main)
    assert "require_curation_ambient=args.require_curation_ambient" in body
    assert '"--require_curation_ambient", action="store_true"' in src
    drv = (ROOT / "tools/run_chunked_training.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--require_curation_ambient", action="store_true"' in drv
    assert ('if args.require_curation_ambient:\n            cmd[cmd.index("--headless"):cmd.index("--headless")] = '
            '["--require_curation_ambient"]') in drv
