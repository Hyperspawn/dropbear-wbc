"""Item 368: the hw-library curation gate re-derives the duty screen's thermal constants and sus thresholds.

A v2 screen (ambient-aware, 0101) records ``screen.thermal`` (60 K rise, 24 C reference, split, tau_w / tau_h, trip,
max continuous ratio, HIGH trip time) and ``thresh.sus`` (warn, HIGH). If those differ from what
``dropbear_wbc.robots.thermal`` gives now at the screen's ambient the curation was made with other constants (or another
HIGH conversion): status ``stale_thermal``, WARN by default, refused under require_curation_ambient, downgraded by
allow_uncurated / continue_run. Recorded in run_info hw_library_curation.ambient_status(.thermal_check).
"""
from __future__ import annotations

import copy
import json
import math
import sys

import pytest

import sdk_test_paths  # noqa: F401  (adds source/)

sys.path.insert(0, str(sdk_test_paths.ROOT / "tools"))
import motion_duty_screen as mds  # noqa: E402

from dropbear_wbc.robots import thermal  # noqa: E402
from dropbear_wbc.robots.thermal import (  # noqa: E402
    THERMAL_ASSUMED, duty_screen_thermal, duty_sus_thresholds, sus_high_trip_s,
)
from dropbear_wbc.tasks.tracking.library_curation import (  # noqa: E402
    CURATION_SCHEMA, SCREEN_SCHEMA_V2, _num, SUS_THRESH_TOL, THERMAL_PARAM_TOL, CurationAmbientError, check_hw_library,
    screen_ambient_status, screen_thermal_check,
)

ROOT = sdk_test_paths.ROOT
SHIPPED = ROOT / "data/motions_v6ts/libraries/lowbody_v1_hw.json"


def _screen(amb=45.0):
    return {"schema": SCREEN_SCHEMA_V2, "map": "user_2026_09_25", "window_s": 2.0, "lp_hz": 6.0, "basis": "dynamic",
            "payload_kg": 0.0, "ambient_C": amb,
            "thresh": {"sus": list(duty_sus_thresholds(amb)), "peak": [1.0, 1.0], "static_sus": [0.8, 1.0]},
            "thermal": duty_screen_thermal(amb)}


def _manifest(screen):
    dec = [{"name": "stand", "severity": "ok", "action": "keep", "weight_in": 1.0, "weight_out": 1.0}]
    cur = {"schema": CURATION_SCHEMA, "source_manifest": "src", "policy": {"HIGH": "exclude"},
           "counts": {"HIGH": 0, "warn": 0, "ok": 1, "unscreened": 0}, "decisions": dec, "screen": screen}
    return {"schema": "dropbear-motion-library-v1", "name": "src_hw",
            "clips": [{"name": "stand", "npz": "stand.npz", "weight": 1.0}], "curation": cur}


def _write(tmp_path, m):
    p = tmp_path / "lib_hw.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    return p


def _status(screen):
    return screen_ambient_status(_manifest(screen))


# ---------------------------------------------------------------- the reference values themselves
def test_reference_values_at_45C():
    assert duty_sus_thresholds(45.0) == (0.9339, 1.1064)
    assert duty_sus_thresholds(25.0) == (1.0905, 1.3)
    assert sus_high_trip_s() == pytest.approx(1041.015, abs=1e-3)
    th = duty_screen_thermal(45.0)
    assert {k: th[k] for k in THERMAL_ASSUMED} == THERMAL_ASSUMED
    assert th["ref_ambient_C"] == 24.0 and th["max_continuous_ratio"] == 0.9339


def test_screen_tool_uses_the_same_source():
    # the screen tool and the gate must not drift apart: the tool re-exports the thermal-module functions
    assert mds.sus_high_trip_s is thermal.sus_high_trip_s
    assert mds.thresholds(45.0)["sus"] == duty_sus_thresholds(45.0)
    assert mds.thresholds(35.0)["sus"] == duty_sus_thresholds(35.0)
    assert mds.SUS_HIGH_REF == (1.3, 25.0)


def test_shipped_manifest_current_and_strict():
    m = json.loads(SHIPPED.read_text(encoding="utf-8"))
    tc = screen_thermal_check(m["curation"]["screen"], 45.0)
    assert tc == {"stale": False, "diffs": []}
    rec, notes = check_hw_library(SHIPPED, "hw_v1", require_curation_ambient=True)
    assert rec["ambient_status"]["status"] == "ok"
    assert rec["ambient_status"]["thermal_check"]["stale"] is False
    assert not any("WARNING" in n for n in notes)


# ---------------------------------------------------------------- classification
def test_num_rejects_bool_and_text():
    assert math.isnan(_num(True)) and math.isnan(_num(False)) and math.isnan(_num("x")) and math.isnan(_num(None))
    assert _num("45") == 45.0 and _num(0.3) == 0.3


@pytest.mark.parametrize("amb", [25.0, 35.0, 45.0])
def test_current_screen_not_stale(amb):
    st = _status(_screen(amb))
    assert st["thermal_check"] == {"stale": False, "diffs": []}
    assert st["status"] == ("ok" if amb == 45.0 else "mismatch")


BIG = {"rise_at_rated_K": 2.0, "ref_ambient_C": 1.0, "split_wh": 0.05, "tau_w_s": 5.0, "tau_h_s": 60.0,
       "t_trip_C": 5.0, "max_continuous_ratio": 0.01, "sus_high_trip_s": 10.0}


@pytest.mark.parametrize("key", sorted(THERMAL_PARAM_TOL))
@pytest.mark.parametrize("sign", [1, -1])
def test_each_thermal_param_perturbed_is_stale(key, sign):
    s = _screen()
    s["thermal"][key] = s["thermal"][key] + sign * BIG[key]
    st = _status(s)
    assert st["status"] == "stale_thermal", st
    assert [d["key"] for d in st["thermal_check"]["diffs"]] == [f"thermal.{key}"]
    assert f"thermal.{key}" in st["detail"] and "now" in st["detail"]


# one unit in the last place the screen records (max_continuous_ratio 1e-4, sus_high_trip_s 1e-3) must not count
LAST_PLACE = {"max_continuous_ratio": 1e-4, "sus_high_trip_s": 1e-3}


@pytest.mark.parametrize("key,unit", sorted(LAST_PLACE.items()))
@pytest.mark.parametrize("sign", [1, -1])
def test_one_recorded_unit_is_tolerated(key, unit, sign):
    s = _screen()
    s["thermal"][key] = round(s["thermal"][key] + sign * unit, 6)
    assert _status(s)["status"] == "ok"
    s["thermal"][key] = round(s["thermal"][key] + sign * unit, 6)  # two units: stale
    assert _status(s)["status"] == "stale_thermal"


@pytest.mark.parametrize("key", sorted(THERMAL_PARAM_TOL))
def test_each_thermal_param_within_tol_is_ok(key):
    s = _screen()
    s["thermal"][key] = s["thermal"][key] + 0.5 * THERMAL_PARAM_TOL[key]
    assert _status(s)["status"] == "ok"


@pytest.mark.parametrize("key", sorted(THERMAL_PARAM_TOL))
@pytest.mark.parametrize("bad", ["missing", None, "x", True, math.nan])
def test_missing_or_bad_param_is_stale(key, bad):
    s = _screen()
    if bad == "missing":
        del s["thermal"][key]
    else:
        s["thermal"][key] = bad
    st = _status(s)
    assert st["status"] == "stale_thermal"
    assert f"thermal.{key}" in [d["key"] for d in st["thermal_check"]["diffs"]]


@pytest.mark.parametrize("i,name", [(0, "warn"), (1, "HIGH")])
@pytest.mark.parametrize("delta,stale", [(1e-3, True), (-1e-3, True), (1e-4, False), (-1e-4, False), (2e-4, True)])
def test_sus_threshold_perturbed(i, name, delta, stale):
    s = _screen()
    s["thresh"]["sus"][i] = round(s["thresh"]["sus"][i] + delta, 6)
    st = _status(s)
    assert st["status"] == ("stale_thermal" if stale else "ok"), st
    if stale:
        assert [d["key"] for d in st["thermal_check"]["diffs"]] == [f"thresh.sus.{name}"]


@pytest.mark.parametrize("sus", [None, [], [0.9339], [0.9339, 1.1064, 2.0], "x", [0.9339, None]])
def test_bad_sus_shape_is_stale(sus):
    s = _screen()
    if sus is None:
        del s["thresh"]["sus"]
    else:
        s["thresh"]["sus"] = sus
    assert _status(s)["status"] == "stale_thermal"


def test_no_thresh_or_thermal_block_is_stale():
    s = _screen()
    del s["thresh"]
    assert _status(s)["status"] == "stale_thermal"
    s = _screen()
    del s["thermal"]
    st = _status(s)
    assert st["status"] == "stale_thermal" and len(st["thermal_check"]["diffs"]) == len(THERMAL_PARAM_TOL)


def test_old_high_conversion_is_stale():
    # the alternative HIGH rule of 0101 (1.3 x continuous = 1.2141 at 45 C) instead of the equal-trip-time rule
    s = _screen()
    s["thresh"]["sus"] = [0.9339, round(1.3 * 0.9339, 4)]
    st = _status(s)
    assert st["status"] == "stale_thermal"
    assert [d["key"] for d in st["thermal_check"]["diffs"]] == ["thresh.sus.HIGH"]


def test_fixed_rated_thresholds_in_v2_screen_are_stale():
    s = _screen()
    s["thresh"]["sus"] = [1.0, 1.3]
    st = _status(s)
    assert st["status"] == "stale_thermal" and len(st["thermal_check"]["diffs"]) == 2


def test_rise_62K_screen_is_stale():
    s = _screen()
    s["thermal"]["rise_at_rated_K"] = 62.0
    st = _status(s)
    assert st["status"] == "stale_thermal"
    d = st["thermal_check"]["diffs"][0]
    assert d["saved"] == 62.0 and d["current"] == 60.0


def test_stale_takes_precedence_over_mismatch():
    s = _screen(25.0)
    s["thresh"]["sus"] = list(duty_sus_thresholds(45.0))  # a 25 C screen carrying 45 C thresholds
    assert _status(s)["status"] == "stale_thermal"


def test_inconsistent_takes_precedence_over_stale():
    s = _screen()
    s["thermal"]["ambient_C"] = 25.0
    s["thermal"]["tau_w_s"] = 99.0
    st = _status(s)
    assert st["status"] == "inconsistent" and st["thermal_check"] is None


def test_absurd_ambient_is_stale_not_a_crash():
    s = _screen()
    s["ambient_C"] = s["thermal"]["ambient_C"] = -273.0
    st = _status(s)
    assert st["status"] == "stale_thermal"
    assert st["thermal_check"]["diffs"][0]["key"] == "thermal_model"
    assert st["thermal_check"]["diffs"][0]["error"].startswith("no ratio trips")


def test_detail_truncates_after_four():
    s = _screen()
    for k in ("rise_at_rated_K", "split_wh", "tau_w_s", "tau_h_s", "t_trip_C"):
        s["thermal"][k] = 1.0
    st = _status(s)
    assert st["detail"].endswith("... 1 more"), st["detail"]
    assert st["detail"].count("(now ") == 4 and "thermal.t_trip_C" not in st["detail"]


# ---------------------------------------------------------------- gate behaviour
def _stale_manifest():
    s = _screen()
    s["thermal"]["tau_h_s"] = 600.0
    return _manifest(s)


def test_default_warns_and_records(tmp_path):
    p = _write(tmp_path, _stale_manifest())
    rec, notes = check_hw_library(p, "hw_v1")
    assert rec["curated"] is True and rec["allowed"] is False
    st = rec["ambient_status"]
    assert st["status"] == "stale_thermal" and st["required"] is False
    assert st["thermal_check"]["diffs"][0]["key"] == "thermal.tau_h_s"
    warn = [n for n in notes if n.startswith("WARNING")]
    assert len(warn) == 1 and "stale_thermal" in warn[0] and "older thermal constants" in warn[0]
    json.dumps(rec)  # run_info-serialisable


def test_strict_refuses(tmp_path):
    p = _write(tmp_path, _stale_manifest())
    with pytest.raises(CurationAmbientError, match="stale_thermal"):
        check_hw_library(p, "hw_v1", require_curation_ambient=True)


@pytest.mark.parametrize("kw,needle", [({"allow_uncurated": True}, "--allow_uncurated"),
                                        ({"continue_run": True}, "--continue_run")])
def test_strict_downgraded(tmp_path, kw, needle):
    p = _write(tmp_path, _stale_manifest())
    rec, notes = check_hw_library(p, "hw_v1", require_curation_ambient=True, **kw)
    assert rec["allowed"] is True and rec["ambient_status"]["required"] is True
    assert any(n.startswith("WARNING") and "stale_thermal" in n and needle in n for n in notes)


def test_current_screen_strict_passes(tmp_path):
    p = _write(tmp_path, _manifest(_screen()))
    rec, notes = check_hw_library(p, "hw_v1", require_curation_ambient=True)
    assert rec["ambient_status"]["status"] == "ok" and not any("WARNING" in n for n in notes)


def test_sim_profile_not_checked(tmp_path):
    p = _write(tmp_path, _stale_manifest())
    rec, notes = check_hw_library(p, "sim", require_curation_ambient=True)
    assert rec["checked"] is False and notes == []


def test_cli_help_mentions_stale_thermal():
    assert "stale_thermal" in (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    assert "stale thermal" in (ROOT / "tools/run_chunked_training.py").read_text(encoding="utf-8")


def test_shipped_screen_deepcopy_mutation_detected():
    m = json.loads(SHIPPED.read_text(encoding="utf-8"))
    m2 = copy.deepcopy(m)
    m2["curation"]["screen"]["thermal"]["t_trip_C"] = 110.0
    assert screen_ambient_status(m)["status"] == "ok"
    assert screen_ambient_status(m2)["status"] == "stale_thermal"
