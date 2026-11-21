"""``tools/curate_hw_library.py`` (item 167): hw library from the arm-duty screen, source never modified, fail closed."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys

import pytest

import sdk_test_paths  # noqa: F401  (adds source/)

sys.path.insert(0, str(sdk_test_paths.ROOT / "tools"))
import curate_hw_library as chl  # noqa: E402

SYN = sdk_test_paths.ROOT / "data/motions_v6ts/synthetic"
CLIPS = ("stand", "wave_right", "arm_swing", "squat_lite")


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def lib(tmp_path):
    d = tmp_path / "data" / "synthetic"
    d.mkdir(parents=True)
    for n in CLIPS:
        for suf in (".npz", ".validation.json"):
            shutil.copy(SYN / f"{n}{suf}", d / f"{n}{suf}")
    libdir = tmp_path / "data" / "libraries"
    libdir.mkdir()
    m = {"schema": "dropbear-motion-library-v1", "name": "src",
         "clips": [{"name": n, "npz": f"../synthetic/{n}.npz", "weight": 2.0 if n == "arm_swing" else 1.0,
                    "sha256": _sha(d / f"{n}.npz")} for n in CLIPS]}
    p = libdir / "src.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    return p


def _screen(lib, worst):
    """Fake screen result: worst per clip name; HIGH via sustained elbow unless the name says peak."""
    clips = []
    for n, w in worst.items():
        f = []
        if w == "HIGH":
            f = [{"motor": "LH_elbow_joint", "metric": "sus", "value": 1.5, "severity": "HIGH", "basis": "dynamic"}]
        elif w == "HIGHpeak":
            w = "HIGH"
            f = [{"motor": "LH_elbow_joint", "metric": "peak", "value": 1.1, "severity": "HIGH", "basis": "dynamic"},
                 {"motor": "LH_elbow_joint", "metric": "sus", "value": 1.1, "severity": "warn", "basis": "dynamic"}]
        elif w == "warn":
            f = [{"motor": "RH_pitch", "metric": "static_sus", "value": 0.9, "severity": "warn", "basis": "static"}]
        clips.append({"clip": n, "path": str(lib.parent.parent / "synthetic" / f"{n}.npz"), "worst": w, "findings": f})
    return {"schema": "dropbear-motion-duty-screen-v2", "map": "user_2026_09_25", "window_s": 2.0, "lp_hz": 6.0,
            "basis": "dynamic", "payload_kg": 0.0, "ambient_C": 45.0, "thresh": {}, "thermal": {},
            "clips": clips, "errors": []}


def _load(lib):
    return json.loads(lib.read_text(encoding="utf-8"))


def test_parse_action():
    assert chl.parse_action("exclude") == ("exclude", 1.0)
    assert chl.parse_action("Keep") == ("keep", 1.0)
    assert chl.parse_action("weight=0.25") == ("weight", 0.25)
    for bad in ("weight=0", "weight=-1", "weight=inf", "weight=x", "drop"):
        with pytest.raises(ValueError):
            chl.parse_action(bad)


def test_default_excludes_high_keeps_warn_and_weights(lib):
    res = _screen(lib, {"stand": "ok", "wave_right": "warn", "arm_swing": "ok", "squat_lite": "HIGH"})
    sel = chl.curate(_load(lib), lib, res)
    assert [c["name"] for c in sel["clips"]] == ["stand", "wave_right", "arm_swing"]
    assert {c["name"]: c["weight"] for c in sel["clips"]}["arm_swing"] == 2.0  # source weight kept
    dec = {d["name"]: d for d in sel["decisions"]}
    assert dec["squat_lite"]["action"] == "exclude" and dec["squat_lite"]["weight_out"] is None


def test_weight_actions_multiply_source_weight(lib):
    res = _screen(lib, {"stand": "ok", "wave_right": "warn", "arm_swing": "HIGH", "squat_lite": "ok"})
    sel = chl.curate(_load(lib), lib, res, high=("weight", 0.1), warn=("weight", 0.5))
    w = {c["name"]: c["weight"] for c in sel["clips"]}
    assert w == {"stand": 1.0, "wave_right": 0.5, "arm_swing": pytest.approx(0.2), "squat_lite": 1.0}


def test_peak_only_high_downgrade(lib):
    res = _screen(lib, {"stand": "HIGHpeak", "wave_right": "HIGH", "arm_swing": "ok", "squat_lite": "ok"})
    assert [c["name"] for c in chl.curate(_load(lib), lib, res)["clips"]] == ["arm_swing", "squat_lite"]
    sel = chl.curate(_load(lib), lib, res, peak_only="warn")
    assert [c["name"] for c in sel["clips"]] == ["stand", "arm_swing", "squat_lite"]  # sustained HIGH still out
    assert {d["name"]: d["severity"] for d in sel["decisions"]}["stand"] == "warn"


def test_unscreened_fails_closed(lib):
    res = _screen(lib, {"stand": "ok", "wave_right": "ok", "arm_swing": "ok"})  # squat_lite missing
    with pytest.raises(RuntimeError, match="could not be screened"):
        chl.curate(_load(lib), lib, res)
    assert [c["name"] for c in chl.curate(_load(lib), lib, res, unscreened="exclude")["clips"]] == \
        ["stand", "wave_right", "arm_swing"]
    assert len(chl.curate(_load(lib), lib, res, unscreened="keep")["clips"]) == 4


def test_changed_npz_bytes_count_as_unscreened(lib):
    m = _load(lib)
    m["clips"][0]["sha256"] = "0" * 64
    res = _screen(lib, {n: "ok" for n in CLIPS})
    with pytest.raises(RuntimeError, match="pin"):
        chl.curate(m, lib, res)


def test_cli_writes_new_manifest_and_never_touches_source(lib, tmp_path):
    before = lib.read_bytes()
    sj = tmp_path / "screen.json"
    sj.write_text(json.dumps(_screen(lib, {"stand": "ok", "wave_right": "warn", "arm_swing": "ok",
                                           "squat_lite": "HIGH"})), encoding="utf-8")
    out_dir = tmp_path / "elsewhere"
    assert chl.main([str(lib), "--screen-json", str(sj), "--out-dir", str(out_dir)]) == 0
    assert lib.read_bytes() == before
    new = json.loads((out_dir / "src_hw.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in new["clips"]] == ["stand", "wave_right", "arm_swing"]
    assert new["curation"]["excluded"] == ["squat_lite"]
    assert new["curation"]["source_manifest_file_sha256"] == hashlib.sha256(before).hexdigest()
    assert new["curation"]["counts"] == {"HIGH": 1, "warn": 1, "ok": 2, "unscreened": 0}
    for c in new["clips"]:  # relative paths re-based to the new directory still resolve to the same bytes
        assert (out_dir / c["npz"]).resolve().is_file()
    # second run refuses to overwrite; unscreened clip refuses with exit 2
    assert chl.main([str(lib), "--screen-json", str(sj), "--out-dir", str(out_dir)]) == 3
    sj.write_text(json.dumps(_screen(lib, {"stand": "ok"})), encoding="utf-8")
    assert chl.main([str(lib), "--screen-json", str(sj), "--name", "x"]) == 2
    assert not (lib.parent / "x.json").exists()


def test_cli_refuses_to_overwrite_source(lib, tmp_path):
    sj = tmp_path / "screen.json"
    sj.write_text(json.dumps(_screen(lib, {n: "ok" for n in CLIPS})), encoding="utf-8")
    with pytest.raises(SystemExit):
        chl.main([str(lib), "--screen-json", str(sj), "--name", "src", "--force"])
    assert json.loads(lib.read_text(encoding="utf-8"))["name"] == "src"


def test_real_screen_on_synthetic_clips(lib):
    pytest.importorskip("mujoco")
    assert chl.main([str(lib), "--dry-run"]) == 0


def test_cli_screen_json_ambient_is_checked_and_recorded(lib, tmp_path):
    """Item 363: a pre-thermal (no ambient_C) screen is refused, a mismatching --ambient_C is refused, and the
    curation records the ambient the severities were computed at."""
    sj = tmp_path / "screen.json"
    res = _screen(lib, {n: "ok" for n in CLIPS})
    legacy = {k: v for k, v in res.items() if k not in ("ambient_C", "thermal")}
    legacy["schema"] = "dropbear-motion-duty-screen-v1"
    sj.write_text(json.dumps(legacy), encoding="utf-8")
    assert chl.main([str(lib), "--screen-json", str(sj), "--name", "a"]) == 2
    assert not (lib.parent / "a.json").exists()
    sj.write_text(json.dumps(res), encoding="utf-8")
    assert chl.main([str(lib), "--screen-json", str(sj), "--name", "b", "--ambient_C", "25"]) == 2
    assert not (lib.parent / "b.json").exists()
    assert chl.main([str(lib), "--screen-json", str(sj), "--name", "c", "--ambient_C", "45"]) == 0
    cur = json.loads((lib.parent / "c.json").read_text(encoding="utf-8"))["curation"]
    assert cur["screen"]["ambient_C"] == 45.0 and "thermal" in cur["screen"]


def test_real_screen_ambient_passthrough(lib, tmp_path):
    pytest.importorskip("mujoco")
    for amb in (25.0, 45.0):
        assert chl.main([str(lib), "--ambient_C", str(amb), "--name", f"t{amb:g}", "--out-dir", str(tmp_path)]) == 0
        cur = json.loads((tmp_path / f"t{amb:g}.json").read_text(encoding="utf-8"))["curation"]
        assert cur["screen"]["ambient_C"] == amb
        assert cur["screen"]["thresh"]["sus"][0] == {25.0: 1.0905, 45.0: 0.9339}[amb]
    assert chl.main([str(lib), "--name", "tdef", "--out-dir", str(tmp_path)]) == 0  # default = design ambient
    assert json.loads((tmp_path / "tdef.json").read_text(encoding="utf-8"))["curation"]["screen"]["ambient_C"] == 45.0
