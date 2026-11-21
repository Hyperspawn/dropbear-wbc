"""Item 349: an hw_* / legacy_hwarm library run refuses a motion library without a valid hardware curation."""
from __future__ import annotations

import ast
import json
import shutil
import sys

import pytest

import sdk_test_paths  # noqa: F401  (adds source/)

from dropbear_wbc.tasks.tracking.library_curation import (
    CURATION_SCHEMA, UncuratedLibraryError, check_hw_library, curation_problems, is_hw_profile,
)

sys.path.insert(0, str(sdk_test_paths.ROOT / "tools"))
import curate_hw_library as chl  # noqa: E402

ROOT = sdk_test_paths.ROOT
SYN = ROOT / "data/motions_v6ts/synthetic"
CLIPS = ("stand", "wave_right", "arm_swing", "squat_lite")


def _curated(policy_high="exclude"):
    dec = [{"name": "stand", "severity": "ok", "action": "keep", "weight_in": 1.0, "weight_out": 1.0},
           {"name": "wave_right", "severity": "HIGH", "action": "exclude", "weight_in": 1.0, "weight_out": None},
           {"name": "arm_swing", "severity": "warn", "action": "keep", "weight_in": 2.0, "weight_out": 2.0}]
    return {"schema": "dropbear-motion-library-v1", "name": "src_hw",
            "clips": [{"name": "stand", "npz": "stand.npz", "weight": 1.0},
                      {"name": "arm_swing", "npz": "arm_swing.npz", "weight": 2.0}],
            "curation": {"schema": CURATION_SCHEMA, "source_manifest": "src", "policy": {"HIGH": policy_high},
                         "counts": {"HIGH": 1, "warn": 1, "ok": 1, "unscreened": 0}, "decisions": dec}}


def _write(tmp_path, m, name="lib.json"):
    p = tmp_path / name
    p.write_text(json.dumps(m), encoding="utf-8")
    return p


@pytest.mark.parametrize("profile,hw", [("hw_v1ie", True), ("hw_cem6", True), ("legacy_hwarm", True),
                                         ("legacy", False), (None, False)])
def test_is_hw_profile(profile, hw):
    assert is_hw_profile(profile) is hw


def test_curated_manifest_passes(tmp_path):
    assert curation_problems(_curated()) == []
    rec, notes = check_hw_library(_write(tmp_path, _curated()), "hw_v1ie")
    assert rec["checked"] and rec["curated"] and not rec["allowed"] and rec["curation"]["source_manifest"] == "src"
    assert "OK" in notes[0]


@pytest.mark.parametrize("policy", ["exclude", "weight=0.25"])
def test_weighting_policies_count_as_curation(policy):
    assert curation_problems(_curated(policy)) == []


@pytest.mark.parametrize("edit,needle", [
    (lambda m: m.pop("curation"), "no 'curation' block"),
    (lambda m: m["curation"].update(schema="other-v0"), "curation schema"),
    (lambda m: m["curation"]["policy"].update(HIGH="keep"), "keeps every HIGH clip"),
    (lambda m: m["clips"].append({"name": "wave_right", "npz": "wave_right.npz"}), "curated as 'exclude'"),
    (lambda m: m["clips"].append({"name": "squat_lite", "npz": "squat_lite.npz"}), "added after curation"),
    (lambda m: m["clips"][1].update(weight=5.0), "weight 5 != curated weight_out 2"),
    (lambda m: m["curation"].update(decisions=[]), "no per-clip decisions"),
])
def test_uncurated_or_edited_refuses_fresh_run(tmp_path, edit, needle):
    m = _curated()
    edit(m)
    probs = curation_problems(m)
    assert any(needle in p for p in probs), probs
    p = _write(tmp_path, m)
    with pytest.raises(UncuratedLibraryError, match="--allow_uncurated"):
        check_hw_library(p, "hw_v1ie")


def test_unscreened_error_decision_refuses():
    m = _curated()
    m["curation"]["decisions"].append({"name": "squat_lite", "severity": "unscreened", "action": "error"})
    m["clips"].append({"name": "squat_lite", "npz": "squat_lite.npz"})
    assert any("curated as 'error'" in p for p in curation_problems(m))


def test_allow_and_continue_only_warn(tmp_path):
    m = _curated()
    m.pop("curation")
    p = _write(tmp_path, m)
    rec, notes = check_hw_library(p, "hw_v1ie", allow_uncurated=True)
    assert rec["allowed"] and rec["curated"] is False and "allowed by --allow_uncurated" in notes[0]
    rec, notes = check_hw_library(p, "hw_v1ie", continue_run=True)
    assert rec["allowed"] and "--continue_run" in notes[0]


def test_sim_profile_and_single_clip_unchecked(tmp_path):
    m = _curated()
    m.pop("curation")
    p = _write(tmp_path, m)
    assert check_hw_library(p, "legacy") == ({"checked": False, "curated": None, "problems": [], "allowed": False,
                                              "curation": None}, [])
    assert check_hw_library(None, "hw_v1ie")[0]["checked"] is False


def test_real_tool_output_passes_and_source_refuses(tmp_path, monkeypatch):
    """End to end with tools/curate_hw_library.py on synthetic clips (fake screen, no mujoco)."""
    d = tmp_path / "data" / "synthetic"
    d.mkdir(parents=True)
    for n in CLIPS:
        for suf in (".npz", ".validation.json"):
            shutil.copy(SYN / f"{n}{suf}", d / f"{n}{suf}")
    libdir = tmp_path / "data" / "libraries"
    libdir.mkdir()
    src = _write(libdir, {"schema": "dropbear-motion-library-v1", "name": "src",
                          "clips": [{"name": n, "npz": f"../synthetic/{n}.npz"} for n in CLIPS]}, "src.json")
    worst = {"stand": "ok", "wave_right": "HIGH", "arm_swing": "warn", "squat_lite": "ok"}
    screen = {"schema": "dropbear-motion-duty-screen-v2", "map": "m", "window_s": 2.0, "lp_hz": 6.0, "basis": "dynamic",
              "payload_kg": 0.0, "ambient_C": 45.0, "thresh": {}, "thermal": {}, "errors": [],
              "clips": [{"clip": n, "path": str(d / f"{n}.npz"), "worst": w,
                         "findings": ([{"motor": "LH_elbow_joint", "metric": "sus", "value": 1.5, "severity": "HIGH",
                                        "basis": "dynamic"}] if w == "HIGH" else [])} for n, w in worst.items()]}
    sj = _write(tmp_path, screen, "screen.json")
    assert chl.main([str(src), "--screen-json", str(sj), "--no-verify"]) == 0
    hw = libdir / "src_hw.json"
    rec, _ = check_hw_library(hw, "hw_v1ie")
    assert rec["curated"] and rec["curation"]["counts"]["HIGH"] == 1
    with pytest.raises(UncuratedLibraryError):
        check_hw_library(src, "hw_v1ie")
    # --high keep: a "curated" file that keeps every HIGH clip is still refused
    assert chl.main([str(src), "--screen-json", str(sj), "--no-verify", "--high", "keep", "--name", "k"]) == 0
    with pytest.raises(UncuratedLibraryError, match="keeps every HIGH clip"):
        check_hw_library(libdir / "k.json", "hw_cem6")


def test_train_py_gates_before_any_write_or_env_build():
    src = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = ast.get_source_segment(src, main)
    i_gate = body.index("check_hw_library(")
    i_prof = body.index("env_cfg.set_actuator_profile(")
    assert i_prof < i_gate < body.index("log_dir.mkdir(") < body.index("gym.make(")
    assert "allow_uncurated=args.allow_uncurated" in body and "continue_run=args.continue_run" in body
    assert '"hw_library_curation": hw_curation' in body
    assert '"--allow_uncurated", action="store_true"' in src


def test_chunked_driver_passes_allow_uncurated_through():
    src = (ROOT / "tools/run_chunked_training.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--allow_uncurated", action="store_true"' in src
    assert 'if args.allow_uncurated:\n            cmd[cmd.index("--headless"):cmd.index("--headless")] = ["--allow_uncurated"]' in src


@pytest.mark.parametrize("lib", ["accepted_v7gen.json", "accepted_v8push.json", "lowbody_v1.json"])
def test_shipped_libraries_are_uncurated(lib):
    """The libraries every hw_* Brev session (3-10) trained on carry no curation: a fresh hw run now refuses them."""
    m = json.loads((ROOT / "data/motions_v6ts/libraries" / lib).read_text(encoding="utf-8"))
    assert curation_problems(m) and "no 'curation' block" in curation_problems(m)[0]
