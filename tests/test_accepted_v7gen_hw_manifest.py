"""Shipped ``accepted_v7gen_hw.json`` (items 350/366): all 138 clips screened at 45 C, a strict subset of the source."""
from __future__ import annotations

import hashlib
import json

import sdk_test_paths  # noqa: F401  (adds source/)

LIBS = sdk_test_paths.ROOT / "data/motions_v6ts/libraries"
SRC = LIBS / "accepted_v7gen.json"
HW = LIBS / "accepted_v7gen_hw.json"

# Screened only on a machine with every NPZ; these 16 were unscreened in the partial screen (notes 363).
NEWLY_SCREENED_HIGH = {
    "lafan1_armraise_dance2_subject1_v4", "lafan1_dance2_subject1_v4", "lafan1_run1_subject2_v4",
    "Neutral_throw_ball_001__A057_ts1.19_v6", "body_stretch_1_004__A069_ts1.19_v6",
    "dance_basic_chaines_180_R_001__A306_ts1.19_v6", "dance_hiphop_shuffle_square_R_fast_002__A318_ts1.19_v6",
    "high_jump_R_001__A277_ts1.19_v6", "G1_Take_102_ts1.19_v6", "G1_gangnam_style_V01_ts1.19_v6",
}
NEWLY_SCREENED_KEPT = {
    "wave_R_001__A428_ts1.19_v6",  # warn
    "lafan1_walk1_subject1_v4", "Neutral_walk_forward_002__A057_ts1.19_v6",
    "item_pick_up_standing_R_001__A410_ts1.19_v6", "big_light_one_hand_pick_up_front_low_R_005__A508_ts1.19_v6",
    "small_light_one_hand_pick_up_front_low_002__A507_ts1.19_v6",
}


def _load():
    return json.loads(SRC.read_text(encoding="utf-8")), json.loads(HW.read_text(encoding="utf-8"))


def test_source_pin_matches_tracked_manifest():
    _, hw = _load()
    assert hw["curation"]["source_manifest"] == "accepted_v7gen"
    assert hw["curation"]["source_manifest_file_sha256"] == hashlib.sha256(SRC.read_bytes()).hexdigest()


def test_every_source_clip_screened_none_unscreened():
    src, hw = _load()
    c = hw["curation"]
    assert c["counts"] == {"HIGH": 38, "warn": 9, "ok": 91, "unscreened": 0}
    assert c["policy"]["unscreened"] == "error" and c["policy"]["HIGH"] == "exclude" and c["policy"]["warn"] == "keep"
    assert sorted(d["name"] for d in c["decisions"]) == sorted(x["name"] for x in src["clips"])
    assert all(d["severity"] in ("HIGH", "warn", "ok") for d in c["decisions"])
    assert c["screen"]["ambient_C"] == 45.0


def test_kept_clips_are_source_entries_verbatim():
    src, hw = _load()
    by = {x["name"]: x for x in src["clips"]}
    assert len(hw["clips"]) == 100
    for clip in hw["clips"]:
        assert clip == by[clip["name"]]  # same npz (same directory), weight, sha256
    kept = {x["name"] for x in hw["clips"]}
    excluded = set(hw["curation"]["excluded"])
    assert kept | excluded == set(by) and not kept & excluded and len(excluded) == 38
    high = {d["name"] for d in hw["curation"]["decisions"] if d["severity"] == "HIGH"}
    assert excluded == high


def test_previously_unscreened_clips_decided():
    _, hw = _load()
    excluded = set(hw["curation"]["excluded"])
    kept = {x["name"] for x in hw["clips"]}
    assert NEWLY_SCREENED_HIGH <= excluded
    assert NEWLY_SCREENED_KEPT <= kept
    assert "dance_arms_ts1.19_v6" in excluded  # the 45 C flip from notes 363


def test_no_machine_paths():
    text = HW.read_text(encoding="utf-8")
    assert ":\\\\" not in text and ":/" not in text and "/home/" not in text
