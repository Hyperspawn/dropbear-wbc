import json

import pytest

from tools import scoreboard as sb
from dropbear_wbc.motion.names import MOTOR_NAMES

PROFILE = "hw_v1"


def _per_motor(profile=PROFILE, rms=0.5, mx=0.8, clip=0.0, speed=0.5, override=None):
    specs = sb.motor_specs(profile)
    per = {n: {"peak_limit_Nm": sp["peak"], "max_Nm": mx * sp["peak"], "rms_Nm": rms * sp["rated"],
               "clip_frac": clip, "max_speed_rad_s": speed * sp["no_load"]} for n, sp in specs.items()}
    for n, kv in (override or {}).items():
        per[n].update(kv)
    return per


def _loco(falls=0, profile=PROFILE, perturbed=True, per=None, created="2026-10-14T10:00:00+05:30"):
    s = {"finite": True, "falls_total": falls, "actuator_profile": profile, "created": created,
         "eval_design": {"perturbed": perturbed},
         "by_scenario": {"fwd": {"envs": 2}, "back": {"envs": 2}}}
    if per is not False:
        s["motor_torque"] = {"per_motor": per if per is not None else _per_motor()}
    return s


def test_pass():
    r = sb.evaluate("locomotion", _loco())
    assert r["pass"], r["why"]
    assert r["scenarios"] == 2 and r["envs"] == 4
    assert abs(r["torque"]["max_over_peak"]["value"] - 0.8) < 1e-6


def test_fall_and_rms_fail():
    per = _per_motor(override={"LL_knee_actuator_joint": {"rms_Nm": 1.59 * sb.motor_specs(PROFILE)["LL_knee_actuator_joint"]["rated"]}})
    r = sb.evaluate("locomotion", _loco(falls=1, per=per))
    assert not r["pass"] and any("1 falls" in w for w in r["why"])
    assert any("rms 1.59x rated (LL_knee_actuator_joint" in w for w in r["why"])


@pytest.mark.parametrize("key,val,frag", [("clip_frac", 0.02, "clip 2.0%"), ("max_speed_rad_s", 1e3, "no-load")])
def test_clip_and_speed_fail(key, val, frag):
    r = sb.evaluate("locomotion", _loco(per=_per_motor(override={"RH_elbow_joint": {key: val}})))
    assert not r["pass"] and any(frag in w for w in r["why"])


def test_pr1_regression_stiff_knee_hw_never_passes():
    # PR #1 looked rated up by the sim limit: hips at 180 on a 200 sim limit were skipped and passed
    per = {n: {"peak_limit_Nm": 200.0, "max_Nm": 180.0, "rms_Nm": 90.0, "clip_frac": 0.0, "max_speed_rad_s": 1.0}
           for n in MOTOR_NAMES}
    r = sb.evaluate("locomotion", _loco(profile="stiff_knee_hw", per=per))
    assert not r["pass"] and any("no motor map" in w for w in r["why"])


def test_peak_limit_mismatch_and_missing_motor():
    per = _per_motor()
    per["LL_hip_joint"]["peak_limit_Nm"] = 200.0
    del per["RL_Revolute81"]
    r = sb.evaluate("locomotion", _loco(per=per))
    assert not r["pass"]
    assert any("peak limit != motor-map peak" in w for w in r["why"])
    assert any("UNKNOWN for 1 motors" in w for w in r["why"])


def test_unknown_torque_and_nominal_fail():
    r = sb.evaluate("locomotion", _loco(per=False, perturbed=False))
    assert not r["pass"]
    assert any("torque UNKNOWN" in w for w in r["why"]) and any("nominal play" in w for w in r["why"])


def test_tracking_without_torque_block_fails(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "play_1.json").write_text(json.dumps(
        {"finite": True, "falls": {"envs_with_a_fall": 0}, "per_clip": {"c": {"envs": [0, 1]}},
         "actuator_profile": PROFILE, "eval_design": {"perturbed": True}}))
    reg = {"skills": {"t": {"kind": "tracking", "summary_glob": "a/play_*.json"},
                      "m": {"kind": "locomotion", "summary_glob": "nope/*.json"}}}
    rows = sb.build(reg, tmp_path)
    assert not rows[0]["pass"] and rows[0]["envs"] == 2 and any("torque UNKNOWN" in w for w in rows[0]["why"])
    assert rows[1]["why"] == ["no summary"]
    assert "0/2" in sb.render(rows)


def test_select_by_created_and_eligibility(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    # lexically last, newest, but nominal -> not eligible
    (d / "play_custom.json").write_text(json.dumps(_loco(perturbed=False, created="2026-10-14T12:00:00+05:30")))
    (d / "play_2026-10-14_09-00-00.json").write_text(json.dumps(_loco(created="2026-10-14T09:00:00+05:30")))
    (d / "play_2026-10-14_08-00-00.json").write_text(json.dumps(_loco(falls=3, created="2026-10-14T08:00:00+05:30")))
    p, s, n = sb.select(str(d / "play_*.json"))
    assert n == 3 and p.name == "play_2026-10-14_09-00-00.json"
    assert sb.evaluate("locomotion", s)["pass"]


def test_select_falls_back_to_newest_ineligible(tmp_path):
    (tmp_path / "play_a.json").write_text(json.dumps(_loco(profile="legacy", created="2026-10-14T09:00:00+05:30")))
    p, s, n = sb.select(str(tmp_path / "play_*.json"))
    assert p.name == "play_a.json" and not sb.evaluate("locomotion", s)["pass"]


def test_allow_profiles_and_every_hw_profile_resolves():
    for prof in sb.HW_PROFILE_MAPS:
        assert set(sb.motor_specs(prof)) == set(MOTOR_NAMES)
    r = sb.evaluate("locomotion", _loco(), allow_profiles=("hw_v1_cad",))
    assert not r["pass"] and any("allow_profiles" in w for w in r["why"])


def test_registry_glob_points_at_library_runner():
    reg = json.loads((sb.ROOT / "tools" / "skills.json").read_text())
    assert "dropbear_tracking_library/" in reg["skills"]["motion_library_tracker"]["summary_glob"]
