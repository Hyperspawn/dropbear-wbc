import json
from tools import scoreboard as sb


def _loco(falls=0, max_nm=60.0, peak=100.0):
    return {"finite": True, "falls_total": falls, "by_scenario": {"fwd": {"envs": 2}, "back": {"envs": 2}},
            "motor_torque": {"per_motor": {"hip": {"peak_limit_Nm": peak, "max_Nm": max_nm}}}}


def test_pass():
    r = sb.evaluate("locomotion", _loco())
    assert r["pass"] and r["units"] == 4 and abs(r["torque_x"] - 1.2) < 1e-9


def test_fall_and_torque_fail():
    r = sb.evaluate("locomotion", _loco(falls=1, max_nm=90.0))
    assert not r["pass"] and len(r["why"]) == 2


def test_tracking_and_missing(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "play_1.json").write_text(json.dumps(
        {"finite": True, "falls": {"envs_with_a_fall": 0}, "per_clip": {"c": {"envs": [0, 1]}}}))
    reg = {"skills": {"t": {"kind": "tracking", "summary_glob": "a/play_*.json"},
                      "m": {"kind": "locomotion", "summary_glob": "nope/*.json"}}}
    rows = sb.build(reg, tmp_path)
    assert rows[0]["pass"] and rows[0]["units"] == 2 and not rows[1]["pass"]
    assert "1/2" in sb.render(rows)
