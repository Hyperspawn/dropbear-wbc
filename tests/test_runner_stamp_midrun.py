"""Item 281: the runner re-validates the LowState stamp on every frame, not only the first one. A publisher whose
clock changes mid-run (restart on another host, placeholder stamp, clock step) must neither trip state_stale on fresh
frames (stamp seconds behind) nor silently disable the transport-age gate (stamp ahead -> negative age read as 0):
the runner warns once, drops transport age for the rest of the run and keeps gating on tick re-use and tick lag."""
from __future__ import annotations

import json
import time

import pytest

import deploy_fixtures as fx

from test_runner_stamp_warning import _stamped_robot
from test_state_freshness_gate import _run_wall

_SWITCH_S = 1.2  # robot wall time of the switch; POLICY starts ~0.7 s into the 2.0 s run


def _switching(foreign):
    t_start: list[float] = []

    def stamp():
        now = time.perf_counter()
        if not t_start:
            t_start.append(now)
        return 0 if now - t_start[0] < _SWITCH_S else foreign()  # 0 -> the transport stamps on this host

    return stamp


_CASES = {
    "behind_1h": lambda: max(1, time.perf_counter_ns() - 3_600_000_000_000),
    "ahead_10s": lambda: time.perf_counter_ns() + 10_000_000_000,
    "placeholder": lambda: 1,
}
_OFFSET = {"behind_1h": 0, "ahead_10s": 2, "placeholder": 4}


@pytest.mark.parametrize("case", list(_CASES))
def test_runner_midrun_foreign_stamp_falls_back_to_tick_gating(tmp_path, capsys, case):
    from dropbear_wbc.sdk.transport import Endpoints

    foreign = _CASES[case]
    if case == "behind_1h" and time.perf_counter_ns() < 3_600_000_000_000:
        foreign = lambda: 1  # noqa: E731  (fresh boot: perf_counter < 1 h)
    ep = Endpoints(cmd_port=27621 + _OFFSET[case], state_port=27622 + _OFFSET[case])
    robot = _stamped_robot(ep, _switching(foreign))
    s = _run_wall(tmp_path, ep, robot)
    out = capsys.readouterr().out
    assert "is not a same-host perf_counter stamp; gating on tick re-use and tick lag only from here on" in out, \
        out[-2000:]
    assert "first LowState stamp age" not in out  # the first frame was a valid same-host stamp
    g = s["state_gate"]
    assert g["stamp_age"] is False and g["stamp_fallback"] is not None, g
    assert g["stamp_fallback"]["t"] > 0.7, g["stamp_fallback"]  # switched after POLICY entry
    names = [t["transition"] for t in s["transitions"]]
    assert "hold->policy" in names and not any("state_" in n for n in names), names
    # Item 289: a fault step is allowed only when the robot's own publish gap explains it (loaded host), never on
    # fresh frames; a strict == {} failed under load although the gate was right (notes/288).
    assert set(g["fault_steps"]) <= {"state_stale"}, g
    assert fx.unexplained_state_faults(tmp_path / "steps.jsonl", robot.pub_t) == [], g
    assert not any(n.startswith("policy->passive") for n in names), names  # ends via motion_end, not a trip
    # Item 283: foreign stamps after the fallback are not ages; base logged ~1 h / 10 s "age" in summary and steps.
    assert s["state_age_ms"] is not None and s["state_age_ms"]["max"] < 1000.0, s["state_age_ms"]
    assert s["state_age_invalid_frames"] > 0, s
    steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["state_age_ms"] is None or 0.0 <= r["state_age_ms"] < 1000.0 for r in steps)
    assert steps[-1]["state_age_ms"] is None


def test_runner_same_host_stamp_has_no_fallback(tmp_path, capsys):
    from dropbear_wbc.sdk.transport import Endpoints

    ep = Endpoints(cmd_port=27627, state_port=27628)
    s = _run_wall(tmp_path, ep, _stamped_robot(ep, lambda: 0))
    assert "is not a same-host perf_counter stamp" not in capsys.readouterr().out
    assert s["state_gate"]["stamp_age"] is True and s["state_gate"]["stamp_fallback"] is None


# Item 283: after the fallback the transport age is gone, so a really stale stream must still trip state_stale via the
# two tick rules alone. Robot wall time: stamp switches at 1.2 s, the stream degrades from 1.25 s; POLICY runs from
# ~0.7 s until the 40-frame fixture motion ends (~1.5 s), so the trip has to land inside ~0.25 s.
_DEGRADE_S = 1.25
_DEGRADE = {
    # publisher/bus goes silent for 0.25 s: the runner re-uses the last frame -> tick re-use age
    "silent_stall": lambda e: None if _DEGRADE_S <= e < _DEGRADE_S + 0.25 else e,
    # frames keep arriving with fresh ticks, but the tick clock runs at 0.5x (backlogged link): only tick lag sees it
    "backlog": lambda e: e if e < _DEGRADE_S else _DEGRADE_S + 0.5 * (e - _DEGRADE_S),
}
_DEGRADE_OFFSET = {"silent_stall": 0, "backlog": 2}


@pytest.mark.parametrize("case", list(_DEGRADE))
def test_runner_stale_stream_after_stamp_fallback_still_trips(tmp_path, capsys, case):
    from dropbear_wbc.sdk.transport import Endpoints

    ep = Endpoints(cmd_port=27631 + _DEGRADE_OFFSET[case], state_port=27632 + _DEGRADE_OFFSET[case])
    s = _run_wall(tmp_path, ep, _stamped_robot(ep, _switching(lambda: 1), clock_fn=_DEGRADE[case]))
    assert "is not a same-host perf_counter stamp" in capsys.readouterr().out
    g = s["state_gate"]
    assert g["stamp_age"] is False and g["stamp_fallback"] is not None, g
    names = [t["transition"] for t in s["transitions"]]
    assert "hold->policy" in names, names
    trip = [t for t in s["transitions"] if t["transition"] == "policy->passive(state_stale)"]
    assert trip, [(round(t["t"], 3), t["transition"]) for t in s["transitions"]]
    assert trip[0]["t"] > g["stamp_fallback"]["t"], (trip, g)  # tripped after the fallback, on tick rules only
    assert g["fault_steps"].get("state_stale", 0) >= 2, g
