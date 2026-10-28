"""Item 281: the runner re-validates the LowState stamp on every frame, not only the first one. A publisher whose
clock changes mid-run (restart on another host, placeholder stamp, clock step) must neither trip state_stale on fresh
frames (stamp seconds behind) nor silently disable the transport-age gate (stamp ahead -> negative age read as 0):
the runner warns once, drops transport age for the rest of the run and keeps gating on tick re-use and tick lag."""
from __future__ import annotations

import time

import pytest

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
    s = _run_wall(tmp_path, ep, _stamped_robot(ep, _switching(foreign)))
    out = capsys.readouterr().out
    print("BASE", [t["transition"] for t in s["transitions"]], s["state_gate"]); assert "is not a same-host perf_counter stamp; gating on tick re-use and tick lag only from here on" in out, \
        out[-2000:]
    assert "first LowState stamp age" not in out  # the first frame was a valid same-host stamp
    g = s["state_gate"]
    assert g["stamp_age"] is False and g["stamp_fallback"] is not None, g
    assert g["stamp_fallback"]["t"] > 0.7, g["stamp_fallback"]  # switched after POLICY entry
    names = [t["transition"] for t in s["transitions"]]
    assert "hold->policy" in names and not any("state_" in n for n in names), names
    assert g["fault_steps"] == {}, g
    assert not any(n.startswith("policy->passive") for n in names), names  # ends via motion_end, not a trip


def test_runner_same_host_stamp_has_no_fallback(tmp_path, capsys):
    from dropbear_wbc.sdk.transport import Endpoints

    ep = Endpoints(cmd_port=27627, state_port=27628)
    s = _run_wall(tmp_path, ep, _stamped_robot(ep, lambda: 0))
    assert "is not a same-host perf_counter stamp" not in capsys.readouterr().out
    assert s["state_gate"]["stamp_age"] is True and s["state_gate"]["stamp_fallback"] is None
