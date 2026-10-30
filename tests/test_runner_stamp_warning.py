"""Item 261: LowState fixtures carry stamp_ns=0 (the transport fills a real same-host stamp), and the runner's
non-same-host stamp path (warning, stamp age dropped, tick re-use / tick lag gating only) is tested explicitly."""
from __future__ import annotations

import time

import numpy as np
import pytest

import deploy_fixtures as fx
from test_state_freshness_gate import _clock_tick, _run_wall


def test_fixture_low_state_leaves_stamp_to_transport():
    assert fx.low_state(5, np.zeros(22)).stamp_ns == 0


def _stamped_robot(ep, stamp_fn, clock_fn=None):
    """500 Hz fake robot that writes ``stamp_fn()`` into every LowState (non-zero => the transport keeps it).

    ``clock_fn(elapsed_s)`` optionally maps robot wall time to the time its tick counter reads; returning ``None``
    publishes nothing (a silent publisher or bus stall), a value behind ``elapsed_s`` delivers frames whose tick trails
    wall time (a backlogged link). Ticks never move backwards (item 283)."""
    import threading

    from dropbear_wbc.sdk.transport import LOWCMD, LOWSTATE, ChannelPublisher, ChannelSubscriber
    from dropbear_wbc.sdk.types import LowCmd, LowState

    class R(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.running, self.ticks = True, 0
            self.q = np.zeros(22)
            self.pub_t = {}  # tick -> perf_counter() at publish (item 288: attribute state faults to publisher gaps)

        def run(self):
            pub = ChannelPublisher(LOWSTATE, LowState, endpoints=ep, role="robot")
            sub = ChannelSubscriber(LOWCMD, LowCmd, endpoints=ep, role="robot")
            pub.Init()
            sub.Init()
            mode = np.zeros(22, dtype=np.uint8)
            nxt = t0 = time.perf_counter()
            try:
                while self.running:
                    cmd = sub.Read(timeout=0.0)
                    if cmd is not None:
                        self.q += 0.05 * (cmd.motor.q.astype(float) - self.q)
                        mode = np.asarray(cmd.motor.mode, dtype=np.uint8).copy()
                    now = time.perf_counter()
                    t_tick = now if clock_fn is None else clock_fn(now - t0)
                    if t_tick is not None and clock_fn is not None:
                        t_tick += t0
                    # clock-driven, not publish-counted (item 280)
                    tick = self.ticks if t_tick is None else _clock_tick(t0, t_tick)
                    if tick <= self.ticks:
                        nxt += 0.002
                        time.sleep(max(0.0, nxt - time.perf_counter()))
                        continue
                    self.ticks = tick
                    st = fx.low_state(self.ticks, self.q)
                    st.stamp_ns = int(stamp_fn())
                    st.motor.mode[:] = mode
                    pub.Write(st)
                    self.pub_t[self.ticks] = time.perf_counter()
                    nxt += 0.002
                    time.sleep(max(0.0, nxt - time.perf_counter()))
            finally:
                pub.Close()
                sub.Close()

    return R()


def _unexplained_state_faults(tmp_path, robot, max_age_s=0.030):
    """State-gate fault steps the fake robot's own publish timing does not explain (item 288).

    Under host load the robot thread can miss its 2 ms period by more than ``state_max_age_s`` (30 ms); the runner then
    re-uses a tick for > 30 ms and correctly records ``state_stale`` (absorbed by the hold-last grace). That is a real
    stale input, not a stamp bug. A fault step is *explained* when the robot's next publish after the faulted tick came
    more than ``max_age_s / 2`` later (the rest is ZMQ delivery / runner read latency); a fault while the robot kept
    publishing at its 2 ms period is the false trip these tests guard against."""
    import json

    pub = robot.pub_t
    ticks = sorted(pub)
    nxt = dict(zip(ticks, ticks[1:]))
    bad = []
    for line in (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if not rec.get("state_fault"):
            continue
        k = rec["tick"]
        gap = (pub[nxt[k]] - pub[k]) if k in nxt else float("inf")  # no later publish: the robot stopped
        if gap <= 0.5 * max_age_s:
            bad.append({"t": rec["t"], "tick": k, "fault": rec["state_fault"], "robot_gap_ms": 1e3 * gap})
    return bad


# placeholder 1 ns, a clock 10 s ahead (negative age), a clock 1 h behind (age >> 1 s), and a same-host control.
_CASES = {
    "placeholder": (lambda: 1, False),
    "ahead_10s": (lambda: time.perf_counter_ns() + 10_000_000_000, False),
    "behind_1h": (lambda: max(1, time.perf_counter_ns() - 3_600_000_000_000), False),
    "same_host": (lambda: 0, True),
}
_OFFSET = {"placeholder": 0, "ahead_10s": 2, "behind_1h": 4, "same_host": 6}


@pytest.mark.parametrize("case", list(_CASES))
def test_runner_non_same_host_stamp_warns_and_gates_on_ticks(tmp_path, capsys, case):
    from dropbear_wbc.sdk.transport import Endpoints

    stamp_fn, expect_stamp_age = _CASES[case]
    if case == "behind_1h" and time.perf_counter_ns() < 3_600_000_000_000:
        stamp_fn = lambda: 1  # noqa: E731  (fresh boot: perf_counter < 1 h; still a non-same-host stamp)
    ep = Endpoints(cmd_port=27601 + _OFFSET[case], state_port=27602 + _OFFSET[case])
    robot = _stamped_robot(ep, stamp_fn)
    s = _run_wall(tmp_path, ep, robot)
    out = capsys.readouterr().out
    warned = "is not a same-host perf_counter stamp" in out
    assert warned is (not expect_stamp_age), out[-2000:]
    assert s["state_gate"]["stamp_age"] is expect_stamp_age, s["state_gate"]
    names = [t["transition"] for t in s["transitions"]]
    # a foreign stamp must neither trip state_stale (seconds of fake age) nor block the policy
    assert "hold->policy" in names and not any("state_" in n for n in names), names
    # Item 288: a loaded host may stall the fake robot past 30 ms (1/6 concurrent runs got {'state_stale': 2}); such
    # faults are real and stay sub-grace. Every fault step must be explained by a robot publish gap.
    assert set(s["state_gate"]["fault_steps"]) <= {"state_stale"}, s["state_gate"]
    assert _unexplained_state_faults(tmp_path, robot) == [], s["state_gate"]


def test_publisher_stall_faults_are_attributed_not_hidden(tmp_path):
    """A 100 ms publisher stall during MOVE_TO_DEFAULT (no trip outside POLICY) yields state_stale steps that the
    attribution explains, so the relaxed assertion above still sees real faults and cannot mask a false trip."""
    from dropbear_wbc.sdk.transport import Endpoints

    stamp = lambda: max(1, time.perf_counter_ns() - 3_600_000_000_000)  # noqa: E731
    if time.perf_counter_ns() < 3_600_000_000_000:
        stamp = lambda: 1  # noqa: E731
    ep = Endpoints(cmd_port=27609, state_port=27610)
    robot = _stamped_robot(ep, stamp, clock_fn=lambda e: None if 0.25 <= e < 0.35 else e)
    s = _run_wall(tmp_path, ep, robot)
    names = [t["transition"] for t in s["transitions"]]
    assert "hold->policy" in names and not any("state_" in n for n in names), names
    assert s["state_gate"]["fault_steps"].get("state_stale", 0) >= 3, s["state_gate"]
    assert _unexplained_state_faults(tmp_path, robot) == [], s["state_gate"]


def test_clock_tick_skips_instead_of_trailing():
    """A fake robot iteration that runs 13 ms late reports the tick of its own clock (item 280)."""
    assert _clock_tick(10.0, 10.0) == 1
    assert _clock_tick(10.0, 10.0019) == 1
    assert _clock_tick(10.0, 10.002) == 2
    assert _clock_tick(10.0, 10.015) == 8  # 13 ms late after tick 2: skips 3..7, no lag vs the wall clock
