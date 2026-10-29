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
                    nxt += 0.002
                    time.sleep(max(0.0, nxt - time.perf_counter()))
            finally:
                pub.Close()
                sub.Close()

    return R()


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
    s = _run_wall(tmp_path, ep, _stamped_robot(ep, stamp_fn))
    out = capsys.readouterr().out
    warned = "is not a same-host perf_counter stamp" in out
    assert warned is (not expect_stamp_age), out[-2000:]
    assert s["state_gate"]["stamp_age"] is expect_stamp_age, s["state_gate"]
    names = [t["transition"] for t in s["transitions"]]
    # a foreign stamp must neither trip state_stale (seconds of fake age) nor block the policy
    assert "hold->policy" in names and not any("state_" in n for n in names), names
    assert s["state_gate"]["fault_steps"] == {}, s["state_gate"]


def test_clock_tick_skips_instead_of_trailing():
    """A fake robot iteration that runs 13 ms late reports the tick of its own clock (item 280)."""
    assert _clock_tick(10.0, 10.0) == 1
    assert _clock_tick(10.0, 10.0019) == 1
    assert _clock_tick(10.0, 10.002) == 2
    assert _clock_tick(10.0, 10.015) == 8  # 13 ms late after tick 2: skips 3..7, no lag vs the wall clock
