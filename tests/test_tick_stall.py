"""Item 263: TickStallWatch (sim-clock loops abort when LowState arrives but its tick stops advancing)."""

import re
from pathlib import Path

import pytest

from dropbear_wbc.sdk.types import TickStallError, TickStallWatch

ROOT = Path(__file__).resolve().parents[1]


def test_advancing_tick_never_trips():
    w = TickStallWatch(100, 0.5, now=0.0)
    for i in range(1, 1000):
        w.check(100 + i, now=i * 0.01)  # 10 s of slow but advancing ticks


def test_frozen_tick_trips_after_stall_s():
    w = TickStallWatch(100, 0.5, now=0.0)
    w.check(100, now=0.4)
    with pytest.raises(TickStallError, match="frozen at tick 100"):
        w.check(100, now=0.51)


def test_restart_trips_until_tick_passes_old_max():
    w = TickStallWatch(1000, 0.5, now=0.0)
    w.check(0, now=0.1)
    w.check(500, now=0.4)
    with pytest.raises(TickStallError, match="producer restart"):
        w.check(600, now=0.6)


def test_disabled():
    w = TickStallWatch(1, 0.0, now=0.0)
    w.check(1, now=1e6)


def test_is_a_timeout_error():
    assert issubclass(TickStallError, TimeoutError)


@pytest.mark.parametrize("tool", ["tools/policy_runner.py", "tools/teleop_arm.py"])
def test_sim_clock_wait_loops_check_the_stall_watch(tool):
    src = (ROOT / tool).read_text(encoding="utf-8")
    assert "--tick-stall-s" in src
    assert len(re.findall(r"stall\.check\(st\.tick", src)) >= 1
