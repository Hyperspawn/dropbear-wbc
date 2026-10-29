"""Item 284: teleop_arm logs latency.state_age_ms only for same-host stamps (same rule as policy_runner 0043).

A foreign stamp (other host's clock, placeholder, other epoch) used to be recorded as (t_rx - stamp_ns), i.e.
hours of "age" in the parquet/npz and the summary percentiles. It is now NaN and counted as invalid.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.kinematics.semantic import DEFAULT_CALIBRATION
from dropbear_wbc.sdk.transport import Endpoints

import test_teleop_loopback as loop

H = 3600 * 10**9


T_RX = 10 * H


@pytest.mark.parametrize("stamp_ns,expect_ms", [
    (0, None),                      # unset
    (T_RX - H, None),               # 1 h behind (other epoch / host)
    (T_RX + 10 * 10**9, None),      # 10 s ahead (other host clock)
    (T_RX - 5 * 10**6, 5.0),        # 5 ms old
    (T_RX - 999 * 10**6, 999.0),    # 999 ms: still an age
    (T_RX - 10**9, None),           # 1 s or more: not an age
])
def test_same_host_age_ms_rule(stamp_ns, expect_ms):
    age = loop._load_tool().same_host_age_ms(T_RX, stamp_ns)
    if expect_ms is None:
        assert math.isnan(age)
    else:
        assert age == pytest.approx(expect_ms)


class ForeignStampRobot(loop.FakeArmRobot):
    """Publishes stamps 1 h behind this host's perf_counter (a cross-host / other-epoch clock)."""

    def state(self):
        st = super().state()
        st.stamp_ns = time.perf_counter_ns() - H
        return st


@pytest.mark.skipif(not DEFAULT_CALIBRATION.exists(), reason="no semantic calibration JSON")
@pytest.mark.parametrize("robot_cls,foreign", [(loop.FakeArmRobot, False), (ForeignStampRobot, True)])
def test_session_state_age_ignores_foreign_stamps(tmp_path, monkeypatch, robot_cls, foreign):
    from dropbear_wbc.teleop.arm_ik import DropbearArmIK

    ep = Endpoints(cmd_port=28565 + 2 * int(foreign), state_port=28566 + 2 * int(foreign))
    monkeypatch.setattr(loop, "EP", ep)
    robot = robot_cls(DropbearArmIK())
    robot.start()
    try:
        tool = loop._load_tool()
        rec, summary_path = tmp_path / "session", tmp_path / "summary.json"
        rc = tool.main(["--device", "scripted", "--duration", "2.0", "--ramp-s", "0.5", "--move-in-s", "0.3",
                        "--settle-s", "0.2", "--home-s", "0.3", "--cmd-port", str(ep.cmd_port),
                        "--state-port", str(ep.state_port), "--connect-timeout-s", "20", "--record", str(rec),
                        "--summary", str(summary_path), "--task", "state age test"])
    finally:
        robot.running = False
        robot.join(timeout=5)
    s = json.loads(summary_path.read_text())
    assert rc == 0, s.get("traceback")
    lat = s["analysis"]["latency.state_age_ms"]
    raw = np.load(rec / "extras" / "episode_000000.npz")["latency__state_age_ms"].reshape(-1)
    if foreign:
        assert lat["n"] == 0 and lat["invalid_frames"] > 10 and "max" not in lat, lat
        assert np.isnan(raw).all(), np.nanmax(raw)
    else:
        assert lat["n"] > 10 and lat["invalid_frames"] == 0, lat
        assert 0.0 <= lat["p50"] and lat["max"] < 1000.0, lat
        assert np.isfinite(raw).all() and raw.max() < 1000.0
