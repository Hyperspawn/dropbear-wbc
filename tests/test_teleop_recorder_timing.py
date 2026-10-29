"""Teleop recorder timing (LeRobot v2.1) and the non-blocking calibration reload (review fixes 2026-09-24). CPU only."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (sys.path)
from dropbear_wbc.teleop.recorder import EXTRA_COLUMNS, SessionRecorder, check_timing
from dropbear_wbc.teleop.reload import BackgroundReloader


def _row(t_sim: float, skipped: int, k: int) -> dict:
    row = {"observation.state": np.zeros(10), "action": np.zeros(10), "timestamp": t_sim}
    for name, (w, _) in EXTRA_COLUMNS.items():
        row[name] = np.zeros(w)
    row["time.sim_s"] = t_sim
    row["time.skipped_steps"] = skipped
    row["time.tick"] = k
    return row


def test_uniform_lerobot_timestamps_and_reported_drift(tmp_path):
    fps = 50.0
    rec = SessionRecorder(tmp_path / "s", fps=fps, task="t")
    t = 0.0
    for k in range(100):
        skipped = 2 if k in (30, 60) else 0  # an overrunning loop skipped 2 control periods twice
        t += (1 + skipped) / fps
        rec.add(**_row(t, skipped, k))
    written = rec.close({"x": 1})
    timing = written["timing"]
    assert timing["lerobot_timestamps_ok"] and timing["lerobot_max_abs_err_s"] <= 1e-4
    assert timing["skipped_steps_total"] == 4 and timing["frames_after_a_skip"] == 2
    assert timing["real_clock_uniform"] is False and timing["sim_clock_drift_max_abs_s"] == pytest.approx(4 / fps)
    root = tmp_path / "s"
    stats = json.loads((root / "meta" / "episodes_stats.jsonl").read_text().splitlines()[0])
    assert stats["episode_index"] == 0 and stats["stats"]["frame_index"]["count"] == [100]
    chk = check_timing(root)
    assert chk["episodes_stats_jsonl"] and chk["lerobot_timestamps_ok"]
    if written.get("parquet"):  # pyarrow present: the parquet itself passes LeRobot's check
        assert chk["timestamp_source"] == "parquet" and chk["lerobot_v21_valid"]


def test_background_reload_never_blocks_the_loop(tmp_path):
    f = tmp_path / "calib.json"
    f.write_text('{"v": 1}')
    built = []

    def slow_factory():
        time.sleep(0.5)  # ~ DropbearArmIK._load on this machine
        built.append(f.read_text())
        return ("ik", "grav")

    import hashlib

    r = BackgroundReloader(f, slow_factory, hashlib.sha256(f.read_bytes()).hexdigest(), poll_s=0.05).start()
    try:
        gaps, swapped, last = [], None, time.perf_counter()
        t_end = time.perf_counter() + 2.0
        touched = False
        while time.perf_counter() < t_end:
            if not touched and time.perf_counter() > t_end - 1.8:
                f.write_text('{"v": 2}')
                touched = True
            got = r.take()
            if got is not None:
                swapped = got
            time.sleep(0.02)  # 50 Hz loop
            now = time.perf_counter()
            gaps.append(now - last)
            last = now
        assert swapped is not None and swapped[1] == ("ik", "grav") and swapped[2] >= 0.45
        assert max(gaps) < 0.05, f"control loop blocked for {max(gaps) * 1e3:.0f} ms"
    finally:
        r.close()


def _loop_gaps(ik, until, take=None):
    """Run the 50 Hz control-loop stand-in until until() is true; return (gaps, last take() result)."""
    gaps, got, last = [], None, time.perf_counter()
    while not until(got):
        if take is not None:
            got = take()
        ik.fk_both(ik.rest_q())  # some control-loop work
        time.sleep(0.02)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now
    return gaps, got


def test_background_reload_with_real_ik(tmp_path):
    """The real rebuild (DropbearArmIK + measured elbow tables) runs off the loop; the loop keeps 50 Hz.

    A shared CI host can stall the whole process (CI run 40821097880: one 155 ms gap while the
    rebuild itself took 0.35 s; locally <= 48 ms even pinned to 1 CPU with 3 CPU hogs). Such a
    stall is not the rebuild blocking the loop, so a slow gap is re-checked against a control
    window of the same loop with no rebuild running: only a gap the control window does not show
    fails the test.
    """
    from dropbear_wbc.kinematics.semantic import DEFAULT_CALIBRATION
    from dropbear_wbc.teleop.arm_ik import DropbearArmIK

    if not DEFAULT_CALIBRATION.exists():
        pytest.skip("no calibration")
    limit_s = 0.1  # the bridge watchdog is 100 ms (CONTRACTS 6.1)
    calib = tmp_path / "calib.json"
    data = json.loads(DEFAULT_CALIBRATION.read_text())
    calib.write_text(json.dumps(data))
    ik0 = DropbearArmIK(calib)
    r = BackgroundReloader(calib, lambda: (DropbearArmIK(calib), None), ik0.sha256, poll_s=0.05).start()
    try:
        data["created"] = str(data.get("created")) + "-touched"
        calib.write_text(json.dumps(data))
        t_end = time.perf_counter() + 4.0
        gaps, got = _loop_gaps(ik0, lambda g: g is not None or time.perf_counter() >= t_end, r.take)
        assert got is not None, f"no rebuild within 4 s (errors {r.errors})"
        print(f"real rebuild {got[2]:.3f} s off-loop; worst loop gap {1e3 * max(gaps):.1f} ms")
    finally:
        r.close()
    if max(gaps) < limit_s:
        return
    n = len(gaps)
    control, _ = _loop_gaps(ik0, lambda _g, c=iter(range(n + 1)): next(c) >= n)
    print(f"control window (no rebuild, {n} iters): worst gap {1e3 * max(control):.1f} ms")
    if max(control) >= limit_s:
        pytest.skip(f"host stalled {1e3 * max(control):.0f} ms with no rebuild running; gap not attributable")
    assert max(gaps) < limit_s, f"rebuild blocked the loop {1e3 * max(gaps):.0f} ms (control {1e3 * max(control):.0f} ms)"
