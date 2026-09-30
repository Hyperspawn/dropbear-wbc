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


def test_background_reload_with_real_ik(tmp_path):
    """The real rebuild (DropbearArmIK + measured elbow tables) runs off the loop; the loop keeps 50 Hz."""
    from dropbear_wbc.kinematics.semantic import DEFAULT_CALIBRATION
    from dropbear_wbc.teleop.arm_ik import DropbearArmIK

    if not DEFAULT_CALIBRATION.exists():
        pytest.skip("no calibration")
    calib = tmp_path / "calib.json"
    data = json.loads(DEFAULT_CALIBRATION.read_text())
    calib.write_text(json.dumps(data))
    ik0 = DropbearArmIK(calib)
    r = BackgroundReloader(calib, lambda: (DropbearArmIK(calib), None), ik0.sha256, poll_s=0.05).start()
    try:
        data["created"] = str(data.get("created")) + "-touched"
        calib.write_text(json.dumps(data))
        gaps, got, last = [], None, time.perf_counter()
        t_end = time.perf_counter() + 4.0
        while time.perf_counter() < t_end and got is None:
            got = r.take()
            ik0.fk_both(ik0.rest_q())  # some control-loop work
            time.sleep(0.02)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now
        assert got is not None, f"no rebuild within 4 s (errors {r.errors})"
        print(f"real rebuild {got[2]:.3f} s off-loop; worst loop gap {1e3 * max(gaps):.1f} ms")
        assert max(gaps) < 0.1  # the bridge watchdog is 100 ms (CONTRACTS 6.1)
    finally:
        r.close()
