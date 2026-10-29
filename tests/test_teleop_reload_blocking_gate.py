"""The real-IK reload timing test must still fail when the rebuild really blocks the loop (CPU only)."""
from __future__ import annotations

import time

import pytest

import sdk_test_paths  # noqa: F401  (sys.path)
import test_teleop_recorder_timing as t


def test_gil_holding_rebuild_still_fails(tmp_path, monkeypatch):
    """A rebuild that holds the GIL for 0.3 s blocks the loop; the control window has no such gap."""
    from dropbear_wbc.teleop import arm_ik

    real = arm_ik.DropbearArmIK
    calls = {"n": 0}

    class Blocking(real):
        def __init__(self, *a, **k):
            calls["n"] += 1
            if calls["n"] > 1:  # the background rebuild, not ik0
                time.sleep(0.05)  # let the loop thread get going
                _hold_gil(0.3)
            super().__init__(*a, **k)

    monkeypatch.setattr(arm_ik, "DropbearArmIK", Blocking)
    with pytest.raises(AssertionError, match="rebuild blocked the loop"):
        t.test_background_reload_with_real_ik(tmp_path)


def _hold_gil(seconds: float) -> None:
    import ctypes
    import sys

    if sys.platform.startswith("win"):
        pytest.skip("usleep via libc is POSIX only")
    libc = ctypes.PyDLL(None)  # PyDLL keeps the GIL held across the call
    libc.usleep(int(seconds * 1e6))
