"""The conftest host lock serialises fixed-port loopback tests across concurrent pytest runs."""
from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import conftest

TESTS = Path(__file__).resolve().parent
LOOPBACK = {"test_sdk_transport.py", "test_runner_loopback.py", "test_lowcmd_stop_signals.py",
            "test_teleop_loopback.py", "test_stop_cause.py"}

HOLDER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    import conftest
    lock = conftest.HostLock(Path(sys.argv[2]), 30.0)
    waited = lock.acquire()
    print(f"{time.time():.6f} {waited:.6f}", flush=True)
    time.sleep(float(sys.argv[3]))
    lock.release()
""")


def test_fixed_port_modules_are_locked_and_free_port_modules_are_not():
    locked = {p.name for p in TESTS.glob("test_*.py") if conftest.module_uses_fixed_ports(p)}
    assert LOOPBACK <= locked
    assert "test_publisher_close_flush.py" not in locked  # binds tcp://127.0.0.1:*
    assert "test_teleop_devices.py" not in locked  # _free_port()
    assert "test_loopback_port_plan.py" not in locked  # only scans source


def test_second_process_waits_for_the_holder(tmp_path):
    lock = tmp_path / "ports.lock"
    hold_s = 0.6
    a = subprocess.Popen([sys.executable, "-c", HOLDER, str(TESTS), str(lock), str(hold_s)],
                         stdout=subprocess.PIPE, text=True)
    t_a = float(a.stdout.readline().split()[0])  # A holds the lock from here
    b = subprocess.run([sys.executable, "-c", HOLDER, str(TESTS), str(lock), "0"],
                       capture_output=True, text=True, timeout=30)
    a.wait(timeout=30)
    t_b, waited_b = map(float, b.stdout.split())
    assert a.returncode == 0 and b.returncode == 0, b.stderr
    assert t_b - t_a >= hold_s - 0.05, (t_a, t_b)
    assert waited_b > 0.2


def test_lock_times_out_instead_of_hanging(tmp_path):
    lock = tmp_path / "ports.lock"
    holder = conftest.HostLock(lock, 5.0)
    holder.acquire()
    try:
        t0 = time.monotonic()
        try:
            conftest.HostLock(lock, 0.3).acquire()
        except TimeoutError:
            pass
        else:
            raise AssertionError("second acquire should time out while the first holds the lock")
        assert 0.25 < time.monotonic() - t0 < 3.0
    finally:
        holder.release()
