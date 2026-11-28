"""The conftest host lock serialises fixed-port loopback tests across concurrent pytest runs."""
from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import conftest

TESTS = Path(__file__).resolve().parent
# Formerly fixed-port modules, now bind-first on OS-assigned ports (item 398): they no longer need the lock.
BIND_FIRST = {"test_sdk_transport.py", "test_runner_loopback.py", "test_lowcmd_stop_signals.py",
              "test_teleop_loopback.py", "test_stop_cause.py", "test_runner_stamp_warning.py"}

# Holder A: takes the lock, reports, keeps it until the parent writes a line to its stdin, reports
# the wall-clock instant it released, then stays alive until a second line (so B must get the lock
# from release(), not from A's process exit).
HOLDER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    import conftest
    lock = conftest.HostLock(Path(sys.argv[2]), 30.0)
    lock.acquire()
    print("held", flush=True)
    sys.stdin.readline()
    t_release = time.time()
    lock.release()
    print(f"{t_release:.6f}", flush=True)
    sys.stdin.readline()
""")

# Waiter B: first proves the lock is held right now (a zero-timeout probe must time out), reports
# that, then blocks in acquire and reports the wall-clock instant it got the lock.
WAITER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    import conftest
    try:
        conftest.HostLock(Path(sys.argv[2]), 0.0).acquire()
    except TimeoutError:
        print("contended", flush=True)
    else:
        print("free", flush=True)
        sys.exit(0)
    lock = conftest.HostLock(Path(sys.argv[2]), 30.0)
    waited = lock.acquire()
    t_acquired = time.time()
    lock.release()
    print(f"{t_acquired:.6f} {waited:.6f}", flush=True)
""")


def test_fixed_port_modules_are_locked_and_free_port_modules_are_not(tmp_path):
    # Samples are built from ``old`` (no digits after cmd_port= in this file), or this module would take the lock.
    old = 27555
    samples = {"endpoints.py": f"ep = Endpoints(cmd_port={old}, state_port={old + 1})\n",
               "cases.py": f'CASES = [("tools/policy_runner.py", "SIGTERM", {old})]\nEndpoints(cmd_port=p)\n',
               "bind_first.py": "ep = robot.start_bound()\nEndpoints(cmd_port=port, state_port=0)\n"}
    for name, text in samples.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    assert {n for n in samples if conftest.module_uses_fixed_ports(tmp_path / n)} == {"endpoints.py", "cases.py"}
    locked = {p.name for p in TESTS.glob("test_*.py") if conftest.module_uses_fixed_ports(p)}
    assert not BIND_FIRST & locked
    assert "test_publisher_close_flush.py" not in locked  # binds tcp://127.0.0.1:*
    assert "test_teleop_devices.py" not in locked  # _free_port()
    assert "test_loopback_port_plan.py" not in locked  # only scans source


def test_second_process_waits_for_the_holder(tmp_path):
    # Ordered by handshakes, not by sleeps: B is known to be blocked on the lock before A is told to
    # release, so the result does not depend on interpreter start-up time or CPU load (item 403: the
    # old waited_b > 0.2 check measured from B's start and gave 0.151 s under a loaded full suite).
    lock = tmp_path / "ports.lock"
    a = subprocess.Popen([sys.executable, "-c", HOLDER, str(TESTS), str(lock)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    b = None
    procs = [a]
    watchdog = threading.Timer(20.0, lambda: [p.kill() for p in procs])  # a hang becomes a failure
    watchdog.start()
    try:
        assert a.stdout.readline().strip() == "held", a.stderr.read()
        b = subprocess.Popen([sys.executable, "-c", WAITER, str(TESTS), str(lock)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append(b)
        probe = b.stdout.readline().strip()
        assert probe == "contended", f"B saw the lock {probe or 'nothing'} while A held it: {b.stderr.read()}"
        time.sleep(0.2)  # B is now polling inside acquire()
        a.stdin.write("release\n")
        a.stdin.flush()
        t_release = float(a.stdout.readline())
        t_acquired, waited_b = map(float, b.stdout.readline().split())
        a.stdin.write("exit\n")
        a.stdin.flush()
        assert a.wait(timeout=30) == 0 and b.wait(timeout=30) == 0, b.stderr.read()
    finally:
        watchdog.cancel()
        for p in procs:
            if p.poll() is None:
                p.kill()
    # B only got the lock after A let go
    assert t_acquired >= t_release, dict(t_release=t_release, t_acquired=t_acquired, waited_b=waited_b)


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
