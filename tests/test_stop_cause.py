"""Item 270: a LowCmd writer stopped by a signal must say so in its summary and exit code.

Before 0033, policy_runner reported status "executed" with rc 0 after SIGINT/SIGTERM/SIGHUP, so a run killed by
systemd, an SSH drop or Popen.terminate looked like a completed --duration run in summaries and run matrices
(tools/run_throughput_matrix.py reruns only when status != "executed"). teleop_arm reported "ok" / rc 0 and
labelled every HOME abort "SIGINT during HOME".
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk.signals import StopRequest
from dropbear_wbc.sdk.transport import Endpoints

ROOT = Path(__file__).resolve().parents[1]


def test_stop_request_records_first_cause_and_exit_code():
    s = StopRequest()
    assert not s.flag and s.exit_code() == 0 and s.to_dict()["cause"] is None
    s.handler(signal.SIGTERM, None)
    s.handler(signal.SIGINT, None)
    s.request("duration")
    assert s.flag and s.cause == "SIGTERM" and s.by_signal and s.count == 2
    assert s.signals == ["SIGTERM", "SIGINT"] and s.exit_code() == 128 + signal.SIGTERM
    d = StopRequest()
    d.request("duration")
    assert d.cause == "duration" and not d.by_signal and d.exit_code() == 0


def _run(rel: str, extra: list[str], ep: Endpoints, summary: Path, signame: str | None):
    from test_runner_loopback import FakeRobot

    robot = FakeRobot(ep)
    robot.start()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "source"), str(ROOT / "tests"),
                                                      os.environ.get("PYTHONPATH", "")]))
    cmd = [sys.executable, "-u", str(ROOT / rel), *extra, "--summary", str(summary),
           "--cmd-port", str(ep.cmd_port), "--state-port", str(ep.state_port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT, env=env, text=True)
    try:
        if signame is not None:
            t0 = time.time()
            while robot.cmds < 300 and proc.poll() is None and time.time() - t0 < 60:
                time.sleep(0.05)
            assert proc.poll() is None, proc.communicate()[0][-2000:]
            proc.send_signal(getattr(signal, signame))
        out, _ = proc.communicate(timeout=90)
    finally:
        if proc.poll() is None:
            proc.kill()
        robot.running = False
        robot.join(timeout=5)
    return proc.returncode, json.loads(summary.read_text(encoding="utf-8")), out


RUNNER = ["--mode", "hold", "--passive-s", "0.05", "--move-s", "0.3", "--connect-timeout-s", "20"]
CASES = [("tools/policy_runner.py", "SIGTERM", 27821), ("tools/policy_runner.py", "SIGINT", 27823),
         ("tools/teleop_arm.py", "SIGTERM", 27825)]


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
@pytest.mark.parametrize("rel,signame,port", CASES)
def test_signal_stop_is_reported_as_interrupted(rel, signame, port, tmp_path):
    extra = RUNNER + ["--duration", "60"] if rel.endswith("policy_runner.py") else ["--device", "scripted",
                                                                                   "--duration", "60"]
    rc, summ, out = _run(rel, extra, Endpoints(cmd_port=port, state_port=port + 1), tmp_path / "s.json", signame)
    assert summ["status"] == "interrupted", out[-2000:]
    assert summ["stop"]["cause"] == signame and summ["stop"]["by_signal"] is True
    assert rc == 128 + getattr(signal, signame)


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
def test_duration_stop_is_executed(tmp_path):
    rc, summ, out = _run("tools/policy_runner.py", RUNNER + ["--duration", "1.5"],
                         Endpoints(cmd_port=27827, state_port=27828), tmp_path / "s.json", None)
    assert summ["status"] == "executed" and rc == 0, out[-2000:]
    assert summ["stop"] == {"cause": "duration", "by_signal": False, "signals": [], "count": 0}
