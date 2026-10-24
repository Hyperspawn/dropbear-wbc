"""Item 268: every rt/lowcmd writer in tools/ and scripts/ must leave the robot in damping when it is stopped.

Before 0030, SIGTERM (kill, systemd/docker stop, Popen.terminate) and SIGHUP (SSH session dropped) killed
policy_runner, teleop_arm and the low-level example with the last hold/policy LowCmd (kp 200-600) still on the
robot; only Ctrl-C reached the damping exit.
"""
from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk.signals import STOP_SIGNALS
from dropbear_wbc.sdk.transport import Endpoints

ROOT = Path(__file__).resolve().parents[1]
WRITERS = {"tools/policy_runner.py", "tools/teleop_arm.py", "scripts/examples/dropbear_low_level_example.py"}


def _is_lowcmd_publisher(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) in
            ("ChannelPublisher", "ZmqPublisher")):
        return False
    args = list(node.args) + [k.value for k in node.keywords]
    return any((isinstance(a, ast.Name) and a.id in ("LOWCMD", "LowCmd")) or
               (isinstance(a, ast.Constant) and a.value == "rt/lowcmd") for a in args)


def _damps(stmts: list[ast.stmt]) -> bool:
    for n in ast.walk(ast.Module(body=stmts, type_ignores=[])):
        if isinstance(n, ast.Call):
            name = getattr(n.func, "id", getattr(n.func, "attr", "")) or ""
            if "damp" in name.lower():
                return True
            if any(k.arg == "kp" and isinstance(k.value, ast.Constant) and k.value.value == 0 for k in n.keywords):
                return True
    return False


def _lowcmd_writers() -> dict[str, ast.Module]:
    out = {}
    for base in ("tools", "scripts"):
        for p in sorted((ROOT / base).rglob("*.py")):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            if any(_is_lowcmd_publisher(n) for n in ast.walk(tree)):
                out[p.relative_to(ROOT).as_posix()] = tree
    return out


def test_lowcmd_writer_scan_finds_the_known_writers():
    assert WRITERS <= set(_lowcmd_writers()), "scan lost a known rt/lowcmd writer"


@pytest.mark.parametrize("rel", sorted(_lowcmd_writers()))
def test_lowcmd_writer_has_damping_abort_path_and_stop_signals(rel):
    tree = _lowcmd_writers()[rel]
    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    assert any(_damps(t.finalbody) or any(_damps(h.body) for h in t.handlers) for t in tries), (
        f"{rel}: no damping write (kp=0 or *damp*()) in an except/finally block")
    calls = {getattr(n.func, "id", getattr(n.func, "attr", "")) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "install_stop_handler" in calls, (
        f"{rel}: register SIGTERM/SIGHUP via dropbear_wbc.sdk.signals.install_stop_handler (only SIGINT is "
        "catchable by default; SIGTERM leaves the last LowCmd on the robot)")


def test_stop_signals_cover_term_and_hup():
    assert signal.SIGINT in STOP_SIGNALS and signal.SIGTERM in STOP_SIGNALS
    if hasattr(signal, "SIGHUP"):
        assert signal.SIGHUP in STOP_SIGNALS


def _cmd(rel: str, ep: Endpoints) -> list[str]:
    ports = ["--cmd-port", str(ep.cmd_port), "--state-port", str(ep.state_port)]
    extra = {"tools/policy_runner.py": ["--mode", "hold", "--duration", "60", "--passive-s", "0.05", "--move-s", "0.3",
                                        "--connect-timeout-s", "20"],
             "tools/teleop_arm.py": ["--device", "scripted", "--duration", "60"],
             "scripts/examples/dropbear_low_level_example.py": ["--duration", "60"]}[rel]
    return [sys.executable, "-u", str(ROOT / rel), *extra, *ports]


CASES = [("tools/policy_runner.py", "SIGTERM", 47811), ("tools/policy_runner.py", "SIGHUP", 47813),
         ("tools/teleop_arm.py", "SIGTERM", 47815),
         ("scripts/examples/dropbear_low_level_example.py", "SIGTERM", 47817)]


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
@pytest.mark.parametrize("rel,signame,port", CASES)
def test_stop_signal_leaves_damping_as_last_lowcmd(rel, signame, port):
    from test_runner_loopback import FakeRobot

    ep = Endpoints(cmd_port=port, state_port=port + 1)
    robot = FakeRobot(ep)
    robot.start()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "source"), str(ROOT / "tests"),
                                                      os.environ.get("PYTHONPATH", "")]))
    proc = subprocess.Popen(_cmd(rel, ep), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
                            text=True)
    try:
        t0 = time.time()
        while robot.cmds < 300 and proc.poll() is None and time.time() - t0 < 60:
            time.sleep(0.05)
        assert proc.poll() is None, proc.communicate()[0][-2000:]
        assert robot.last_kp is not None and float(np.abs(robot.last_kp).max()) > 0  # gains were live
        proc.send_signal(getattr(signal, signame))
        out, _ = proc.communicate(timeout=20)
        time.sleep(0.2)
    finally:
        if proc.poll() is None:
            proc.kill()
        robot.running = False
        robot.join(timeout=5)
    assert proc.returncode >= 0, f"{rel} died on {signame} (rc {proc.returncode}) without its exit path\n{out[-2000:]}"
    assert float(np.abs(robot.last_kp).max()) == 0.0, f"{rel}: last LowCmd after {signame} still has kp>0"
