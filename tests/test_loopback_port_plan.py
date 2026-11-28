"""Loopback test ports must sit below the OS ephemeral range and must not be shared between test files.

A fixed port inside the ephemeral range (Linux 32768-60999, Windows/macOS 49152-65535) can be taken as the
local port of any outgoing TCP connect made earlier in the same run (ZMQ client sockets, subprocesses),
and the later bind fails with EADDRINUSE. That is how CI run 40163897030 failed on 127.0.0.1:46636.

The plan only covers one pytest run; a second process on the host holding a literal port still breaks it. Fake
robots therefore bind port 0 and hand the bound ports to the client (tests/loopback_ports.py, item 398); the
BIND_FIRST modules must stay free of literal ports, and any literal added elsewhere must still obey the plan.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

TESTS = Path(__file__).resolve().parent
EPHEMERAL_LOW = 32768  # lowest default ephemeral port across Linux, Windows and macOS
CONTRACT_DEFAULTS = {5555, 5556}
# Literal ports handed to Endpoints / --cmd-port, plus offset bases like `cmd_port=<base> + offset`.
PORT_RE = re.compile(r"\b(?:cmd_port|state_port)\s*=\s*(\d{4,5})")
CASE_RE = re.compile(r"\"SIG[A-Z]+\",\s*(\d{4,5})\)")
MAX_OFFSET = 100  # _endpoints(offset) helpers use offsets < 100
# Converted to bind-first OS-assigned ports (item 398, notes/397/fixed_port_inventory.json + the two below).
BIND_FIRST = {"test_runner_loopback.py", "test_runner_stamp_midrun.py", "test_runner_stamp_warning.py",
              "test_state_freshness_gate.py", "test_fsm_motor_mode_gate.py", "test_sdk_transport.py",
              "test_stop_cause.py", "test_tau_valid_recording.py", "test_teleop_loopback.py",
              "test_teleop_state_age.py", "test_wall_schedule_rt.py", "test_lowcmd_stop_signals.py",
              "test_teleop_motion_mode_gains.py"}


def _scan(text: str) -> list[int]:
    ports = [int(m.group(1)) for m in PORT_RE.finditer(text)]
    for m in CASE_RE.finditer(text):  # (script, signal, cmd_port); state_port = cmd_port + 1
        ports += [int(m.group(1)), int(m.group(1)) + 1]
    return ports


def _ports() -> dict[int, set[str]]:
    found: dict[int, set[str]] = defaultdict(set)
    for path in sorted(TESTS.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        for port in _scan(path.read_text(encoding="utf-8")):
            found[port].add(path.name)
    return found


def test_port_scan_finds_literal_ports():
    old = 27555  # samples built at run time: digits after cmd_port= in this file would make conftest lock it
    assert _scan(f"ep = Endpoints(cmd_port={old}, state_port={old + 1})") == [old, old + 1]
    assert _scan(f"Endpoints(cmd_port={old} + offset, state_port={old + 1} + offset)") == [old, old + 1]
    assert _scan(f'CASES = [("tools/policy_runner.py", "SIGTERM", {old})]') == [old, old + 1]
    assert _scan("ep = robot.start_bound()\nEndpoints(cmd_port=port, state_port=0)") == []


def test_bind_first_modules_have_no_literal_ports():
    assert BIND_FIRST <= {p.name for p in TESTS.glob("test_*.py")}
    left = {p: sorted(f & BIND_FIRST) for p, f in _ports().items() if f & BIND_FIRST}
    assert not left, f"bind port 0 via tests/loopback_ports.py instead of fixed ports: {left}"


def test_loopback_ports_below_ephemeral_range():
    bad = {p: sorted(f) for p, f in _ports().items() if not (1024 <= p and p + MAX_OFFSET < EPHEMERAL_LOW)}
    assert not bad, f"test ports inside the ephemeral range or privileged: {bad}"


def test_loopback_ports_avoid_contract_defaults_and_are_not_shared():
    ports = _ports()
    assert not CONTRACT_DEFAULTS & set(ports), "tests must not bind the CONTRACTS sec 6 default ports"
    shared = {p: sorted(f) for p, f in ports.items() if len(f) > 1}
    assert not shared, f"same port used by more than one test file: {shared}"
