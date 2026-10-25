"""Loopback test ports must sit below the OS ephemeral range and must not be shared between test files.

A fixed port inside the ephemeral range (Linux 32768-60999, Windows/macOS 49152-65535) can be taken as the
local port of any outgoing TCP connect made earlier in the same run (ZMQ client sockets, subprocesses),
and the later bind fails with EADDRINUSE. That is how CI run 40163897030 failed on 127.0.0.1:46636.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

TESTS = Path(__file__).resolve().parent
EPHEMERAL_LOW = 32768  # lowest default ephemeral port across Linux, Windows and macOS
CONTRACT_DEFAULTS = {5555, 5556}
# Literal ports handed to Endpoints / --cmd-port, plus offset bases like `cmd_port=26555 + offset`.
PORT_RE = re.compile(r"\b(?:cmd_port|state_port)\s*=\s*(\d{4,5})")
CASE_RE = re.compile(r"\"SIG[A-Z]+\",\s*(\d{4,5})\)")
MAX_OFFSET = 100  # _endpoints(offset) helpers use offsets < 100


def _ports() -> dict[int, set[str]]:
    found: dict[int, set[str]] = defaultdict(set)
    for path in sorted(TESTS.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        for m in PORT_RE.finditer(text):
            found[int(m.group(1))].add(path.name)
        for m in CASE_RE.finditer(text):  # (script, signal, cmd_port); state_port = cmd_port + 1
            found[int(m.group(1))].add(path.name)
            found[int(m.group(1)) + 1].add(path.name)
    return found


def test_port_scan_finds_the_loopback_tests():
    files = {f for names in _ports().values() for f in names}
    assert {"test_sdk_transport.py", "test_runner_loopback.py", "test_lowcmd_stop_signals.py"} <= files


def test_loopback_ports_below_ephemeral_range():
    bad = {p: sorted(f) for p, f in _ports().items() if not (1024 <= p and p + MAX_OFFSET < EPHEMERAL_LOW)}
    assert not bad, f"test ports inside the ephemeral range or privileged: {bad}"


def test_loopback_ports_avoid_contract_defaults_and_are_not_shared():
    ports = _ports()
    assert not CONTRACT_DEFAULTS & set(ports), "tests must not bind the CONTRACTS sec 6 default ports"
    shared = {p: sorted(f) for p, f in ports.items() if len(f) > 1}
    assert not shared, f"same port used by more than one test file: {shared}"
