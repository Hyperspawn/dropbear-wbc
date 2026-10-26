"""A bare `pytest` from the repo root must collect tests/ only.

tools/hw_stand_test.py matches pytest's default *_test.py pattern but is an
Isaac Lab script; collecting it off-Isaac aborts the whole run.
"""

from __future__ import annotations

import configparser
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _ini() -> configparser.SectionProxy:
    cfg = configparser.ConfigParser()
    assert cfg.read(REPO / "pytest.ini"), "pytest.ini missing at repo root"
    return cfg["pytest"]


def test_testpaths_is_tests_only():
    assert _ini()["testpaths"].split() == ["tests"]


def test_test_named_files_outside_tests_are_not_recursed():
    skip = set(_ini()["norecursedirs"].split())
    offenders = []
    for pattern in ("test_*.py", "*_test.py"):
        for path in REPO.rglob(pattern):
            rel = path.relative_to(REPO)
            if rel.parts[0] == "tests":
                continue
            if not any(part in skip for part in rel.parts[:-1]):
                offenders.append(str(rel))
    assert not offenders, f"collectable test-named files outside tests/: {offenders}"


def test_bare_pytest_collects_without_errors():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-2000:]
    lines = out.splitlines()
    assert not [ln for ln in lines if ln.startswith("tools/") or ln.startswith("ERROR")]
    assert "error" not in lines[-1].lower()
