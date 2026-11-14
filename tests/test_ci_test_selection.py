"""CI and the documented CPU test command must not deselect tests by name.

`-k "not isaac"` was a name filter, not an Isaac-availability check: it also
dropped the CPU-only Isaac-boundary gates (test_isaac_quat_boundary.py,
test_isaac_cfg_quat.py, test_isaac_qmath.py) and every other test whose id
contains "isaac", so those gates never ran in CI. Tests that need Isaac Sim
skip themselves; no keyword filter is needed.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
DOCS = (ROOT / "README.md", ROOT / "docs" / "INSTALL.md", ROOT / "docs" / "QUICKSTART.md")
GATES = ("test_isaac_quat_boundary.py", "test_isaac_cfg_quat.py", "test_isaac_qmath.py")
_PYTEST = re.compile(r"python\s+-m\s+pytest\b[^\n#]*")


def _pytest_commands(text: str) -> list[list[str]]:
    return [shlex.split(m.group(0)) for m in _PYTEST.finditer(text)]


def _selection_flags(argv: list[str]) -> list[str]:
    bad = []
    for i, a in enumerate(argv):
        if a in ("-k", "-m", "--deselect", "--ignore", "--ignore-glob"):
            bad.append(" ".join(argv[i:i + 2]))
        elif a.startswith(("-k", "-m", "--deselect=", "--ignore=", "--ignore-glob=")):
            bad.append(a)  # glued forms: -knot, -m"x", --deselect=...
    return bad


def test_workflow_runs_pytest_without_name_or_marker_selection():
    cmds = _pytest_commands(WORKFLOW.read_text(encoding="utf-8"))
    assert cmds, "tests.yml has no python -m pytest step"
    for argv in cmds:
        # argv[1] == "-m" is `python -m pytest`; check only the pytest args.
        assert _selection_flags(argv[3:]) == [], f"CI deselects tests: {' '.join(argv)}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_documented_cpu_command_matches_ci(doc):
    for argv in _pytest_commands(doc.read_text(encoding="utf-8")):
        assert _selection_flags(argv[3:]) == [], f"{doc.name} deselects tests: {' '.join(argv)}"


@pytest.mark.parametrize("gate", GATES)
def test_isaac_boundary_gates_live_under_testpaths(gate):
    assert (ROOT / "tests" / gate).is_file(), f"gate {gate} moved out of tests/"
