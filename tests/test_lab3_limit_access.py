"""Isaac Lab 3.0 renamed the actuator limit fields (``actuator_effort_limit`` / ``actuator_velocity_limit``; the old
names are warning aliases removed in 3.1). Every Isaac-side read or write goes through ``hw_introspect`` (runtime
tensors) or ``hw_actuators.LIMIT_FIELDS`` (configs). AST scan, CPU only."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
ALLOWED = {
    "source/dropbear_wbc/robots/hw_introspect.py",  # the 2.2 fallback itself
}
SKIP_DIRS = ("source/dropbear_wbc/sdk/",)  # SDK dataclasses (no Isaac): their own effort_limit field
RENAMED = {"effort_limit", "velocity_limit"}


def _files() -> list[str]:
    out = []
    for root in ("tools", "scripts", "source/dropbear_wbc"):
        for p in sorted((_REPO / root).rglob("*.py")):
            rel = p.relative_to(_REPO).as_posix()
            if rel not in ALLOWED and not rel.startswith(SKIP_DIRS):
                out.append(rel)
    return out


@pytest.mark.parametrize("rel", ("tools/hw_stand_test.py", "tools/probe_knee_authority.py"))
def test_the_two_tools_are_scanned(rel):
    assert rel in _files()


def test_no_direct_renamed_limit_attribute():
    bad = []
    for rel in _files():
        tree = ast.parse((_REPO / rel).read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and n.attr in RENAMED:
                bad.append(f"{rel}:{n.lineno} .{n.attr}")
    assert not bad, "use hw_introspect.actuator_effort_limit / LIMIT_FIELDS:\n" + "\n".join(bad)
