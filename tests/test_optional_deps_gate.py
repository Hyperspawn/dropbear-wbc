"""Tests that need an optional heavy dependency must skip, not fail, when it is absent.

CI installs torch (and mujoco), so a bare `import torch` in a test passes there and only
fails on a torch-free machine (the c3e64f7 symmetry test did exactly that). Two gates:

1. Static: every import of an optional module inside tests/ is preceded, in the same or an
   enclosing scope, by `pytest.importorskip("<module>")`, or sits in a try/except that
   catches ImportError.
2. Dynamic: collecting tests/ in a subprocess with those modules blocked at import time
   yields zero collection errors (catches module-level and transitive imports through the
   project's own packages).
"""
from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
OPTIONAL = ("torch", "rsl_rl", "isaaclab", "isaacsim", "omni", "pxr", "newton", "warp", "mujoco")
_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _roots(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name.split(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module.split(".")[0]]
    return []


def _skip_target(node: ast.AST) -> str | None:
    """Module name if node is a call pytest.importorskip("x...")."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "importorskip" and node.args
            and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
        return node.args[0].value.split(".")[0]
    return None


def _skip_targets(stmt: ast.AST, helpers: dict[str, set[str]]) -> set[str]:
    """Modules a statement guards: importorskip("x"), importorskip(m) in `for m in ("x", ...)`,
    or a call to a local helper that itself guards them."""
    out: set[str] = set()
    for sub in ast.walk(stmt):
        t = _skip_target(sub)
        if t:
            out.add(t)
        if isinstance(sub, ast.For) and isinstance(sub.iter, (ast.Tuple, ast.List)) and isinstance(sub.target, ast.Name):
            consts = [e.value.split(".")[0] for e in sub.iter.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            for c in ast.walk(sub):
                if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "importorskip"
                        and c.args and isinstance(c.args[0], ast.Name) and c.args[0].id == sub.target.id):
                    out.update(consts)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in helpers:
            out |= helpers[sub.func.id]
    return out


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:
        return True
    names = t.elts if isinstance(t, ast.Tuple) else [t]
    return any(isinstance(n, ast.Name) and n.id in _IMPORT_ERRORS for n in names)


def unguarded_optional_imports(source: str, filename: str = "<src>") -> list[tuple[int, str]]:
    """(line, module) for each optional import that is not guarded."""
    tree = ast.parse(source, filename)
    bad: list[tuple[int, str]] = []

    def walk(body: list[ast.stmt], guarded: frozenset[str]) -> None:
        g = set(guarded)
        for stmt in body:
            for root in _roots(stmt):
                if root in OPTIONAL and root not in g:
                    bad.append((stmt.lineno, root))
            if isinstance(stmt, ast.Try):
                inner = frozenset(g | set(OPTIONAL)) if any(map(_catches_import_error, stmt.handlers)) else frozenset(g)
                walk(stmt.body, inner)
                for h in stmt.handlers:
                    walk(h.body, frozenset(g))
                walk(stmt.orelse, frozenset(g))
                walk(stmt.finalbody, frozenset(g))
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(stmt.body, frozenset(g))
            elif isinstance(stmt, (ast.If, ast.For, ast.While, ast.With, ast.AsyncWith, ast.AsyncFor)):
                walk(stmt.body, frozenset(g))
                walk(getattr(stmt, "orelse", []), frozenset(g))
            # a skip call in this statement guards the statements after it in this scope
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                g |= _skip_targets(stmt, helpers)

    # Module-level helpers (e.g. _need_usd) that skip when a module is missing.
    helpers: dict[str, set[str]] = {}
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef):
            mods = _skip_targets(fn, {})
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Try) and any(map(_catches_import_error, sub.handlers)):
                    mods |= {r for st in sub.body for r in _roots(st) if r in OPTIONAL}
            if mods:
                helpers[fn.name] = mods
    walk(tree.body, frozenset())
    return bad


def test_static_gate_flags_bare_and_accepts_guarded():
    bare = "def test_x():\n    import torch\n    from rsl_rl.runners import OnPolicyRunner\n"
    assert unguarded_optional_imports(bare) == [(2, "torch"), (3, "rsl_rl")]
    ok = textwrap.dedent("""
        import pytest
        mujoco = pytest.importorskip("mujoco")
        def test_a():
            pytest.importorskip("torch")
            import torch
        def test_b():
            try:
                import newton
            except ImportError:
                pytest.skip("no newton")
        def _need_pxr():
            try:
                from pxr import Usd
            except ImportError:
                pytest.skip("no pxr")
        def test_d():
            _need_pxr()
            from pxr import Usd
        def test_e():
            for m in ("torch", "rsl_rl"):
                pytest.importorskip(m)
            import torch
        def test_c():
            import torch  # not guarded: test_a's skip is a different scope
    """)
    assert unguarded_optional_imports(ok) == [(25, "torch")]


def test_no_unguarded_optional_imports_in_tests():
    bad = []
    for f in sorted(TESTS.rglob("*.py")):
        for line, mod in unguarded_optional_imports(f.read_text(encoding="utf-8"), str(f)):
            bad.append(f"{f.relative_to(ROOT)}:{line} imports {mod} without pytest.importorskip({mod!r})")
    assert not bad, "\n".join(bad)


_BLOCKED_COLLECT = textwrap.dedent("""
    import importlib.abc, sys
    BLOCK = set(sys.argv[1].split(","))
    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCK:
                raise ModuleNotFoundError("No module named %r (blocked)" % name, name=name)
            return None
    for m in list(sys.modules):
        if m.split(".")[0] in BLOCK:
            del sys.modules[m]
    sys.meta_path.insert(0, _Block())
    import pytest
    sys.exit(pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", "tests"]))
""")


def test_collection_survives_missing_optional_deps():
    r = subprocess.run([sys.executable, "-c", _BLOCKED_COLLECT, ",".join(OPTIONAL)],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    tail = "\n".join((r.stdout + r.stderr).splitlines()[-40:])
    assert r.returncode == 0, f"collection with {OPTIONAL} blocked failed (rc {r.returncode}):\n{tail}"
    assert " error" not in r.stdout.splitlines()[-1], tail
