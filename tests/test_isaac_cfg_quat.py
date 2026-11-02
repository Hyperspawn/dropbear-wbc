"""Isaac cfg quaternion literals (``rot=`` of InitialStateCfg / OffsetCfg) go through compat.cfg_quat.

Lab 2.x reads those fields as wxyz (default (1, 0, 0, 0)); Lab 3.0 reads them as xyzw (default (0, 0, 0, 1),
assets/asset_base_cfg.py:41, sensors/*/..._cfg.py OffsetCfg). A wxyz identity fed raw to 3.0 is a 180 deg
rotation about x: the robot spawns upside down and the tabletop cameras look the wrong way.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "source"))

from dropbear_wbc.isaac import compat  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCAN = [ROOT / "source" / "dropbear_wbc", ROOT / "tools", ROOT / "scripts"]
CFG_SUFFIXES = ("InitialStateCfg", "OffsetCfg")


def _func_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def raw_cfg_rot_sites(paths) -> list[str]:
    """``rot=`` keywords of *InitialStateCfg / *OffsetCfg calls whose value is not a cfg_quat(...) call."""
    bad = []
    for base in paths:
        files = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for f in files:
            try:
                tree = ast.parse(f.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and _func_name(node.func).endswith(CFG_SUFFIXES)):
                    continue
                for kw in node.keywords:
                    if kw.arg == "rot" and not (isinstance(kw.value, ast.Call) and _func_name(kw.value.func) == "cfg_quat"):
                        bad.append(f"{f.relative_to(ROOT) if f.is_relative_to(ROOT) else f}:{kw.value.lineno}")
    return bad


def test_no_raw_cfg_rot_literals():
    bad = raw_cfg_rot_sites(SCAN)
    assert not bad, f"wrap these rot= values in compat.cfg_quat(wxyz): {bad}"


def test_gate_catches_planted_raw_rot(tmp_path):
    f = tmp_path / "planted.py"
    f.write_text(
        "a = ArticulationCfg.InitialStateCfg(pos=(0, 0, 1), rot=(1.0, 0.0, 0.0, 0.0))\n"
        "b = CameraCfg.OffsetCfg(pos=p, rot=q, convention='world')\n"
        "c = AssetBaseCfg.InitialStateCfg(rot=cfg_quat((1.0, 0.0, 0.0, 0.0)))\n"
    )
    assert len(raw_cfg_rot_sites([f])) == 2


@pytest.mark.parametrize(
    "order, q, expect",
    [
        ("wxyz", (1, 0, 0, 0), (1.0, 0.0, 0.0, 0.0)),
        ("xyzw", (1, 0, 0, 0), (0.0, 0.0, 0.0, 1.0)),  # = Lab 3.0 InitialStateCfg default
        ("xyzw", (0.5, 0.1, 0.2, 0.3), (0.1, 0.2, 0.3, 0.5)),
        ("wxyz", np.array([0.5, 0.1, 0.2, 0.3]), (0.5, 0.1, 0.2, 0.3)),
    ],
)
def test_cfg_quat_order(monkeypatch, order, q, expect):
    monkeypatch.setenv("DROPBEAR_ISAAC_QUAT_ORDER", order)
    out = compat.cfg_quat(q)
    assert out == expect and all(type(v) is float for v in out)


def test_cfg_quat_rejects_wrong_length():
    with pytest.raises(ValueError):
        compat.cfg_quat((1.0, 0.0, 0.0))


def _rotate_xyzw(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))


def test_raw_wxyz_identity_is_180_about_x_on_lab3():
    """Why the wrapper matters: (1, 0, 0, 0) read as xyzw flips up (+z) to -z and forward (+x) stays +x."""
    up = _rotate_xyzw((1.0, 0.0, 0.0, 0.0), np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(up, [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(_rotate_xyzw((0.0, 0.0, 0.0, 1.0), np.array([0.0, 0.0, 1.0])), [0, 0, 1], atol=1e-12)
