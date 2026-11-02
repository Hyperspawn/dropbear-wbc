"""dropbear_wbc.isaac.qmath: wxyz semantics over isaaclab.utils.math on Lab 2.x (identity) and 3.x (xyzw).

Gate: no raw ``isaaclab.utils.math`` import in the package outside qmath.py, and every name imported
from qmath has a quaternion-argument spec. The fake Lab 3 math below is written natively in xyzw,
independent of qmath, so a missing or reversed reorder fails.
"""

from __future__ import annotations

import pathlib
import re
import sys
import types

import pytest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from dropbear_wbc.isaac import qmath  # noqa: E402

PKG = ROOT / "source" / "dropbear_wbc"
QMATH = PKG / "isaac" / "qmath.py"
RAW_MATH = re.compile(r"^\s*(from\s+isaaclab\.utils\.math\s+import|import\s+isaaclab\.utils\.math)\b")


def _pkg_files():
    return [p for p in PKG.rglob("*.py") if p != QMATH and "__pycache__" not in p.parts]


def test_no_raw_isaaclab_math_import_in_package():
    hits = [
        f"{p.relative_to(ROOT)}:{i}: {line.strip()}"
        for p in _pkg_files()
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if RAW_MATH.search(line)
    ]
    assert not hits, "import from dropbear_wbc.isaac.qmath instead (wxyz on Lab 2 and 3):\n" + "\n".join(hits)


def test_every_imported_name_has_a_spec():
    used = set()
    for p in _pkg_files():
        src = p.read_text(encoding="utf-8")
        for block in re.findall(r"from dropbear_wbc\.isaac\.qmath import \(([^)]*)\)", src):
            used |= {n.strip() for n in block.split(",") if n.strip()}
        for line in re.findall(r"from dropbear_wbc\.isaac\.qmath import ([^(\n]+)", src):
            used |= {n.strip() for n in line.split(",") if n.strip()}
        if "qmath as math_utils" in src:
            used |= set(re.findall(r"math_utils\.(\w+)", src))
    assert used, "gate found no qmath users; regex broken?"
    missing = used - set(qmath.__all__)
    assert not missing, f"names without a quaternion spec in qmath._SPEC: {sorted(missing)}"


def test_gate_regex_catches_planted_imports():
    assert RAW_MATH.search("from isaaclab.utils.math import quat_mul")
    assert RAW_MATH.search("    import isaaclab.utils.math as math_utils")
    assert not RAW_MATH.search("from dropbear_wbc.isaac.qmath import quat_mul")
    assert not RAW_MATH.search("``isaaclab.utils.math`` functions of the same names.")


def test_unknown_name_raises():
    with pytest.raises(AttributeError):
        qmath.resolve("quat_slerp", math_module=types.SimpleNamespace(), order="xyzw")


# --------------------------------------------------------------------------- fake Lab 3 (xyzw) math

def _mul_xyzw(q1, q2):
    a, b = q1, q2
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    return torch.stack(
        [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
         w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)


def _conj_xyzw(q):
    return torch.cat([-q[..., :3], q[..., 3:]], -1)


def _apply_xyzw(q, v):
    vq = torch.cat([v, torch.zeros_like(v[..., :1])], -1)
    return _mul_xyzw(_mul_xyzw(q, vq), _conj_xyzw(q))[..., :3]


def _sft_xyzw(t01, q01, t02=None, q02=None):
    qi = _conj_xyzw(q01)
    q12 = _mul_xyzw(qi, q02) if q02 is not None else qi
    t12 = _apply_xyzw(qi, t02 - t01) if t02 is not None else _apply_xyzw(qi, -t01)
    return t12, q12


LAB3 = types.SimpleNamespace(
    quat_mul=_mul_xyzw, quat_conjugate=_conj_xyzw, quat_inv=_conj_xyzw, quat_apply=_apply_xyzw,
    quat_apply_inverse=lambda q, v: _apply_xyzw(_conj_xyzw(q), v), subtract_frame_transforms=_sft_xyzw,
    sample_uniform=lambda lo, hi, size, device: torch.full(size, 0.5),
)

S = 0.5 ** 0.5


@needs_torch
def test_lab3_wrap_known_rotation():
    z90 = torch.tensor([[S, 0.0, 0.0, S]], dtype=torch.float64)  # wxyz, +90 deg about z
    ex = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    ey = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    apply = qmath.resolve("quat_apply", LAB3, "xyzw")
    assert torch.allclose(apply(z90, ex), ey, atol=1e-12)
    assert torch.allclose(qmath.resolve("quat_apply_inverse", LAB3, "xyzw")(z90, ey), ex, atol=1e-12)
    z180 = qmath.resolve("quat_mul", LAB3, "xyzw")(z90, z90)
    assert torch.allclose(z180.abs(), torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(qmath.resolve("quat_conjugate", LAB3, "xyzw")(z90), torch.tensor([[S, 0, 0, -S]], dtype=torch.float64))


@needs_torch
def test_lab3_wrap_tuple_and_kwargs_and_none():
    z90 = torch.tensor([[S, 0.0, 0.0, S]], dtype=torch.float64)
    one = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    t0 = torch.zeros(1, 3, dtype=torch.float64)
    t2 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    sft = qmath.resolve("subtract_frame_transforms", LAB3, "xyzw")
    t12, q12 = sft(t0, z90, t2, one)
    assert torch.allclose(t12, torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(q12, torch.tensor([[S, 0, 0, -S]], dtype=torch.float64), atol=1e-12)
    t12k, q12k = sft(t0, q01=z90, t02=t2, q02=None)
    assert torch.allclose(q12k, q12, atol=1e-12) and torch.allclose(t12k, t12, atol=1e-12)
    m = qmath.resolve("quat_mul", LAB3, "xyzw")
    assert torch.allclose(m(q1=z90, q2=one), z90)


def test_lab2_and_passthrough_are_the_same_object():
    fake = types.SimpleNamespace(**{n: (lambda *a, **k: None) for n in qmath.__all__})
    for n in qmath.__all__:
        assert qmath.resolve(n, fake, "wxyz") is getattr(fake, n), n
    assert qmath.resolve("sample_uniform", fake, "xyzw") is fake.sample_uniform


def test_importing_qmath_does_not_import_isaaclab():
    assert "isaaclab.utils.math" not in sys.modules or sys.modules["isaaclab.utils.math"] is not None
    src = QMATH.read_text(encoding="utf-8")
    top = src.split("def _isaac_math", 1)[0]
    assert "import isaaclab" not in top
