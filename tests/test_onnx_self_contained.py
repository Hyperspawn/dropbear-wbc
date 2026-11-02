"""Policy exports must be single-file ONNX.

torch >= 2.9 (Isaac Lab 3.0 ships torch 2.11) exports through the dynamo exporter by default and writes the
weights to ``policy.onnx.data``. Deploy bundles carry ``policy.onnx`` + ``policy.json`` only; onnxruntime
rejects the graph without its data file. Exports are inlined by ``make_onnx_self_contained`` or by
``_attach_onnx_metadata`` (load + save), and the side file is removed.
"""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401,E402  (adds source/ and third_party/pydeps)

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from dropbear_wbc.deploy.policy import OnnxPolicy, make_onnx_self_contained, onnx_external_data_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OBS, ACT = 12, 5


def _split_model(d: Path, name: str = "policy.onnx") -> tuple[Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((OBS, ACT)).astype(np.float32)
    b = rng.standard_normal(ACT).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "W"], ["h"]), helper.make_node("Add", ["h", "B"], ["actions"])],
        "pi",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, OBS])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, ACT])],
        [numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 10
    p = d / name
    onnx.save_model(m, str(p), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=name + ".data", size_threshold=0)
    return p, w, b


def test_split_export_does_not_load_alone(tmp_path):
    p, _, _ = _split_model(tmp_path)
    assert onnx_external_data_files(p) == ["policy.onnx.data"]
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(p, alone)
    with pytest.raises(Exception):
        OnnxPolicy(alone / "policy.onnx")


def test_make_self_contained_inlines_and_removes_side_file(tmp_path):
    p, w, b = _split_model(tmp_path)
    assert make_onnx_self_contained(p) is True
    assert onnx_external_data_files(p) == []
    assert not (tmp_path / "policy.onnx.data").exists()
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(p, alone)
    pol = OnnxPolicy(alone / "policy.onnx")
    x = np.linspace(-1, 1, OBS).astype(np.float32)
    np.testing.assert_allclose(pol.act(x), x @ w + b, rtol=1e-6, atol=1e-6)
    assert make_onnx_self_contained(p) is False  # idempotent on an inline model


def test_attach_metadata_inlines_and_removes_side_file(tmp_path):
    pytest.importorskip("torch")
    try:
        from dropbear_wbc.tasks.tracking.export import _attach_onnx_metadata
    except ImportError:  # isaaclab not importable in this venv: exercise the same code from source
        src = (ROOT / "source/dropbear_wbc/tasks/tracking/export.py").read_text()
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_attach_onnx_metadata")
        ns: dict = {"Path": Path, "Any": object}
        exec(compile(ast.Module([fn], []), "export.py", "exec"), ns)
        _attach_onnx_metadata = ns["_attach_onnx_metadata"]
    p, _, _ = _split_model(tmp_path)
    _attach_onnx_metadata(p, {"joint_names": ["a", "b"], "dt": 0.02})
    assert onnx_external_data_files(p) == []
    assert not (tmp_path / "policy.onnx.data").exists()
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(p, alone)
    assert OnnxPolicy(alone / "policy.onnx").metadata == {"joint_names": "a,b", "dt": "0.02"}


def _calls(tree: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def test_every_package_onnx_export_is_made_single_file():
    """Each package module that writes ONNX must also inline it (make_onnx_self_contained or _attach_onnx_metadata)."""
    offenders = []
    for f in sorted((ROOT / "source/dropbear_wbc").rglob("*.py")):
        text = f.read_text()
        calls = _calls(ast.parse(text))
        writes = "export_policy_as_onnx" in calls or "torch.onnx.export(" in text
        if writes and not calls & {"make_onnx_self_contained", "_attach_onnx_metadata"}:
            offenders.append(str(f.relative_to(ROOT)))
    assert offenders == [], f"ONNX written without inlining external data: {offenders}"
