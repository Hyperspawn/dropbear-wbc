"""Export call sites must work on rsl-rl 2.x ActorCritic and rsl-rl 5.x MLPModel.

Isaac Lab 3.0 ``export_policy_as_onnx`` raises ValueError on an rsl-rl 5.x ``MLPModel`` (no actor/student),
and ``policy.actor[0].in_features`` raises AttributeError. ``dropbear_wbc.tasks.policy_export`` routes both:
2.x delegates unchanged to the Isaac Lab exporters, 5.x uses ``as_jit()`` / ``as_onnx()``.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401,E402

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from dropbear_wbc.tasks import policy_export as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OBS, ACT = 14, 5


class _Norm(nn.Module):  # EmpiricalNormalization shape: _mean/_std/eps buffers
    def __init__(self, n):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.register_buffer("_mean", torch.randn(1, n, generator=g))
        self.register_buffer("_std", torch.rand(1, n, generator=g) + 0.5)
        self.eps = 1e-2

    def forward(self, x):
        return (x - self._mean) / (self._std + self.eps)


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(OBS, 32), nn.ELU(), nn.Linear(32, ACT))


class _ActorCritic2(nn.Module):  # rsl-rl 2.x surface used by the exporters
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.actor = _mlp()

    def act_inference(self, x):
        return self.actor(x)


class _Det(nn.Module):
    def forward(self, x):
        return x


class _Dist:
    def as_deterministic_output_module(self):
        return _Det()


class _Wrap(nn.Module):  # rsl-rl 5.5.1 _TorchMLPModel / _OnnxMLPModel
    def __init__(self, m):
        super().__init__()
        import copy

        self.obs_normalizer = copy.deepcopy(m.obs_normalizer)
        self.mlp = copy.deepcopy(m.mlp)
        self.deterministic_output = _Det()

    def forward(self, x):
        return self.deterministic_output(self.mlp(self.obs_normalizer(x)))


class _MLPModel5(nn.Module):  # rsl-rl 5.x surface: no actor, own normalizer, as_jit/as_onnx
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.obs_dim = OBS
        self.obs_normalizer = _Norm(OBS)
        self.mlp = _mlp()
        self.distribution = _Dist()

    def as_jit(self):
        return _Wrap(self)

    def as_onnx(self, verbose=False):
        return _Wrap(self)


def _real_mlp_model():
    models = pytest.importorskip("rsl_rl.models")
    if not hasattr(models, "MLPModel"):
        pytest.skip("rsl-rl < 5")
    from tensordict import TensorDict

    obs = TensorDict({"policy": torch.zeros(2, OBS), "critic": torch.zeros(2, OBS + 3)}, batch_size=[2])
    m = models.MLPModel(obs, {"actor": ["policy"], "critic": ["critic"]}, "actor", ACT, [32, 16], "elu", True,
                        {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"})
    m.obs_normalizer.update(torch.randn(256, OBS) * 3 + 1)
    return m.eval()


@pytest.fixture
def isaac_stub(monkeypatch):
    calls = []
    m = types.ModuleType("isaaclab_rl.rsl_rl")
    m.export_policy_as_jit = lambda *a, **k: calls.append(("jit", a, k))
    m.export_policy_as_onnx = lambda *a, **k: calls.append(("onnx", a, k))
    monkeypatch.setitem(sys.modules, "isaaclab_rl", types.ModuleType("isaaclab_rl"))
    monkeypatch.setitem(sys.modules, "isaaclab_rl.rsl_rl", m)
    return calls


def test_rslrl2_delegates_unchanged(isaac_stub, tmp_path):
    p, n = _ActorCritic2(), _Norm(OBS)
    assert not pe.is_rslrl5_model(p)
    assert pe.actor_obs_dim(p) == OBS
    pe.export_actor_jit(p, n, tmp_path, "policy.pt")
    with pytest.raises(FileNotFoundError):  # stub writes nothing; helper still tries to inline the path
        pe.export_actor_onnx(p, n, tmp_path, "policy.onnx")
    assert isaac_stub[0] == ("jit", (p, n), {"path": str(tmp_path), "filename": "policy.pt"})
    assert isaac_stub[1] == ("onnx", (p,), {"normalizer": n, "path": str(tmp_path), "filename": "policy.onnx"})
    a, nn_ = pe.actor_and_normalizer(p, n)
    x = torch.randn(8, OBS)
    assert torch.equal(a(nn_(x)), p.actor(n(x)))
    assert torch.equal(pe.live_actions(p, n, x), p.actor(n(x)))
    assert pe.effective_normalizer(p, n) is n


@pytest.mark.parametrize("make", [_MLPModel5, _real_mlp_model], ids=["stub5", "rslrl5"])
def test_rslrl5_exports_contract(make, tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setitem(sys.modules, "isaaclab_rl.rsl_rl", None)  # 5.x path must not touch the Isaac exporter
    from dropbear_wbc.deploy.policy import OnnxPolicy, onnx_external_data_files

    p = make()
    assert pe.is_rslrl5_model(p) and pe.actor_obs_dim(p) == OBS
    assert pe.effective_normalizer(p, None) is p.obs_normalizer
    x = torch.randn(64, OBS) * 2
    live = pe.live_actions(p, None, x)
    o = pe.export_actor_onnx(p, None, tmp_path, "policy.onnx")
    j = pe.export_actor_jit(p, p.obs_normalizer, tmp_path, "policy.pt")
    assert onnx_external_data_files(o) == [] and sorted(f.name for f in tmp_path.iterdir()) == ["policy.onnx", "policy.pt"]
    import onnx

    mdl = onnx.load(str(o))
    assert [i.name for i in mdl.graph.input] == ["obs"] and [i.name for i in mdl.graph.output] == ["actions"]
    assert max(op.version for op in mdl.opset_import if op.domain in ("", "ai.onnx")) == pe.ONNX_OPSET_RSLRL5
    pol = OnnxPolicy(o)
    assert pol.obs_input == "obs"
    s = ort.InferenceSession(str(o), providers=["CPUExecutionProvider"])
    y = np.concatenate([s.run(None, {"obs": x[i:i + 1].numpy()})[0] for i in range(len(x))])
    assert np.abs(y - live.numpy()).max() < 1e-5
    assert (torch.jit.load(str(j))(x) - live).abs().max() < 1e-5
    a, n = pe.actor_and_normalizer(p, None)
    assert isinstance(n, nn.Identity) and (a(x) - live).abs().max() < 1e-6
    assert (live - p.mlp(x)).abs().max() > 1e-3  # normalizer really applied


def test_rslrl5_rejects_second_normalizer():
    p = _MLPModel5()
    for fn in (pe.actor_and_normalizer, pe.effective_normalizer):
        with pytest.raises(ValueError, match="twice"):
            fn(p, _Norm(OBS))
    pe.actor_and_normalizer(p, p.obs_normalizer)  # the model's own one is fine
    pe.actor_and_normalizer(p, nn.Identity())


EXPORTERS = ("source/dropbear_wbc/tasks/tracking/export.py", "source/dropbear_wbc/tasks/tracking/export_library.py",
             "source/dropbear_wbc/tasks/locomotion/export.py")


def test_package_exports_route_through_policy_export():
    """No package module outside policy_export.py reads policy.actor/.student or imports the Isaac exporters."""
    offenders = []
    for f in sorted((ROOT / "source/dropbear_wbc").rglob("*.py")):
        rel = str(f.relative_to(ROOT))
        if rel.endswith("tasks/policy_export.py"):
            continue
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and n.attr in ("actor", "student") and isinstance(n.value, ast.Name) \
                    and n.value.id == "policy":
                offenders.append(f"{rel}:{n.lineno} policy.{n.attr}")
            if isinstance(n, ast.ImportFrom) and n.module == "isaaclab_rl.rsl_rl" and \
                    {a.name for a in n.names} & {"export_policy_as_onnx", "export_policy_as_jit"}:
                offenders.append(f"{rel}:{n.lineno} isaac exporter import")
    assert offenders == []
    for rel in EXPORTERS:
        calls = {getattr(n.func, "id", "") for n in ast.walk(ast.parse((ROOT / rel).read_text())) if isinstance(n, ast.Call)}
        assert {"export_actor_jit", "export_actor_onnx", "actor_obs_dim", "live_actions"} <= calls, rel
