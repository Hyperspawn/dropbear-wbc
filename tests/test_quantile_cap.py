"""``dropbear_wbc.isaac.quantile.quantile``: torch.quantile without the 2**24-element cap of torch 2.5.1 / 2.7.0
(item 177). Play summaries (motor_torque p95, closure-gap quantiles) reduce envs x steps samples; above the cap
the old code raised after the whole rollout, before summary.json was written."""
from __future__ import annotations

import ast
import functools

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ and third_party/pydeps)

torch = pytest.importorskip("torch")

from dropbear_wbc.isaac import motor_torque as mt  # noqa: E402
from dropbear_wbc.isaac.quantile import TORCH_QUANTILE_MAX_NUMEL, quantile  # noqa: E402

ROOT = sdk_test_paths.ROOT
QS = [0.0, 0.01, 0.5, 0.95, 0.99, 1.0]


@pytest.mark.parametrize("n", [1, 2, 3, 7, 100, 1001])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sort_path_matches_torch_quantile(n, dtype):
    """float64 exact; float32 within 1 ulp (torch 2.7 forms float32 ranks, 2.14 double ranks)."""
    g = torch.Generator().manual_seed(n)
    x = torch.randn(n, generator=g, dtype=dtype)
    x[: n // 3] = x[0]  # ties
    ref = torch.quantile(x, torch.tensor(QS, dtype=dtype))
    got = quantile(x, QS, max_numel=0)
    tol = dict(rtol=0, atol=0) if dtype == torch.float64 else dict(rtol=2.4e-7, atol=2.4e-7 * 4)
    torch.testing.assert_close(got, ref, **tol)
    for q in QS:  # scalar q -> 0-d, like torch.quantile
        r, s = torch.quantile(x, q), quantile(x, q, max_numel=0)
        assert s.shape == r.shape == ()
        torch.testing.assert_close(s, r, **tol)


def test_sort_path_nan_and_shape_and_errors():
    x = torch.tensor([[1.0, 2.0], [3.0, float("nan")]])
    assert torch.isnan(quantile(x, [0.1, 0.5], max_numel=0)).all()
    assert torch.equal(quantile(torch.arange(10).view(2, 5), 0.5, max_numel=0), torch.tensor(4.5))
    with pytest.raises(RuntimeError):
        quantile(torch.zeros(0), 0.5, max_numel=0)
    with pytest.raises(RuntimeError):
        quantile(torch.zeros(4), 1.5, max_numel=0)
    with pytest.raises(RuntimeError):
        quantile(torch.zeros(4), -0.1, max_numel=0)


def test_above_real_cap_matches_numpy():
    n = TORCH_QUANTILE_MAX_NUMEL + 3
    x = torch.rand(n, generator=torch.Generator().manual_seed(0))
    got = quantile(x, [0.5, 0.95, 0.99]).double().numpy()
    ref = np.quantile(x.double().numpy(), [0.5, 0.95, 0.99])
    np.testing.assert_allclose(got, ref, atol=1e-6)


def _capped_torch_quantile(cap):
    real = torch.quantile

    @functools.wraps(real)
    def capped(inp, *a, **k):
        if inp.numel() > cap:
            raise RuntimeError("quantile() input tensor is too large")
        return real(inp, *a, **k)
    return capped


def test_motor_torque_block_survives_capped_torch(monkeypatch):
    """Emulate torch 2.7's cap at 10 elements: the sampler block must still be produced, identical to uncapped."""
    names = ["a", "b"]
    act = type("A", (), {})()
    act.joint_names, act.effort_limit = names, torch.full((4, 2), 10.0)
    sampler = mt.MotorTorqueSampler({"g": act}, names, 4, "cpu")
    g = torch.Generator().manual_seed(1)
    for _ in range(13):  # 52 samples per motor > cap
        tq = torch.randn(4, 2, generator=g) * 5
        sampler.add(torch.randn(4, 2, generator=g), torch.ones(4, dtype=torch.bool), efforts=(tq, tq * 1.1))
    ref = sampler.block("n")
    monkeypatch.setattr(torch, "quantile", _capped_torch_quantile(10))
    with pytest.raises(RuntimeError, match="too large"):
        torch.quantile(torch.zeros(11), 0.5)
    monkeypatch.setattr(mt, "quantile", functools.partial(mt.quantile, max_numel=10))
    assert sampler.block("n") == ref


def test_no_direct_torch_quantile_in_rollout_code():
    """Every play/probe reduction goes through the helper (np.quantile is uncapped and fine)."""
    bad = []
    for sub in ("scripts", "tools", "source"):
        for p in (ROOT / sub).rglob("*.py"):
            if p.name == "quantile.py" and p.parent.name == "isaac":
                continue
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("quantile", "nanquantile")
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"):
                    bad.append(f"{p.relative_to(ROOT)}:{node.lineno}")
    assert not bad, bad
