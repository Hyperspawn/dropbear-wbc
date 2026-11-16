"""``torch.quantile`` without its input-size cap.

torch 2.5.1 / 2.7.0 (the CPU test pin and the Isaac Lab 2.2 pin, docs/INSTALL.md) raise
``RuntimeError: quantile() input tensor is too large`` above 2**24 = 16,777,216 elements. Play rollouts reduce
envs x steps samples per motor (``motor_torque``) or per env (closure gap), so a 4096-env play longer than 4096
steps (82 s) crashed after the whole rollout, before ``summary.json`` was written. Newer torch (2.14) has no cap.

:func:`quantile` calls ``torch.quantile`` up to the cap (bit-identical to before) and above it uses a full sort
with the same linear interpolation (``torch.lerp`` between the floor and ceil ranks), so results match
``torch.quantile`` / ``numpy.quantile(method="linear")`` (float64 exactly; float32 to ~1 ulp, the rank rounding of
torch.quantile itself differs between torch versions). NaN anywhere gives NaN, as in ``torch.quantile``.
"""
from __future__ import annotations

import torch

TORCH_QUANTILE_MAX_NUMEL = 2**24


def quantile(x: torch.Tensor, q, *, max_numel: int = TORCH_QUANTILE_MAX_NUMEL) -> torch.Tensor:
    """Linear-interpolation quantile(s) of all elements of ``x`` (flattened). ``q``: float or 1-D sequence/tensor
    in [0, 1]. Returns a 0-d tensor for a scalar ``q``, else shape ``[len(q)]`` (as ``torch.quantile``)."""
    x = x.reshape(-1)
    if not x.is_floating_point():
        x = x.float()
    scalar = not isinstance(q, torch.Tensor) and not isinstance(q, (list, tuple))
    qt = torch.as_tensor(q, dtype=x.dtype, device=x.device).reshape(-1)
    if x.numel() == 0:
        raise RuntimeError("quantile() input tensor must be non-empty")
    if bool(((qt < 0) | (qt > 1)).any()):
        raise RuntimeError("quantile() q values must be in the range [0, 1]")
    if x.numel() <= max_numel:
        out = torch.quantile(x, qt)
    else:
        s = torch.sort(x).values
        n = s.numel()
        rank = qt.double() * (n - 1)  # double ranks: equals torch 2.14 bitwise, torch 2.7 within 1 float32 ulp
        lo = rank.floor().long().clamp(0, n - 1)
        hi = rank.ceil().long().clamp(0, n - 1)
        out = torch.lerp(s[lo], s[hi], (rank - lo.double()).to(x.dtype))
        if bool(torch.isnan(s[-1])):
            out = torch.full_like(out, float("nan"))
    return out.reshape(()) if scalar else out
