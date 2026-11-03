"""Policy export across rsl-rl 2.x (Isaac Lab 2.2) and rsl-rl 5.x (Isaac Lab 3.0).

rsl-rl 2.x hands the runner an ``ActorCritic`` (``policy.actor`` Sequential, normalizer on the runner as
``runner.obs_normalizer``). rsl-rl 5.x hands an ``MLPModel``: no ``actor``/``student`` attribute, the
normalizer lives inside the model (``model.obs_normalizer``) and export goes through ``as_jit()`` /
``as_onnx()``. Isaac Lab 3.0's ``export_policy_as_onnx`` raises ``ValueError`` on an ``MLPModel``.

Every export module goes through this file, so the file contract stays the same on both stacks:
``policy.pt`` / ``policy.onnx`` with input ``obs`` (1, obs_dim) and output ``actions``, normalizer baked in,
deterministic (mean) output, single-file ONNX.

On rsl-rl 2.x every function delegates to exactly the code path used before (Isaac Lab exporters,
``policy.actor``), so 2.x exports are unchanged.
"""
from __future__ import annotations

import copy
import inspect
import os
from pathlib import Path

import torch

ONNX_OPSET_RSLRL5 = 18
"""Opset for rsl-rl 5.x exports (same as rsl-rl 5.5.1 ``OnPolicyRunner.export_policy_to_onnx`` and the
Isaac Lab 3.0 exporter); needs onnxruntime >= 1.14 on the deploy computer."""


def is_rslrl5_model(policy) -> bool:
    """True for an rsl-rl 5.x ``MLPModel``-style policy (``as_onnx`` and no ``actor``/``student``)."""
    return hasattr(policy, "as_onnx") and not hasattr(policy, "actor") and not hasattr(policy, "student")


def _model_normalizer(policy, normalizer):
    """rsl-rl 5.x: the model's own normalizer. An extra, different normalizer would normalize twice."""
    own = getattr(policy, "obs_normalizer", None)
    if normalizer is not None and normalizer is not own and not isinstance(normalizer, torch.nn.Identity):
        raise ValueError(
            "rsl-rl 5.x model carries its own obs_normalizer; passing a second normalizer would apply it twice"
        )
    return own if own is not None else torch.nn.Identity()


def effective_normalizer(policy, normalizer):
    """The normalizer that the exported graph applies (for sidecar statistics and live parity)."""
    if is_rslrl5_model(policy):
        return _model_normalizer(policy, normalizer)
    return normalizer if normalizer is not None else torch.nn.Identity()


def actor_obs_dim(policy) -> int:
    """Actor input width (concatenated policy observation)."""
    if is_rslrl5_model(policy):
        return int(policy.obs_dim)
    return int(policy.actor[0].in_features)


def actor_and_normalizer(policy, normalizer) -> tuple[torch.nn.Module, torch.nn.Module]:
    """CPU copies ``(actor, normalizer)`` with ``actions = actor(normalizer(obs))`` deterministic.

    rsl-rl 2.x: ``policy.actor`` and the runner normalizer (attribute names kept, so graphs built from
    them are unchanged). rsl-rl 5.x: ``as_onnx()`` (normalizer + mlp + deterministic output) and Identity.
    """
    if is_rslrl5_model(policy):
        _model_normalizer(policy, normalizer)  # reject a second normalizer
        return policy.as_onnx(verbose=False).cpu().eval(), torch.nn.Identity()
    actor = copy.deepcopy(policy.actor).cpu()
    norm = copy.deepcopy(normalizer).cpu() if normalizer is not None else torch.nn.Identity()
    return actor, norm


@torch.no_grad()
def live_actions(policy, normalizer, x: torch.Tensor) -> torch.Tensor:
    """Deterministic actions of the live (training-side) policy for parity checks."""
    if is_rslrl5_model(policy):
        norm = _model_normalizer(policy, normalizer)
        dist = getattr(policy, "distribution", None)
        head = dist.as_deterministic_output_module() if dist is not None else torch.nn.Identity()
        return head(policy.mlp(norm(x)))
    return policy.act_inference(normalizer(x))


def _onnx_export_kwargs() -> dict:
    # torch >= 2.9 defaults to the dynamo exporter (external .onnx.data, opset forced to 18); keep the
    # TorchScript exporter where the switch exists so the graph is one file at the requested opset.
    return {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}


def export_actor_jit(policy, normalizer, path: str | os.PathLike, filename: str = "policy.pt") -> Path:
    """TorchScript ``actions = actor(normalizer(obs))``."""
    os.makedirs(path, exist_ok=True)
    out = Path(path) / filename
    if is_rslrl5_model(policy):
        _model_normalizer(policy, normalizer)
        m = policy.as_jit().cpu().eval()
        torch.jit.script(m).save(str(out))
        return out
    from isaaclab_rl.rsl_rl import export_policy_as_jit

    export_policy_as_jit(policy, normalizer, path=str(path), filename=filename)
    return out


def export_actor_onnx(policy, normalizer, path: str | os.PathLike, filename: str = "policy.onnx") -> Path:
    """Single-file ONNX, input ``obs`` (1, obs_dim), output ``actions``."""
    from dropbear_wbc.deploy.policy import make_onnx_self_contained

    os.makedirs(path, exist_ok=True)
    out = Path(path) / filename
    if is_rslrl5_model(policy):
        _model_normalizer(policy, normalizer)
        m = policy.as_onnx(verbose=False).cpu().eval()
        torch.onnx.export(
            m,
            (torch.zeros(1, actor_obs_dim(policy)),),
            str(out),
            export_params=True,
            opset_version=ONNX_OPSET_RSLRL5,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
            **_onnx_export_kwargs(),
        )
    else:
        from isaaclab_rl.rsl_rl import export_policy_as_onnx

        export_policy_as_onnx(policy, normalizer=normalizer, path=str(path), filename=filename)
    make_onnx_self_contained(out)
    return out
