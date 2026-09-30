"""Resolve gym-registered config entry points without importing ``isaaclab_tasks`` (whose package import
walks every Isaac Lab task and needs ``isaaclab_assets``, which kit python does not have installed)."""
from __future__ import annotations

import importlib
from typing import Any


def load_entry_point(task: str, key: str) -> Any:
    """Instantiate ``gym.spec(task).kwargs[key]`` (``"module:Class"`` string, class, or instance)."""
    import gymnasium as gym

    spec = gym.spec(task)
    if key not in spec.kwargs:
        raise KeyError(f"{task} has no {key!r} (have {sorted(spec.kwargs)})")
    entry = spec.kwargs[key]
    if isinstance(entry, str):
        module_name, attr = entry.split(":")
        entry = getattr(importlib.import_module(module_name), attr)
    return entry() if isinstance(entry, type) else entry
