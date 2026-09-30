"""Hardware speed gate of the live text -> motion service (tools/kimodo_live_service.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")  # the service module imports torch at the top
_spec = importlib.util.spec_from_file_location(
    "kimodo_live_service", Path(__file__).resolve().parents[1] / "tools" / "kimodo_live_service.py")
kls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kls)


def _sine(freq_hz: float, amp: float = 0.5, fps: float = 50.0, seconds: float = 4.0) -> np.ndarray:
    t = np.arange(int(seconds * fps)) / fps
    return np.stack([amp * np.sin(2 * np.pi * freq_hz * t), 0.1 * t], axis=1)  # motor 0 oscillates, motor 1 creeps


def test_slow_clip_passes_unchanged():
    g = kls.speed_gate(_sine(0.5), ["a", "b"], 50.0, {"a": 10.0, "b": 10.0}, max_stretch=1.6)
    assert g["stretch"] == 1.0 and not g["refuse"] and g["worst_motor"] == "a"


def test_fast_clip_is_slowed_to_the_no_load_speed_then_refused_beyond_the_limit():
    no_load = {"a": 2.0, "b": 10.0}
    q = _sine(1.0)  # peak speed 2*pi*1*0.5 = 3.14 rad/s -> ~1.57x motor a's no-load speed
    g = kls.speed_gate(q, ["a", "b"], 50.0, no_load, max_stretch=1.8)
    assert 1.5 < g["stretch"] < 1.8 and not g["refuse"]
    slowed = kls._stretch(q, g["stretch"])
    assert kls.speed_gate(slowed, ["a", "b"], 50.0, no_load, max_stretch=1.8)["p99_over_noload"] <= 1.0
    assert kls.speed_gate(q, ["a", "b"], 50.0, no_load, max_stretch=1.4)["refuse"]


def test_stretch_keeps_endpoints_and_lengthens():
    x = np.linspace(0.0, 1.0, 11)[:, None]
    y = kls._stretch(x, 1.5)
    assert len(y) == 16 and y[0, 0] == 0.0 and y[-1, 0] == pytest.approx(1.0) and np.all(np.diff(y[:, 0]) > 0)
