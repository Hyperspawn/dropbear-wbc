"""Render-glitch check of the demo renderer (demo_eval, 2026-09-24): ``demo_render.frame_is_valid`` must reject the blank
frames the RTX annotator sometimes returns (all black; uniform washed-out grey without floor or robot) and accept a real
scene frame (coloured grid floor + a robot silhouette). CPU only, numpy only (no Isaac import)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc.tasks.tracking.demo_render import frame_is_valid  # noqa: E402

H, W = 720, 1280


def _scene() -> np.ndarray:
    img = np.empty((H, W, 3), np.uint8)
    img[: H // 3] = (222, 222, 222)  # light grey sky
    img[H // 3:] = (96, 140, 176)  # blue floor
    img[H // 3:, ::80] = (245, 245, 245)  # grid lines
    img[200:650, 600:680] = (235, 235, 230)  # robot silhouette
    return img


def test_real_scene_frame_is_valid():
    assert frame_is_valid(_scene())


def test_black_frame_is_invalid():
    assert not frame_is_valid(np.zeros((H, W, 3), np.uint8))


def test_washed_out_grey_frame_is_invalid():
    img = np.full((H, W, 3), 214, np.uint8)
    img += np.random.default_rng(0).integers(0, 4, img.shape, dtype=np.uint8)  # faint noise, still colourless
    assert not frame_is_valid(img)


def test_empty_or_wrong_rank_is_invalid():
    assert not frame_is_valid(np.zeros((0,), np.uint8))
    assert not frame_is_valid(np.zeros((H, W), np.uint8))


def test_rgba_input_accepted():
    rgba = np.concatenate([_scene(), np.full((H, W, 1), 255, np.uint8)], axis=2)
    assert frame_is_valid(rgba)
