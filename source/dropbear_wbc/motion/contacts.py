"""Foot-contact hints from foot trajectories (height + horizontal-speed thresholds).

Inputs are world-frame sole heights [m] and foot-centre positions [m] sampled at ``fps``. A foot is
in contact when it is within ``height_thresh`` of the *rolling ground* and moves slower than
``speed_thresh`` horizontally; the boolean track is then cleaned with a majority filter and short
segments are removed. The rolling ground is the moving minimum (``ground_window_s``) of the lower
sole, so video-derived clips whose feet float by several cm (ASAP) still get stance contacts, while a
foot held up during single-leg balance or a short jump stays above it. These are *hints* for the
settle/tracking stages, not ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ContactParams", "HysteresisParams", "detect_contacts", "detect_contacts_hysteresis", "hysteresis_track",
           "estimate_ground", "horizontal_speed", "contact_segments"]


@dataclass(frozen=True)
class ContactParams:
    height_thresh: float = 0.05  # [m] above ground (source-robot scale)
    speed_thresh: float = 0.40  # [m/s] horizontal foot speed (source-robot scale)
    median_window_s: float = 0.10  # [s] majority filter
    min_segment_s: float = 0.05  # [s] drop shorter on/off segments
    ground_window_s: float = 2.0  # [s] rolling-ground window (moving min of the lower sole)


def horizontal_speed(foot_pos: np.ndarray, fps: float) -> np.ndarray:
    """|d/dt xy| [m/s] with central differences, (T,)."""
    xy = foot_pos[:, :2]
    v = np.gradient(xy, 1.0 / fps, axis=0)
    return np.linalg.norm(v, axis=1)


def estimate_ground(sole_heights: np.ndarray) -> float:
    """Ground height estimate [m]: 5th percentile of the per-frame lowest sole (robust to noise)."""
    return float(np.percentile(np.min(sole_heights, axis=1), 5.0))


def _majority(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = win | 1
    pad = k // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    c = np.convolve(xp, np.ones(k) / k, mode="valid")
    return c > 0.5


def _moving_min(x: np.ndarray, win: int) -> np.ndarray:
    k = max(1, win) | 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(xp, k).min(axis=1)


def _remove_short(x: np.ndarray, min_len: int) -> np.ndarray:
    x = x.copy()
    if min_len <= 1:
        return x
    for value in (True, False):
        segs = contact_segments(x) if value else contact_segments(~x)
        for s, e in segs:
            if e - s < min_len and s > 0 and e < len(x):  # never flip boundary segments
                x[s:e] = not value
    return x


def contact_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


@dataclass(frozen=True)
class HysteresisParams:
    """Contact detection with hysteresis (foot_contact track, 2026-09-24): a foot touches down when it is within
    ``h_on`` of the rolling ground AND slower than ``v_on``; it lifts off only when it rises above ``h_off`` OR moves
    faster than ``v_off``. Removes the single-threshold chatter of :func:`detect_contacts` near the threshold."""

    h_on: float = 0.035  # [m] source-robot scale
    h_off: float = 0.06
    v_on: float = 0.35  # [m/s]
    v_off: float = 0.60
    ground_window_s: float = 2.0
    min_stance_s: float = 0.10
    min_swing_s: float = 0.08


def hysteresis_track(height: np.ndarray, speed: np.ndarray, p: HysteresisParams) -> np.ndarray:
    """Two-threshold state machine per frame (T,) -> bool. Starts in contact if the first frame qualifies."""
    out = np.zeros(len(height), dtype=bool)
    on = bool(height[0] < p.h_on and speed[0] < p.v_on)
    for i in range(len(height)):
        if on:
            on = not (height[i] > p.h_off or speed[i] > p.v_off)
        else:
            on = bool(height[i] < p.h_on and speed[i] < p.v_on)
        out[i] = on
    return out


def detect_contacts_hysteresis(
    sole_heights: np.ndarray,
    foot_pos: np.ndarray,
    fps: float,
    params: HysteresisParams = HysteresisParams(),
    ground: float | None = None,
    speeds: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict]:
    """Like :func:`detect_contacts` (same rolling ground) but with hysteresis and min stance / swing durations.

    ``speeds`` (T, 2) optionally replaces the horizontal speed of ``foot_pos`` (e.g. the slower of a foot's toe and
    ankle, so that a foot pivoting on its toe or heel counts as planted and a switching proxy point does not create
    speed spikes). Returns ``(contacts (T, 2) bool, ground_z, signals)`` where ``signals`` holds the per-foot height
    above the rolling ground and the horizontal speed actually used (for reports)."""
    sole_heights = np.asarray(sole_heights, dtype=np.float64)
    g = estimate_ground(sole_heights) if ground is None else float(ground)
    rolling = _moving_min(sole_heights.min(axis=1), int(round(params.ground_window_s * fps)))
    rolling = np.maximum(rolling, g)
    out = np.zeros(sole_heights.shape, dtype=bool)
    heights = np.zeros(sole_heights.shape)
    speeds_used = np.zeros(sole_heights.shape)
    n_sw = max(1, int(round(params.min_swing_s * fps)))
    n_st = max(1, int(round(params.min_stance_s * fps)))
    for s in range(sole_heights.shape[1]):
        speed = horizontal_speed(foot_pos[:, s], fps) if speeds is None else np.asarray(speeds[:, s], dtype=np.float64)
        h = sole_heights[:, s] - rolling
        c = hysteresis_track(h, speed, params)
        for a, b in contact_segments(~c):  # short lifts: keep planted
            if b - a < n_sw and a > 0 and b < len(c):
                c[a:b] = True
        for a, b in contact_segments(c):  # short touches: swing
            if b - a < n_st:
                c[a:b] = False
        out[:, s] = c
        heights[:, s], speeds_used[:, s] = h, speed
    return out, g, {"height_above_ground": heights, "speed": speeds_used}


def detect_contacts(
    sole_heights: np.ndarray,
    foot_pos: np.ndarray,
    fps: float,
    params: ContactParams = ContactParams(),
    ground: float | None = None,
) -> tuple[np.ndarray, float]:
    """Per-frame contact booleans.

    ``sole_heights``: (T, 2) lowest sole z for [left, right]; ``foot_pos``: (T, 2, 3) foot centres.
    Returns ``(contacts (T, 2) bool, ground_z)``.
    """
    sole_heights = np.asarray(sole_heights, dtype=np.float64)
    g = estimate_ground(sole_heights) if ground is None else float(ground)
    rolling = _moving_min(sole_heights.min(axis=1), int(round(params.ground_window_s * fps)))
    rolling = np.maximum(rolling, g)  # never below the clip-level ground estimate
    out = np.zeros(sole_heights.shape, dtype=bool)
    win = max(1, int(round(params.median_window_s * fps)))
    min_len = max(1, int(round(params.min_segment_s * fps)))
    for s in range(2):
        speed = horizontal_speed(foot_pos[:, s], fps)
        raw = (sole_heights[:, s] - rolling < params.height_thresh) & (speed < params.speed_thresh)
        out[:, s] = _remove_short(_majority(raw, win), min_len)
    return out, g
