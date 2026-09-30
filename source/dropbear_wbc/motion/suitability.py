"""Heuristic suitability flags for a retargeted clip (catalog tags, not guarantees).

Inputs are the G1-scale diagnostics written by ``g1_to_semantic`` (``meta``), the contact track and the
saturation summary. Thresholds are deliberately simple and listed in :data:`THRESHOLDS` so the catalog
records exactly how each flag was decided.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contacts import contact_segments

__all__ = ["THRESHOLDS", "suitability_flags"]

THRESHOLDS: dict[str, float] = {
    "locomotion_mean_speed_mps": 0.25,  # G1 hip-centre mean horizontal speed
    "locomotion_path_m": 1.0,
    "upper_body_only_root_path_m": 0.30,
    "upper_body_only_leg_range_rad": 0.35,  # max-min of every G1 leg joint
    "upper_body_only_contact_frac": 0.90,
    "jump_flight_s": 0.08,  # both feet off the ground ...
    "jump_launch_speed_mps": 0.5,  # ... and the G1 hip centre rises at least this fast
    "single_leg_s": 0.60,  # one foot off while the other is on
    "deep_knee_rad": 1.2,
    "low_pelvis_ratio": 0.70,  # hip height / standing hip height
    "turn_deg": 90.0,
    "high_saturation_frac": 0.25,  # frames with any semantic saturation
    "noisy_ground_median_m": 0.04,  # median lowest-sole height above estimated ground
}


def suitability_flags(
    meta: dict[str, Any],
    contacts: np.ndarray | None,
    fps: float,
    leg_ranges_rad: np.ndarray,
    saturation_frac: float,
) -> list[str]:
    th = THRESHOLDS
    flags: list[str] = []
    rs = meta["g1_root_speed_mps"]
    if rs["mean"] > th["locomotion_mean_speed_mps"] or rs["path_length_m"] > th["locomotion_path_m"]:
        flags.append("locomotion")
    cfrac = 1.0
    if contacts is not None:
        both = contacts.all(axis=1)
        cfrac = float(both.mean())
        flight = ~contacts.any(axis=1)
        if rs["max_up_velocity"] > th["jump_launch_speed_mps"] and any(
            (e - s) / fps >= th["jump_flight_s"] for s, e in contact_segments(flight)
        ):
            flags.append("jump")
        single = contacts.any(axis=1) & ~both
        if any((e - s) / fps >= th["single_leg_s"] for s, e in contact_segments(single)):
            flags.append("single-leg-support")
    if (
        rs["path_length_m"] < th["upper_body_only_root_path_m"]
        and float(np.max(leg_ranges_rad)) < th["upper_body_only_leg_range_rad"]
        and cfrac >= th["upper_body_only_contact_frac"]
    ):
        flags.append("upper-body-only")
    if meta["g1_max_knee_rad"] > th["deep_knee_rad"]:
        flags.append("deep-knee")
    if meta["g1_min_hip_height_ratio"] < th["low_pelvis_ratio"]:
        flags.append("low-pelvis")
    if meta["g1_yaw_change_deg"] > th["turn_deg"]:
        flags.append("turning")
    if saturation_frac > th["high_saturation_frac"]:
        flags.append("high-saturation")
    if meta["g1_sole_height_raw"]["median"] - meta["g1_ground_z_m"] > th["noisy_ground_median_m"]:
        flags.append("noisy-ground-contact")
    if not flags or flags == ["high-saturation"]:
        flags.append("in-place")
    return flags
