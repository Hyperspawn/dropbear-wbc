"""Left/right mirror consistency of the motor target clamp under ``--symmetry`` (item 250; pure numpy, no Isaac).

The velocity task clips each motor position target to ``MOTOR_HARD_LIMITS_DEG`` +- margin (gait-shaped runs,
``flat_env_cfg.clamp_targets_to_limits``). PPO symmetry augmentation (``symmetry.compute_symmetric_states``) mirrors
observations and actions with the affine motor map ``q' = q0 + M (q - q0)`` (``M = motor_mirror_matrix``, ``q0`` =
calibration ``standing_motor_pos``, the action offset). A transition is only a valid mirrored sample if the clamp box
maps onto itself. It does not: the USD authors ``RL_Revolute81`` as (-60, 50) deg while ``LL_Revolute81`` and both
``*_Revolute67`` are (-50, 60), and ``M`` maps ``LL_Revolute81 -> RL_Revolute81`` with gain +0.996, so the mirror of
the left box is about (-50, 60) on the right: a ~10 deg offset at both ends. A left target held on its -50 bound mirrors
to a right target the right clamp does not stop (the real right joint goes to -60), and a right joint between -60 and
-50 mirrors to a left joint position beyond the left USD limit. Every other motor mirrors within 1.6 deg.

``mirrored_limits_deg`` gives the interval hull of the mirrored box, ``clamp_mirror_report`` the per-motor offsets,
``symmetric_limits_deg`` the box intersected with its mirror (``--symmetric_clamp``), ``check_symmetry_clamp`` the
train_locomotion gate. Whether the hardware R81 stop is at -60/50 or -50/60 needs CAD / the robot (notes/250).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from dropbear_wbc.kinematics.semantic import DEFAULT_CALIBRATION
from dropbear_wbc.robots.dropbear_names import MOTOR_HARD_LIMITS_DEG, MOTOR_NAMES

MIRROR_TOL_DEG: float = 2.0
"""A clamp bound whose mirror lands more than this from the partner's bound is mirror-inconsistent. Above the
calibration's own L/R gain residue (elbows 1.55 deg from the 1.052 elbow gain, ankle R67 0.70 deg), far below the
RL_Revolute81 offset (10.1 deg)."""


def _limits_array(limits: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    missing = [n for n in MOTOR_NAMES if n not in limits]
    if missing:
        raise KeyError(f"clamp limits miss motors {missing}")
    lo = np.array([float(limits[n][0]) for n in MOTOR_NAMES])
    hi = np.array([float(limits[n][1]) for n in MOTOR_NAMES])
    if np.any(lo > hi):
        raise ValueError("clamp limits with lo > hi")
    return lo, hi


def _mirror_and_q0(calibration) -> tuple[np.ndarray, np.ndarray]:
    from dropbear_wbc.tasks.locomotion.symmetry import motor_mirror_matrix

    calib = json.loads(Path(calibration).read_text(encoding="utf-8"))
    q0 = np.degrees(np.asarray(calib["standing_motor_pos"], dtype=float))
    return np.asarray(motor_mirror_matrix(calibration)), q0


def _hull(lo: np.ndarray, hi: np.ndarray, m: np.ndarray, q0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interval hull of ``q0 + M (box - q0)`` (exact per row: centre + |M| half-widths)."""
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    c = q0 + m @ (mid - q0)
    h = np.abs(m) @ half
    return c - h, c + h


def mirrored_limits_deg(limits: dict[str, tuple[float, float]] = MOTOR_HARD_LIMITS_DEG,
                        calibration: str | Path = DEFAULT_CALIBRATION) -> dict[str, tuple[float, float]]:
    """Per motor: the range its target spans when the partner side's clamp box is mirrored onto it [deg]."""
    lo, hi = _limits_array(limits)
    m, q0 = _mirror_and_q0(calibration)
    mlo, mhi = _hull(lo, hi, m, q0)
    return {n: (float(a), float(b)) for n, a, b in zip(MOTOR_NAMES, mlo, mhi)}


def clamp_mirror_report(limits: dict[str, tuple[float, float]] = MOTOR_HARD_LIMITS_DEG,
                        calibration: str | Path = DEFAULT_CALIBRATION) -> dict[str, dict]:
    """``{motor: {limits, mirrored, d_lo, d_hi, max_abs_deg}}``; ``d = mirrored - own`` [deg]."""
    mir = mirrored_limits_deg(limits, calibration)
    out = {}
    for n in MOTOR_NAMES:
        lo, hi = (float(v) for v in limits[n])
        d_lo, d_hi = mir[n][0] - lo, mir[n][1] - hi
        out[n] = {"limits": [lo, hi], "mirrored": [round(mir[n][0], 3), round(mir[n][1], 3)],
                  "d_lo": round(d_lo, 3), "d_hi": round(d_hi, 3), "max_abs_deg": round(max(abs(d_lo), abs(d_hi)), 3)}
    return out


def asymmetric_motors(limits: dict[str, tuple[float, float]] = MOTOR_HARD_LIMITS_DEG,
                      calibration: str | Path = DEFAULT_CALIBRATION, tol_deg: float = MIRROR_TOL_DEG) -> dict[str, float]:
    """``{motor: max |mirrored - own| [deg]}`` for the motors above ``tol_deg``."""
    return {n: r["max_abs_deg"] for n, r in clamp_mirror_report(limits, calibration).items() if r["max_abs_deg"] > tol_deg}


def symmetric_limits_deg(limits: dict[str, tuple[float, float]] = MOTOR_HARD_LIMITS_DEG,
                         calibration: str | Path = DEFAULT_CALIBRATION) -> dict[str, tuple[float, float]]:
    """The clamp box intersected with its mirror: each bound is the tighter of its own and the partner's mirrored one
    (RL/LL_Revolute81 -> about (-50, 50)). Never wider than ``limits``, so no target leaves either side's USD range.
    The residue (mirror of a box is a slightly sheared box: ankle / hip cross terms) stays < 0.7 deg."""
    lo, hi = _limits_array(limits)
    m, q0 = _mirror_and_q0(calibration)
    mlo, mhi = _hull(lo, hi, m, q0)
    slo, shi = np.maximum(lo, mlo), np.minimum(hi, mhi)
    if np.any(slo > shi):
        raise ValueError("clamp box and its mirror do not overlap")
    return {n: (float(a), float(b)) for n, a, b in zip(MOTOR_NAMES, slo, shi)}


def clamp_clip_rad(margin_deg: float = 0.0, symmetric: bool = False,
                   limits: dict[str, tuple[float, float]] = MOTOR_HARD_LIMITS_DEG,
                   calibration: str | Path = DEFAULT_CALIBRATION) -> dict[str, tuple[float, float]]:
    """The ``actions.joint_pos.clip`` dict [rad]: (symmetric) limits +- ``margin_deg``."""
    lim = symmetric_limits_deg(limits, calibration) if symmetric else limits
    m = math.radians(margin_deg)
    return {n: (math.radians(lo) - m, math.radians(hi) + m) for n, (lo, hi) in lim.items()}


def clamp_mirror_record(clip_rad: dict[str, tuple[float, float]] | None, symmetric_clamp: bool,
                        calibration: str | Path = DEFAULT_CALIBRATION, tol_deg: float = MIRROR_TOL_DEG) -> dict | None:
    """run_info block for the clamp actually built: worst mirror offset and the motors above ``tol_deg``."""
    if not clip_rad:
        return None
    lim = {n: (math.degrees(a), math.degrees(b)) for n, (a, b) in clip_rad.items()}
    rep = clamp_mirror_report(lim, calibration)
    worst = max(rep, key=lambda n: rep[n]["max_abs_deg"])
    return {"symmetric_clamp": bool(symmetric_clamp), "tol_deg": float(tol_deg),
            "max_mismatch_deg": rep[worst]["max_abs_deg"], "worst_motor": worst,
            "asymmetric_motors": {n: rep[n]["max_abs_deg"] for n in rep if rep[n]["max_abs_deg"] > tol_deg},
            "clip_deg": {n: [round(lo, 4), round(hi, 4)] for n, (lo, hi) in lim.items()}}


def check_symmetry_clamp(*, symmetry: bool, clamp_active: bool, symmetric_clamp: bool, continue_run: bool,
                         allow_asymmetric: bool = False,
                         limits: dict[str, tuple[float, float]] = MOTOR_HARD_LIMITS_DEG,
                         calibration: str | Path = DEFAULT_CALIBRATION, tol_deg: float = MIRROR_TOL_DEG) -> list[str]:
    """train_locomotion gate. ``--symmetry`` on a mirror-inconsistent target clamp is refused for a fresh run unless
    ``--symmetric_clamp`` (clamp to the mirrored intersection) or ``--allow_asymmetric_clamp``; a ``--continue_run``
    chunk keeps its run's clamp (changing it is a settings change, item 182) and only warns."""
    if not (symmetry and clamp_active) or symmetric_clamp:
        return []
    bad = asymmetric_motors(limits, calibration, tol_deg)
    if not bad:
        return []
    desc = ", ".join(f"{n} {v:.1f} deg" for n, v in bad.items())
    msg = (f"--symmetry on a target clamp that is not left/right mirror consistent ({desc} > {tol_deg} deg; "
           "RL_Revolute81 USD (-60, 50) vs LL (-50, 60), notes/250): mirrored transitions near those bounds do not "
           "match the env")
    if continue_run:
        return [f"WARNING {msg} (--continue_run keeps the run's clamp)"]
    if allow_asymmetric:
        return [f"WARNING {msg} (--allow_asymmetric_clamp)"]
    raise ValueError(f"{msg}. Pass --symmetric_clamp (clamp both sides to the mirrored intersection) or "
                     "--allow_asymmetric_clamp")
