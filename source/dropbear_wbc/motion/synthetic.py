"""Dropbear-native synthetic clips generated directly in the semantic joint space (no G1 source).

Clips (50 Hz, min-jerk keyframes; both feet planted and flat the whole time):

* ``stand``         10 s still at the calibration standing pose.
* ``wave_right``    right arm raised sideways, elbow bent, waved by shoulder yaw; legs still.
* ``wave_right_v2`` expressive wave: right hand raised high (forearm vertical), +-30 deg shoulder-yaw wave with an
                    in-phase wrist roll at 1.5 Hz for ~5 s, then return; legs still (added 2026-09-24, demo_eval).
* ``arm_swing``     both shoulders swing in anti-phase (walking-like), legs still.
* ``weight_shift``  slow lateral pelvis shift centre -> left -> centre -> right -> centre; hips/ankles
                    compensate via leg IK so the feet stay planted and flat.
* ``squat_lite``    two shallow squats; depth chosen so every leg joint stays inside the calibration's
                    valid range with margin (knee target = 70 % of its measured max).

Root consistency: the semantic pelvis pose is chosen so that, with the calibration segment lengths,
the soles of the semantic leg model rest on z = 0 at the standing feet positions (``leg_ik`` for the
shift/squat clips). The articulation root ``world`` pose follows from ``pelvis_T_root``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .calibration_view import CalibrationView
from .g1_to_dropbear import SemanticTrajectory
from .names import SEMANTIC_INDEX, SEMANTIC_NAMES
from .rotations import matrix_to_quat
from .semantic_skeleton import LegGeometry, leg_fk, leg_ik

__all__ = ["SYNTHETIC_CLIPS", "min_jerk", "probe_semantic_ranges", "make_clip", "SyntheticClip"]

FPS = 50.0


def min_jerk(tau: np.ndarray) -> np.ndarray:
    """Minimum-jerk blend s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 for tau in [0, 1] (clipped)."""
    tau = np.clip(tau, 0.0, 1.0)
    return tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)


def _keyframes(t: np.ndarray, keys: list[tuple[float, float]]) -> np.ndarray:
    """Piecewise min-jerk through (time [s], value) keys; constant outside."""
    out = np.full_like(t, keys[0][1], dtype=np.float64)
    for (t0, v0), (t1, v1) in zip(keys[:-1], keys[1:]):
        m = (t >= t0) & (t <= t1)
        out[m] = v0 + (v1 - v0) * min_jerk((t[m] - t0) / max(t1 - t0, 1e-9))
    out[t > keys[-1][0]] = keys[-1][1]
    return out


def probe_semantic_ranges(cal: CalibrationView, n: int = 721) -> dict[str, tuple[float, float]]:
    """Valid [lo, hi] per semantic DoF by round-tripping a sweep through the calibration SemanticMap,
    other DoFs at the standing pose. Works for any SemanticMap implementing the contract API."""
    base = cal.standing_semantic_pos
    grid = np.linspace(-np.pi, np.pi, n)
    out: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(SEMANTIC_NAMES):
        q = np.tile(base, (n, 1))
        q[:, i] = grid
        m, _ = cal.semantic_to_motor(q)
        back = cal.motor_to_semantic(m)
        ok = np.abs(back[:, i] - grid) < 1e-4
        if not ok.any():
            out[name] = (float(base[i]), float(base[i]))
        else:
            out[name] = (float(grid[ok].min()), float(grid[ok].max()))
    return out


@dataclass
class SyntheticClip:
    name: str
    semantic: SemanticTrajectory
    ik_residual: float
    suitability: list[str]
    description: str


def _standing(cal: CalibrationView, geom: LegGeometry) -> tuple[np.ndarray, float, np.ndarray]:
    """Standing semantic pose, pelvis height that puts the soles on z=0, and the ankle positions."""
    q0 = cal.standing_semantic_pos.copy()
    rot = np.eye(3)[None]
    fk = leg_fk(np.zeros((1, 3)), rot, q0[None], geom)
    z = -float(fk.sole_height.min())
    ankles = fk.ankle[0] + np.array([0.0, 0.0, z])
    return q0, z, ankles


def _with_ik(
    t: np.ndarray, q: np.ndarray, pelvis: np.ndarray, ankles: np.ndarray, geom: LegGeometry
) -> tuple[np.ndarray, float]:
    rot = np.tile(np.eye(3), (len(t), 1, 1))
    target = np.tile(ankles, (len(t), 1, 1))
    frot = np.tile(np.eye(3), (len(t), 2, 1, 1))
    return leg_ik(pelvis, rot, target, frot, q, geom)


def make_clip(name: str, cal: CalibrationView) -> SyntheticClip:
    """Build one synthetic clip (see module docstring)."""
    geom = LegGeometry.from_calibration(cal)
    q0, z0, ankles = _standing(cal, geom)
    ranges = probe_semantic_ranges(cal)
    si = SEMANTIC_INDEX

    def base(duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.arange(0.0, duration + 1e-9, 1.0 / FPS)
        q = np.tile(q0, (len(t), 1))
        pelvis = np.tile(np.array([0.0, 0.0, z0]), (len(t), 1))
        return t, q, pelvis

    def clamp(name_: str, v: float, frac: float = 0.9) -> float:
        lo, hi = ranges[name_]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * frac
        return float(np.clip(v, mid - half, mid + half))

    residual = 0.0
    if name == "stand":
        t, q, pelvis = base(10.0)
        flags, desc = ["in-place", "static"], "10 s still at the calibration standing pose"
    elif name == "wave_right":
        t, q, pelvis = base(10.0)
        roll = clamp("right_shoulder_roll", -1.3)
        elbow = clamp("right_elbow", q0[si["right_elbow"]] - 0.9)  # G1 elbow: smaller = more flexed
        q[:, si["right_shoulder_roll"]] = _keyframes(t, [(1.0, q0[si["right_shoulder_roll"]]), (2.5, roll), (7.5, roll), (9.0, q0[si["right_shoulder_roll"]])])
        q[:, si["right_elbow"]] = _keyframes(t, [(1.0, q0[si["right_elbow"]]), (2.5, elbow), (7.5, elbow), (9.0, q0[si["right_elbow"]])])
        amp = _keyframes(t, [(2.5, 0.0), (3.0, 1.0), (7.0, 1.0), (7.5, 0.0)])
        yaw0 = q0[si["right_shoulder_yaw"]]
        a = clamp("right_shoulder_yaw", yaw0 + 0.35) - yaw0
        q[:, si["right_shoulder_yaw"]] = yaw0 + a * amp * np.sin(2 * np.pi * 1.2 * (t - 2.5))
        flags, desc = ["upper-body-only"], "right arm raise (shoulder roll), elbow bend, 1.2 Hz shoulder-yaw wave"
    elif name == "wave_right_v2":
        # Expressive "hello" wave (demo_eval track, 2026-09-24). Raised pose solved in the idealised G1 semantic axes
        # (R = Ry(pitch) Rx(roll) Rz(yaw), forearm = R Ry(elbow) x): upper arm 15 deg above horizontal pointing
        # 40 deg to the right of forward, forearm vertical, hand above shoulder height. The wave is a shoulder-yaw
        # (humeral rotation) oscillation of +-30 deg at 1.5 Hz -- the forearm swings side to side about the upper-arm
        # axis -- plus an in-phase +-15 deg wrist roll; full amplitude 3.3-7.7 s, 0.8 s min-jerk amplitude ramps.
        t, q, pelvis = base(11.0)
        raised = {
            "right_shoulder_pitch": clamp("right_shoulder_pitch", np.deg2rad(-109.3)),
            "right_shoulder_roll": clamp("right_shoulder_roll", np.deg2rad(-38.4)),
            "right_shoulder_yaw": clamp("right_shoulder_yaw", np.deg2rad(12.3)),
            "right_elbow": clamp("right_elbow", np.deg2rad(15.0)),  # G1 elbow: 15 deg = 75 deg flexion
        }
        for jn, v in raised.items():
            j0 = q0[si[jn]]
            q[:, si[jn]] = _keyframes(t, [(1.0, j0), (2.5, v), (8.5, v), (10.0, j0)])
        amp = _keyframes(t, [(2.5, 0.0), (3.3, 1.0), (7.7, 1.0), (8.5, 0.0)])
        wave = np.sin(2 * np.pi * 1.5 * (t - 2.5))
        yc = raised["right_shoulder_yaw"]
        a_yaw = min(clamp("right_shoulder_yaw", yc + np.deg2rad(30.0)) - yc, yc - clamp("right_shoulder_yaw", yc - np.deg2rad(30.0)))
        q[:, si["right_shoulder_yaw"]] += a_yaw * amp * wave
        w0 = q0[si["right_wrist_roll"]]
        a_wr = min(clamp("right_wrist_roll", w0 + np.deg2rad(15.0)) - w0, w0 - clamp("right_wrist_roll", w0 - np.deg2rad(15.0)))
        q[:, si["right_wrist_roll"]] = w0 + a_wr * amp * wave
        flags = ["upper-body-only"]
        desc = (f"right hand raised high (upper arm 15 deg above horizontal, forearm vertical), 1.5 Hz wave: shoulder yaw "
                f"+-{np.degrees(a_yaw):.0f} deg + wrist roll +-{np.degrees(a_wr):.0f} deg for ~5 s, then return")
    elif name == "arm_swing":
        t, q, pelvis = base(10.0)
        amp = _keyframes(t, [(1.0, 0.0), (2.0, 1.0), (8.0, 1.0), (9.0, 0.0)])
        for side, ph in (("left", 0.0), ("right", np.pi)):
            j = si[f"{side}_shoulder_pitch"]
            a = clamp(f"{side}_shoulder_pitch", q0[j] + 0.45) - q0[j]
            q[:, j] = q0[j] + a * amp * np.sin(2 * np.pi * 0.9 * (t - 1.0) + ph)
        flags, desc = ["upper-body-only"], "anti-phase shoulder-pitch swing, 0.9 Hz, +-0.45 rad"
    elif name == "weight_shift":
        t, q, pelvis = base(14.0)
        shift = 0.40 * geom.hip_width  # [m] pelvis lateral travel
        dy = _keyframes(t, [(1.0, 0.0), (4.0, shift), (7.0, 0.0), (10.0, -shift), (13.0, 0.0)])
        pelvis[:, 1] = dy
        # Keep the hip->ankle distance of the standing pose: the pelvis follows an arc (drops slightly).
        leg_v = z0 - float(ankles[:, 2].mean())
        pelvis[:, 2] = z0 - (leg_v - np.sqrt(leg_v**2 - dy**2))
        q, residual = _with_ik(t, q, pelvis, ankles, geom)
        flags, desc = ["in-place", "balance"], f"lateral pelvis shift +-{shift:.3f} m, feet planted (leg IK)"
    elif name == "squat_lite":
        t, q, pelvis = base(18.0)
        k_lo, k_hi = ranges["left_knee"]
        knee_target = q0[si["left_knee"]] + 0.7 * (k_hi - q0[si["left_knee"]])
        depth = _squat_depth_for_knee(knee_target, q0, z0, ankles, geom, ranges)
        keys = [(1.5, 0.0), (4.0, -depth), (5.0, -depth), (7.5, 0.0), (10.0, 0.0), (12.5, -depth), (13.5, -depth), (16.0, 0.0)]
        pelvis[:, 2] = z0 + _keyframes(t, keys)
        q, residual = _with_ik(t, q, pelvis, ankles, geom)
        flags, desc = ["in-place", "squat"], f"two squats, pelvis drop {depth:.3f} m (knee target {knee_target:.2f} rad)"
    else:
        raise KeyError(f"unknown synthetic clip {name!r}; choose from {list(SYNTHETIC_CLIPS)}")

    rot = np.tile(np.eye(3), (len(t), 1, 1))
    sem = SemanticTrajectory(
        fps=FPS,
        pelvis_pos=pelvis,
        pelvis_quat_wxyz=matrix_to_quat(rot),
        q=q,
        contacts=np.ones((len(t), 2), dtype=bool),
        meta={"standing_pelvis_height_m": z0, "ik_max_residual": residual, "semantic_ranges_probe": ranges},
        q_requested=q.copy(),
    )
    return SyntheticClip(name=name, semantic=sem, ik_residual=residual, suitability=flags, description=desc)


def _squat_depth_for_knee(
    knee_target: float,
    q0: np.ndarray,
    z0: float,
    ankles: np.ndarray,
    geom: LegGeometry,
    ranges: dict[str, tuple[float, float]],
    margin: float = 0.9,
) -> float:
    """Largest pelvis drop [m] whose IK knee <= knee_target and all leg joints inside ``margin`` of range."""
    best = 0.0
    for depth in np.linspace(0.0, 0.5, 101)[1:]:
        pel = np.array([[0.0, 0.0, z0 - depth]])
        q, res = leg_ik(pel, np.eye(3)[None], ankles[None], np.tile(np.eye(3), (1, 2, 1, 1)), q0[None], geom)
        if res > 1e-6:
            break
        ok = q[0, SEMANTIC_INDEX["left_knee"]] <= knee_target
        for side in ("left", "right"):
            for j in ("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll"):
                n = f"{side}_{j}"
                lo, hi = ranges[n]
                mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * margin
                ok &= bool(mid - half <= q[0, SEMANTIC_INDEX[n]] <= mid + half)
        if not ok:
            break
        best = float(depth)
    return best


SYNTHETIC_CLIPS: dict[str, Callable[[CalibrationView], SyntheticClip]] = {
    n: (lambda cal, _n=n: make_clip(_n, cal))
    for n in ("stand", "wave_right", "wave_right_v2", "arm_swing", "weight_shift", "squat_lite")
}
