"""Ankle-roll target clamp for the pinned tie-rod ankle (docs/ROADMAP.md, "Tie-rod top ends").

The real tie-rod top ends are pins, so the ankle binds at about +-8 deg of roll (docs/CONTRACTS.md 0.2). The per-motor
target clamp (``MOTOR_HARD_LIMITS_DEG``) is a box in (Revolute67, Revolute81) space, and the bind boundary is diagonal
in that space: 64 % (left) / 67 % (right) of the box is beyond 8 deg of roll, so a target on the motor clamp can hold
kp * gap into the pins (left R81 at -50: 12 deg past the bind, 16.8 N*m at kp 80).

:class:`AnkleRollBind` clamps a motor target in semantic space instead: (a, b) -> (pitch, roll) through the measured
calibration; if |roll| > limit, the target becomes the motors for (pitch, +-limit) (pitch kept, then projected onto the
feasible set). Targets inside the limit are returned unchanged (bit for bit).

* :meth:`AnkleRollBind.clamp` - exact, numpy, any batch shape; the deploy FSM uses it (sidecar ``ankle_roll_bind``).
* :meth:`AnkleRollBind.lut` - the same map tabulated on the calibration's motor grid as a correction (da, db) that is
  zero inside the limit; :func:`apply_lut_torch` evaluates it bilinearly on GPU for training.
Units: radians unless the name says deg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dropbear_wbc.kinematics.semantic import MOTOR_NAMES, SemanticMap

DEFAULT_ROLL_LIMIT_DEG = 8.0
"""Pinned tie-rod bind (docs/ROADMAP.md, docs/CONTRACTS.md 0.2)."""


@dataclass(frozen=True)
class AnkleLut:
    side: str
    motors: tuple[str, str]  # (a, b) motor names
    a_grid: np.ndarray  # (na,)
    b_grid: np.ndarray  # (nb,)
    delta: np.ndarray  # (na, nb, 2) clamped - original motors at each node; 0 inside the limit


class AnkleRollBind:
    """Semantic ankle-roll clamp on motor targets: |roll| <= ``roll_limit_deg - margin_deg``."""

    def __init__(self, smap: SemanticMap | None = None, roll_limit_deg: float = DEFAULT_ROLL_LIMIT_DEG,
                 margin_deg: float = 1.0):
        self.smap = smap if smap is not None else SemanticMap.load()
        self.roll_limit_deg = float(roll_limit_deg)
        self.margin_deg = float(margin_deg)
        self.limit = math.radians(self.roll_limit_deg - self.margin_deg)
        if not self.limit > 0.0:
            raise ValueError(f"roll limit {roll_limit_deg} - margin {margin_deg} deg must be > 0")
        sides = ["left", "right"]
        self.pairs = []  # (side, pair)
        from dropbear_wbc.kinematics.semantic import SEMANTIC_NAMES

        for ip, _ir, pair in self.smap._pairs:
            self.pairs.append((SEMANTIC_NAMES[ip].split("_")[0], pair))
        if sorted(s for s, _ in self.pairs) != sorted(sides):
            raise ValueError("calibration must have left and right ankle_pairs")

    @property
    def motor_indices(self) -> list[tuple[int, int]]:
        """(a, b) indices into MOTOR_NAMES per side (left first)."""
        return [(p.ma, p.mb) for _, p in sorted(self.pairs, key=lambda t: t[0] != "left")]

    def sidecar(self) -> dict:
        return {"roll_limit_deg": self.roll_limit_deg, "margin_deg": self.margin_deg}

    def _clamp_pair(self, pair, ab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ab (..., 2) -> (clamped ab, over mask)."""
        m = np.zeros(ab.shape[:-1] + (len(MOTOR_NAMES),))
        m[..., pair.ma], m[..., pair.mb] = ab[..., 0], ab[..., 1]
        pr = pair.fwd(m)
        over = np.abs(pr[..., 1]) > self.limit
        out = ab.copy()
        if over.any():
            tgt = pr[over].copy()
            tgt[:, 1] = np.clip(tgt[:, 1], -self.limit, self.limit)
            pr2 = pair.project(tgt)
            far = np.abs(pr2[:, 1]) > self.limit + 1e-9  # (pitch, +-limit) outside the feasible set: the projection
            if far.any():  # fell back to a grid node beyond the limit; use the nearest node inside it instead
                vp = pair.valid_points[np.abs(pair.valid_points[:, 1]) <= self.limit]
                d2 = ((tgt[far][:, None, :] - vp[None, :, :]) ** 2).sum(-1)
                pr2[far] = vp[np.argmin(d2, axis=1)]
            out[over] = pair.inv(pr2)
        return out, over

    def clamp(self, q_motor: np.ndarray, return_mask: bool = False):
        """Motor targets (..., 22) in MOTOR_NAMES order [rad] -> clamped copy. ``return_mask``: also return a
        (..., 2) bool array (left, right) of the targets that were moved."""
        q = np.array(q_motor, dtype=np.float64, copy=True)
        if q.shape[-1] != len(MOTOR_NAMES):
            raise ValueError(f"expected (..., {len(MOTOR_NAMES)}), got {q.shape}")
        masks = []
        for _, pair in sorted(self.pairs, key=lambda t: t[0] != "left"):
            ab, over = self._clamp_pair(pair, np.stack([q[..., pair.ma], q[..., pair.mb]], -1))
            q[..., pair.ma], q[..., pair.mb] = ab[..., 0], ab[..., 1]
            masks.append(over)
        return (q, np.stack(masks, -1)) if return_mask else q

    def roll(self, q_motor: np.ndarray) -> np.ndarray:
        """Semantic ankle roll (..., 2) (left, right) [rad] of motor targets (calibration forward map)."""
        q = np.asarray(q_motor, dtype=np.float64)
        return np.stack([p.fwd(q)[..., 1] for _, p in sorted(self.pairs, key=lambda t: t[0] != "left")], -1)

    def lut(self, oversample: int = 2) -> list[AnkleLut]:
        """Tabulated correction on the calibration motor grid (``oversample`` x finer), left first."""
        out = []
        for side, pair in sorted(self.pairs, key=lambda t: t[0] != "left"):
            ag = np.linspace(pair.ag[0], pair.ag[-1], (len(pair.ag) - 1) * oversample + 1)
            bg = np.linspace(pair.bg[0], pair.bg[-1], (len(pair.bg) - 1) * oversample + 1)
            ab = np.stack(np.meshgrid(ag, bg, indexing="ij"), -1)
            c, _ = self._clamp_pair(pair, ab)
            out.append(AnkleLut(side, (MOTOR_NAMES[pair.ma], MOTOR_NAMES[pair.mb]), ag, bg, c - ab))
        return out


def apply_lut_torch(targets, luts, columns, torch_tables=None):
    """Torch version for training: ``targets`` (N, k) motor targets, ``columns`` [(ia, ib), ...] their column indices
    per entry of ``luts``. Adds the bilinearly interpolated correction in place-free form and returns the new tensor.
    ``torch_tables``: cache from :func:`lut_tables_torch` (device/dtype of ``targets``)."""
    import torch

    tabs = torch_tables if torch_tables is not None else lut_tables_torch(luts, targets.device, targets.dtype)
    out = targets.clone()
    for (ia, ib), (a0, da_, na, b0, db_, nb, delta) in zip(columns, tabs):
        a, b = targets[:, ia], targets[:, ib]
        fa = ((a - a0) / da_).clamp(0.0, na - 1.0)
        fb = ((b - b0) / db_).clamp(0.0, nb - 1.0)
        i0 = fa.floor().long().clamp(max=na - 2)
        j0 = fb.floor().long().clamp(max=nb - 2)
        ta, tb = (fa - i0).unsqueeze(-1), (fb - j0).unsqueeze(-1)
        d = (delta[i0, j0] * (1 - ta) * (1 - tb) + delta[i0 + 1, j0] * ta * (1 - tb)
             + delta[i0, j0 + 1] * (1 - ta) * tb + delta[i0 + 1, j0 + 1] * ta * tb)
        out[:, ia] = a + d[:, 0]
        out[:, ib] = b + d[:, 1]
    return out


def lut_tables_torch(luts, device=None, dtype=None):
    import torch

    tabs = []
    for L in luts:
        tabs.append((float(L.a_grid[0]), float(L.a_grid[1] - L.a_grid[0]), len(L.a_grid),
                     float(L.b_grid[0]), float(L.b_grid[1] - L.b_grid[0]), len(L.b_grid),
                     torch.as_tensor(L.delta, device=device, dtype=dtype)))
    return tabs
