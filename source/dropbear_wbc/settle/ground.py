"""Foot sole geometry and the ground (root z) correction of the settle tool.

The sole points are the convex-hull vertices of each foot body's collision meshes, in the body link
frame (``data/calibration/dropbear_foot_sole_hulls.json``, written by ``tools/extract_foot_soles.py``).
The lowest point of a foot in a pose ``(p, R)`` is ``min_v (R v + p)_z`` -- exact for the (per-mesh
convex hull) collision geometry.

Ground correction (per frame ``t``):

1. ``h_side[t]`` = lowest sole point of each foot (both bodies of a foot are considered: sole plate and
   ankle cross).
2. Contact feet: the sidecar ``contact_hint`` when present; otherwise the lower foot only.
3. Raw correction ``dz_raw[t] = -min_{contact feet} h_side[t]`` (NaN when the hint says no foot is in
   contact; filled by linear interpolation, edges held).
4. ``dz[t]`` = zero-phase Gaussian smoothing of ``dz_raw`` (``sigma_s`` seconds; 0 disables).
5. Every body z and the root z are shifted by ``dz[t]``.

Units: metres, seconds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dropbear_wbc.motion.rotations import quat_to_matrix

FOOT_GROUPS: dict[str, tuple[str, ...]] = {
    "left": ("LL_skateboard_bearing_left_2", "LL_basis_left_1"),
    "right": ("RL_skateboard_bearing_left_2", "RL_basis_left_1"),
}


@dataclass
class SoleModel:
    """Sole hull vertices per foot body (link frame)."""

    vertices: dict[str, np.ndarray]
    usd_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "SoleModel":
        d = json.loads(Path(path).read_text())
        return cls({k: np.asarray(v["vertices_b"], dtype=np.float64) for k, v in d["feet"].items()},
                   d.get("usd_sha256", ""))

    def lowest_z(self, body_pos: np.ndarray, body_quat: np.ndarray, body_names: list[str]) -> dict[str, np.ndarray]:
        """Lowest sole point z [m] per foot side, each shape (T,).

        Args:
            body_pos: (T, B, 3) world positions of link frames.
            body_quat: (T, B, 4) world orientations (wxyz).
            body_names: articulation body names (length B).
        """
        out: dict[str, np.ndarray] = {}
        for side, bodies in FOOT_GROUPS.items():
            zs = []
            for b in bodies:
                i = body_names.index(b)
                r = quat_to_matrix(body_quat[:, i])  # (T, 3, 3)
                z = np.einsum("tj,vj->tv", r[:, 2, :], self.vertices[b]) + body_pos[:, i, 2:3]
                zs.append(z.min(axis=1))
            out[side] = np.min(np.stack(zs, axis=0), axis=0)
        return out


def gaussian_smooth(x: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing along axis 0 with edge padding (``sigma_frames`` <= 0: identity)."""
    if sigma_frames <= 0 or x.shape[0] < 3:
        return x.copy()
    half = int(np.ceil(3.0 * sigma_frames))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_frames) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], half, axis=0), x, np.repeat(x[-1:], half, axis=0)], axis=0)
    return np.convolve(pad, k, mode="valid")


@dataclass
class GroundFix:
    """Result of :func:`ground_correction`."""

    dz: np.ndarray
    """(T,) applied z shift [m]."""
    dz_raw: np.ndarray
    """(T,) unsmoothed correction [m] (NaN-filled)."""
    contact: np.ndarray
    """(T, 2) contact feet used (left, right)."""
    contact_source: str
    """'sidecar' or 'lowest_foot'."""
    sole_z_after: dict[str, np.ndarray]
    """Lowest sole point per side after the shift [m]."""
    contact_sole_z_after: np.ndarray
    """(T,) lowest sole point among contact feet after the shift [m] (0 = exact contact)."""


def ground_correction(lowest: dict[str, np.ndarray], fps: float, contact_hint: np.ndarray | None = None,
                      sigma_s: float = 0.1) -> GroundFix:
    """Compute the per-frame z shift that puts the contact foot sole on z = 0 (see module doc)."""
    hl, hr = lowest["left"], lowest["right"]
    t = hl.shape[0]
    if contact_hint is not None:
        contact = np.asarray(contact_hint, dtype=bool).copy()
        source = "sidecar"
    else:
        contact = np.stack([hl <= hr, hr < hl], axis=-1)
        source = "lowest_foot"
    h = np.stack([hl, hr], axis=-1)
    masked = np.where(contact, h, np.inf)
    lowest_contact = masked.min(axis=-1)
    dz_raw = np.where(np.isfinite(lowest_contact), -lowest_contact, np.nan)
    if np.all(np.isnan(dz_raw)):
        dz_raw = -np.minimum(hl, hr)  # no contact anywhere: fall back to the lowest foot
    idx = np.arange(t)
    good = ~np.isnan(dz_raw)
    dz_filled = np.interp(idx, idx[good], dz_raw[good])
    dz = gaussian_smooth(dz_filled, sigma_s * fps)
    after = {"left": hl + dz, "right": hr + dz}
    h_after = np.stack([after["left"], after["right"]], axis=-1)
    contact_after = np.where(contact, h_after, np.inf).min(axis=-1)
    contact_after = np.where(np.isfinite(contact_after), contact_after, np.nan)
    return GroundFix(dz=dz, dz_raw=dz_filled, contact=contact, contact_source=source, sole_z_after=after,
                     contact_sole_z_after=contact_after)
