"""Numpy loop-closure residuals of the Dropbear articulation (27 excluded joints).

Given link-frame body poses (any common frame), each excluded joint has two anchors
``a_k = p_k + R_k @ localPos_k`` (k = 0, 1) that must coincide, and joint frames ``F_k = R_k @ R(localRot_k)``
whose relative rotation must respect the joint type:

* fixed      -- angular residual = geodesic angle of ``F0^T F1``;
* revolute   -- the free axis ``e`` (joint-frame X/Y/Z) must stay aligned: residual = angle(F0 e, F1 e);
* spherical  -- no angular constraint (residual 0).

Units: metres (gaps) and radians (angles). Quaternions wxyz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dropbear_wbc.motion.rotations import quat_to_matrix

_AXIS = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}


@dataclass
class ClosureTable:
    """Arrays describing the excluded joints (as saved by ``QuasiStaticDropbear.closure_table``)."""

    names: np.ndarray
    types: np.ndarray
    axes: np.ndarray
    body0: np.ndarray
    body1: np.ndarray
    local_pos0: np.ndarray
    local_pos1: np.ndarray
    local_rot0: np.ndarray
    local_rot1: np.ndarray

    @classmethod
    def from_npz(cls, data) -> "ClosureTable":
        return cls(
            names=np.asarray(data["closure_names"]).astype(str), types=np.asarray(data["closure_types"]).astype(str),
            axes=np.asarray(data["closure_axes"]).astype(str), body0=np.asarray(data["closure_body0"]),
            body1=np.asarray(data["closure_body1"]), local_pos0=np.asarray(data["closure_local_pos0"]),
            local_pos1=np.asarray(data["closure_local_pos1"]), local_rot0=np.asarray(data["closure_local_rot0"]),
            local_rot1=np.asarray(data["closure_local_rot1"]),
        )

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "closure_names": self.names, "closure_types": self.types, "closure_axes": self.axes,
            "closure_body0": self.body0, "closure_body1": self.body1, "closure_local_pos0": self.local_pos0,
            "closure_local_pos1": self.local_pos1, "closure_local_rot0": self.local_rot0,
            "closure_local_rot1": self.local_rot1,
        }


def closure_residuals(body_pos: np.ndarray, body_quat: np.ndarray, table: ClosureTable) -> tuple[np.ndarray, np.ndarray]:
    """Anchor gaps [m] and angular residuals [rad], each of shape (..., C).

    Args:
        body_pos: (..., B, 3) link positions.
        body_quat: (..., B, 4) link orientations (wxyz), same frame as ``body_pos``.
        table: closure description.
    """
    r = quat_to_matrix(body_quat)  # (..., B, 3, 3)
    r0, r1 = r[..., table.body0, :, :], r[..., table.body1, :, :]
    p0, p1 = body_pos[..., table.body0, :], body_pos[..., table.body1, :]
    a0 = p0 + np.einsum("...ij,...j->...i", r0, table.local_pos0)
    a1 = p1 + np.einsum("...ij,...j->...i", r1, table.local_pos1)
    gaps = np.linalg.norm(a0 - a1, axis=-1)
    f0 = r0 @ quat_to_matrix(table.local_rot0)
    f1 = r1 @ quat_to_matrix(table.local_rot1)
    ang = np.zeros_like(gaps)
    for c, (jtype, axis) in enumerate(zip(table.types, table.axes)):
        if "Fixed" in jtype:
            rel = np.swapaxes(f0[..., c, :, :], -1, -2) @ f1[..., c, :, :]
            tr = np.trace(rel, axis1=-2, axis2=-1)
            ang[..., c] = np.arccos(np.clip(0.5 * (tr - 1.0), -1.0, 1.0))
        elif "Revolute" in jtype and axis in _AXIS:
            e0 = f0[..., c, :, :] @ _AXIS[axis]
            e1 = f1[..., c, :, :] @ _AXIS[axis]
            ang[..., c] = np.arccos(np.clip(np.sum(e0 * e1, axis=-1), -1.0, 1.0))
    return gaps, ang
