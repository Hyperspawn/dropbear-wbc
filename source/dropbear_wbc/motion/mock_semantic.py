"""MOCK semantic<->motor map for developing the motion pipeline before the real calibration exists.

!!! NOT A CALIBRATION !!!  The maps are identity-ish linear guesses whose signs come from the USD joint
axes (logs/calibrate_settle/usd_joint_frames.json) and whose knee/elbow four-bar ratios are invented.
Outputs produced with it are labelled ``MOCK`` in every sidecar and must not be used for training.

It implements the same call surface as the contract API
``dropbear_wbc.kinematics.semantic.SemanticMap`` (CONTRACTS.md section 2):
``load(path)``, ``semantic_to_motor(q_sem[..., 22]) -> (q_motor[..., 22], info)`` and
``motor_to_semantic(q_motor[..., 22]) -> (q_sem[..., 22], info)``, vectorised, clipping to the valid
semantic range and reporting saturation in ``info``.

JSON layout read here (``tests/fixtures/mock_semantic_calibration.json``)::

    {"schema": "dropbear-semantic-calibration-v1", "MOCK": true, "semantic_names": [...22],
     "motor_names": [...22],
     "dofs": {<semantic>: {"map": "linear", "motors": [m], "scale": k, "offset": o, "range": [lo, hi]}
             | {"map": "linear2", "motors": [a, b], "partner": <semantic>, "matrix": [[..],[..]],
                "offset": [oa, ob], "range": [lo, hi]}}, ...}

``linear``:  semantic = scale * (motor - offset)  (identical to the real SemanticMap ``linear``).
``linear2``: [motor_a, motor_b] = offset + matrix @ [this, partner] (ankle pitch/roll pairs; the pitch
entry carries the matrix, the roll entry names the pitch entry as partner with ``"map": "pair"``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .names import MOTOR_NAMES, SEMANTIC_NAMES

__all__ = ["MockSemanticMap"]


@dataclass
class MockSemanticMap:
    semantic_names: tuple[str, ...]
    motor_names: tuple[str, ...]
    lo: np.ndarray  # (22,) semantic lower bound [rad]
    hi: np.ndarray  # (22,) semantic upper bound [rad]
    fwd: np.ndarray  # (22, 22) motor = offset + fwd @ semantic
    offset: np.ndarray  # (22,)
    is_mock: bool = True

    @classmethod
    def load(cls, path: Path | str) -> "MockSemanticMap":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if not d.get("MOCK", False):
            raise ValueError(f"{path} is not flagged MOCK; use dropbear_wbc.kinematics.semantic.SemanticMap")
        sem = tuple(d["semantic_names"])
        mot = tuple(d["motor_names"])
        if sem != SEMANTIC_NAMES or mot != MOTOR_NAMES:
            raise ValueError("mock calibration names do not match the contract")
        s_idx = {n: i for i, n in enumerate(sem)}
        m_idx = {n: i for i, n in enumerate(mot)}
        fwd = np.zeros((22, 22))
        off = np.zeros(22)
        lo = np.zeros(22)
        hi = np.zeros(22)
        for name, spec in d["dofs"].items():
            i = s_idx[name]
            lo[i], hi[i] = spec["range"]
            kind = spec["map"]
            if kind == "linear":
                # same convention as the real SemanticMap: q_sem = scale * (m - offset)
                (m,) = spec["motors"]
                fwd[m_idx[m], i] = 1.0 / spec["scale"]
                off[m_idx[m]] = spec["offset"]
            elif kind == "linear2":
                a, b = (m_idx[m] for m in spec["motors"])
                j = s_idx[spec["partner"]]
                mat = np.asarray(spec["matrix"], dtype=np.float64)
                fwd[a, i], fwd[a, j] = mat[0]
                fwd[b, i], fwd[b, j] = mat[1]
                off[a], off[b] = spec["offset"]
            elif kind == "pair":
                pass  # coefficients stored on the partner entry
            else:
                raise ValueError(f"unknown map {kind!r} for {name}")
        if abs(np.linalg.det(fwd)) < 1e-9:
            raise ValueError("mock map is singular")
        return cls(sem, mot, lo, hi, fwd, off)

    def semantic_to_motor(self, q_sem: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        q = np.asarray(q_sem, dtype=np.float64)
        if q.shape[-1] != 22:
            raise ValueError(f"expected (..., 22), got {q.shape}")
        qc = np.clip(q, self.lo, self.hi)
        motor = self.offset + np.einsum("ij,...j->...i", self.fwd, qc)
        return motor, {"saturated": qc != q, "excess": q - qc, "q_sem_clipped": qc}

    def motor_to_semantic(self, q_motor: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        m = np.asarray(q_motor, dtype=np.float64)
        q = np.einsum("ij,...j->...i", np.linalg.inv(self.fwd), m - self.offset)
        qc = np.clip(q, self.lo, self.hi)
        return qc, {"saturated": qc != q, "excess": q - qc}
