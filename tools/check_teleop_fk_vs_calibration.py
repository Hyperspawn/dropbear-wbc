"""Compare the teleop IK's serial arm model (``teleop.arm_ik``) with the calibration's own measured forward model
(``kinematics.serial_model.CalibrationFK``: raw Isaac sweep, interpolated tables for the elbow four-bar) over random
arm configurations inside the IK limits. The difference is the IK model error w.r.t. the Isaac plant (best-fit elbow
pivot, idealised axes); the Newton plant adds its own offset (docs/TELEOP.md 5.3).

    .venv-teleop/Scripts/python.exe tools/check_teleop_fk_vs_calibration.py --out logs/teleop/fk_vs_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source")]

import numpy as np  # noqa: E402

from dropbear_wbc.kinematics.serial_model import CalibrationFK  # noqa: E402
from dropbear_wbc.teleop.arm_ik import ARM_SEMANTIC_INDEX, SIDES, DropbearArmIK  # noqa: E402

HAND = {"left": "LH_shoulder_ex_al_interface_1", "right": "RH_shoulder_ex_al_interface_1"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "teleop" / "fk_vs_calibration.json")
    a = ap.parse_args()
    ik = DropbearArmIK()
    cfk = CalibrationFK(ik.path)
    rng = np.random.default_rng(0)
    out = {"calibration": str(ik.path), "calibration_sha256": ik.sha256, "n": a.n, "sides": {}}
    for side in SIDES:
        ch = ik.chains[side]
        qs = rng.uniform(ch.lower, ch.upper, size=(a.n, 5))
        q22 = np.tile(ik.standing_semantic, (a.n, 1))
        q22[:, list(ARM_SEMANTIC_INDEX[side])] = qs
        segs, used, _ = cfk.segments(q22)
        p_cal = segs[HAND[side]][1]
        used_arm = used[:, list(ARM_SEMANTIC_INDEX[side])]
        p_mod, _ = ch.fk_batch(used_arm)          # compare at the reachable (clipped) pose
        e = np.linalg.norm(p_mod - p_cal, axis=-1) * 1e3
        el = used_arm[:, 3]
        bins = np.linspace(ch.lower[3], ch.upper[3], 6)
        by_elbow = {f"{bins[i]:+.2f}..{bins[i+1]:+.2f}": float(np.median(e[(el >= bins[i]) & (el <= bins[i + 1])]))
                    for i in range(5) if ((el >= bins[i]) & (el <= bins[i + 1])).any()}
        out["sides"][side] = {"wrist_err_mm": {"p50": float(np.percentile(e, 50)), "p95": float(np.percentile(e, 95)),
                                               "max": float(e.max()), "rms": float(np.sqrt(np.mean(e ** 2)))},
                              "median_by_elbow_rad": by_elbow}
        print(side, json.dumps(out["sides"][side]), flush=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
