#!/usr/bin/env python3
"""Write the calibration-derived loaded-stand hold pose (item 241): ``data/calibration/dropbear_loaded_stand.json``.

The FSM MoveToDefault / Hold target defaults to this pose (``policy_runner --hold-pose loaded``); the calibration's
``standing_motor_pos`` stays available (``--hold-pose default``). Re-run after every calibration change; the
committed-JSON test fails when the calibration hash no longer matches.

    python tools/derive_loaded_stand.py [--knee-crank-deg 3.0] [--out data/calibration/dropbear_loaded_stand.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from dropbear_wbc.kinematics.loaded_stand import StandPlant, build_record  # noqa: E402

DEFAULT_CALIB = ROOT / "data/calibration/dropbear_semantic_calibration.json"
DEFAULT_XML = ROOT / "data/robot/dropbear_serial.xml"
DEFAULT_OUT = ROOT / "data/calibration/dropbear_loaded_stand.json"
DEFAULT_KNEE_CRANK_DEG = 3.0  # trained walkers stand at crank ~2-4 deg (Isaac, hw_v4, logs/loads/s9/stand_v4.npz)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--knee-crank-deg", type=float, default=DEFAULT_KNEE_CRANK_DEG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    calib = json.loads(args.calibration.read_text(encoding="utf-8"))
    rec = build_record(StandPlant(calib, args.xml), args.calibration, args.xml, math.radians(args.knee_crank_deg))
    args.out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    t, t0 = rec["static_motor_torque_Nm"], rec["reference_static_motor_torque_Nm"]
    print(f"wrote {args.out} (plant {rec['plant_mass_kg']} kg)")
    for n in ("LL_knee_actuator_joint", "RL_knee_actuator_joint", "LL_Revolute67", "LL_Revolute81", "LL_hip_joint"):
        print(f"  {n:24s} loaded {t[n]:8.2f}  reference {t0[n]:8.2f} N*m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
