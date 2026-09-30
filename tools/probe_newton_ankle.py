"""Newton check of the CONTRACTS 0.2 ankle retype: build the plant with the spherical (default) and the authored
(revolute) ankle tie rods and compare the MuJoCo equality constraints and a single-calf-motor step.

In PhysX (Isaac) moving ``*_Revolute81`` alone binds after ~50 % of a 0.3 rad step on the authored plant
(``logs/calibrate_settle/step_response/``). This probe runs the same kind of step in Newton/MuJoCo (fixed base,
hanging, gravity on) for both variants so the two simulators can be compared.

CPU only (MuJoCo C), no GPU lock needed::

    CUDA_VISIBLE_DEVICES=-1 .venv-newton/Scripts/python.exe tools/probe_newton_ankle.py \
        --report logs/gpu_pipeline/newton_ankle_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

import numpy as np  # noqa: E402

from dropbear_wbc.sdk import motors  # noqa: E402


def run_variant(authored: bool, args) -> dict:
    from dropbear_wbc.newton_sim.plant import DropbearNewtonPlant, PlantConfig

    t0 = time.perf_counter()
    plant = DropbearNewtonPlant(PlantConfig(fixed_base=True, device="cpu", mujoco_cpu=True, use_cuda_graph=False,
                                            authored_ankle_tierods=authored), log=lambda m: print(m, flush=True))
    out: dict = {"authored_ankle_tierods": authored, "build_s": time.perf_counter() - t0,
                 "spherical_retyped": plant.report["usd_fixes_contract_0_1"]["spherical"]}
    m = plant.solver.mj_model
    import mujoco

    eq_types = [int(t) for t in m.eq_type]
    out["mujoco_neq"] = int(m.neq)
    out["mujoco_eq_connect"] = sum(t == int(mujoco.mjtEq.mjEQ_CONNECT) for t in eq_types)
    out["mujoco_eq_weld"] = sum(t == int(mujoco.mjtEq.mjEQ_WELD) for t in eq_types)
    labels = [plant.joint_labels[j].rsplit("/", 1)[-1] for j in plant.closure_joints]
    r0 = plant.initial_readout()
    q0 = r0.motor_q.copy()
    kp = np.asarray(motors.DEFAULT_KP) * args.kp_scale
    kd = np.asarray(motors.DEFAULT_KD) * args.kp_scale
    i81 = motors.MOTOR_NAMES.index(args.motor)
    ticks_hold, ticks_ramp, ticks_after = int(0.5 / plant.cfg.tick_dt), int(0.5 / plant.cfg.tick_dt), int(1.0 / plant.cfg.tick_dt)
    q_des = q0.copy()
    trace = []
    for k in range(ticks_hold + ticks_ramp + ticks_after):
        if k >= ticks_hold:
            ratio = min(1.0, (k - ticks_hold) / ticks_ramp)
            q_des = q0.copy()
            q_des[i81] = q0[i81] + ratio * args.step_rad
        plant.set_motor_command(q_des, np.zeros(22), np.zeros(22), kp, kd, np.ones(22, np.int32))
        plant.step()
        if k % 50 == 0 or k == ticks_hold + ticks_ramp + ticks_after - 1:
            r = plant.readout()
            res = plant.closure_residuals_m()
            trace.append({"t": round(r.time_s, 3), "q_motor": float(r.motor_q[i81] - q0[i81]),
                          "max_closure_m": float(res.max()), "worst": labels[int(res.argmax())]})
    r = plant.readout()
    res = plant.closure_residuals_m()
    out["step"] = {"motor": args.motor, "step_rad": args.step_rad, "kp_scale": args.kp_scale,
                   "reached_rad": float(r.motor_q[i81] - q0[i81]),
                   "reached_fraction": float((r.motor_q[i81] - q0[i81]) / args.step_rad),
                   "final_tau": float(r.motor_tau[i81]),
                   "closure_residual_m": {n: float(v) for n, v in zip(labels, res)},
                   "max_closure_m": float(res.max()), "finite": plant.is_finite()}
    out["trace"] = trace
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--motor", default="LL_Revolute81")
    ap.add_argument("--step-rad", type=float, default=0.3)
    ap.add_argument("--kp-scale", type=float, default=1.0, help="multiplier on the legacy motor kp/kd")
    ap.add_argument("--variants", default="spherical,authored")
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("CPU-only probe: set CUDA_VISIBLE_DEVICES=-1 (no GPU lock is taken)")
    report = {"args": {k: str(v) for k, v in vars(args).items()}, "variants": {}}
    for v in args.variants.split(","):
        report["variants"][v] = run_variant(v == "authored", args)
        s = report["variants"][v]["step"]
        print(json.dumps({"variant": v, "neq": report["variants"][v]["mujoco_neq"],
                          "reached_fraction": round(s["reached_fraction"], 3),
                          "max_closure_mm": round(1e3 * s["max_closure_m"], 3)}), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(f"[probe_newton_ankle] wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
