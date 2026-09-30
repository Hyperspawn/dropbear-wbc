"""Probe the Newton Dropbear plant: build time, per-tick cost and a PD hold without ZMQ.

Holds the 22 motors with the legacy DROPBEAR_CFG gains for ``--ticks`` control
ticks, either at their initial (USD-authored) positions (``--target initial``) or
ramping linearly to the legacy default pose over ``--ramp-s`` and holding it
(``--target default``). Reports timing, per-motor tracking error, the model's
motor joint limits, root height and closure residuals. Acquires the GPU lock,
except with ``--no-gpu-lock``, which is only accepted for a pure-CPU run
(``--device cpu --mujoco-cpu`` with ``CUDA_VISIBLE_DEVICES=-1``).

Example::

    .venv-newton/Scripts/python.exe tools/probe_newton_plant.py --fixed-base --ticks 1000 \
        --report logs/sdk_bridge/probe_gpu_fixed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "third_party" / "pydeps")]

import numpy as np  # noqa: E402

from dropbear_wbc.newton_sim.gpu_lock import GpuLock  # noqa: E402
from dropbear_wbc.sdk import motors  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixed-base", action="store_true")
    ap.add_argument("--ticks", type=int, default=500)
    ap.add_argument("--sim-dt", type=float, default=0.002)
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mujoco-cpu", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--ls-iterations", type=int, default=50)
    ap.add_argument("--collisions", default="convex-hull", choices=["convex-hull", "box"])
    ap.add_argument("--target", choices=["initial", "default"], default="initial")
    ap.add_argument("--ramp-s", type=float, default=2.0)
    ap.add_argument("--passive-damping", type=float, default=motors.PASSIVE_DAMPING)
    ap.add_argument("--no-gpu-lock", action="store_true")
    ap.add_argument("--disable-contacts", action="store_true", help="diagnostic: no MuJoCo contacts at all")
    ap.add_argument("--raw-usd", action="store_true", help="skip the CONTRACTS 0.1 in-memory plant fixes")
    ap.add_argument("--save-mjcf", type=Path, default=None)
    ap.add_argument("--diag-axis", action="append", default=[], metavar="JOINT=AXIS",
                    help="DIAGNOSTIC in-memory revolute axis override (repeatable); never plant authority")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    if args.no_gpu_lock and not (args.device == "cpu" and args.mujoco_cpu
                                 and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"):
        ap.error("--no-gpu-lock requires --device cpu --mujoco-cpu and CUDA_VISIBLE_DEVICES=-1")

    report: dict = {"args": {k: str(v) for k, v in vars(args).items()}, "status": "failed"}
    lock = GpuLock(owner=f"sdk_bridge/probe_newton_plant {args.report.name}")
    try:
        if not args.no_gpu_lock:
            lock.acquire()
            report["gpu_lock_wait_s"] = lock.waited_s
        from dropbear_wbc.newton_sim.plant import DropbearNewtonPlant, PlantConfig

        t0 = time.perf_counter()
        plant = DropbearNewtonPlant(PlantConfig(
            fixed_base=args.fixed_base, sim_dt=args.sim_dt, substeps=args.substeps, device=args.device,
            mujoco_cpu=args.mujoco_cpu, use_cuda_graph=not args.no_graph, iterations=args.iterations,
            ls_iterations=args.ls_iterations, collisions=args.collisions, passive_damping=args.passive_damping,
            disable_contacts=args.disable_contacts, save_mjcf=args.save_mjcf,
            diag_axis_overrides=tuple(tuple(x.split("=", 1)) for x in args.diag_axis), usd_fixes=not args.raw_usd))
        report["build_s"] = time.perf_counter() - t0
        report["plant"] = plant.report
        r0 = plant.initial_readout()
        q_init = r0.motor_q.copy()
        q_goal = np.asarray(motors.DEFAULT_POS) if args.target == "default" else q_init
        report["initial_motor_q"] = q_init.tolist()
        report["initial_root_pos_w"] = r0.root_pos_w.tolist()
        lo = plant.model.joint_limit_lower.numpy()[plant.motor_qd_idx]
        hi = plant.model.joint_limit_upper.numpy()[plant.motor_qd_idx]
        report["motor_limits_rad"] = {n: [float(a), float(b)] for n, a, b in zip(motors.MOTOR_NAMES, lo, hi)}
        kp, kd = np.asarray(motors.DEFAULT_KP), np.asarray(motors.DEFAULT_KD)
        step_t, read_t, samples = [], [], []
        q_hold = q_init
        for k in range(args.ticks):
            ratio = min(1.0, k * plant.cfg.tick_dt / max(args.ramp_s, 1e-6))
            q_hold = q_init + ratio * (q_goal - q_init)
            plant.set_motor_command(q_hold, np.zeros(22), np.zeros(22), kp, kd, np.ones(22, np.int32))
            a = time.perf_counter()
            plant.step()
            b = time.perf_counter()
            r = plant.readout()
            c = time.perf_counter()
            step_t.append(b - a)
            read_t.append(c - b)
            if not np.isfinite(r.motor_q).all():
                report["nonfinite_tick"] = k
                break
            if k % 50 == 0 or k == args.ticks - 1:
                samples.append({"tick": r.tick, "t": r.time_s, "root_z": float(r.root_pos_w[2]),
                                "root_quat_wxyz": r.root_quat_wxyz.tolist(),
                                "max_abs_track_err_rad": float(np.abs(r.motor_q - q_hold).max()),
                                "max_abs_tau": float(np.abs(r.motor_tau).max()),
                                "worst_motor": motors.MOTOR_NAMES[int(np.abs(r.motor_q - q_hold).argmax())],
                                "max_closure_residual_m": float(plant.closure_residuals_m().max())})
        report["final_per_motor"] = {n: {"q": float(q), "q_des": float(qd), "err": float(q - qd), "tau": float(t)}
                                     for n, q, qd, t in zip(motors.MOTOR_NAMES, r.motor_q, q_hold, r.motor_tau)}
        st, rt = np.asarray(step_t[5:]), np.asarray(read_t[5:])
        tick_dt = args.sim_dt * args.substeps
        report["timing"] = {
            "tick_dt_s": tick_dt, "step_ms_mean": 1e3 * st.mean(), "step_ms_p50": 1e3 * np.median(st),
            "step_ms_p99": 1e3 * np.percentile(st, 99), "readout_ms_mean": 1e3 * rt.mean(),
            "achievable_tick_hz": 1.0 / (st.mean() + rt.mean()),
            "realtime_factor": tick_dt / (st.mean() + rt.mean())}
        report["samples"] = samples
        solver = plant.solver
        if args.mujoco_cpu and getattr(solver, "mj_data", None) is not None:
            import mujoco
            m, dat = solver.mj_model, solver.mj_data
            report["mujoco_ncon_final"] = int(dat.ncon)
            pairs: dict[str, float] = {}
            for i in range(dat.ncon):
                c = dat.contact[i]
                b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom1]))
                b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom2]))
                key = " | ".join(sorted([str(b1), str(b2)]))
                pairs[key] = min(pairs.get(key, 1e9), float(c.dist))
            report["mujoco_contact_body_pairs_min_dist_m"] = pairs
        report["status"] = "executed"
    except Exception:  # noqa: BLE001
        import traceback
        report["error"] = traceback.format_exc()
    finally:
        lock.release()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({"status": report["status"], "timing": report.get("timing"),
                          "error": report.get("error")}, indent=2))
    return 0 if report["status"] == "executed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
