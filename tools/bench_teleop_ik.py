"""Benchmark the Dropbear arm IK (``dropbear_wbc.teleop.arm_ik``) on FK-generated targets (CPU only).

For each arm: random semantic configurations inside the IK limits -> FK pose (reachable, orientation-consistent)
-> IK from (a) a warm start 0.1 rad away (the teleop case) and (b) the standing pose (cold start, multi-start
seeds). Both the default priority solver and xr_teleoperate's single weighted cost are measured. Also: control-loop
style tracking of a figure-8 (warm-started, 5 iterations per step) and the per-call costs of the motor mapping and
gravity feed-forward.

    .venv-teleop/Scripts/python.exe tools/bench_teleop_ik.py --out logs/teleop/ik_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source")]

import numpy as np  # noqa: E402

from dropbear_wbc.teleop.arm_ik import (  # noqa: E402
    MIRROR_SIGN, SIDES, DropbearArmIK, IKWeights, pose, solve_chain_robust,
)
from dropbear_wbc.teleop.devices import ScriptedSource  # noqa: E402


def pct(v, q):
    return float(np.percentile(np.asarray(v), q))


def stats_mm(e):
    e = np.asarray(e) * 1e3
    return {"p50_mm": pct(e, 50), "p95_mm": pct(e, 95), "max_mm": float(e.max()), "frac_gt_5mm": float((e > 5).mean()),
            "n": int(len(e))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "teleop" / "ik_benchmark.json")
    a = ap.parse_args()
    ik = DropbearArmIK()
    rng = np.random.default_rng(a.seed)
    out: dict = {"calibration": str(ik.path), "calibration_sha256": ik.sha256, "n": a.n, "seed": a.seed,
                 "limits": ik.info()["limits"], "results": {}}
    for mode in ("priority", "weighted"):
        w = IKWeights(mode=mode)
        res = {}
        for side in SIDES:
            ch = ik.chains[side]
            qs = rng.uniform(ch.lower, ch.upper, size=(a.n, 5))
            warm, warm_rot, cold, cold_rot, t_warm, t_cold = [], [], [], [], [], []
            for q in qs:
                t = pose(*ch.fk(q))
                q0 = ch.clip(q + rng.normal(0.0, 0.1, 5))
                t0 = time.perf_counter()
                r = solve_chain_robust(ch, t, q0, q0, weights=w)
                t_warm.append(time.perf_counter() - t0)
                warm.append(r.pos_err_m)
                warm_rot.append(r.rot_err_rad)
                t0 = time.perf_counter()
                r = solve_chain_robust(ch, t, ch.q_rest, weights=w)
                t_cold.append(time.perf_counter() - t0)
                cold.append(r.pos_err_m)
                cold_rot.append(r.rot_err_rad)
            res[side] = {"warm_start_pos": stats_mm(warm), "warm_start_rot_p50_rad": pct(warm_rot, 50),
                         "cold_start_pos": stats_mm(cold), "cold_start_rot_p50_rad": pct(cold_rot, 50),
                         "warm_ms": {"p50": 1e3 * pct(t_warm, 50), "p95": 1e3 * pct(t_warm, 95)},
                         "cold_ms": {"p50": 1e3 * pct(t_cold, 50), "p95": 1e3 * pct(t_cold, 95)}}
        out["results"][mode] = res
        print(mode, json.dumps(res, indent=1), flush=True)
    # control-loop style tracking of the default scripted figure-8 (both arms, 50 Hz, 5 iterations per step)
    q_ref = np.array([-0.7, 0.15, 0.0, 0.0, 0.0])
    ref = {"left": ik.fk("left", q_ref), "right": ik.fk("right", q_ref * MIRROR_SIGN)}
    src = ScriptedSource(ref, kind="figure8", amplitude=(0.02, 0.06, 0.05), period=6.0, ramp_s=0.0)
    src.start()
    ik.reset(np.concatenate([q_ref, q_ref * MIRROR_SIGN]))
    err, ts, dq, last = [], [], [], None
    for k in range(600):
        wt = src.get(k * 0.02)
        t0 = time.perf_counter()
        r = ik.solve(wt.left, wt.right, max_iters=5, restarts="light", restart_above_m=0.01)
        ts.append(time.perf_counter() - t0)
        err.append(max(x.pos_err_m for x in r.arms.values()))
        if last is not None:
            dq.append(float(np.abs(r.q_sem - last).max()))
        last = r.q_sem
    m10, _ = ik.to_motor_fast(last)
    t0 = time.perf_counter()
    for _ in range(200):
        ik.to_motor(last)
    t_full = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(200):
        ik.to_motor_fast(last)
    t_fast = (time.perf_counter() - t0) / 200
    from dropbear_wbc.teleop.gravity import ArmGravity
    grav = ArmGravity(ik)
    t0 = time.perf_counter()
    for _ in range(200):
        grav.motor_torque(last, m10)
    t_grav = (time.perf_counter() - t0) / 200
    out["figure8_loop"] = {"steps": 600, "rate_hz": 50, "iters_per_step": 5, "pos_err": stats_mm(err),
                           "max_step_dq_rad": max(dq), "solve_ms_both_arms": {"p50": 1e3 * pct(ts, 50),
                                                                              "p95": 1e3 * pct(ts, 95),
                                                                              "max": 1e3 * max(ts)}}
    out["mapping_ms"] = {"semantic_to_motor_full": 1e3 * t_full, "to_motor_fast": 1e3 * t_fast,
                         "gravity_ff": 1e3 * t_grav}
    print(json.dumps({k: out[k] for k in ("figure8_loop", "mapping_ms")}, indent=1), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
