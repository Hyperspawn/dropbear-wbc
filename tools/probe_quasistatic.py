"""Probe: cost and accuracy of the quasi-static Dropbear articulation vs solver iterations / env count.

Parent mode (default, plain python, the caller holds the GPU lock via ``tools/gpu_lock_run.py``): runs one
child Isaac process per ``--configs`` entry (``<pos_iters>x<num_envs>``) sequentially and writes a summary.
Child mode (``--child``, run with ``C:/isaac-sim/python.bat``) for one configuration:

1. rest hold (all motors 0) for 60 steps: worst closure gap, root drift, ms/step;
2. a COARSE version of the calibration sweep (every motor over its full range, closed-loop motors
   forward + back, 3 ankle rows per side) with the fixed ``--ramp``/``--hold`` stage lengths; per program
   the worst / 95th-percentile closure gap, joint position change over the last 4 hold steps, motor
   tracking error.

Example (from ``<repo>``)::

    python tools/gpu_lock_run.py --owner calibrate_settle --log logs/calibrate_settle/probe_iters.log \
        --timeout 1200 -- python tools/probe_quasistatic.py --configs 8x64 16x64 32x64 16x128
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--configs", nargs="+", default=["8x64", "16x64", "32x64"])
ap.add_argument("--child", action="store_true")
ap.add_argument("--pos-iters", type=int, default=16)
ap.add_argument("--num-envs", type=int, default=64)
ap.add_argument("--ramp", type=int, default=4)
ap.add_argument("--hold", type=int, default=6)
ap.add_argument("--motor-kp", type=float, default=3000.0)
ap.add_argument("--motor-kd", type=float, default=60.0)
ap.add_argument("--passive-damping", type=float, default=0.5)
ap.add_argument("--serial-step-deg", type=float, default=6.0)
ap.add_argument("--loop-step-deg", type=float, default=1.5)
ap.add_argument("--no-fixes", action="store_true", help="DIAGNOSTIC: authored plant without contract 0.1 fixes")
ap.add_argument("--tag", default="")
ap.add_argument("--out-dir", type=Path, default=REPO / "logs/calibrate_settle/probe")
ARGS, KIT = ap.parse_known_args()


def coarse_programs(motor_limits):
    import numpy as np

    from dropbear_wbc.kinematics.sweeps import grid2d_row, sweep1d
    from dropbear_wbc.robots.dropbear_names import CLOSURE_MOTORS, MOTOR_NAMES

    deg = np.pi / 180
    progs = []
    for i, name in enumerate(MOTOR_NAMES):
        lo, hi = motor_limits[i, 0] + 0.25 * deg, motor_limits[i, 1] - 0.25 * deg
        loop = name in CLOSURE_MOTORS
        st = (ARGS.loop_step_deg if loop else ARGS.serial_step_deg) * deg
        progs.append(sweep1d(i, lo, hi, st, max_step=min(st, 2.0 * deg) if loop else st, return_pass=loop,
                            name=f"sweep1d:{name}"))
    for side in ("LL", "RL"):
        ia, ib = MOTOR_NAMES.index(f"{side}_Revolute67"), MOTOR_NAMES.index(f"{side}_Revolute81")
        bv = np.linspace(motor_limits[ib, 0] + 0.25 * deg, motor_limits[ib, 1] - 0.25 * deg, 7)
        for a in np.linspace(motor_limits[ia, 0] + 0.25 * deg, motor_limits[ia, 1] - 0.25 * deg, 3):
            progs.append(grid2d_row(ia, ib, float(a), bv, max_step=2.0 * deg, name=f"grid2d:{side}:{a:.6f}"))
    return progs


def child() -> int:
    from dropbear_wbc.isaac.launch import close_app_and_exit, prepare_kit_python

    prepare_kit_python()
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args(KIT)
    a.headless = True
    app = AppLauncher(a).app
    tag = ARGS.tag or f"it{ARGS.pos_iters}_n{ARGS.num_envs}_r{ARGS.ramp}h{ARGS.hold}"
    out = ARGS.out_dir / f"probe_{tag}.json"
    report: dict = {"tag": tag, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(ARGS).items()}}
    rc = 1
    try:
        import numpy as np
        import torch

        from dropbear_wbc.isaac.quasistatic import QuasiStaticCfg, QuasiStaticDropbear
        from dropbear_wbc.isaac.sweep_runner import run_programs

        cfg = QuasiStaticCfg(num_envs=ARGS.num_envs, pos_iters=ARGS.pos_iters, motor_kp=ARGS.motor_kp,
                             motor_kd=ARGS.motor_kd, passive_damping=ARGS.passive_damping,
                             contract_fixes=not ARGS.no_fixes)
        t = time.time()
        qs = QuasiStaticDropbear(cfg)
        report["setup_s"] = time.time() - t
        report["stage_fixes"] = qs.stage_fixes
        report["config"] = cfg.to_dict()
        print(f"[probe] {tag} ready in {report['setup_s']:.1f}s fixes={qs.stage_fixes}", flush=True)
        # 1. rest hold
        zero = torch.zeros(qs.num_envs, len(qs.joint_names), device=qs.device)
        qs.write_joint_state(zero)
        root0 = qs.robot.data.root_link_pos_w.clone()
        rest = []
        torch.cuda.synchronize()
        t = time.time()
        for k in range(6):
            qs.step(10)
            rest.append({"step": 10 * (k + 1), "worst_gap_mm": 1e3 * float(qs.worst_gap().max()),
                         "root_drift_mm": 1e3 * float((qs.robot.data.root_link_pos_w - root0).norm(dim=-1).max()),
                         "root_offset_from_env_origin_mm": 1e3 * float(
                             (qs.robot.data.root_link_pos_w - qs.env_origins).norm(dim=-1).max()),
                         "max_abs_joint_vel": float(qs.robot.data.joint_vel.abs().max())})
        torch.cuda.synchronize()
        report["rest_ms_per_step"] = 1e3 * (time.time() - t) / 60
        report["rest"] = rest
        print(f"[probe] rest ms/step={report['rest_ms_per_step']:.2f} last={rest[-1]}", flush=True)
        # 2. coarse sweep
        progs = coarse_programs(qs.motor_limits)
        rec = run_programs(qs, progs, ARGS.ramp, ARGS.hold, window=4, tag=f"probe:{tag}")
        report["sweep_ms_per_step"] = float(rec["ms_per_step"])
        report["sweep_wall_s"] = float(rec["wall_s"])
        report["sweep_physics_steps"] = int(rec["physics_steps"])
        report["root_drift_after_sweep_mm"] = 1e3 * float((qs.robot.data.root_link_pos_w - root0).norm(dim=-1).max())
        cl_names = [c.name for c in qs.closures]
        per = {}
        for pid, pr in enumerate(progs):
            sel = rec["program"] == pid
            if not sel.any():
                continue
            g, dq, me = rec["gap"][sel], rec["dq_window"][sel], rec["motor_err"][sel]
            worst_cl = cl_names[int(np.argmax(rec["gaps"][sel].max(axis=0)))]
            per[pr.name] = {"n": int(sel.sum()), "gap_max_mm": 1e3 * float(g.max()),
                            "gap_p95_mm": 1e3 * float(np.percentile(g, 95)),
                            "gap_median_mm": 1e3 * float(np.median(g)),
                            "frac_gap_gt_3mm": float((g > 3e-3).mean()), "worst_closure": worst_cl,
                            "dq_max": float(dq.max()), "dq_median": float(np.median(dq)),
                            "motor_err_max_deg": float(np.degrees(me.max())),
                            "motor_err_median_deg": float(np.degrees(np.median(me)))}
        report["programs"] = per
        allg = rec["gap"]
        report["summary"] = {"gap_max_mm": 1e3 * float(allg.max()), "gap_p95_mm": 1e3 * float(np.percentile(allg, 95)),
                             "gap_median_mm": 1e3 * float(np.median(allg)),
                             "frac_gap_gt_3mm": float((allg > 3e-3).mean()),
                             "dq_p95": float(np.percentile(rec["dq_window"], 95)),
                             "motor_err_p95_deg": float(np.degrees(np.percentile(rec["motor_err"], 95)))}
        print(f"[probe] {tag} sweep ms/step={report['sweep_ms_per_step']:.2f} steps={report['sweep_physics_steps']} "
              f"wall={report['sweep_wall_s']:.1f}s summary={report['summary']}", flush=True)
        for k, v in per.items():
            print(f"[probe]   {k:40s} gap_max={v['gap_max_mm']:7.3f} p95={v['gap_p95_mm']:6.3f} mm "
                  f">3mm={v['frac_gap_gt_3mm']:.2f} dq_max={v['dq_max']:.1e} err_max={v['motor_err_max_deg']:.2f} deg "
                  f"worst={v['worst_closure']}", flush=True)
        report["status"] = "ok"
        rc = 0
    except Exception:
        import traceback

        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"[probe] wrote {out}", flush=True)
    close_app_and_exit(app, rc)
    return rc


def parent() -> int:
    if not (REPO / ".locks/gpu.lock").exists():
        print("[probe] refusing to run: the GPU lock is not held (use tools/gpu_lock_run.py)", flush=True)
        return 3
    summary = []
    for c in ARGS.configs:
        it, n = (int(x) for x in c.split("x"))
        cmd = [_paths.isaac_python(), "-u", str(Path(__file__).relative_to(REPO)), "--child",
               "--pos-iters", str(it), "--num-envs", str(n), "--ramp", str(ARGS.ramp), "--hold", str(ARGS.hold),
               "--motor-kp", str(ARGS.motor_kp), "--motor-kd", str(ARGS.motor_kd),
               "--passive-damping", str(ARGS.passive_damping), "--serial-step-deg", str(ARGS.serial_step_deg),
               "--loop-step-deg", str(ARGS.loop_step_deg), "--out-dir", str(ARGS.out_dir), "--headless"]
        if ARGS.no_fixes:
            cmd.append("--no-fixes")
        print(f"[probe] === config {c}: {' '.join(cmd)}", flush=True)
        t = time.time()
        rc = subprocess.call(cmd, cwd=str(REPO))
        tag = f"it{it}_n{n}_r{ARGS.ramp}h{ARGS.hold}"
        path = ARGS.out_dir / f"probe_{tag}.json"
        row = {"config": c, "rc": rc, "wall_s": time.time() - t}
        if path.exists():
            d = json.loads(path.read_text())
            row.update({k: d.get(k) for k in ("rest_ms_per_step", "sweep_ms_per_step", "sweep_wall_s", "summary",
                                                "root_drift_after_sweep_mm", "status")})
        summary.append(row)
        print(f"[probe] === {json.dumps(row)}", flush=True)
    (ARGS.out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=1))
    return 0 if all(r["rc"] == 0 for r in summary) else 1


if __name__ == "__main__":
    raise SystemExit(child() if ARGS.child else parent())
