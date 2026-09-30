"""Run the tracking-env throughput matrix (one Isaac process per config, each under the GPU lock) and
collect ``logs/robot_task/throughput_matrix.json`` + a markdown table on stdout.

Each cell: warm-up, ``--steps`` timed zero-action policy steps of the TRAINING config (pushes, RSI noise,
resets), then a standing hold (pushes off, no RSI noise, reset to frame 0, ``--hold_steps`` zero-action
steps) that records loop-closure gap statistics while the robot still stands (see
``tools/tracking_env_probe.py:standing_hold``). Existing cell JSONs are reused unless ``--force``.

    python tools/run_throughput_matrix.py --envs 256 1024 2048 --solvers 8,4 16,4 32,4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def cell(n: int, pos: int, vel: int, args) -> dict:
    tag = f"throughput_n{n}_s{pos}-{vel}"
    out = REPO / "logs" / "robot_task" / f"{tag}.json"
    rc = None
    if args.force or not out.is_file() or json.loads(out.read_text(encoding="utf-8")).get("status") != "executed":
        cmd = [
            sys.executable, str(REPO / "tools" / "gpu_lock_run.py"), "--owner", "robot_task",
            "--log", f"logs/robot_task/{tag}.log", "--timeout", str(args.timeout), "--",
            _paths.isaac_python(), "-u", "tools/tracking_env_probe.py", "--mode", "throughput",
            "--num_envs", str(n), "--steps", str(args.steps), "--warmup", str(args.warmup),
            "--hold_steps", str(args.hold_steps), "--solver_iters", str(pos), str(vel),
            "--out", str(out), "--headless",
        ]
        print(f"[matrix] {tag}", flush=True)
        rc = subprocess.run(cmd, cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    row = {"num_envs": n, "solver": f"{pos}/{vel}", "rc": rc, "log": f"logs/robot_task/{tag}.log", "json": f"logs/robot_task/{tag}.json"}
    if out.is_file():
        r = json.loads(out.read_text(encoding="utf-8"))
        row.update({k: r.get(k) for k in ("status", "env_steps_per_s", "physics_steps_per_s", "gpu_used_mib_during",
                                           "gpu_baseline_mib", "gpu_process_delta_mib", "torch_max_allocated_mib",
                                           "env_create_s", "finite", "wall_s", "steps")})
        hold = r.get("standing_hold") or {}
        row["hold"] = {k: hold.get(k) for k in ("worst_gap_m", "frac_samples_over_3mm", "worst_closure", "gap_after_reset_m",
                                                 "standing_fraction_mean", "standing_fraction_last", "samples")}
        if r.get("status") == "failed":
            row["error_tail"] = (r.get("error") or "")[-400:]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", type=int, nargs="+", default=[256, 1024, 2048])
    ap.add_argument("--solvers", nargs="+", default=["8,4", "16,4", "32,4"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--hold_steps", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "logs" / "robot_task" / "throughput_matrix.json")
    args = ap.parse_args()
    rows = []
    for n in args.envs:
        for solver in args.solvers:
            pos, vel = (int(x) for x in solver.split(","))
            row = cell(n, pos, vel, args)
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print("\n| envs | solver pos/vel | env-steps/s | GPU used MiB (total) | delta vs idle MiB | hold gap median / p95 / max [mm] | >3 mm | standing frac | status |")
    print("|---:|:---:|---:|---:|---:|:---:|---:|---:|:---|")
    for r in rows:
        g = (r.get("hold") or {}).get("worst_gap_m") or {}
        gap = f"{g['median'] * 1e3:.2f} / {g['p95'] * 1e3:.2f} / {g['max'] * 1e3:.2f}" if g else "-"
        over = (r.get("hold") or {}).get("frac_samples_over_3mm")
        sf = (r.get("hold") or {}).get("standing_fraction_mean")
        eps = r.get("env_steps_per_s")
        print(f"| {r['num_envs']} | {r['solver']} | {eps:.0f} | {r.get('gpu_used_mib_during')} | {r.get('gpu_process_delta_mib')} | {gap} | "
              f"{'-' if over is None else f'{over:.3f}'} | {'-' if sf is None else f'{sf:.2f}'} | {r.get('status')} rc={r['rc']} |"
              if eps else f"| {r['num_envs']} | {r['solver']} | - | - | - | - | - | - | {r.get('status')} rc={r['rc']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
