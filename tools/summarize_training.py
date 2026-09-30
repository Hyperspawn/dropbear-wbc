"""Summarize a Dropbear tracking run's ``metrics.jsonl`` (written by ``DropbearOnPolicyRunner``).

Prints a compact table (every ``--every`` iterations), learning-signal statistics (episode length and
reward over the first/last windows, termination shares) and writes ``<run>/training_summary.json`` plus a
PNG plot ``<run>/training_curves.png`` (matplotlib, if available). System python, CPU only.

    python tools/summarize_training.py logs/rsl_rl/dropbear_tracking/<run>
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def window_mean(rows, key, lo, hi):
    vals = [r[key] for r in rows[lo:hi] if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()
    rows = [json.loads(l) for l in (args.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("empty metrics.jsonl")
    w = min(args.window, max(1, len(rows) // 3))
    term_keys = sorted({k for r in rows for k in r["episode"] if k.startswith("Episode_Termination/")})
    print(f"{'it':>5} {'ep_len':>7} {'reward':>8} {'act_std':>7} {'steps/s':>8} " + " ".join(k.split('/')[-1][:10].rjust(10) for k in term_keys))
    for r in rows[:: args.every] + ([rows[-1]] if (len(rows) - 1) % args.every else []):
        print(f"{r['it']:>5} {r['mean_episode_length'] or 0:>7.1f} {r['mean_reward'] or 0:>8.3f} {r['action_std']:>7.3f} "
              f"{r['steps_per_s'] or 0:>8.0f} " + " ".join(f"{r['episode'].get(k, 0):>10.3f}" for k in term_keys))
    first_len, last_len = window_mean(rows, "mean_episode_length", 0, w), window_mean(rows, "mean_episode_length", -w, None)
    best = max(rows, key=lambda r: r["mean_episode_length"] or 0)
    summary = {
        "run_dir": str(args.run_dir),
        "iterations": len(rows),
        "first_it": rows[0]["it"],
        "last_it": rows[-1]["it"],
        "window": w,
        "episode_length_first_window": first_len,
        "episode_length_last_window": last_len,
        "episode_length_max": {"it": best["it"], "value": best["mean_episode_length"]},
        "min_episode_length": min((r["mean_episode_length"] or 0) for r in rows),
        "reward_first_window": window_mean(rows, "mean_reward", 0, w),
        "reward_last_window": window_mean(rows, "mean_reward", -w, None),
        "all_finite": all(r["finite"] for r in rows),
        "mean_steps_per_s": statistics.mean(r["steps_per_s"] for r in rows if r["steps_per_s"]),
        "termination_share_last": {k.split("/")[-1]: rows[-1]["episode"].get(k) for k in term_keys},
        "tracking_error_last": {k.split("/")[-1]: v for k, v in rows[-1]["episode"].items() if k.startswith("Metrics/motion/error")},
        "max_episode_length_steps": None,
    }
    try:
        info = json.loads((args.run_dir / "run_info.json").read_text(encoding="utf-8"))
        summary["run_info"] = {k: info.get(k) for k in ("num_envs", "solver_iterations", "motion_file", "default_pose_source")}
    except FileNotFoundError:
        pass
    print(json.dumps(summary, indent=1))
    (args.run_dir / "training_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        its = [r["it"] for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(its, [r["mean_episode_length"] or 0 for r in rows])
        ax[0].set_title("mean episode length [policy steps, max 500]")
        ax[1].plot(its, [r["mean_reward"] or 0 for r in rows])
        ax[1].set_title("mean episode reward")
        for k in term_keys:
            ax[2].plot(its, [r["episode"].get(k, 0) for r in rows], label=k.split("/")[-1])
        ax[2].set_title("termination share")
        ax[2].legend()
        for a in ax:
            a.set_xlabel("PPO iteration")
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.run_dir / "training_curves.png", dpi=90)
        print(f"wrote {args.run_dir / 'training_curves.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
