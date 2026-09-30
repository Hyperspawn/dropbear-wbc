"""Bounded PPO training in GPU-lock-friendly chunks (each chunk is its own Isaac process under the team lock).

Chunk 1 starts a new run ``logs/rsl_rl/dropbear_tracking/<stamp>_<run_name>``; chunk k > 1 resumes the latest
checkpoint of that run with ``--resume --continue_run`` (same directory; ``metrics.jsonl`` appends and the
iteration count continues). Between chunks the lock is released, so other team members' GPU jobs can run.

    python tools/run_chunked_training.py --run_name static_stand --chunks 3 --iters_per_chunk 100 \
        --num_envs 2048 --solver_iters 16 4 --motion_file data/motions/smoke/dropbear_static_stand.npz

Prints one JSON line per chunk (rc, wall time, last metrics) and writes ``<run>/chunks.json``.

Calibration pinning: the default pose (action offset) comes from the calibration JSON, which another component
may rewrite while a run is in progress. The driver copies it once to
``data/calibration/snapshots/dropbear_semantic_calibration_<sha8>.json`` and passes that path to every chunk via
``$DROPBEAR_CALIBRATION_JSON`` (recorded in ``chunks.json``; ``scripts/play.py`` re-uses the run's pinned file).
Rejected motions (``meta.status`` or ``<clip>.validation.json`` verdict ``rejected``) fail closed in the env unless
``--allow_rejected_motion`` (recorded in ``<run>/motion_acceptance.json``, inherited by resumed chunks). The learning
rate and the adaptive-sampling statistics are checkpointed and restored across chunks (``--restore_train_state``;
runs started before 2026-09-24 13:00 keep their original restart-from-scratch behaviour under ``auto``).
It also refuses a motion NPZ whose ``meta.default_motor_pos`` (the pose a static clip was built for) differs
from the pinned calibration's ``standing_motor_pos`` by more than 1e-4 rad (``--allow_stale_npz`` overrides).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "logs" / "rsl_rl" / "dropbear_tracking"
CALIB = REPO / "data" / "calibration" / "dropbear_semantic_calibration.json"


def pin_calibration(src: Path) -> Path | None:
    """Copy ``src`` to ``data/calibration/snapshots/<name>_<sha8>.json`` (once); return the snapshot path."""
    if not src.is_file():
        return None
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if src.parent.name == "snapshots" and src.stem.endswith(f"_{sha[:8]}"):
        return src  # already a pinned snapshot (e.g. --calibration data/calibration/snapshots/<name>_<sha8>.json)
    dst = src.parent / "snapshots" / f"{src.stem}_{sha[:8]}.json"
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return dst


def npz_pose_mismatch(motion: Path, calibration: Path) -> float | None:
    """Max |NPZ meta.default_motor_pos - calibration standing_motor_pos| [rad], or None if the NPZ records none."""
    import numpy as np

    sys.path.insert(0, str(REPO / "source"))
    from dropbear_wbc.robots.defaults import load_standing_motor_pos

    with np.load(motion, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    built_for = meta.get("default_motor_pos")
    if not isinstance(built_for, dict):
        return None
    pose = load_standing_motor_pos(calibration)
    return max(abs(float(built_for[n]) - v) for n, v in pose.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--motion_file", default="data/motions/smoke/dropbear_static_stand.npz")
    ap.add_argument("--motion_library", default=None,
                    help="motion-library manifest (dropbear-motion-library-v1, added 2026-09-24 by multiclip): passed to "
                    "scripts/train.py --motion_library instead of --motion_file; needs a library --task "
                    "(Dropbear-Tracking-Library-v0 / -Future-v0). Every clip that records a default pose is checked "
                    "against the pinned calibration like --motion_file")
    ap.add_argument("--experiment_name", default=None,
                    help="logs/rsl_rl/<experiment> of the task's PPO config (default: dropbear_tracking, or "
                    "dropbear_tracking_library with --motion_library)")
    ap.add_argument("--task", default="Dropbear-Tracking-Flat-v0")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--iters_per_chunk", type=int, default=100)
    ap.add_argument("--num_envs", type=int, default=2048)
    ap.add_argument("--solver_iters", type=int, nargs=2, default=(16, 4))
    ap.add_argument("--save_interval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=900.0, help="per-chunk wall-clock limit [s]")
    ap.add_argument("--resume_run", default=None, help="continue an existing run dir name instead of starting one")
    ap.add_argument("--init_run", default=None,
                    help="warm start (added 2026-09-24, demo_eval): chunk 1 of a NEW run loads this run dir's checkpoint "
                    "(scripts/train.py --resume --load_run <init_run> without --continue_run: new run dir, weights + "
                    "optimizer + iteration counter from the checkpoint). Needs an identical observation/action layout.")
    ap.add_argument("--init_checkpoint", default=None, help="checkpoint file name in --init_run (default: latest)")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="calibration JSON to pin (default: the run's pinned file when resuming, else the live one)")
    ap.add_argument("--allow_stale_npz", action="store_true")
    ap.add_argument("--log_dir", default="logs/robot_task", help="where the per-chunk logs go (repo-relative)")
    ap.add_argument("--owner", default="robot_task", help="GPU lock owner name")
    ap.add_argument("--allow_rejected_motion", action="store_true",
                    help="EXPLORATORY: pass --allow_rejected_motion to scripts/train.py (recorded in <run>/motion_acceptance.json)")
    ap.add_argument("--accept_reason", default="", help="reason recorded with --allow_rejected_motion")
    ap.add_argument("--restore_train_state", choices=("auto", "on", "off"), default="auto",
                    help="scripts/train.py --restore_train_state (learning rate + adaptive sampling across chunks)")
    ap.add_argument("--actuator_profile", default=None, help="scripts/train.py --actuator_profile (legacy / hw_v1 / ...)")
    ap.add_argument("--hw_regularizers", action="store_true", help="scripts/train.py --hw_regularizers")
    ap.add_argument("--knee_stop_penalty", type=float, default=0.0, help="scripts/train.py --knee_stop_penalty (< 0)")
    ap.add_argument("--feet_slide_penalty", type=float, default=0.0, help="scripts/train.py --feet_slide_penalty (< 0)")
    ap.add_argument("--target_clamp_margin_deg", type=float, default=None, help="scripts/train.py --target_clamp_margin_deg")
    ap.add_argument("--action_rate_weight", type=float, default=None, help="scripts/train.py --action_rate_weight")
    ap.add_argument("--isaac_python", default=_paths.isaac_python(),
                    help="command that runs an Isaac Lab script (split like a shell word list), e.g. on Linux "
                    "'/isaac-sim/python.sh' or '/opt/IsaacLab/isaaclab.sh -p' or 'python' (pip isaaclab; docs/BREV.md)")
    ap.add_argument("--gap_s", type=float, default=45.0,
                    help="pause between chunks so other team members polling the GPU lock (30 s) can take it")
    args = ap.parse_args()
    exp = EXP
    if args.experiment_name or args.motion_library:
        exp = EXP.parent / (args.experiment_name or "dropbear_tracking_library")
    motion_files = [args.motion_file]
    if args.motion_library:
        sys.path.insert(0, str(REPO / "source"))
        from dropbear_wbc.tasks.tracking.motion_library import load_manifest

        motion_files = [str(c.npz) for c in load_manifest(REPO / args.motion_library).clips]

    pinned: Path | None = None
    if args.calibration is not None:
        pinned = pin_calibration(args.calibration)
    elif args.resume_run and (exp / args.resume_run / "chunks.json").is_file():
        prev = json.loads((exp / args.resume_run / "chunks.json").read_text(encoding="utf-8"))
        pinned = Path(prev[0]["calibration"]) if prev and prev[0].get("calibration") else None
    else:
        pinned = pin_calibration(CALIB)
    env = dict(os.environ)
    if pinned is not None:
        env["DROPBEAR_CALIBRATION_JSON"] = str(pinned)
        diffs = [d for d in (npz_pose_mismatch(REPO / f, pinned) for f in motion_files) if d is not None]
        diff = max(diffs) if diffs else None
        if diff is not None and diff > 1e-4 and not args.allow_stale_npz:
            raise SystemExit(f"motion NPZ was built for a default pose {diff:.4f} rad away from the pinned calibration "
                             f"{pinned.name}: rebuild the clip (tools/make_static_npz.py) or pass --allow_stale_npz")
        print(json.dumps({"npz_default_pose_vs_pinned_max_abs_rad": diff}), flush=True)
    print(json.dumps({"pinned_calibration": str(pinned) if pinned else None}), flush=True)

    run_dir: Path | None = exp / args.resume_run if args.resume_run else None
    k0 = 0
    if run_dir is not None and (run_dir / "chunks.json").is_file():
        k0 = len(json.loads((run_dir / "chunks.json").read_text(encoding="utf-8")))
    records = []
    for k in range(k0 + 1, k0 + args.chunks + 1):
        log = f"{args.log_dir.rstrip('/')}/train_{args.run_name}_chunk{k}.log"
        cmd = [
            sys.executable, str(REPO / "tools" / "gpu_lock_run.py"), "--owner", args.owner, "--log", log,
            "--timeout", str(args.timeout), "--", *shlex.split(args.isaac_python), "-u", "scripts/train.py",
            "--task", args.task, *(["--motion_library", args.motion_library] if args.motion_library else
                                   ["--motion_file", args.motion_file]), "--num_envs", str(args.num_envs),
            "--max_iterations", str(args.iters_per_chunk), "--save_interval", str(args.save_interval),
            "--solver_iters", str(args.solver_iters[0]), str(args.solver_iters[1]), "--seed", str(args.seed + k - 1),
            "--run_name", args.run_name, "--headless",
        ]
        if run_dir is not None:
            cmd[cmd.index("--headless"):cmd.index("--headless")] = ["--resume", "--load_run", run_dir.name, "--continue_run"]
        elif args.init_run:
            init = ["--resume", "--load_run", args.init_run] + (
                ["--checkpoint", args.init_checkpoint] if args.init_checkpoint else [])
            cmd[cmd.index("--headless"):cmd.index("--headless")] = init
        cmd[cmd.index("--headless"):cmd.index("--headless")] = ["--restore_train_state", args.restore_train_state]
        if args.actuator_profile:
            cmd[cmd.index("--headless"):cmd.index("--headless")] = ["--actuator_profile", args.actuator_profile]
        if args.hw_regularizers:
            cmd[cmd.index("--headless"):cmd.index("--headless")] = ["--hw_regularizers"]
        if args.knee_stop_penalty:
            cmd[cmd.index("--headless"):cmd.index("--headless")] = [f"--knee_stop_penalty={args.knee_stop_penalty}"]
        if args.feet_slide_penalty:
            cmd[cmd.index("--headless"):cmd.index("--headless")] = [f"--feet_slide_penalty={args.feet_slide_penalty}"]
        if args.target_clamp_margin_deg is not None:
            cmd[cmd.index("--headless"):cmd.index("--headless")] = [f"--target_clamp_margin_deg={args.target_clamp_margin_deg}"]
        if args.action_rate_weight is not None:
            cmd[cmd.index("--headless"):cmd.index("--headless")] = [f"--action_rate_weight={args.action_rate_weight}"]
        if args.experiment_name:
            cmd[cmd.index("--headless"):cmd.index("--headless")] = ["--experiment_name", args.experiment_name]
        if args.allow_rejected_motion:
            cmd[cmd.index("--headless"):cmd.index("--headless")] = ["--allow_rejected_motion", "--accept_reason",
                                                                    args.accept_reason or "run_chunked_training"]
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if run_dir is None:
            runs = sorted(exp.glob(f"*_{args.run_name}"), key=lambda p: p.stat().st_mtime)
            run_dir = runs[-1] if runs else None
        rec = {"chunk": k, "rc": rc, "wall_s": round(time.time() - t0, 1), "log": log, "run_dir": str(run_dir),
               "calibration": str(pinned) if pinned else None}
        if k == 1 and args.init_run and not args.resume_run:
            rec["warm_start"] = {"init_run": args.init_run, "init_checkpoint": args.init_checkpoint or "latest"}
        if run_dir is not None and (run_dir / "metrics.jsonl").is_file():
            lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if lines:
                last = json.loads(lines[-1])
                rec.update(last_it=last["it"], mean_episode_length=last["mean_episode_length"],
                           mean_reward=last["mean_reward"], finite=last["finite"], steps_per_s=last["steps_per_s"])
        records.append(rec)
        print(json.dumps(rec), flush=True)
        if run_dir is not None:
            old = []
            if args.resume_run and (run_dir / "chunks.json").is_file() and k == k0 + 1:
                old = json.loads((run_dir / "chunks.json").read_text(encoding="utf-8"))
                records[:0] = old
            (run_dir / "chunks.json").write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
        if rc != 0:
            return rc
        if k < k0 + args.chunks:
            time.sleep(args.gap_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
