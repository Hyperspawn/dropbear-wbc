"""Chunked PPO training of the velocity task (each chunk = one Isaac process under the team GPU lock).

Adapted from ``tools/run_chunked_training.py`` (tracking) for ``scripts/train_locomotion.py``:

* chunk 1 starts ``logs/rsl_rl/dropbear_velocity/<stamp>_<run_name>`` (or ``--resume_run`` continues one); chunk k > 1
  resumes the latest checkpoint with ``--resume --continue_run`` (same directory, ``metrics.jsonl`` appends, the
  iteration count continues, learning rate + command-curriculum ranges restored from the checkpoint);
* the calibration (action offset = standing pose) is pinned once to ``data/calibration/snapshots/..._<sha8>.json`` and
  passed to every chunk via ``$DROPBEAR_CALIBRATION_JSON`` (recorded in ``chunks.json``);
* ``--first_wait_minutes``: GPU-lock wait of the FIRST chunk (queue a run behind a long job of another track);
  later chunks wait ``--wait_minutes``; a lock timeout (gpu_lock_run rc 125) is retried up to ``--lock_retries`` times;
* ``--after_pid``: do not even queue for the lock until these processes (e.g. another track's long-run driver) have
  exited (polled every 30 s via ``tasklist``, image name checked; ``--after_pid_max_hours`` bounds the wait);
* graceful stop: create ``logs/locomotion/STOP_<run_name>`` -- the driver exits before starting the next chunk.

    python tools/run_chunked_locomotion.py --run_name vel_flat_s8 --chunks 20 --iters_per_chunk 150 \
        --num_envs 2048 --solver_iters 8 4 --save_interval 50 --timeout 1200 --first_wait_minutes 240

Prints one JSON line per chunk (rc, wall time, last metrics) and writes ``<run>/chunks.json``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
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
EXP = REPO / "logs" / "rsl_rl" / "dropbear_velocity"
CALIB = REPO / "data" / "calibration" / "dropbear_semantic_calibration.json"
LOCK_TIMEOUT_RC = 125


def pin_calibration(src: Path) -> Path | None:
    """Copy ``src`` to ``data/calibration/snapshots/<name>_<sha8>.json`` (once); return the snapshot path."""
    if not src.is_file():
        return None
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    dst = src.parent / "snapshots" / f"{src.stem}_{sha[:8]}.json"
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return dst


def pid_alive(pid: int, image: str = "python") -> bool:
    """True if ``pid`` runs and its image name contains ``image`` (Windows ``tasklist``; never signals it)."""
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True,
                         check=False).stdout
    return any(f'"{pid}"' in line and image.lower() in line.lower() for line in out.splitlines())


def boot_time() -> float:
    """System boot time (epoch seconds; Windows GetTickCount64, else /proc/uptime)."""
    try:
        import ctypes

        return time.time() - ctypes.windll.kernel32.GetTickCount64() / 1000.0
    except Exception:  # noqa: BLE001
        return time.time() - float(Path("/proc/uptime").read_text().split()[0])


def clear_pre_boot_lock() -> str | None:
    """Delete ``.locks/gpu.lock`` ONLY if the file predates the current boot (its holder died with the machine; a
    reboot skips gpu_lock_run's ``finally``). A lock written after the boot is never touched."""
    lock = REPO / ".locks" / "gpu.lock"
    try:
        mtime = lock.stat().st_mtime
    except FileNotFoundError:
        return None
    if mtime < boot_time() - 60.0:
        content = lock.read_text(encoding="utf-8", errors="replace")
        lock.unlink(missing_ok=True)
        return content
    return None


def latest_iteration(run_dir: Path | None) -> int | None:
    """Highest ``model_<it>.pt`` iteration in ``run_dir`` (None if there is no checkpoint)."""
    if run_dir is None:
        return None
    its = [int(p.stem.split("_")[1]) for p in run_dir.glob("model_*.pt") if p.stem.split("_")[1].isdigit()]
    return max(its) if its else None


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--task", default="Dropbear-Velocity-Flat-v0")
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--iters_per_chunk", type=int, default=150)
    ap.add_argument("--num_envs", type=int, default=2048)
    ap.add_argument("--solver_iters", type=int, nargs=2, default=(8, 4))
    ap.add_argument("--save_interval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=1200.0, help="per-chunk wall-clock limit [s] (after the lock)")
    ap.add_argument("--wait_minutes", type=float, default=60.0, help="GPU-lock wait of chunks 2.. [min]")
    ap.add_argument("--first_wait_minutes", type=float, default=60.0, help="GPU-lock wait of the first chunk [min]")
    ap.add_argument("--lock_retries", type=int, default=3, help="retries of a chunk whose lock wait timed out (rc 125)")
    ap.add_argument("--resume_run", default=None, help="continue an existing run dir name instead of starting one")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="calibration JSON to pin (default: the run's pinned file when resuming, else the live one)")
    ap.add_argument("--stand_npz", default=None, help="reset NPZ (scripts/train_locomotion.py --stand_npz)")
    ap.add_argument("--pushes", action="store_true")
    ap.add_argument("--actuator_profile", default=None, help="scripts/train_locomotion.py --actuator_profile")
    ap.add_argument("--thermal_penalty", type=float, default=None, help="scripts/train_locomotion.py --thermal_penalty")
    ap.add_argument("--gait_shaping", action="store_true", help="scripts/train_locomotion.py --gait_shaping")
    ap.add_argument("--gait_v2", action="store_true", help="scripts/train_locomotion.py --gait_v2")
    ap.add_argument("--gait_v3", action="store_true", help="scripts/train_locomotion.py --gait_v3")
    ap.add_argument("--terrain", choices=("flat", "rough"), default="flat", help="scripts/train_locomotion.py --terrain")
    ap.add_argument("--feet_width_weight", type=float, default=None, help="scripts/train_locomotion.py --feet_width_weight")
    ap.add_argument("--action_rate_weight", type=float, default=None, help="scripts/train_locomotion.py --action_rate_weight")
    ap.add_argument("--target_clamp_margin_deg", type=float, default=None,
                    help="scripts/train_locomotion.py --target_clamp_margin_deg (0 for new runs; docs/ISSUES.md #23)")
    ap.add_argument("--log_dir", default="logs/locomotion", help="per-chunk logs (repo-relative)")
    ap.add_argument("--owner", default="locomotion", help="GPU lock owner name")
    ap.add_argument("--gap_s", type=float, default=45.0, help="pause between chunks so other tracks can take the lock")
    ap.add_argument("--after_pid", type=int, nargs="*", default=[], help="wait until these (python) processes exit")
    ap.add_argument("--after_pid_max_hours", type=float, default=8.0)
    ap.add_argument("--until_iteration", type=int, default=None,
                    help="stop once the run's latest checkpoint reaches this learning iteration (resume-safe total; the "
                         "last chunk is shortened); --chunks is then only an upper bound")
    ap.add_argument("--isaac_python", default=_paths.isaac_python(),
                    help="Isaac Lab python launcher (Linux/Brev: e.g. \"python\" inside the Isaac venv); added 2026-09-25")
    ap.add_argument("--crash_retries", type=int, default=1,
                    help="re-run a chunk that exited non-zero (not a lock timeout) this many times, from the latest checkpoint")
    args = ap.parse_args()

    if args.after_pid:
        t_wait = time.time()
        print(json.dumps({"time": _now(), "waiting_for_pids": args.after_pid}), flush=True)
        while any(pid_alive(p) for p in args.after_pid):
            if time.time() - t_wait > args.after_pid_max_hours * 3600:
                print(json.dumps({"time": _now(), "after_pid_wait": "max hours reached, starting anyway"}), flush=True)
                break
            if (REPO / args.log_dir / f"STOP_{args.run_name}").is_file():
                print(json.dumps({"time": _now(), "stopped_by": "STOP file while waiting"}), flush=True)
                return 0
            time.sleep(30)
        print(json.dumps({"time": _now(), "pids_done": args.after_pid, "waited_s": round(time.time() - t_wait)}),
              flush=True)

    if args.calibration is not None:
        pinned = pin_calibration(args.calibration)
    elif args.resume_run and (EXP / args.resume_run / "chunks.json").is_file():
        prev = json.loads((EXP / args.resume_run / "chunks.json").read_text(encoding="utf-8"))
        pinned = Path(prev[0]["calibration"]) if prev and prev[0].get("calibration") else None
    else:
        pinned = pin_calibration(CALIB)
    env = dict(os.environ)
    if pinned is not None:
        env["DROPBEAR_CALIBRATION_JSON"] = str(pinned)
    print(json.dumps({"time": _now(), "pinned_calibration": str(pinned) if pinned else None, "pid": os.getpid(),
                      "argv": sys.argv[1:]}), flush=True)

    if args.resume_run and not args.actuator_profile:
        infos = sorted((EXP / args.resume_run).glob("run_info*.json"))
        if infos:
            args.actuator_profile = json.loads(infos[0].read_text(encoding="utf-8")).get("actuator_profile") or None
        print(json.dumps({"time": _now(), "actuator_profile_from_run_info": args.actuator_profile}), flush=True)
    stop_file = REPO / args.log_dir / f"STOP_{args.run_name}"
    run_dir: Path | None = EXP / args.resume_run if args.resume_run else None
    k0 = 0
    old_records: list = []
    if run_dir is not None and (run_dir / "chunks.json").is_file():
        old_records = json.loads((run_dir / "chunks.json").read_text(encoding="utf-8"))
        k0 = len(old_records)
    records: list = []
    for k in range(k0 + 1, k0 + args.chunks + 1):
        if stop_file.is_file():
            print(json.dumps({"time": _now(), "stopped_by": str(stop_file), "before_chunk": k}), flush=True)
            return 0
        stale = clear_pre_boot_lock()
        if stale is not None:
            print(json.dumps({"time": _now(), "removed_pre_boot_gpu_lock": stale}), flush=True)
        wait = args.first_wait_minutes if k == k0 + 1 else args.wait_minutes
        iters = args.iters_per_chunk
        if args.until_iteration is not None:
            done_it = latest_iteration(run_dir)
            if done_it is not None and done_it + 1 >= args.until_iteration:
                print(json.dumps({"time": _now(), "until_iteration_reached": done_it}), flush=True)
                return 0
            iters = min(iters, args.until_iteration - (done_it + 1 if done_it is not None else 0))
        t0 = time.time()
        rc, log, crashes = LOCK_TIMEOUT_RC, "", 0
        while True:
            log = f"{args.log_dir.rstrip('/')}/train_{args.run_name}_chunk{k}{'_retry' + str(crashes) if crashes else ''}.log"
            cmd = [
                sys.executable, str(REPO / "tools" / "gpu_lock_run.py"), "--owner", args.owner, "--log", log,
                "--timeout", str(args.timeout), "--wait-minutes", str(wait), "--",
                *shlex.split(args.isaac_python), "-u", "scripts/train_locomotion.py",
                "--task", args.task, "--num_envs", str(args.num_envs),
                "--max_iterations", str(iters), "--save_interval", str(args.save_interval),
                "--solver_iters", str(args.solver_iters[0]), str(args.solver_iters[1]),
                "--seed", str(args.seed + k - 1 + 100 * crashes), "--run_name", args.run_name,
            ]
            if args.stand_npz:
                cmd += ["--stand_npz", args.stand_npz]
            if args.pushes:
                cmd += ["--pushes"]
            if args.actuator_profile:
                cmd += ["--actuator_profile", args.actuator_profile]
            if args.thermal_penalty is not None:
                cmd += [f"--thermal_penalty={args.thermal_penalty}"]
            if args.gait_shaping:
                cmd += ["--gait_shaping"]
            if args.gait_v2:
                cmd += ["--gait_v2"]
            if args.gait_v3:
                cmd += ["--gait_v3"]
            if args.target_clamp_margin_deg is not None:
                cmd += [f"--target_clamp_margin_deg={args.target_clamp_margin_deg}"]
            if args.terrain != "flat":
                cmd += ["--terrain", args.terrain]
            if args.feet_width_weight is not None:
                cmd += [f"--feet_width_weight={args.feet_width_weight}"]
            if args.action_rate_weight is not None:
                cmd += [f"--action_rate_weight={args.action_rate_weight}"]
            if run_dir is not None and any(run_dir.glob("model_*.pt")):
                cmd += ["--resume", "--load_run", run_dir.name, "--continue_run"]
            cmd += ["--headless"]
            for attempt in range(args.lock_retries + 1):
                rc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL).returncode
                if rc != LOCK_TIMEOUT_RC:
                    break
                print(json.dumps({"time": _now(), "chunk": k, "lock_timeout_attempt": attempt + 1}), flush=True)
            if run_dir is None:
                runs = sorted(EXP.glob(f"*_{args.run_name}"), key=lambda p: p.stat().st_mtime)
                run_dir = runs[-1] if runs else None
            if rc in (0, LOCK_TIMEOUT_RC) or crashes >= args.crash_retries:
                break
            crashes += 1
            print(json.dumps({"time": _now(), "chunk": k, "rc": rc, "log": log, "crash_retry": crashes,
                              "note": "resuming from the latest checkpoint; metrics.jsonl may repeat iterations "
                                      "after it"}), flush=True)
            time.sleep(args.gap_s)
        rec = {"chunk": k, "rc": rc, "crash_retries_used": crashes, "end": _now(), "wall_s": round(time.time() - t0, 1), "log": log,
               "run_dir": str(run_dir), "calibration": str(pinned) if pinned else None}
        if run_dir is not None and (run_dir / "metrics.jsonl").is_file():
            lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if lines:
                last = json.loads(lines[-1])
                ep = last.get("episode", {})
                rec.update(last_it=last["it"], mean_episode_length=last["mean_episode_length"],
                           mean_reward=last["mean_reward"], finite=last["finite"], steps_per_s=last["steps_per_s"],
                           track_lin=ep.get("Episode_Reward/track_lin_vel_xy_exp"),
                           error_vel_xy=ep.get("Metrics/base_velocity/error_vel_xy"),
                           cmd_vx_max=ep.get("Curriculum/cmd_vx_max"))
        records.append(rec)
        print(json.dumps(rec), flush=True)
        if run_dir is not None:
            (run_dir / "chunks.json").write_text(json.dumps(old_records + records, indent=1) + "\n", encoding="utf-8")
        if rc != 0:
            return rc
        if k < k0 + args.chunks:
            time.sleep(args.gap_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
