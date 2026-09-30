"""Sequential GPU verification of the motion-library tracker (multiclip track; launch detached, e.g. via WMI).

Every GPU step goes through ``tools/gpu_lock_run.py`` (owner ``multiclip``) and is capped at ``--step_timeout`` s
(default 900 = 15 min; the GPU is shared). Before each step the driver YIELDS: it waits while the lock is held or while a
``gpu_lock_run.py --owner <yield_to>`` process (default ``demo_eval``, the track with priority) is waiting, and only then
asks for the lock (short lock wait, retried), so queued priority jobs go first. State (step, command, rc, timings) is
appended to ``logs/multiclip/verify_<plan>_state.jsonl`` after every step. A failed step is recorded; steps that need
its outputs are skipped.

Plan ``smoke`` (default):
  probe       tools/library_env_probe.py on Dropbear-Tracking-Library-Future-v0 (256 envs, 150 zero-action steps)
  smoke20     scripts/train.py Dropbear-Tracking-Library-v0, 2048 envs, 8/4, 20 PPO iterations (throughput + finite)
  per_clip    (CPU) per-clip metrics of smoke20 from metrics.jsonl -> logs/multiclip/smoke20_per_clip.json
  play_export scripts/play.py Dropbear-Tracking-Library-Play-v0 on the smoke20 run: one env per clip, 32/4,
              600 steps, --export (runtime-reference export + parity_rollout.pt)
  check       tools/check_library_export.py on that export (CPU; deploy runner fed the clip NPZ at runtime)
  smoke20_fut scripts/train.py Dropbear-Tracking-Library-Future-v0, 2048 envs, 8/4, 20 iterations (+ per-clip summary)

    C:/.../Python312/python.exe tools/run_multiclip_verification.py --plan smoke
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGS = REPO / "logs" / "multiclip"
EXP = REPO / "logs" / "rsl_rl" / "dropbear_tracking_library"
LOCK = REPO / ".locks" / "gpu.lock"
MANIFEST = "data/motions/libraries/accepted_v1.json"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
ISAAC = _paths.isaac_python()


def lock_waiters(owners: set[str]) -> list[str]:
    """Command lines of running ``gpu_lock_run.py`` processes whose ``--owner`` is in ``owners`` (Windows, via CIM)."""
    if not owners:
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*gpu_lock_run.py*' } | "
             "ForEach-Object { $_.CommandLine }"], capture_output=True, text=True, timeout=90).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    hits = []
    for line in out.splitlines():
        toks = line.split()
        for i, t in enumerate(toks):
            if (t == "--owner" and i + 1 < len(toks) and toks[i + 1].strip('"') in owners) or \
                    (t.startswith("--owner=") and t.split("=", 1)[1].strip('"') in owners):
                hits.append(line.strip())
    return hits


def summarize_per_clip(run_dir: Path, out: Path) -> dict:
    """Per-clip metrics (Metrics/motion/clip_*/<name>) and the aggregates of every iteration of a library run."""
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    keys = sorted({k for r in rows for k in r.get("episode", {}) if "/clip_" in k or "sampling_" in k})
    first, last = rows[0], rows[-1]
    per_clip: dict = {}
    for k in keys:
        if k.count("/") >= 3:  # Metrics/motion/clip_prob/<name>
            _, _, metric, name = k.split("/", 3)
            per_clip.setdefault(name, {})[metric] = {"first": first["episode"].get(k), "last": last["episode"].get(k)}
    rep = {"run_dir": str(run_dir), "iterations": len(rows), "finite_all": all(r.get("finite") for r in rows),
           "steps_per_s": [r.get("steps_per_s") for r in rows],
           "steps_per_s_mean_after_first": (sum(r["steps_per_s"] for r in rows[1:]) / max(len(rows) - 1, 1))
           if len(rows) > 1 else None,
           "mean_episode_length": [r.get("mean_episode_length") for r in rows],
           "mean_reward": [r.get("mean_reward") for r in rows],
           "aggregates_last": {k: last["episode"].get(k) for k in keys if k.count("/") < 3},
           "per_clip": per_clip, "metric_keys": keys,
           "note": "20-iteration smoke: interface/finite/throughput evidence only, not a tracking result."}
    out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return rep


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="smoke", choices=["smoke"])
    ap.add_argument("--skip", default="", help="comma list of steps to skip")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--wait_minutes", type=float, default=240.0, help="overall budget to get the GPU per step")
    ap.add_argument("--step_timeout", type=int, default=900, help="hard cap per GPU step [s] (gpu_lock_run --timeout)")
    ap.add_argument("--yield_to", default="demo_eval", help="comma list of lock owners that go first")
    ap.add_argument("--tag", default="", help="suffix for run names / logs (re-runs)")
    ap.add_argument("--calibration", default="data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    yield_to = {s.strip() for s in args.yield_to.split(",") if s.strip()}
    tag = f"_{args.tag}" if args.tag else ""
    state = LOGS / f"verify_{args.plan}{tag}_state.jsonl"
    env = dict(os.environ)
    env["DROPBEAR_CALIBRATION_JSON"] = str((REPO / args.calibration).resolve())
    manifest = args.manifest

    def record(**kw):
        kw["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(state, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\n")

    def gpu(name: str, log: str, *cmd: str) -> int:
        deadline = time.time() + args.wait_minutes * 60
        record(step=name, event="queued", yield_to=sorted(yield_to))
        rc = 125
        while True:
            # yield: wait for a free lock and no waiting priority job
            while True:
                pri = lock_waiters(yield_to)
                if not LOCK.exists() and not pri:
                    break
                if time.time() > deadline:
                    record(step=name, event="gave_up", reason="GPU not available within --wait_minutes")
                    return 125
                time.sleep(20)
            full = [PY, str(REPO / "tools" / "gpu_lock_run.py"), "--owner", "multiclip", "--log", log, "--timeout",
                    str(args.step_timeout), "--wait-minutes", "2", "--", *cmd]
            record(step=name, event="begin", command=full)
            t0 = time.time()
            rc = subprocess.run(full, cwd=str(REPO), env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL).returncode
            if rc != 125:  # 125 = someone else took the lock first: go back to waiting
                record(step=name, event="end", rc=rc, wall_s=round(time.time() - t0, 1), log=log)
                return rc
            record(step=name, event="lost_lock_race", wall_s=round(time.time() - t0, 1))
            if time.time() > deadline:
                return rc

    def latest_run(run_name: str) -> Path | None:
        runs = sorted(EXP.glob(f"*_{run_name}"), key=lambda p: p.stat().st_mtime) if EXP.is_dir() else []
        return runs[-1] if runs else None

    record(step="plan", event="start", plan=args.plan, manifest=manifest, calibration=env["DROPBEAR_CALIBRATION_JSON"],
           step_timeout=args.step_timeout)
    if "probe" not in skip:
        gpu("probe", f"logs/multiclip/library_env_probe{tag}.log", ISAAC, "-u", "tools/library_env_probe.py",
            "--task", "Dropbear-Tracking-Library-Future-v0", "--motion_library", manifest, "--num_envs", "256",
            "--steps", "150", "--out", f"logs/multiclip/library_env_probe{tag}.json", "--headless")
    run_dir = None
    if "smoke20" not in skip:
        rn = f"lib_smoke20{tag}"
        rc = gpu("smoke20", f"logs/multiclip/train_lib_smoke20{tag}.log", ISAAC, "-u", "scripts/train.py",
                 "--task", "Dropbear-Tracking-Library-v0", "--motion_library", manifest, "--num_envs", "2048",
                 "--max_iterations", "20", "--save_interval", "10", "--solver_iters", "8", "4", "--seed", "7",
                 "--run_name", rn, "--headless")
        run_dir = latest_run(rn) if rc == 0 else None
        record(step="smoke20", event="run_dir", run_dir=str(run_dir) if run_dir else None)
        if run_dir is not None and (run_dir / "metrics.jsonl").is_file():
            try:
                rep = summarize_per_clip(run_dir, LOGS / f"smoke20{tag}_per_clip.json")
                record(step="per_clip", event="end", out=f"logs/multiclip/smoke20{tag}_per_clip.json",
                       finite_all=rep["finite_all"], steps_per_s=rep["steps_per_s_mean_after_first"],
                       clips=sorted(rep["per_clip"]))
            except Exception as exc:  # noqa: BLE001
                record(step="per_clip", event="error", error=repr(exc))
    if run_dir is not None and "play_export" not in skip:
        n_clips = len(json.loads((REPO / manifest).read_text(encoding="utf-8"))["clips"])
        rc = gpu("play_export", f"logs/multiclip/play_lib_smoke20{tag}_export.log", ISAAC, "-u", "scripts/play.py",
                 "--task", "Dropbear-Tracking-Library-Play-v0", "--motion_library", manifest, "--load_run", run_dir.name,
                 "--num_envs", str(n_clips), "--steps", "600", "--solver_iters", "32", "4", "--export", "--headless")
        if rc == 0 and "check" not in skip:
            t0 = time.time()
            out = subprocess.run([PY, str(REPO / "tools" / "check_library_export.py"), str(run_dir / "exported"),
                                  "--out", f"logs/multiclip/check_library_export_smoke20{tag}.json"], cwd=str(REPO),
                                 env=env, capture_output=True, text=True)
            (LOGS / f"check_library_export_smoke20{tag}.log").write_text(out.stdout + out.stderr, encoding="utf-8")
            record(step="check", event="end", rc=out.returncode, wall_s=round(time.time() - t0, 1),
                   log=f"logs/multiclip/check_library_export_smoke20{tag}.log")
    if "smoke20_fut" not in skip:
        rn = f"lib_future_smoke20{tag}"
        rc = gpu("smoke20_fut", f"logs/multiclip/train_lib_future_smoke20{tag}.log", ISAAC, "-u", "scripts/train.py",
                 "--task", "Dropbear-Tracking-Library-Future-v0", "--motion_library", manifest, "--num_envs", "2048",
                 "--max_iterations", "20", "--save_interval", "10", "--solver_iters", "8", "4", "--seed", "8",
                 "--run_name", rn, "--headless")
        fut_dir = latest_run(rn) if rc == 0 else None
        record(step="smoke20_fut", event="run_dir", run_dir=str(fut_dir) if fut_dir else None)
        if fut_dir is not None and (fut_dir / "metrics.jsonl").is_file():
            try:
                rep = summarize_per_clip(fut_dir, LOGS / f"smoke20_future{tag}_per_clip.json")
                record(step="per_clip_fut", event="end", finite_all=rep["finite_all"],
                       steps_per_s=rep["steps_per_s_mean_after_first"])
            except Exception as exc:  # noqa: BLE001
                record(step="per_clip_fut", event="error", error=repr(exc))
    record(step="plan", event="done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
