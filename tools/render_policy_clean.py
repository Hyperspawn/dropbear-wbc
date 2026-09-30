"""Glitch-free clean policy videos: record the policy rollout, then render it kinematically (demo_eval, 2026-09-24).

Why: ``scripts/play.py --clean_video`` renders while physics runs; in 3 of 4 play renders on 2026-09-24 the RTX ``rgb``
annotator returned 25-61 % blank frames (black or washed-out grey), even with re-render retries, while every kinematic
replay (``scripts/render_reference.py``: write state, ``sim.render()``, read) was clean. So each job here runs two Isaac
processes:

1. ``scripts/play.py --record_rollout <npz>`` (no cameras): the policy rollout of env 0, saved per policy step as a
   contract motion NPZ of the SIMULATED state (all joints, all body link poses; ``meta.not_a_reference``);
2. ``scripts/render_reference.py --motion_file <npz>``: writes that state frame by frame and renders it with the clean
   renderer (``demo_render.DemoRecorder``); physics is never stepped, so the video shows exactly the simulated motion.

Then every frame of every output mp4 is checked (``demo_render.frame_is_valid``) and the result is written to
``<video_dir>/<tag>_frame_check.json``. The caller holds the GPU lock for the whole batch:

    python tools/gpu_lock_run.py --owner demo_eval --log logs/demo_eval/render_jobs_x.log --timeout 1800 -- \
        python -u tools/render_policy_clean.py --jobs logs/demo_eval/render_jobs_x.json

Job keys: ``tag``, ``motion_file``, ``checkpoint`` | ``load_run``, ``steps``, ``solver_iters`` [32, 4], ``caption``
(literal ``\\n`` separates lines), ``cams`` ("track,front,feet"), ``allow_rejected_motion`` (false), ``play_args`` ([]),
``rollout`` (default ``logs/demo_eval/rollouts/<tag>.npz``), ``video_dir`` (``logs/demo_eval/media``).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
ISAAC = _paths.isaac_python()


def check_frames(mp4: Path) -> dict:
    import numpy as np

    sys.path.insert(0, str(REPO / "source"))
    from dropbear_wbc.tasks.tracking.demo_render import ffmpeg_exe, frame_is_valid

    w, h = 1280, 720
    probe = subprocess.run([str(Path(ffmpeg_exe()).with_name("ffprobe.exe")), "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(mp4)],
                           capture_output=True, text=True)
    if probe.returncode == 0 and "," in probe.stdout:
        w, h = (int(v) for v in probe.stdout.strip().split(",")[:2])
    proc = subprocess.Popen([ffmpeg_exe(), "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    n, bad, size = 0, [], w * h * 3
    while True:  # stream frame by frame (a whole 30 s clip at 1280x720 would be ~4 GB)
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        if not frame_is_valid(np.frombuffer(buf, np.uint8).reshape(h, w, 3)):
            bad.append(n)
        n += 1
    proc.wait()
    return {"mp4": str(mp4), "frames": n, "invalid_frames": len(bad), "first_invalid": bad[:20]}


def run(cmd: list[str], log) -> int:
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT).returncode
    log.write(f"[render_policy_clean] rc={rc} wall_s={time.time() - t0:.1f}\n")
    log.flush()
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=Path, default=None, help="JSON file with a list of jobs")
    ap.add_argument("--job", action="append", default=[], help="one job as an inline JSON object (repeatable)")
    a = ap.parse_args()
    jobs = []
    if a.jobs is not None:
        jobs += json.loads((a.jobs if a.jobs.is_absolute() else REPO / a.jobs).read_text(encoding="utf-8"))
    jobs += [json.loads(j) for j in a.job]
    if not jobs:
        ap.error("give --jobs <file> and/or --job <json>")
    results, rc_all = [], 0
    for j in jobs:
        tag = j["tag"]
        video_dir = j.get("video_dir", "logs/demo_eval/media")
        rollout = j.get("rollout", f"logs/demo_eval/rollouts/{tag}.npz")
        cams = j.get("cams", "track,front,feet")
        sel = ["--checkpoint", j["checkpoint"]] if j.get("checkpoint") else ["--load_run", j["load_run"]]
        rej = ["--allow_rejected_motion"] if j.get("allow_rejected_motion") else []
        si = [str(v) for v in j.get("solver_iters", [32, 4])]
        log_path = REPO / "logs" / "demo_eval" / f"render_clean_{tag}.log"
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            rc1 = run([ISAAC, "-u", "scripts/play.py", "--task", j.get("task", "Dropbear-Tracking-Flat-Play-v0"),
                       "--motion_file", j["motion_file"], *sel, "--solver_iters", *si, *rej, "--num_envs", "1",
                       "--steps", str(j["steps"]), "--record_rollout", rollout, *j.get("play_args", []), "--headless"], log)
            rc2 = 1
            if rc1 == 0:
                cap = ["--video_caption", j["caption"]] if j.get("caption") else []
                rc2 = run([ISAAC, "-u", "scripts/render_reference.py", "--motion_file", rollout, "--video_cams", cams,
                           "--video_dir", video_dir, "--video_tag", tag, *cap, "--headless"], log)
            checks = []
            if rc2 == 0:
                for cam in cams.split(","):
                    checks.append(check_frames(REPO / video_dir / f"{tag}_{cam.strip()}.mp4"))
            res = {"tag": tag, "rollout_rc": rc1, "render_rc": rc2, "rollout": rollout, "log": str(log_path),
                   "frame_check": checks}
            (REPO / video_dir / f"{tag}_frame_check.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
            log.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)
        results.append(res)
        rc_all |= int(rc1 != 0 or rc2 != 0 or any(c["invalid_frames"] for c in checks))
    print(json.dumps({"render_policy_clean": results}), flush=True)
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
