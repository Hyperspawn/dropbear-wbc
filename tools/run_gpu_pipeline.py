"""Sequential, session-independent GPU pipeline for the gpu_pipeline track (launch it detached, e.g. via WMI).

Runs the steps of a named plan one after another; every GPU step goes through ``tools/gpu_lock_run.py`` (directly
or via ``tools/run_chunked_training.py``). State (step, command, rc, timings) is appended to
``logs/gpu_pipeline/pipeline_<plan>_state.jsonl`` after every step, so a successor can see what ran. A failed
evaluation step does not stop the plan; a failed training step skips the steps that need its checkpoints.

Plans:
  wave_then_dance  -- resume demo #1 (wave_right) training, summarize, eval at 32/4 (16 envs, 1010 steps, export),
                      export parity, video (1 env, 520 steps) + PNG frames, then launch the long demo #2 run
                      (G1_Take_102, chunked) and wait for it.

    C:/.../Python312/python.exe tools/run_gpu_pipeline.py --plan wave_then_dance
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGS = REPO / "logs" / "gpu_pipeline"
EXP = REPO / "logs" / "rsl_rl" / "dropbear_tracking"
FFMPEG = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"

WAVE_RUN = "2026-09-24_10-33-16_wave_right_s8"
WAVE_NPZ = "data/motions/synthetic/wave_right.npz"
DANCE_NPZ = "data/motions/unitree_rl_lab_mimic/G1_Take_102.npz"


def lock(log: str, timeout: int, *cmd: str) -> list[str]:
    return [PY, str(REPO / "tools" / "gpu_lock_run.py"), "--owner", "gpu_pipeline", "--log", log,
            "--timeout", str(timeout), "--", *cmd]


def isaac(script: str, *args: str) -> list[str]:
    return [_paths.isaac_python(), "-u", script, *args, "--headless"]


def plan_wave_then_dance(a) -> list[dict]:
    run = WAVE_RUN
    exported = f"logs/rsl_rl/dropbear_tracking/{run}/exported"
    return [
        {"name": "train_wave", "critical": True,
         "cmd": [PY, "-u", "tools/run_chunked_training.py", "--run_name", "wave_right_s8", "--resume_run", run,
                 "--motion_file", WAVE_NPZ, "--chunks", str(a.wave_chunks), "--iters_per_chunk", str(a.wave_iters),
                 "--num_envs", "2048", "--solver_iters", "8", "4", "--save_interval", "50", "--timeout", "780",
                 "--log_dir", "logs/gpu_pipeline", "--owner", "gpu_pipeline"],
         "log": "logs/gpu_pipeline/train_wave_right_s8_driver_resume.log"},
        {"name": "summarize_wave", "needs": "train_wave",
         "cmd": [PY, "-u", "tools/summarize_training.py", f"logs/rsl_rl/dropbear_tracking/{run}"],
         "log": "logs/gpu_pipeline/wave_right_s8_summary.log"},
        {"name": "eval_export_wave", "needs": "train_wave",
         "cmd": lock("logs/gpu_pipeline/play_wave_eval_32_4.log", 1200,
                     *isaac("scripts/play.py", "--task", "Dropbear-Tracking-Flat-Play-v0", "--motion_file", WAVE_NPZ,
                            "--load_run", run, "--num_envs", "16", "--steps", "1010", "--solver_iters", "32", "4",
                            "--export")),
         "log": "logs/gpu_pipeline/play_wave_eval_32_4.wrapper.out"},
        {"name": "parity_wave", "needs": "eval_export_wave",
         "cmd": [PY, "-u", "tools/check_export_parity.py", exported],
         "log": "logs/gpu_pipeline/check_export_parity_wave.log"},
        {"name": "video_wave", "needs": "train_wave",
         "cmd": lock("logs/gpu_pipeline/play_wave_video.log", 1200,
                     *isaac("scripts/play.py", "--task", "Dropbear-Tracking-Flat-Play-v0", "--motion_file", WAVE_NPZ,
                            "--load_run", run, "--num_envs", "1", "--steps", "520", "--solver_iters", "32", "4",
                            "--video", "--video_length", "520")),
         "log": "logs/gpu_pipeline/play_wave_video.wrapper.out"},
        {"name": "frames_wave", "needs": "video_wave", "func": "extract_frames", "args": {"run": run, "tag": "wave_right"},
         "log": "logs/gpu_pipeline/media/extract_frames_wave.log"},
        {"name": "long_dance", "critical": True,
         "cmd": [PY, "-u", "tools/run_chunked_training.py", "--run_name", "take102_s8", "--motion_file", DANCE_NPZ,
                 "--chunks", str(a.dance_chunks), "--iters_per_chunk", str(a.dance_iters), "--num_envs", "2048",
                 "--solver_iters", "8", "4", "--save_interval", "100", "--timeout", "900",
                 "--log_dir", "logs/gpu_pipeline", "--owner", "gpu_pipeline"],
         "log": "logs/gpu_pipeline/train_take102_s8_driver.log"},
    ]


def extract_frames(run: str, tag: str, log) -> int:
    vids = sorted((EXP / run / "videos" / "play").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not vids:
        log.write("no mp4 found\n")
        return 1
    media = LOGS / "media"
    media.mkdir(parents=True, exist_ok=True)
    src = vids[-1]
    dst = media / f"{tag}_play_32_4.mp4"
    dst.write_bytes(src.read_bytes())
    log.write(f"copied {src} -> {dst}\n")
    rc = 0
    for t in (1.0, 3.0, 5.5, 8.5):
        out = media / f"{tag}_t{t:04.1f}s.png"
        r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", str(t), "-i", str(dst), "-frames:v", "1",
                            str(out)], capture_output=True, text=True)
        log.write(f"frame t={t}: rc={r.returncode} {out} {r.stderr.strip()}\n")
        rc |= r.returncode
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", choices=["wave_then_dance"], required=True)
    ap.add_argument("--wave_chunks", type=int, default=7)
    ap.add_argument("--wave_iters", type=int, default=100)
    ap.add_argument("--dance_chunks", type=int, default=20)
    ap.add_argument("--dance_iters", type=int, default=120)
    ap.add_argument("--skip", default="", help="comma list of step names to skip (resume a broken plan)")
    a = ap.parse_args()
    steps = plan_wave_then_dance(a)
    skip = {s for s in a.skip.split(",") if s}
    state_path = LOGS / f"pipeline_{a.plan}_state.jsonl"
    ok: dict[str, bool] = {s: True for s in skip}
    with state_path.open("a", encoding="utf-8") as st:
        st.write(json.dumps({"event": "start", "pid": os.getpid(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "args": vars(a)}) + "\n")
        st.flush()
        for s in steps:
            if s["name"] in skip:
                continue
            if s.get("needs") and not ok.get(s["needs"], False):
                st.write(json.dumps({"step": s["name"], "skipped": f"needs {s['needs']}"}) + "\n")
                st.flush()
                ok[s["name"]] = False
                continue
            t0 = time.time()
            log_path = REPO / s["log"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            st.write(json.dumps({"step": s["name"], "event": "begin", "time": time.strftime("%H:%M:%S"),
                                 "cmd": s.get("cmd"), "log": s["log"]}) + "\n")
            st.flush()
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                if "func" in s:
                    rc = globals()[s["func"]](log=log, **s["args"])
                else:
                    proc = subprocess.Popen(s["cmd"], cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
                    st.write(json.dumps({"step": s["name"], "child_pid": proc.pid}) + "\n")
                    st.flush()
                    rc = proc.wait()
            ok[s["name"]] = rc == 0
            st.write(json.dumps({"step": s["name"], "event": "end", "rc": rc, "wall_s": round(time.time() - t0, 1),
                                 "time": time.strftime("%H:%M:%S")}) + "\n")
            st.flush()
        st.write(json.dumps({"event": "done", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "ok": ok}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
