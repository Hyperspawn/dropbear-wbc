"""Session-independent pipeline of the demo_eval track (launch it detached via WMI; one GPU step at a time).

Every GPU step goes through ``tools/gpu_lock_run.py`` (owner ``demo_eval``) or ``tools/run_chunked_training.py``.
State (step, command, rc, timings) is appended to ``logs/demo_eval/pipeline_<plan>_state.jsonl`` after every step,
so a successor can see what ran; ``--skip a,b`` resumes a broken plan. A failed evaluation step does not stop the
plan; a failed critical step (the wait for the dance run, training) skips the steps that need it.

Plans:
  take102  -- wait for the detached G1_Take_102 run (gpu_pipeline, 20 chunks) to end; summarize it; Isaac eval at 32/4
              on 16 envs from frame 0: nominal full clip (+ --export) / continuous 2 loops / perturbed continuous 2
              loops; export parity; clean policy render (track, front, feet) + reference replay (track, feet) +
              contact sheets + side-by-side; Newton sim2sim (CPU MuJoCo C, passive damping 0, eq_solref 0.004).
  wave_v2  -- warm-start training of wave_right_v2 from the wave_right policy (chunked), summarize, the same evals,
              clean render, sim2sim.
  all      -- take102 then wave_v2.

    python tools/run_demo_eval_pipeline.py --plan all
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
LOGS = REPO / "logs" / "demo_eval"
MEDIA = LOGS / "media"
EXP = REPO / "logs" / "rsl_rl" / "dropbear_tracking"
BASH = r"C:\Program Files\Git\bin\bash.exe"

DANCE_RUN = "2026-09-24_11-49-58_take102_s8"
DANCE_NPZ = "data/motions/unitree_rl_lab_mimic/G1_Take_102.npz"
DANCE_DRIVER_LOG = REPO / "logs" / "gpu_pipeline" / "train_take102_s8_driver.log"
DANCE_STATE = REPO / "logs" / "gpu_pipeline" / "pipeline_wave_then_dance_state.jsonl"
WAVE_RUN = "2026-09-24_10-33-16_wave_right_s8"
WAVE2_NPZ = "data/motions/synthetic/wave_right_v2.npz"
WAVE2_NAME = "wave_right_v2_ws8"


def lock(log: str, timeout: int, *cmd: str) -> list[str]:
    return [PY, str(REPO / "tools" / "gpu_lock_run.py"), "--owner", "demo_eval", "--log", log, "--timeout", str(timeout),
            "--wait-minutes", "240", "--", *cmd]


def isaac(script: str, *args: str) -> list[str]:
    return [_paths.isaac_python(), "-u", script, *args, "--headless"]


def npz_frames(npz: str) -> int:
    import numpy as np

    with np.load(REPO / npz, allow_pickle=False) as d:
        return int(d["joint_pos"].shape[0])


def latest_run(name: str) -> str | None:
    runs = sorted(EXP.glob(f"*_{name}"), key=lambda p: p.stat().st_mtime)
    return runs[-1].name if runs else None


# ---------------------------------------------------------------------------------------------- python steps
def wait_dance(log, max_h: float = 3.5, chunks: int = 20) -> int:
    """Wait until the detached take102 run ended: all chunk records in its driver log, or its pipeline step ended,
    or its driver process is gone (then the latest checkpoint is evaluated and the log says so)."""
    t0 = time.time()
    while True:
        recs = [json.loads(l) for l in DANCE_DRIVER_LOG.read_text(encoding="utf-8").splitlines()
                if l.startswith('{"chunk"')]
        ended = any(json.loads(l).get("step") == "long_dance" and json.loads(l).get("event") == "end"
                    for l in DANCE_STATE.read_text(encoding="utf-8").splitlines() if l.strip())
        last = recs[-1] if recs else {}
        log.write(f"{time.strftime('%H:%M:%S')} chunks={len(recs)} last_it={last.get('last_it')} "
                  f"last_rc={last.get('rc')} pipeline_step_end={ended}\n")
        log.flush()
        if len(recs) >= chunks or ended:
            bad = [r["chunk"] for r in recs if r.get("rc") != 0]
            log.write(f"DANCE RUN ENDED: {len(recs)} chunk records, non-zero rc chunks {bad}\n")
            return 0
        if time.time() - t0 > max_h * 3600:
            log.write("timeout waiting for the dance run\n")
            return 1
        time.sleep(120)


def media_post(log, tag: str, pairs: list[tuple[str, str, str]], sheets: list[str]) -> int:
    """Contact sheets (fps=2 tile) for ``sheets`` and hstacked side-by-sides for ``pairs`` (left, right, out)."""
    sys.path.insert(0, str(REPO / "source"))
    from dropbear_wbc.tasks.tracking.demo_render import contact_sheet, ffmpeg_exe

    rc = 0
    for name in sheets:
        src = MEDIA / f"{name}.mp4"
        if not src.is_file():
            log.write(f"missing {src}\n")
            rc = 1
            continue
        out = contact_sheet(src, MEDIA / f"{name}_contact_sheet_2fps.png", fps=2.0, cols=8)
        log.write(f"sheet {out}\n")
    for left, right, out in pairs:
        a, b = MEDIA / f"{left}.mp4", MEDIA / f"{right}.mp4"
        if not (a.is_file() and b.is_file()):
            log.write(f"missing {a} or {b}\n")
            rc = 1
            continue
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(a), "-i", str(b), "-filter_complex",
               "[0:v]scale=960:540[l];[1:v]scale=960:540[r];[l][r]hstack=inputs=2:shortest=1", "-c:v", "libx264",
               "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(MEDIA / f"{out}.mp4")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        log.write(f"side-by-side {out}: rc={r.returncode} {r.stderr.strip()}\n")
        rc |= r.returncode
    log.write(f"{tag} media done rc={rc}\n")
    return rc


# ---------------------------------------------------------------------------------------------- plans
def eval_steps(prefix: str, run: str | None, npz: str, frames: int, needs: str, allow_rejected: bool,
               run_name: str | None = None, checkpoint: str | None = None, caption: str | None = None,
               ref_caption: str | None = None) -> list[dict]:
    """Summary, nominal eval (+export), parity, clean policy render, continuous + perturbed evals, reference render,
    media, sim2sim. ``checkpoint`` (e.g. ``model_2300.pt``, relative to the run dir) pins the evaluated model (default:
    the run's highest-iteration checkpoint); ``caption`` / ``ref_caption`` override the burned-in video captions (a
    literal ``\\n`` separates lines)."""
    runsel = ["--load_run", run] if run else []
    if checkpoint and run:
        runsel = ["--checkpoint", f"logs/rsl_rl/dropbear_tracking/{run}/{checkpoint}"]
    rej = ["--allow_rejected_motion"] if allow_rejected else []
    one = str(frames - 1)  # policy steps of one pass from frame 0 (frame f is reached after f steps)
    two = str(2 * (frames - 1))
    play = ["--task", "Dropbear-Tracking-Flat-Play-v0", "--motion_file", npz, *runsel, "--solver_iters", "32", "4", *rej]
    cap = ["--video_caption", caption] if caption else []
    rcap = ["--video_caption", ref_caption] if ref_caption else []
    return [
        {"name": f"{prefix}_summary", "needs": needs, "cmd": [PY, "-u", "tools/summarize_training.py", "RUN_DIR"],
         "run_name": run_name, "log": f"logs/demo_eval/{prefix}_training_summary.log"},
        {"name": f"{prefix}_eval_nominal", "needs": needs, "run_name": run_name,
         "cmd": lock(f"logs/demo_eval/play_{prefix}_nominal_32_4.log", 1800,
                     *isaac("scripts/play.py", *play, "--num_envs", "16", "--steps", one, "--export")),
         "log": f"logs/demo_eval/play_{prefix}_nominal_32_4.wrapper.out"},
        {"name": f"{prefix}_parity", "needs": f"{prefix}_eval_nominal", "run_name": run_name,
         "cmd": [PY, "-u", "tools/check_export_parity.py", "RUN_DIR/exported"],
         "log": f"logs/demo_eval/check_export_parity_{prefix}.log"},
        # policy video = recorded rollout rendered kinematically (tools/render_policy_clean.py; 2026-09-24 22:30: play.py
        # --clean_video renders produced 25-61 % blank frames, kinematic replays none); '@RUN@' is resolved at run time
        {"name": f"{prefix}_render_policy", "needs": needs, "run_name": run_name,
         "cmd": lock(f"logs/demo_eval/render_{prefix}_policy.log", 1800, PY, "-u", "tools/render_policy_clean.py", "--job",
                     json.dumps({"tag": f"{prefix}_policy", "motion_file": npz, "steps": int(one), "solver_iters": [32, 4],
                                 "cams": "track,front,feet", "allow_rejected_motion": allow_rejected,
                                 **({"checkpoint": f"logs/rsl_rl/dropbear_tracking/{run}/{checkpoint}"}
                                    if checkpoint and run else {"load_run": run or "@RUN@"}),
                                 "caption": caption or (f"Dropbear | {Path(npz).stem} | policy @RUN@ (latest checkpoint)\\n"
                                                        "Isaac Lab PhysX 32/4 - nominal - deterministic policy\\n"
                                                        "rendered from the recorded rollout (kinematic replay)")})),
         "log": f"logs/demo_eval/render_{prefix}_policy.wrapper.out"},
        {"name": f"{prefix}_eval_continuous", "needs": needs, "run_name": run_name,
         "cmd": lock(f"logs/demo_eval/play_{prefix}_continuous_32_4.log", 1800,
                     *isaac("scripts/play.py", *play, "--num_envs", "16", "--steps", two, "--continuous_loop")),
         "log": f"logs/demo_eval/play_{prefix}_continuous_32_4.wrapper.out"},
        {"name": f"{prefix}_eval_perturbed", "needs": needs, "run_name": run_name,
         "cmd": lock(f"logs/demo_eval/play_{prefix}_perturbed_32_4.log", 1800,
                     *isaac("scripts/play.py", *play, "--num_envs", "16", "--steps", two, "--continuous_loop",
                            "--perturbed", "--seed", "7")),
         "log": f"logs/demo_eval/play_{prefix}_perturbed_32_4.wrapper.out"},
        {"name": f"{prefix}_render_reference",
         "cmd": lock(f"logs/demo_eval/render_{prefix}_reference.log", 1800,
                     *isaac("scripts/render_reference.py", "--motion_file", npz, "--video_cams", "track,feet",
                            "--video_dir", "logs/demo_eval/media", "--video_tag", f"{prefix}_reference", *rcap)),
         "log": f"logs/demo_eval/render_{prefix}_reference.wrapper.out"},
        {"name": f"{prefix}_media", "func": "media_post",
         "args": {"tag": prefix, "sheets": [f"{prefix}_policy_track", f"{prefix}_policy_front", f"{prefix}_reference_feet"],
                  "pairs": [(f"{prefix}_reference_track", f"{prefix}_policy_track", f"{prefix}_reference_vs_policy_track"),
                            (f"{prefix}_reference_feet", f"{prefix}_policy_feet", f"{prefix}_reference_vs_policy_feet")]},
         "log": f"logs/demo_eval/{prefix}_media.log"},
        {"name": f"{prefix}_sim2sim", "needs": f"{prefix}_eval_nominal", "run_name": run_name,
         "cmd": [BASH, "logs/demo_eval/sim2sim/run_sim2sim_cpu.sh", prefix, "RUN_NAME", npz,
                 f"{(frames - 1) / 50.0 + 1.0:.2f}", "5755", "5756", "--diag-eq-solref", "0.004", "1"],
         "log": f"logs/demo_eval/sim2sim/{prefix}_sim2sim.out"},
    ]


DANCE_CAPTION = ("Dropbear tracking Unitree G1 dance_102 (retargeted), policy model_2300, Isaac Sim\\n"
                 "Isaac Lab PhysX 32/4 - nominal - deterministic policy\\n"
                 "reference FAILS the contact gate (stance feet float up to 12-14 cm)")
DANCE_REF_CAPTION = ("REFERENCE: Unitree G1 dance_102 retargeted to Dropbear (kinematic replay, no physics)\\n"
                     "FAILS the contact gate: stance feet float up to 12-14 cm above the floor")


def plan_take102(a) -> list[dict]:
    """The take102_s8 run ENDED at the 15:35 reboot (last checkpoint model_2300, chunk 20 of 20 incomplete): pass
    ``--skip wait_dance`` (the wait counts chunk records and would block); ``--dance_checkpoint`` pins the model."""
    frames = npz_frames(DANCE_NPZ)
    return [{"name": "wait_dance", "critical": True, "func": "wait_dance", "args": {"max_h": a.max_wait_h},
             "log": "logs/demo_eval/wait_dance.log"}] + eval_steps(
        "take102", DANCE_RUN, DANCE_NPZ, frames, needs="wait_dance", allow_rejected=True, run_name=None,
        checkpoint=a.dance_checkpoint, caption=DANCE_CAPTION, ref_caption=DANCE_REF_CAPTION)


WAVE_NPZ = "data/motions/synthetic/wave_right.npz"


def wave_v1_render_step() -> dict:
    """Clean re-render of the wave_right v1 policy (model_743) with the current framing (track, front, feet)."""
    frames = npz_frames(WAVE_NPZ)
    one = str(frames - 1)
    return {"name": "wave_v1_render_policy",
            "cmd": lock("logs/demo_eval/render_wave_v1_policy.log", 1800, PY, "-u", "tools/render_policy_clean.py", "--job",
                        json.dumps({"tag": "wave_v1_policy", "motion_file": WAVE_NPZ, "load_run": WAVE_RUN,
                                    "steps": int(one), "solver_iters": [32, 4], "cams": "track,front,feet",
                                    "caption": "Dropbear tracking synthetic wave_right (v1), policy wave_right_s8 model_743, "
                                               "Isaac Sim\\nIsaac Lab PhysX 32/4 - nominal - deterministic policy\\n"
                                               "rendered from the recorded rollout (kinematic replay of the simulated state)"})),
            "log": "logs/demo_eval/render_wave_v1_policy.wrapper.out"}


def plan_wave_v2(a) -> list[dict]:
    frames = npz_frames(WAVE2_NPZ)
    train = {"name": "train_wave_v2", "critical": True,
             "cmd": [PY, "-u", "tools/run_chunked_training.py", "--run_name", WAVE2_NAME, "--motion_file", WAVE2_NPZ,
                     "--init_run", WAVE_RUN, "--chunks", str(a.wave_chunks), "--iters_per_chunk", str(a.wave_iters),
                     "--num_envs", "2048", "--solver_iters", "8", "4", "--save_interval", "50", "--timeout", "900",
                     "--seed", "101", "--log_dir", "logs/demo_eval", "--owner", "demo_eval"],
             "log": f"logs/demo_eval/train_{WAVE2_NAME}_driver.log"}
    if a.wave_resume_run:
        train["cmd"][train["cmd"].index("--init_run"):train["cmd"].index("--init_run") + 2] = [
            "--resume_run", a.wave_resume_run]
    return [wave_v1_render_step(), train] + eval_steps("wave_v2", None, WAVE2_NPZ, frames, needs="train_wave_v2",
                                                       allow_rejected=False, run_name=WAVE2_NAME)


def resolve(cmd: list[str], run_name: str | None, default_run: str | None) -> list[str]:
    run = latest_run(run_name) if run_name else default_run
    out = []
    for c in cmd:
        c = c.replace("RUN_DIR", f"logs/rsl_rl/dropbear_tracking/{run}") if "RUN_DIR" in c else c
        c = run if c == "RUN_NAME" else c
        c = c.replace("@RUN@", str(run)) if "@RUN@" in c else c
        out.append(c)
    if run_name and "--load_run" not in out and any("scripts/play.py" in c for c in out):
        i = out.index("--motion_file")
        out[i + 2:i + 2] = ["--load_run", run]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", choices=["take102", "wave_v2", "all"], required=True)
    ap.add_argument("--max_wait_h", type=float, default=3.5)
    ap.add_argument("--dance_checkpoint", default="model_2300.pt",
                    help="take102 checkpoint (relative to the run dir); '' = the run's latest")
    ap.add_argument("--wave_chunks", type=int, default=4)
    ap.add_argument("--wave_iters", type=int, default=150)
    ap.add_argument("--wave_resume_run", default=None, help="resume a broken wave_v2 run dir instead of a warm start")
    ap.add_argument("--skip", default="", help="comma list of step names to skip (resume a broken plan)")
    ap.add_argument("--after_pid", type=int, default=None,
                    help="wait until this process (e.g. another pipeline of this track) has exited before the first step, "
                    "so the two do not race each other for the GPU lock")
    a = ap.parse_args()
    if a.after_pid:
        import ctypes

        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x00100000, False, a.after_pid)  # SYNCHRONIZE
        if h:
            k32.WaitForSingleObject(h, 0xFFFFFFFF)
            k32.CloseHandle(h)
    steps = []
    if a.plan in ("take102", "all"):
        steps += [dict(s, default_run=DANCE_RUN) for s in plan_take102(a)]
    if a.plan in ("wave_v2", "all"):
        steps += plan_wave_v2(a)
    skip = {s for s in a.skip.split(",") if s}
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "sim2sim").mkdir(exist_ok=True)
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
            cmd = resolve(s["cmd"], s.get("run_name"), s.get("default_run")) if "cmd" in s else None
            st.write(json.dumps({"step": s["name"], "event": "begin", "time": time.strftime("%H:%M:%S"), "cmd": cmd,
                                 "log": s["log"]}) + "\n")
            st.flush()
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                if "func" in s:
                    try:
                        rc = globals()[s["func"]](log=log, **s["args"])
                    except Exception as e:  # noqa: BLE001
                        log.write(f"EXCEPTION {e!r}\n")
                        rc = 1
                else:
                    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
                    st.write(json.dumps({"step": s["name"], "child_pid": proc.pid}) + "\n")
                    st.flush()
                    rc = proc.wait()
            ok[s["name"]] = rc == 0
            st.write(json.dumps({"step": s["name"], "event": "end", "rc": rc, "wall_s": round(time.time() - t0, 1),
                                 "time": time.strftime("%H:%M:%S")}) + "\n")
            st.flush()
            if rc != 0 and s.get("critical"):
                st.write(json.dumps({"step": s["name"], "note": "critical step failed; dependent steps skipped"}) + "\n")
                st.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
