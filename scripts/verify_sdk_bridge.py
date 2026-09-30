"""End-to-end verification of the dropbear_hg SDK: Newton bridge + policy runner in 'hold' mode.

Cases:
    * ``fixed``: ``--fixed-base`` (hanging). Passive 0.1 s -> MoveToDefault 2 s -> Hold.
      Reports per-motor joint tracking error over the last second of Hold.
    * ``free``: free base on the ground, same FSM. No balance policy exists, so the
      robot is expected to fall; reports the time of fall honestly.

Bridge and runner run as separate processes over ZMQ (the bridge takes the GPU
lock itself). Their logs, reports and the bridge trace go to ``logs/sdk_bridge/``;
the merged verdict goes to ``logs/sdk_bridge/verify_<case>_summary.json``.

Example::

    .venv-newton/Scripts/python.exe scripts/verify_sdk_bridge.py --case fixed --hold-s 5

Under the team GPU wrapper (absolute paths; the bridge must not re-acquire its caller's lock)::

    python tools/gpu_lock_run.py --owner sdk_bridge --log logs/sdk_bridge/verify_x.log --timeout 600 --         .venv-newton/Scripts/python.exe -u scripts/verify_sdk_bridge.py         --case fixed --hold-s 3 --tag _x "--bridge-extra=--gpu-lock-held"
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGS = ROOT / "logs" / "sdk_bridge"
READY_MARK = "[bridge] paused: waiting for the first LowCmd"


def tilt_of(qwxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(qwxyz, -1, 0)
    zz = 1.0 - 2.0 * (x * x + y * y)  # world-z component of the body z axis
    return np.arccos(np.clip(zz, -1.0, 1.0))


def analyse_trace(path: Path, move_end_s: float, hold_window_s: float, tag: str) -> dict:
    d = np.load(path)
    t = d["time_s"]
    names = [str(n) for n in d["motor_names"]]
    err = d["motor_q"] - d["q_des"]
    out: dict = {"trace": str(path), "samples": int(len(t)), "t_end_s": float(t[-1]) if len(t) else None}
    hold = (t >= move_end_s) & (~d["damping"].astype(bool))
    if hold.any():
        th = t[hold]
        win = hold & (t >= th[-1] - hold_window_s)
        e = err[win]
        out["hold_window"] = [float(t[win][0]), float(t[win][-1])]
        out["per_motor_rms_rad"] = dict(zip(names, np.sqrt((e ** 2).mean(0)).round(6).tolist()))
        out["per_motor_mean_rad"] = dict(zip(names, e.mean(0).round(6).tolist()))
        out["per_motor_max_abs_rad"] = dict(zip(names, np.abs(e).max(0).round(6).tolist()))
        out["overall_rms_rad"] = float(np.sqrt((e ** 2).mean()))
        out["overall_max_abs_rad"] = float(np.abs(e).max())
        out["worst_motor"] = names[int(np.abs(e).max(0).argmax())]
        tau = d["motor_tau"][win]
        out["per_motor_mean_abs_tau_nm"] = dict(zip(names, np.abs(tau).mean(0).round(3).tolist()))
    tilt = tilt_of(d["root_quat_wxyz"])
    z = d["root_pos_w"][:, 2]
    out["root_z_start_end_m"] = [float(z[0]), float(z[-1])] if len(z) else None
    out["tilt_max_rad"] = float(tilt.max()) if len(tilt) else None
    for thr in (0.5, 1.0):
        idx = np.nonzero(tilt > thr)[0]
        out[f"first_tilt_gt_{thr}_s"] = float(t[idx[0]]) if len(idx) else None
    if len(z):
        dz = z - z[0]
        idx = np.nonzero(dz < -0.15)[0]
        out["first_root_drop_gt_0.15m_s"] = float(t[idx[0]]) if len(idx) else None
    # Downsampled time series for the record (every ~0.1 s).
    step = max(1, int(round(0.1 / max(np.median(np.diff(t)), 1e-6)))) if len(t) > 1 else 1
    out["series_0p1s"] = {"t": t[::step].round(3).tolist(), "root_z": z[::step].round(4).tolist(),
                          "tilt_rad": tilt[::step].round(4).tolist(),
                          "max_abs_err_rad": np.abs(err[::step]).max(1).round(4).tolist()}
    return out


def run_case(case: str, a: argparse.Namespace) -> dict:
    tag = f"{case}{a.tag}"
    passive, move = 0.1, 2.0
    runner_dur = passive + move + a.hold_s + (a.policy_s if a.runner_mode == "policy" else 0.0)
    bridge_dur = runner_dur + 0.4  # bridge outlives the runner -> exercises the watchdog at the end
    bridge_cmd = [PY, "-u", str(ROOT / "tools" / "newton_bridge.py"), "--duration", f"{bridge_dur}",
                  "--report", str(LOGS / f"bridge_{tag}.json"), "--trace", str(LOGS / f"bridge_{tag}_trace.npz"),
                  "--trace-every", "5", "--sim-dt", str(a.sim_dt), "--substeps", str(a.substeps),
                  "--device", a.device] + (["--fixed-base"] if case == "fixed" else []) + \
        (["--realtime"] if a.realtime else []) + (["--mujoco-cpu"] if a.mujoco_cpu else []) + \
        (["--device", "cpu", "--mujoco-cpu", "--no-gpu-lock"] if a.cpu_only else []) + shlex.split(a.bridge_extra)
    runner_cmd = [PY, "-u", str(ROOT / "tools" / "policy_runner.py"), "--mode", a.runner_mode,
                  "--duration", f"{runner_dur}", "--hold-s", f"{a.hold_s}",
                  "--passive-s", f"{passive}", "--move-s", f"{move}", "--clock", a.clock,
                  "--tick-dt", str(a.sim_dt * a.substeps), "--log", str(LOGS / f"runner_{tag}.jsonl"),
                  "--summary", str(LOGS / f"runner_{tag}.json"), "--connect-timeout-s", "120"] + shlex.split(a.runner_extra)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"bridge_{tag}.log", "w", encoding="utf-8") as blog, \
            open(LOGS / f"runner_{tag}.log", "w", encoding="utf-8") as rlog:
        blog.write("$ " + " ".join(bridge_cmd) + "\n")
        blog.flush()
        rlog.write("$ " + " ".join(runner_cmd) + "\n")
        rlog.flush()
        t0 = time.time()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1") if a.cpu_only else None
        bridge = subprocess.Popen(bridge_cmd, stdout=blog, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        # Start the runner only once the bridge holds the GPU lock and has built the plant: the lock
        # may be held by another agent for a long time (CONTRACTS 0: poll up to 60 min).
        ready = False
        while bridge.poll() is None and time.time() - t0 < a.timeout_s:
            if READY_MARK in (LOGS / f"bridge_{tag}.log").read_text(encoding="utf-8", errors="replace"):
                ready = True
                break
            time.sleep(1.0)
        if not ready:
            bridge.kill()
            raise RuntimeError(f"bridge never became ready (rc={bridge.poll()}); see bridge_{tag}.log")
        t_ready = time.time()
        runner = subprocess.Popen(runner_cmd, stdout=rlog, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            rc_runner = runner.wait(timeout=a.timeout_s)
            rc_bridge = bridge.wait(timeout=a.timeout_s)
        finally:
            for p in (runner, bridge):
                if p.poll() is None:
                    p.kill()
    res: dict = {"case": case, "tag": tag, "wall_s": time.time() - t0, "bridge_ready_after_s": t_ready - t0,
                 "rc_bridge": rc_bridge,
                 "rc_runner": rc_runner, "bridge_cmd": bridge_cmd, "runner_cmd": runner_cmd}
    try:
        res["bridge"] = json.loads((LOGS / f"bridge_{tag}.json").read_text(encoding="utf-8"))
        res["runner"] = json.loads((LOGS / f"runner_{tag}.json").read_text(encoding="utf-8"))
        res["trace_analysis"] = analyse_trace(LOGS / f"bridge_{tag}_trace.npz", passive + move, 1.0, tag)
    except (OSError, ValueError, KeyError) as e:
        res["analysis_error"] = repr(e)
    b, r = res.get("bridge", {}), res.get("runner", {})
    res["headline"] = {
        "bridge_hz": b.get("bridge_hz"), "realtime_factor": b.get("realtime_factor"),
        "tick_wall_ms": b.get("tick_wall_ms"), "physics_step_ms": b.get("physics_step_ms"),
        "cmd_latency_ticks": (b.get("commands") or {}).get("latency_ticks"),
        "cmd_transport_ms": (b.get("commands") or {}).get("transport_ms"),
        "watchdog_events": b.get("watchdog_events"),
        "runner_hz_wall": r.get("runner_hz_wall"), "runner_ticks_per_step": r.get("ticks_per_step"),
        "state_age_ms": r.get("state_age_ms"), "runner_fall": r.get("fall"),
        "runner_hold_tracking_overall_rms_rad": (r.get("hold_tracking") or {}).get("overall_rms_rad"),
        "trace_hold_overall_rms_rad": res.get("trace_analysis", {}).get("overall_rms_rad"),
        "trace_hold_overall_max_abs_rad": res.get("trace_analysis", {}).get("overall_max_abs_rad"),
        "trace_first_tilt_gt_1.0_s": res.get("trace_analysis", {}).get("first_tilt_gt_1.0_s"),
        "final_root_pos_w": (b.get("final") or {}).get("root_pos_w"),
    }
    (LOGS / f"verify_{tag}_summary.json").write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"case": tag, "rc": [rc_bridge, rc_runner], **res["headline"]}, indent=2, default=str),
          flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=["fixed", "free", "all"], default="all")
    ap.add_argument("--hold-s", type=float, default=5.0, help="seconds in Hold (before Policy in policy mode)")
    ap.add_argument("--sim-dt", type=float, default=0.002)
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mujoco-cpu", action="store_true")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--clock", choices=["sim", "wall"], default="sim")
    ap.add_argument("--tag", default="", help="suffix for log names")
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--cpu-only", action="store_true",
                    help="MuJoCo C on CPU with CUDA hidden (no GPU, so no GPU lock); slower than real time")
    ap.add_argument("--runner-mode", choices=["hold", "policy"], default="hold")
    ap.add_argument("--policy-s", type=float, default=5.0, help="policy mode: seconds in Policy after Hold")
    ap.add_argument("--bridge-extra", default="", help="extra bridge args (one quoted string)")
    ap.add_argument("--runner-extra", default="", help="extra runner args (one quoted string)")
    a = ap.parse_args()
    cases = ["fixed", "free"] if a.case == "all" else [a.case]
    results = [run_case(c, a) for c in cases]
    return 0 if all(r["rc_bridge"] == 0 and r["rc_runner"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
