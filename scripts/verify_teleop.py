"""End-to-end teleop session: Newton bridge (fixed base, real time) + ``tools/teleop_arm.py`` as two processes.

The bridge publishes ground-truth hand-plate poses (``--sim-bodies``) so the summary can compare the IK model's
wrist (FK of the measured joints) with the simulated wrist. Logs and the merged summary go to ``logs/teleop/``;
the recording goes to ``data/teleop/<session>/``.

Examples::

    # CPU (MuJoCo C, CUDA hidden, no GPU lock needed)
    .venv-teleop/Scripts/python.exe scripts/verify_teleop.py --backend cpu --tag cpu_fig8 --duration 25
    # GPU under the team lock (the bridge must not take the lock again)
    python tools/gpu_lock_run.py --owner teleop --log logs/teleop/gpu_fig8.wrapper.log --timeout 900 -- \
        .venv-teleop/Scripts/python.exe -u scripts/verify_teleop.py \
        --backend gpu --gpu-lock-held --tag gpu_fig8 --duration 25
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs" / "teleop"
NEWTON_PY = ROOT / ".venv-newton" / "Scripts" / "python.exe"
TELEOP_PY = ROOT / ".venv-teleop" / "Scripts" / "python.exe"
READY_MARK = "[bridge] paused: waiting for the first LowCmd"
TELEOP_MARK = "s TELEOP ("  # "[teleop] t=3.02s TELEOP (tracking)"
SIM_BODIES = "LH_shoulder_ex_al_interface_1,RH_shoulder_ex_al_interface_1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--gpu-lock-held", action="store_true", help="caller (gpu_lock_run.py) holds .locks/gpu.lock")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--duration", type=float, default=25.0, help="TELEOP phase length [s]")
    ap.add_argument("--session", default=None, help="recording name (default <date>_<tag>)")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--ports", type=int, nargs=2, default=(5555, 5556), metavar=("CMD", "STATE"),
                    help="ZMQ ports (use non-default ones when another bridge may be running)")
    ap.add_argument("--xr-sim", choices=["off", "controllers", "hands"], default="off",
                    help="drive tools/teleop_arm.py --device webxr with tools/sim_xr_client.py (simulated headset)")
    ap.add_argument("--xr-port", type=int, default=8012)
    ap.add_argument("--kb-sim", action="store_true",
                    help="drive tools/teleop_arm.py --device keyboard (stdin backend) with a scripted key sequence")
    ap.add_argument("--xr-keys", action="store_true",
                    help="with --xr-sim hands: start / calibrate / stop from terminal keys (stdin) instead of --auto-start")
    ap.add_argument("--bridge-extra", default="", help="extra bridge args (one quoted string)")
    ap.add_argument("--teleop-extra", default="", help="extra teleop_arm.py args (one quoted string)")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    tag = a.tag
    session = a.session or f"{dt.datetime.now():%Y-%m-%d_%H-%M}_{tag}"
    rec_dir = ROOT / "data" / "teleop" / session
    bridge_dur = a.duration + 2.0 + 1.0 + 1.5 + 2.0
    bridge_cmd = [str(NEWTON_PY), "-u", str(ROOT / "tools" / "newton_bridge.py"), "--fixed-base", "--realtime",
                  "--duration", f"{bridge_dur}", "--sim-bodies", SIM_BODIES,
                  "--report", str(LOGS / f"bridge_{tag}.json"), "--trace", str(LOGS / f"bridge_{tag}_trace.npz"),
                  "--trace-every", "5", "--cmd-port", str(a.ports[0]), "--state-port", str(a.ports[1])]
    env = None
    if a.backend == "cpu":
        bridge_cmd += ["--device", "cpu", "--mujoco-cpu", "--no-gpu-lock"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    elif a.gpu_lock_held:
        bridge_cmd += ["--gpu-lock-held"]
    bridge_cmd += shlex.split(a.bridge_extra)
    device = "keyboard" if a.kb_sim else ("scripted" if a.xr_sim == "off" else "webxr")
    teleop_cmd = [str(TELEOP_PY), "-u", str(ROOT / "tools" / "teleop_arm.py"), "--device", device,
                  "--duration", f"{a.duration}", "--summary", str(LOGS / f"teleop_{tag}.json"),
                  "--connect-timeout-s", "120", "--cmd-port", str(a.ports[0]), "--state-port", str(a.ports[1])]
    if not a.no_record:
        teleop_cmd += ["--record", str(rec_dir)]
    xr_cmd = None
    if a.xr_sim != "off":
        teleop_cmd += ["--xr-port", str(a.xr_port), "--clock", "wall",
                       "--task", f"dropbear arm teleop (simulated WebXR {a.xr_sim})"]
        if a.xr_sim == "controllers":
            teleop_cmd += ["--xr-controllers"]
        elif a.xr_keys:
            teleop_cmd += ["--keyboard-backend", "stdin"]
        else:
            teleop_cmd += ["--auto-start"]
        # the simulated operator: calibrate at 1 s, start at 1.5 s, stop (both thumbsticks) at duration - 2 s
        xr_cmd = [str(TELEOP_PY), "-u", str(ROOT / "tools" / "sim_xr_client.py"), "--port", str(a.xr_port),
                  "--mode", a.xr_sim, "--duration", f"{a.duration + 0.5}", "--stop-at", f"{a.duration - 2.0}",
                  "--summary", str(LOGS / f"xrclient_{tag}.json")]
    if a.kb_sim:
        teleop_cmd += ["--keyboard-backend", "stdin", "--clock", "wall", "--task", "dropbear arm teleop (scripted keys)"]
    # key script: (time after connect [s], keys). r = start; w/s a/d q/e jog the left hand 1 cm per key; i/k j/l u/o
    # the right hand; [ ] ; ' wrist roll; h = home; x = stop
    kb_script = [(1.0, "r"), (1.5, "wwww"), (2.0, "wwww"), (2.5, "iiii"), (3.0, "iiii"), (4.0, "qqqq"),
                 (4.5, "uuuu"), (5.5, "aaaa"), (6.0, "llll"), (7.0, "[[[["), (7.5, ";;;;"), (9.0, "h"),
                 (11.0, "eeeeee"), (11.5, "oooooo"), (13.0, "x")]
    teleop_cmd += shlex.split(a.teleop_extra)
    t0 = time.time()
    with open(LOGS / f"bridge_{tag}.log", "w", encoding="utf-8") as blog, \
            open(LOGS / f"teleop_{tag}.log", "w", encoding="utf-8") as tlog:
        blog.write("$ " + " ".join(bridge_cmd) + "\n")
        blog.flush()
        tlog.write("$ " + " ".join(teleop_cmd) + "\n")
        tlog.flush()
        bridge = subprocess.Popen(bridge_cmd, stdout=blog, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        ready = False
        while bridge.poll() is None and time.time() - t0 < a.timeout_s:
            if READY_MARK in (LOGS / f"bridge_{tag}.log").read_text(encoding="utf-8", errors="replace"):
                ready = True
                break
            time.sleep(1.0)
        if not ready:
            bridge.kill()
            print(f"bridge never became ready (rc={bridge.poll()}); see {LOGS / f'bridge_{tag}.log'}", flush=True)
            return 2
        t_ready = time.time()
        piped = a.kb_sim or a.xr_keys
        teleop = subprocess.Popen(teleop_cmd, stdout=tlog, stderr=subprocess.STDOUT, cwd=ROOT,
                                  stdin=subprocess.PIPE if piped else None, text=piped or None)
        tlog_path = LOGS / f"teleop_{tag}.log"
        if a.kb_sim:
            while teleop.poll() is None and TELEOP_MARK not in tlog_path.read_text(encoding="utf-8", errors="replace"):
                time.sleep(0.1)
            tk = time.time()
            for at, keys in kb_script:
                while time.time() - tk < at:
                    time.sleep(0.02)
                teleop.stdin.write(keys + chr(10))
                teleop.stdin.flush()
        xr = None
        if xr_cmd is not None:
            # start the simulated operator when the TELEOP phase begins, so its script (calibrate at 1 s, start at
            # 1.5 s, stop at duration - 2 s) is aligned with the teleop timeline
            while teleop.poll() is None and TELEOP_MARK not in tlog_path.read_text(encoding="utf-8", errors="replace"):
                time.sleep(0.2)
            xlog = open(LOGS / f"xrclient_{tag}.log", "w", encoding="utf-8")
            xlog.write("$ " + " ".join(xr_cmd) + chr(10))
            xlog.flush()
            xr = subprocess.Popen(xr_cmd, stdout=xlog, stderr=subprocess.STDOUT, cwd=ROOT)
            if a.xr_keys:  # terminal keys aligned with the simulated operator: c at 1.0 s, r at 1.5 s, x at end - 2 s
                tk = time.time()
                for at, k in ((1.0, "c"), (1.5, "r"), (a.duration - 2.0, "x")):
                    while time.time() - tk < at:
                        time.sleep(0.02)
                    teleop.stdin.write(k + chr(10))
                    teleop.stdin.flush()
        try:
            rc_t = teleop.wait(timeout=a.timeout_s)
            rc_b = bridge.wait(timeout=a.timeout_s)
            if xr is not None:
                xr.wait(timeout=60)
        finally:
            for p in (teleop, bridge, xr):
                if p is not None and p.poll() is None:
                    p.kill()
    res: dict = {"tag": tag, "backend": a.backend, "wall_s": time.time() - t0, "bridge_ready_after_s": t_ready - t0,
                 "rc_bridge": rc_b, "rc_teleop": rc_t, "bridge_cmd": bridge_cmd, "teleop_cmd": teleop_cmd,
                 "recording": None if a.no_record else str(rec_dir)}
    try:
        b = json.loads((LOGS / f"bridge_{tag}.json").read_text(encoding="utf-8"))
        tsum = json.loads((LOGS / f"teleop_{tag}.json").read_text(encoding="utf-8"))
        res["bridge"] = {k: b.get(k) for k in ("status", "bridge_hz", "realtime_factor", "tick_wall_ms",
                                                 "physics_step_ms", "commands", "watchdog_events", "final", "plant",
                                                 "closure_residual_m", "authored_ankle_tierods")}
        res["teleop"] = {k: tsum.get(k) for k in ("status", "loop", "steps", "keepalives", "loop_wall_period_ms",
                                                   "transitions", "gravity_ff", "error", "events", "device")}
        res["analysis"] = tsum.get("analysis")
    except (OSError, ValueError) as e:
        res["merge_error"] = repr(e)
    (LOGS / f"verify_{tag}_summary.json").write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    an = res.get("analysis") or {}
    head = {"rc": [rc_b, rc_t], "bridge_hz": (res.get("bridge") or {}).get("bridge_hz"),
            "cmd_latency_ticks": ((res.get("bridge") or {}).get("commands") or {}).get("latency_ticks")}
    for s in ("left", "right"):
        if s in an:
            head[s] = {"fk_vs_target": an[s]["wrist_tracking_fk_vs_target"],
                       "sim_vs_target": an[s].get("wrist_tracking_sim_vs_target"),
                       "model_sim_vs_fk": an[s].get("model_sim_vs_fk"), "lag": an[s]["lag_fk_vs_target"]}
    print(json.dumps(head, indent=1, default=str), flush=True)
    return 0 if rc_b == 0 and rc_t == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
