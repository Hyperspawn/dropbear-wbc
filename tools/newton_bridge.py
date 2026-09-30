"""Newton <-> ``dropbear_hg-v1`` bridge: Dropbear in simulation behind a Unitree-style low-level API.

The bridge is the "robot side" of CONTRACTS section 6. It binds
``rt/lowstate`` (PUB, ``tcp://127.0.0.1:5556``) and ``rt/lowcmd`` (SUB,
``tcp://127.0.0.1:5555``), publishes a LowState every control tick (500 Hz by
default: ``--sim-dt 0.002 --substeps 1``) and applies the newest LowCmd with the
motor law ``tau = tau_ff + kp*(q* - q) + kd*(dq* - dq)`` (clipped to the effort
limits) on the 22 body motors each physics step. The last command is held
(zero-order hold).

Safety/time semantics (mirroring a Unitree robot):
    * Until the first LowCmd arrives the simulation is paused and tick-0 state is
      published at 50 Hz (``--no-wait-for-cmd`` starts immediately in damping mode).
    * Watchdog: no LowCmd for more than ``--watchdog-ms`` (wall clock) switches all
      motors to damping (``kp = 0``, ``kd = --damping-kd`` or the legacy group kd).
    * ``--realtime`` throttles to wall-clock time when the simulation is faster;
      when it is slower the bridge runs as fast as it can and reports the
      real-time factor. ``--lockstep-ticks N`` instead blocks after every N-th
      tick until a command computed from that tick arrives (deterministic sim2sim).

GPU lock: acquired before the plant is built and always released (CONTRACTS 0). Under
``tools/gpu_lock_run.py`` pass ``--gpu-lock-held`` so the bridge does not wait on its own caller.

Example (hanging bench test, headless, 10 s of simulated time)::

    .venv-newton/Scripts/python.exe tools/newton_bridge.py --fixed-base --duration 10 \
        --report logs/sdk_bridge/bridge_fixed.json --trace logs/sdk_bridge/bridge_fixed_trace.npz
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "third_party" / "pydeps")]

import numpy as np  # noqa: E402

from dropbear_wbc.newton_sim.gpu_lock import GpuLock  # noqa: E402
from dropbear_wbc.sdk import motors  # noqa: E402
from dropbear_wbc.sdk.transport import LOWCMD, LOWSTATE, ChannelPublisher, ChannelSubscriber, Endpoints  # noqa: E402
from dropbear_wbc.sdk.types import LowCmd, LowState, MotorMode  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", type=Path, default=None, help="plant USD (default $DROPBEAR_USD / contract path)")
    ap.add_argument("--fixed-base", action="store_true", help="weld the torso to the world (hanging bench mode)")
    ap.add_argument("--hang-clearance", type=float, default=0.30, help="fixed base: feet clearance [m]")
    ap.add_argument("--viewer", choices=["null", "gl"], default="null")
    ap.add_argument("--render-hz", type=float, default=30.0)
    ap.add_argument("--realtime", action="store_true", help="throttle to wall clock")
    ap.add_argument("--duration", type=float, default=0.0, help="simulated seconds to run (0 = until Ctrl-C)")
    ap.add_argument("--max-wall-s", type=float, default=0.0, help="wall-clock limit after start (0 = none)")
    ap.add_argument("--sim-dt", type=float, default=0.002)
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mujoco-cpu", action="store_true", help="MuJoCo C backend instead of MuJoCo Warp")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--ls-iterations", type=int, default=50)
    ap.add_argument("--collisions", choices=["convex-hull", "box"], default="convex-hull")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cmd-port", type=int, default=5555)
    ap.add_argument("--state-port", type=int, default=5556)
    ap.add_argument("--watchdog-ms", type=float, default=100.0)
    ap.add_argument("--motor-profile", default=None,
                    help="training motor twin in the plant (hw_v1, hw_v1i, hw_v1_knee_x10s2, ...): per-motor armature, "
                         "peak torque, torque-speed envelope and friction (robots.hw_motor_specs.hw_motor_torque); "
                         "default: legacy PD clipped to the legacy effort limits")
    ap.add_argument("--target-ramp-ms", type=float, default=0.0,
                    help="ramp each new position target linearly over this time instead of a step, as the motor side "
                         "must for policies trained with target interpolation (sidecar target_interp_steps x 5 ms, "
                         "e.g. 20 for hw_v1i; dropbear_wbc.sdk.target_ramp). 0 = zero-order hold")
    ap.add_argument("--damping-kd", type=float, default=None, help="damping-mode kd (default: legacy group kd)")
    ap.add_argument("--no-wait-for-cmd", action="store_true")
    ap.add_argument("--wait-timeout-s", type=float, default=300.0)
    ap.add_argument("--lockstep-ticks", type=int, default=0)
    ap.add_argument("--lockstep-timeout-ms", type=float, default=1000.0)
    ap.add_argument("--no-sim-block", action="store_true", help="do not publish privileged sim ground truth")
    ap.add_argument("--sim-bodies", default="", help="comma-separated extra body names for the sim block")
    ap.add_argument("--trace", type=Path, default=None, help="write a per-tick trace (npz)")
    ap.add_argument("--trace-every", type=int, default=5)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--status-every-s", type=float, default=1.0)
    ap.add_argument("--passive-damping", type=float, default=None, help="override passive-joint damping")
    ap.add_argument("--raw-usd", action="store_true",
                    help="skip the CONTRACTS 0.1 in-memory plant fixes (diagnostic only)")
    ap.add_argument("--start-npz", type=Path, default=None,
                    help="free-base sim2sim: pre-settle at the motor pose of this motion NPZ frame (root pinned) "
                         "and start from it (PlantConfig.start_motor_q)")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--presettle-s", type=float, default=1.5)
    ap.add_argument("--diag-eq-solref", type=float, nargs=2, default=None, metavar=("TIMECONST", "DAMPRATIO"),
                    help="DIAGNOSTIC (--mujoco-cpu only): MuJoCo eq_solref for the loop-closure equalities")
    ap.add_argument("--authored-ankle", action="store_true",
                    help="keep the authored revolute ankle tie rods (CONTRACTS 0.2 opt-out; default spherical, "
                         "or $DROPBEAR_AUTHORED_ANKLE=1)")
    ap.add_argument("--no-gpu-lock", action="store_true",
                    help="only for pure-CPU runs: --device cpu --mujoco-cpu with CUDA_VISIBLE_DEVICES=-1")
    ap.add_argument("--gpu-lock-held", action="store_true",
                    help="the caller (tools/gpu_lock_run.py) already holds .locks/gpu.lock; do not acquire it again")
    args = ap.parse_args(argv)
    if args.no_gpu_lock and not (args.device == "cpu" and args.mujoco_cpu
                                 and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"):
        ap.error("--no-gpu-lock requires --device cpu --mujoco-cpu and CUDA_VISIBLE_DEVICES=-1")
    return args


def pct(values, q) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None


class Trace:
    """Decimated per-tick record written as npz."""

    def __init__(self, every: int):
        self.every = every
        self.rows: dict[str, list] = {k: [] for k in (
            "tick", "time_s", "wall_s", "root_pos_w", "root_quat_wxyz", "root_lin_vel_w", "motor_q", "motor_dq",
            "motor_tau", "q_des", "kp", "kd", "damping", "cmd_tick", "neck_q", "body_pos_w", "body_quat_wxyz")}
        self.body_names: tuple[str, ...] = ()

    def add(self, r, wall_s: float, q_des, kp, kd, damping: bool, cmd_tick: int) -> None:
        if r.tick % self.every:
            return
        for k, v in (("tick", r.tick), ("time_s", r.time_s), ("wall_s", wall_s), ("root_pos_w", r.root_pos_w),
                     ("root_quat_wxyz", r.root_quat_wxyz), ("root_lin_vel_w", r.root_lin_vel_w),
                     ("motor_q", r.motor_q), ("motor_dq", r.motor_dq), ("motor_tau", r.motor_tau),
                     ("q_des", q_des), ("kp", kp), ("kd", kd), ("damping", damping), ("cmd_tick", cmd_tick),
                     ("neck_q", r.neck_q), ("body_pos_w", r.body_pos_w), ("body_quat_wxyz", r.body_quat_wxyz)):
            self.rows[k].append(np.array(v, copy=True))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, motor_names=np.array(motors.MOTOR_NAMES), body_names=np.array(self.body_names),
                            **{k: np.asarray(v) for k, v in self.rows.items()})


def _npz_motor_pose(path: Path, frame: int) -> tuple[float, ...]:
    """22 motor positions (motor-contract order) of ``frame`` of a ``dropbear-motion-npz-v1`` file."""
    import numpy as np

    from dropbear_wbc.sdk import motors as _motors

    with np.load(path, allow_pickle=False) as d:
        names = [str(n) for n in d["joint_names"]]
        q = np.asarray(d["joint_pos"][frame], dtype=float)
    return tuple(float(q[names.index(n)]) for n in _motors.MOTOR_NAMES)


def main(argv=None) -> int:
    args = parse_args(argv)
    report: dict = {"status": "failed", "args": {k: str(v) for k, v in vars(args).items()}}
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    lock = GpuLock(owner="sdk_bridge/newton_bridge")
    trace = Trace(args.trace_every) if args.trace else None
    pub = sub = viewer = None
    try:
        if args.gpu_lock_held:
            if not lock.path.exists():
                raise RuntimeError(f"--gpu-lock-held given but {lock.path} does not exist")
            report["gpu_lock"] = "held by caller: " + lock.path.read_text(encoding="utf-8", errors="replace")
        elif not args.no_gpu_lock:
            lock.acquire()
            report["gpu_lock_wait_s"] = lock.waited_s
        from dropbear_wbc.newton_sim.imu import LowStateAssembler
        from dropbear_wbc.newton_sim.plant import DEFAULT_USD, DropbearNewtonPlant, PlantConfig

        bodies = tuple(b for b in args.sim_bodies.split(",") if b)
        cfg = PlantConfig(usd_path=args.usd or DEFAULT_USD, fixed_base=args.fixed_base,
                          hang_clearance=args.hang_clearance, sim_dt=args.sim_dt, substeps=args.substeps,
                          device=args.device, mujoco_cpu=args.mujoco_cpu, use_cuda_graph=not args.no_graph,
                          iterations=args.iterations, ls_iterations=args.ls_iterations, collisions=args.collisions,
                          extra_bodies=bodies, usd_fixes=not args.raw_usd, motor_profile=args.motor_profile,
                          authored_ankle_tierods=True if args.authored_ankle else None,
                          start_motor_q=_npz_motor_pose(args.start_npz, args.start_frame) if args.start_npz else None,
                          presettle_s=args.presettle_s,
                          diag_eq_solref=tuple(args.diag_eq_solref) if args.diag_eq_solref else None)
        if args.passive_damping is not None:
            cfg.passive_damping = args.passive_damping
        t_build = time.perf_counter()
        plant = DropbearNewtonPlant(cfg, log=lambda m: print(m, flush=True))
        if trace is not None:
            trace.body_names = tuple(plant.extra_body_names)
        report["build_s"] = time.perf_counter() - t_build
        report["plant"] = plant.report
        tick_dt = cfg.tick_dt
        asm = LowStateAssembler(tick_dt, publish_sim=not args.no_sim_block, body_names=bodies)

        if args.viewer == "gl":
            import newton
            viewer = newton.viewer.ViewerGL()
            viewer.set_model(plant.model)

        ep = Endpoints(args.host, args.cmd_port, args.state_port)
        pub = ChannelPublisher(LOWSTATE, LowState, endpoints=ep, role="robot")
        sub = ChannelSubscriber(LOWCMD, LowCmd, endpoints=ep, role="robot")
        pub.Init()
        sub.Init()
        print(f"[bridge] rt/lowstate PUB {ep.for_channel(LOWSTATE)} | rt/lowcmd SUB {ep.for_channel(LOWCMD)} | "
              f"tick {1e3 * tick_dt:.1f} ms | fixed_base={cfg.fixed_base}", flush=True)

        n = motors.NUM_MOTORS
        damping_kd = (np.full(n, args.damping_kd) if args.damping_kd is not None
                      else np.asarray(motors.DEFAULT_KD, float))
        cur = {"q": np.zeros(n), "dq": np.zeros(n), "tau": np.zeros(n), "kp": np.zeros(n), "kd": damping_kd.copy(),
               "mode": np.ones(n, np.int32)}
        damping = True
        plant.set_motor_command(cur["q"], cur["dq"], cur["tau"], cur["kp"], cur["kd"], cur["mode"])
        r0 = plant.initial_readout()
        from dropbear_wbc.sdk.target_ramp import TargetRamp

        ramp = TargetRamp.from_ms(args.target_ramp_ms, tick_dt)
        report["target_ramp"] = {"ms": args.target_ramp_ms, "ticks": ramp.n}

        # ---------------------------------------------------------------- wait for the first command
        first_cmd: LowCmd | None = None
        if not args.no_wait_for_cmd:
            print("[bridge] paused: waiting for the first LowCmd", flush=True)
            t_wait = time.perf_counter()
            while first_cmd is None and not stop["flag"]:
                pub.Write(asm.build(r0, cur["mode"]))
                first_cmd = sub.Read(timeout=0.02)
                if time.perf_counter() - t_wait > args.wait_timeout_s:
                    raise TimeoutError(f"no LowCmd within {args.wait_timeout_s} s")
            report["waited_for_first_cmd_s"] = time.perf_counter() - t_wait
            asm = LowStateAssembler(tick_dt, publish_sim=not args.no_sim_block, body_names=bodies)

        # ---------------------------------------------------------------- main loop
        tick_wall, step_ms, cmd_lat_ticks, cmd_lat_ms, watchdog_events = [], [], [], [], []
        cmds_applied, last_cmd_tick = 0, -1
        last_cmd_wall = time.perf_counter()
        t_start = time.perf_counter()
        next_status = t_start + args.status_every_s
        next_render = t_start
        status_ticks0, status_t0 = 0, t_start
        lockstep_timeouts = 0

        def apply(cmd: LowCmd) -> None:
            nonlocal damping, cmds_applied, last_cmd_tick, last_cmd_wall
            now_ns = time.perf_counter_ns()
            m = cmd.motor
            cur.update(q=m.q.astype(float), dq=m.dq.astype(float), tau=m.tau.astype(float), kp=m.kp.astype(float),
                       kd=m.kd.astype(float), mode=(m.mode != MotorMode.DISABLE).astype(np.int32))
            q_now = ramp.on_command(cur["q"], snap=damping)  # cur["q"] = the commanded target, q_now = applied
            plant.set_motor_command(q_now, cur["dq"], cur["tau"], cur["kp"], cur["kd"], cur["mode"])
            if cmd.neck is not None:
                plant.set_neck_targets(cmd.neck.q.astype(float))
            if damping and watchdog_events and watchdog_events[-1].get("recovered_tick") is None:
                watchdog_events[-1]["recovered_tick"] = plant.tick
            damping = False
            cmds_applied += 1
            last_cmd_tick = cmd.tick
            last_cmd_wall = time.perf_counter()
            cmd_lat_ticks.append(plant.tick - cmd.tick)
            if cmd.stamp_ns:
                cmd_lat_ms.append((now_ns - cmd.stamp_ns) * 1e-6)

        if first_cmd is not None:
            apply(first_cmd)
        r = r0
        while not stop["flag"]:
            if args.duration and plant.time_s >= args.duration - 1e-9:
                break
            if args.max_wall_s and time.perf_counter() - t_start > args.max_wall_s:
                report["stopped_by"] = "max_wall_s"
                break
            t_tick = time.perf_counter()
            cmd = sub.Read(timeout=0.0)
            if cmd is not None:
                apply(cmd)
            elif not damping and (time.perf_counter() - last_cmd_wall) * 1e3 > args.watchdog_ms:
                damping = True
                ramp.cancel()
                cur.update(kp=np.zeros(n), kd=damping_kd.copy(), tau=np.zeros(n), dq=np.zeros(n),
                           mode=np.ones(n, np.int32))
                plant.set_motor_command(cur["q"], cur["dq"], cur["tau"], cur["kp"], cur["kd"], cur["mode"])
                watchdog_events.append({"tick": plant.tick, "time_s": plant.time_s,
                                        "silence_ms": (time.perf_counter() - last_cmd_wall) * 1e3,
                                        "recovered_tick": None})
                print(f"[bridge] WATCHDOG: no LowCmd for {args.watchdog_ms:.0f} ms at t={plant.time_s:.3f}s -> "
                      "damping mode", flush=True)
            q_ramp = ramp.tick() if not damping else None
            if q_ramp is not None:
                plant.set_motor_command(q_ramp, cur["dq"], cur["tau"], cur["kp"], cur["kd"], cur["mode"])
            a = time.perf_counter()
            plant.step()
            r = plant.readout()
            step_ms.append((time.perf_counter() - a) * 1e3)
            if not (np.isfinite(r.motor_q).all() and np.isfinite(r.root_pos_w).all()):
                raise FloatingPointError(f"non-finite plant state at tick {r.tick}")
            pub.Write(asm.build(r, cur["mode"]))
            if trace is not None:
                trace.add(r, time.perf_counter() - t_start, cur["q"], cur["kp"], cur["kd"], damping, last_cmd_tick)
            # Lockstep: block for the command computed from this tick, unless the watchdog already
            # declared the client gone (damping), so a vanished client cannot stall the simulation.
            if args.lockstep_ticks and r.tick % args.lockstep_ticks == 0 and not damping:
                deadline = time.perf_counter() + args.lockstep_timeout_ms * 1e-3
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        lockstep_timeouts += 1
                        break
                    c = sub.Read(timeout=remaining)
                    if c is not None:
                        apply(c)
                        if c.tick >= r.tick:
                            break
            if viewer is not None and time.perf_counter() >= next_render:
                viewer.begin_frame(plant.time_s)
                viewer.log_state(plant.current_state())
                viewer.end_frame()
                next_render += 1.0 / args.render_hz
                if not viewer.is_running():
                    break
            if args.realtime:
                target = t_start + plant.tick * tick_dt
                while (slack := target - time.perf_counter()) > 0:
                    time.sleep(slack - 3e-4) if slack > 1e-3 else None
            tick_wall.append(time.perf_counter() - t_tick)
            now = time.perf_counter()
            if now >= next_status:
                hz = (plant.tick - status_ticks0) / (now - status_t0)
                print(f"[bridge] t_sim={plant.time_s:7.3f}s tick={plant.tick} {hz:6.1f} Hz "
                      f"RTF={hz * tick_dt:5.2f} root_z={r.root_pos_w[2]:+.3f} cmds={cmds_applied} "
                      f"damping={damping}", flush=True)
                status_ticks0, status_t0 = plant.tick, now
                next_status = now + args.status_every_s

        wall = time.perf_counter() - t_start
        tw = np.asarray(tick_wall) * 1e3
        report.update({
            "status": "executed",
            "ticks": plant.tick, "sim_time_s": plant.time_s, "wall_s": wall,
            "bridge_hz": plant.tick / wall if wall > 0 else None,
            "realtime_factor": plant.time_s / wall if wall > 0 else None,
            "tick_wall_ms": {"mean": float(tw.mean()) if len(tw) else None, "p50": pct(tw, 50), "p99": pct(tw, 99),
                             "max": float(tw.max()) if len(tw) else None},
            "physics_step_ms": {"mean": float(np.mean(step_ms)) if step_ms else None, "p50": pct(step_ms, 50),
                                "p99": pct(step_ms, 99)},
            "commands": {"applied": cmds_applied, "latency_ticks": {"mean": float(np.mean(cmd_lat_ticks))
                         if cmd_lat_ticks else None, "p50": pct(cmd_lat_ticks, 50), "p99": pct(cmd_lat_ticks, 99),
                         "max": int(np.max(cmd_lat_ticks)) if cmd_lat_ticks else None},
                         "transport_ms": {"p50": pct(cmd_lat_ms, 50), "p99": pct(cmd_lat_ms, 99),
                                          "max": float(np.max(cmd_lat_ms)) if cmd_lat_ms else None},
                         "subscriber": sub.stats},
            "publisher": pub.stats, "watchdog_events": watchdog_events, "lockstep_timeouts": lockstep_timeouts,
            "final": {"root_pos_w": r.root_pos_w.tolist(), "root_quat_wxyz": r.root_quat_wxyz.tolist(),
                      "damping": damping, "finite": bool(plant.is_finite()),
                      "max_closure_residual_m": float(plant.closure_residuals_m().max())},
        })
    except Exception:  # noqa: BLE001
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        for ch in (pub, sub):
            if ch is not None:
                ch.Close()
        if viewer is not None:
            viewer.close()
        lock.release()
        if trace is not None:
            trace.save(args.trace)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({k: report.get(k) for k in ("status", "ticks", "sim_time_s", "wall_s", "bridge_hz",
                                                      "realtime_factor", "error")}, default=str), flush=True)
    return 0 if report["status"] == "executed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
