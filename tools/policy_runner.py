"""Unitree-deploy-style policy runner for Dropbear over ``dropbear_hg-v1`` (client side).

Connects to ``rt/lowstate`` / ``rt/lowcmd`` (CONTRACTS section 6) and runs the FSM
``Passive -> MoveToDefault (2 s) -> Hold -> Policy`` of
:mod:`dropbear_wbc.deploy.fsm` at the control rate (``step_dt``, 0.02 s).

Modes:
    * ``hold``: no ONNX. Passive for ``--passive-s``, MoveToDefault, then Hold for the
      rest of ``--duration``. Default pose: ``--default-pose legacy`` (DROPBEAR_CFG
      init pose) or a semantic calibration JSON (``standing_motor_pos``).
    * ``policy``: needs ``--deploy-yaml`` (unitree_rl_lab ``deploy.yaml`` format) or
      ``--sidecar`` (``dropbear-policy-sidecar-v1`` JSON next to an ONNX export),
      and switches Hold -> Policy after ``--hold-s``.
    * Reference for tracking policies: embedded in the export (``policy_motion.onnx``, single-clip BeyondMimic) or
      fed at runtime with ``--motion <clip.npz|clip.csv>``. A motion-library export (sidecar ``motion.source:
      runtime``, CONTRACTS 5.3) embeds none and REQUIRES ``--motion``: any clip meeting the sidecar's
      ``motion.requirements`` (plant USD, ankle variant, fps; a validator-rejected clip needs
      ``--allow-rejected-motion``).

Clocks:
    * ``--clock sim`` (default): one control step whenever the LowState tick has
      advanced by ``step_dt / tick_dt`` ticks (policy runs at 50 Hz of *simulated*
      time even when the bridge is slower or faster than real time).
    * ``--clock wall``: fixed wall-clock period like the Unitree C++ deploy.

Writes a JSONL step log and a JSON summary (loop rates, latency, per-motor
tracking error, fall time from the privileged sim block when present).
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "third_party" / "pydeps")]

import numpy as np  # noqa: E402

from dropbear_wbc.deploy import quat as Q  # noqa: E402
from dropbear_wbc.deploy.config import (DeployConfig, hold_config, load_deploy_yaml, load_sidecar,  # noqa: E402
                                        load_standing_pose)
from dropbear_wbc.deploy.fsm import DeployController, FsmState  # noqa: E402
from dropbear_wbc.sdk import motors  # noqa: E402
from dropbear_wbc.sdk.transport import LOWCMD, LOWSTATE, ChannelPublisher, ChannelSubscriber, Endpoints  # noqa: E402
from dropbear_wbc.sdk.types import LowCmd, LowState, MotorCmdBlock  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["hold", "policy"], default="hold")
    ap.add_argument("--deploy-yaml", type=Path)
    ap.add_argument("--sidecar", type=Path)
    ap.add_argument("--onnx", type=Path, help="override the policy path of the yaml/sidecar")
    ap.add_argument("--default-pose", default="legacy", help="'legacy' or a semantic calibration JSON (hold mode)")
    ap.add_argument("--allow-privileged", action="store_true", help="allow sim-only observation terms")
    ap.add_argument("--live_dir", type=Path, default=None,
                    help="LIVE reference (deploy.motion.LiveMotion): --motion plays first and is the idle clip; every "
                    "contract NPZ dropped into this folder (tools/text_to_motion.py) is checked like --motion and "
                    "spliced at the playhead; docs/TEXT_TO_MOTION.md")
    ap.add_argument("--motion", type=Path, default=None,
                    help="reference fed at runtime (contract NPZ or motion CSV): REQUIRED for a motion-library export "
                    "(sidecar motion.source='runtime'); for other exports it replaces the configured reference. Checked "
                    "against the sidecar's motion.requirements (plant USD, ankle variant, fps, validator verdict)")
    ap.add_argument("--motion-fps", type=float, default=None, help="frame rate of a --motion CSV without a sidecar")
    ap.add_argument("--allow-rejected-motion", action="store_true",
                    help="accept a --motion clip whose validator verdict / meta.status is 'rejected' (experiments)")
    ap.add_argument("--velocity-cmd", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    ap.add_argument("--clock", choices=["sim", "wall"], default="sim")
    ap.add_argument("--tick-dt", type=float, default=0.002, help="bridge tick period (sim clock without sim block)")
    ap.add_argument("--duration", type=float, default=10.0, help="controller seconds to run (sim or wall clock)")
    ap.add_argument("--passive-s", type=float, default=0.1)
    ap.add_argument("--hold-s", type=float, default=1.0, help="policy mode: time in Hold before Policy")
    ap.add_argument("--move-s", type=float, default=None, help="override MoveToDefault duration [s]")
    ap.add_argument("--no-bad-orientation", action="store_true", help="disable Policy->Passive tilt check")
    ap.add_argument("--fall-tilt-rad", type=float, default=1.0, help="summary: tilt counted as a fall")
    ap.add_argument("--fall-height-m", type=float, default=None, help="summary: root z counted as a fall")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cmd-port", type=int, default=5555)
    ap.add_argument("--state-port", type=int, default=5556)
    ap.add_argument("--connect-timeout-s", type=float, default=600.0, help="wait for the first LowState")
    ap.add_argument("--state-timeout-s", type=float, default=5.0, help="abort if LowState stops")
    ap.add_argument("--log", type=Path, help="JSONL per-step log")
    ap.add_argument("--summary", type=Path, help="JSON summary")
    ap.add_argument("--settle-window-s", type=float, default=1.0, help="tracking stats over the last N s of Hold")
    return ap.parse_args(argv)


def build(args) -> tuple[DeployConfig, DeployController]:
    if args.mode == "hold":
        pose = None if args.default_pose == "legacy" else load_standing_pose(Path(args.default_pose))
        cfg = hold_config(pose)
        policy = motion = None
    else:
        from dropbear_wbc.deploy.motion import OnnxMotion, load_motion
        from dropbear_wbc.deploy.policy import OnnxPolicy

        if args.sidecar:
            onnx_meta = None
            cfg = load_sidecar(args.sidecar)
            onnx_path = args.onnx or cfg.policy_path
            policy = OnnxPolicy(onnx_path, cfg.onnx_obs_input, cfg.onnx_time_input)
            if policy.metadata:
                onnx_meta = policy.metadata
                cfg = load_sidecar(args.sidecar, onnx_meta)
        elif args.deploy_yaml:
            cfg = load_deploy_yaml(args.deploy_yaml)
            onnx_path = args.onnx or cfg.policy_path
            if onnx_path is None:
                raise SystemExit("policy mode needs an ONNX path (--onnx or 'policy:' in the yaml)")
            policy = OnnxPolicy(onnx_path)
        else:
            raise SystemExit("policy mode needs --deploy-yaml or --sidecar")
        if policy.action_dim is not None and policy.action_dim != cfg.num_actions:
            raise SystemExit(f"ONNX action dim {policy.action_dim} != config joints {cfg.num_actions}")
        motion = None
        if getattr(args, "motion", None) is not None:
            from dropbear_wbc.deploy.config import MotionCfg
            from dropbear_wbc.deploy.motion import load_runtime_motion

            base = cfg.motion if cfg.motion is not None else MotionCfg(source="runtime")
            if base.source not in ("runtime", "npz", "csv"):
                print(f"[runner] --motion {args.motion} replaces the export's {base.source!r} reference", flush=True)
            motion, report = load_runtime_motion(base, args.motion, cfg.joint_names, cfg.step_dt,
                                                 allow_rejected=args.allow_rejected_motion, fps=args.motion_fps)
            if getattr(args, "live_dir", None) is not None:
                import dataclasses

                from dropbear_wbc.deploy.motion import LiveMotion

                motion = LiveMotion(dataclasses.replace(base, file=Path(args.motion).resolve(), source="npz"),
                                    cfg.joint_names, cfg.step_dt, watch_dir=args.live_dir,
                                    allow_rejected=args.allow_rejected_motion)
                print(f"[runner] LIVE reference: watching {args.live_dir} (idle clip {Path(args.motion).name})",
                      flush=True)
            cfg.meta["runtime_motion"] = report
            print(f"[runner] runtime reference {report['file']} ({report['num_frames']} frames, "
                  f"verdict {report['checks'].get('validation')})", flush=True)
        elif cfg.motion is not None:
            if cfg.motion.source == "runtime":
                raise SystemExit("this export embeds no reference (motion.source='runtime', motion-library policy): "
                                 "pass --motion <clip.npz|clip.csv>")
            if cfg.motion.source == "onnx":
                if not policy.has_reference:
                    raise SystemExit("motion.source=onnx but the ONNX has no time_step/reference outputs")
                side = json.loads(args.sidecar.read_text(encoding="utf-8")) if args.sidecar else {}
                mside = side.get("motion", {})
                ref_joints = mside.get("reference_joint_names") or _meta_list(policy.metadata, "joint_names")
                bodies = mside.get("body_names") or _meta_list(policy.metadata, "body_names")
                nframes = int(mside.get("num_frames", 0)) or int(1e9)
                motion = OnnxMotion(cfg.motion, cfg.joint_names, cfg.step_dt, policy.reference, ref_joints, bodies,
                                    nframes)
            else:
                motion = load_motion(cfg.motion, cfg.joint_names, cfg.step_dt)
    if args.move_s is not None:
        cfg.fsm.move_to_default_s = args.move_s
    if args.no_bad_orientation:
        cfg.fsm.bad_orientation_rad = None
    if cfg.target_interp_steps:
        print(f"[runner] NOTE: trained with each new target ramped over {cfg.target_interp_steps} x 5 ms; the robot "
              f"side must ramp too (newton_bridge.py --target-ramp-ms {5 * cfg.target_interp_steps}; ESP32 firmware)",
              flush=True)
    if cfg.target_clip is not None:
        print("[runner] motor targets clipped to the sidecar's target_clip (as in training)", flush=True)
    ctrl = DeployController(cfg, policy, motion, allow_privileged=args.allow_privileged,
                            velocity_command=np.asarray(args.velocity_cmd))
    return cfg, ctrl


def _meta_list(meta: dict, key: str) -> list[str]:
    v = meta.get(key, "")
    return [s.strip() for s in v.split(",") if s.strip()] if isinstance(v, str) else list(v)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary: dict = {"status": "failed", "args": {k: str(v) for k, v in vars(args).items()}}
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    pub = sub = None
    logf = None
    try:
        cfg, ctrl = build(args)
        summary["config"] = {"source": cfg.source, "runtime_motion": cfg.meta.get("runtime_motion"),
                             "joint_names": list(cfg.joint_names), "step_dt": cfg.step_dt,
                             "default_pose_sdk": cfg.default_pose_sdk.tolist(),
                             "hold_kp": cfg.fsm.hold_kp.tolist(), "hold_kd": cfg.fsm.hold_kd.tolist(),
                             "move_to_default_s": cfg.fsm.move_to_default_s,
                             "observations": [t.__dict__ for t in cfg.observations]}
        ep = Endpoints(args.host, args.cmd_port, args.state_port)
        sub = ChannelSubscriber(LOWSTATE, LowState, endpoints=ep, role="client")
        pub = ChannelPublisher(LOWCMD, LowCmd, endpoints=ep, role="client")
        sub.Init()
        pub.Init()
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            logf = open(args.log, "w", encoding="utf-8")

        print(f"[runner] waiting for rt/lowstate on {ep.for_channel(LOWSTATE)} ...", flush=True)
        st = sub.Read(timeout=args.connect_timeout_s)
        if st is None:
            raise TimeoutError("no LowState received")
        tick_dt = args.tick_dt
        decim = max(1, int(round(cfg.step_dt / tick_dt)))
        t0_tick = st.tick
        t0_wall = time.perf_counter()
        ctrl_time = lambda s: ((s.tick - t0_tick) * tick_dt) if args.clock == "sim" else time.perf_counter() - t0_wall  # noqa: E731
        print(f"[runner] connected at tick {st.tick}; mode={args.mode} clock={args.clock} step={cfg.step_dt}s "
              f"({decim} ticks)", flush=True)

        schedule = [(args.passive_s, FsmState.MOVE_TO_DEFAULT)]
        policy_at = None
        if args.mode == "policy":
            policy_at = args.passive_s + cfg.fsm.move_to_default_s + args.hold_s
        steps, wall_steps, state_age_ms, compute_ms, tick_gaps, policy_ms = 0, [], [], [], [], []
        hold_err: list[np.ndarray] = []
        transitions, fall = [], None
        last_tick = st.tick - decim
        last_rx = time.perf_counter()
        last_cmd: LowCmd | None = None
        last_pub = time.perf_counter()
        keepalives = 0
        next_wall = time.perf_counter()
        seq_i = 0
        while not stop["flag"]:
            # ---- acquire the state for this step
            if args.clock == "sim":
                while True:
                    new = sub.Read(timeout=0.05)
                    if new is not None:
                        st, last_rx = new, time.perf_counter()
                        if st.tick >= last_tick + decim:
                            break
                    if time.perf_counter() - last_rx > args.state_timeout_s:
                        raise TimeoutError(f"LowState stopped for {args.state_timeout_s} s")
                    if stop["flag"]:
                        break
                    # Keepalive: re-send the last command while the sim tick does not advance. This
                    # breaks the start-up deadlock (paused bridge + lost first command, ZMQ slow joiner)
                    # and keeps the bridge watchdog fed when the simulation runs slowly.
                    if last_cmd is not None and time.perf_counter() - last_pub > 0.05:
                        last_cmd.stamp_ns = 0
                        pub.Write(last_cmd)
                        last_pub = time.perf_counter()
                        keepalives += 1
            else:
                now = time.perf_counter()
                if next_wall > now:
                    time.sleep(next_wall - now)
                next_wall += cfg.step_dt
                new = sub.Read(timeout=0.0)
                if new is not None:
                    st, last_rx = new, time.perf_counter()
                elif time.perf_counter() - last_rx > args.state_timeout_s:
                    raise TimeoutError(f"LowState stopped for {args.state_timeout_s} s")
            t_rx_ns = time.perf_counter_ns()
            t = ctrl_time(st)
            if t >= args.duration:
                break
            if seq_i < len(schedule) and t >= schedule[seq_i][0] and ctrl.state == FsmState.PASSIVE:
                ctrl.request(schedule[seq_i][1], st, t)
                seq_i += 1
            if policy_at is not None and t >= policy_at and ctrl.state == FsmState.HOLD:
                ctrl.request(FsmState.POLICY, st, t)
                policy_at = None
            a = time.perf_counter()
            cmd, info = ctrl.step(st, t)
            pub.Write(cmd)
            last_cmd, last_pub = cmd, time.perf_counter()
            compute_ms.append((time.perf_counter() - a) * 1e3)
            state_age_ms.append((t_rx_ns - st.stamp_ns) * 1e-6)
            wall_steps.append(time.perf_counter())
            tick_gaps.append(st.tick - last_tick)
            last_tick = st.tick
            if info.policy_ms:
                policy_ms.append(info.policy_ms)
            if info.transition:
                transitions.append({"t": t, "tick": st.tick, "transition": info.transition})
                print(f"[runner] t={t:6.3f}s {info.transition}", flush=True)
            q = st.motor.q.astype(float)
            err = q - info.q_des
            tilt = Q.tilt_angle(st.imu.quat_wxyz.astype(float))
            root_z = float(st.sim.root_pos_w[2]) if st.sim is not None else None
            if info.state == FsmState.HOLD and ctrl.state == FsmState.HOLD:
                hold_err.append(np.concatenate([[t], err]))
            if fall is None and (tilt > args.fall_tilt_rad or (args.fall_height_m is not None and root_z is not None
                                                                and root_z < args.fall_height_m)):
                fall = {"t": t, "tick": st.tick, "tilt_rad": tilt, "root_z": root_z, "fsm": info.state.value}
                print(f"[runner] FALL detected at t={t:.3f}s tilt={tilt:.2f} rad root_z={root_z}", flush=True)
            if logf is not None:
                rec = {"step": steps, "t": t, "tick": st.tick, "fsm": info.state.value, "tilt": tilt,
                       "root_z": root_z, "state_age_ms": state_age_ms[-1], "q": q.tolist(),
                       "q_des": info.q_des.tolist(), "tau_est": st.motor.tau_est.astype(float).tolist()}
                if info.action is not None:
                    rec["action"] = info.action.tolist()
                logf.write(json.dumps(rec) + "\n")
            steps += 1

        # ---- leave the robot in damping (Passive) like Unitree deploy on exit
        exit_cmd = LowCmd(tick=st.tick)
        exit_cmd.motor = MotorCmdBlock.from_arrays(motors.NUM_MOTORS, q=st.motor.q, kp=0.0, kd=cfg.fsm.passive_kd)
        pub.Write(exit_cmd)

        wall = np.diff(wall_steps)
        herr = np.asarray(hold_err)
        tracking = None
        if len(herr):
            t_end = herr[-1, 0]
            win = herr[herr[:, 0] >= t_end - args.settle_window_s, 1:]
            tracking = {"window_s": args.settle_window_s, "samples": int(len(win)),
                        "per_motor_rms_rad": dict(zip(motors.MOTOR_NAMES, np.sqrt((win ** 2).mean(0)).tolist())),
                        "per_motor_max_abs_rad": dict(zip(motors.MOTOR_NAMES, np.abs(win).max(0).tolist())),
                        "overall_rms_rad": float(np.sqrt((win ** 2).mean())),
                        "overall_max_abs_rad": float(np.abs(win).max())}
        summary.update({
            "status": "executed", "steps": steps, "controller_time_s": ctrl_time(st),
            "runner_hz_wall": float(1.0 / wall.mean()) if len(wall) else None,
            "step_interval_wall_ms": {"p50": float(np.percentile(wall, 50) * 1e3) if len(wall) else None,
                                      "p99": float(np.percentile(wall, 99) * 1e3) if len(wall) else None},
            "ticks_per_step": {"mean": float(np.mean(tick_gaps[1:])) if len(tick_gaps) > 1 else None,
                               "max": int(np.max(tick_gaps[1:])) if len(tick_gaps) > 1 else None,
                               "expected": decim},
            "state_age_ms": {"p50": float(np.percentile(state_age_ms, 50)), "p99": float(np.percentile(state_age_ms, 99)),
                             "max": float(np.max(state_age_ms))} if state_age_ms else None,
            "compute_ms": {"p50": float(np.percentile(compute_ms, 50)), "p99": float(np.percentile(compute_ms, 99))}
            if compute_ms else None,
            "policy_ms_p50": float(np.percentile(policy_ms, 50)) if policy_ms else None,
            "transitions": transitions, "fall": fall, "hold_tracking": tracking,
            "final_fsm": ctrl.state.value, "subscriber": sub.stats, "publisher": pub.stats, "keepalives": keepalives,
            "privileged_terms": ctrl.obs_builder.privileged_terms if ctrl.obs_builder else [],
        })
    except Exception:  # noqa: BLE001
        summary["error"] = traceback.format_exc()
        print(summary["error"], file=sys.stderr, flush=True)
    finally:
        if logf is not None:
            logf.close()
        for ch in (pub, sub):
            if ch is not None:
                ch.Close()
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        print(json.dumps({k: summary.get(k) for k in ("status", "steps", "runner_hz_wall", "fall", "error")},
                         default=str), flush=True)
    return 0 if summary["status"] == "executed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
