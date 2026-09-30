"""xr_teleoperate-style arm teleop for Dropbear over ``dropbear_hg-v1`` (client side, simulation).

Pipeline (one loop, ``--rate`` Hz, default 50)::

    device (scripted | keyboard | webxr) -> wrist targets (torso frame)
      -> DropbearArmIK (semantic, warm-started, priority IK) -> joint-velocity limit
      -> SemanticMap (semantic -> 10 arm motor targets) [+ gravity feed-forward]
      -> LowCmd: arms from IK, legs held at the calibrated standing pose with stiff gains
      -> Newton bridge (``tools/newton_bridge.py --fixed-base``: hanging / bench mode, like Unitree's)

Phases: MOVE_IN (all 22 motors ramp from the measured pose to the standing pose) -> SETTLE -> TELEOP -> HOME
(arms ramp back to standing) -> exit with a damping command (Unitree deploy ``Passive``).

Keyboard / XR start in a paused TELEOP (arms hold) until the operator sends ``start`` (``r`` key, right controller
A) unless ``--auto-start``; ``c`` / left controller X calibrates the XR mapping onto the robot's current wrists;
``x`` / Esc / both thumbsticks = stop (-> HOME). Scripted runs start immediately.

Clocks: ``--clock sim`` (default) steps every ``1 / rate / tick_dt`` bridge ticks (simulated time, like
``tools/policy_runner.py``); ``--clock wall`` uses a fixed wall-clock period (interactive devices with a real-time
bridge).

Recording (``--record``): ``data/teleop/<session>/`` in a LeRobot-v2.1-like layout (see
``dropbear_wbc.teleop.recorder``); ``--summary`` writes the measured tracking error and latency.

Example (bridge in another process; see docs/TELEOP.md and scripts/verify_teleop.py)::

    .venv-teleop/Scripts/python.exe tools/teleop_arm.py --device scripted --duration 25 \
        --record data/teleop/demo --summary logs/teleop/demo_summary.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "third_party" / "pydeps")]

import numpy as np  # noqa: E402

from dropbear_wbc.sdk import motors  # noqa: E402
from dropbear_wbc.sdk.transport import LOWCMD, LOWSTATE, ChannelPublisher, ChannelSubscriber, Endpoints  # noqa: E402
from dropbear_wbc.sdk.types import LowCmd, LowState, MotorCmdBlock  # noqa: E402
from dropbear_wbc.teleop.arm_ik import (  # noqa: E402
    ARM_MOTOR_SLOTS, MIRROR_SIGN, SIDES, DropbearArmIK, IKWeights, pose, so3_log,
)
from dropbear_wbc.teleop.devices import KeyboardSource, ScriptedSource, VuerXRSource  # noqa: E402
from dropbear_wbc.teleop.frames import OperatorMapping  # noqa: E402
from dropbear_wbc.teleop.recorder import SessionRecorder, pose7  # noqa: E402

HAND_BODY = {"left": "LH_shoulder_ex_al_interface_1", "right": "RH_shoulder_ex_al_interface_1"}
"""Hand-plate bodies: their frame origin is the wrist point of the IK model (calibration ``hand_body_origin``)."""

REF_POSES = {
    # left-arm semantic (pitch, roll, yaw, elbow, wrist_roll); right = mirrored
    "standing": None,
    # hands in front, forearms forward: upper arm 40 deg forward, elbow 90 deg (G1 0). With the default figure-8
    # (+-2 / 6 / 5 cm; mostly tangential) the wrist stays 0.296..0.389 m from the shoulder centre, inside the
    # reachable shell 0.281 (max elbow flexion) .. 0.405 m (straight arm); IK residual 0 along the path.
    "forward": (-0.7, 0.15, 0.0, 0.0, 0.0),
    # first verification reference (set 1 / first GPU set): its figure-8 (+-4/6/5 cm) leaves the shell by ~2.4 cm
    "forward_extended": (-0.35, 0.12, 0.0, 0.25, 0.0),
    # second attempt: violates the inner boundary (too close to the shoulder) -> IK residual up to 30 mm
    "chest": (-0.5, 0.15, 0.0, -0.3, 0.0),
    "reach": (-0.8, 0.15, 0.0, 0.6, 0.0),
}

PROBE_SETS = {
    # sequences of left-arm semantic poses (right mirrored), used with --scripted-kind poses
    "elbow": [(0.0, 0.05, 0.0, e, 0.0) for e in (1.40, 1.00, 0.60, 0.20, -0.20, -0.50, 0.40, 1.20)],
    "shoulder": [(-0.5, 0.1, 0.0, 1.0, 0.0), (-1.0, 0.1, 0.0, 1.0, 0.0), (-0.5, 0.6, 0.0, 1.0, 0.0),
                 (-0.5, 0.2, 0.6, 0.6, 0.0), (-0.5, 0.2, -0.6, 0.6, 0.5), (0.3, 0.1, 0.0, 1.2, -0.5)],
}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["scripted", "keyboard", "webxr"], default="scripted")
    ap.add_argument("--duration", type=float, default=25.0, help="TELEOP phase length [s] (0 = until stopped)")
    ap.add_argument("--rate", type=float, default=50.0, help="IK / command rate [Hz]")
    ap.add_argument("--clock", choices=["sim", "wall"], default="sim")
    ap.add_argument("--tick-dt", type=float, default=0.002, help="bridge tick period [s]")
    ap.add_argument("--move-in-s", type=float, default=2.0)
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--home-s", type=float, default=1.5)
    ap.add_argument("--auto-start", action="store_true", help="keyboard/webxr: track without waiting for 'start'")
    # scripted device
    ap.add_argument("--scripted-kind", choices=["figure8", "sine", "hold", "poses"], default="figure8")
    ap.add_argument("--ref-pose", choices=sorted(REF_POSES), default="forward",
                    help="scripted/keyboard reference (centre) arm pose")
    ap.add_argument("--amplitude", type=float, nargs=3, default=(0.02, 0.06, 0.05), metavar=("AX", "AY", "AZ"),
                    help="figure-8 amplitude [m] (x is radial for the forward pose; the reachable shell is thin)")
    ap.add_argument("--period", type=float, default=6.0)
    ap.add_argument("--ramp-s", type=float, default=2.0, help="scripted: move from standing to the pattern")
    ap.add_argument("--roll-amplitude", type=float, default=0.0, help="scripted wrist-roll oscillation [rad]")
    ap.add_argument("--probe-set", choices=sorted(PROBE_SETS), default="elbow")
    ap.add_argument("--hold-s", type=float, default=2.0, help="poses: hold time per pose")
    ap.add_argument("--pose-move-s", type=float, default=1.0, help="poses: transition time")
    # XR
    ap.add_argument("--xr-host", default="127.0.0.1", help="Vuer bind address (127.0.0.1 = this PC only)")
    ap.add_argument("--xr-port", type=int, default=8012)
    ap.add_argument("--xr-cert", default=None)
    ap.add_argument("--xr-key", default=None)
    ap.add_argument("--xr-controllers", action="store_true", help="controller tracking (default hand tracking)")
    ap.add_argument("--xr-scale", type=float, default=0.70, help="robot/human arm length ratio")
    ap.add_argument("--xr-reference", choices=["head_yaw", "head_position"], default="head_yaw")
    ap.add_argument("--keyboard-backend", default="auto", choices=["auto", "msvcrt", "pynput", "stdin", "none"])
    # IK
    ap.add_argument("--calibration", type=Path, default=None, help="semantic calibration JSON")
    ap.add_argument("--ik-mode", choices=["priority", "weighted"], default="priority")
    ap.add_argument("--ik-rot-weight", type=float, default=None, help="default 0.5 (0.05 for keyboard)")
    ap.add_argument("--ik-iters", type=int, default=5, help="max IK iterations per step (warm-started)")
    ap.add_argument("--ik-restart-mm", type=float, default=2.0,
                    help="retry from closed-form seeds (light) when the wrist error exceeds this [mm]")
    ap.add_argument("--max-joint-vel", type=float, default=4.0, help="semantic joint velocity limit [rad/s]")
    # control
    ap.add_argument("--kp-shoulder", type=float, default=200.0)
    ap.add_argument("--kd-shoulder", type=float, default=5.0)
    ap.add_argument("--kp-elbow", type=float, default=600.0,
                    help="elbow MOTOR gain; the four-bar gear ratio (~4.5 semantic/motor) makes the forearm ~20x softer")
    ap.add_argument("--kd-elbow", type=float, default=10.0)
    ap.add_argument("--kp-wrist", type=float, default=60.0)
    ap.add_argument("--kd-wrist", type=float, default=2.0)
    ap.add_argument("--leg-gain-scale", type=float, default=2.0, help="legs: legacy kp/kd x this (stiff hold)")
    ap.add_argument("--gravity-ff", choices=["off", "model"], default="model")
    ap.add_argument("--no-dq-ff", dest="dq_ff", action="store_false",
                    help="send dq* = 0 like xr_teleoperate (default: the commanded motor velocity, finite difference "
                         "of the targets, which cuts the tracking lag)")
    ap.add_argument("--inertia", type=Path, default=ROOT / "data" / "teleop" / "arm_inertia.json")
    # io
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cmd-port", type=int, default=5555)
    ap.add_argument("--state-port", type=int, default=5556)
    ap.add_argument("--connect-timeout-s", type=float, default=600.0)
    ap.add_argument("--state-timeout-s", type=float, default=5.0)
    ap.add_argument("--record", type=Path, default=None, help="session directory (LeRobot-v2-like)")
    ap.add_argument("--task", default="dropbear arm teleop (scripted figure-8)")
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--analysis-skip-s", type=float, default=None, help="ignore the first N s of TELEOP in stats")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------ helpers
def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def rot_angle(r: np.ndarray) -> float:
    return float(np.linalg.norm(so3_log(r)))


def ref_arm_q(ik: DropbearArmIK, name: str) -> np.ndarray:
    if REF_POSES[name] is None:
        return ik.rest_q()
    q = np.asarray(REF_POSES[name], dtype=float)
    return np.concatenate([ik.chains["left"].clip(q), ik.chains["right"].clip(q * MIRROR_SIGN)])


def gains(args) -> tuple[np.ndarray, np.ndarray]:
    kp = np.asarray(motors.DEFAULT_KP, dtype=float) * args.leg_gain_scale
    kd = np.asarray(motors.DEFAULT_KD, dtype=float) * args.leg_gain_scale
    for side_slots in ((12, 13, 14, 15, 16), (17, 18, 19, 20, 21)):
        for k, slot in enumerate(side_slots):
            if k < 3:
                kp[slot], kd[slot] = args.kp_shoulder, args.kd_shoulder
            elif k == 3:
                kp[slot], kd[slot] = args.kp_elbow, args.kd_elbow
            else:
                kp[slot], kd[slot] = args.kp_wrist, args.kd_wrist
    return kp, kd


def sim_hand_torso(ik: DropbearArmIK, st: LowState) -> dict:
    """Ground-truth hand-plate (wrist) poses in the torso frame from the privileged sim block, if present."""
    out = {"left": None, "right": None}
    sim = st.sim
    if sim is None or not sim.body_names:
        return out
    from dropbear_wbc.motion.rotations import quat_to_matrix

    r_root = quat_to_matrix(np.asarray(sim.root_quat_w, dtype=float))
    p_root = np.asarray(sim.root_pos_w, dtype=float)
    for s in SIDES:
        if HAND_BODY[s] in sim.body_names:
            p, q = sim.body_pose(HAND_BODY[s])
            r = r_root.T @ quat_to_matrix(np.asarray(q, dtype=float))
            pr = r_root.T @ (np.asarray(p, dtype=float) - p_root)
            out[s] = pose(pr - ik.torso_origin_root, r)
    return out


def build_device(args, ik: DropbearArmIK, q_ref: np.ndarray):
    ref = {"left": ik.fk("left", q_ref[:5]), "right": ik.fk("right", q_ref[5:])}
    start = {"left": ik.fk("left", ik.rest_q()[:5]), "right": ik.fk("right", ik.rest_q()[5:])}
    if args.device == "scripted":
        if args.scripted_kind == "poses":
            seq = PROBE_SETS[args.probe_set]
            poses = {"left": [ik.fk("left", ik.chains["left"].clip(np.asarray(q))) for q in seq],
                     "right": [ik.fk("right", ik.chains["right"].clip(np.asarray(q) * MIRROR_SIGN)) for q in seq]}
            dev = ScriptedSource(start, kind="poses", poses=poses, hold_s=args.hold_s, move_s=args.pose_move_s)
        else:
            dev = ScriptedSource(ref, kind=args.scripted_kind, amplitude=args.amplitude, period=args.period,
                                 ramp_s=args.ramp_s, start_from=start, roll_amplitude=args.roll_amplitude)
    elif args.device == "keyboard":
        dev = KeyboardSource(ref, backend=args.keyboard_backend)
    else:
        mapping = OperatorMapping(scale=args.xr_scale)
        dev = VuerXRSource(mapping=mapping, hand_tracking=not args.xr_controllers, host=args.xr_host,
                           port=args.xr_port, cert=args.xr_cert, key=args.xr_key, arm_reference_mode=args.xr_reference)
    return dev, ref


def build_key_events(args, ref: dict):
    """Terminal keys for the XR device (xr_teleoperate listens to r / q / s in the terminal while the headset
    streams): r = start, c = calibrate, x / Esc = stop, h = home. Hand tracking has no buttons, so this is how its
    operator starts / calibrates / stops. Only events are used, never the key targets."""
    if args.device != "webxr" or args.keyboard_backend == "none":
        return None
    return KeyboardSource(ref, backend=args.keyboard_backend)


# ------------------------------------------------------------------------------------------------ analysis
def best_lag(target: np.ndarray, actual: np.ndarray, dt_s: float, max_lag_s: float = 0.6) -> dict:
    """Delay [s] that minimises RMS |actual(t + lag) - target(t)| (frames are uniformly spaced by dt_s)."""
    n = len(target)
    best = (0, float("inf"))
    rms0 = None
    for k in range(0, max(1, int(max_lag_s / dt_s)) + 1):
        if n - k < 10:
            break
        e = np.linalg.norm(actual[k:] - target[:n - k], axis=1)
        r = float(np.sqrt(np.mean(e ** 2)))
        if k == 0:
            rms0 = r
        if r < best[1]:
            best = (k, r)
    return {"lag_s": best[0] * dt_s, "rms_at_lag_m": best[1], "rms_at_zero_m": rms0}


def stats_m(e: np.ndarray) -> dict:
    e = e[np.isfinite(e)]
    if not len(e):
        return {}
    return {"mean_mm": 1e3 * float(e.mean()), "rms_mm": 1e3 * float(np.sqrt(np.mean(e ** 2))),
            "p95_mm": 1e3 * float(np.percentile(e, 95)), "max_mm": 1e3 * float(e.max()), "n": int(len(e))}


def analyze(arr: dict, rate: float, skip_s: float) -> dict:
    t = arr["timestamp"][:, 0]
    trk = arr["teleop.tracking"][:, 0] > 0.5
    t_track0 = float(t[trk][0]) if trk.any() else float("inf")
    m = trk & (t >= t_track0 + skip_s)
    out: dict = {"frames": int(len(t)), "frames_tracking": int(trk.sum()), "frames_analyzed": int(m.sum()),
                 "tracking_started_s": t_track0, "skip_s_after_tracking_start": skip_s}
    if m.sum() < 10:
        return out
    dt_s = 1.0 / rate
    for s in SIDES:
        tgt, fk, cmd, sim = (arr[f"{k}.{s}_wrist"][m] for k in ("target", "fk", "fk_cmd", "sim"))
        d = {"wrist_tracking_fk_vs_target": stats_m(np.linalg.norm(fk[:, :3] - tgt[:, :3], axis=1)),
             "ik_cmd_vs_target": stats_m(np.linalg.norm(cmd[:, :3] - tgt[:, :3], axis=1)),
             "servo_fk_vs_cmd": stats_m(np.linalg.norm(fk[:, :3] - cmd[:, :3], axis=1)),
             "lag_fk_vs_target": best_lag(tgt[:, :3], fk[:, :3], dt_s),
             "lag_fk_vs_cmd": best_lag(cmd[:, :3], fk[:, :3], dt_s),
             "target_path_length_m": float(np.linalg.norm(np.diff(tgt[:, :3], axis=0), axis=1).sum()),
             "target_speed_max_mps": float(np.linalg.norm(np.diff(tgt[:, :3], axis=0), axis=1).max() * rate)}
        if np.isfinite(sim).all():
            d["wrist_tracking_sim_vs_target"] = stats_m(np.linalg.norm(sim[:, :3] - tgt[:, :3], axis=1))
            d["model_sim_vs_fk"] = stats_m(np.linalg.norm(sim[:, :3] - fk[:, :3], axis=1))
            d["lag_sim_vs_target"] = best_lag(tgt[:, :3], sim[:, :3], dt_s)
        out[s] = d
    q_err = arr["observation.state"][m] - arr["action"][m]
    out["semantic_rms_rad"] = np.sqrt(np.mean(q_err ** 2, axis=0)).round(5).tolist()
    mq = arr["observation.motor_q"][m][:, list(ARM_MOTOR_SLOTS)] - arr["action.motor_q"][m]
    out["arm_motor_rms_rad"] = np.sqrt(np.mean(mq ** 2, axis=0)).round(5).tolist()
    out["arm_motor_mean_rad"] = mq.mean(0).round(5).tolist()
    out["ik_pos_err_max_mm"] = 1e3 * float(np.nanmax(arr["ik.pos_err_m"][m]))
    out["ik_rot_err_median_rad"] = float(np.nanmedian(arr["ik.rot_err_rad"][m]))
    for k in ("latency.compute_ms", "latency.state_age_ms", "latency.device_age_ms"):
        v = arr[k][m, 0]
        out[k] = {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}
    return out


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    args = parse_args(argv)
    stop = {"flag": False, "count": 0}

    def _on_sigint(*_):
        stop["flag"] = True
        stop["count"] += 1  # a second Ctrl-C also aborts the HOME ramp

    signal.signal(signal.SIGINT, _on_sigint)
    summary: dict = {"status": "failed", "args": {k: str(v) for k, v in vars(args).items()},
                     "started": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    pub = sub = dev = None
    rec = None
    st = None
    try:
        rot_w = args.ik_rot_weight if args.ik_rot_weight is not None else (0.05 if args.device == "keyboard" else 0.5)
        ik = DropbearArmIK(args.calibration, weights=IKWeights(rotation=rot_w, mode=args.ik_mode))
        grav = None
        if args.gravity_ff == "model":
            from dropbear_wbc.teleop.gravity import ArmGravity
            grav = ArmGravity(ik, args.inertia)
        kp, kd = gains(args)

        def _rebuild():  # runs on the reloader thread (never inside the control loop)
            ik2 = DropbearArmIK(ik.path, weights=IKWeights(rotation=rot_w, mode=args.ik_mode))
            g2 = None
            if args.gravity_ff == "model":
                from dropbear_wbc.teleop.gravity import ArmGravity as _AG
                g2 = _AG(ik2, args.inertia)
            return ik2, g2

        from dropbear_wbc.teleop.reload import BackgroundReloader

        reloader = BackgroundReloader(ik.path, _rebuild, ik.sha256).start()
        q_stand22 = ik.standing_motor.copy()
        q_rest10 = ik.rest_q()
        q_ref10 = ref_arm_q(ik, args.ref_pose)
        dev, ref = build_device(args, ik, q_ref10)
        keys = build_key_events(args, ref)
        summary["ik"] = ik.info()
        summary["gains"] = {"kp": kp.tolist(), "kd": kd.tolist()}
        summary["gravity_ff"] = None if grav is None else {"inertia": str(grav.path), "arm_mass_kg": grav.arm_mass}
        decim = max(1, int(round(1.0 / args.rate / args.tick_dt)))
        rate = 1.0 / (decim * args.tick_dt)
        summary["loop"] = {"rate_hz": rate, "ticks_per_step": decim, "clock": args.clock}
        if args.record:
            rec = SessionRecorder(args.record, fps=rate, task=args.task)

        ep = Endpoints(args.host, args.cmd_port, args.state_port)
        sub = ChannelSubscriber(LOWSTATE, LowState, endpoints=ep, role="client")
        pub = ChannelPublisher(LOWCMD, LowCmd, endpoints=ep, role="client")
        sub.Init()
        pub.Init()
        dev.start()
        if keys is not None:
            keys.start()
        if isinstance(dev, VuerXRSource):
            print(f"[teleop] WebXR: open {dev.url} in the headset browser (see docs/TELEOP.md)", flush=True)
        print(f"[teleop] waiting for rt/lowstate on {ep.for_channel(LOWSTATE)} ...", flush=True)
        st = sub.Read(timeout=args.connect_timeout_s)
        if st is None:
            raise TimeoutError("no LowState received")
        q_start22 = st.motor.q.astype(float).copy()
        t0_tick, t0_wall = st.tick, time.perf_counter()
        print(f"[teleop] connected at tick {st.tick}; device={args.device} rate={rate:.1f} Hz ({decim} ticks) "
              f"clock={args.clock} ik={args.ik_mode} rot_w={rot_w} gravity_ff={args.gravity_ff}", flush=True)

        phase = "move_in"
        phase_t0 = 0.0
        tracking = args.device == "scripted" or args.auto_start
        q_cmd10 = q_rest10.copy()     # commanded semantic arm angles (after the velocity limit)
        q_home_from = None
        t_teleop0 = None
        last_tick = st.tick - decim
        sched_tick = st.tick  # sim clock: control steps sit on a fixed tick grid (sched_tick + k * decim)
        last_rx = time.perf_counter()
        last_cmd = None
        last_pub = time.perf_counter()
        keepalives = 0
        overruns = 0
        next_wall = time.perf_counter()
        transitions = []
        events_log = []
        steps = 0
        loop_wall = []
        dt_step = decim * args.tick_dt
        q_des_prev = None
        while not stop["flag"]:
            if args.clock == "sim":
                while True:
                    new = sub.Read(timeout=0.05)
                    if new is not None:
                        st, last_rx = new, time.perf_counter()
                        if st.tick >= sched_tick:
                            break
                    if time.perf_counter() - last_rx > args.state_timeout_s:
                        raise TimeoutError(f"LowState stopped for {args.state_timeout_s} s")
                    if stop["flag"]:
                        break
                    if last_cmd is not None and time.perf_counter() - last_pub > 0.05:
                        last_cmd.stamp_ns = 0
                        pub.Write(last_cmd)
                        last_pub = time.perf_counter()
                        keepalives += 1
            else:
                now = time.perf_counter()
                if next_wall > now:
                    time.sleep(next_wall - now)
                next_wall += dt_step
                new = sub.Read(timeout=0.0)
                if new is not None:
                    st, last_rx = new, time.perf_counter()
                elif time.perf_counter() - last_rx > args.state_timeout_s:
                    raise TimeoutError(f"LowState stopped for {args.state_timeout_s} s")
            t_rx_ns = time.perf_counter_ns()
            t = (st.tick - t0_tick) * args.tick_dt if args.clock == "sim" else time.perf_counter() - t0_wall
            # control periods missed before this step (sim clock: the newest LowState is more than one period ahead)
            # (review fix 2026-09-24) stay on the grid: a late state is used for its grid step; whole periods missed
            # are counted (the old rule re-based the grid on every late state, so the sim time of the frames drifted)
            skipped_steps = 0
            if args.clock == "sim":
                skipped_steps = max(0, (st.tick - sched_tick) // decim)
                sched_tick += decim * (skipped_steps + 1)
            overruns += int(skipped_steps > 0)
            last_tick = st.tick
            loop_wall.append(time.perf_counter())
            swap = reloader.take()  # non-blocking: the rebuild (~0.5 s) ran on the reloader thread
            if swap is not None:
                _sha, (ik_new, grav_new), build_s = swap
                ik_new.reset(q_cmd10)  # keep the IK warm start continuous across the swap
                ik, grav = ik_new, grav_new
                q_stand22 = ik.standing_motor.copy()
                q_rest10 = ik.rest_q()
                print(f"[teleop] calibration changed on disk -> swapped in (sha {ik.sha256[:8]}, built in "
                      f"{build_s:.2f} s off the control loop)", flush=True)
                events_log.append({"t": t, "event": "calibration_reloaded", "sha256": ik.sha256, "build_s": build_s,
                                   "note": "device mappings built at start keep the old calibration"})

            q_meas22 = st.motor.q.astype(float)
            tau_ff10 = np.zeros(10)
            ik_res = None
            targets = None
            q_ik10 = q_cmd10.copy()
            # ---------------------------------------------------------------- phase logic
            if phase == "move_in":
                a = smoothstep((t - phase_t0) / max(args.move_in_s, 1e-6))
                q_des22 = q_start22 + a * (q_stand22 - q_start22)
                if a >= 1.0:
                    phase, phase_t0 = "settle", t
                    transitions.append({"t": t, "to": phase})
            elif phase == "settle":
                q_des22 = q_stand22.copy()
                if t - phase_t0 >= args.settle_s:
                    phase, phase_t0, t_teleop0 = "teleop", t, t
                    transitions.append({"t": t, "to": phase})
                    ik.reset(q_rest10)
                    q_cmd10 = q_rest10.copy()
                    print(f"[teleop] t={t:.2f}s TELEOP ({'tracking' if tracking else 'waiting for start'})", flush=True)
            # operator events are processed in every phase (an operator may press start / calibrate while the robot
            # is still moving in; the arms only follow once TELEOP begins)
            tt = t - t_teleop0 if phase == "teleop" else 0.0
            targets = dev.get(tt) if (phase == "teleop" or args.device != "scripted") else None
            key_events = keys.get(tt).events if keys is not None else []
            for ev in (targets.events if targets is not None else []) + key_events:
                events_log.append({"t": t, "phase": phase, "event": ev})
                if ev in ("start",):
                    tracking = True
                elif ev == "toggle":
                    tracking = not tracking
                elif ev == "stop":
                    stop["flag"] = True
                elif ev == "calibrate" and isinstance(dev, VuerXRSource):
                    try:
                        info = dev.calibrate(ik.fk_both(q_cmd10))
                        events_log.append({"t": t, "event": "xr_calibrated", "mapping": info})
                        print("[teleop] XR mapping calibrated onto the current robot wrists", flush=True)
                    except RuntimeError as e:
                        events_log.append({"t": t, "event": "xr_calibrate_failed", "error": str(e)})
            if phase == "teleop":
                if tracking and targets.valid:
                    ik_res = ik.solve(targets.left, targets.right, max_iters=args.ik_iters, restarts="light",
                                      restart_above_m=1e-3 * args.ik_restart_mm)
                    q_ik10 = ik_res.q_sem
                    dq = np.clip(q_ik10 - q_cmd10, -args.max_joint_vel * dt_step, args.max_joint_vel * dt_step)
                    q_cmd10 = q_cmd10 + dq  # the IK keeps its own solution as the warm start
                if args.duration > 0 and tt >= args.duration:
                    stop["flag"] = True
            if phase in ("teleop",):
                m10, _sat = ik.to_motor_fast(q_cmd10)
                q_des22 = ik.motor22(m10, q_stand22)
                if grav is not None:
                    tau_ff10 = grav.motor_torque(q_cmd10, m10)
            elif phase in ("move_in", "settle") and grav is not None:
                # gravity FF on the arms at the (interpolated) commanded arm pose
                sem_des = ik.motor_to_semantic_arms_fast(q_des22)
                tau_ff10 = grav.motor_torque(sem_des, q_des22[list(ARM_MOTOR_SLOTS)])
            # ---------------------------------------------------------------- command
            tau22 = np.zeros(22)
            tau22[list(ARM_MOTOR_SLOTS)] = tau_ff10
            dq22 = np.zeros(22)
            if args.dq_ff and q_des_prev is not None and phase == "teleop":
                dq22[list(ARM_MOTOR_SLOTS)] = (q_des22 - q_des_prev)[list(ARM_MOTOR_SLOTS)] / dt_step
            q_des_prev = q_des22.copy()
            cmd = LowCmd(tick=st.tick)
            cmd.motor = MotorCmdBlock.from_arrays(motors.NUM_MOTORS, q=q_des22, dq=dq22, tau=tau22, kp=kp, kd=kd)
            pub.Write(cmd)
            t_tx_ns = time.perf_counter_ns()
            last_cmd, last_pub = cmd, time.perf_counter()
            steps += 1
            # ---------------------------------------------------------------- record
            if rec is not None and phase == "teleop":
                q_sem_meas = ik.motor_to_semantic_arms_fast(q_meas22)
                fk_m = ik.fk_both(q_sem_meas)
                fk_c = ik.fk_both(q_cmd10)
                simw = sim_hand_torso(ik, st)
                tl = targets.left if targets is not None and targets.left is not None else fk_c["left"]
                tr = targets.right if targets is not None and targets.right is not None else fk_c["right"]
                arms = ik_res.arms if ik_res is not None else None
                rec.add(**{
                    "observation.state": q_sem_meas, "action": q_cmd10, "timestamp": t - t_teleop0,
                    "observation.motor_q": q_meas22, "observation.motor_dq": st.motor.dq.astype(float),
                    "observation.motor_tau": st.motor.tau_est.astype(float),
                    "action.motor_q": q_des22[list(ARM_MOTOR_SLOTS)], "action.tau_ff": tau_ff10,
                    "action.ik_q": q_ik10,
                    "target.left_wrist": pose7(tl), "target.right_wrist": pose7(tr),
                    "fk.left_wrist": pose7(fk_m["left"]), "fk.right_wrist": pose7(fk_m["right"]),
                    "fk_cmd.left_wrist": pose7(fk_c["left"]), "fk_cmd.right_wrist": pose7(fk_c["right"]),
                    "sim.left_wrist": pose7(simw["left"]), "sim.right_wrist": pose7(simw["right"]),
                    "ik.pos_err_m": [arms["left"].pos_err_m, arms["right"].pos_err_m] if arms else [np.nan, np.nan],
                    "ik.rot_err_rad": [arms["left"].rot_err_rad, arms["right"].rot_err_rad] if arms else [np.nan, np.nan],
                    "time.sim_s": st.sim.time_s if st.sim is not None else t, "time.wall_s": time.perf_counter() - t0_wall,
                    "time.tick": st.tick, "time.skipped_steps": skipped_steps,
                    "teleop.tracking": float(tracking and targets is not None and targets.valid),
                    "latency.compute_ms": (t_tx_ns - t_rx_ns) * 1e-6,
                    "latency.state_age_ms": (t_rx_ns - st.stamp_ns) * 1e-6,
                    "latency.device_age_ms": ((t_tx_ns - targets.stamp_ns) * 1e-6) if targets is not None else np.nan,
                })
        # ---------------------------------------------------------------- HOME + exit
        if st is not None and phase == "teleop":
            print(f"[teleop] HOME: arms back to standing over {args.home_s:.1f} s", flush=True)
            q_home_from = q_cmd10.copy()
            t_h0 = (st.tick - t0_tick) * args.tick_dt
            last_tick = st.tick
            home_rx = time.perf_counter()
            home_deadline = home_rx + args.home_s + args.state_timeout_s
            sigints_at_home = stop["count"]
            home_abort = None
            while True:
                # the HOME ramp must terminate even if the bridge dies or the operator insists (review fix 2026-09-24):
                # no LowState for state_timeout_s, the wall-clock deadline, or another Ctrl-C ends it; the recording
                # and the summary below are still written.
                if time.perf_counter() - home_rx > args.state_timeout_s:
                    home_abort = f"no LowState for {args.state_timeout_s} s"
                elif time.perf_counter() > home_deadline:
                    home_abort = f"HOME deadline home_s + state_timeout_s = {args.home_s + args.state_timeout_s:.1f} s"
                elif stop["count"] > sigints_at_home:
                    home_abort = "SIGINT during HOME"
                if home_abort is not None:
                    print(f"[teleop] HOME aborted: {home_abort}", flush=True)
                    break
                new = sub.Read(timeout=0.05)
                if new is None:
                    if time.perf_counter() - last_pub > 0.05 and last_cmd is not None:
                        last_cmd.stamp_ns = 0
                        pub.Write(last_cmd)
                        last_pub = time.perf_counter()
                    continue
                st, home_rx = new, time.perf_counter()
                if st.tick < last_tick + decim:
                    continue
                last_tick = st.tick
                th = (st.tick - t0_tick) * args.tick_dt - t_h0
                a = smoothstep(th / max(args.home_s, 1e-6))
                qh = q_home_from + a * (q_rest10 - q_home_from)
                m10, _ = ik.to_motor_fast(qh)
                tau22 = np.zeros(22)
                if grav is not None:
                    tau22[list(ARM_MOTOR_SLOTS)] = grav.motor_torque(qh, m10)
                cmd = LowCmd(tick=st.tick)
                cmd.motor = MotorCmdBlock.from_arrays(motors.NUM_MOTORS, q=ik.motor22(m10, q_stand22), tau=tau22,
                                                      kp=kp, kd=kd)
                pub.Write(cmd)
                last_cmd, last_pub = cmd, time.perf_counter()
                if a >= 1.0:
                    break
            transitions.append({"t": (st.tick - t0_tick) * args.tick_dt,
                                "to": "home_done" if home_abort is None else "home_aborted",
                                **({} if home_abort is None else {"reason": home_abort})})
        if st is not None and pub is not None:
            exit_cmd = LowCmd(tick=st.tick)
            exit_cmd.motor = MotorCmdBlock.from_arrays(motors.NUM_MOTORS, q=st.motor.q, kp=0.0,
                                                       kd=np.asarray(motors.DEFAULT_KD, dtype=float))
            pub.Write(exit_cmd)
            time.sleep(0.05)
        lw = np.diff(np.asarray(loop_wall)) if len(loop_wall) > 1 else np.zeros(0)
        summary.update({
            "status": "ok", "steps": steps, "keepalives": keepalives, "overrun_steps": overruns,
            "transitions": transitions,
            "events": events_log, "subscriber": sub.stats, "publisher": pub.stats,
            "loop_wall_period_ms": {"p50": float(np.percentile(lw, 50) * 1e3), "p95": float(np.percentile(lw, 95) * 1e3),
                                    "max": float(lw.max() * 1e3)} if len(lw) else None,
            "device": {"name": dev.name, **({"xr_counts": dev.counts, "url": dev.url} if isinstance(dev, VuerXRSource)
                                            else {})},
        })
        if rec is not None and len(rec):
            arr = rec.arrays()
            if args.analysis_skip_s is not None:
                skip = args.analysis_skip_s
            elif args.device == "scripted":
                skip = args.ramp_s + 0.5 if args.scripted_kind != "poses" else 0.0
            else:
                skip = 2.5  # live devices: skip the first 2.5 s after tracking starts (operator ramp / catch-up)
            summary["analysis"] = analyze(arr, rate, skip)
            meta = {k: v for k, v in summary.items() if k != "analysis"}
            meta["analysis"] = summary["analysis"]
            if isinstance(dev, VuerXRSource):
                meta["xr_mapping"] = dev.mapping.to_dict()
            summary["recording"] = rec.close(meta)
            print(f"[teleop] recorded {len(rec)} frames -> {args.record}", flush=True)
        rc = 0
    except Exception as e:  # noqa: BLE001
        summary["error"] = repr(e)
        summary["traceback"] = traceback.format_exc()
        print(summary["traceback"], flush=True)
        rc = 1
    finally:
        if locals().get("reloader") is not None:
            locals()["reloader"].close()
        for c in (dev, locals().get("keys")):
            try:
                if c is not None:
                    c.close()
            except Exception:  # noqa: BLE001
                pass
        for c in (sub, pub):
            try:
                if c is not None:
                    c.Close()
            except Exception:  # noqa: BLE001
                pass
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    an = summary.get("analysis", {})
    for s in SIDES:
        if s in an:
            d = an[s]
            print(f"[teleop] {s}: wrist FK-vs-target {d['wrist_tracking_fk_vs_target']}  lag {d['lag_fk_vs_target']}",
                  flush=True)
            if "wrist_tracking_sim_vs_target" in d:
                print(f"[teleop] {s}: wrist SIM-vs-target {d['wrist_tracking_sim_vs_target']}  "
                      f"model SIM-vs-FK {d['model_sim_vs_fk']}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
