"""Dropbear port of unitree_sdk2_python ``example/g1/low_level/g1_low_level_example.py``.

Only the imports, motor count/indices and gains change; the control structure
(ChannelFactoryInitialize, ChannelPublisher("rt/lowcmd"), ChannelSubscriber
("rt/lowstate", handler), a 2 ms write loop, CRC filled on write) is Unitree's.

Stage 1 (3 s): interpolate every motor from its measured position to the default pose (``--default-pose``).
Stage 2: sine on the calf motors A/B (parallel ankle) and wrist rolls around the default pose.

Dropbear safety deltas vs Unitree (item 266): no command is written until a LowState arrives within
``--connect-timeout-s`` (Unitree's 600 s wait fell through with q0 = 0); gains are clipped to the
motion-mode field (knee kd 12 -> 5, else the HAL rejects every LowCmd); LowState older than
``--state-timeout-s`` aborts; every exit path (end, abort, Ctrl-C, SIGTERM, SIGHUP) sends damping (kp 0).

Default pose (item 377): ``--default-pose legacy`` (default, unchanged: sdk ``motors.DEFAULT_POS``, both hip rolls
-0.1 rad = semantic L -5.9 / R -5.6 deg, mirror error 11.5 deg, item 370), ``calibration`` (repo calibration
``standing_motor_pos``) or a semantic calibration JSON path. The pose is mirror-checked before anything is
sent (same report as tools/policy_runner.py, warn only) and recorded in the summary as ``default_pose``.

Run against the bridge (hang the robot)::

    .venv-newton/Scripts/python.exe tools/newton_bridge.py --fixed-base
    .venv-newton/Scripts/python.exe scripts/examples/dropbear_low_level_example.py --duration 8
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source")]

import numpy as np  # noqa: E402

from dropbear_wbc.deploy.config import (load_standing_pose, motion_mode_field_kd, motion_mode_field_kp,  # noqa: E402
                                        pose_mirror_report)
from dropbear_wbc.sdk import motors  # noqa: E402
from dropbear_wbc.sdk.signals import install_stop_handler  # noqa: E402
from dropbear_wbc.sdk.transport import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from dropbear_wbc.sdk.types import LowCmd, LowState  # noqa: E402

NUM_MOTOR = motors.NUM_MOTORS
Kp = list(motion_mode_field_kp())  # DEFAULT_KP clipped to the motion-mode field (0..500)
Kd = list(motion_mode_field_kd())  # DEFAULT_KD clipped to 0..5 (knees 12 -> 5)
PASSIVE_KD = motion_mode_field_kd()  # damping on exit, same as DeployConfig.fsm.passive_kd default
CALIBRATION_JSON = ROOT / "data/calibration/dropbear_semantic_calibration.json"


def resolve_default_pose(choice: str | None = "legacy") -> tuple[np.ndarray, Path, str]:
    """``--default-pose`` -> (Stage-1 target, SDK order; calibration it is mirror-checked against; source label).

    'legacy' = sdk ``motors.DEFAULT_POS`` (checked against the repo calibration); 'calibration' = the repo calibration
    ``standing_motor_pos``; anything else = a semantic calibration JSON path (missing/invalid file raises before any
    command is written)."""
    if choice in (None, "legacy"):
        return np.asarray(motors.DEFAULT_POS, dtype=float).copy(), CALIBRATION_JSON, "legacy"
    calib = CALIBRATION_JSON if choice == "calibration" else Path(choice)
    pose = load_standing_pose(calib)
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"standing_motor_pos in {calib} is not finite")
    return pose, calib, "calibration" if choice == "calibration" else "path"


def default_pose_symmetry(pose, calibration: Path, source: str) -> dict | None:
    """Left/right mirror report of the Stage-1 target (warn only, as the runner's hold target check, item 376).
    None = calibration file missing (NOTE printed)."""
    calibration = Path(calibration)
    if not calibration.is_file():
        print(f"[example] NOTE: default pose mirror check skipped: calibration {calibration} not found", flush=True)
        return None
    rep = pose_mirror_report(pose, calibration)
    rep["default_pose_source"] = source
    if rep["asymmetric"]:
        print(f"[example] WARNING: default pose ({source}) is not left/right symmetric: max semantic mirror error "
              f"{rep['max_err_deg']:.1f} deg ({rep['worst']}; hip roll {rep['hip_roll_deg']}). The legacy pose rolls "
              f"both hips the same way (item 370); symmetric: --default-pose calibration", flush=True)
    return rep


class StateLost(RuntimeError):
    """No LowState at start, or LowState went stale mid-run."""


class DropbearJointIndex:
    LeftCalfMotorA = motors.motor_index("LL_Revolute67")
    LeftCalfMotorB = motors.motor_index("LL_Revolute81")
    RightCalfMotorA = motors.motor_index("RL_Revolute67")
    RightCalfMotorB = motors.motor_index("RL_Revolute81")
    LeftWristRoll = motors.motor_index("LH_wrist_roll")
    RightWristRoll = motors.motor_index("RH_wrist_roll")


class Custom:
    def __init__(self, duration: float, connect_timeout_s: float = 20.0, state_timeout_s: float = 0.5,
                 default_pose=None):
        self.connect_timeout_s = connect_timeout_s
        self.state_timeout_s = state_timeout_s  # <= 0 disables
        self.last_rx = 0.0
        self.time_ = 0.0
        self.control_dt_ = 0.002  # [2 ms]
        self.duration_ = 3.0  # [3 s]
        self.total_ = duration
        self.low_cmd = LowCmd()
        self.low_state: LowState | None = None
        self.first_state = threading.Event()
        self.errors: list[float] = []
        self.q0 = np.zeros(NUM_MOTOR)
        self.default = np.asarray(motors.DEFAULT_POS if default_pose is None else default_pose, dtype=float).copy()
        if self.default.shape != (NUM_MOTOR,) or not np.all(np.isfinite(self.default)):
            raise ValueError(f"default pose must be {NUM_MOTOR} finite motor angles, got shape {self.default.shape}")

    def Init(self):
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def LowStateHandler(self, msg: LowState):
        self.low_state = msg
        self.last_rx = time.perf_counter()
        if not self.first_state.is_set():
            self.q0 = msg.motor.q.astype(float).copy()
            self.first_state.set()

    def Start(self):
        if not self.first_state.wait(timeout=self.connect_timeout_s):
            raise StateLost(f"no LowState within {self.connect_timeout_s} s; no command was written")
        next_t = time.perf_counter()
        while self.time_ < self.total_:
            age = time.perf_counter() - self.last_rx
            if self.state_timeout_s > 0 and age > self.state_timeout_s:
                raise StateLost(f"LowState {age:.3f} s old (> {self.state_timeout_s} s) at t={self.time_:.3f} s")
            self.LowCmdWrite()
            next_t += self.control_dt_
            time.sleep(max(0.0, next_t - time.perf_counter()))

    def LowCmdWrite(self):
        self.time_ += self.control_dt_
        if self.time_ < self.duration_:
            # [Stage 1]: move to the default posture
            ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
            for i in range(NUM_MOTOR):
                self.low_cmd.motor_cmd[i].mode = 1  # 1: Enable, 0: Disable
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.q0[i] + ratio * self.default[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = Kp[i]
                self.low_cmd.motor_cmd[i].kd = Kd[i]
        else:
            # [Stage 2]: swing the parallel-ankle calf motors and wrists
            t = self.time_ - self.duration_
            amp_a, amp_b, amp_w = np.deg2rad(15.0), np.deg2rad(10.0), np.deg2rad(30.0)
            J = DropbearJointIndex
            for idx, amp, phase in ((J.LeftCalfMotorA, amp_a, 0.0), (J.LeftCalfMotorB, amp_b, np.pi),
                                    (J.RightCalfMotorA, amp_a, 0.0), (J.RightCalfMotorB, amp_b, np.pi),
                                    (J.LeftWristRoll, amp_w, 0.0), (J.RightWristRoll, amp_w, 0.0)):
                self.low_cmd.motor_cmd[idx].q = self.default[idx] + amp * np.sin(2.0 * np.pi * 0.5 * t + phase)
            if self.low_state is not None:
                self.errors.append(float(np.abs(self.low_state.motor.q - self.low_cmd.motor.q).max()))
        self.low_cmd.tick = self.low_state.tick if self.low_state is not None else 0
        self.low_cmd.stamp_ns = 0  # let Write() stamp it
        self.lowcmd_publisher_.Write(self.low_cmd)  # CRC is filled by Write()

    def Damp(self) -> bool:
        """Leave the robot in damping (kp 0, passive kd) at the last measured position. Best effort."""
        if self.low_state is None:
            return False  # nothing was ever commanded (Start refuses to write before the first state)
        cmd = LowCmd(tick=self.low_state.tick)
        for i in range(NUM_MOTOR):
            cmd.motor_cmd[i].mode = 1
            cmd.motor_cmd[i].q = float(self.low_state.motor.q[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = float(PASSIVE_KD[i])
        cmd.stamp_ns = 0
        self.lowcmd_publisher_.Write(cmd)
        return True


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=8.0, help="wall seconds")
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--cmd-port", type=int, default=5555)
    ap.add_argument("--state-port", type=int, default=5556)
    ap.add_argument("--connect-timeout-s", type=float, default=20.0, help="wait for the first LowState")
    ap.add_argument("--state-timeout-s", type=float, default=0.5, help="abort on older LowState (<=0 off)")
    ap.add_argument("--default-pose", default="legacy",
                    help="Stage-1 target: 'legacy' (default; sdk motors.DEFAULT_POS, both hip rolls -0.1 rad, NOT "
                         "mirror-symmetric, item 370), 'calibration' (repo data/calibration/"
                         "dropbear_semantic_calibration.json standing_motor_pos) or a semantic calibration JSON path")
    return ap.parse_args(argv)


def default_pose_record(choice) -> tuple[np.ndarray, dict]:
    """Resolve ``--default-pose`` and mirror-check it -> (pose, summary ``default_pose`` block). Runs before any
    transport is opened."""
    pose, calib, source = resolve_default_pose(choice)
    mirror = default_pose_symmetry(pose, calib, source)
    print(f"[example] default pose: {source} ({calib.as_posix() if source != 'legacy' else 'sdk motors.DEFAULT_POS'})",
          flush=True)
    return pose, {"choice": choice, "source": source, "calibration": calib.as_posix(), "mirror": mirror,
                  "default_pose_sdk": pose.tolist()}


if __name__ == "__main__":
    args = parse_args()
    default_pose, default_pose_info = default_pose_record(args.default_pose)
    ChannelFactoryInitialize(0, cmd_port=args.cmd_port, state_port=args.state_port)
    custom = Custom(args.duration, args.connect_timeout_s, args.state_timeout_s, default_pose)
    custom.Init()
    error, rc = None, 0
    install_stop_handler()  # item 268: SIGTERM/SIGHUP raise KeyboardInterrupt too, so the damping finally runs
    try:
        custom.Start()
    except (StateLost, KeyboardInterrupt) as exc:
        error, rc = f"{type(exc).__name__}: {exc}", 1
    finally:
        damped = custom.Damp()
    e = np.asarray(custom.errors)
    summary = {"rc": rc, "error": error, "damping_sent": damped,"stage2_samples": len(e), "stage2_max_abs_err_rad_p50": float(np.median(e)) if len(e) else None,
               "stage2_max_abs_err_rad_max": float(e.max()) if len(e) else None,
               "published": custom.lowcmd_publisher_.stats, "received": custom.lowstate_subscriber.stats,
               "default_pose": dict(default_pose_info, commanded_sdk=custom.default.tolist())}
    print(json.dumps(summary))
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    custom.lowstate_subscriber.Close()
    custom.lowcmd_publisher_.Close()
    if error:
        print(error, file=sys.stderr)
    sys.exit(rc)
