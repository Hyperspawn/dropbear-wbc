"""Dropbear port of unitree_sdk2_python ``example/g1/low_level/g1_low_level_example.py``.

Only the imports, motor count/indices and gains change; the control structure
(ChannelFactoryInitialize, ChannelPublisher("rt/lowcmd"), ChannelSubscriber
("rt/lowstate", handler), a 2 ms write loop, CRC filled on write) is Unitree's.

Stage 1 (3 s): interpolate every motor from its measured position to the default pose.
Stage 2: sine on the calf motors A/B (parallel ankle) and wrist rolls around the default pose.

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

from dropbear_wbc.sdk import motors  # noqa: E402
from dropbear_wbc.sdk.transport import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from dropbear_wbc.sdk.types import LowCmd, LowState  # noqa: E402

NUM_MOTOR = motors.NUM_MOTORS
Kp = list(motors.DEFAULT_KP)
Kd = list(motors.DEFAULT_KD)


class DropbearJointIndex:
    LeftCalfMotorA = motors.motor_index("LL_Revolute67")
    LeftCalfMotorB = motors.motor_index("LL_Revolute81")
    RightCalfMotorA = motors.motor_index("RL_Revolute67")
    RightCalfMotorB = motors.motor_index("RL_Revolute81")
    LeftWristRoll = motors.motor_index("LH_wrist_roll")
    RightWristRoll = motors.motor_index("RH_wrist_roll")


class Custom:
    def __init__(self, duration: float):
        self.time_ = 0.0
        self.control_dt_ = 0.002  # [2 ms]
        self.duration_ = 3.0  # [3 s]
        self.total_ = duration
        self.low_cmd = LowCmd()
        self.low_state: LowState | None = None
        self.first_state = threading.Event()
        self.errors: list[float] = []
        self.q0 = np.zeros(NUM_MOTOR)
        self.default = np.asarray(motors.DEFAULT_POS)

    def Init(self):
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def LowStateHandler(self, msg: LowState):
        self.low_state = msg
        if not self.first_state.is_set():
            self.q0 = msg.motor.q.astype(float).copy()
            self.first_state.set()

    def Start(self):
        self.first_state.wait(timeout=600)
        next_t = time.perf_counter()
        while self.time_ < self.total_:
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=8.0, help="wall seconds")
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--cmd-port", type=int, default=5555)
    ap.add_argument("--state-port", type=int, default=5556)
    args = ap.parse_args()
    ChannelFactoryInitialize(0, cmd_port=args.cmd_port, state_port=args.state_port)
    custom = Custom(args.duration)
    custom.Init()
    custom.Start()
    e = np.asarray(custom.errors)
    summary = {"stage2_samples": len(e), "stage2_max_abs_err_rad_p50": float(np.median(e)) if len(e) else None,
               "stage2_max_abs_err_rad_max": float(e.max()) if len(e) else None,
               "published": custom.lowcmd_publisher_.stats, "received": custom.lowstate_subscriber.stats}
    print(json.dumps(summary))
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    custom.lowstate_subscriber.Close()
    custom.lowcmd_publisher_.Close()
