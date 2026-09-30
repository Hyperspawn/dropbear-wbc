"""Simulated WebXR headset for testing ``tools/teleop_arm.py --device webxr`` without hardware.

Connects to the teleop tool's Vuer server over the same websocket a headset browser uses and streams, at ``--hz``,
``CAMERA_MOVE`` (head pose) and ``CONTROLLER_MOVE`` (controller poses + buttons) or ``HAND_MOVE`` (hand tracking)
events in the Vuer/WebXR format (msgpack, OpenXR basis, column-major 4x4). The operator's wrists move on a
figure-8 in front of the head; the head slowly yaws (tests the ``head_yaw`` reference). Controller script:
t = ``--calib-at`` press left X (calibrate the mapping onto the robot's current wrists), t = ``--start-at`` press
right A (start tracking), t = ``--stop-at`` press both thumbsticks (stop -> HOME).

    .venv-teleop/Scripts/python.exe tools/sim_xr_client.py --port 8012 --duration 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source")]

import numpy as np  # noqa: E402

from dropbear_wbc.teleop.frames import head_relative_to_xr, xr_flat  # noqa: E402


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def pose(p, r=None):
    t = np.eye(4)
    t[:3, 3] = p
    if r is not None:
        t[:3, :3] = r
    return t


async def run(a) -> dict:
    import aiohttp
    import msgpack

    url = f"ws://{a.host}:{a.port}/"
    stats = {"sent": 0, "url": url}
    deadline = time.time() + a.connect_timeout_s
    while True:
        try:
            session = aiohttp.ClientSession()
            ws = await session.ws_connect(url)
            break
        except Exception:  # noqa: BLE001 - server not up yet
            await session.close()
            if time.time() > deadline:
                raise
            await asyncio.sleep(0.5)
    hand = a.mode == "hands"
    t0 = time.time()
    base = {"left": np.array([0.35, 0.20, -0.45]), "right": np.array([0.35, -0.20, -0.45])}
    try:
        while True:
            t = time.time() - t0
            if t > a.duration:
                break
            w = 2 * math.pi / a.period
            yaw = a.head_yaw_amp * math.sin(2 * math.pi * t / 10.0)
            head = pose([0.0, 0.0, 1.65], rot_z(yaw))
            ramp = min(1.0, max(0.0, (t - a.start_at) / 2.0))
            rel = {}
            for s, sgn in (("left", 1.0), ("right", -1.0)):
                off = ramp * np.array([a.amp[0] * math.sin(0.5 * w * t), sgn * a.amp[1] * math.sin(w * t),
                                       a.amp[2] * math.sin(2 * w * t)])
                rel[s] = pose(base[s] + off)
            hx, lx, rx = head_relative_to_xr(head, rel["left"], rel["right"], hand)
            ts = int(time.time() * 1000)
            events = [{"etype": "CAMERA_MOVE", "ts": ts, "key": "defaultCamera",
                       "value": {"camera": {"matrix": xr_flat(hx)}}}]
            if hand:
                st = {"pinch": False, "pinchValue": 0.1, "squeeze": False, "squeezeValue": 0.0}
                events.append({"etype": "HAND_MOVE", "ts": ts, "key": "hands",
                               "value": {"left": xr_flat(lx) * 25, "right": xr_flat(rx) * 25,
                                         "leftState": st, "rightState": st}})
            else:
                def btn(at):
                    return at <= t < at + 0.3
                base_st = {"trigger": False, "triggerValue": 0.0, "squeeze": False, "squeezeValue": 0.0,
                           "thumbstickValue": [0.0, 0.0], "bButton": False}
                ls = dict(base_st, aButton=btn(a.calib_at), thumbstick=btn(a.stop_at))
                rs = dict(base_st, aButton=btn(a.start_at), thumbstick=btn(a.stop_at))
                events.append({"etype": "CONTROLLER_MOVE", "ts": ts, "key": "motionControllers",
                               "value": {"left": xr_flat(lx), "right": xr_flat(rx), "leftState": ls, "rightState": rs}})
            try:
                for ev in events:
                    await ws.send_bytes(msgpack.packb(ev, use_bin_type=True))
                    stats["sent"] += 1
            except (ConnectionResetError, aiohttp.ClientError) as e:  # the teleop tool closed its server
                stats["closed_by_server_at_s"] = round(t, 3)
                stats["close_reason"] = repr(e)
                break
            await asyncio.sleep(1.0 / a.hz)
    finally:
        await ws.close()
        await session.close()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--mode", choices=["controllers", "hands"], default="controllers")
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--period", type=float, default=6.0)
    ap.add_argument("--amp", type=float, nargs=3, default=(0.05, 0.08, 0.07), help="operator motion [m]")
    ap.add_argument("--head-yaw-amp", type=float, default=0.3, help="head yaw oscillation [rad] (must not matter)")
    ap.add_argument("--calib-at", type=float, default=1.0)
    ap.add_argument("--start-at", type=float, default=1.5)
    ap.add_argument("--stop-at", type=float, default=18.0)
    ap.add_argument("--connect-timeout-s", type=float, default=120.0)
    ap.add_argument("--summary", type=Path, default=None)
    a = ap.parse_args()
    stats = asyncio.run(run(a))
    print(json.dumps(stats), flush=True)
    if a.summary:
        a.summary.write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
