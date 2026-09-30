"""Teleop input sources and XR frame conventions (``dropbear_wbc.teleop.devices`` / ``.frames``).

The WebXR test starts a real Vuer server on 127.0.0.1 (random free port) and plays a simulated headset over its
websocket (msgpack ``CAMERA_MOVE`` / ``HAND_MOVE`` / ``CONTROLLER_MOVE`` events, as the Vuer web client sends them).
It needs ``vuer`` (``.venv-teleop``) and skips elsewhere.
"""
from __future__ import annotations

import asyncio
import math
import socket
import time

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.teleop.devices import KeyboardSource, ScriptedSource, VuerXRSource
from dropbear_wbc.teleop.frames import (
    OperatorMapping, head_relative_to_xr, head_yaw_rot, xr_flat, xr_matrix, xr_to_head_relative,
)


def _pose(p, r=None):
    t = np.eye(4)
    t[:3, 3] = p
    if r is not None:
        t[:3, :3] = r
    return t


def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * k @ k


REF = {"left": _pose([0.10, 0.25, 0.20], _rot([0, 1, 0], 0.3)), "right": _pose([0.10, -0.25, 0.20])}


# ------------------------------------------------------------------------------------------------ scripted
def test_scripted_figure8_ramps_in_and_mirrors():
    src = ScriptedSource(REF, kind="figure8", amplitude=(0.04, 0.06, 0.05), period=6.0, ramp_s=2.0)
    src.start()
    first = src.get(10.0)  # t0 = 10 s -> tau 0
    for s in ("left", "right"):
        assert np.allclose(getattr(first, s), REF[s])
    prev = first
    max_step = 0.0
    for k in range(1, 1200):
        w = src.get(10.0 + k * 0.01)
        max_step = max(max_step, max(np.linalg.norm(getattr(w, s)[:3, 3] - getattr(prev, s)[:3, 3]) for s in ("left", "right")))
        dl = w.left[:3, 3] - REF["left"][:3, 3]
        dr = w.right[:3, 3] - REF["right"][:3, 3]
        assert np.allclose(dl * [1, -1, 1], dr, atol=1e-12)  # right hand mirrors the left (outward together)
        assert np.allclose(w.left[:3, :3], REF["left"][:3, :3])
        prev = w
    assert max_step < 0.003  # <= 0.3 m/s at 100 Hz
    tau = 7.5  # after the ramp: exact Lissajous
    w = src.get(10.0 + tau)
    om = 2 * math.pi / 6.0
    exp = np.array([0.04 * math.sin(0.5 * om * tau), 0.06 * math.sin(om * tau), 0.05 * math.sin(2 * om * tau)])
    assert np.allclose(w.left[:3, 3] - REF["left"][:3, 3], exp, atol=1e-12)


def test_scripted_pose_sequence():
    poses = {s: [_pose(REF[s][:3, 3] + [0.05, 0, 0]), _pose(REF[s][:3, 3] + [0, 0, -0.05])] for s in REF}
    src = ScriptedSource(REF, kind="poses", poses=poses, hold_s=1.0, move_s=0.5)
    src.start()
    src.get(0.0)
    w = src.get(0.5 + 0.9)  # end of the first hold
    assert np.allclose(w.left[:3, 3], poses["left"][0][:3, 3])
    w = src.get(2 * 1.5 + 5.0)  # past the end: last pose
    assert np.allclose(w.right, poses["right"][1])


# ------------------------------------------------------------------------------------------------ keyboard
def test_keyboard_jog_and_events():
    kb = KeyboardSource(REF, step_m=0.01, step_rad=0.1, max_offset_m=0.05, backend="none")
    for ch in "www" + "aa" + "e" + "i" + "[":
        kb.feed(ch)
    kb.feed("r")
    w = kb.get(0.0)
    assert np.allclose(w.left[:3, 3] - REF["left"][:3, 3], [0.03, 0.02, -0.01])
    assert np.allclose(w.right[:3, 3] - REF["right"][:3, 3], [0.01, 0.0, 0.0])
    x_axis = (REF["left"][:3, :3].T @ w.left[:3, :3])  # relative rotation = Rx(0.1)
    assert abs(math.atan2(x_axis[2, 1], x_axis[1, 1]) - 0.1) < 1e-12
    assert w.events == ["start"] and kb.get(0.0).events == []
    for _ in range(20):
        kb.feed("w")
    assert np.linalg.norm(kb.get(0.0).left[:3, 3] - REF["left"][:3, 3]) <= 0.05 + 1e-12  # clamped
    kb.feed("h")
    w = kb.get(0.0)
    assert np.allclose(w.left, REF["left"]) and w.events == ["home"]


# ------------------------------------------------------------------------------------------------ frames
@pytest.mark.parametrize("hand_tracking", [True, False])
def test_xr_roundtrip_and_head_yaw(hand_tracking):
    head = _pose([0.3, -0.2, 1.6], _rot([0, 0, 1], 0.7) @ _rot([0, 1, 0], 0.4) @ _rot([1, 0, 0], -0.2))
    rel_l = _pose([0.35, 0.25, -0.40], _rot([1, 1, 0], 0.5))
    rel_r = _pose([0.30, -0.22, -0.45], _rot([0, 1, 1], -0.3))
    hx, lx, rx = head_relative_to_xr(head, rel_l, rel_r, hand_tracking)
    h2, l2, r2 = xr_to_head_relative(hx, lx, rx, hand_tracking)
    assert np.allclose(h2, head) and np.allclose(l2, rel_l) and np.allclose(r2, rel_r)
    # head pitch/roll do not move the targets in head_yaw mode (only yaw + position)
    head2 = head.copy()
    head2[:3, :3] = head_yaw_rot(head[:3, :3]) @ _rot([0, 1, 0], -0.3)
    world_l = np.eye(4)
    ry = head_yaw_rot(head[:3, :3])
    world_l[:3, :3] = ry @ rel_l[:3, :3]
    world_l[:3, 3] = ry @ rel_l[:3, 3] + head[:3, 3]
    hx2, lx2, rx2 = head_relative_to_xr(head2, rel_l, rel_r, hand_tracking)
    _, l3, _ = xr_to_head_relative(hx2, lx2, rx2, hand_tracking)
    assert np.allclose(l3, rel_l)
    # column-major flattening as Vuer sends it
    assert np.allclose(xr_matrix(xr_flat(lx)), lx)


def test_operator_mapping_calibration():
    m = OperatorMapping(scale=0.7)
    rel = {"left": _pose([0.1, 0.2, -0.5]), "right": _pose([0.1, -0.2, -0.5])}
    info = m.calibrate(rel, REF)
    for s in ("left", "right"):
        assert np.allclose(m.apply(s, rel[s])[:3, 3], REF[s][:3, 3])
        moved = rel[s].copy()
        moved[:3, 3] += [0.1, 0, 0]
        assert np.allclose(m.apply(s, moved)[:3, 3] - REF[s][:3, 3], [0.07, 0, 0])
    assert info["scale"] == 0.7 and m.calibrated


# ------------------------------------------------------------------------------------------------ WebXR / Vuer
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _play_headset(port: int, events: list[dict], expect_index: bool = True) -> dict:
    import aiohttp
    import msgpack

    out = {}
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/") as r:
            out["index_status"] = r.status
            out["index_len"] = len(await r.read())
        async with http.get(f"http://127.0.0.1:{port}/static/test_teleop_devices.py") as r:
            out["static_status"] = r.status
        async with http.ws_connect(f"ws://127.0.0.1:{port}/") as ws:
            # the server's session main upserts the Hands / MotionControllers element
            first = await ws.receive(timeout=5.0)
            out["first_server_msg"] = msgpack.unpackb(first.data, raw=False) if first.type == aiohttp.WSMsgType.BINARY else None
            for ev in events:
                await ws.send_bytes(msgpack.packb(ev, use_bin_type=True))
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.2)
    return out


@pytest.mark.parametrize("hand_tracking,backend", [(True, "process"), (False, "process"), (False, "thread")])
def test_vuer_server_receives_simulated_headset(hand_tracking, backend):
    pytest.importorskip("vuer")
    pytest.importorskip("aiohttp")
    port = _free_port()
    mapping = OperatorMapping(scale=0.7)
    src = VuerXRSource(mapping=mapping, hand_tracking=hand_tracking, host="127.0.0.1", port=port, backend=backend)
    src.start()
    try:
        head = _pose([0.0, 0.0, 1.65], _rot([0, 0, 1], 0.4) @ _rot([0, 1, 0], 0.25))
        rel_l = _pose([0.30, 0.20, -0.45], _rot([0, 1, 0], 0.2))
        rel_r = _pose([0.28, -0.21, -0.47], _rot([1, 0, 0], -0.3))
        hx, lx, rx = head_relative_to_xr(head, rel_l, rel_r, hand_tracking)
        ts = int(time.time() * 1000)
        events = [{"etype": "CAMERA_MOVE", "ts": ts, "key": "defaultCamera", "value": {"camera": {"matrix": xr_flat(hx)}}}]
        if hand_tracking:
            state = {"pinch": False, "pinchValue": 0.1, "squeeze": False, "squeezeValue": 0.0}
            events.append({"etype": "HAND_MOVE", "ts": ts + 5, "key": "hands",
                           "value": {"left": xr_flat(lx) * 25, "right": xr_flat(rx) * 25,
                                     "leftState": state, "rightState": state}})
        else:
            st = {"trigger": False, "triggerValue": 0.0, "squeeze": False, "squeezeValue": 0.0, "thumbstick": False,
                  "thumbstickValue": [0.0, 0.0], "aButton": False, "bButton": False}
            events.append({"etype": "CONTROLLER_MOVE", "ts": ts + 5, "key": "motionControllers",
                           "value": {"left": xr_flat(lx), "right": xr_flat(rx), "leftState": st,
                                     "rightState": dict(st, aButton=True)}})
        assert src.get(0.0).valid is False  # nothing received yet
        info = asyncio.run(_play_headset(port, events))
        assert info["index_status"] == 200 and info["index_len"] > 100  # the web client is served locally
        assert info["static_status"] == 404                              # working directory not exposed
        assert src.counts["CAMERA_MOVE"] == 1
        assert src.counts["HAND_MOVE" if hand_tracking else "CONTROLLER_MOVE"] == 1
        w = src.get(1.0)
        assert w.valid
        assert np.allclose(w.left, mapping.apply("left", rel_l), atol=1e-6)
        assert np.allclose(w.right, mapping.apply("right", rel_r), atol=1e-6)
        if not hand_tracking:
            assert w.events == ["toggle"]  # right A rising edge
        # calibration maps the current operator pose onto the robot reference
        summary = src.calibrate(REF)
        w = src.get(1.1)
        for s in ("left", "right"):
            assert np.allclose(getattr(w, s)[:3, 3], REF[s][:3, 3], atol=1e-6)
        assert summary["scale"] == 0.7
        # stale data -> invalid (loop holds the arms)
        src.stale_s = 0.0
        time.sleep(0.01)
        assert src.get(2.0).valid is False
    finally:
        src.close()
