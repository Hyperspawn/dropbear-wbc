"""Bind a live :class:`~dropbear_wbc.motion.stream.MotionStream` to the tracking env's motion command.

The single-clip ``MotionCommand`` reads its reference from ``MotionLoader`` tensors indexed by ``time_steps``. Here
those tensors are replaced by stream-sized buffers (capacity frames), so new clips spliced into the stream appear in
the running reference: the policy keeps stepping, the command keeps advancing one frame per step, and the next frames
are whatever was spliced last (text -> motion, a behaviour script, teleop). Only the changed frame range is copied to
the device; the hold after the last clip (last pose, zero velocity) is filled on the device.

Command sources (``poll(t)`` once per policy step):

* a timed script ``[(t_s, npz), ...]`` (``scripts/play.py --live_script "0:a.npz,2.5:b.npz"``);
* a watched folder (``--live_dir``): every new ``*.npz`` (contract motion NPZ) dropped in is spliced when it appears,
  in name order. This is the hook for a text-to-motion generator running in another process.

After a fall (the env resets the robot to reference frame 0) call ``restart()``: the reference starts over at the idle
clip, so the robot gets up standing and waits for the next command rather than replaying the past ones.

A clip is spliced ``lead_s`` ahead of the current reference frame (the time the device copy may take) and cross-faded
over ``blend_s``; after it, the reference blends (``idle_blend_s``) into the idle clip (default: the first clip, a
stand) placed where the clip ended, so the robot does not freeze mid-stride. Import after the Isaac Sim app is
running.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from dropbear_wbc.motion.stream import Clip, MotionStream
from dropbear_wbc.robots.dropbear_names import ROOT_BODY
from dropbear_wbc.tasks.tracking.motion_npz import load_motion_npz


class LiveReference:
    def __init__(self, command, first_npz: str | Path, capacity_s: float = 300.0, lead_s: float = 0.1,
                 blend_s: float = 0.4, script: list[tuple[float, str]] | None = None, watch_dir: str | Path | None = None,
                 idle_npz: str | Path | None = None, idle_blend_s: float = 0.8):
        self.cmd = command
        m = command.motion
        first = load_motion_npz(first_npz)
        self.root = first.body_names.index(ROOT_BODY)
        self.stream = MotionStream(first.fps, Clip.from_arrays(first), self.root, capacity_s=capacity_s)
        self.lead = int(round(lead_s * first.fps))
        self.blend_s = float(blend_s)
        self.script = sorted(script or [], key=lambda e: e[0])
        self.watch = Path(watch_dir) if watch_dir else None
        self._seen: set[str] = set(p.name for p in self.watch.glob("*.npz")) if self.watch and self.watch.is_dir() else set()
        self.events: list[dict] = []
        # after every commanded clip the reference settles into this idle clip (default: the first clip, e.g. a stand)
        # instead of freezing on the clip's last frame (a walk would freeze mid-stride)
        self.idle = Clip.from_arrays(load_motion_npz(idle_npz or first_npz))
        self.idle_blend_s = float(idle_blend_s)
        dev = m.full_joint_pos.device
        self._motor_idx = torch.as_tensor(
            [first.joint_names.index(n) for n in first.motor_names], dtype=torch.long, device=dev)
        s = self.stream
        m.full_joint_pos = torch.zeros(s.joint_pos.shape, dtype=torch.float32, device=dev)
        m.full_joint_vel = torch.zeros_like(m.full_joint_pos)
        m._body_pos_w = torch.zeros(s.body_pos_w.shape, dtype=torch.float32, device=dev)
        m._body_quat_w = torch.zeros(s.body_quat_w.shape, dtype=torch.float32, device=dev)
        m._body_lin_vel_w = torch.zeros(s.body_lin_vel_w.shape, dtype=torch.float32, device=dev)
        m._body_ang_vel_w = torch.zeros(s.body_ang_vel_w.shape, dtype=torch.float32, device=dev)
        m.time_step_total = s.capacity
        self._sync(full=True)
        print(f"[live] armed: {len(self.script)} scripted clips, watching {self.watch or '-'}", flush=True)

    # ------------------------------------------------------------------ device sync
    def _sync(self, full: bool = False) -> None:
        m, s = self.cmd.motion, self.stream
        lo, hi = (0, s.capacity) if full else s.dirty

        def put(dst: torch.Tensor, src: np.ndarray) -> None:
            dst[lo:hi] = torch.as_tensor(src[lo:hi], dtype=torch.float32, device=dst.device)

        put(m.full_joint_pos, s.joint_pos)
        put(m.full_joint_vel, s.joint_vel)
        put(m._body_pos_w, s.body_pos_w)
        put(m._body_quat_w, s.body_quat_w)
        put(m._body_lin_vel_w, s.body_lin_vel_w)
        put(m._body_ang_vel_w, s.body_ang_vel_w)
        e = s.clip_end
        if not full and e < s.capacity:  # the hold after the clip: last pose, zero velocity (device-side fill)
            m.full_joint_pos[e:] = m.full_joint_pos[e - 1]
            m._body_pos_w[e:] = m._body_pos_w[e - 1]
            m._body_quat_w[e:] = m._body_quat_w[e - 1]
            for t in (m.full_joint_vel, m._body_lin_vel_w, m._body_ang_vel_w):
                t[e:] = 0.0
        m.joint_pos = m.full_joint_pos[:, self._motor_idx].contiguous()
        m.joint_vel = m.full_joint_vel[:, self._motor_idx].contiguous()
        m.root_pos_w = m._body_pos_w[:, self.root].contiguous()
        m.root_quat_w = m._body_quat_w[:, self.root].contiguous()
        m.root_lin_vel_w = m._body_lin_vel_w[:, self.root].contiguous()
        m.root_ang_vel_w = m._body_ang_vel_w[:, self.root].contiguous()

    # ------------------------------------------------------------------ commands
    @property
    def frame(self) -> int:
        return int(self.cmd.time_steps[0])

    def play(self, npz: str | Path, source: str = "") -> dict:
        """Splice ``npz`` into the running reference now (+ lead). Returns the event record."""
        t0 = time.perf_counter()
        arrays = load_motion_npz(npz)
        if abs(arrays.fps - self.stream.fps) > 1e-6:
            raise ValueError(f"{npz}: fps {arrays.fps} != stream fps {self.stream.fps}")
        at = self.frame + self.lead
        start, end = self.stream.splice(Clip.from_arrays(arrays), at, blend_s=self.blend_s)
        self.stream.splice(self.idle, end, blend_s=self.idle_blend_s)
        lo = self.stream.dirty[1]
        self.stream.dirty = (max(0, start - 1), lo)  # both splices in one device copy
        self._sync()
        ev = {"clip": Path(npz).stem, "source": source, "frame": start, "end": end,
              "latency_ms": round(1e3 * (time.perf_counter() - t0), 1)}
        self.events.append(ev)
        print(f"[live] t={start / self.stream.fps:6.2f}s splice {ev['clip']} ({(end - start) / self.stream.fps:.1f} s, "
              f"{ev['latency_ms']} ms)", flush=True)
        return ev

    def restart(self, reason: str = "fall") -> None:
        """After a reset of the robot (a fall): the reference restarts at the idle clip and waits for the next command,
        instead of replaying the clips already played (the command clock is back at frame 0)."""
        s = self.stream
        self.stream = MotionStream(s.fps, self.idle, self.root, capacity_s=s.capacity / s.fps)
        self._sync(full=True)
        self.events.append({"clip": "<idle restart>", "source": reason, "frame": 0, "end": 0, "latency_ms": 0.0})
        print(f"[live] {reason}: reference restarted at the idle clip", flush=True)

    def poll(self, t_s: float) -> None:
        """Run due script events and new files in the watched folder (call once per policy step)."""
        while self.script and self.script[0][0] <= t_s + 1e-9:
            _, npz = self.script.pop(0)
            self.play(npz, source="script")
        if self.watch is not None and self.watch.is_dir():
            for p in sorted(self.watch.glob("*.npz")):
                if p.name not in self._seen:
                    self._seen.add(p.name)
                    try:
                        self.play(p, source="watch_dir")
                    except Exception as exc:  # noqa: BLE001 - a bad clip must not stop the robot
                        print(f"[live] rejected {p.name}: {type(exc).__name__}: {exc}", flush=True)


def parse_live_script(spec: str) -> list[tuple[float, str]]:
    """``"0:a.npz,2.5:b.npz"`` -> [(0.0, 'a.npz'), (2.5, 'b.npz')] (paths may contain ':' after a drive letter)."""
    out = []
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        t, path = item.split(":", 1)
        out.append((float(t), path))
    return out
