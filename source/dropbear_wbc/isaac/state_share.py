"""Latest-robot-state channel between two processes (Isaac-free; numpy only).

Why: one Isaac process cannot both simulate Dropbear in real time and draw a GUI viewport. On the RTX 4080 laptop the
motor-twin physics runs at ~1.1x real time on the CPU pipeline (``scripts/play.py --device cpu --realtime``), but each
viewport frame costs ~28 ms in the same loop (0.43x). So the physics process publishes env 0's state here every
policy step and ``scripts/live_viewer.py`` (a second, GUI process) poses a kinematic copy of the robot from it at its
own frame rate.

Layout: a memory-mapped float64 file ``[seq, sim_t, root_pos(3), root_quat_wxyz(4), joint_pos(J), body_pos(B*3),
body_quat_wxyz(B*4)]`` (positions env-local) and a JSON sidecar (``<path>.json``) with the joint and body names. The
body poses (optional, B may be 0) let a viewer that has no Isaac articulation (``tools/live_viewer_gl.py``, Newton
OpenGL) place every link directly. ``seq`` is a seqlock: odd while a write is in progress, so a
reader retries instead of seeing a torn row. Only the newest state is kept (a viewer that falls behind skips frames).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_HDR = 2  # seq, sim_t


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".json")


class StateWriter:
    def __init__(self, path: Path | str, joint_names: list[str], body_names: list[str] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = len(joint_names)
        self.nb = len(body_names or [])
        size = _HDR + 3 + 4 + self.n + 7 * self.nb
        _sidecar(self.path).write_text(json.dumps({
            "joint_names": list(joint_names), "body_names": list(body_names or []), "size": size,
            "layout": "seq, sim_t, root_pos[3], root_quat_wxyz[4], joint_pos[J], body_pos[B*3], body_quat_wxyz[B*4]"}),
            encoding="utf-8")
        self.buf = np.memmap(self.path, dtype=np.float64, mode="w+", shape=(size,))
        self.buf[:] = 0.0
        self.seq = 0

    def write(self, sim_t: float, root_pos, root_quat_wxyz, joint_pos, body_pos=None, body_quat_wxyz=None) -> None:
        b = self.buf
        self.seq += 1
        b[0] = self.seq  # odd: writing
        b[1] = sim_t
        b[2:5] = np.asarray(root_pos, dtype=np.float64).reshape(3)
        b[5:9] = np.asarray(root_quat_wxyz, dtype=np.float64).reshape(4)
        j1 = 9 + self.n
        b[9:j1] = np.asarray(joint_pos, dtype=np.float64).reshape(self.n)
        if self.nb:
            b[j1:j1 + 3 * self.nb] = np.asarray(body_pos, dtype=np.float64).reshape(-1)
            b[j1 + 3 * self.nb:j1 + 7 * self.nb] = np.asarray(body_quat_wxyz, dtype=np.float64).reshape(-1)
        self.seq += 1
        b[0] = self.seq  # even: complete


class StateReader:
    def __init__(self, path: Path | str, wait_s: float = 600.0):
        self.path = Path(path)
        t0 = time.time()
        while not (self.path.is_file() and _sidecar(self.path).is_file()):
            if time.time() - t0 > wait_s:
                raise TimeoutError(f"no state file {self.path} after {wait_s:.0f} s (start the physics process)")
            time.sleep(0.2)
        meta = json.loads(_sidecar(self.path).read_text(encoding="utf-8"))
        self.joint_names: list[str] = meta["joint_names"]
        self.body_names: list[str] = meta.get("body_names", [])
        self.n, self.nb = len(self.joint_names), len(self.body_names)
        self.buf = np.memmap(self.path, dtype=np.float64, mode="r", shape=(int(meta["size"]),))
        self.last_seq = -1.0

    def read(self) -> dict | None:
        """The newest complete state, or None if nothing new since the last read (or a write is in progress)."""
        for _ in range(100):
            s1 = float(self.buf[0])
            if s1 == self.last_seq or s1 <= 0:
                return None
            if int(s1) % 2:
                continue
            row = np.array(self.buf)
            if float(self.buf[0]) == s1:
                self.last_seq = s1
                j1, nb = 9 + self.n, self.nb
                return {"seq": int(s1), "sim_t": float(row[1]), "root_pos": row[2:5], "root_quat_wxyz": row[5:9],
                        "joint_pos": row[9:j1], "body_pos": row[j1:j1 + 3 * nb].reshape(nb, 3),
                        "body_quat_wxyz": row[j1 + 3 * nb:j1 + 7 * nb].reshape(nb, 4)}
        return None
