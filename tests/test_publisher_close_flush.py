"""Item 277: the last frame written before ChannelPublisher.Close() must reach the robot.

ZmqPublisher closed with linger 0 right after a write; when the process then exits, ZMQ's I/O thread may not have
sent the frame yet. For a LowCmd writer that last frame is the damping command (kp 0), so a quick exit left the
previous hold/policy gains on the robot. Measured on the box before the fix: a final frame followed by an immediate
close + exit arrived in 45/60 trials (20/30 idle, 25/30 with 16 busy-loop processes); 60/60 with a 50 ms hold.
This flaked tests/test_lowcmd_stop_signals.py for the low-level example, whose Damp() is followed directly by Close().

The subscriber binds an OS-assigned port: this test runs ~14 s, and a fixed port (was 27901) failed with EADDRINUSE
whenever a second suite in another venv ran it at the same time. That collision, not a lost frame, was the one
failure seen with two concurrent suites (item 286); the 50 ms hold lost 0/120 final frames with 0, 16 and 32 busy
processes on 8 CPUs; with no hold it lost 7/40 idle and 25/40 with 16 busy processes.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import zmq

import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk.transport import ZmqPublisher

ROOT = Path(__file__).resolve().parents[1]
TRIALS = 20

CHILD = r"""
import sys, time
from dropbear_wbc.sdk.transport import ZmqPublisher
port, tid = int(sys.argv[1]), int(sys.argv[2]).to_bytes(4, "little")
pub = ZmqPublisher(f"tcp://127.0.0.1:{port}", "rt/lowcmd", bind=False)
for _ in range(150):  # live 500 Hz command stream, gains > 0
    pub.publish(tid + b"\x01" * 1400)
    time.sleep(0.002)
pub.publish(tid + b"\x00" * 1400)  # final damping frame
pub.close()  # then exit at once, like a writer finishing its finally block
"""


def test_close_waits_for_the_last_frame_but_not_when_idle():
    pub = ZmqPublisher("inproc://flush-idle", "rt/lowcmd", bind=True)
    import time

    t0 = time.perf_counter()
    pub.close()
    assert time.perf_counter() - t0 < 0.02, "close() without a recent send must not wait"
    pub = ZmqPublisher("inproc://flush-recent", "rt/lowcmd", bind=True, flush_s=0.05)
    pub.publish(b"x")
    t0 = time.perf_counter()
    pub.close()
    assert time.perf_counter() - t0 >= 0.04


@pytest.mark.skipif(os.name != "posix", reason="subprocess exit timing measured on POSIX")
def test_final_frame_survives_close_and_exit():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "source"), os.environ.get("PYTHONPATH", "")]))
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 300)
    sub.setsockopt(zmq.LINGER, 0)
    sub.bind("tcp://127.0.0.1:*")  # OS-assigned free port: no clash with a concurrent run of this test
    port = int(sub.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])
    lost = []
    try:
        for trial in range(1, TRIALS + 1):
            tid = trial.to_bytes(4, "little")
            proc = subprocess.Popen([sys.executable, "-c", CHILD, str(port), str(trial)], env=env, cwd=ROOT)
            last = None
            while True:
                try:
                    msg = sub.recv()
                except zmq.Again:
                    if proc.poll() is not None:
                        break
                    continue
                body = msg[msg.index(tid):] if tid in msg else None
                if body is not None:
                    last = body
            assert proc.wait(timeout=10) == 0
            if last is None or last[4] != 0:
                lost.append(trial)
    finally:
        sub.close(linger=0)
    assert not lost, f"final (damping) frame lost on close+exit in {len(lost)}/{TRIALS} trials: {lost}"
