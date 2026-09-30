"""CPU-only unit tests for the Newton bridge pieces: IMU/LowState assembly and the motor PD kernel.

The PD-kernel test runs warp on the CPU device in a subprocess with CUDA hidden
(``CUDA_VISIBLE_DEVICES=-1``), so it never touches the GPU (no GPU lock needed).
It is skipped when warp is not installed (e.g. system Python).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.newton_sim.imu import LowStateAssembler, rotate_inv_wxyz
from dropbear_wbc.sdk.types import LowState

ROOT = Path(__file__).resolve().parents[1]


def _readout(tick, quat=(1, 0, 0, 0), lin_v=(0, 0, 0), ang_v=(0, 0, 0), dq=None):
    return SimpleNamespace(tick=tick, time_s=tick * 0.002, motor_q=np.zeros(22),
                           motor_dq=np.zeros(22) if dq is None else np.asarray(dq, float), motor_tau=np.ones(22),
                           neck_q=np.zeros(6), neck_dq=np.zeros(6), root_pos_w=np.array([0, 0, 1.0]),
                           root_quat_wxyz=np.asarray(quat, float), root_lin_vel_w=np.asarray(lin_v, float),
                           root_ang_vel_w=np.asarray(ang_v, float), body_pos_w=np.zeros((0, 3)),
                           body_quat_wxyz=np.zeros((0, 4)))


def test_imu_at_rest_reads_plus_g_and_body_gyro():
    asm = LowStateAssembler(0.002)
    angle = np.pi / 2  # 90 deg roll about +x
    q = (np.cos(angle / 2), np.sin(angle / 2), 0, 0)
    st = asm.build(_readout(1, quat=q, ang_v=(0, 0, 1.0)), np.ones(22))
    # Body y axis points to world z after a +90 deg roll, so gravity reaction appears on body +y.
    np.testing.assert_allclose(st.imu.accel, [0, 9.81, 0], atol=1e-5)
    np.testing.assert_allclose(st.imu.gyro, [0, 1.0, 0], atol=1e-6)  # world z spin seen on body +y
    np.testing.assert_allclose(st.imu.rpy, [angle, 0, 0], atol=1e-6)
    assert LowState.from_bytes(st.to_bytes()).sim.root_pos_w[2] == 1.0


def test_imu_finite_difference_accel_and_ddq():
    asm = LowStateAssembler(0.002)
    asm.build(_readout(1, lin_v=(0, 0, 0), dq=np.zeros(22)), np.ones(22))
    st = asm.build(_readout(2, lin_v=(0.02, 0, 0), dq=np.full(22, 0.004)), np.ones(22))
    np.testing.assert_allclose(st.imu.accel, [10.0, 0, 9.81], atol=1e-4)
    np.testing.assert_allclose(st.motor.ddq, 2.0, atol=1e-5)
    np.testing.assert_allclose(rotate_inv_wxyz(np.array([1.0, 0, 0, 0]), np.array([1.0, 2, 3])), [1, 2, 3])


PD_SCRIPT = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
os.environ["WARP_CACHE_PATH"] = sys.argv[2]
from dropbear_wbc.newton_sim.plant import configure_warp_cache, _kernels
configure_warp_cache()
import numpy as np, warp as wp
wp.init()
pd, _ = _kernels()
n = 4
f = lambda v: wp.array(np.asarray(v, np.float32), dtype=float, device="cpu")
i = lambda v: wp.array(np.asarray(v, np.int32), dtype=int, device="cpu")
joint_q = f([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
joint_qd = f([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
q_idx, qd_idx = i([7, 1, 3, 5]), i([6, 0, 2, 4])
q_des, dq_des, tau_ff = f([1.0, 0.1, 0.5, 0.0]), f([0.0, 0.0, 1.0, 0.0]), f([0.5, 0.0, 0.0, 0.0])
kp, kd, en, lim = f([10.0, 100.0, 1000.0, 5.0]), f([1.0, 2.0, 3.0, 4.0]), i([1, 1, 1, 0]), f([100.0, 100.0, 50.0, 100.0])
joint_f, tau = wp.zeros(7, dtype=float, device="cpu"), wp.zeros(n, dtype=float, device="cpu")
_one = wp.array(np.ones(n, np.float32), dtype=float, device="cpu")  # hw-law arrays (unused: hw = 0, legacy law)
wp.launch(pd, dim=n, device="cpu", inputs=[joint_q, joint_qd, q_idx, qd_idx, q_des, dq_des, tau_ff, kp, kd, en, lim,
                                           0, _one, _one, _one, _one],
          outputs=[joint_f, tau])
print(json.dumps({"tau": tau.numpy().tolist(), "joint_f": joint_f.numpy().tolist(),
                  "device_count_cuda": wp.get_cuda_device_count()}))
"""


def test_motor_pd_kernel_law_clip_and_disable(tmp_path):
    pytest.importorskip("warp")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    out = subprocess.run([sys.executable, "-c", PD_SCRIPT, str(ROOT / "source"), str(ROOT / ".warp-cache")],
                         capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["device_count_cuda"] == 0
    # motor 0: q=0.7 dq=6.0 -> 0.5 + 10*(1-0.7) + 1*(0-6) = -2.5
    # motor 1: q=0.1 dq=0.0 -> 100*(0.1-0.1) + 2*(0-0) = 0
    # motor 2: q=0.3 dq=2.0 -> 1000*(0.5-0.3) + 3*(1-2) = 197 -> clipped to 50
    # motor 3: disabled -> 0
    np.testing.assert_allclose(res["tau"], [-2.5, 0.0, 50.0, 0.0], atol=1e-5)
    jf = np.asarray(res["joint_f"])
    np.testing.assert_allclose(jf[[6, 0, 2, 4]], [-2.5, 0.0, 50.0, 0.0], atol=1e-5)
    assert jf[[1, 3, 5]].tolist() == [0.0, 0.0, 0.0]


def _bridge_module(monkeypatch, lock_path: Path):
    """tools/newton_bridge.py with its GpuLock pointed at ``lock_path`` (never the team lock)."""
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    import newton_bridge
    from dropbear_wbc.newton_sim.gpu_lock import GpuLock

    monkeypatch.setattr(newton_bridge, "GpuLock", lambda owner: GpuLock(owner, path=lock_path, poll_s=0.01,
                                                                          timeout_s=0.05))
    return newton_bridge


def test_bridge_gpu_lock_held_requires_the_callers_lock(tmp_path, monkeypatch):
    nb = _bridge_module(monkeypatch, tmp_path / "gpu.lock")
    report = tmp_path / "r.json"
    assert nb.main(["--gpu-lock-held", "--duration", "0.01", "--report", str(report)]) == 1
    rep = json.loads(report.read_text())
    assert rep["status"] == "failed" and "--gpu-lock-held given but" in rep["error"]


def test_bridge_gpu_lock_held_never_touches_the_callers_lock(tmp_path, monkeypatch):
    import types

    lock = tmp_path / "gpu.lock"
    lock.write_text(json.dumps({"owner": "gpu_lock_run", "pid": 1, "token": "abc"}))
    fake_plant = types.ModuleType("dropbear_wbc.newton_sim.plant")
    fake_plant.DEFAULT_USD = Path("x.usd")
    fake_plant.PlantConfig = lambda **kw: SimpleNamespace(**kw)

    def _no_sim(*a, **k):
        raise RuntimeError("stop before the simulator starts")

    fake_plant.DropbearNewtonPlant = _no_sim
    monkeypatch.setitem(sys.modules, "dropbear_wbc.newton_sim.plant", fake_plant)
    nb = _bridge_module(monkeypatch, lock)
    report = tmp_path / "r.json"
    assert nb.main(["--gpu-lock-held", "--duration", "0.01", "--report", str(report)]) == 1
    rep = json.loads(report.read_text())
    assert rep["gpu_lock"].startswith("held by caller") and "gpu_lock_wait_s" not in rep
    assert "stop before the simulator starts" in rep["error"]
    assert lock.exists()  # the wrapper (tools/gpu_lock_run.py) owns and releases it


def test_target_ramp_matches_datasheet_motor_law():
    """``sdk.target_ramp.TargetRamp`` = ``DatasheetMotor._interpolate``: linear over n ticks, ending on the target;
    repeated commands do not restart it; snap / cancel jump (docs/ISSUES.md #11)."""
    import numpy as np

    from dropbear_wbc.sdk.target_ramp import TargetRamp

    r = TargetRamp.from_ms(20.0, 0.005)
    assert r.n == 4
    np.testing.assert_allclose(r.on_command(np.zeros(2)), 0.0)  # first command snaps
    assert r.tick() is None
    np.testing.assert_allclose(r.on_command(np.ones(2)), 0.0)  # ramp starts from the applied target
    seen = [r.tick()[0] for _ in range(4)]
    np.testing.assert_allclose(seen, [0.25, 0.5, 0.75, 1.0])
    assert r.tick() is None
    r.on_command(np.full(2, 3.0))
    r.tick()  # 1.5
    r.on_command(np.full(2, 3.0))  # repeated: keeps ramping
    np.testing.assert_allclose(r.tick(), 2.0)
    r.on_command(np.full(2, -1.0))  # a new target mid-ramp starts from where the ramp is
    np.testing.assert_allclose(r.tick(), 2.0 + 0.25 * (-3.0))
    r.cancel()
    np.testing.assert_allclose(r.on_command(np.full(2, 7.0)), 7.0)
    assert TargetRamp.from_ms(0.0, 0.002).n == 1


def test_hw_motor_law_matches_datasheet_motor_and_the_plant_kernel():
    """``hw_motor_specs.hw_motor_torque`` == ``DatasheetMotor._clip_effort`` math, and the Newton plant's warp kernel
    (``--motor-profile``) == ``hw_motor_torque`` (warp on CPU; skipped where warp is missing)."""
    import numpy as np
    import pytest

    from dropbear_wbc.robots.hw_motor_specs import hw_motor_torque

    peak, sat, v0, cou, vis = 60.0, 170.0, 6.3, 0.8, 0.05
    tau_pd = np.array([500.0, -500.0, 30.0, 59.0, -59.0, 200.0])
    dq = np.array([0.0, 0.0, 5.0, 6.0, -6.0, 9.0])
    applied, clipped = hw_motor_torque(tau_pd, dq, peak, sat, v0, cou, vis)
    # envelope by hand (Isaac DCMotor): top = min(sat (1 - v/v0), peak)
    ve = v0 * (1 + peak / sat)
    v = np.clip(dq, -ve, ve)
    np.testing.assert_allclose(clipped, np.clip(tau_pd, np.maximum(sat * (-1 - v / v0), -peak),
                                                np.minimum(sat * (1 - v / v0), peak)))
    assert clipped[0] == 60.0 and clipped[1] == -60.0 and clipped[5] < 1e-9  # beyond the no-load speed: no drive
    np.testing.assert_allclose(applied, clipped - (cou * np.tanh(dq / 0.05) + vis * dq))

    wp = pytest.importorskip("warp")
    from dropbear_wbc.newton_sim.plant import _kernels

    wp.init()
    kernel, _ = _kernels()
    n = len(tau_pd)
    f = lambda x: wp.array(np.asarray(x, np.float32), dtype=float, device="cpu")  # noqa: E731
    i = lambda x: wp.array(np.asarray(x, np.int32), dtype=int, device="cpu")  # noqa: E731
    kp = 100.0
    q_des = tau_pd / kp  # q = 0, dq_des = dq  -> kernel PD demand == tau_pd
    joint_f, tau_out = wp.zeros(n, dtype=float, device="cpu"), wp.zeros(n, dtype=float, device="cpu")
    wp.launch(kernel, dim=n, device="cpu",
              inputs=[f(np.zeros(n)), f(dq), i(range(n)), i(range(n)), f(q_des), f(dq), f(np.zeros(n)), f([kp] * n),
                      f([1.0] * n), i([1] * n), f([peak] * n), 1, f([sat] * n), f([v0] * n), f([cou] * n), f([vis] * n)],
              outputs=[joint_f, tau_out])
    np.testing.assert_allclose(tau_out.numpy(), clipped, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(joint_f.numpy(), applied, rtol=1e-5, atol=1e-3)
