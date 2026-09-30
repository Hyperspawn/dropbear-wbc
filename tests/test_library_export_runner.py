"""The policy runner feeds a motion-LIBRARY export (no baked reference, sidecar ``motion.source: runtime``) from ANY
contract NPZ given at runtime (``tools/policy_runner.py --motion``), with the exact observation layout of training.

The export is made by the REAL ``tasks/tracking/export_library.py`` via ``scripts/emulate_tracking_export.py``
(Isaac stubs, mocked env handles, the real ``accepted_v0`` library report, a random untrained rsl-rl ActorCritic).
The controller is stepped on LowStates that put the robot exactly on the fed NPZ, and every observation slice is
checked against the NPZ -- including the ``-Future`` command (+5/+10 frames, clamped at the clip end) -- the action
against the plain ``policy.onnx`` and the LowCmd against the target law. Fail-closed cases: no ``--motion``, rejected
clip, other ankle variant, other plant USD, other frame rate. Needs torch + onnx + onnxruntime (system Python).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401

ROOT = sdk_test_paths.ROOT
for p in (ROOT / "tools", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MANIFEST = ROOT / "data" / "motions" / "libraries" / "accepted_v0.json"
WAVE = ROOT / "data" / "motions" / "synthetic" / "wave_right.npz"
SQUAT = ROOT / "data" / "motions" / "synthetic" / "squat_lite.npz"
REJECTED = ROOT / "data" / "motions" / "kimodo_g1" / "output_wave.npz"
FUTURE = (5, 10)


def _export(tmp_path_factory, future):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    for p in (MANIFEST, WAVE, SQUAT):
        if not p.is_file():
            pytest.skip(f"{p} not present")
    import emulate_tracking_export as emu

    out = tmp_path_factory.mktemp("library_export_future" if future else "library_export")
    emu.emulate_library_export(out, MANIFEST, seed=5, future_steps=future)
    return out


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    return _export(tmp_path_factory, ())


@pytest.fixture(scope="module")
def export_future_dir(tmp_path_factory):
    return _export(tmp_path_factory, FUTURE)


def _npz(path: Path):
    d = np.load(path, allow_pickle=False)
    return d, [str(j) for j in d["joint_names"]], [str(b) for b in d["body_names"]]


def _state_on_reference(npz: Path, k: int, tick: int):
    """LowState with the robot exactly at frame ``k`` of ``npz`` (root = articulation root 'world')."""
    from dropbear_wbc.deploy import quat as Q
    from dropbear_wbc.sdk import motors
    from dropbear_wbc.sdk.types import IMUState, LowState, MotorStateBlock, SimState

    d, joints, bodies = _npz(npz)
    ids = [joints.index(m) for m in motors.MOTOR_NAMES]
    r = bodies.index("world")
    q = np.asarray(d["body_quat_w"][k, r], float)
    f32 = lambda v: np.asarray(v, np.float32)  # noqa: E731
    st = LowState(tick=tick, stamp_ns=1)
    st.imu = IMUState(quat_wxyz=f32(q), gyro=f32(Q.rotate_inv(q, d["body_ang_vel_w"][k, r])),
                      accel=f32([0, 0, 9.81]), rpy=np.zeros(3, np.float32))
    st.motor = MotorStateBlock.zeros(motors.NUM_MOTORS)
    st.motor.q[:] = f32(d["joint_pos"][k, ids])
    st.motor.dq[:] = f32(d["joint_vel"][k, ids])
    st.sim = SimState(time_s=k * 0.02, root_pos_w=f32(d["body_pos_w"][k, r]), root_quat_w=f32(q),
                      root_lin_vel_w=f32(d["body_lin_vel_w"][k, r]), root_ang_vel_w=f32(d["body_ang_vel_w"][k, r]))
    return st


def _build(export: Path, *extra: str):
    import policy_runner

    args = policy_runner.parse_args(["--mode", "policy", "--sidecar", str(export / "policy.json"), "--allow-privileged",
                                     *extra])
    return policy_runner.build(args)


def test_library_sidecar_is_a_runtime_reference_export(export_dir, export_future_dir):
    from dropbear_wbc.deploy.config import load_sidecar

    for d, future, obs_dim in ((export_dir, [], 125), (export_future_dir, list(FUTURE), 125 + 88)):
        assert not (d / "policy_motion.onnx").exists()
        side = json.loads((d / "policy.json").read_text(encoding="utf-8"))
        cfg = load_sidecar(d / "policy.json")
        assert cfg.policy_path.name == "policy.onnx" and cfg.onnx_time_input is None
        assert cfg.motion.source == "runtime" and cfg.motion.file is None
        assert cfg.obs_dim == obs_dim == sum(t.dim for t in cfg.observations)
        assert cfg.observations[0].func == "motion_command"
        assert list(cfg.observations[0].params.get("future_steps", [])) == future
        req = cfg.motion.requirements
        assert req["fps"] == 50.0 and req["authored_ankle_tierods"] is False and req["usd_sha256"].startswith("45586414")
        assert side["dropbear_tracking"]["export_type"] == "library_runtime_reference"
        assert side["motion"]["library"]["num_clips"] == 5
        assert cfg.motion.anchor_offset_quat == pytest.approx((0.5, 0.5, 0.5, 0.5), abs=1e-4)


def test_runtime_export_without_motion_fails_closed(export_dir):
    with pytest.raises(SystemExit, match="--motion"):
        _build(export_dir)


@pytest.mark.parametrize("clip", [WAVE, SQUAT])
def test_runner_feeds_any_npz_with_training_layout(export_dir, clip):
    import onnxruntime as ort

    from dropbear_wbc.deploy.fsm import FsmState
    from dropbear_wbc.deploy.motion import NpzMotion
    from dropbear_wbc.sdk import motors

    cfg, ctrl = _build(export_dir, "--motion", str(clip))
    assert isinstance(ctrl.motion, NpzMotion) and ctrl.motion.cfg.file == clip.resolve()
    rep = cfg.meta["runtime_motion"]
    assert rep["checks"]["validation"] == "accepted" and rep["num_frames"] == ctrl.motion.num_frames
    plain = ort.InferenceSession(str(export_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
    d, joints, _ = _npz(clip)
    ids = [joints.index(m) for m in motors.MOTOR_NAMES]
    ctrl.request(FsmState.POLICY, _state_on_reference(clip, 0, 1000), 0.0)
    prev = np.zeros(22)
    for k in range(4):
        st = _state_on_reference(clip, k, 1000 + 10 * k)
        cmd, info = ctrl.step(st, 0.02 * k)
        assert info.state == FsmState.POLICY and info.motion_frame == k
        obs = info.obs
        assert obs.shape == (125,)
        np.testing.assert_allclose(obs[0:22], d["joint_pos"][k, ids], atol=1e-5)
        np.testing.assert_allclose(obs[22:44], d["joint_vel"][k, ids], atol=1e-5)
        assert np.abs(obs[44:47]).max() < 1e-4  # robot on the reference: anchor offset ~0
        np.testing.assert_allclose(obs[47:53], [1, 0, 0, 1, 0, 0], atol=1e-5)
        np.testing.assert_allclose(obs[59:81], d["joint_pos"][k, ids] - cfg.default_joint_pos, atol=1e-5)
        np.testing.assert_allclose(obs[103:125], prev, atol=1e-6)
        (ref,) = plain.run(["actions"], {"obs": obs[None]})
        np.testing.assert_allclose(info.action, ref[0], atol=1e-5)
        np.testing.assert_allclose(cmd.motor.q, cfg.action_offset + cfg.action_scale * info.action, atol=1e-5)
        prev = info.action


def test_future_command_window_and_clamping(export_future_dir):
    import onnxruntime as ort

    from dropbear_wbc.deploy.fsm import FsmState
    from dropbear_wbc.sdk import motors

    cfg, ctrl = _build(export_future_dir, "--motion", str(WAVE))
    d, joints, _ = _npz(WAVE)
    ids = [joints.index(m) for m in motors.MOTOR_NAMES]
    T = d["joint_pos"].shape[0]
    plain = ort.InferenceSession(str(export_future_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
    ctrl.request(FsmState.POLICY, _state_on_reference(WAVE, 0, 1000), 0.0)
    for k in (0, 1):
        _, info = ctrl.step(_state_on_reference(WAVE, k, 1000 + 10 * k), 0.02 * k)
        obs = info.obs
        assert obs.shape == (213,)
        for j, off in enumerate(FUTURE):  # [q_t, dq_t, q_t+5, dq_t+5, q_t+10, dq_t+10], then the 125-44 other terms
            np.testing.assert_allclose(obs[44 + 44 * j: 66 + 44 * j], d["joint_pos"][k + off, ids], atol=1e-5)
            np.testing.assert_allclose(obs[66 + 44 * j: 88 + 44 * j], d["joint_vel"][k + off, ids], atol=1e-5)
        (ref,) = plain.run(["actions"], {"obs": obs[None]})
        np.testing.assert_allclose(info.action, ref[0], atol=1e-5)
    # near the clip end the future frames clamp to the last frame (training: min(t + k, T - 1))
    ctrl._policy_step = T - 7
    ctx = ctrl._ctx(_state_on_reference(WAVE, T - 7, 5000))
    obs = ctrl.obs_builder.compute(ctx)
    np.testing.assert_allclose(obs[44:66], d["joint_pos"][T - 2, ids], atol=1e-5)
    np.testing.assert_allclose(obs[88:110], d["joint_pos"][T - 1, ids], atol=1e-5)


def _variant(tmp: Path, name: str, fps: float | None = None, **meta_updates) -> Path:
    from dropbear_wbc.tasks.tracking.motion_npz import load_motion_npz, save_motion_npz

    m = load_motion_npz(WAVE)
    m.meta = {**m.meta, **meta_updates}
    if fps is not None:
        m.fps = fps
    return save_motion_npz(tmp / f"{name}.npz", m)


def test_runtime_reference_fails_closed(export_dir, tmp_path):
    from dropbear_wbc.deploy.motion import RuntimeReferenceError

    if REJECTED.is_file():
        with pytest.raises(RuntimeReferenceError, match="REJECTED"):
            _build(export_dir, "--motion", str(REJECTED))
        cfg, _ = _build(export_dir, "--motion", str(REJECTED), "--allow-rejected-motion")
        assert cfg.meta["runtime_motion"]["checks"]["validation"] == "rejected"
    with pytest.raises(RuntimeReferenceError, match="ankle"):
        _build(export_dir, "--motion", str(_variant(tmp_path, "authored", authored_ankle_tierods=True)))
    with pytest.raises(RuntimeReferenceError, match="USD"):
        _build(export_dir, "--motion", str(_variant(tmp_path, "usd", usd_sha256="f" * 64)))
    with pytest.raises(RuntimeReferenceError, match="fps"):
        _build(export_dir, "--motion", str(_variant(tmp_path, "fps60", fps=60.0)))
    # an unvalidated clip (no verdict file next to it) is usable and reported as such
    plain = tmp_path / "plain_copy.npz"
    shutil.copyfile(WAVE, plain)
    cfg, _ = _build(export_dir, "--motion", str(plain))
    assert cfg.meta["runtime_motion"]["checks"]["validation"] == "unvalidated"
