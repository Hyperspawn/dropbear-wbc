"""The policy runner consumes the tracking export exactly as ``scripts/play.py --export`` writes it.

The export is made by the REAL ``tasks/tracking/export.py`` (via ``scripts/emulate_tracking_export.py``: Isaac
stubs, mocked env handles and a real contract NPZ, with a random untrained rsl-rl ActorCritic and normalizer).
The runner is then built from that ``policy.json`` with ``tools/policy_runner.build``, and the controller is
stepped on LowStates that put the robot exactly on the NPZ reference. Every observation slice is checked
against the NPZ ground truth, the action against the plain ``policy.onnx``, and the LowCmd against the target
law. Needs torch + onnx + onnxruntime (system Python). Skips in ``.venv-newton``, which has no torch.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ and third_party/pydeps)

ROOT = sdk_test_paths.ROOT
for p in (ROOT / "tools", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

NPZ = ROOT / "data" / "motions" / "smoke" / "dropbear_static_stand.npz"


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    if not NPZ.is_file():
        pytest.skip(f"contract NPZ {NPZ} not present")
    import emulate_tracking_export as emu

    out = tmp_path_factory.mktemp("export_emulated")
    emu.emulate_export(out, NPZ, seed=3)
    return out


def _npz():
    d = np.load(NPZ, allow_pickle=False)
    return d, [str(j) for j in d["joint_names"]], [str(b) for b in d["body_names"]]


def _state_on_reference(k: int, tick: int):
    """LowState with the robot exactly at NPZ frame ``k`` (root = articulation root 'world')."""
    from dropbear_wbc.deploy import quat as Q
    from dropbear_wbc.sdk import motors
    from dropbear_wbc.sdk.types import IMUState, LowState, MotorStateBlock, SimState

    d, joints, bodies = _npz()
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


def test_export_sidecar_parses_into_deploy_config(export_dir):
    from dropbear_wbc.deploy.config import load_sidecar
    from dropbear_wbc.sdk import motors

    side = json.loads((export_dir / "policy.json").read_text(encoding="utf-8"))
    cfg = load_sidecar(export_dir / "policy.json")
    assert cfg.joint_names == motors.MOTOR_NAMES
    assert cfg.policy_path.name == "policy_motion.onnx" and cfg.policy_path.is_file()
    assert (cfg.onnx_obs_input, cfg.onnx_time_input) == ("obs", "time_step")
    assert cfg.step_dt == pytest.approx(0.02)
    assert [t.func for t in cfg.observations] == ["motion_command", "motion_anchor_pos_b", "motion_anchor_ori_b",
                                                  "base_lin_vel", "base_ang_vel", "joint_pos_rel", "joint_vel_rel",
                                                  "last_action"]
    assert sum(t.dim for t in cfg.observations) == cfg.obs_dim == 125
    np.testing.assert_allclose(cfg.action_scale, side["action_scale"])
    np.testing.assert_allclose(cfg.action_offset, side["default_joint_pos"])
    np.testing.assert_allclose(cfg.kp, side["joint_stiffness"])
    np.testing.assert_allclose(cfg.default_pose_sdk, side["default_joint_pos"])  # all 22 motors in the policy
    assert cfg.motion.source == "onnx" and cfg.motion.anchor_body == side["motion"]["anchor_body_name"]
    assert cfg.motion.anchor_offset_quat == pytest.approx((0.5, 0.5, 0.5, 0.5), abs=1e-4)  # CONTRACTS 5.1
    assert cfg.meta["usd_sha256"].startswith("45586414")


def test_runner_fails_closed_without_allow_privileged(export_dir):
    import policy_runner
    from dropbear_wbc.deploy.observations import PrivilegedObservationError

    args = policy_runner.parse_args(["--mode", "policy", "--sidecar", str(export_dir / "policy.json")])
    with pytest.raises(PrivilegedObservationError):
        policy_runner.build(args)


def test_runner_policy_steps_match_export_semantics(export_dir):
    import onnxruntime as ort

    import policy_runner
    from dropbear_wbc.deploy.fsm import FsmState
    from dropbear_wbc.deploy.motion import OnnxMotion
    from dropbear_wbc.deploy import quat as Q
    from dropbear_wbc.sdk import motors

    args = policy_runner.parse_args(["--mode", "policy", "--sidecar", str(export_dir / "policy.json"),
                                     "--allow-privileged"])
    cfg, ctrl = policy_runner.build(args)
    assert isinstance(ctrl.motion, OnnxMotion) and ctrl.motion.num_frames == 200
    assert ctrl.obs_builder.privileged_terms == ["motion_anchor_pos_b", "base_lin_vel"]
    plain = ort.InferenceSession(str(export_dir / "policy.onnx"), providers=["CPUExecutionProvider"])

    d, joints, bodies = _npz()
    ids = [joints.index(m) for m in motors.MOTOR_NAMES]
    a = bodies.index(cfg.motion.anchor_body)
    r = bodies.index("world")
    # env -> sidecar -> runner: the target law uses the env's default pose and action scale (EMULATED.json
    # records what the mocked env handed to export.py).
    emu = json.loads((export_dir / "EMULATED.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(cfg.default_joint_pos, np.asarray(emu["default_full"])[emu["motor_ids"]], atol=1e-6)
    np.testing.assert_allclose(cfg.action_offset, cfg.default_joint_pos)
    np.testing.assert_allclose(cfg.action_scale, emu["action_scale"], rtol=1e-6)
    # The embedded ONNX reference equals the NPZ motor columns and the NPZ anchor pose.
    for k in (0, 57, 199):
        raw = ctrl.motion.raw(k)
        np.testing.assert_allclose(raw.joint_pos, d["joint_pos"][k, ids], atol=1e-6)
        np.testing.assert_allclose(raw.anchor_pos_w, d["body_pos_w"][k, a], atol=1e-6)

    # The runner rebuilds the robot anchor from root pose (IMU quat + sim root position) and the sidecar's rigid
    # anchor offset. That must equal the NPZ anchor link pose. (yaw_xy alignment would hide an xy error here.)
    for k in (0, 120):
        ctx = ctrl._ctx(_state_on_reference(k, tick=k))
        np.testing.assert_allclose(ctx.robot_anchor_pos_w(), d["body_pos_w"][k, a], atol=2e-5)
        assert abs(float(np.dot(ctx.robot_anchor_quat_w(), d["body_quat_w"][k, a]))) == pytest.approx(1.0, abs=1e-6)

    st0 = _state_on_reference(0, tick=1000)
    ctrl.request(FsmState.POLICY, st0, 0.0)
    prev_action = np.zeros(22)
    for k in range(4):
        st = _state_on_reference(k, tick=1000 + 10 * k)
        cmd, info = ctrl.step(st, 0.02 * k)
        assert info.state == FsmState.POLICY and info.motion_frame == k
        obs = info.obs
        assert obs.shape == (125,) and obs.dtype == np.float32
        # command: reference motor joint_pos (22) then joint_vel (22), contract order
        np.testing.assert_allclose(obs[0:22], d["joint_pos"][k, ids], atol=1e-5)
        np.testing.assert_allclose(obs[22:44], d["joint_vel"][k, ids], atol=1e-5)
        # robot on the reference: anchor offset ~0, relative rotation ~identity (first two matrix columns)
        assert np.abs(obs[44:47]).max() < 1e-4
        np.testing.assert_allclose(obs[47:53], [1, 0, 0, 1, 0, 0], atol=1e-5)
        # base_lin_vel = root CoM velocity in the root link frame (Isaac Lab root_lin_vel_b); base_ang_vel = gyro
        qr = np.asarray(d["body_quat_w"][k, r], float)
        np.testing.assert_allclose(obs[53:56], Q.rotate_inv(qr, d["body_lin_vel_w"][k, r]), atol=1e-5)
        np.testing.assert_allclose(obs[56:59], Q.rotate_inv(qr, d["body_ang_vel_w"][k, r]), atol=1e-5)
        np.testing.assert_allclose(obs[59:81], d["joint_pos"][k, ids] - cfg.default_joint_pos, atol=1e-5)
        np.testing.assert_allclose(obs[81:103], d["joint_vel"][k, ids], atol=1e-5)
        np.testing.assert_allclose(obs[103:125], prev_action, atol=1e-6)
        # action == the plain exported policy (normalizer baked in) on the same observation
        (ref,) = plain.run(["actions"], {"obs": obs[None]})
        np.testing.assert_allclose(info.action, ref[0], atol=1e-5)
        # LowCmd: q* = default + scale * action on all 22 motors, gains from the sidecar
        np.testing.assert_allclose(cmd.motor.q, cfg.action_offset + cfg.action_scale * info.action, atol=1e-5)
        np.testing.assert_allclose(cmd.motor.kp, cfg.kp, rtol=1e-6)
        np.testing.assert_allclose(cmd.motor.kd, cfg.kd, rtol=1e-6)
        assert cmd.tick == st.tick
        prev_action = info.action


def test_export_without_anchor_offset_fails_closed(export_dir, tmp_path):
    import shutil

    import policy_runner

    for f in export_dir.iterdir():
        shutil.copy(f, tmp_path / f.name)
    side = json.loads((tmp_path / "policy.json").read_text(encoding="utf-8"))
    del side["motion"]["anchor_offset_pos"], side["motion"]["anchor_offset_quat"]
    (tmp_path / "policy.json").write_text(json.dumps(side), encoding="utf-8")
    args = policy_runner.parse_args(["--mode", "policy", "--sidecar", str(tmp_path / "policy.json"),
                                     "--allow-privileged"])
    with pytest.raises(ValueError, match="anchor_offset"):
        policy_runner.build(args)
