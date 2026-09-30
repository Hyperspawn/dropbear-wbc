"""Offline tests of the deploy runner pieces: config parsing, observations, motion, FSM, ONNX (CPU only)."""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

import deploy_fixtures as fx
import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy import quat as Q
from dropbear_wbc.deploy.config import MotionCfg, ObsTermCfg, hold_config, load_deploy_yaml, load_sidecar
from dropbear_wbc.deploy.fsm import DeployController, FsmState
from dropbear_wbc.deploy.motion import NpzMotion
from dropbear_wbc.deploy.observations import (ObsContext, ObservationBuilder, PrivilegedObservationError,
                                              motion_anchor_ori_b)
from dropbear_wbc.sdk import motors


# ----------------------------------------------------------------------------- FSM (hold mode)

def test_fsm_passive_move_hold_sequence():
    cfg = hold_config()
    ctrl = DeployController(cfg)
    q0 = np.full(22, 0.05)
    st = fx.low_state(0, q0)
    cmd, info = ctrl.step(st, 0.0)
    assert info.state == FsmState.PASSIVE
    np.testing.assert_allclose(cmd.motor.kp, 0.0)
    np.testing.assert_allclose(cmd.motor.kd, np.asarray(motors.DEFAULT_KD, np.float32))
    np.testing.assert_allclose(cmd.motor.q, q0.astype(np.float32))
    ctrl.request(FsmState.MOVE_TO_DEFAULT, st, 0.0)
    cmd, info = ctrl.step(st, 1.0)  # halfway through the 2 s interpolation
    target = np.asarray(motors.DEFAULT_POS)
    np.testing.assert_allclose(cmd.motor.q, (q0 + 0.5 * (target - q0)).astype(np.float32), atol=1e-6)
    np.testing.assert_allclose(cmd.motor.kp, np.asarray(motors.DEFAULT_KP, np.float32))
    cmd, info = ctrl.step(st, 2.0)
    assert info.transition == "move_to_default->hold" and ctrl.state == FsmState.HOLD
    cmd, info = ctrl.step(st, 3.0)
    np.testing.assert_allclose(cmd.motor.q, target.astype(np.float32))
    assert cmd.tick == st.tick
    with pytest.raises(RuntimeError):
        ctrl.request(FsmState.POLICY, st, 3.0)


# ----------------------------------------------------------------------------- observations

def test_projected_gravity_and_joint_order():
    cfg = load_sidecar_cfg_without_motion()
    q = np.arange(22) * 0.01
    angle = 0.3  # pitch about +y
    quat = np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0])
    st = fx.low_state(0, q, quat_wxyz=quat, gyro=(0.1, 0.2, 0.3))
    ctx = ObsContext(state=st, cfg=cfg, last_action=np.zeros(22))
    b = ObservationBuilder([ObsTermCfg("g", "projected_gravity"), ObsTermCfg("w", "base_ang_vel", scale=2.0),
                            ObsTermCfg("jp", "joint_pos_rel", clip=(-0.1, 0.1))])
    obs = b.compute(ctx)
    np.testing.assert_allclose(obs[:3], [np.sin(angle), 0.0, -np.cos(angle)], atol=1e-6)
    np.testing.assert_allclose(obs[3:6], [0.2, 0.4, 0.6], atol=1e-6)
    expect = np.clip(q[cfg.motor_ids] - cfg.default_joint_pos, -0.1, 0.1)
    np.testing.assert_allclose(obs[6:], expect, atol=1e-6)
    assert cfg.joint_names[1] == motors.MOTOR_NAMES[6]  # policy order differs from SDK order


def test_history_stacking_oldest_first_and_reset_fill():
    cfg = hold_config()
    b = ObservationBuilder([ObsTermCfg("w", "base_ang_vel", history_length=3)])
    mk = lambda g: ObsContext(state=fx.low_state(0, gyro=(g, 0, 0)), cfg=cfg, last_action=np.zeros(22))  # noqa
    np.testing.assert_allclose(b.compute(mk(1.0))[::3], [1, 1, 1])
    np.testing.assert_allclose(b.compute(mk(2.0))[::3], [1, 1, 2])
    np.testing.assert_allclose(b.compute(mk(3.0))[::3], [1, 2, 3])
    b.reset()
    np.testing.assert_allclose(b.compute(mk(4.0))[::3], [4, 4, 4])


def test_privileged_terms_refused_by_default():
    with pytest.raises(PrivilegedObservationError):
        ObservationBuilder([ObsTermCfg("v", "base_lin_vel")])
    b = ObservationBuilder([ObsTermCfg("v", "base_lin_vel")], allow_privileged=True)
    st = fx.low_state(0, sim=False)
    with pytest.raises(PrivilegedObservationError):
        b.compute(ObsContext(state=st, cfg=hold_config(), last_action=np.zeros(22), allow_privileged=True))


def test_motion_anchor_ori_b_matches_isaaclab_layout():
    cfg = hold_config()
    st = fx.low_state(0)
    from dropbear_wbc.deploy.observations import ReferenceFrame
    ref = ReferenceFrame(np.zeros(22), np.zeros(22), np.zeros(3), Q.from_yaw(np.pi / 2))
    ctx = ObsContext(state=st, cfg=cfg, last_action=np.zeros(22), reference=ref)
    v = motion_anchor_ori_b(ctx, {})
    # rel = q_robot^-1 q_ref = yaw 90 deg: R = [[0,-1,0],[1,0,0],[0,0,1]] -> first two columns row-major.
    np.testing.assert_allclose(v, [0, -1, 1, 0, 0, 0], atol=1e-12)


# ----------------------------------------------------------------------------- config formats

def load_sidecar_cfg_without_motion():
    import json
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    side = fx.write_sidecar(d / "policy.json", "policy.onnx")
    side.pop("motion")
    (d / "policy.json").write_text(json.dumps(side), encoding="utf-8")
    return load_sidecar(d / "policy.json")


def test_unitree_deploy_yaml_parse(tmp_path):
    ids = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 12, 17, 13, 18, 14, 19, 15, 20, 16, 21]
    (tmp_path / "deploy.yaml").write_text(textwrap.dedent(f"""
        joint_ids_map: {ids}
        step_dt: 0.02
        stiffness: {[10.0 + i for i in range(22)]}
        damping: {[0.5 + 0.25 * i for i in range(22)]}
        default_joint_pos: {[0.01 * i for i in range(22)]}
        actions:
          JointPositionAction:
            scale: {[0.25] * 22}
            offset: {[0.1] * 22}
            clip: null
        observations:
          base_ang_vel: {{params: {{}}, clip: null, scale: [0.2, 0.2, 0.2], history_length: 1}}
          projected_gravity: {{params: {{}}, clip: null, scale: [1.0, 1.0, 1.0], history_length: 1}}
          velocity_commands: {{params: {{command_name: base_velocity}}, clip: null, scale: [1, 1, 1]}}
          joint_pos_rel: {{scale: {[1.0] * 22}}}
          joint_vel_rel: {{scale: {[0.05] * 22}}}
          last_action: {{}}
        policy: exported/policy.onnx
        fsm: {{move_to_default_s: 3.0}}
        """), encoding="utf-8")
    cfg = load_deploy_yaml(tmp_path / "deploy.yaml")
    assert cfg.joint_names[1] == motors.MOTOR_NAMES[6]
    # unitree_rl_lab: stiffness/damping are SDK slot order; default_joint_pos is policy order
    np.testing.assert_allclose(cfg.kp, [10.0 + s for s in ids])
    np.testing.assert_allclose(cfg.kd, [0.5 + 0.25 * s for s in ids])
    np.testing.assert_allclose(cfg.default_joint_pos, [0.01 * i for i in range(22)])
    np.testing.assert_allclose(cfg.default_pose_sdk[ids], [0.01 * i for i in range(22)])
    ctrl = DeployController(cfg)
    np.testing.assert_allclose(ctrl.full_kp, [10.0 + s for s in range(22)])  # LowCmd slot s gets stiffness[s]
    assert [t.func for t in cfg.observations][:3] == ["base_ang_vel", "projected_gravity", "velocity_commands"]
    assert cfg.fsm.move_to_default_s == 3.0
    assert cfg.policy_path.name == "policy.onnx"
    np.testing.assert_allclose(cfg.action_offset, [0.1] * 22)


# ----------------------------------------------------------------------------- NPZ reference

def test_npz_motion_columns_offset_alignment(tmp_path):
    tables = fx.write_contract_npz(tmp_path / "clip.npz")
    m = NpzMotion(MotionCfg(file=tmp_path / "clip.npz", anchor_body="anchor_body", align="yaw_xy"),
                  fx.POLICY_JOINTS, 0.02)
    cols = [fx.REF_JOINTS.index(j) for j in fx.POLICY_JOINTS]
    np.testing.assert_allclose(m.raw(3).joint_pos, tables["joint_pos"][3, cols])
    # Offset is anchor pose in the root frame at frame 0.
    rq, rp = tables["body_quat_w"][0, 0], tables["body_pos_w"][0, 0]
    ap = tables["body_pos_w"][0, 1]
    np.testing.assert_allclose(Q.rotate(rq, m.anchor_offset_pos) + rp, ap, atol=1e-9)
    # Align: robot anchor heading 1.0 rad at (5, 6); reference step 0 must map onto it.
    robot_q = Q.from_yaw(1.0)
    m.align(robot_q, np.array([5.0, 6.0, 0.9]))
    s0 = m.sample(0)
    assert abs(Q.yaw(s0.anchor_quat_w) - 1.0) < 1e-9
    np.testing.assert_allclose(s0.anchor_pos_w[:2], [5.0, 6.0], atol=1e-9)
    assert m.frame_index(1000) == fx.T_FRAMES - 1 and m.done(fx.T_FRAMES)


# ----------------------------------------------------------------------------- ONNX sidecar end-to-end

def test_beyondmimic_sidecar_policy_end_to_end(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from dropbear_wbc.deploy.motion import OnnxMotion
    from dropbear_wbc.deploy.policy import OnnxPolicy

    tables = fx.write_beyondmimic_onnx(tmp_path / "policy.onnx")
    fx.write_sidecar(tmp_path / "policy.json", "policy.onnx")
    policy = OnnxPolicy(tmp_path / "policy.onnx")
    cfg = load_sidecar(tmp_path / "policy.json", policy.metadata)
    assert policy.has_reference and policy.obs_dim == fx.OBS_DIM and policy.time_input == "time_step"
    assert cfg.motion.source == "onnx" and cfg.motion.anchor_body == "anchor_body"
    motion = OnnxMotion(cfg.motion, cfg.joint_names, cfg.step_dt, policy.reference, list(fx.REF_JOINTS),
                        list(fx.BODIES), fx.T_FRAMES)
    ctrl = DeployController(cfg, policy, motion)
    st = fx.low_state(0, np.asarray(motors.DEFAULT_POS))
    ctrl.request(FsmState.MOVE_TO_DEFAULT, st, 0.0)
    ctrl.step(st, 2.0)
    assert ctrl.state == FsmState.HOLD
    ctrl.request(FsmState.POLICY, st, 2.0)
    prev_action = np.zeros(22)
    for k in range(3):
        cmd, info = ctrl.step(fx.low_state(10 * k, np.asarray(motors.DEFAULT_POS)), 2.0 + 0.02 * k)
        assert info.state == FsmState.POLICY and info.motion_frame == k
        # last_action term (last 22 obs values) carries the previous step's raw action (zeros at start).
        np.testing.assert_allclose(info.obs[-22:], prev_action, atol=1e-6)
        prev_action = info.action.copy()
        # Obs layout: command = reference motor pos/vel (policy order) at frame k.
        cols = [fx.REF_JOINTS.index(j) for j in cfg.joint_names]
        np.testing.assert_allclose(info.obs[:22], tables["joint_pos"][k, cols], atol=1e-6)
        np.testing.assert_allclose(info.obs[22:44], tables["joint_vel"][k, cols], atol=1e-6)
        # Action -> command mapping: q* = default + 0.25 * action at the policy joints' SDK slots.
        expect_action = info.obs.astype(np.float32) @ fx.action_matrix()
        np.testing.assert_allclose(info.action, expect_action, atol=1e-5)
        np.testing.assert_allclose(cmd.motor.q[cfg.motor_ids], (cfg.default_joint_pos + 0.25 * info.action)
                                   .astype(np.float32), atol=1e-6)
        np.testing.assert_allclose(cmd.motor.kp[cfg.motor_ids], cfg.kp.astype(np.float32))
    # Bad orientation (upside down) drops to Passive.
    cmd, info = ctrl.step(fx.low_state(40, np.asarray(motors.DEFAULT_POS), quat_wxyz=(0, 1, 0, 0)), 2.1)
    assert info.transition == "policy->passive(bad_orientation)" and ctrl.state == FsmState.PASSIVE


def test_sidecar_target_clip_clamps_policy_targets(tmp_path):
    """``target_clip`` (training-time motor target clamp, docs/ISSUES.md #25) is parsed and applied after scale + offset;
    ``target_interp_steps`` is carried for the motor-side loop."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import json

    from dropbear_wbc.deploy.motion import OnnxMotion
    from dropbear_wbc.deploy.policy import OnnxPolicy

    fx.write_beyondmimic_onnx(tmp_path / "policy.onnx", weight_scale=50.0)  # large actions: targets far out
    side = fx.write_sidecar(tmp_path / "policy.json", "policy.onnx")
    lo = np.asarray(side["default_joint_pos"]) - 0.01
    side["target_clip"] = [[float(a), float(a) + 0.02] for a in lo]
    side["target_interp_steps"] = 4
    (tmp_path / "policy.json").write_text(json.dumps(side), encoding="utf-8")
    policy = OnnxPolicy(tmp_path / "policy.onnx")
    cfg = load_sidecar(tmp_path / "policy.json", policy.metadata)
    assert cfg.target_clip.shape == (22, 2) and cfg.target_interp_steps == 4
    motion = OnnxMotion(cfg.motion, cfg.joint_names, cfg.step_dt, policy.reference, list(fx.REF_JOINTS),
                        list(fx.BODIES), fx.T_FRAMES)
    ctrl = DeployController(cfg, policy, motion)
    st = fx.low_state(0, np.asarray(motors.DEFAULT_POS))
    ctrl.request(FsmState.MOVE_TO_DEFAULT, st, 0.0)
    ctrl.step(st, 2.0)
    ctrl.request(FsmState.POLICY, st, 2.0)
    cmd, info = ctrl.step(fx.low_state(10, np.asarray(motors.DEFAULT_POS)), 2.02)
    unclipped = cfg.action_offset + cfg.action_scale * info.action
    assert np.abs(unclipped - cfg.default_joint_pos).max() > 0.05  # the clamp is exercised
    q = cmd.motor.q[cfg.motor_ids].astype(float)
    assert np.all(q >= cfg.target_clip[:, 0] - 1e-6) and np.all(q <= cfg.target_clip[:, 1] + 1e-6)
    bad = dict(side, target_clip=[[0.1, -0.1]] * 22)
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="target_clip"):
        load_sidecar(tmp_path / "bad.json", policy.metadata)


def test_live_motion_splices_dropped_clips_at_the_playhead(tmp_path):
    """deploy.motion.LiveMotion (docs/TEXT_TO_MOTION.md): the first clip plays; a clip dropped into the watched folder
    is spliced at the playhead (+ lead) and followed by the idle clip; a clip with another layout is rejected and the
    stream keeps running; never done."""
    import json

    from dropbear_wbc.deploy.config import MotionCfg
    from dropbear_wbc.deploy.motion import LiveMotion

    first = tmp_path / "first.npz"
    fx.write_contract_npz(first)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    cfg = MotionCfg(file=first, source="npz", anchor_body="anchor_body")
    live = LiveMotion(cfg, tuple(fx.POLICY_JOINTS), 0.02, watch_dir=inbox)
    assert not live.done(10_000) and live.poll(0) == []
    d = dict(np.load(first, allow_pickle=False))
    d["joint_pos"] = d["joint_pos"] + 0.5  # a visibly different clip
    np.savez(inbox / "001_new.npz", **d)
    ev = live.poll(20)
    assert len(ev) == 1 and ev[0]["frame"] == live.frame_index(20) + live.lead
    col = list(fx.REF_JOINTS).index(fx.POLICY_JOINTS[0])
    k = ev[0]["frame"] + 30  # past the 0.4 s (20-frame) cross-fade, inside the 40-frame clip
    np.testing.assert_allclose(live.raw(k).joint_pos[0], d["joint_pos"][k - ev[0]["frame"], col], atol=1e-9)
    bad = dict(d)
    bad["joint_names"] = np.array(list(d["joint_names"])[::-1])
    np.savez(inbox / "002_bad.npz", **bad)
    ev2 = live.poll(30)
    assert "rejected" in ev2[0]
    assert live.poll(31) == []  # each file is handled once
    assert json.dumps(live.events)


def test_onnx_motion_non_root_anchor_needs_an_offset():
    """A reference without the root body (e.g. the tracking export's 14 bodies) needs motion.anchor_offset_*."""
    from dropbear_wbc.deploy.motion import OnnxMotion

    def query(frame):
        return {"joint_pos": np.zeros(22), "joint_vel": np.zeros(22), "body_pos_w": np.array([[0.1, 0.0, 1.6], [0, 0, 1]]),
                "body_quat_w": np.array([[0.5, 0.5, 0.5, 0.5], [1.0, 0, 0, 0]])}

    bodies, joints = ["anchor", "head"], list(motors.MOTOR_NAMES)
    with pytest.raises(ValueError, match="anchor_offset"):
        OnnxMotion(MotionCfg(source="onnx", anchor_body="anchor"), motors.MOTOR_NAMES, 0.02, query, joints, bodies, 10)
    m = OnnxMotion(MotionCfg(source="onnx", anchor_body="anchor", anchor_offset_pos=(0.1, 0.0, 1.6),
                             anchor_offset_quat=(0.5, 0.5, 0.5, 0.5)), motors.MOTOR_NAMES, 0.02, query, joints, bodies, 10)
    np.testing.assert_allclose(m.anchor_offset_quat, [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(m.anchor_offset_pos, [0.1, 0.0, 1.6])
