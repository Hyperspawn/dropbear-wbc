"""Synthetic fixtures for the deploy tests: a BeyondMimic-layout ONNX, a contract NPZ, LowStates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk import motors
from dropbear_wbc.sdk.types import IMUState, LowState, MotorStateBlock, SimState

# Policy joint order deliberately differs from SDK order (Isaac-style interleaving).
POLICY_JOINTS = tuple(motors.MOTOR_NAMES[i] for i in (0, 6, 12, 17, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20,
                                                      4, 10, 16, 21, 5, 11))
REF_JOINTS = ("some_passive_joint",) + POLICY_JOINTS[::-1] + ("head_LeadScrew1",)  # all-joint Isaac order
BODIES = ("world", "anchor_body", "left_foot")
T_FRAMES = 40
OBS_TERMS = [  # BeyondMimic policy group, deployable variant (no privileged terms)
    {"name": "command", "func": "motion_command", "dim": 44},
    {"name": "motion_anchor_ori_b", "func": "motion_anchor_ori_b", "dim": 6},
    {"name": "base_ang_vel", "func": "base_ang_vel", "dim": 3},
    {"name": "joint_pos", "func": "joint_pos_rel", "dim": 22},
    {"name": "joint_vel", "func": "joint_vel_rel", "dim": 22},
    {"name": "actions", "func": "last_action", "dim": 22},
]
OBS_DIM = sum(t["dim"] for t in OBS_TERMS)
# Full BeyondMimic policy ObsGroup (tracking_env_cfg.PolicyCfg), including the sim-only terms.
OBS_TERMS_FULL = [
    {"name": "command", "func": "motion_command", "dim": 44},
    {"name": "motion_anchor_pos_b", "func": "motion_anchor_pos_b", "dim": 3},
    {"name": "motion_anchor_ori_b", "func": "motion_anchor_ori_b", "dim": 6},
    {"name": "base_lin_vel", "func": "base_lin_vel", "dim": 3},
    {"name": "base_ang_vel", "func": "base_ang_vel", "dim": 3},
    {"name": "joint_pos", "func": "joint_pos_rel", "dim": 22},
    {"name": "joint_vel", "func": "joint_vel_rel", "dim": 22},
    {"name": "actions", "func": "last_action", "dim": 22},
]


def reference_tables(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    jp = rng.normal(scale=0.2, size=(T_FRAMES, len(REF_JOINTS)))
    jv = rng.normal(scale=0.5, size=(T_FRAMES, len(REF_JOINTS)))
    bp = rng.normal(size=(T_FRAMES, len(BODIES), 3))
    yaw = np.linspace(0, 0.5, T_FRAMES)
    bq = np.zeros((T_FRAMES, len(BODIES), 4))
    bq[..., 0] = np.cos(yaw / 2)[:, None]
    bq[..., 3] = np.sin(yaw / 2)[:, None]
    return {"joint_pos": jp, "joint_vel": jv, "body_pos_w": bp, "body_quat_w": bq}


def action_matrix(seed: int = 1, obs_dim: int = OBS_DIM, scale: float = 0.01) -> np.ndarray:
    return np.random.default_rng(seed).normal(scale=scale, size=(obs_dim, 22)).astype(np.float32)


def write_beyondmimic_onnx(path: Path, obs_dim: int = OBS_DIM, weight_scale: float = 0.01,
                           tables: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """ONNX with inputs obs[1,N], time_step[1,1] -> actions = obs @ W, plus reference tables at time_step."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    tables = reference_tables() if tables is None else tables
    n_frames = tables["joint_pos"].shape[0]
    w = action_matrix(obs_dim=obs_dim, scale=weight_scale)
    inits = [numpy_helper.from_array(w, "W")]
    nodes = [helper.make_node("MatMul", ["obs", "W"], ["actions"]),
             helper.make_node("Cast", ["time_step"], ["ts_i"], to=TensorProto.INT64),
             helper.make_node("Clip", ["ts_i", "zero", "tmax"], ["ts_c"]),
             helper.make_node("Reshape", ["ts_c", "shape1"], ["idx"])]
    inits += [numpy_helper.from_array(np.array(0, np.int64), "zero"),
              numpy_helper.from_array(np.array(n_frames - 1, np.int64), "tmax"),
              numpy_helper.from_array(np.array([1], np.int64), "shape1")]
    outputs = [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 22])]
    for name, tab in tables.items():
        inits.append(numpy_helper.from_array(tab.astype(np.float32), f"tab_{name}"))
        nodes.append(helper.make_node("Gather", [f"tab_{name}", "idx"], [name], axis=0))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, *tab.shape[1:]]))
    graph = helper.make_graph(nodes, "beyondmimic_like",
                              [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim]),
                               helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1])],
                              outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    for k, v in {"joint_names": ",".join(REF_JOINTS), "body_names": ",".join(BODIES),
                 "anchor_body_name": "anchor_body"}.items():
        e = model.metadata_props.add()
        e.key, e.value = k, v
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return tables


def write_sidecar(path: Path, onnx_name: str, terms: list[dict] | None = None, num_frames: int = T_FRAMES,
                  body_names: tuple[str, ...] = BODIES, anchor: str = "anchor_body", align: str = "yaw") -> dict:
    terms = OBS_TERMS if terms is None else terms
    side = {
        "schema": "dropbear-policy-sidecar-v1", "policy_onnx": onnx_name,
        "onnx_inputs": {"obs": "obs", "time_step": "time_step"}, "step_dt": 0.02,
        "joint_names": list(POLICY_JOINTS),
        "default_joint_pos": [motors.DEFAULT_POS[motors.MOTOR_NAMES.index(j)] for j in POLICY_JOINTS],
        "joint_stiffness": [motors.DEFAULT_KP[motors.MOTOR_NAMES.index(j)] for j in POLICY_JOINTS],
        "joint_damping": [motors.DEFAULT_KD[motors.MOTOR_NAMES.index(j)] for j in POLICY_JOINTS],
        "action_scale": 0.25, "observations": terms, "obs_dim": sum(t["dim"] for t in terms),
        "motion": {"source": "onnx", "anchor_body_name": anchor, "num_frames": num_frames,
                   "reference_joint_names": list(REF_JOINTS), "body_names": list(body_names), "align": align},
    }
    path.write_text(json.dumps(side, indent=1), encoding="utf-8")
    return side


def write_contract_npz(path: Path) -> dict[str, np.ndarray]:
    tables = reference_tables(seed=5)
    np.savez(path, fps=np.array(50.0), joint_pos=tables["joint_pos"], joint_vel=tables["joint_vel"],
             body_pos_w=tables["body_pos_w"], body_quat_w=tables["body_quat_w"],
             body_lin_vel_w=np.zeros_like(tables["body_pos_w"]), body_ang_vel_w=np.zeros_like(tables["body_pos_w"]),
             joint_names=np.array(REF_JOINTS), body_names=np.array(BODIES), motor_names=np.array(motors.MOTOR_NAMES),
             closure_residual_m=np.zeros(T_FRAMES), meta=np.array(json.dumps({"source": "synthetic"})))
    return tables


def low_state(tick: int, q: np.ndarray | None = None, quat_wxyz=(1.0, 0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0),
              sim: bool = True, imu_valid: int = 1, imu_age_us: int = 0) -> LowState:
    st = LowState(tick=tick, stamp_ns=0)
    st.imu = IMUState(quat_wxyz=np.asarray(quat_wxyz, np.float32), gyro=np.asarray(gyro, np.float32),
                      accel=np.array([0, 0, 9.81], np.float32), rpy=np.zeros(3, np.float32),
                      valid=imu_valid, age_us=imu_age_us)
    st.motor = MotorStateBlock.zeros(22)
    if q is not None:
        st.motor.q[:] = np.asarray(q, np.float32)
    if sim:
        st.sim = SimState(time_s=tick * 0.002, root_pos_w=np.array([0.1, -0.2, 0.9], np.float32),
                          root_quat_w=np.asarray(quat_wxyz, np.float32))
    return st


def unexplained_state_faults(steps_path, pub_t, max_age_s=0.030):
    """State-gate fault steps that the fake robot's own publish timing does not explain (items 288-289).

    Under host load a fake robot thread can miss its 2 ms period by more than ``state_max_age_s`` (30 ms); the runner
    then re-uses a tick for > 30 ms and correctly records ``state_stale`` (absorbed by the hold-last grace). That is a
    real stale input, not a gate bug. A fault step is *explained* when the robot's next publish after the faulted tick
    came more than ``max_age_s / 2`` later (the rest is ZMQ delivery / runner read latency); a fault while the robot
    kept publishing at its 2 ms period is the false trip the wall-clock runner tests guard against.

    A fault is also explained when the step's logged ``state_age_ms`` (the runner's same-host transport age, logged from
    the raw stamp independently of the gated age and only when it is a valid [0, 1 s) age) is itself >= ``max_age_s``:
    the frame really was that old at receive time (ZMQ delivery or the runner thread starved under load), item 289.

    ``steps_path`` is the runner --log JSONL, ``pub_t`` maps tick -> perf_counter() at publish."""
    import json
    from pathlib import Path

    ticks = sorted(pub_t)
    nxt = dict(zip(ticks, ticks[1:]))
    bad = []
    for line in Path(steps_path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if not rec.get("state_fault"):
            continue
        k = rec["tick"]
        gap = (pub_t[nxt[k]] - pub_t[k]) if k in nxt else float("inf")  # no later publish: the robot stopped
        age_ms = rec.get("state_age_ms")
        if gap <= 0.5 * max_age_s and not (age_ms is not None and age_ms >= 1e3 * max_age_s):
            bad.append({"t": rec["t"], "tick": k, "fault": rec["state_fault"], "robot_gap_ms": 1e3 * gap,
                        "state_age_ms": age_ms})
    return bad
