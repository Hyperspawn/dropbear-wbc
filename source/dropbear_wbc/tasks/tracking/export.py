"""Export a trained Dropbear tracking policy for deployment / sim2sim.

Writes into ``out_dir``:

* ``policy.pt``    TorchScript, ``actions = actor(normalizer(obs))`` (Isaac Lab ``export_policy_as_jit``);
* ``policy.onnx``  same graph as ONNX (input ``obs`` (1, obs_dim), output ``actions`` (1, 22));
* ``policy_motion.onnx``  BeyondMimic-style: inputs ``obs``, ``time_step`` -> ``actions`` plus the embedded
  reference (motor ``joint_pos``/``joint_vel`` (22), tracked-body link poses/velocities, world frame);
* ``policy.json``  sidecar in the deploy schema ``dropbear-policy-sidecar-v1`` (CONTRACTS 6.1: joint order,
  gains, default pose, action scale, ordered observation terms, motion/anchor info) plus a
  ``dropbear_tracking`` block (observation layout, normalizer statistics, timing, provenance hashes, parity);
* ``parity_samples.pt``  64 random observations and the live policy's actions (for offline parity checks).

Joint target law (per motor, contract order): ``q_target = default_pos + action_scale * action``,
tracked by the implicit PD ``tau = kp (q_target - q) - kd dq`` (clipped to effort_limit).
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

import torch

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

from dropbear_wbc.isaac.launch import sha256_file
from dropbear_wbc.robots import dropbear_names as N

SIDECAR_SCHEMA = "dropbear-tracking-policy-v1"
DEPLOY_SIDECAR_SCHEMA = "dropbear-policy-sidecar-v1"
OBS_FUNC: dict[str, str] = {
    "command": "motion_command",
    "motion_anchor_pos_b": "motion_anchor_pos_b",
    "motion_anchor_ori_b": "motion_anchor_ori_b",
    "base_lin_vel": "base_lin_vel",
    "base_ang_vel": "base_ang_vel",
    "joint_pos": "joint_pos_rel",
    "joint_vel": "joint_vel_rel",
    "actions": "last_action",
}
"""Policy ObsGroup term name -> deploy observation function (``dropbear_wbc.deploy.observations``)."""


class _MotionPolicyOnnx(torch.nn.Module):
    """(obs, time_step) -> (actions, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w)."""

    def __init__(self, policy, normalizer, command):
        super().__init__()
        self.actor = copy.deepcopy(policy.actor).cpu()
        self.normalizer = copy.deepcopy(normalizer).cpu() if normalizer is not None else torch.nn.Identity()
        m = command.motion
        self.register_buffer("joint_pos", m.joint_pos.detach().cpu())
        self.register_buffer("joint_vel", m.joint_vel.detach().cpu())
        self.register_buffer("body_pos_w", m.body_pos_w.detach().cpu())
        self.register_buffer("body_quat_w", m.body_quat_w.detach().cpu())
        self.register_buffer("body_lin_vel_w", m.body_lin_vel_w.detach().cpu())
        self.register_buffer("body_ang_vel_w", m.body_ang_vel_w.detach().cpu())
        self.t_total = int(m.time_step_total)

    def forward(self, x, time_step):
        t = torch.clamp(time_step.long().squeeze(-1), max=self.t_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[t],
            self.joint_vel[t],
            self.body_pos_w[t],
            self.body_quat_w[t],
            self.body_lin_vel_w[t],
            self.body_ang_vel_w[t],
        )


def _attach_onnx_metadata(path: Path, metadata: dict[str, Any]) -> None:
    import onnx

    model = onnx.load(str(path))
    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = ",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
        model.metadata_props.append(entry)
    onnx.save(model, str(path))


def _obs_layout(env, group: str) -> list[dict[str, Any]]:
    names = env.observation_manager.active_terms[group]
    dims = env.observation_manager.group_obs_term_dim[group]
    layout, start = [], 0
    for name, dim in zip(names, dims):
        size = 1
        for d in dim:
            size *= int(d)
        layout.append({"name": name, "start": start, "dim": size})
        start += size
    return layout


def export_tracking_policy(
    env,
    runner,
    out_dir: str | Path,
    *,
    task: str,
    checkpoint: str | Path,
    motion_file: str | Path,
) -> dict[str, Any]:
    """Export ``runner``'s policy (with its observation normalizer) for the unwrapped tracking ``env``.

    Returns the sidecar dict (also written to ``policy.json``). Includes a TorchScript-vs-live parity check
    on random observations (``parity.torchscript_max_abs_diff``).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy = runner.alg.policy
    normalizer = runner.obs_normalizer
    policy.eval()
    if hasattr(normalizer, "eval"):
        normalizer.eval()  # EmpiricalNormalization updates its statistics in train mode
    export_policy_as_jit(policy, normalizer, path=str(out), filename="policy.pt")
    export_policy_as_onnx(policy, normalizer=normalizer, path=str(out), filename="policy.onnx")

    command = env.command_manager.get_term("motion")
    motion_onnx = _MotionPolicyOnnx(policy, normalizer, command).eval()
    obs_dim = int(policy.actor[0].in_features)
    torch.onnx.export(
        motion_onnx,
        (torch.zeros(1, obs_dim), torch.zeros(1, 1)),
        str(out / "policy_motion.onnx"),
        export_params=True,
        opset_version=11,
        input_names=["obs", "time_step"],
        output_names=["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"],
        dynamic_axes={},
    )

    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    motor_names = list(term._joint_names)
    if tuple(motor_names) != N.MOTOR_NAMES:
        raise RuntimeError(f"action joints {motor_names} != motor contract")
    motor_ids = robot.find_joints(motor_names, preserve_order=True)[0]
    nominal = getattr(robot.data, "default_joint_pos_nominal", robot.data.default_joint_pos[0])
    scale = term._scale
    scale_list = scale[0].tolist() if isinstance(scale, torch.Tensor) else [float(scale)] * len(motor_names)
    kp = robot.data.joint_stiffness[0, motor_ids].tolist()
    kd = robot.data.joint_damping[0, motor_ids].tolist()
    effort = robot.data.joint_effort_limits[0, motor_ids].tolist()
    # explicit (hw_*) motor models run the PD themselves: read their own gains and peak torque
    from dropbear_wbc.robots.hw_introspect import overlay_explicit_gains, target_interp_steps

    overlay_explicit_gains(robot, motor_names, kp, kd, effort)
    default = [float(nominal[i]) for i in motor_ids]

    norm_state = {}
    if hasattr(normalizer, "_mean") and hasattr(normalizer, "_std"):
        norm_state = {"mean": normalizer._mean.squeeze(0).tolist(), "std": normalizer._std.squeeze(0).tolist(), "eps": float(normalizer.eps)}

    cfg = env.cfg
    layout = _obs_layout(env, "policy")
    # anchor pose in the root ('world') link frame -- rigid (fixed joint); taken from the robot's current state
    from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

    a_idx = robot.body_names.index(command.cfg.anchor_body_name)
    root_q = robot.data.body_link_quat_w[0:1, 0]
    rel_p = quat_apply_inverse(root_q, robot.data.body_link_pos_w[0:1, a_idx] - robot.data.body_link_pos_w[0:1, 0])[0]
    rel_q = quat_mul(quat_inv(root_q), robot.data.body_link_quat_w[0:1, a_idx])[0]
    sidecar: dict[str, Any] = {
        # ---- dropbear-policy-sidecar-v1 (CONTRACTS 6.1; parsed by dropbear_wbc.deploy.config.load_sidecar) ----
        "schema": DEPLOY_SIDECAR_SCHEMA,
        "policy_onnx": "policy_motion.onnx",
        "onnx_inputs": {"obs": "obs", "time_step": "time_step"},
        "step_dt": cfg.sim.dt * cfg.decimation,
        "joint_names": motor_names,
        "default_joint_pos": default,
        "joint_stiffness": kp,
        "joint_damping": kd,
        "action_scale": scale_list,
        "action_offset": default,
        "action_clip": None,
        # motor-side interpolation of each new position target over N physics steps (0 = step change)
        "target_interp_steps": target_interp_steps(robot),
        # motor position-target clamp [rad] per joint (lo, hi), applied AFTER scale + offset (docs/ISSUES.md #25)
        "target_clip": ([[float(term._clip[0, i, 0]), float(term._clip[0, i, 1])] for i in range(len(motor_names))]
                        if getattr(getattr(term, "cfg", None), "clip", None) is not None else None),
        "observations": [
            {"name": t["name"], "func": OBS_FUNC.get(t["name"], t["name"]), "dim": t["dim"], "scale": 1.0,
             "clip": None, "history_length": 1, "params": {}}
            for t in layout
        ],
        "obs_dim": obs_dim,
        "motion": {
            "source": "onnx",
            "anchor_body_name": command.cfg.anchor_body_name,
            "anchor_offset_pos": [round(float(v), 6) for v in rel_p],
            "anchor_offset_quat": [round(float(v), 6) for v in rel_q],
            "align": "yaw_xy",
            "reference_joint_names": motor_names,
            "body_names": list(command.cfg.body_names),
            "num_frames": int(command.motion.time_step_total),
        },
        "task": task,
        "usd_sha256": N.USD_SHA256,
        "run_path": str(Path(checkpoint).parent),
        # ---- Dropbear tracking details (ignored by the deploy parser) ----
        "dropbear_tracking": {
            "schema": SIDECAR_SCHEMA,
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "motor_contract": "dropbear-wbc-motors-v1",
            "timing": {"physics_dt": cfg.sim.dt, "decimation": cfg.decimation, "policy_dt": cfg.sim.dt * cfg.decimation},
            "observation_layout": layout,
            "critic_observation_layout": _obs_layout(env, "critic"),
            "normalization": "EmpiricalNormalization baked into policy.pt / policy.onnx / policy_motion.onnx",
            "normalizer": norm_state,
            "command_layout": "reference motor joint_pos (22, contract order) then joint_vel (22)",
            "frames": "anchor = raw link frame of anchor_body (BeyondMimic convention); base_lin_vel/base_ang_vel = "
            "root 'world' CoM velocity in the root link frame; policy_motion.onnx body_* outputs are link poses / CoM "
            "velocities in the motion's world frame",
            "sim_only_terms": [t["name"] for t in layout if t["name"] in ("motion_anchor_pos_b", "base_lin_vel")],
            "target_law": "q_target = default_joint_pos + action_scale * action (no clipping); implicit PD",
            "effort_limit": effort,
            "root_body": command.cfg.root_body_name,
            "tracked_bodies": list(command.cfg.body_names),
            "default_pose_source": getattr(cfg, "default_pose_source", ""),
            "default_pose_info": dict(getattr(cfg, "default_pose_info", {}) or {}),
            "solver_iterations": [
                cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
                cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count,
            ],
            "files": {"policy_plain_onnx": "policy.onnx (obs -> actions)", "torchscript": "policy.pt (obs -> actions)"},
            "provenance": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "motion_file": str(motion_file),
                "motion_sha256": sha256_file(motion_file),
                "motion_meta_status": (getattr(command.motion, "arrays_meta", {}) or {}).get("status"),
                # tools/validate_motion_npz.py verdict of the reference; a policy trained/exported on a
                # 'rejected' clip carries the flag (review finding 2026-09-24)
                "motion_validation": getattr(command.motion, "validation", None),
                "allow_rejected_motion": bool(getattr(command.cfg, "allow_rejected_motion", False)),
                "usd_path": cfg.scene.robot.spawn.usd_path,
            },
        },
    }
    # ONNX metadata (BeyondMimic keys, motor-only joint lists)
    meta = {
        "joint_names": motor_names,
        "joint_stiffness": kp,
        "joint_damping": kd,
        "default_joint_pos": default,
        "action_scale": scale_list,
        "observation_names": env.observation_manager.active_terms["policy"],
        "anchor_body_name": command.cfg.anchor_body_name,
        "body_names": list(command.cfg.body_names),
        "command_names": env.command_manager.active_terms,
        "sidecar": "policy.json",
    }
    for name in ("policy.onnx", "policy_motion.onnx"):
        _attach_onnx_metadata(out / name, meta)

    # parity: TorchScript export vs live policy (+ normalizer) on random observations
    with torch.inference_mode():
        x = torch.randn(64, obs_dim, device=runner.device) * 2.0
        live = policy.act_inference(normalizer(x)).cpu()
        jit = torch.jit.load(str(out / "policy.pt"), map_location="cpu")
        exp = jit(x.cpu())
    sidecar["dropbear_tracking"]["parity"] = {"torchscript_max_abs_diff": float((live - exp).abs().max()), "samples": 64}
    sidecar["dropbear_tracking"]["sha256"] = {n: sha256_file(out / n) for n in ("policy.pt", "policy.onnx", "policy_motion.onnx")}
    (out / "policy.json").write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8")
    torch.save({"obs": x.cpu(), "actions": live}, out / "parity_samples.pt")
    return sidecar


def latest_checkpoint(run_dir: str | Path) -> Path:
    """Highest-iteration ``model_<it>.pt`` in ``run_dir``."""
    run = Path(run_dir)
    models = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not models:
        raise FileNotFoundError(f"no model_*.pt in {run}")
    return models[-1]


def resolve_run_dir(log_root: str | Path, load_run: str | None) -> Path:
    """``log_root/load_run`` or the most recent run directory in ``log_root``."""
    root = Path(log_root)
    if load_run:
        return root / load_run
    runs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: os.path.getmtime(p))
    if not runs:
        raise FileNotFoundError(f"no runs in {root}")
    return runs[-1]
