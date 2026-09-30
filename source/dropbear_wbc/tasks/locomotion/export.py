"""Export a trained Dropbear velocity policy for deployment / sim2sim (``tools/policy_runner.py --sidecar``).

Writes into ``out_dir``:

* ``policy.pt``    TorchScript, ``actions = actor(normalizer(obs))`` (Isaac Lab ``export_policy_as_jit``);
* ``policy.onnx``  same graph as ONNX (input ``obs`` (1, obs_dim), output ``actions`` (1, 22));
* ``policy.json``  sidecar ``dropbear-policy-sidecar-v1`` (CONTRACTS 6.1; ``motion`` null, ``onnx_inputs.time_step``
  null) whose ordered ``observations`` use the ``dropbear_wbc.deploy.observations`` functions ``base_lin_vel``
  (sim-only), ``base_ang_vel``, ``projected_gravity``, ``velocity_commands``, ``joint_pos_rel``, ``joint_vel_rel``,
  ``last_action``; plus a ``dropbear_velocity`` block (layout, normalizer, command ranges, provenance, parity);
* ``parity_samples.pt``  64 random observations and the live policy's actions.

Joint target law (per motor, contract order): ``q_target = default_pos + action_scale * action`` (implicit PD).
The command is ``(vx, vy, wz)``: planar CoM velocity of the root body in its yaw frame [m/s] and yaw rate [rad/s].
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import torch

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

from dropbear_wbc.isaac.launch import sha256_file
from dropbear_wbc.robots import dropbear_names as N
from dropbear_wbc.tasks.tracking.export import _attach_onnx_metadata, _obs_layout

SIDECAR_SCHEMA = "dropbear-velocity-policy-v1"
DEPLOY_SIDECAR_SCHEMA = "dropbear-policy-sidecar-v1"
OBS_FUNC: dict[str, str] = {
    "base_lin_vel": "base_lin_vel",
    "base_ang_vel": "base_ang_vel",
    "projected_gravity": "projected_gravity",
    "velocity_commands": "velocity_commands",
    "joint_pos": "joint_pos_rel",
    "joint_vel": "joint_vel_rel",
    "actions": "last_action",
}
"""Policy ObsGroup term name -> deploy observation function (``dropbear_wbc.deploy.observations``)."""
SIM_ONLY_TERMS = ("base_lin_vel",)


def export_velocity_policy(env, runner, out_dir: str | Path, *, task: str, checkpoint: str | Path) -> dict[str, Any]:
    """Export ``runner``'s policy (with its observation normalizer) for the unwrapped velocity ``env``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy = runner.alg.policy
    normalizer = runner.obs_normalizer
    policy.eval()
    if hasattr(normalizer, "eval"):
        normalizer.eval()
    export_policy_as_jit(policy, normalizer, path=str(out), filename="policy.pt")
    export_policy_as_onnx(policy, normalizer=normalizer, path=str(out), filename="policy.onnx")

    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    motor_names = list(term._joint_names)
    if tuple(motor_names) != N.MOTOR_NAMES:
        raise RuntimeError(f"action joints {motor_names} != motor contract")
    motor_ids = robot.find_joints(motor_names, preserve_order=True)[0]
    scale = term._scale
    scale_list = scale[0].tolist() if isinstance(scale, torch.Tensor) else [float(scale)] * len(motor_names)
    kp = robot.data.joint_stiffness[0, motor_ids].tolist()
    kd = robot.data.joint_damping[0, motor_ids].tolist()
    effort = robot.data.joint_effort_limits[0, motor_ids].tolist()
    # explicit (hw_*) motor models run the PD themselves: read their own gains and peak torque
    from dropbear_wbc.robots.hw_introspect import overlay_explicit_gains, target_interp_steps

    overlay_explicit_gains(robot, motor_names, kp, kd, effort)
    default = [float(robot.data.default_joint_pos[0, i]) for i in motor_ids]
    obs_dim = int(policy.actor[0].in_features)
    norm_state = {}
    if hasattr(normalizer, "_mean") and hasattr(normalizer, "_std"):
        norm_state = {"mean": normalizer._mean.squeeze(0).tolist(), "std": normalizer._std.squeeze(0).tolist(),
                      "eps": float(normalizer.eps)}
    cfg = env.cfg
    layout = _obs_layout(env, "policy")
    unknown = [t["name"] for t in layout if t["name"] not in OBS_FUNC]
    if unknown:
        raise RuntimeError(f"policy observation terms without a deploy function: {unknown}")
    cmd_cfg = env.command_manager.get_term("base_velocity").cfg
    ranges = {k: [float(v) for v in getattr(cmd_cfg.ranges, k)] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")}
    reset_term = env.event_manager.get_term_cfg("reset_robot").func
    sidecar: dict[str, Any] = {
        "schema": DEPLOY_SIDECAR_SCHEMA,
        "policy_onnx": "policy.onnx",
        "onnx_inputs": {"obs": "obs", "time_step": None},
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
        # motor position-target clamp [rad] per joint (lo, hi), applied AFTER scale + offset (gait-shaped runs)
        "target_clip": ([[float(term._clip[0, i, 0]), float(term._clip[0, i, 1])] for i in range(len(motor_names))]
                        if getattr(getattr(term, "cfg", None), "clip", None) is not None else None),
        "observations": [
            {"name": t["name"], "func": OBS_FUNC[t["name"]], "dim": t["dim"], "scale": 1.0, "clip": None,
             "history_length": 1, "params": {}}
            for t in layout
        ],
        "obs_dim": obs_dim,
        "motion": None,
        "task": task,
        "usd_sha256": N.USD_SHA256,
        "run_path": str(Path(checkpoint).parent),
        "dropbear_velocity": {
            "schema": SIDECAR_SCHEMA,
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "motor_contract": "dropbear-wbc-motors-v1",
            "timing": {"physics_dt": cfg.sim.dt, "decimation": cfg.decimation, "policy_dt": cfg.sim.dt * cfg.decimation},
            "observation_layout": layout,
            "critic_observation_layout": _obs_layout(env, "critic"),
            "normalization": "EmpiricalNormalization baked into policy.pt / policy.onnx",
            "normalizer": norm_state,
            "command": "velocity_commands = (vx, vy, wz): root CoM planar velocity in the root yaw frame [m/s], yaw "
                       "rate [rad/s]; command ranges at export below",
            "command_ranges_at_export": ranges,
            "frames": "base_lin_vel/base_ang_vel = root 'world' CoM linear / angular velocity in the root link frame; "
                      "projected_gravity = unit gravity in the root link frame (root = rigid torso+pelvis, world-aligned "
                      "at the standing pose; its frame ORIGIN is ~12.5 cm below the soles)",
            "sim_only_terms": [t["name"] for t in layout if t["name"] in SIM_ONLY_TERMS],
            "target_law": ("q_target = default_joint_pos + action_scale * action, then clipped to target_clip if set; "
                           "PD by the motor model (implicit, or explicit hw_* DatasheetMotor)"),
            "effort_limit": effort,
            "actuator_profile": getattr(cfg, "actuator_profile", ""),
            "default_pose_source": getattr(cfg, "default_pose_source", ""),
            "default_pose_info": dict(getattr(cfg, "default_pose_info", {}) or {}),
            "solver_iterations": [
                cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
                cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count,
            ],
            "reset_state": {"npz": reset_term.npz_path, "npz_sha256": sha256_file(reset_term.npz_path),
                            "note": "start a deployment from this settled standing state (full joint row + root)"},
            "provenance": {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                           "usd_path": cfg.scene.robot.spawn.usd_path},
        },
    }
    meta = {
        "joint_names": motor_names,
        "joint_stiffness": kp,
        "joint_damping": kd,
        "default_joint_pos": default,
        "action_scale": scale_list,
        "observation_names": env.observation_manager.active_terms["policy"],
        "command_names": env.command_manager.active_terms,
        "sidecar": "policy.json",
    }
    _attach_onnx_metadata(out / "policy.onnx", meta)

    with torch.inference_mode():
        x = torch.randn(64, obs_dim, device=runner.device) * 2.0
        live = policy.act_inference(normalizer(x)).cpu()
        jit = torch.jit.load(str(out / "policy.pt"), map_location="cpu")
        exp = jit(x.cpu())
    parity = {"torchscript_max_abs_diff": float((live - exp).abs().max()), "samples": 64,
              "live_max_abs_action": float(live.abs().max())}
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out / "policy.onnx"), providers=["CPUExecutionProvider"])
        onnx_out = np.concatenate([sess.run(None, {"obs": x[i:i + 1].cpu().numpy()})[0] for i in range(x.shape[0])])
        parity["onnx_max_abs_diff"] = float(np.abs(onnx_out - live.numpy()).max())
    except Exception as exc:  # noqa: BLE001
        parity["onnx_check"] = f"skipped: {type(exc).__name__}: {exc}"
    sidecar["dropbear_velocity"]["parity"] = parity
    sidecar["dropbear_velocity"]["sha256"] = {n: sha256_file(out / n) for n in ("policy.pt", "policy.onnx")}
    (out / "policy.json").write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8")
    torch.save({"obs": x.cpu(), "actions": live}, out / "parity_samples.pt")
    return sidecar
