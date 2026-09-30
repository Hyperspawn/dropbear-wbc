"""Export a motion-LIBRARY tracking policy with a RUNTIME reference (docs/CONTRACTS.md 5.3 / 6.1).

A single-clip BeyondMimic export bakes its clip into ``policy_motion.onnx`` (inputs ``obs`` + ``time_step``). A library
policy tracks whatever reference it is fed, so its export bakes NO reference: the deploy side streams reference frames
from any contract NPZ / CSV (``tools/policy_runner.py --motion <file>``; later a live stream) into the same observation
builder. This is the SONIC-like interface: policy = f(robot state, reference window).

Writes into ``out_dir``:

* ``policy.pt`` / ``policy.onnx``: ``actions = actor(normalizer(obs))`` (Isaac Lab exporters; ONNX input ``obs``);
* ``policy.json``: ``dropbear-policy-sidecar-v1`` with ``policy_onnx: policy.onnx``, ``onnx_inputs.time_step: null`` and
  ``motion.source: "runtime"`` (+ the reference requirements a runtime NPZ must meet: plant USD, ankle variant, fps,
  anchor body/offset, future frames), plus the ``dropbear_tracking`` block with ``export_type:
  "library_runtime_reference"``, the library fingerprint and per-clip provenance;
* ``parity_samples.pt``: 64 random observations and the live policy's actions.

No ``policy_motion.onnx`` is written.
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
from dropbear_wbc.tasks.tracking.export import (
    DEPLOY_SIDECAR_SCHEMA,
    OBS_FUNC,
    SIDECAR_SCHEMA,
    _attach_onnx_metadata,
    _obs_layout,
)

EXPORT_TYPE = "library_runtime_reference"
MOTION_SOURCE_RUNTIME = "runtime"


def export_library_policy(
    env,
    runner,
    out_dir: str | Path,
    *,
    task: str,
    checkpoint: str | Path,
    manifest: str | Path,
) -> dict[str, Any]:
    """Export ``runner``'s policy for the unwrapped library tracking ``env`` (command term ``motion`` must be a
    :class:`~.mdp.library_commands.MotionLibraryCommand`). Returns the sidecar dict (also ``policy.json``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    command = env.command_manager.get_term("motion")
    if not hasattr(command, "library_info"):
        raise TypeError("export_library_policy needs a motion-library command (use export.export_tracking_policy)")
    policy = runner.alg.policy
    normalizer = runner.obs_normalizer
    policy.eval()
    if hasattr(normalizer, "eval"):
        normalizer.eval()
    export_policy_as_jit(policy, normalizer, path=str(out), filename="policy.pt")
    export_policy_as_onnx(policy, normalizer=normalizer, path=str(out), filename="policy.onnx")
    stale = out / "policy_motion.onnx"
    if stale.exists():  # never leave a baked single-clip reference next to a runtime-reference export
        stale.unlink()

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
    default = [float(nominal[i]) for i in motor_ids]
    obs_dim = int(policy.actor[0].in_features)

    norm_state = {}
    if hasattr(normalizer, "_mean") and hasattr(normalizer, "_std"):
        norm_state = {"mean": normalizer._mean.squeeze(0).tolist(), "std": normalizer._std.squeeze(0).tolist(),
                      "eps": float(normalizer.eps)}

    cfg = env.cfg
    layout = _obs_layout(env, "policy")
    future = [int(k) for k in getattr(command.cfg, "future_steps", ()) or ()]
    cmd_dim = 44 * (1 + len(future))
    for t in layout:
        if t["name"] == "command" and t["dim"] != cmd_dim:
            raise RuntimeError(f"command term has {t['dim']} values, expected {cmd_dim} for future_steps={future}")
    from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

    a_idx = robot.body_names.index(command.cfg.anchor_body_name)
    root_q = robot.data.body_link_quat_w[0:1, 0]
    rel_p = quat_apply_inverse(root_q, robot.data.body_link_pos_w[0:1, a_idx] - robot.data.body_link_pos_w[0:1, 0])[0]
    rel_q = quat_mul(quat_inv(root_q), robot.data.body_link_quat_w[0:1, a_idx])[0]
    spawn = cfg.scene.robot.spawn
    authored_ankle = len(tuple(getattr(spawn, "spherical_joint_overrides", ()) or ())) == 0
    info = dict(command.library_info)
    fps = float(info.get("fps") or 1.0 / (cfg.sim.dt * cfg.decimation))

    def obs_entry(t: dict) -> dict:
        params = {"future_steps": future} if (t["name"] == "command" and future) else {}
        return {"name": t["name"], "func": OBS_FUNC.get(t["name"], t["name"]), "dim": t["dim"], "scale": 1.0,
                "clip": None, "history_length": 1, "params": params}

    sidecar: dict[str, Any] = {
        # ---- dropbear-policy-sidecar-v1 (CONTRACTS 6.1; dropbear_wbc.deploy.config.load_sidecar) ----
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
        "observations": [obs_entry(t) for t in layout],
        "obs_dim": obs_dim,
        "motion": {
            "source": MOTION_SOURCE_RUNTIME,
            "anchor_body_name": command.cfg.anchor_body_name,
            "anchor_offset_pos": [round(float(v), 6) for v in rel_p],
            "anchor_offset_quat": [round(float(v), 6) for v in rel_q],
            "align": "yaw_xy",
            "reference_joint_names": motor_names,
            "body_names": list(command.cfg.body_names),
            "num_frames": None,
            "fps": fps,
            "future_steps": future,
            "requirements": {
                "schema": "dropbear-motion-npz-v1 (or dropbear-motion-csv-v1)",
                "fps": fps,
                "usd_sha256": N.USD_SHA256,
                "authored_ankle_tierods": authored_ankle,
                "motor_names": motor_names,
                "anchor_body_name": command.cfg.anchor_body_name,
                "validator_verdict": "accepted (tools/validate_motion_npz.py); a rejected clip needs "
                                     "--allow-rejected-motion",
            },
            "library": {k: info.get(k) for k in ("name", "sha256", "num_clips")},
        },
        "task": task,
        "usd_sha256": N.USD_SHA256,
        "run_path": str(Path(checkpoint).parent),
        # ---- Dropbear tracking details (ignored by the deploy parser) ----
        "dropbear_tracking": {
            "schema": SIDECAR_SCHEMA,
            "export_type": EXPORT_TYPE,
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "motor_contract": "dropbear-wbc-motors-v1",
            "timing": {"physics_dt": cfg.sim.dt, "decimation": cfg.decimation, "policy_dt": cfg.sim.dt * cfg.decimation},
            "observation_layout": layout,
            "critic_observation_layout": _obs_layout(env, "critic"),
            "normalization": "EmpiricalNormalization baked into policy.pt / policy.onnx",
            "normalizer": norm_state,
            "command_layout": ("reference motor joint_pos (22, contract order) then joint_vel (22) at frame t"
                               + "".join(f", then joint_pos (22) + joint_vel (22) at frame t+{k}" for k in future)
                               + " (future frames of the same clip, clamped to its last frame)"),
            "reference": "NOT embedded: fed at runtime (tools/policy_runner.py --motion <npz|csv>); no policy_motion.onnx",
            "frames": "anchor = raw link frame of anchor_body (BeyondMimic convention); base_lin_vel/base_ang_vel = "
                      "root 'world' CoM velocity in the root link frame",
            "sim_only_terms": [t["name"] for t in layout if t["name"] in ("motion_anchor_pos_b", "base_lin_vel")],
            "target_law": "q_target = default_joint_pos + action_scale * action (no clipping); implicit PD",
            "effort_limit": effort,
            "root_body": command.cfg.root_body_name,
            "tracked_bodies": list(command.cfg.body_names),
            "default_pose_source": getattr(cfg, "default_pose_source", ""),
            "default_pose_info": dict(getattr(cfg, "default_pose_info", {}) or {}),
            "solver_iterations": [spawn.articulation_props.solver_position_iteration_count,
                                  spawn.articulation_props.solver_velocity_iteration_count],
            "library": info,
            "provenance": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "library_sha256": info.get("sha256"),
                "all_clips_accepted": info.get("all_accepted"),
                "allow_rejected_motion": bool(getattr(command.cfg, "allow_rejected_motion", False)),
                "authored_ankle_tierods": authored_ankle,
                "usd_path": spawn.usd_path,
            },
        },
    }
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
        "motion_source": MOTION_SOURCE_RUNTIME,
        "future_steps": future if future else "none",
    }
    _attach_onnx_metadata(out / "policy.onnx", meta)

    with torch.inference_mode():
        x = torch.randn(64, obs_dim, device=runner.device) * 2.0
        live = policy.act_inference(normalizer(x)).cpu()
        jit = torch.jit.load(str(out / "policy.pt"), map_location="cpu")
        exp = jit(x.cpu())
    sidecar["dropbear_tracking"]["parity"] = {"torchscript_max_abs_diff": float((live - exp).abs().max()), "samples": 64}
    sidecar["dropbear_tracking"]["sha256"] = {n: sha256_file(out / n) for n in ("policy.pt", "policy.onnx")}
    (out / "policy.json").write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8")
    torch.save({"obs": x.cpu(), "actions": live}, out / "parity_samples.pt")
    return sidecar
