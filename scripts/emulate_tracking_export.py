"""Run the REAL ``tasks/tracking/export.py`` on CPU without Isaac, to check its output against the runner.

``scripts/play.py --export`` needs Isaac Sim, a trained checkpoint and the GPU. This script runs the same
``export_tracking_policy`` function in plain Python instead:

* **Stubbed (Isaac-only):** ``isaaclab_rl.rsl_rl.export_policy_as_jit/onnx`` (re-implemented with Isaac Lab
  2.2's exporter semantics: ``actor(normalizer(obs))``, ONNX input ``obs`` / output ``actions``, opset 11)
  and ``isaaclab.utils.math.quat_mul/quat_inv/quat_apply_inverse`` (wxyz).
* **Mocked:** the env and runner handles that the export reads. Their joint and body order, default pose,
  body poses and reference come from a real contract NPZ.
* **Real:** the vendored rsl-rl 2.3.3 ``ActorCritic`` and ``EmpiricalNormalization``. The weights are
  random and the normalizer statistics are synthetic.

The output directory has the same files as a real export (``policy.json``, ``policy.onnx``,
``policy_motion.onnx``, ``policy.pt``, ``parity_samples.pt``) plus ``EMULATED.json``, which says what was
mocked. The policy is untrained, so this tells you nothing about tracking quality. It checks only the
format and conventions of the export seen by ``tools/policy_runner.py --sidecar``.

Needs torch + onnx + onnxruntime (the system Python 3.12)::

    python scripts/emulate_tracking_export.py \
        --npz data/motions/smoke/dropbear_static_stand.npz --out logs/sdk_bridge/export_emulated
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPORT_PY = ROOT / "source" / "dropbear_wbc" / "tasks" / "tracking" / "export.py"
DEFAULT_NPZ = ROOT / "data" / "motions" / "smoke" / "dropbear_static_stand.npz"
for p in (ROOT / "source", ROOT / "third_party" / "pydeps"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Policy observation group of the tracking task (CONTRACTS 5.1, tracking_env_cfg.PolicyCfg), in order.
POLICY_TERMS = (("command", 44), ("motion_anchor_pos_b", 3), ("motion_anchor_ori_b", 6), ("base_lin_vel", 3),
                ("base_ang_vel", 3), ("joint_pos", 22), ("joint_vel", 22), ("actions", 22))
CRITIC_TERMS = (("command", 44), ("motion_anchor_pos_b", 3), ("motion_anchor_ori_b", 6), ("body_pos", 42),
                ("body_ori", 84), ("base_lin_vel", 3), ("base_ang_vel", 3), ("joint_pos", 22), ("joint_vel", 22),
                ("actions", 22))


# --------------------------------------------------------------------------------------------- stubs
def _isaac_math_module():
    import torch

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=-1)

    def quat_inv(q, eps: float = 1e-9):
        conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
        return conj / (q * q).sum(-1, keepdim=True).clamp(min=eps)

    def quat_apply_inverse(quat, vec):
        xyz = quat[..., 1:]
        t = torch.cross(xyz, vec, dim=-1) * 2
        return vec - quat[..., :1] * t + torch.cross(xyz, t, dim=-1)

    m = types.ModuleType("isaaclab.utils.math")
    m.quat_mul, m.quat_inv, m.quat_apply_inverse = quat_mul, quat_inv, quat_apply_inverse
    return m


def _isaac_rsl_rl_module():
    import torch

    class _Exporter(torch.nn.Module):  # Isaac Lab 2.2 _TorchPolicyExporter/_OnnxPolicyExporter (MLP branch)
        def __init__(self, policy, normalizer=None):
            super().__init__()
            self.actor = copy.deepcopy(policy.actor)
            self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

        def forward(self, x):
            return self.actor(self.normalizer(x))

    def export_policy_as_jit(policy, normalizer, path, filename="policy.pt"):
        os.makedirs(path, exist_ok=True)
        m = _Exporter(policy, normalizer).to("cpu")
        torch.jit.script(m).save(os.path.join(path, filename))

    def export_policy_as_onnx(policy, path, normalizer=None, filename="policy.onnx", verbose=False):
        os.makedirs(path, exist_ok=True)
        m = _Exporter(policy, normalizer).to("cpu").eval()
        obs = torch.zeros(1, m.actor[0].in_features)
        torch.onnx.export(m, obs, os.path.join(path, filename), export_params=True, opset_version=11,
                          verbose=verbose, input_names=["obs"], output_names=["actions"], dynamic_axes={})

    m = types.ModuleType("isaaclab_rl.rsl_rl")
    m.export_policy_as_jit, m.export_policy_as_onnx = export_policy_as_jit, export_policy_as_onnx
    return m


@contextlib.contextmanager
def stubbed_isaac_modules():
    """Temporarily install the stub modules (restores ``sys.modules`` afterwards)."""
    stubs = {"isaaclab": types.ModuleType("isaaclab"), "isaaclab.utils": types.ModuleType("isaaclab.utils"),
             "isaaclab.utils.math": _isaac_math_module(), "isaaclab_rl": types.ModuleType("isaaclab_rl"),
             "isaaclab_rl.rsl_rl": _isaac_rsl_rl_module()}
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def load_export_module():
    """Import ``tasks/tracking/export.py`` by path (its package ``__init__`` needs Isaac Lab)."""
    spec = importlib.util.spec_from_file_location("_dropbear_tracking_export_under_test", EXPORT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------- mocks
def _ns(**kw):
    return types.SimpleNamespace(**kw)


def build_mocks(npz_path: Path, seed: int = 0, hidden: tuple[int, ...] = (64, 64), frame: int = 0):
    """(env, runner, info) built from a contract NPZ and a random rsl-rl 2.3.3 ActorCritic."""
    import torch
    from rsl_rl.modules import ActorCritic, EmpiricalNormalization

    from dropbear_wbc.robots import dropbear_names as N

    d = np.load(npz_path, allow_pickle=False)
    joints = [str(j) for j in d["joint_names"]]
    bodies = [str(b) for b in d["body_names"]]
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    motor_ids = [joints.index(m) for m in N.MOTOR_NAMES]
    tracked = [bodies.index(b) for b in N.TRACKED_BODIES]
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731

    default = np.asarray(d["joint_pos"][frame], float).copy()
    dmp = meta.get("default_motor_pos")
    if isinstance(dmp, dict):
        for name, v in dmp.items():
            default[joints.index(name)] = float(v)
    kp, kd, effort = np.zeros(len(joints)), np.zeros(len(joints)), np.zeros(len(joints))
    for name, j in zip(N.MOTOR_NAMES, motor_ids):
        effort[j], kp[j], kd[j] = (N.motor_param(name, i) for i in (0, 1, 2))
    scale = np.array([0.25 * effort[j] / kp[j] for j in motor_ids])  # BeyondMimic rule (CONTRACTS 5.1)

    def find_joints(names, preserve_order=True):
        return [joints.index(n) for n in names], list(names)

    robot = _ns(joint_names=joints, body_names=bodies, find_joints=find_joints,
                data=_ns(default_joint_pos=t(default)[None], joint_stiffness=t(kp)[None], joint_damping=t(kd)[None],
                         joint_effort_limits=t(effort)[None], body_link_pos_w=t(d["body_pos_w"][frame])[None],
                         body_link_quat_w=t(d["body_quat_w"][frame])[None]))
    motion = _ns(joint_pos=t(d["joint_pos"][:, motor_ids]), joint_vel=t(d["joint_vel"][:, motor_ids]),
                 body_pos_w=t(d["body_pos_w"][:, tracked]), body_quat_w=t(d["body_quat_w"][:, tracked]),
                 body_lin_vel_w=t(d["body_lin_vel_w"][:, tracked]), body_ang_vel_w=t(d["body_ang_vel_w"][:, tracked]),
                 time_step_total=int(d["joint_pos"].shape[0]))
    command = _ns(motion=motion, cfg=_ns(anchor_body_name=N.ANCHOR_BODY, body_names=list(N.TRACKED_BODIES),
                                         root_body_name=N.ROOT_BODY))
    env = _ns(
        scene={"robot": robot},
        command_manager=_ns(get_term=lambda name: command, active_terms=["motion"]),
        action_manager=_ns(get_term=lambda name: _ns(_joint_names=list(N.MOTOR_NAMES), _scale=t(scale)[None])),
        observation_manager=_ns(active_terms={"policy": [n for n, _ in POLICY_TERMS], "critic": [n for n, _ in CRITIC_TERMS]},
                                group_obs_term_dim={"policy": [(k,) for _, k in POLICY_TERMS],
                                                    "critic": [(k,) for _, k in CRITIC_TERMS]}),
        cfg=_ns(sim=_ns(dt=0.005), decimation=4, default_pose_source=meta.get("default_pose_source", "npz_frame0"),
                scene=_ns(robot=_ns(spawn=_ns(usd_path=N.DEFAULT_USD_PATH, articulation_props=_ns(
                    solver_position_iteration_count=32, solver_velocity_iteration_count=4))))),
    )
    obs_dim, critic_dim = sum(k for _, k in POLICY_TERMS), sum(k for _, k in CRITIC_TERMS)
    torch.manual_seed(seed)
    policy = ActorCritic(obs_dim, critic_dim, len(N.MOTOR_NAMES), actor_hidden_dims=list(hidden),
                         critic_hidden_dims=list(hidden), activation="elu")
    normalizer = EmpiricalNormalization(shape=[obs_dim], until=1.0e8)
    with torch.no_grad():  # synthetic statistics so the baked-in normalizer is not the identity
        normalizer._mean.copy_(torch.randn(1, obs_dim) * 0.1)
        normalizer._std.copy_(torch.rand(1, obs_dim) + 0.5)
    runner = _ns(alg=_ns(policy=policy), obs_normalizer=normalizer, device="cpu")
    info = {"npz": str(npz_path), "motor_ids": motor_ids, "tracked_body_ids": tracked, "default_full": default.tolist(),
            "action_scale": scale.tolist(), "obs_dim": obs_dim, "seed": seed, "hidden": list(hidden)}
    return env, runner, info


def emulate_export(out_dir: Path, npz_path: Path = DEFAULT_NPZ, seed: int = 0) -> dict:
    """Run ``export_tracking_policy`` on mocks; returns the sidecar dict it wrote."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env, runner, info = build_mocks(Path(npz_path), seed=seed)
    ckpt = out_dir / "model_emulated.pt"
    ckpt.write_bytes(b"emulated checkpoint placeholder (random weights; not a trained model)\n")
    with stubbed_isaac_modules():
        mod = load_export_module()
        sidecar = mod.export_tracking_policy(env, runner, out_dir, task="Dropbear-Tracking-Flat-Play-v0 (EMULATED)",
                                             checkpoint=ckpt, motion_file=npz_path)
    (out_dir / "EMULATED.json").write_text(json.dumps({
        "what": "tasks/tracking/export.py run on CPU with Isaac stubs and mocked env/runner",
        "stubbed": ["isaaclab_rl.rsl_rl.export_policy_as_jit", "isaaclab_rl.rsl_rl.export_policy_as_onnx",
                    "isaaclab.utils.math.quat_mul/quat_inv/quat_apply_inverse"],
        "real": ["tasks/tracking/export.py", "rsl_rl 2.3.3 ActorCritic + EmpiricalNormalization (random init)",
                 "contract NPZ joint/body order, default pose, reference"],
        "not_a_trained_policy": True, **info}, indent=1) + "\n", encoding="utf-8")
    return sidecar


# --------------------------------------------------------------------------------------------- motion library
EXPORT_LIBRARY_PY = ROOT / "source" / "dropbear_wbc" / "tasks" / "tracking" / "export_library.py"
DEFAULT_MANIFEST = ROOT / "data" / "motions" / "libraries" / "accepted_v0.json"


def build_library_mocks(manifest: Path, seed: int = 0, future_steps: tuple[int, ...] = (), hidden=(64, 64)):
    """(env, runner, info) for ``export_library.export_library_policy``: the single-clip mocks built on the library's
    first clip (joint/body order, default pose, anchor offset), a library command (real ``build_library`` report,
    no reference arrays: the library export must not need them) and the command term sized for ``future_steps``."""
    import torch

    from dropbear_wbc.robots import dropbear_names as N
    from dropbear_wbc.tasks.tracking.motion_library import build_library, library_report

    data = build_library(manifest, keep_body_names=list(N.TRACKED_BODIES))
    env, _, info = build_mocks(Path(data.clips[0].path), seed=seed, hidden=hidden)
    k = len(future_steps)
    terms = (("command", 44 * (1 + k)),) + POLICY_TERMS[1:]
    critic = (("command", 44 * (1 + k)),) + CRITIC_TERMS[1:]
    env.observation_manager = _ns(active_terms={"policy": [n for n, _ in terms], "critic": [n for n, _ in critic]},
                                  group_obs_term_dim={"policy": [(d,) for _, d in terms], "critic": [(d,) for _, d in critic]})
    command = _ns(library_info=library_report(data), cfg=_ns(
        anchor_body_name=N.ANCHOR_BODY, body_names=list(N.TRACKED_BODIES), root_body_name=N.ROOT_BODY,
        future_steps=tuple(future_steps), allow_rejected_motion=False))
    env.command_manager = _ns(get_term=lambda name: command, active_terms=["motion"])
    env.cfg.scene.robot.spawn.spherical_joint_overrides = N.SPHERICAL_JOINT_FIXES
    from rsl_rl.modules import ActorCritic, EmpiricalNormalization

    obs_dim, critic_dim = sum(d for _, d in terms), sum(d for _, d in critic)
    torch.manual_seed(seed)
    policy = ActorCritic(obs_dim, critic_dim, len(N.MOTOR_NAMES), actor_hidden_dims=list(hidden),
                         critic_hidden_dims=list(hidden), activation="elu")
    normalizer = EmpiricalNormalization(shape=[obs_dim], until=1.0e8)
    with torch.no_grad():
        normalizer._mean.copy_(torch.randn(1, obs_dim) * 0.1)
        normalizer._std.copy_(torch.rand(1, obs_dim) + 0.5)
    runner = _ns(alg=_ns(policy=policy), obs_normalizer=normalizer, device="cpu")
    info.update(manifest=str(manifest), library_sha256=data.fingerprint["sha256"], future_steps=list(future_steps),
                obs_dim=obs_dim)
    return env, runner, info


def emulate_library_export(out_dir: Path, manifest: Path = DEFAULT_MANIFEST, seed: int = 0,
                           future_steps: tuple[int, ...] = ()) -> dict:
    """Run the REAL ``export_library.export_library_policy`` on mocks; returns the sidecar dict it wrote."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env, runner, info = build_library_mocks(Path(manifest), seed=seed, future_steps=tuple(future_steps))
    ckpt = out_dir / "model_emulated.pt"
    ckpt.write_bytes(b"emulated checkpoint placeholder (random weights; not a trained model)\n")
    with stubbed_isaac_modules():
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "dropbear_wbc.tasks.tracking.export"}
        try:
            spec = importlib.util.spec_from_file_location("_dropbear_library_export_under_test", EXPORT_LIBRARY_PY)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sidecar = mod.export_library_policy(env, runner, out_dir, task="Dropbear-Tracking-Library-Play-v0 (EMULATED)",
                                                checkpoint=ckpt, manifest=Path(manifest))
        finally:  # never leave the stub-bound export module importable
            sys.modules.pop("dropbear_wbc.tasks.tracking.export", None)
            sys.modules.update(saved)
    (out_dir / "EMULATED.json").write_text(json.dumps({
        "what": "tasks/tracking/export_library.py run on CPU with Isaac stubs and mocked env/runner",
        "stubbed": ["isaaclab_rl.rsl_rl.export_policy_as_jit", "isaaclab_rl.rsl_rl.export_policy_as_onnx",
                    "isaaclab.utils.math.quat_mul/quat_inv/quat_apply_inverse"],
        "real": ["tasks/tracking/export_library.py", "tasks/tracking/motion_library.build_library/library_report",
                 "rsl_rl 2.3.3 ActorCritic + EmpiricalNormalization (random init)"],
        "not_a_trained_policy": True, **info}, indent=1) + "\n", encoding="utf-8")
    return sidecar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "sdk_bridge" / "export_emulated")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--library", type=Path, default=None,
                    help="emulate a motion-library export (runtime reference) from this manifest instead")
    ap.add_argument("--future_steps", type=int, nargs="*", default=[], help="with --library: future command frames")
    a = ap.parse_args()
    if a.library is not None:
        side = emulate_library_export(a.out, a.library, a.seed, tuple(a.future_steps))
        print(json.dumps({"out": str(a.out), "policy_onnx": side["policy_onnx"], "obs_dim": side["obs_dim"],
                          "observations": [(o["name"], o["func"], o["dim"], o["params"]) for o in side["observations"]],
                          "motion": {k: v for k, v in side["motion"].items() if k not in ("reference_joint_names",)},
                          "parity": side["dropbear_tracking"]["parity"]}, indent=1))
        return 0
    side = emulate_export(a.out, a.npz, a.seed)
    print(json.dumps({"out": str(a.out), "policy_onnx": side["policy_onnx"], "obs_dim": side["obs_dim"],
                      "observations": [(o["name"], o["func"], o["dim"]) for o in side["observations"]],
                      "motion": {k: v for k, v in side["motion"].items() if k not in ("reference_joint_names",)},
                      "parity": side["dropbear_tracking"]["parity"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
