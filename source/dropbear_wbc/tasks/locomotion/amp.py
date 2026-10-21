"""Adversarial Motion Prior (AMP) style reward for the velocity walker.

A discriminator learns to tell short transitions of the robot's own motion apart from transitions of retargeted human
motion clips (walking, turning, stepping, balancing; ``AMP_CLIPS``). The policy is rewarded for motion the
discriminator takes for human, so arm swing, weight shift and balance reactions come from the data instead of
hand-written gait terms (Peng et al. 2021, "AMP: Adversarial Motion Priors for Stylized Physics-Based Character
Control"; the least-squares GAN form with a gradient penalty on the reference samples).

It is a reward TERM (:class:`AmpStyleReward`) that owns the discriminator and trains it online from a replay buffer of
the robot's transitions, so the rsl-rl runner is unchanged. The discriminator is saved next to the policy
(``ckpt_path``, every update) and reloaded by the next chunk of a chunked run.

Features of one state (66 values; the same function for the robot and the clips): the 22 motor angles relative to the
standing pose and their velocities, the root height above the lower foot (negative: the USD root frame sits below
the soles; only its change matters), gravity, linear and angular velocity in the
root frame, and the feet and hands relative to the root in the root frame. A transition is two consecutive 50 Hz states.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

AMP_CLIPS: tuple[str, ...] = (
    # walking at several speeds, turning, stopping, backwards (generated, AMASS, LAFAN1; retargeted and settled)
    "walk_normal_ts1.19_v6", "walk_slow_ts1.19_v6", "walk_brisk_ts1.19_v6", "walk_circle_ts1.19_v6",
    "walk_turn_left_ts1.19_v6", "walk_turn_right_ts1.19_v6", "walk_back_ts1.19_v6", "walk_stop_ts1.19_v6",
    "walk_look_ts1.19_v6", "output_walk_ts1.19_v6", "Neutral_walk_forward_002__A057_ts1.19_v6", "turn_around_ts1.19_v6",
    "lafan1_walk1_subject1_v4",  # (lafan1_run1 left out: running is outside the walker's speed range)
    "motions_raw_tairantestbed_smpl_video_walk_level1_filter_amass_ts1.19_v6",
    "motions_raw_tairantestbed_smpl_video_walk_level2_filter_amass_ts1.19_v6",
    "motions_raw_tairantestbed_smpl_video_walk_level3_filter_amass_ts1.19_v6",
    "motions_raw_tairantestbed_smpl_video_walk_level4_filter_amass_ts1.19_v6",
    # stepping and balance (weight shift, single support)
    "step_fwd_back_ts1.19_v6",
    *(f"motions_raw_tairantestbed_smpl_video_step_forward_{d}_level{i}_filter_amass_ts1.19_v6"
      for d in ("back", "forward") for i in (1, 2, 3, 4)),
    *(f"motions_raw_tairantestbed_smpl_video_single_foot_balance_level{i}_filter_amass_ts1.19_v6" for i in (1, 2, 3, 4)),
)
DEFAULT_LIBRARY = "data/motions_v6ts/libraries/accepted_v7gen.json"
FEET = ("LL_skateboard_bearing_left_2", "RL_skateboard_bearing_left_2")
HANDS = ("LH_shoulder_ex_al_interface_1", "RH_shoulder_ex_al_interface_1")


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` (..., 3) by the inverse of the unit quaternion ``q`` (..., 4, wxyz)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


def amp_features(dq: torch.Tensor, qd: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor,
                 root_lin_vel_w: torch.Tensor, root_ang_vel_w: torch.Tensor, feet_pos: torch.Tensor,
                 hands_pos: torch.Tensor) -> torch.Tensor:
    """(N, 66) state features. ``dq``/``qd`` (N, 22): motor angles from the standing pose and velocities;
    ``root_*`` world frame (quat wxyz); ``feet_pos``/``hands_pos`` (N, 2, 3) world positions."""
    n = dq.shape[0]
    height = root_pos[:, 2:3] - torch.min(feet_pos[:, :, 2], dim=1, keepdim=True).values
    grav = _quat_rotate_inverse(root_quat, torch.tensor([0.0, 0.0, -1.0], device=dq.device).expand(n, 3))
    v = _quat_rotate_inverse(root_quat, root_lin_vel_w)
    w = _quat_rotate_inverse(root_quat, root_ang_vel_w)
    key = torch.cat([feet_pos, hands_pos], dim=1) - root_pos[:, None, :]
    key = _quat_rotate_inverse(root_quat[:, None, :].expand(-1, 4, -1), key).reshape(n, 12)
    return torch.cat([dq, qd, height, grav, v, w, key], dim=1)


def load_reference_transitions(motor_names, default_motor_pos: np.ndarray, library: Path, clips=AMP_CLIPS,
                               fps: float = 50.0, device: str = "cpu") -> tuple[torch.Tensor, dict]:
    """(M, 132) reference transitions (state_t, state_t+1) from the library clips, plus a short report."""
    lib = json.loads(library.read_text(encoding="utf-8"))
    by_name = {c["name"]: (library.parent / c["npz"]).resolve() for c in lib["clips"]}
    out, used = [], {}
    for name in clips:
        p = by_name.get(name)
        if p is None or not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        if abs(float(z["fps"]) - fps) > 1e-6:
            continue  # all clips are 50 Hz (the control rate); others would need resampling
        jn = [str(j) for j in z["joint_names"]]
        bn = [str(b) for b in z["body_names"]]
        mi = [jn.index(m) for m in motor_names]
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)  # noqa: E731
        dq = t(z["joint_pos"][:, mi] - default_motor_pos[None, :])
        qd = t(z["joint_vel"][:, mi])
        pos, quat = t(z["body_pos_w"]), t(z["body_quat_w"])
        f = amp_features(dq, qd, pos[:, 0], quat[:, 0], t(z["body_lin_vel_w"][:, 0]), t(z["body_ang_vel_w"][:, 0]),
                         pos[:, [bn.index(b) for b in FEET]], pos[:, [bn.index(b) for b in HANDS]])
        out.append(torch.cat([f[:-1], f[1:]], dim=1))
        used[name] = int(f.shape[0])
    if not out:
        raise FileNotFoundError(f"no AMP reference clip found in {library}")
    return torch.cat(out, dim=0), {"clips": used, "transitions": int(sum(v - 1 for v in used.values()))}


class Discriminator(nn.Module):
    def __init__(self, dim: int, hidden=(1024, 512), mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        super().__init__()
        layers, d = [], dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        self.register_buffer("mean", mean if mean is not None else torch.zeros(dim))
        self.register_buffer("std", std if std is not None else torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk((x - self.mean) / self.std)).squeeze(-1)


try:  # the reward term needs Isaac Lab; the data and network above do not (unit tests run without it)
    from isaaclab.managers import ManagerTermBase
except ImportError:  # pragma: no cover
    ManagerTermBase = object  # type: ignore[misc,assignment]


class AmpStyleReward(ManagerTermBase):
    """Style reward ``max(0, 1 - 0.25 (D(s, s') - 1)^2)`` with an online-trained discriminator ``D``.

    params: ``library`` (manifest path), ``ckpt_path`` (save / resume the discriminator; None = in memory only),
    ``update_every`` (env steps between discriminator updates; 24 = one rsl-rl rollout), ``num_updates`` and
    ``batch`` (per update), ``lr``, ``grad_penalty``, ``buffer_size`` (robot transitions kept)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES

        p = cfg.params
        self._robot = env.scene["robot"]
        self._motor_ids = self._robot.find_joints(list(MOTOR_NAMES), preserve_order=True)[0]
        self._feet = self._robot.find_bodies(list(FEET), preserve_order=True)[0]
        self._hands = self._robot.find_bodies(list(HANDS), preserve_order=True)[0]
        self._default = self._robot.data.default_joint_pos[0, self._motor_ids].clone()
        lib = Path(p.get("library", DEFAULT_LIBRARY))
        lib = lib if lib.is_absolute() else Path(__file__).resolve().parents[4] / lib
        self._ref, report = load_reference_transitions(list(MOTOR_NAMES), self._default.cpu().numpy(), lib, device=env.device)
        mean, std = self._ref.mean(0), self._ref.std(0).clamp_min(1e-3)
        self._disc = Discriminator(self._ref.shape[1], mean=mean, std=std).to(env.device)
        self._opt = torch.optim.Adam(self._disc.parameters(), lr=float(p.get("lr", 1e-4)))
        self._ckpt = Path(p["ckpt_path"]) if p.get("ckpt_path") else None
        if self._ckpt is not None and self._ckpt.exists():
            state = torch.load(self._ckpt, map_location=env.device)
            self._disc.load_state_dict(state["disc"])
            self._opt.load_state_dict(state["opt"])
            report["resumed"] = str(self._ckpt)
        self._buf = torch.zeros(int(p.get("buffer_size", 500_000)), self._ref.shape[1], device=env.device)
        self._buf_n, self._buf_i, self._calls, self._updates = 0, 0, 0, 0
        self._prev = None
        print(json.dumps({"amp": report | {"ref_shape": list(self._ref.shape)}}), flush=True)

    def _state(self) -> torch.Tensor:
        d = self._robot.data
        return amp_features(d.joint_pos[:, self._motor_ids] - self._default, d.joint_vel[:, self._motor_ids],
                            d.root_pos_w, d.root_quat_w, d.root_lin_vel_w, d.root_ang_vel_w,
                            d.body_link_pos_w[:, self._feet], d.body_link_pos_w[:, self._hands])

    def _store(self, x: torch.Tensor) -> None:
        n, cap = x.shape[0], self._buf.shape[0]
        idx = (torch.arange(n, device=x.device) + self._buf_i) % cap
        self._buf[idx] = x
        self._buf_i = (self._buf_i + n) % cap
        self._buf_n = min(self._buf_n + n, cap)

    def _update(self, num_updates: int, batch: int, gp: float) -> dict:
        # rsl-rl steps the env under torch.inference_mode(); the discriminator needs autograd
        with torch.inference_mode(False), torch.enable_grad():
            return self._train(num_updates, batch, gp)

    def _train(self, num_updates: int, batch: int, gp: float) -> dict:
        stats = []
        for _ in range(num_updates):
            pol = self._buf[torch.randint(0, self._buf_n, (batch,), device=self._buf.device)]
            ref = self._ref[torch.randint(0, self._ref.shape[0], (batch,), device=self._ref.device)].clone().requires_grad_(True)
            d_ref, d_pol = self._disc(ref), self._disc(pol)
            loss = torch.mean((d_ref - 1.0) ** 2) + torch.mean((d_pol + 1.0) ** 2)
            grad = torch.autograd.grad(d_ref.sum(), ref, create_graph=True)[0]
            loss = loss + 0.5 * gp * torch.mean(torch.sum(grad**2, dim=1)) + 0.01 * torch.sum(self._disc.head.weight**2)
            self._opt.zero_grad(set_to_none=True)
            loss.backward()
            self._opt.step()
            stats.append((float(loss), float(d_ref.mean()), float(d_pol.mean())))
        self._updates += 1
        if self._ckpt is not None:
            self._ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"disc": self._disc.state_dict(), "opt": self._opt.state_dict(), "updates": self._updates}, self._ckpt)
        s = np.mean(np.array(stats), axis=0)
        return {"loss": round(float(s[0]), 4), "d_ref": round(float(s[1]), 3), "d_robot": round(float(s[2]), 3)}

    def __call__(self, env, library: str = DEFAULT_LIBRARY, ckpt_path: str | None = None, update_every: int = 24,
                 num_updates: int = 4, batch: int = 4096, lr: float = 1e-4, grad_penalty: float = 5.0,
                 buffer_size: int = 500_000) -> torch.Tensor:
        s1 = self._state()
        reward = torch.zeros(env.num_envs, device=env.device)
        if self._prev is not None:
            valid = env.episode_length_buf > 1  # the stored state of a just-reset env is the pre-reset state
            trans = torch.cat([self._prev, s1], dim=1)
            with torch.no_grad():
                d = self._disc(trans)
            reward = torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0) * valid.float()
            if bool(valid.any()):
                self._store(trans[valid].detach())
        self._prev = s1.detach()
        self._calls += 1
        if self._calls % int(update_every) == 0 and self._buf_n >= batch:
            st = self._update(int(num_updates), int(batch), float(grad_penalty))
            if self._updates % 10 == 1:
                print(json.dumps({"amp_update": self._updates, **st, "style_reward_mean": round(float(reward.mean()), 3)}),
                      flush=True)
        return reward
