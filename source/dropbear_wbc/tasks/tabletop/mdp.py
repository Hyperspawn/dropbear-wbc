"""MDP terms of the tabletop task (Isaac Lab 2.2; import after the app has started).

Per-episode placements come from a :class:`PlacementFeed` attached to the env as ``env.tabletop_feed`` by the runner
(``scripts/tabletop_collect.py``) BEFORE the first ``env.reset()``; without it every reset uses the layout's default
placement. The success counter lives on the env (``env.tabletop_hold``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from pathlib import Path

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from .layout import LAYOUT, Placement, TabletopLayout

DEFAULT_STAND_NPZ = Path(__file__).resolve().parents[4] / "data" / "motions" / "smoke" / "dropbear_static_stand.npz"

DEFAULT_PLACEMENT = Placement(pid="default", split="none", side="left", block_xy=(0.29, 0.34), block_yaw=0.0,
                              zone_xy=(0.27, 0.24))


@dataclass
class PlacementFeed:
    """Queue of placements handed out on resets (env i gets the next one when it resets)."""

    placements: list
    cursor: int = 0
    current: dict = field(default_factory=dict)   # env_id -> Placement
    episode_index: dict = field(default_factory=dict)  # env_id -> running attempt counter
    loop: bool = False

    def next(self) -> Placement | None:
        if self.cursor >= len(self.placements):
            if not self.loop or not self.placements:
                return None
            self.cursor = 0
        p = self.placements[self.cursor]
        self.cursor += 1
        return p

    @property
    def exhausted(self) -> bool:
        return self.cursor >= len(self.placements) and not self.loop


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(0.5 * yaw)
    q[:, 3] = torch.sin(0.5 * yaw)
    return q


class reset_tabletop(ManagerTermBase):
    """Reset event: robot to a closure-consistent standing joint state; block and zone from the placement feed.

    The robot joint state is the FULL settled joint row (22 motors + 63 passive DOFs + 6 neck screws) of frame 0 of a
    contract standing NPZ (default ``data/motions/smoke/dropbear_static_stand.npz``), validated like the velocity
    task's reset (names/order vs the live articulation, USD SHA, ankle variant, validator verdict; fail closed).
    Writing ``default_joint_pos`` instead would put every passive four-bar DOF at its CAD value (0) while the motors
    sit at the standing pose, i.e. torn loop closures that PhysX snaps shut after the reset (CONTRACTS section 1:
    "a reset writes the full, physically settled joint state"). The root is fixed (``fix_root_link``), so only the
    joint row is used; the NPZ root pose is ignored. Position targets = that row (only the motors and neck screws have
    stiffness), effort targets 0.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES
        from dropbear_wbc.tasks.tracking.motion_npz import (
            expected_usd_sha256,
            load_motion_npz,
            load_validation_verdict,
            validate_against_articulation,
            validate_provenance,
        )

        robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        robot: Articulation = env.scene[robot_cfg.name]
        path = cfg.params.get("npz_path") or str(DEFAULT_STAND_NPZ)
        arrays = load_motion_npz(path)
        validate_against_articulation(arrays, robot.joint_names, robot.body_names, list(MOTOR_NAMES))
        spawn = getattr(robot.cfg, "spawn", None)
        spherical = tuple(getattr(spawn, "spherical_joint_overrides", ()) or ())
        verdict = load_validation_verdict(path)
        validate_provenance(arrays, expected_usd_sha256=expected_usd_sha256(), allow_rejected=False,
                            expected_authored_ankle=len(spherical) == 0, validation=verdict)
        self.joint_pos = torch.as_tensor(arrays.joint_pos[0], dtype=torch.float32, device=env.device)
        motor_ids, _ = robot.find_joints(list(MOTOR_NAMES), preserve_order=True)
        default_motor = robot.data.default_joint_pos[0, motor_ids]
        env.tabletop_reset_info = {
            "npz": str(path), "frame": 0, "closure_residual_max_m": float(arrays.closure_residual_m.max()),
            "validation": (verdict or {}).get("verdict"), "validation_stale": (verdict or {}).get("stale"),
            "npz_vs_default_motor_max_abs_rad": float((self.joint_pos[motor_ids] - default_motor).abs().max()),
        }

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, npz_path: str | None = None,
                 layout: TabletopLayout = LAYOUT, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
                 zone_cfg: SceneEntityCfg = SceneEntityCfg("zone")) -> None:
        robot: Articulation = env.scene[robot_cfg.name]
        ids = env_ids if env_ids is not None else torch.arange(env.num_envs, device=env.device)
        jp = self.joint_pos.unsqueeze(0).repeat(len(ids), 1)
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
        robot.set_joint_position_target(jp, env_ids=ids)
        robot.set_joint_velocity_target(torch.zeros_like(jp), env_ids=ids)
        robot.set_joint_effort_target(torch.zeros_like(jp), env_ids=ids)
        place_objects(env, ids, layout, block_cfg, zone_cfg)


def _zone_buffer(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs, 3) target-zone pose in the ROOT frame: x, y, yaw. The zone is a visual-only prim (no physics), so this
    buffer is the zone's source of truth for the success rule and the observations."""
    buf = getattr(env, "tabletop_zone", None)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.zeros(env.num_envs, 3, device=env.device)
        buf[:, 0], buf[:, 1] = DEFAULT_PLACEMENT.zone_xy[0], DEFAULT_PLACEMENT.zone_xy[1]
        env.tabletop_zone = buf
    return buf


def _author_zone_prims(env: ManagerBasedRLEnv, ids: list[int], layout: TabletopLayout = LAYOUT) -> None:
    """Move the zone prims of ``ids`` to their buffered poses by authoring a USD transform op (local to the env prim,
    whose own transform is the env origin). Writing the pose of a kinematic, collision-free rigid body through PhysX
    did NOT move its render (logs/tabletop/record_train_v1_0: the green square stayed at the spawn pose in every env
    while the physics buffer had the placement); USD xform ops are what the renderer reads (demo_render moves its
    cameras the same way)."""
    from pxr import Gf, UsdGeom

    ops = getattr(env, "_tabletop_zone_ops", None)
    if ops is None:
        ops = {}
        env._tabletop_zone_ops = ops
    buf = _zone_buffer(env).cpu().numpy()
    root = [float(v) for v in layout.root_pos_w]
    z = root[2] + layout.table_top_z + 0.5 * layout.zone_thickness + 0.0005
    for i in ids:
        op = ops.get(i)
        if op is None:
            prim = env.scene.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[i]}/Zone")
            if not prim.IsValid():
                raise RuntimeError(f"zone prim {env.scene.env_prim_paths[i]}/Zone not found")
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            op = xf.AddTransformOp(opSuffix="tabletop")
            ops[i] = op
        x, y, yaw = (float(v) for v in buf[i])
        c, s_ = float(np.cos(yaw)), float(np.sin(yaw))
        # row-vector convention (USD): rows are the local axes, last row the translation
        op.Set(Gf.Matrix4d(c, s_, 0.0, 0.0, -s_, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, root[0] + x, root[1] + y, z, 1.0))


def place_objects(env: ManagerBasedRLEnv, ids: torch.Tensor, layout: TabletopLayout = LAYOUT,
                  block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
                  zone_cfg: SceneEntityCfg = SceneEntityCfg("zone")) -> None:
    """Block and zone poses from the placement feed (the layout's default placement without a feed)."""
    block: RigidObject = env.scene[block_cfg.name]
    feed: PlacementFeed | None = getattr(env, "tabletop_feed", None)
    n = len(ids)
    bxy = torch.zeros(n, 2, device=env.device)
    zxy = torch.zeros(n, 2, device=env.device)
    byaw = torch.zeros(n, device=env.device)
    zyaw = torch.zeros(n, device=env.device)
    for k, i in enumerate(ids.tolist()):
        p = feed.next() if feed is not None else None
        if p is None:
            p = DEFAULT_PLACEMENT
        if feed is not None:
            feed.current[i] = p
            feed.episode_index[i] = feed.episode_index.get(i, -1) + 1
        bxy[k] = torch.tensor(p.block_xy[:2], device=env.device)
        zxy[k] = torch.tensor(p.zone_xy[:2], device=env.device)
        byaw[k] = float(p.block_yaw)
        zyaw[k] = float(p.zone_yaw)
    root = torch.tensor(layout.root_pos_w, device=env.device)
    origins = env.scene.env_origins[ids]
    bpos = torch.zeros(n, 3, device=env.device)
    bpos[:, :2] = bxy + root[:2]
    bpos[:, 2] = root[2] + layout.block_rest_z + 0.001
    bpose = torch.cat([bpos + origins, _yaw_quat(byaw)], dim=-1)
    block.write_root_pose_to_sim(bpose, env_ids=ids)
    block.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=ids)
    zbuf = _zone_buffer(env)
    zbuf[ids, 0:2] = zxy
    zbuf[ids, 2] = zyaw
    _author_zone_prims(env, ids.tolist(), layout)
    hold = getattr(env, "tabletop_hold", None)
    if hold is None or hold.shape[0] != env.num_envs:
        env.tabletop_hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.tabletop_hold[ids] = 0


def _block_state_root(env: ManagerBasedRLEnv, layout: TabletopLayout):
    block: RigidObject = env.scene["block"]
    root = torch.tensor(layout.root_pos_w, device=env.device)
    pos = block.data.root_pos_w - env.scene.env_origins - root
    return pos, block.data.root_quat_w, block.data.root_lin_vel_w, block.data.root_ang_vel_w


def success_now(env: ManagerBasedRLEnv, layout: TabletopLayout = LAYOUT) -> torch.Tensor:
    """Instantaneous success condition (no hold): centre in the zone (margin), on the table, upright, at rest."""
    pos, quat, lv, av = _block_state_root(env, layout)
    zbuf = _zone_buffer(env)
    zyaw = zbuf[:, 2]
    d = pos[:, :2] - zbuf[:, :2]
    c, s = torch.cos(zyaw), torch.sin(zyaw)
    lx, ly = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    h = 0.5 * layout.zone_size - layout.success_margin
    inside = (lx.abs() <= h) & (ly.abs() <= h)
    on_table = (pos[:, 2] - layout.block_rest_z).abs() <= layout.success_z_tol
    # block z axis vs world z: upright up to the cube's symmetry -> use the largest |component| of any body axis on z
    w, x, y, z = quat.unbind(-1)
    zz = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1).abs().amax(-1)
    upright = zz >= math.cos(math.radians(layout.success_max_tilt_deg))
    still = (lv.norm(dim=-1) <= layout.success_max_lin_vel) & (av.norm(dim=-1) <= layout.success_max_ang_vel)
    return inside & on_table & upright & still


def block_in_zone_stable(env: ManagerBasedRLEnv, layout: TabletopLayout = LAYOUT) -> torch.Tensor:
    """Success termination: :func:`success_now` held for ``success_hold_s``."""
    hold = getattr(env, "tabletop_hold", None)
    if hold is None or hold.shape[0] != env.num_envs:
        env.tabletop_hold = hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    ok = success_now(env, layout)
    env.tabletop_hold = torch.where(ok, hold + 1, torch.zeros_like(hold))
    need = max(1, int(round(layout.success_hold_s / env.step_dt)))
    return env.tabletop_hold >= need


def block_dropped(env: ManagerBasedRLEnv, layout: TabletopLayout = LAYOUT) -> torch.Tensor:
    pos, *_ = _block_state_root(env, layout)
    off = ~((pos[:, 0] >= layout.table_front_x - 0.02) & (pos[:, 0] <= layout.table_front_x + layout.table_depth + 0.02)
            & ((pos[:, 1] - layout.table_center_y).abs() <= 0.5 * layout.table_width + 0.02))
    return (pos[:, 2] < layout.table_top_z - layout.drop_margin) | off


def success_reward(env: ManagerBasedRLEnv, layout: TabletopLayout = LAYOUT) -> torch.Tensor:
    return success_now(env, layout).float()


def block_pose_root(env: ManagerBasedRLEnv, layout: TabletopLayout = LAYOUT) -> torch.Tensor:
    pos, quat, *_ = _block_state_root(env, layout)
    return torch.cat([pos, quat], dim=-1)


def zone_pos_root(env: ManagerBasedRLEnv, layout: TabletopLayout = LAYOUT) -> torch.Tensor:
    """(num_envs, 3) zone centre in the root frame (top surface centre height)."""
    zbuf = _zone_buffer(env)
    z = torch.full_like(zbuf[:, :1], layout.table_top_z + 0.5 * layout.zone_thickness + 0.0005)
    return torch.cat([zbuf[:, :2], z], dim=-1)
