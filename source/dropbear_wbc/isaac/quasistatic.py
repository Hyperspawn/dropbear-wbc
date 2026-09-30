"""Quasi-static (gravity-free, root-FIXED) Dropbear articulation for kinematic measurement.

Used by ``tools/calibrate_semantics.py`` (motor sweeps) and ``tools/settle_motion.py`` (closure-consistent
full joint states for motion frames). Must run inside an Isaac Lab 2.2 app (``AppLauncher`` already
started) -- import this module only after the app is up.

Physics setup (one robot per env, no ground, collisions disabled entirely):

* the contract 0.1 spawn-time plant fixes are applied in memory (``dropbear_wbc.robots.spawn``): the 3
  joint-less orphan bodies are deactivated, joint friction is 0, the left-knee closure ``LL_Revolute121``
  axis is X -> Z, zero principal inertias are raised to 2e-6 kg*m^2; and (contract 0.2, default) the ankle
  tie-rod closures ``*_Revolute111/112`` are retyped revolute -> spherical (``authored_ankle_tierods`` /
  ``$DROPBEAR_AUTHORED_ANKLE=1`` keeps them revolute). The USD file is never modified.
* gravity off; the root link ``world`` is FIXED to the env origin (``fix_root_link``) -- no drift.
* the 22 body motors: stiff implicit PD to the commanded targets (``motor_kp``/``motor_kd``);
* neck lead screws: PD held at 0 (their authored init value);
* every other joint (passive four-bar / tie-rod / U-joint DOFs): stiffness 0, damping ``passive_damping``;
* the 27 excluded loop closures are solved by PhysX as maximal-coordinate joints.

Motion is commanded in fixed-length stages (:meth:`QuasiStaticDropbear.move`): the motor targets are
ramped linearly from the previous to the new value over ``ramp_steps`` physics steps, then held for
``hold_steps``. Convergence is judged by POSITIONS, not velocities: the passive loop joints keep a
~0.02 rad/s velocity noise floor from the closure solver even when nothing moves
(``logs/calibrate_settle/probe_quasistatic.json``). Per stage we report

* ``dq_window``  -- max |joint position change| over the last ``window`` steps [rad or m];
* ``gap``        -- worst loop-closure anchor gap [m];
* ``motor_err``  -- max |motor position - target| [rad].

All poses returned by :meth:`read_state` are **relative to the root link frame** (``world`` body).

Units: SI (m, rad, s). Quaternions wxyz. Poses are link (actor) frames, not COM frames.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from dropbear_wbc.robots.dropbear_names import (
    JOINT_AXIS_FIXES,
    MIN_PRINCIPAL_INERTIA,
    MOTOR_NAMES,
    NECK_NAMES,
    ORPHAN_BODIES,
    PASSIVE_ARM_REGEX,
    PASSIVE_HEAD_REGEX,
    PASSIVE_LEG_REGEX,
    authored_ankle_requested,
    resolve_usd_path,
    spherical_joint_fixes,
)


@dataclass
class QuasiStaticCfg:
    """Settings of the quasi-static articulation (see module docstring)."""

    usd_path: str = field(default_factory=resolve_usd_path)
    num_envs: int = 64
    dt: float = 0.005
    """Physics step [s]."""
    pos_iters: int = 16
    """PhysX TGS position iterations (chosen by ``tools/probe_quasistatic.py``; contract training default 32)."""
    vel_iters: int = 4
    motor_kp: float = 3000.0
    """Motor PD stiffness [N*m/rad] (implicit drive)."""
    motor_kd: float = 60.0
    """Motor PD damping [N*m*s/rad]."""
    motor_effort: float = 1.0e4
    """Motor effort limit [N*m] (large: a jammed mechanism should show up as tracking error / closure gap,
    not as clipping)."""
    neck_kp: float = 5000.0
    neck_kd: float = 50.0
    passive_damping: float = 0.5
    """Damping of passive DOFs [N*m*s/rad]; small, only to kill free oscillation of light links."""
    armature: float = 0.01
    max_angular_velocity_deg_s: float = math.degrees(50.0)
    """Per-link angular velocity cap (Isaac Lab unit: deg/s). Attempt 1 used 100 deg/s, which throttles
    the knee/elbow four-bar outputs during ramps."""
    env_spacing: float = 4.0
    device: str = "cuda:0"
    fix_root: bool = True
    """Fix the root ``world`` link to the env origin (``fix_root_link``)."""
    contract_fixes: bool = True
    """Apply the contract 0.1 spawn-time fixes (orphans, friction 0, LL_Revolute121 axis Z, inertia)."""
    extra_axis_overrides: dict[str, str] = field(default_factory=dict)
    """DIAGNOSTIC ONLY: additional ``{joint: 'X'|'Y'|'Z'}`` axis overrides (in memory)."""
    diag_spherical_joints: tuple[str, ...] = ()
    """DIAGNOSTIC ONLY: joints whose prim type is changed to ``PhysicsSphericalJoint`` in memory before the
    simulation starts (e.g. the ankle crank/tie-rod closures ``*_Revolute111``/``*_Revolute112``, to test the
    rod-end hypothesis). Empty = contract plant."""
    authored_ankle_tierods: bool | None = None
    """``True`` keeps the authored revolute ankle tie-rod closures (CONTRACTS 0.2 opt-out); ``False`` applies the
    default spherical retype (only together with ``contract_fixes``); ``None`` is resolved from
    ``$DROPBEAR_AUTHORED_ANKLE`` at construction, so ``to_dict()`` always records the plant that was used."""

    def __post_init__(self) -> None:
        self.authored_ankle_tierods = authored_ankle_requested(self.authored_ankle_tierods)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClosureSpec:
    """One excluded (loop-closing) joint read from the live stage."""

    name: str
    joint_type: str
    body0: int
    body1: int
    local_pos0: tuple[float, float, float]
    local_pos1: tuple[float, float, float]
    local_rot0: tuple[float, float, float, float]
    """Joint frame in body0 (wxyz)."""
    local_rot1: tuple[float, float, float, float]
    axis: str | None
    """Free rotation axis in the joint frame for revolute closures ('X'/'Y'/'Z'), else None."""


def _wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi


class QuasiStaticDropbear:
    """N gravity-free, root-fixed Dropbear articulations driven by motor position targets."""

    def __init__(self, cfg: QuasiStaticCfg):
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass

        from dropbear_wbc.robots.spawn import DropbearUsdFileCfg

        self.cfg = cfg
        sim_cfg = sim_utils.SimulationCfg(
            dt=cfg.dt, device=cfg.device, gravity=(0.0, 0.0, 0.0),
            physx=sim_utils.PhysxCfg(solver_type=1, enable_stabilization=False),
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)

        axis_fix = dict(JOINT_AXIS_FIXES) if cfg.contract_fixes else {}
        axis_fix.update(cfg.extra_axis_overrides)
        spawn = DropbearUsdFileCfg(
            usd_path=cfg.usd_path,
            deactivate_prims=tuple(ORPHAN_BODIES) if cfg.contract_fixes else (),
            joint_friction_override=0.0 if cfg.contract_fixes else None,
            joint_axis_overrides=axis_fix,
            min_principal_inertia=MIN_PRINCIPAL_INERTIA if cfg.contract_fixes else None,
            spherical_joint_overrides=spherical_joint_fixes(cfg.authored_ankle_tierods) if cfg.contract_fixes else (),
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=50.0, max_angular_velocity=cfg.max_angular_velocity_deg_s,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=cfg.pos_iters,
                solver_velocity_iteration_count=cfg.vel_iters,
                fix_root_link=cfg.fix_root,
            ),
        )
        robot_cfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
                                                       joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
            actuators={
                "motors": ImplicitActuatorCfg(
                    joint_names_expr=list(MOTOR_NAMES), effort_limit_sim=cfg.motor_effort,
                    stiffness=cfg.motor_kp, damping=cfg.motor_kd, armature=cfg.armature,
                ),
                "neck": ImplicitActuatorCfg(
                    joint_names_expr=list(NECK_NAMES), effort_limit_sim=1.0e4,
                    stiffness=cfg.neck_kp, damping=cfg.neck_kd, armature=0.001,
                ),
                "passive": ImplicitActuatorCfg(
                    joint_names_expr=[PASSIVE_LEG_REGEX, PASSIVE_ARM_REGEX, PASSIVE_HEAD_REGEX],
                    effort_limit_sim=1.0e4, stiffness=0.0, damping=cfg.passive_damping, armature=cfg.armature,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

        @configclass
        class _SceneCfg(InteractiveSceneCfg):
            robot: ArticulationCfg = robot_cfg

        self.scene = InteractiveScene(_SceneCfg(num_envs=cfg.num_envs, env_spacing=cfg.env_spacing))
        self.diag_patched = self._make_spherical(cfg.diag_spherical_joints)
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.device = self.robot.device
        self.num_envs = cfg.num_envs
        self.joint_names: list[str] = list(self.robot.joint_names)
        self.body_names: list[str] = list(self.robot.body_names)
        ids, names = self.robot.find_joints(list(MOTOR_NAMES), preserve_order=True)
        if tuple(names) != tuple(MOTOR_NAMES):
            raise RuntimeError(f"motor resolution mismatch: {names}")
        self.motor_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self.motor_ids_list = list(ids)
        self.env_origins = self.scene.env_origins.clone()
        lim = self.robot.data.joint_pos_limits[0].cpu().numpy()
        self.joint_limits = lim  # (J, 2) rad or m
        self.motor_limits = lim[self.motor_ids_list]
        self.revolute_mask = torch.tensor([not n.startswith("head_LeadScrew") for n in self.joint_names],
                                          device=self.device)
        self.closures = self._read_closures()
        self._cl_i0 = torch.tensor([c.body0 for c in self.closures], device=self.device)
        self._cl_i1 = torch.tensor([c.body1 for c in self.closures], device=self.device)
        self._cl_l0 = torch.tensor([c.local_pos0 for c in self.closures], dtype=torch.float32, device=self.device)
        self._cl_l1 = torch.tensor([c.local_pos1 for c in self.closures], dtype=torch.float32, device=self.device)
        self.com_pos_b = self.robot.data.body_com_pos_b[0].cpu().numpy().astype(np.float64)
        self.body_masses = self.robot.data.default_mass[0].cpu().numpy().astype(np.float64)
        self.stage_fixes = self._check_stage_fixes()
        self.physics_steps = 0

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _make_spherical(names: tuple[str, ...]) -> list[str]:
        """DIAGNOSTIC: retype the named joints of every env to spherical (in-memory stage only)."""
        if not names:
            return []
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        done = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs")):
            if prim.GetName() in names and prim.IsA(UsdPhysics.Joint):
                if prim.IsInstanceProxy():
                    raise RuntimeError(f"cannot retype instance proxy {prim.GetPath()}")
                prim.SetTypeName("PhysicsSphericalJoint")
                done.append(str(prim.GetPath()))
        missing = set(names) - {p.rsplit("/", 1)[-1] for p in done}
        if missing:
            raise RuntimeError(f"spherical diagnostic targets not found: {sorted(missing)}")
        print(f"[quasistatic] DIAGNOSTIC: {len(done)} joint prims retyped to PhysicsSphericalJoint: "
              f"{sorted(set(p.rsplit('/', 1)[-1] for p in done))}", flush=True)
        return done

    def _check_stage_fixes(self) -> dict:
        """Read back the spawn-time fixes from env_0's live stage (evidence for the logs)."""
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        out: dict = {"orphans_active": {}, "axes": {}, "friction_nonzero": []}
        for name in ORPHAN_BODIES:
            p = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
            out["orphans_active"][name] = bool(p.IsValid() and p.IsActive())
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            if prim.GetName() in ("LL_Revolute121", "RL_Revolute121"):
                out["axes"][prim.GetName()] = str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
            if prim.GetName() in ("LL_Revolute111", "LL_Revolute112", "RL_Revolute111", "RL_Revolute112"):
                out.setdefault("ankle_tierod_types", {})[prim.GetName()] = prim.GetTypeName()
            a = prim.GetAttribute("physxJoint:jointFriction")
            if a and a.HasAuthoredValue() and float(a.Get()) != 0.0:
                out["friction_nonzero"].append(prim.GetName())
        out["fix_root"] = self.cfg.fix_root
        out["authored_ankle_tierods"] = bool(self.cfg.authored_ankle_tierods)
        out["diag_spherical_joints"] = sorted(set(p.rsplit("/", 1)[-1] for p in self.diag_patched))
        return out

    def _read_closures(self) -> list[ClosureSpec]:
        """Excluded joints of env_0's robot, mapped to articulation body indices."""
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        index = {n: i for i, n in enumerate(self.body_names)}
        out: list[ClosureSpec] = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(prim)
            if not j.GetExcludeFromArticulationAttr().Get() or j.GetJointEnabledAttr().Get() is False:
                continue
            t0, t1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if len(t0) != 1 or len(t1) != 1 or t0[0].name not in index or t1[0].name not in index:
                continue
            r0, r1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()
            axis = None
            if prim.IsA(UsdPhysics.RevoluteJoint):
                axis = str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
            out.append(ClosureSpec(
                name=prim.GetName(), joint_type=prim.GetTypeName(),
                body0=index[t0[0].name], body1=index[t1[0].name],
                local_pos0=tuple(float(v) for v in j.GetLocalPos0Attr().Get()),
                local_pos1=tuple(float(v) for v in j.GetLocalPos1Attr().Get()),
                local_rot0=(float(r0.GetReal()), *map(float, r0.GetImaginary())),
                local_rot1=(float(r1.GetReal()), *map(float, r1.GetImaginary())),
                axis=axis,
            ))
        if not out:
            raise RuntimeError("no loop closures found on the live stage")
        return out

    # ------------------------------------------------------------------------------------------------
    def write_joint_state(self, joint_pos: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Write a full joint state (all J joints, Isaac order) with zero velocity (also resets targets
        of the motors to the written motor positions)."""
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        if env_ids is None:
            self.set_motor_targets(joint_pos[:, self.motor_ids])

    def set_motor_targets(self, targets: torch.Tensor) -> None:
        """Motor position targets [rad], shape (N, 22) in motor-contract order."""
        self.robot.set_joint_position_target(targets, joint_ids=self.motor_ids_list)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(self.cfg.dt)
        self.physics_steps += n

    def gaps(self) -> torch.Tensor:
        """Closure anchor gaps [m], shape (N, C)."""
        from isaaclab.utils.math import quat_apply

        bp, bq = self.robot.data.body_link_pos_w, self.robot.data.body_link_quat_w
        n, c = bp.shape[0], self._cl_i0.numel()
        a0 = bp[:, self._cl_i0] + quat_apply(bq[:, self._cl_i0].reshape(-1, 4),
                                             self._cl_l0.expand(n, c, 3).reshape(-1, 3)).reshape(n, c, 3)
        a1 = bp[:, self._cl_i1] + quat_apply(bq[:, self._cl_i1].reshape(-1, 4),
                                             self._cl_l1.expand(n, c, 3).reshape(-1, 3)).reshape(n, c, 3)
        return torch.linalg.vector_norm(a0 - a1, dim=-1)

    def worst_gap(self) -> torch.Tensor:
        """Worst closure anchor gap per env [m], shape (N,)."""
        return self.gaps().max(dim=-1).values

    def joint_delta(self, q_ref: torch.Tensor) -> torch.Tensor:
        """max |q - q_ref| per env (revolute differences wrapped to [-pi, pi)), shape (N,)."""
        d = self.robot.data.joint_pos - q_ref
        d = torch.where(self.revolute_mask, _wrap(d), d)
        return d.abs().max(dim=-1).values

    def move(self, start: torch.Tensor, end: torch.Tensor, ramp_steps: int, hold_steps: int,
             window: int = 4) -> dict:
        """Ramp motor targets linearly ``start`` -> ``end`` (N, 22) over ``ramp_steps``, hold ``hold_steps``.

        Returns torch tensors (N,): ``dq_window`` (max joint position change over the last ``window``
        steps), ``gap`` (worst closure gap [m]), ``motor_err`` (max |motor pos - end| [rad]) and the int
        ``steps``.
        """
        total = ramp_steps + hold_steps
        snap_at = max(0, total - window)
        q_ref = self.robot.data.joint_pos.clone() if snap_at == 0 else None
        k = 0
        for i in range(1, ramp_steps + 1):
            self.set_motor_targets(start + (end - start) * (i / ramp_steps))
            self.step(1)
            k += 1
            if k == snap_at:
                q_ref = self.robot.data.joint_pos.clone()
        self.set_motor_targets(end)
        for _ in range(hold_steps):
            self.step(1)
            k += 1
            if k == snap_at:
                q_ref = self.robot.data.joint_pos.clone()
        motor_err = (self.robot.data.joint_pos[:, self.motor_ids] - end).abs().max(dim=-1).values
        return {"dq_window": self.joint_delta(q_ref), "gap": self.worst_gap(), "motor_err": motor_err,
                "steps": total}

    def settle(self, targets: torch.Tensor, hold_min: int = 8, hold_max: int = 200, window: int = 4,
               pos_tol: float = 2e-4) -> dict:
        """Hold ``targets`` (N, 22) until every env's joints moved < ``pos_tol`` over ``window`` steps.

        Position-based criterion (the velocity noise floor makes |qd| useless). Returns the same keys as
        :meth:`move` plus ``converged`` (N,) bool.
        """
        self.set_motor_targets(targets)
        info = self.move(targets, targets, 0, hold_min, window)
        steps = hold_min
        while steps < hold_max and bool((info["dq_window"] >= pos_tol).any()):
            info = self.move(targets, targets, 0, window, window)
            steps += window
        info["steps"] = steps
        info["converged"] = info["dq_window"] < pos_tol
        return info

    def read_state(self) -> dict[str, np.ndarray]:
        """Current state of every env as float64 numpy arrays, poses relative to the root link frame.

        Keys: ``joint_pos`` (N,J), ``joint_vel`` (N,J), ``body_pos`` (N,B,3), ``body_quat`` (N,B,4, wxyz),
        ``motor_pos`` (N,22), ``motor_target`` (N,22), ``root_pos_w`` (N,3, minus env origin),
        ``root_quat_w`` (N,4), ``gaps`` (N,C).
        """
        from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

        d = self.robot.data
        root_p = d.root_link_pos_w
        root_q = d.root_link_quat_w
        bp, bq = d.body_link_pos_w, d.body_link_quat_w
        n, b = bp.shape[:2]
        rq = root_q[:, None, :].expand(n, b, 4).reshape(-1, 4)
        rel_p = quat_apply_inverse(rq, (bp - root_p[:, None, :]).reshape(-1, 3)).reshape(n, b, 3)
        rel_q = quat_mul(quat_conjugate(rq), bq.reshape(-1, 4)).reshape(n, b, 4)
        rel_q = rel_q * torch.where(rel_q[..., :1] < 0, -1.0, 1.0)
        out = {
            "joint_pos": d.joint_pos, "joint_vel": d.joint_vel, "body_pos": rel_p, "body_quat": rel_q,
            "motor_pos": d.joint_pos[:, self.motor_ids], "motor_target": d.joint_pos_target[:, self.motor_ids],
            "root_pos_w": root_p - self.env_origins, "root_quat_w": root_q, "gaps": self.gaps(),
        }
        return {k: v.detach().cpu().double().numpy() for k, v in out.items()}

    def joint_frame_table(self) -> dict[str, np.ndarray]:
        """Every USD joint of env_0's robot: bodies (articulation indices, -1 = none), local frames, axis.

        Used offline to place joint anchors/axes on bodies (e.g. hip centre = hip-pitch anchor)."""
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        index = {n: i for i, n in enumerate(self.body_names)}
        rows = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(prim)
            t0, t1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            b0 = index.get(t0[0].name, -1) if len(t0) == 1 else -1
            b1 = index.get(t1[0].name, -1) if len(t1) == 1 else -1
            lp0, lp1 = j.GetLocalPos0Attr().Get(), j.GetLocalPos1Attr().Get()
            r0, r1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()
            axis_attr = prim.GetAttribute("physics:axis")
            axis = str(axis_attr.Get()) if axis_attr and axis_attr.IsValid() and axis_attr.Get() is not None else ""
            rows.append((prim.GetName(), prim.GetTypeName(), b0, b1,
                         tuple(float(v) for v in (lp0 if lp0 is not None else (0, 0, 0))),
                         tuple(float(v) for v in (lp1 if lp1 is not None else (0, 0, 0))),
                         (float(r0.GetReal()), *map(float, r0.GetImaginary())) if r0 is not None else (1.0, 0, 0, 0),
                         (float(r1.GetReal()), *map(float, r1.GetImaginary())) if r1 is not None else (1.0, 0, 0, 0),
                         axis, bool(j.GetExcludeFromArticulationAttr().Get())))
        return {
            "jf_names": np.array([r[0] for r in rows]), "jf_types": np.array([r[1] for r in rows]),
            "jf_body0": np.array([r[2] for r in rows], dtype=np.int64),
            "jf_body1": np.array([r[3] for r in rows], dtype=np.int64),
            "jf_local_pos0": np.array([r[4] for r in rows]), "jf_local_pos1": np.array([r[5] for r in rows]),
            "jf_local_rot0": np.array([r[6] for r in rows]), "jf_local_rot1": np.array([r[7] for r in rows]),
            "jf_axis": np.array([r[8] for r in rows]), "jf_excluded": np.array([r[9] for r in rows]),
        }

    def closure_table(self) -> dict[str, np.ndarray]:
        """Closure specs as arrays (for saving next to raw data and for numpy residual evaluation)."""
        return {
            "closure_names": np.array([c.name for c in self.closures]),
            "closure_types": np.array([c.joint_type for c in self.closures]),
            "closure_axes": np.array([c.axis or "" for c in self.closures]),
            "closure_body0": np.array([c.body0 for c in self.closures], dtype=np.int64),
            "closure_body1": np.array([c.body1 for c in self.closures], dtype=np.int64),
            "closure_local_pos0": np.array([c.local_pos0 for c in self.closures], dtype=np.float64),
            "closure_local_pos1": np.array([c.local_pos1 for c in self.closures], dtype=np.float64),
            "closure_local_rot0": np.array([c.local_rot0 for c in self.closures], dtype=np.float64),
            "closure_local_rot1": np.array([c.local_rot1 for c in self.closures], dtype=np.float64),
        }
