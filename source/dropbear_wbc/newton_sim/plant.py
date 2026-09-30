"""Dropbear in Newton 1.6 (MuJoCo solver) with Unitree-style per-motor impedance control.

The user's USD (CONTRACTS section 0) is imported through Codex's in-memory
adapter ``<DROPBEAR_CONTROL_DIR>/tools/newton_dropbear_asset.py`` (github.com/robit-man/dropbear_control)
(``prepare_stage``: expands instances, gives collision meshes convex-hull
approximations and orients the tree; path-imported read-only). The 27 loop
closures stay as Newton loop joints, which SolverMuJoCo turns into equality
constraints.

Actuation model (``dropbear_hg-v1`` motor law, CONTRACTS section 6):

* 22 body motors: ``JointTargetMode.EFFORT`` (no MuJoCo actuator). Each physics
  substep a warp kernel computes ``tau = tau_ff + kp*(q* - q) + kd*(dq* - dq)``,
  clips it to the per-motor effort limit and writes ``Control.joint_f``.
  Motors whose mode is 0 (disabled) apply zero torque.
* 6 neck lead screws: stiff implicit MuJoCo position actuators (legacy ``head``
  group gains, 5000 N/m, 50 N*s/m), target = initial position unless a LowCmd
  carries a neck block.
* 55 passive joints (83 DOF-carrying joints minus the above): no actuator, MuJoCo
  DOF damping ``PlantConfig.passive_damping`` (default ``motors.PASSIVE_DAMPING`` = 0, the value Isaac actually
  simulates: the legacy ``parasitic_*`` damping 50 is a drive parameter PhysX ignores on these undriven joints,
  docs/CONTRACTS.md 0.3) and armature 0.01. Before 2026-09-24 13:00 the default was 50.

Frames: world z-up, ground plane at z = 0. The root body is ``/humanoid/world``;
its frame origin sits well below the soles (about 12.5 cm in the standing pose),
so the spawn height is computed from the collision geometry, not assumed.
Newton body velocities are ``(linear COM velocity, angular velocity)`` in the
world frame; quaternions are converted from Newton's xyzw to wxyz on readout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Must be set before USD/TBB initialize (reproduced native crash in the Windows USD parser).
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import numpy as np

from .. import paths as _paths
from ..sdk import motors

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USD = _paths.usd_path()  # $DROPBEAR_USD / .dropbear.env, else assets/dropbear.usd
USD_SHA256 = "45586414b065cd982d487cbd868fe982108b3b8ccec64d3dfcf629652ed8db0f"
# github.com/robit-man/dropbear_control (clone next to this repo, or set DROPBEAR_CONTROL_DIR)
CODEX_ASSET_MODULE = _paths.dropbear_control_dir() / "tools" / "newton_dropbear_asset.py"
ROOT_BODY_LABEL = "/humanoid/world"
GRAVITY = 9.81


def configure_warp_cache() -> Path:
    """Point warp's kernel cache at the repo (C: is nearly full; CONTRACTS section 0)."""
    import warp as wp

    cache = Path(os.environ.get("WARP_CACHE_PATH", REPO_ROOT / ".warp-cache"))
    cache.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(cache)
    return cache


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_prepare_stage(path: Path = CODEX_ASSET_MODULE):
    """Path-import Codex's ``prepare_stage`` without modifying ``dropbear_control``."""
    spec = importlib.util.spec_from_file_location("codex_newton_dropbear_asset", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_stage


@dataclass
class PlantConfig:
    """Build and runtime options for :class:`DropbearNewtonPlant`.

    Attributes:
        usd_path: plant USD (default: ``$DROPBEAR_USD`` or the contract path).
        verify_sha: refuse to run if the USD SHA-256 differs from the contract.
        fixed_base: weld the root body to the world ("hanging" bench mode).
        hang_clearance: fixed base only: lowest collision point height above ground [m].
        spawn_clearance: free base only: lowest collision point height at spawn [m].
        sim_dt: physics step [s].
        substeps: physics steps per control tick (tick period = ``sim_dt * substeps``).
        device: warp device (``"cuda:0"`` or ``"cpu"``).
        mujoco_cpu: use MuJoCo C (CPU) instead of MuJoCo Warp.
        use_cuda_graph: capture each tick in a CUDA graph (GPU only).
        iterations, ls_iterations: MuJoCo solver and line-search iterations.
        njmax, nconmax: MuJoCo constraint-row and contact capacities per world.
        collisions: ``"convex-hull"`` (Codex default) or ``"box"`` (bounding boxes).
        extra_bodies: body names (last path component) whose poses are read out each tick.
        start_motor_q: optional 22-motor start pose [rad] (e.g. frame 0 of a tracking reference). Before the
            first tick the motors are ramped from the USD rest pose to it over ``presettle_s / 2`` and held for the
            rest of ``presettle_s`` with the root PINNED (free base: root coordinates re-written and root velocity
            zeroed after every tick; gravity and contacts on). The loop closures are thus solved by the simulator
            itself. The settled configuration becomes the initial state (velocities zeroed, root re-placed at
            ``spawn_clearance``); tick and time restart at 0. This replaces an RSI-style start for sim2sim.
        presettle_s: duration of that pre-settle [s].
        presettle_gain: multiplier on the legacy motor kp/kd during the pre-settle.
        initial_motor_q: optional 22 initial motor positions [rad]; ``None`` keeps the USD
            authored (closure-consistent) configuration.
        save_mjcf: optional path to write the generated MJCF (diagnostic).
        passive_damping: passive-joint DOF damping [N*m*s/rad]; default 0 = Isaac's effective value (CONTRACTS 0.3).
            MuJoCo applies any value given here, uncapped (Isaac's drive would clip at 50 N*m if it acted).
        disable_contacts: turn off all MuJoCo contacts (diagnostic; only meaningful with ``fixed_base``).
        usd_fixes: apply the CONTRACTS section 0.1 in-memory plant fixes (deactivate the three
            jointless knee bearings, zero ``physxJoint:jointFriction``, ``LL_Revolute121`` axis X->Z,
            raise zero principal inertias) exactly as the Isaac spawner does. ``False`` loads the raw
            USD (then Newton makes each jointless body its own root: welded to the world with
            ``fixed_base``, loose debris otherwise, outside the self-collision filter).
        authored_ankle_tierods: CONTRACTS 0.2 opt-out (only with ``usd_fixes``): ``True`` keeps the authored
            revolute ankle tie-rod closures; ``False`` retypes them to spherical (default plant); ``None`` reads
            ``$DROPBEAR_AUTHORED_ANKLE``.
        diag_axis_overrides: DIAGNOSTIC ONLY: ``((joint_name, "X"|"Y"|"Z"), ...)`` revolute-axis edits
            applied in the in-memory session layer (the USD file is never modified). Not plant authority.
    """

    usd_path: Path = DEFAULT_USD
    verify_sha: bool = True
    fixed_base: bool = False
    hang_clearance: float = 0.30
    spawn_clearance: float = 0.005
    sim_dt: float = 0.002
    substeps: int = 1
    device: str = "cuda:0"
    mujoco_cpu: bool = False
    use_cuda_graph: bool = True
    iterations: int = 100
    ls_iterations: int = 50
    njmax: int = 1024
    nconmax: int = 256
    collisions: str = "convex-hull"
    extra_bodies: tuple[str, ...] = ()
    initial_motor_q: tuple[float, ...] | None = None
    start_motor_q: tuple[float, ...] | None = None
    diag_eq_solref: tuple[float, float] | None = None
    """DIAGNOSTIC (MuJoCo C only): override the MuJoCo ``eq_solref`` (time constant, damping ratio) of all equality
    constraints (the loop closures). Default MuJoCo (0.02, 1) is soft; e.g. (0.004, 1) is about as stiff as dt allows."""
    presettle_s: float = 1.5
    presettle_gain: float = 4.0
    save_mjcf: Path | None = None
    passive_damping: float = motors.PASSIVE_DAMPING
    disable_contacts: bool = False
    diag_axis_overrides: tuple[tuple[str, str], ...] = ()
    usd_fixes: bool = True
    authored_ankle_tierods: bool | None = None
    motor_profile: str | None = None
    """``None``: legacy motors (plain PD clipped to ``motors.EFFORT_LIMIT``, legacy armature). An ``hw_*`` actuator
    profile (``robots.hw_motor_specs.HW_PROFILE_MAPS``, e.g. ``hw_v1``): the training motor twin, i.e. per-motor
    armature (rotor x ratio^2) and peak torque, the PD demand clipped to the DC-motor torque-speed envelope and the
    output-side Coulomb + viscous friction subtracted (``hw_motor_specs.hw_motor_torque``, nominal friction scale 1, no
    extra delay: the command link adds its own). Gains still come from each LowCmd (the export's sidecar)."""

    @property
    def tick_dt(self) -> float:
        return self.sim_dt * self.substeps


@dataclass
class PlantReadout:
    """One tick of plant state (host numpy, float64). Frames/units as in the module docstring."""

    tick: int
    time_s: float
    motor_q: np.ndarray
    motor_dq: np.ndarray
    motor_tau: np.ndarray
    neck_q: np.ndarray
    neck_dq: np.ndarray
    root_pos_w: np.ndarray
    root_quat_wxyz: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    body_pos_w: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    body_quat_wxyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))


def _hw_params(profile: str | None) -> dict | None:
    """Per-motor twin parameters of an ``hw_*`` actuator profile (``None`` -> legacy motors)."""
    if not profile:
        return None
    from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS, joint_hw_params

    if profile not in HW_PROFILE_MAPS:
        raise ValueError(f"unknown motor profile {profile!r}; known: {sorted(HW_PROFILE_MAPS)}")
    return joint_hw_params(HW_PROFILE_MAPS[profile])


def _kernels():
    """Define warp kernels lazily (after warp's cache directory is configured)."""
    import warp as wp

    @wp.kernel
    def motor_pd_kernel(
        joint_q: wp.array(dtype=float),
        joint_qd: wp.array(dtype=float),
        q_idx: wp.array(dtype=int),
        qd_idx: wp.array(dtype=int),
        q_des: wp.array(dtype=float),
        dq_des: wp.array(dtype=float),
        tau_ff: wp.array(dtype=float),
        kp: wp.array(dtype=float),
        kd: wp.array(dtype=float),
        enable: wp.array(dtype=int),
        effort_limit: wp.array(dtype=float),
        hw: int,
        saturation: wp.array(dtype=float),
        no_load_speed: wp.array(dtype=float),
        coulomb: wp.array(dtype=float),
        viscous: wp.array(dtype=float),
        joint_f: wp.array(dtype=float),
        tau_out: wp.array(dtype=float),
    ):
        i = wp.tid()
        q = joint_q[q_idx[i]]
        dq = joint_qd[qd_idx[i]]
        tau = float(0.0)
        if enable[i] != 0:
            tau = tau_ff[i] + kp[i] * (q_des[i] - q) + kd[i] * (dq_des[i] - dq)
        lim = effort_limit[i]
        if hw != 0:  # hw_motor_specs.hw_motor_torque (the DatasheetMotor law)
            v0 = no_load_speed[i]
            ve = v0 * (1.0 + lim / saturation[i])
            v = wp.clamp(dq, -ve, ve)
            top = wp.min(saturation[i] * (1.0 - v / v0), lim)
            bottom = wp.max(saturation[i] * (-1.0 - v / v0), -lim)
            tau = wp.clamp(tau, bottom, top)
            tau_out[i] = tau
            tau = tau - (coulomb[i] * wp.tanh(dq / 0.05) + viscous[i] * dq)
        else:
            tau = wp.clamp(tau, -lim, lim)
            tau_out[i] = tau
        joint_f[qd_idx[i]] = tau

    @wp.kernel
    def pack_readout_kernel(
        joint_q: wp.array(dtype=float),
        joint_qd: wp.array(dtype=float),
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
        motor_q_idx: wp.array(dtype=int),
        motor_qd_idx: wp.array(dtype=int),
        tau_applied: wp.array(dtype=float),
        neck_q_idx: wp.array(dtype=int),
        neck_qd_idx: wp.array(dtype=int),
        bodies: wp.array(dtype=int),
        out: wp.array(dtype=float),
    ):
        # Single thread: the readout is ~100 floats; one launch + one D2H copy per tick.
        nm = motor_q_idx.shape[0]
        nn = neck_q_idx.shape[0]
        o = int(0)
        for i in range(nm):
            out[o + i] = joint_q[motor_q_idx[i]]
            out[o + nm + i] = joint_qd[motor_qd_idx[i]]
            out[o + 2 * nm + i] = tau_applied[i]
        o = 3 * nm
        for i in range(nn):
            out[o + i] = joint_q[neck_q_idx[i]]
            out[o + nn + i] = joint_qd[neck_qd_idx[i]]
        o = 3 * nm + 2 * nn
        for b in range(bodies.shape[0]):
            t = body_q[bodies[b]]
            p = wp.transform_get_translation(t)
            r = wp.transform_get_rotation(t)
            v = body_qd[bodies[b]]
            base = o + b * 13
            out[base + 0] = p[0]
            out[base + 1] = p[1]
            out[base + 2] = p[2]
            out[base + 3] = r[3]  # w
            out[base + 4] = r[0]
            out[base + 5] = r[1]
            out[base + 6] = r[2]
            for k in range(6):
                out[base + 7 + k] = v[k]

    return motor_pd_kernel, pack_readout_kernel


class DropbearNewtonPlant:
    """Newton model of Dropbear with explicit per-motor PD torques.

    Typical use::

        plant = DropbearNewtonPlant(PlantConfig(fixed_base=True))
        plant.set_motor_command(q, dq, tau, kp, kd, enable)
        plant.step()                 # one control tick (``substeps`` physics steps)
        r = plant.readout()          # PlantReadout
    """

    def __init__(self, cfg: PlantConfig, log: Any = print):
        self.cfg = cfg
        self.log = log
        self.report: dict[str, Any] = {"config": {k: (str(v) if isinstance(v, Path) else v)
                                                  for k, v in cfg.__dict__.items()}}
        configure_warp_cache()
        import warp as wp
        import newton

        self.wp, self.newton = wp, newton
        wp.init()
        self.device = wp.get_device(cfg.device)
        self._pd_kernel, self._pack_kernel = _kernels()

        usd = Path(cfg.usd_path)
        t0 = time.perf_counter()
        sha = sha256_file(usd)
        self.report["usd"] = {"path": str(usd), "sha256": sha, "contract_sha256": USD_SHA256}
        if cfg.verify_sha and sha != USD_SHA256:
            raise ValueError(f"USD SHA-256 {sha} != contract {USD_SHA256}; pass verify_sha=False to override")
        self.log(f"[plant] USD sha256 ok ({time.perf_counter() - t0:.1f} s)")

        t0 = time.perf_counter()
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        prepare_stage = load_prepare_stage()
        stage, changes = prepare_stage(usd)
        from pxr import UsdPhysics

        overrides = []
        for jname, axis in cfg.diag_axis_overrides:
            prims = [p for p in stage.Traverse() if p.GetName() == jname and p.IsA(UsdPhysics.RevoluteJoint)]
            if len(prims) != 1:
                raise KeyError(f"diag axis override: {len(prims)} revolute joints named {jname!r}")
            attr = UsdPhysics.RevoluteJoint(prims[0]).GetAxisAttr()
            overrides.append({"joint": str(prims[0].GetPath()), "from": str(attr.Get()), "to": axis})
            attr.Set(axis)
        if overrides:
            self.report["DIAGNOSTIC_axis_overrides"] = overrides
            self.log(f"[plant] DIAGNOSTIC in-memory axis overrides (not plant authority): {overrides}")

        if cfg.usd_fixes:
            from .usd_fixes import apply_contract_fixes

            fixes = apply_contract_fixes(stage, "/humanoid", ROOT_BODY_LABEL,
                                         authored_ankle_tierods=cfg.authored_ankle_tierods)
            self.report["usd_fixes_contract_0_1"] = fixes
            self.report["authored_ankle_tierods"] = fixes["authored_ankle_tierods"]
            self.log(f"[plant] CONTRACTS 0.1 fixes: deactivated {fixes['deactivated']}, friction->0 on "
                     f"{[f['joint'] for f in fixes['friction']]}, axis {fixes['axis']}, "
                     f"inertia {[f['body'] for f in fixes['inertia']]}; 0.2 spherical ankle tie rods "
                     f"{[f['joint'] for f in fixes['spherical']]} (authored ankle: {fixes['authored_ankle_tierods']})")
        else:
            self.report["usd_fixes_contract_0_1"] = None
            self.log("[plant] WARNING: raw USD without the CONTRACTS 0.1 fixes")
        closure_paths = {str(p.GetPath()) for p in stage.Traverse()
                         if p.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()
                         and UsdPhysics.Joint(p).GetJointEnabledAttr().Get()}
        builder.add_usd(stage, floating=not cfg.fixed_base, enable_self_collisions=False,
                        load_visual_shapes=False, force_show_colliders=True, root_path="/humanoid")
        if cfg.collisions == "box":
            builder.approximate_meshes("bounding_box")
        elif cfg.collisions != "convex-hull":
            raise ValueError(f"unknown collisions mode {cfg.collisions!r}")
        self.report["import"] = {"seconds": time.perf_counter() - t0, "reversed_joints": changes["reversedJoints"],
                                 "collision_meshes": changes["collisionMeshes"],
                                 "expanded_instances": len(changes["expandedInstances"])}
        self.log(f"[plant] USD imported in {time.perf_counter() - t0:.1f} s: {len(builder.body_q)} bodies, "
                 f"{len(builder.joint_type)} joints")

        self._configure_actuators(builder, closure_paths)
        self.report["motor_joint_friction_after_import"] = {
            n: float(builder.joint_friction[d]) for n, d in zip(motors.MOTOR_NAMES, self.motor_qd_idx)}
        builder.add_ground_plane()

        labels = list(builder.joint_label)
        missing = sorted(closure_paths - set(labels))
        if missing:
            raise ValueError(f"loop closures lost during import: {missing}")
        self.closure_joints = [i for i, lab in enumerate(labels) if lab in closure_paths]
        self.body_labels = list(builder.body_label)
        self.joint_labels = labels
        self.root_body = self.body_labels.index(ROOT_BODY_LABEL)
        self.root_joint = next(i for i, (c, par) in enumerate(zip(builder.joint_child, builder.joint_parent))
                               if c == self.root_body and par == -1)
        short = {lab.rsplit("/", 1)[-1]: i for i, lab in enumerate(self.body_labels)}
        missing_bodies = [b for b in cfg.extra_bodies if b not in short]
        if missing_bodies:
            raise KeyError(f"unknown extra bodies {missing_bodies}; available e.g. {list(short)[:8]}")
        self.extra_body_names = tuple(cfg.extra_bodies)
        self._readout_bodies = [self.root_body] + [short[b] for b in cfg.extra_bodies]

        if cfg.initial_motor_q is not None:
            q0 = np.asarray(cfg.initial_motor_q, dtype=float)
            for k, qi in enumerate(self.motor_q_idx):
                builder.joint_q[qi] = float(q0[k])

        self.model = builder.finalize(device=self.device)
        self.report["model"] = {"bodies": self.model.body_count, "joints": self.model.joint_count,
                                "dofs": self.model.joint_dof_count, "coords": self.model.joint_coord_count,
                                "closures": len(self.closure_joints),
                                "mass_kg": float(self.model.body_mass.numpy().sum())}

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._place_root()

        t0 = time.perf_counter()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, iterations=cfg.iterations, ls_iterations=cfg.ls_iterations, njmax=cfg.njmax,
            nconmax=cfg.nconmax, use_mujoco_contacts=True, use_mujoco_cpu=cfg.mujoco_cpu,
            disable_contacts=cfg.disable_contacts,
            save_to_mjcf=None if cfg.save_mjcf is None else str(cfg.save_mjcf))
        if cfg.diag_eq_solref is not None:
            if not cfg.mujoco_cpu:
                raise NotImplementedError("diag_eq_solref is implemented for the MuJoCo C backend only")
            before = self.solver.mj_model.eq_solref.copy()
            self.solver.mj_model.eq_solref[:] = np.asarray(cfg.diag_eq_solref, dtype=float)
            self.report["DIAGNOSTIC_eq_solref"] = {"from": np.unique(before, axis=0).tolist(),
                                                   "to": list(cfg.diag_eq_solref), "neq": int(self.solver.mj_model.neq)}
            self.log(f"[plant] DIAGNOSTIC eq_solref {np.unique(before, axis=0).tolist()} -> {list(cfg.diag_eq_solref)}")
        # MuJoCo C computes its own contacts internally and has no Newton contact buffer.
        self.contacts = None if cfg.mujoco_cpu else newton.Contacts(
            self.solver.get_max_contact_count(), 0, device=self.device)
        self.report["solver_build_s"] = time.perf_counter() - t0

        self._alloc_buffers()
        self.tick = 0
        self.time_s = 0.0
        self._graphs: list[Any] = []
        self._parity = 0
        if cfg.start_motor_q is not None:
            self._presettle(np.asarray(cfg.start_motor_q, dtype=float))
        # MuJoCo C steps copy through host memory and cannot be captured in a CUDA graph.
        if self.device.is_cuda and cfg.use_cuda_graph and not cfg.mujoco_cpu:
            self._capture()

    # ------------------------------------------------------------------ build helpers

    def _configure_actuators(self, builder, closure_paths: set[str]) -> None:
        """Assign motor/neck/passive DOF parameters and record motor/neck coordinate indices.

        Loop-closure joints (``closure_paths``) are left untouched: SolverMuJoCo turns them into
        equality constraints, so their coordinates are not simulated DOFs.
        """
        from newton import JointTargetMode

        by_name: dict[str, int] = {}
        for ji, lab in enumerate(builder.joint_label):
            by_name.setdefault(lab.rsplit("/", 1)[-1], ji)

        def dofs(ji: int) -> range:
            start = builder.joint_qd_start[ji]
            return range(start, start + sum(builder.joint_dof_dim[ji]))

        self.motor_q_idx: list[int] = []
        self.motor_qd_idx: list[int] = []
        hw = _hw_params(self.cfg.motor_profile)
        for k, name in enumerate(motors.MOTOR_NAMES):
            ji = by_name[name]
            if sum(builder.joint_dof_dim[ji]) != 1:
                raise ValueError(f"motor {name} is not a 1-DOF joint")
            d = builder.joint_qd_start[ji]
            builder.joint_target_mode[d] = int(JointTargetMode.EFFORT)
            builder.joint_target_ke[d] = 0.0
            builder.joint_target_kd[d] = 0.0
            builder.joint_damping[d] = 0.0
            builder.joint_armature[d] = hw[name]["armature"] if hw else motors.ARMATURE[k]
            builder.joint_effort_limit[d] = hw[name]["peak_torque"] if hw else motors.EFFORT_LIMIT[k]
            self.motor_q_idx.append(builder.joint_q_start[ji])
            self.motor_qd_idx.append(d)

        coord_targets = len(builder.joint_target_q) == len(builder.joint_q)
        self.neck_q_idx: list[int] = []
        self.neck_qd_idx: list[int] = []
        self._neck_target_idx: list[int] = []
        for name in motors.NECK_NAMES:
            ji = by_name[name]
            d = builder.joint_qd_start[ji]
            q = builder.joint_q_start[ji]
            builder.joint_target_mode[d] = int(JointTargetMode.POSITION)
            builder.joint_target_ke[d] = motors.NECK_KP
            builder.joint_target_kd[d] = motors.NECK_KD
            builder.joint_armature[d] = motors.NECK_ARMATURE
            builder.joint_effort_limit[d] = motors.NECK_EFFORT_LIMIT
            t = q if coord_targets else d
            builder.joint_target_q[t] = builder.joint_q[q]
            self.neck_q_idx.append(q)
            self.neck_qd_idx.append(d)
            self._neck_target_idx.append(t)

        passive_dofs = 0
        for name in motors.PASSIVE_JOINTS:
            for d in dofs(by_name[name]):
                builder.joint_target_mode[d] = int(JointTargetMode.NONE)
                builder.joint_target_ke[d] = 0.0
                builder.joint_target_kd[d] = 0.0
                builder.joint_damping[d] = self.cfg.passive_damping
                builder.joint_armature[d] = motors.PASSIVE_ARMATURE
                passive_dofs += 1

        known = set(motors.MOTOR_NAMES) | set(motors.NECK_NAMES) | set(motors.PASSIVE_JOINTS)
        unknown = []
        for ji, lab in enumerate(builder.joint_label):
            name = lab.rsplit("/", 1)[-1]
            n = sum(builder.joint_dof_dim[ji])
            if n and name not in known and builder.joint_parent[ji] != -1 and lab not in closure_paths:
                unknown.append(name)
        if unknown:
            raise ValueError(f"joints with DOFs outside the motor/neck/passive lists: {unknown}")
        self.report["actuation"] = {"motor_dofs": len(self.motor_qd_idx), "neck_dofs": len(self.neck_qd_idx),
                                    "passive_dofs": passive_dofs, "coord_layout_targets": coord_targets}

    def lowest_collision_z(self) -> float:
        """Lowest world z of any robot collision-shape vertex in ``state_0`` [m]."""
        from newton import GeoType, ShapeFlags

        body_q = self.state_0.body_q.numpy()
        shape_body = self.model.shape_body.numpy()
        shape_tf = self.model.shape_transform.numpy()
        shape_type = self.model.shape_type.numpy()
        shape_scale = self.model.shape_scale.numpy()
        shape_flags = self.model.shape_flags.numpy()
        zmin = np.inf
        for s in range(len(shape_body)):
            b = int(shape_body[s])
            if b < 0 or not int(shape_flags[s]) & int(ShapeFlags.COLLIDE_SHAPES):
                continue
            if int(shape_type[s]) in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)):
                verts = np.asarray(self.model.shape_source[s].vertices, dtype=float) * shape_scale[s]
            else:  # primitive: use the scale as a conservative half-extent box
                e = np.abs(shape_scale[s])
                verts = np.array([[sx, sy, sz] for sx in (-e[0], e[0]) for sy in (-e[1], e[1])
                                  for sz in (-e[2], e[2])], dtype=float)
            w = _transform_points(body_q[b], _transform_points(shape_tf[s], verts))
            zmin = min(zmin, float(w[:, 2].min()))
        return zmin

    def _place_root(self) -> None:
        """Lift/lower the robot so its lowest collision point sits at the configured height."""
        zmin = self.lowest_collision_z()
        target = self.cfg.hang_clearance if self.cfg.fixed_base else self.cfg.spawn_clearance
        dz = target - zmin
        if self.cfg.fixed_base:
            x_p = self.model.joint_X_p.numpy()
            x_p[self.root_joint][2] += dz
            self.model.joint_X_p.assign(x_p)
        else:
            q = self.model.joint_q.numpy()
            qs = self.model.joint_q_start.numpy()[self.root_joint]
            q[qs + 2] += dz
            self.model.joint_q.assign(q)
        for st in (self.state_0, self.state_1):
            st.joint_q.assign(self.model.joint_q)
            st.joint_qd.assign(self.model.joint_qd)
        self.newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.report["placement"] = {"lowest_point_before_m": zmin, "dz_m": dz,
                                    "lowest_point_after_m": self.lowest_collision_z(),
                                    "root_pos_w": self.state_0.body_q.numpy()[self.root_body][:3].tolist()}
        self.log(f"[plant] placed root: lowest collision point {zmin:+.4f} -> {target:+.4f} m (dz {dz:+.4f})")

    def _presettle(self, q_goal: np.ndarray) -> None:
        """Solve the loop closures at ``q_goal`` with the root pinned (see ``PlantConfig.start_motor_q``)."""
        if q_goal.shape != (motors.NUM_MOTORS,) or not np.isfinite(q_goal).all():
            raise ValueError(f"start_motor_q must be 22 finite values, got {q_goal.shape}")
        t0 = time.perf_counter()
        n = max(2, int(round(self.cfg.presettle_s / self.cfg.tick_dt)))
        n_ramp = max(1, n // 2)
        q0 = self.initial_readout().motor_q.copy()
        kp = np.asarray(motors.DEFAULT_KP) * self.cfg.presettle_gain
        kd = np.asarray(motors.DEFAULT_KD) * self.cfg.presettle_gain
        zeros, ones = np.zeros(motors.NUM_MOTORS), np.ones(motors.NUM_MOTORS, np.int32)
        pin = not self.cfg.fixed_base
        qs = int(self.model.joint_q_start.numpy()[self.root_joint])
        ds = int(self.model.joint_qd_start.numpy()[self.root_joint])
        root_q = self.state_0.joint_q.numpy()[qs:qs + 7].copy() if pin else None
        worst = []
        for k in range(n):
            a = min(1.0, (k + 1) / n_ramp)
            self.set_motor_command(q0 + a * (q_goal - q0), zeros, zeros, kp, kd, ones)
            self.step()
            if pin:
                st = self.current_state()
                jq, jqd = st.joint_q.numpy(), st.joint_qd.numpy()
                jq[qs:qs + 7] = root_q
                jqd[ds:ds + 6] = 0.0
                st.joint_q.assign(jq)
                st.joint_qd.assign(jqd)
            if k % 50 == 0 or k == n - 1:
                worst.append(float(self.closure_residuals_m().max()))
        st = self.current_state()
        motor_q = st.joint_q.numpy()[self.motor_q_idx].astype(float)
        closure = float(self.closure_residuals_m().max())
        self.model.joint_q.assign(st.joint_q.numpy())
        self.model.joint_qd.zero_()
        for s in (self.state_0, self.state_1):
            s.joint_q.assign(self.model.joint_q)
            s.joint_qd.zero_()
        self.newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._place_root()
        self.solver.reset(self.state_0)
        self.control.joint_f.zero_()
        self._tau.zero_()
        self._parity = 0
        self.tick = 0
        self.time_s = 0.0
        self.report["presettle"] = {
            "ticks": n, "ramp_ticks": n_ramp, "gain": self.cfg.presettle_gain, "root_pinned": pin,
            "motor_err_max_rad": float(np.abs(motor_q - q_goal).max()),
            "worst_motor": motors.MOTOR_NAMES[int(np.abs(motor_q - q_goal).argmax())],
            "closure_residual_max_m": closure, "closure_trace_m": worst, "seconds": time.perf_counter() - t0,
            "finite": self.is_finite()}
        self.log(f"[plant] pre-settled at the start pose in {n} ticks: motor err max "
                 f"{self.report['presettle']['motor_err_max_rad']:.4f} rad ({self.report['presettle']['worst_motor']}), "
                 f"closure residual {1e3 * closure:.2f} mm")

    def _alloc_buffers(self) -> None:
        wp, dev = self.wp, self.device
        n = motors.NUM_MOTORS
        i32 = lambda v: wp.array(np.asarray(v, np.int32), dtype=int, device=dev)  # noqa: E731
        f32 = lambda v: wp.array(np.asarray(v, np.float32), dtype=float, device=dev)  # noqa: E731
        self._motor_q_idx = i32(self.motor_q_idx)
        self._motor_qd_idx = i32(self.motor_qd_idx)
        self._neck_q_idx = i32(self.neck_q_idx)
        self._neck_qd_idx = i32(self.neck_qd_idx)
        self._bodies = i32(self._readout_bodies)
        hw = _hw_params(self.cfg.motor_profile)
        col = lambda key: [hw[m][key] for m in motors.MOTOR_NAMES] if hw else [1.0] * n  # noqa: E731
        self._effort = f32(col("peak_torque") if hw else motors.EFFORT_LIMIT)
        self._hw = 1 if hw else 0
        self._sat, self._v0 = f32(col("saturation_effort")), f32(col("no_load_speed"))
        self._coulomb = f32(col("coulomb_friction") if hw else [0.0] * n)
        self._viscous = f32(col("viscous_friction") if hw else [0.0] * n)
        self.report["motor_profile"] = self.cfg.motor_profile or "legacy"
        self._q_des = f32(np.zeros(n))
        self._dq_des = f32(np.zeros(n))
        self._tau_ff = f32(np.zeros(n))
        self._kp = f32(np.zeros(n))
        self._kd = f32(np.zeros(n))
        self._enable = i32(np.zeros(n))
        self._tau = f32(np.zeros(n))
        self._readout_len = 3 * n + 2 * motors.NUM_NECK + 13 * len(self._readout_bodies)
        self._out = wp.zeros(self._readout_len, dtype=float, device=dev)
        if self.control.joint_f is None:
            self.control.joint_f = wp.zeros(self.model.joint_dof_count, dtype=float, device=dev)
        self._neck_target_idx_np = np.asarray(self._neck_target_idx, np.int64)

    # ------------------------------------------------------------------ stepping

    def _substeps(self, src, dst):
        """Run ``substeps`` physics steps starting in ``src``; returns the state holding the result."""
        wp = self.wp
        a, b = src, dst
        for _ in range(self.cfg.substeps):
            a.clear_forces()
            wp.launch(self._pd_kernel, dim=motors.NUM_MOTORS, device=self.device,
                      inputs=[a.joint_q, a.joint_qd, self._motor_q_idx, self._motor_qd_idx, self._q_des,
                              self._dq_des, self._tau_ff, self._kp, self._kd, self._enable, self._effort, self._hw,
                              self._sat, self._v0, self._coulomb, self._viscous],
                      outputs=[self.control.joint_f, self._tau])
            self.solver.step(a, b, self.control, self.contacts, self.cfg.sim_dt)
            a, b = b, a
        wp.launch(self._pack_kernel, dim=1, device=self.device,
                  inputs=[a.joint_q, a.joint_qd, a.body_q, a.body_qd, self._motor_q_idx, self._motor_qd_idx,
                          self._tau, self._neck_q_idx, self._neck_qd_idx, self._bodies], outputs=[self._out])
        return a

    def _capture(self) -> None:
        wp = self.wp
        t0 = time.perf_counter()
        self._substeps(self.state_0, self.state_1)  # warm-up: compile/load kernels outside capture
        states = [(self.state_0, self.state_1)]
        if self.cfg.substeps % 2 == 1:
            states.append((self.state_1, self.state_0))
        # The warm-up advanced the physics by one tick: restore model defaults (the placed initial
        # configuration) and clear MuJoCo's warm-start/applied-force buffers.
        self.solver.reset(self.state_0)
        self.newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.control.joint_f.zero_()
        self._tau.zero_()
        self._graphs = []
        for src, dst in states:
            with wp.ScopedCapture(device=self.device) as cap:
                self._substeps(src, dst)
            self._graphs.append(cap.graph)
        self.report["graph_capture_s"] = time.perf_counter() - t0

    def current_state(self):
        """Newton ``State`` holding the latest tick."""
        if self._graphs and len(self._graphs) == 2 and self._parity == 1:
            return self.state_1
        return self.state_0

    def set_motor_command(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray, kp: np.ndarray, kd: np.ndarray,
                          enable: np.ndarray) -> None:
        """Set the 22-motor impedance command held until the next call (host -> device copy)."""
        self._q_des.assign(np.asarray(q, np.float32))
        self._dq_des.assign(np.asarray(dq, np.float32))
        self._tau_ff.assign(np.asarray(tau, np.float32))
        self._kp.assign(np.asarray(kp, np.float32))
        self._kd.assign(np.asarray(kd, np.float32))
        self._enable.assign(np.asarray(enable, np.int32))

    def set_neck_targets(self, q: np.ndarray) -> None:
        """Set the 6 neck lead-screw position targets [m] of the stiff implicit PD."""
        tgt = self.control.joint_target_q.numpy()
        tgt[self._neck_target_idx_np] = np.asarray(q, np.float32)
        self.control.joint_target_q.assign(tgt)

    def step(self) -> None:
        """Advance one control tick (``substeps`` physics steps)."""
        if self._graphs:
            g = self._graphs[self._parity if len(self._graphs) == 2 else 0]
            self.wp.capture_launch(g)
            if len(self._graphs) == 2:
                self._parity ^= 1
        else:
            final = self._substeps(self.state_0, self.state_1)
            if final is not self.state_0:
                self.state_0, self.state_1 = self.state_1, self.state_0
        self.tick += 1
        self.time_s += self.cfg.tick_dt

    def readout(self) -> PlantReadout:
        """Download the packed tick readout (one device->host copy)."""
        out = self._out.numpy().astype(np.float64)
        n, nn = motors.NUM_MOTORS, motors.NUM_NECK
        o = 3 * n + 2 * nn
        bodies = out[o:].reshape(-1, 13)
        return PlantReadout(
            tick=self.tick, time_s=self.time_s, motor_q=out[0:n], motor_dq=out[n:2 * n], motor_tau=out[2 * n:3 * n],
            neck_q=out[3 * n:3 * n + nn], neck_dq=out[3 * n + nn:o], root_pos_w=bodies[0, 0:3],
            root_quat_wxyz=bodies[0, 3:7], root_lin_vel_w=bodies[0, 7:10], root_ang_vel_w=bodies[0, 10:13],
            body_pos_w=bodies[1:, 0:3], body_quat_wxyz=bodies[1:, 3:7])

    def initial_readout(self) -> PlantReadout:
        """Readout of the initial state without stepping."""
        self.wp.launch(self._pack_kernel, dim=1, device=self.device,
                       inputs=[self.state_0.joint_q, self.state_0.joint_qd, self.state_0.body_q, self.state_0.body_qd,
                               self._motor_q_idx, self._motor_qd_idx, self._tau, self._neck_q_idx, self._neck_qd_idx,
                               self._bodies], outputs=[self._out])
        return self.readout()

    def closure_residuals_m(self) -> np.ndarray:
        """Anchor separation [m] of each loop-closure joint in the current state (diagnostic; slow)."""
        st = self.current_state()
        body_q = st.body_q.numpy()
        parents = self.model.joint_parent.numpy()
        children = self.model.joint_child.numpy()
        xp = self.model.joint_X_p.numpy()
        xc = self.model.joint_X_c.numpy()
        res = []
        for j in self.closure_joints:
            pa = xp[j][:3] if parents[j] < 0 else _transform_points(body_q[parents[j]], xp[j][None, :3])[0]
            pc = _transform_points(body_q[children[j]], xc[j][None, :3])[0]
            res.append(float(np.linalg.norm(pa - pc)))
        return np.asarray(res)

    def is_finite(self) -> bool:
        st = self.current_state()
        return bool(np.isfinite(st.joint_q.numpy()).all() and np.isfinite(st.joint_qd.numpy()).all())


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate points ``v`` (N,3) by xyzw quaternion ``q``."""
    u, w = np.asarray(q[:3], float), float(q[3])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _transform_points(tf: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a Newton/warp transform ``(px,py,pz,qx,qy,qz,qw)`` to points (N,3)."""
    tf = np.asarray(tf, float)
    return _quat_rotate_xyzw(tf[3:7], np.asarray(pts, float)) + tf[:3]
