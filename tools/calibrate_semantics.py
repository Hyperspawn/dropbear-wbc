"""Measure Dropbear's semantic (G1-named) joint space from the USD physics and write the calibration.

Stages (``--stage``):

* ``sweep``  (Isaac Lab 2.2, GPU): quasi-static motor sweeps in a gravity-free, root-FIXED articulation
  with the contract 0.1 spawn fixes (``dropbear_wbc.isaac.quasistatic``). All programs run in parallel
  envs (``dropbear_wbc.isaac.sweep_runner``): every motor over its authored range (closed-loop motors
  forward and back), and each ankle's two calf motors over a ``--grid`` x ``--grid`` grid (one env per
  grid row). Fixed-length stages: ``--ramp`` steps of linear target ramp + ``--hold`` steps of hold per
  recorded sample. Per sample the closure gaps, the joint-position change over the last ``--window``
  steps and the motor tracking error are stored; samples with a closure gap > ``--max-gap`` are treated
  as outside the valid range by the fit. Raw poses of all bodies (relative to the root link) go to ``--raw``.
* ``fit``    (CPU, numpy): derive per-DOF maps, semantic zero, standing pose, key bodies and segment
  lengths from ``--raw`` and write ``--out`` (schema ``dropbear-semantic-calibration-v1``).
* ``verify`` (Isaac Lab 2.2, GPU): drive the fitted ``semantic_zero_motor_pos`` / ``standing_motor_pos``,
  named extreme poses and random semantic poses (reached along paths interpolated in semantic space),
  compare the measured semantic angles with the requested ones, store in ``--verify-raw``; then re-run
  ``fit`` to embed the verification summary.
* ``all``: sweep -> fit -> verify -> fit (two GPU processes are preferable: run the stages separately).

GPU stages must hold the team GPU lock (``docs/CONTRACTS.md`` section 0). Run them through
``tools/gpu_lock_run.py`` and pass ``--gpu-lock-held``, e.g. (from ``<repo>``)::

    python tools/gpu_lock_run.py --owner calibrate_settle --log logs/<component>/calib_sweep_vN.log -- \
        C:/isaac-sim/python.bat -u tools/calibrate_semantics.py --stage sweep --gpu-lock-held --headless \
        --raw logs/<component>/calib/raw/semantic_sweep_vN.npz
    python tools/calibrate_semantics.py --stage fit --raw logs/<component>/calib/raw/semantic_sweep_vN.npz
    python tools/gpu_lock_run.py ... -- C:/isaac-sim/python.bat -u tools/calibrate_semantics.py --stage verify \
        --raw <sweep npz> --verify-raw logs/<component>/calib/raw/semantic_verify_vN.npz --gpu-lock-held --headless
    python tools/calibrate_semantics.py --stage fit --raw <sweep npz> --verify-raw <verify npz>

Raw evidence paths (review fix 2026-09-24): ``--raw`` / ``--verify-raw`` have NO hard-coded default any more (they
used to point at the v2 authored-ankle evidence, so a bare ``--stage fit`` silently refit an authored-ankle
calibration over the current one). When omitted they default to the paths recorded in the provenance of the
calibration at ``--out`` (``raw_sweep``, ``verification.verify_npz``). A GPU stage refuses to overwrite an existing raw
file unless ``--force``, and ``fit`` fails when the sweep and verify raws come from different plant variants.

Units: radians, metres. Quaternions wxyz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

DEFAULT_OUT = REPO / "data/calibration/dropbear_semantic_calibration.json"
DEFAULT_SOLE = REPO / "data/calibration/dropbear_foot_sole_hulls.json"


def _parse() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["sweep", "fit", "verify", "all"], default="all")
    p.add_argument("--raw", type=Path, default=None,
                   help="raw sweep NPZ (default: provenance.raw_sweep of the calibration at --out)")
    p.add_argument("--verify-raw", type=Path, default=None,
                   help="raw verify NPZ (default: verification.verify_npz of the calibration at --out)")
    p.add_argument("--force", action="store_true", help="allow a GPU stage to overwrite an existing raw NPZ")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sole", type=Path, default=DEFAULT_SOLE, help="foot sole hull JSON (tools/extract_foot_soles.py)")
    p.add_argument("--log", type=str, default="", help="log path recorded as provenance (the wrapper's --log)")
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--pos-iters", type=int, default=16)
    p.add_argument("--vel-iters", type=int, default=4)
    p.add_argument("--motor-kp", type=float, default=10000.0)
    p.add_argument("--motor-kd", type=float, default=100.0)
    p.add_argument("--passive-damping", type=float, default=0.5)
    p.add_argument("--ramp", type=int, default=4, help="physics steps of linear target ramp per stage")
    p.add_argument("--hold", type=int, default=12, help="physics steps of hold per recorded stage")
    p.add_argument("--window", type=int, default=4, help="steps over which dq_window is measured")
    p.add_argument("--serial-step-deg", type=float, default=4.0)
    p.add_argument("--loop-step-deg", type=float, default=0.5, help="knee crank / elbow motor sample step")
    p.add_argument("--calf-step-deg", type=float, default=2.0, help="1D calf-motor sweep sample step")
    p.add_argument("--loop-max-step-deg", type=float, default=1.0, help="max target change per stage, closed loops")
    p.add_argument("--grid", type=int, default=41, help="ankle grid points per calf motor")
    p.add_argument("--max-gap", type=float, default=3e-3, help="closure gap above which a sample is invalid [m]")
    p.add_argument("--verify-samples", type=int, default=64, help="random semantic poses in verify")
    p.add_argument("--verify-hold", type=int, default=16)
    p.add_argument("--gpu-lock-held", action="store_true", help="caller (tools/gpu_lock_run.py) holds the GPU lock")
    p.add_argument("--authored-ankle", action="store_true",
                   help="calibrate the authored revolute ankle tie rods (CONTRACTS 0.2 opt-out; default: spherical, "
                        "or $DROPBEAR_AUTHORED_ANKLE=1)")
    p.add_argument("--diag-spherical-tie-rods", action="store_true",
                   help="DIAGNOSTIC sweep (ankle grid only) with *_Revolute111/112 retyped to spherical joints; "
                        "never used for the contract calibration")
    return p.parse_known_args()


ARGS, KIT_ARGS = _parse()


# ----------------------------------------------------------------------------------------------------
# GPU stages
# ----------------------------------------------------------------------------------------------------
def _launch_app():
    from dropbear_wbc.isaac.launch import prepare_kit_python

    prepare_kit_python()
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    app_args = ap.parse_args(KIT_ARGS)
    app_args.headless = True
    return AppLauncher(app_args).app


def _qs_cfg():
    from dropbear_wbc.isaac.quasistatic import QuasiStaticCfg

    diag = ("LL_Revolute111", "LL_Revolute112", "RL_Revolute111", "RL_Revolute112") if ARGS.diag_spherical_tie_rods else ()
    return QuasiStaticCfg(num_envs=ARGS.num_envs, pos_iters=ARGS.pos_iters, vel_iters=ARGS.vel_iters,
                          motor_kp=ARGS.motor_kp, motor_kd=ARGS.motor_kd, passive_damping=ARGS.passive_damping,
                          diag_spherical_joints=diag, authored_ankle_tierods=True if ARGS.authored_ankle else None)


def build_sweep_programs(motor_limits, grid_n: int):
    """1D sweeps of all 22 motors + one program per ankle grid row (see module doc)."""
    import numpy as np

    from dropbear_wbc.kinematics.sweeps import grid2d_row, single_pose, sweep1d
    from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES

    deg = np.pi / 180.0
    margin = 0.25 * deg
    loop = {"LL_knee_actuator_joint", "RL_knee_actuator_joint", "LH_elbow_joint", "RH_elbow_joint"}
    calf = {"LL_Revolute67", "LL_Revolute81", "RL_Revolute67", "RL_Revolute81"}
    programs = [single_pose(np.zeros(22), max_step=1.0, name="rest")]
    for i, name in enumerate(MOTOR_NAMES):
        lo, hi = float(motor_limits[i, 0]) + margin, float(motor_limits[i, 1]) - margin
        if name in loop:
            step, max_step = ARGS.loop_step_deg * deg, ARGS.loop_max_step_deg * deg
        elif name in calf:
            step = max_step = ARGS.calf_step_deg * deg
        else:
            step = max_step = ARGS.serial_step_deg * deg
        programs.append(sweep1d(i, lo, hi, step, max_step=max_step, return_pass=name in loop | calf,
                                name=f"sweep1d:{name}"))
    for side in ("LL", "RL"):
        ia, ib = MOTOR_NAMES.index(f"{side}_Revolute67"), MOTOR_NAMES.index(f"{side}_Revolute81")
        a_vals = np.linspace(motor_limits[ia, 0] + margin, motor_limits[ia, 1] - margin, grid_n)
        b_vals = np.linspace(motor_limits[ib, 0] + margin, motor_limits[ib, 1] - margin, grid_n)
        for a in a_vals:
            programs.append(grid2d_row(ia, ib, float(a), b_vals, max_step=ARGS.loop_max_step_deg * 2 * deg,
                                       name=f"grid2d:{side}:{a:.6f}"))
    return programs


def _resolve_raw_paths() -> None:
    """Fill ``--raw`` / ``--verify-raw`` from the current calibration's provenance when omitted (see module doc)."""
    prov, ver = {}, {}
    if ARGS.out.is_file():
        cal = json.loads(ARGS.out.read_text(encoding="utf-8"))
        prov, ver = cal.get("provenance") or {}, cal.get("verification") or {}

    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else REPO / p

    if ARGS.raw is None:
        if not prov.get("raw_sweep"):
            raise SystemExit("--raw is required (the calibration at --out records no raw_sweep)")
        ARGS.raw = _abs(prov["raw_sweep"])
        print(f"[calibrate] --raw from provenance: {ARGS.raw}", flush=True)
    if ARGS.verify_raw is None and ver.get("verify_npz"):
        ARGS.verify_raw = _abs(ver["verify_npz"])
        print(f"[calibrate] --verify-raw from provenance: {ARGS.verify_raw}", flush=True)


def _refuse_overwrite(path: Path) -> None:
    if path.exists() and not ARGS.force:
        raise SystemExit(f"refusing to overwrite existing raw evidence {path} (pass a new path or --force)")


def _save_raw(path: Path, qs, cfg, programs, rec, extra: dict) -> None:
    import numpy as np

    from dropbear_wbc.isaac.launch import sha256_file
    from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES

    path.parent.mkdir(parents=True, exist_ok=True)
    run = {"ramp": ARGS.ramp, "hold": ARGS.hold, "window": ARGS.window, "max_gap_m": ARGS.max_gap,
           "stage_fixes": qs.stage_fixes}
    np.savez_compressed(
        path,
        program_names=np.array([p.name for p in programs]),
        joint_names=np.array(qs.joint_names), body_names=np.array(qs.body_names),
        motor_names=np.array(MOTOR_NAMES), motor_ids=np.array(qs.motor_ids_list),
        motor_limits=qs.motor_limits, joint_limits=qs.joint_limits,
        com_pos_b=qs.com_pos_b, body_masses=qs.body_masses,
        cfg_json=np.array(json.dumps(cfg.to_dict())), run_json=np.array(json.dumps(run)),
        usd_sha256=np.array(sha256_file(cfg.usd_path)), usd_path=np.array(cfg.usd_path),
        **qs.closure_table(), **qs.joint_frame_table(), **rec, **extra,
    )


def _stage_sweep() -> None:
    import numpy as np

    from dropbear_wbc.isaac.quasistatic import QuasiStaticDropbear
    from dropbear_wbc.isaac.sweep_runner import run_programs

    cfg = _qs_cfg()
    t_setup = time.time()
    qs = QuasiStaticDropbear(cfg)
    print(f"[sweep] articulation ready in {time.time() - t_setup:.1f}s: J={len(qs.joint_names)} "
          f"B={len(qs.body_names)} closures={len(qs.closures)} fixes={qs.stage_fixes}", flush=True)
    programs = build_sweep_programs(qs.motor_limits, ARGS.grid)
    if ARGS.diag_spherical_tie_rods:
        programs = [p for p in programs if p.name == "rest" or p.name.startswith("grid2d:")
                    or "Revolute67" in p.name or "Revolute81" in p.name]
    lens = sorted(len(p) for p in programs)
    print(f"[sweep] {len(programs)} programs on {qs.num_envs} envs, longest {lens[-1]} stages, "
          f"ramp={ARGS.ramp} hold={ARGS.hold}", flush=True)
    rec = run_programs(qs, programs, ARGS.ramp, ARGS.hold, ARGS.window, tag="sweep")
    _save_raw(ARGS.raw, qs, cfg, programs, rec, {})
    g = rec["gap"]
    print(f"[sweep] wrote {ARGS.raw} records={len(rec['program'])} physics_steps={int(rec['physics_steps'])} "
          f"ms/step={float(rec['ms_per_step']):.2f} wall={float(rec['wall_s']):.1f}s gap max={1e3 * g.max():.3f} "
          f"p95={1e3 * np.percentile(g, 95):.3f} mm; >{1e3 * ARGS.max_gap:.0f} mm: {int((g > ARGS.max_gap).sum())}",
          flush=True)


def verify_poses(smap, n_random: int, seed: int = 0) -> dict:
    """Named + random semantic poses -> {name: (semantic target (22,) or None, motor waypoints [...])}."""
    import numpy as np

    from dropbear_wbc.kinematics.semantic import SEMANTIC_NAMES

    rng = np.random.default_rng(seed)
    lo, hi = smap.semantic_limits[:, 0], smap.semantic_limits[:, 1]
    s_zero = smap.motor_to_semantic(smap.semantic_zero_motor_pos)
    s_stand = smap.motor_to_semantic(smap.standing_motor_pos)
    targets: dict[str, np.ndarray] = {"semantic_zero": s_zero, "standing": s_stand}

    def named(name: str, **kv) -> None:
        s = s_stand.copy()
        for k, v in kv.items():
            s[SEMANTIC_NAMES.index(k)] = v
        targets[name] = s

    frac = lambda j, f: lo[SEMANTIC_NAMES.index(j)] + f * (hi[SEMANTIC_NAMES.index(j)] - lo[SEMANTIC_NAMES.index(j)])  # noqa: E731
    for side in ("left", "right"):
        named(f"{side}_knee_max", **{f"{side}_knee": frac(f"{side}_knee", 0.97)})
        named(f"{side}_elbow_min", **{f"{side}_elbow": frac(f"{side}_elbow", 0.03)})
        named(f"{side}_ankle_pitch_hi", **{f"{side}_ankle_pitch": frac(f"{side}_ankle_pitch", 0.9)})
        named(f"{side}_ankle_pitch_lo", **{f"{side}_ankle_pitch": frac(f"{side}_ankle_pitch", 0.1)})
        named(f"{side}_ankle_roll_hi", **{f"{side}_ankle_roll": frac(f"{side}_ankle_roll", 0.9)})
        named(f"{side}_ankle_roll_lo", **{f"{side}_ankle_roll": frac(f"{side}_ankle_roll", 0.1)})
        named(f"{side}_hip_combo", **{f"{side}_hip_pitch": -0.8, f"{side}_hip_yaw": 0.3 * (1 if side == "left" else -1),
                                      f"{side}_hip_roll": 0.2 * (1 if side == "left" else -1)})
        named(f"{side}_shoulder_combo", **{f"{side}_shoulder_pitch": -1.2,
                                           f"{side}_shoulder_roll": 0.4 * (1 if side == "left" else -1),
                                           f"{side}_shoulder_yaw": 0.5})
    for k in range(n_random):
        span = 0.8 * (hi - lo)
        targets[f"random_{k:03d}"] = np.clip(s_stand + (rng.random(len(s_stand)) - 0.5) * span, lo, hi)
    return targets


def _stage_verify() -> None:
    import numpy as np

    from dropbear_wbc.isaac.quasistatic import QuasiStaticDropbear
    from dropbear_wbc.isaac.sweep_runner import run_programs
    from dropbear_wbc.kinematics.semantic import SemanticMap
    from dropbear_wbc.kinematics.sweeps import waypoint_program

    smap = SemanticMap.load(ARGS.out)
    targets = verify_poses(smap, ARGS.verify_samples)
    s0 = smap.motor_to_semantic(smap.semantic_zero_motor_pos)
    programs, requested, motor_cmd = [], [], []
    for name, s in targets.items():
        # path: rest -> semantic zero (motor space), then linear in semantic space to the target
        wps = [smap.semantic_zero_motor_pos] + [smap.semantic_to_motor(s0 + (s - s0) * f) for f in np.linspace(0.1, 1.0, 10)]
        m, rep = smap.semantic_to_motor(s, return_report=True)
        requested.append(rep.used)
        motor_cmd.append(m)
        programs.append(waypoint_program(wps, np.deg2rad(ARGS.loop_max_step_deg * 2), name))
    cfg = _qs_cfg()
    qs = QuasiStaticDropbear(cfg)
    rec = run_programs(qs, programs, ARGS.ramp, ARGS.verify_hold, ARGS.window, tag="verify")
    _save_raw(ARGS.verify_raw, qs, cfg, programs, rec,
              {"pose_semantic_requested": np.stack(requested), "pose_motor_targets": np.stack(motor_cmd)})
    print(f"[verify] wrote {ARGS.verify_raw} records={len(rec['program'])} gap max={1e3 * rec['gap'].max():.3f} mm "
          f"wall={float(rec['wall_s']):.1f}s", flush=True)


def _stage_fit() -> None:
    from dropbear_wbc.kinematics.calib_build import fit_calibration

    verify = ARGS.verify_raw if ARGS.verify_raw is not None and ARGS.verify_raw.exists() else None
    fit_calibration(raw_path=ARGS.raw, out_path=ARGS.out, sole_path=ARGS.sole, verify_path=verify,
                    script=str(Path(__file__).relative_to(REPO)).replace("\\", "/"), log_path=ARGS.log,
                    max_gap=ARGS.max_gap)


def main() -> int:
    stages = ["sweep", "fit", "verify", "fit"] if ARGS.stage == "all" else [ARGS.stage]
    if "sweep" in stages and ARGS.raw is None:
        raise SystemExit("--stage sweep/all needs an explicit --raw output path")
    if "verify" in stages and ARGS.verify_raw is None:
        raise SystemExit("--stage verify/all needs an explicit --verify-raw output path")
    _resolve_raw_paths()
    if "sweep" in stages:
        _refuse_overwrite(ARGS.raw)
    if "verify" in stages:
        _refuse_overwrite(ARGS.verify_raw)
    needs_gpu = any(s in ("sweep", "verify") for s in stages)
    app = None
    rc = 1
    try:
        if needs_gpu:
            if not ARGS.gpu_lock_held or not (REPO / ".locks/gpu.lock").exists():
                raise RuntimeError("GPU stages must run under tools/gpu_lock_run.py with --gpu-lock-held")
            app = _launch_app()
        for s in stages:
            print(f"[calibrate] stage {s}", flush=True)
            {"sweep": _stage_sweep, "fit": _stage_fit, "verify": _stage_verify}[s]()
        rc = 0
    except Exception:
        import traceback

        traceback.print_exc()
        rc = 1
    if app is not None:
        from dropbear_wbc.isaac.launch import close_app_and_exit

        close_app_and_exit(app, rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
