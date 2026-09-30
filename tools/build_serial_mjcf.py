"""Build the DERIVED serial ("output-space") Dropbear MJCF from the current semantic calibration.

    DERIVED, NOT CANONICAL: the USD is the plant. Closed loops are collapsed into the dropbear-semantic-v1
    joints; motors are reached through SemanticMap (docs/CONTRACTS.md section 2, docs/SERIAL_MODEL_AND_GMR.md).

One command regenerates everything from whatever calibration JSON is current::

    .venv-newton/Scripts/python.exe tools/build_serial_mjcf.py

Outputs (default ``data/robot/``):

* ``dropbear_serial.xml``        robot MJCF: free root ``pelvis`` + 22 hinges named like the semantic DOFs
* ``dropbear_serial_scene.xml``  the same with a floor and a light (viewers, GMR, mjlab)
* ``dropbear_serial.json``       metadata (inputs + SHAs, frames, fit residuals, mass lumping, surrogate actuators)

Inputs: the calibration JSON (``--calibration``), the raw physics sweep named in its provenance (the
calibration's own forward model), ``data/robot/usd_body_properties_<sha8>.json`` (mass/inertia/collision hulls
read from the USD with pxr; extracted automatically with the .venv-newton interpreter if missing) and the foot
sole hulls. CPU only; no Isaac, no GPU.

After writing, the script compiles the model with MuJoCo and runs a quick FK self-check against the
calibration forward model (the full parity test is ``tests/test_serial_mjcf.py``).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

import numpy as np  # noqa: E402

from dropbear_wbc.kinematics import serial_model as sm  # noqa: E402

NEWTON_PY = REPO / ".venv-newton" / "Scripts" / "python.exe"


class Tee:
    def __init__(self, path: Path | None):
        self.f = open(path, "w", encoding="utf-8") if path else None

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + "\n")
            self.f.flush()


def ensure_body_props(usd_sha: str, log) -> Path:
    path = REPO / f"data/robot/usd_body_properties_{usd_sha[:8]}.json"
    if path.is_file():
        return path
    log(f"[build_serial_mjcf] {path.name} missing -> extracting from the USD with {NEWTON_PY}")
    cmd = [str(NEWTON_PY), str(REPO / "tools/extract_usd_body_properties.py"), "--out", str(path)]
    subprocess.run(cmd, check=True, cwd=str(REPO))
    return path


def self_check(xml: Path, meta: dict, calibration: Path, log, n: int = 300, seed: int = 1) -> dict:
    """Compile + FK at the zero pose + quick random-pose parity against the calibration forward model."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(REPO / "data/robot/dropbear_serial_scene.xml"))
    log(f"[self_check] scene compiles: nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} "
        f"nsite={model.nsite} total mass {float(model.body_subtreemass[1]):.3f} kg")
    smod = sm.SerialModel(xml)
    cfk = sm.CalibrationFK(calibration)
    rng = np.random.default_rng(seed)
    lim = cfk.smap.semantic_limits
    q = rng.uniform(lim[:, 0], lim[:, 1], (n, 22))
    q[0] = 0.0
    segs, used, _ = cfk.segments(q)
    pel = np.asarray(meta["frames"]["pelvis_in_root"]["pos"])
    errs: dict[str, list[float]] = {}
    for k in range(n):
        smod.fk(smod.qpos(pel, np.array([1.0, 0, 0, 0]), used[k]))
        for sname, st in meta["sites"].items():
            b = st["usd_body"]
            if b is None or b not in segs:
                continue
            _, p = smod.site_pose(sname)
            errs.setdefault(st["label"], []).append(float(np.linalg.norm(p - segs[b][1][k])))
    out = {}
    for lab, e in errs.items():
        e = 1e3 * np.asarray(e)
        out[lab] = {"zero_mm": float(e[0]), "p50_mm": float(np.median(e)), "p95_mm": float(np.percentile(e, 95)),
                    "max_mm": float(e.max())}
        log(f"[self_check] {lab:18s} zero {e[0]:6.2f} mm  p50 {np.median(e):6.2f}  p95 {np.percentile(e, 95):6.2f}  "
            f"max {e.max():6.2f} mm  (n={len(e)} uniform random semantic poses)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", type=Path, default=sm.DEFAULT_CALIBRATION)
    ap.add_argument("--body-props", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/robot")
    ap.add_argument("--log", type=Path, default=REPO / "logs/serial_mjcf_gmr/build_serial_mjcf.log")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args(argv)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = Tee(args.log)
    t0 = time.time()
    cal = json.loads(Path(args.calibration).read_text())
    log(f"[build_serial_mjcf] calibration {args.calibration} created {cal.get('created')} "
        f"sha256 {sm.sha256_file(args.calibration)[:16]} plant_variant: {cal.get('plant_variant')}")
    props = args.body_props or ensure_body_props(cal["usd_sha256"], log)
    spec = sm.fit_serial_model(args.calibration, body_props=props, log=log)
    xml = args.out_dir / "dropbear_serial.xml"
    meta = sm.write_mjcf(spec, xml, args.out_dir / "dropbear_serial.json", args.out_dir / "dropbear_serial_scene.xml")
    log(f"[build_serial_mjcf] wrote {xml} ({meta['xml_sha256'][:16]}), dropbear_serial.json, dropbear_serial_scene.xml")
    for k, v in meta["fit"].items():
        brief = {kk: (round(vv["rms_mm"], 2), round(vv["p95_mm"], 2), round(vv["max_mm"], 2)) for kk, vv in v.items()
                 if isinstance(vv, dict) and "rms_mm" in vv}
        extra = {kk: float(f"{vv:.3g}") for kk, vv in v.items() if kk.endswith("_deg") and isinstance(vv, float)}
        log(f"[fit] {k:16s} (rms, p95, max mm) {brief} {extra}")
    log(f"[mass] total {meta['total_mass_kg']:.3f} kg; placeholders {meta['placeholder_bodies']}")
    for n, b in meta["bodies"].items():
        log(f"[mass] {n:28s} {b['mass']:7.3f} kg  members {len(b['usd_members'])}")
    if not args.no_check:
        meta["self_check_vs_calibration_fk"] = self_check(xml, meta, args.calibration, log)
        (args.out_dir / "dropbear_serial.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"[build_serial_mjcf] done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
