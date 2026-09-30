"""Batch-retarget every on-disk G1 motion source to Dropbear and write ``<out-root>/catalog.json``.

Uses a process pool (one calibration load per worker; BLAS pinned to 1 thread per process because
numpy/OpenBLAS sized for 32 threads per worker exhausted the Windows commit limit on this machine).
If the pool breaks (e.g. a worker dies with "paging file is too small"), the remaining clips are
processed serially in the parent. Sources without files on disk (LAFAN1) or
with git-lfs pointer stubs (GR00T SONIC reference) are listed under ``skipped`` in the catalog,
failures under ``failed`` -- nothing is dropped silently. Synthetic clips already present under
``<out-root>/synthetic`` (tools/make_synthetic_motion.py) are indexed too.

Examples::

    python tools/build_motion_library.py                      # real calibration -> data/motions
    python tools/build_motion_library.py --calibration tests/fixtures/mock_semantic_calibration.json \
        --allow-mock                                          # -> data/motions_mock (MOCK labelled)
    python tools/build_motion_library.py --sources kimodo_g1,asap_g1 --workers 8

Library v4 (foot_contact track, 2026-09-24), DEFAULT: the foot-contact stage runs (``dropbear_wbc.motion.foot_contact``;
contacts with hysteresis), the output is written at the settle rate (``--output-fps`` default 50) and every motor
trajectory is projected under the 10 rad/s gate (``--max-motor-step`` default 0.18). ``--no-foot-contact`` restores the
legacy v3 route (source rate, no projection)::

    python tools/build_motion_library.py --out-root data/motions --workers 3
"""

from __future__ import annotations

import os

# Must precede any numpy import (here and in spawned workers, which inherit the environment).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import datetime as dt
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "source", REPO / "third_party" / "pydeps"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dropbear_wbc.motion.calibration_view import DEFAULT_CALIBRATION, load_calibration  # noqa: E402
from dropbear_wbc.motion.g1_sources import SOURCES, discover_source_files, is_git_lfs_pointer  # noqa: E402
from dropbear_wbc.motion.foot_contact import FootContactParams  # noqa: E402
from dropbear_wbc.motion.g1_to_dropbear import RetargetOptions  # noqa: E402
from dropbear_wbc.motion.pipeline import PIPELINE_VERSION, retarget_file  # noqa: E402

DEFAULT_OUT = REPO / "data" / "motions"
MOCK_OUT = REPO / "data" / "motions_mock"

_WORKER: dict[str, Any] = {}


def _init_worker(cal_path: str, allow_mock: bool, out_root: str, opts: RetargetOptions) -> None:
    _WORKER["cal"] = load_calibration(cal_path, allow_mock=allow_mock)
    _WORKER["out_root"] = out_root
    _WORKER["opts"] = opts


def _job(item: tuple[str, str]) -> dict[str, Any]:
    source, path = item
    try:
        entry = retarget_file(path, _WORKER["out_root"], source=source, cal=_WORKER["cal"], opts=_WORKER["opts"])
        return {"ok": True, "entry": entry}
    except Exception as exc:  # noqa: BLE001 - reported in the catalog
        return {"ok": False, "source": source, "file": path, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=4)}


def _synthetic_entries(out_root: Path) -> list[dict[str, Any]]:
    entries = []
    for side_path in sorted((out_root / "synthetic").glob("*.json")):
        if "." in side_path.stem:  # <clip>.report.json / <clip>.validation.json are not clip sidecars
            continue
        side = json.loads(side_path.read_text(encoding="utf-8"))
        if side.get("schema") != "dropbear-motion-csv-v1" or "clip" not in side:
            continue
        entries.append(
            {
                "clip": side["clip"],
                "source": side["source"],
                "csv": str(side_path.with_suffix(".csv")).replace("\\", "/"),
                "license": side["source_license"]["license"],
                "redistributable": side["source_license"]["redistributable"],
                "fps": side["fps"],
                "num_frames": side["num_frames"],
                "duration_s": round(side["duration_s"], 3),
                "suitability": side.get("suitability", []),
                "saturation_summary": {
                    "frames_with_any_saturation_frac": side["saturation"]["frames_with_any_saturation_frac"],
                    "worst_joint": side["saturation"]["worst_joint"],
                },
                "mock_calibration": side["calibration"]["calibration_is_mock"],
            }
        )
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="all", help=f"comma list of {sorted(SOURCES)} or 'all'")
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    ap.add_argument("--allow-mock", action="store_true")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="0 = serial")
    ap.add_argument("--waist-mode", choices=("auto", "fold", "drop"), default="auto")
    ap.add_argument("--joint-mapping", choices=("anatomical", "by_name"), default="anatomical")
    ap.add_argument("--limit", type=int, default=0, help="max clips per source (0 = all), for quick tests")
    ap.add_argument("--foot-contact", action=argparse.BooleanOptionalAction, default=None,
                    help="run the foot-contact stage (library v4; default ON except with a MOCK calibration, whose "
                         "forward model does not exist; --no-foot-contact = legacy v3 route)")
    ap.add_argument("--output-fps", type=float, default=None,
                    help="resample the source first (default: 50 = settle rate with the foot stage, else source rate)")
    ap.add_argument("--max-motor-step", type=float, default=None,
                    help="project motor trajectories to this max step per output frame [rad] (default: 0.18 with the "
                         "foot stage = 9 rad/s at 50 Hz, else off; <= 0 = off)")
    ap.add_argument("--only", nargs="*", default=None, help="clip names (stems) to process; default all")
    ap.add_argument("--catalog-name", default="catalog.json")
    args = ap.parse_args(argv)

    cal = load_calibration(args.calibration, allow_mock=args.allow_mock)  # fail fast in the parent
    out_root = args.out_root or (MOCK_OUT if cal.is_mock else DEFAULT_OUT)
    if cal.is_mock and out_root.resolve() == DEFAULT_OUT.resolve():
        print("error: refusing to write MOCK-calibrated clips into data/motions", file=sys.stderr)
        return 2
    keys = sorted(SOURCES) if args.sources == "all" else [k.strip() for k in args.sources.split(",")]
    items: list[tuple[str, str]] = []
    skipped: list[dict[str, Any]] = []
    for key in keys:
        files = discover_source_files(key)
        if not files:
            skipped.append({"source": key, "reason": "no files on disk", "patterns": list(SOURCES[key].patterns)})
            continue
        if args.limit:
            files = files[: args.limit]
        for f in files:
            if args.only and not any(o == f.stem or o == f.name or o in str(f) for o in args.only):
                continue
            probe = f / "joint_pos.csv" if f.is_dir() else f
            if is_git_lfs_pointer(probe):
                skipped.append({"source": key, "file": str(f).replace("\\", "/"), "reason": "git-lfs pointer"})
                continue
            items.append((key, str(f)))
    if args.foot_contact is None:
        args.foot_contact = not cal.is_mock
        if cal.is_mock:
            print("[build_motion_library] MOCK calibration: foot-contact stage OFF (needs the real forward model)")
    if args.foot_contact and args.output_fps is None:
        args.output_fps = 50.0  # the stage's motor-step hinge is per OUTPUT frame: solve at the settle rate
    if args.max_motor_step is None:
        args.max_motor_step = 0.18 if args.foot_contact else None
    elif args.max_motor_step <= 0:
        args.max_motor_step = None
    opts = RetargetOptions(waist_mode=args.waist_mode, joint_mapping=args.joint_mapping, output_fps=args.output_fps,
                           foot_contact=FootContactParams() if args.foot_contact else None,
                           max_motor_step_rad=args.max_motor_step)
    print(f"[build_motion_library] {len(items)} clips, {len(skipped)} skipped, workers={args.workers}, "
          f"calibration={cal.path} mock={cal.is_mock} impl={cal.semantic_map_impl[:60]}", flush=True)
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    done: set[tuple[str, str]] = set()
    initargs = (str(cal.path), args.allow_mock, str(out_root), opts)

    def record(item: tuple[str, str], r: dict[str, Any]) -> None:
        done.add(item)
        results.append(r)
        tag = r["entry"]["clip"] if r["ok"] else f"FAILED {r['file']}: {r['error']}"
        print(f"  [{len(results)}/{len(items)}] {tag}", flush=True)

    pool_error = ""
    if args.workers > 0:
        try:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=initargs) as ex:
                futs = {ex.submit(_job, it): it for it in items}
                for fut in as_completed(futs):
                    record(futs[fut], fut.result())
        except BrokenProcessPool as exc:
            pool_error = f"process pool broke ({exc}); finishing serially"
            print(f"[build_motion_library] {pool_error}", flush=True)
    remaining = [it for it in items if it not in done]
    if remaining:
        _init_worker(*initargs)
        for it in remaining:
            record(it, _job(it))
    wall = time.perf_counter() - t0
    ok = sorted((r["entry"] for r in results if r["ok"]), key=lambda e: (e["source"], e["clip"]))
    failed = [r for r in results if not r["ok"]]
    synthetic = _synthetic_entries(out_root)
    catalog = {
        "schema": "dropbear-motion-catalog-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "pipeline_version": PIPELINE_VERSION,
        "calibration": cal.provenance(),
        "mock_calibration": cal.is_mock,
        "retarget_options": {"waist_mode": opts.waist_mode, "joint_mapping": opts.joint_mapping,
                             "ground_fix": opts.ground_fix,
                             "foot_lock_xy": opts.foot_lock_xy, "output_fps": opts.output_fps,
                             "foot_contact": opts.foot_contact is not None,
                             "max_motor_step_rad": opts.max_motor_step_rad},
        "num_clips": len(ok) + len(synthetic),
        "total_duration_s": round(sum(e["duration_s"] or 0 for e in ok + synthetic), 2),
        "wall_time_s": round(wall, 2),
        "workers": args.workers,
        "pool_error": pool_error,
        "clips": ok + synthetic,
        "skipped": skipped,
        "failed": failed,
        "licenses": {k: SOURCES[k].license.as_dict() for k in keys},
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / args.catalog_name).write_text(json.dumps(catalog, indent=1), encoding="utf-8")
    print(f"[build_motion_library] ok={len(ok)} synthetic={len(synthetic)} failed={len(failed)} "
          f"skipped={len(skipped)} wall={wall:.1f}s -> {out_root / args.catalog_name}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
