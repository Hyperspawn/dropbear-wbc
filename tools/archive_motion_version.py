"""Copy the current retargeted clips of a motion source to versioned names before a library rebuild.

For every ``dropbear-motion-csv-v1`` clip ``<dir>/<clip>.csv`` (+ ``<clip>.json`` sidecar, + ``semantic/<clip>.semantic.csv``)
this writes ``<clip><suffix>.csv`` / ``.json`` / ``semantic/<clip><suffix>.semantic.csv`` (default suffix ``_v3``). It
never overwrites an existing target, never touches NPZs (a settled ``<clip>.npz`` stays where it is and is the settle of
the archived CSV), and skips files that already carry a version suffix. The copied sidecar gets ``clip`` /
``semantic_trajectory`` updated and an ``archived`` block (original path, reason, time). Catalog files
(``catalog*.json``) are copied to ``catalog*<suffix>.json``.

    python tools/archive_motion_version.py data/motions/unitree_rl_lab_mimic data/motions/kimodo_g1 ... --suffix _v3 \
        --reason "library v4 rebuild (foot_contact track)"

foot_contact track, 2026-09-24.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path

VERSIONED = re.compile(r"_v\d+$")


def archive_dir(d: Path, suffix: str, reason: str, dry: bool = False) -> dict:
    out = {"dir": str(d).replace("\\", "/"), "copied": [], "skipped_existing": [], "skipped_versioned": []}
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    for side in sorted(d.glob("*.json")):
        stem = side.stem
        if stem.startswith("catalog"):
            dst = side.with_name(f"{stem}{suffix}.json")
            if not dst.exists():
                out["copied"].append(dst.name)
                if not dry:
                    shutil.copy2(side, dst)
            continue
        if "." in stem or VERSIONED.search(stem):  # .validation / .report / already versioned
            if VERSIONED.search(stem):
                out["skipped_versioned"].append(side.name)
            continue
        try:
            meta = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("schema") != "dropbear-motion-csv-v1":
            continue
        csv = side.with_suffix(".csv")
        if not csv.is_file():
            continue
        new = f"{stem}{suffix}"
        dst_csv, dst_json = d / f"{new}.csv", d / f"{new}.json"
        if dst_csv.exists() or dst_json.exists():
            out["skipped_existing"].append(new)
            continue
        sem_src = d / "semantic" / f"{stem}.semantic.csv"
        sem_dst = d / "semantic" / f"{new}.semantic.csv"
        meta["archived"] = {"from": str(csv).replace("\\", "/"), "suffix": suffix, "reason": reason, "time": now,
                            "note": "copy made before the clip was rebuilt in place; a settled <clip>.npz next to it "
                                    "(if any) was produced from THIS content"}
        meta["clip"] = new
        if sem_src.is_file():
            meta["semantic_trajectory"] = f"semantic/{new}.semantic.csv"
        if not dry:
            shutil.copy2(csv, dst_csv)
            dst_json.write_text(json.dumps(meta, indent=1), encoding="utf-8")
            if sem_src.is_file() and not sem_dst.exists():
                shutil.copy2(sem_src, sem_dst)
        out["copied"].append(new)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--reason", default="library rebuild")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not VERSIONED.search(args.suffix):
        raise SystemExit("--suffix must look like _v<N>")
    res = [archive_dir(d, args.suffix, args.reason, args.dry_run) for d in args.dirs]
    for r in res:
        print(f"[archive] {r['dir']}: copied {len(r['copied'])}, existing {len(r['skipped_existing'])}, "
              f"already versioned {len(r['skipped_versioned'])}")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
