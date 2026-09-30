"""Pack what a cloud (Brev) training box needs into one .tgz with repo-relative paths (docs/BREV.md section 4).

Stdlib only (any Python 3.10+, Windows or WSL). Contents:

* code: ``source/ scripts/ tools/ third_party/pydeps/ docs/ requirements/`` (no ``__pycache__``/``*.pyc``);
* ``data/calibration/`` (the default calibration and its pinned snapshots);
* for every ``--manifest`` (``dropbear-motion-library-v1``): the manifest, every clip NPZ it lists and the clip's
  ``<clip>.validation.json`` verdict (the library refuses clips without an accepted verdict, CONTRACTS 5.3);
* ``--extra`` files/dirs (e.g. ``data/motions/smoke/dropbear_static_stand.npz`` for locomotion resets);
* ``BUNDLE.json``: file list with SHA-256, the manifests' clip pins, the source host and time.

The plant USD (421 MB, read-only on P:) is NOT packed: copy it separately and point ``$DROPBEAR_USD`` at it.

    python tools/pack_brev_bundle.py --manifest data/motions/libraries/accepted_v1.json --out ../dropbear-wbc-brev.tgz
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import os
import platform
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE_DIRS = ("source", "scripts", "tools", "third_party/pydeps", "docs", "requirements")
DATA_DIRS = ("data/calibration",)
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError as exc:
        raise SystemExit(f"{path} is outside the repo {REPO}; the bundle keeps repo-relative paths") from exc


def _walk(root: Path):
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_PARTS)
        for fn in sorted(filenames):
            if not fn.endswith((".pyc", ".pyo")):
                yield Path(dirpath) / fn


def manifest_files(manifest: Path) -> tuple[list[Path], dict]:
    """The manifest, its clip NPZs and their verdicts (fail closed on missing files)."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != "dropbear-motion-library-v1":
        raise SystemExit(f"{manifest}: not a dropbear-motion-library-v1 manifest")
    files, pins = [manifest], {}
    for c in data.get("clips", []):
        npz = Path(c["npz"] if isinstance(c, dict) else c)
        npz = (npz if npz.is_absolute() else manifest.parent / npz).resolve()
        verdict = npz.with_name(npz.stem + ".validation.json")
        missing = [str(p) for p in (npz, verdict) if not p.is_file()]
        if missing:
            raise SystemExit(f"{manifest.name}: missing {missing}")
        files += [npz, verdict]
        pins[_rel(npz)] = c.get("sha256") if isinstance(c, dict) else None
    return files, pins


def build_bundle(out: Path, manifests: list[Path], extra: list[Path], include_code: bool = True) -> dict:
    roots = ([REPO / d for d in CODE_DIRS] if include_code else []) + [REPO / d for d in DATA_DIRS]
    files: dict[str, Path] = {}
    for r in roots:
        if r.exists():
            for p in _walk(r):
                files[_rel(p)] = p
    info_manifests = {}
    for m in manifests:
        mf, pins = manifest_files(m.resolve())
        for p in mf:
            files[_rel(p)] = p
        info_manifests[_rel(m)] = {"clips": pins}
    for e in extra:
        if not e.exists():
            raise SystemExit(f"--extra {e} does not exist")
        for p in _walk(e.resolve()):
            files[_rel(p)] = p
    listing = {k: {"sha256": _sha256(p), "bytes": p.stat().st_size} for k, p in sorted(files.items())}
    for m in info_manifests.values():  # a pinned clip must be packed with the pinned bytes
        for rel, pin in m["clips"].items():
            if pin and listing[rel]["sha256"] != pin:
                raise SystemExit(f"{rel}: sha256 differs from its manifest pin; regenerate the manifest")
    info = {"schema": "dropbear-brev-bundle-v1", "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "host": platform.node(), "repo": str(REPO), "manifests": info_manifests,
            "usd_note": "plant USD not included: copy it and set $DROPBEAR_USD (sha256 must be 45586414...)",
            "files": listing}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for rel, p in sorted(files.items()):
            tar.add(str(p), arcname=rel, recursive=False)
        blob = (json.dumps(info, indent=1) + "\n").encode()
        ti = tarfile.TarInfo("BUNDLE.json")
        ti.size = len(blob)
        tar.addfile(ti, io.BytesIO(blob))
    return {"out": str(out), "sha256": _sha256(out), "bytes": out.stat().st_size, "num_files": len(listing),
            "manifests": list(info_manifests)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, action="append", default=[], help="library manifest(s) to ship")
    ap.add_argument("--extra", type=Path, action="append", default=[], help="additional repo files/dirs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-code", action="store_true", help="data only (tests)")
    a = ap.parse_args(argv)
    rep = build_bundle(a.out.resolve(), a.manifest, a.extra, include_code=not a.no_code)
    print(json.dumps(rep, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
