"""Download the large dropbear-wbc assets from the Hugging Face Hub and verify their SHA-256.

The list, sizes and checksums are in ``assets_manifest.json`` (committed). Files land where the code expects them:
the plant USD in ``assets/dropbear.usd`` (the default of ``dropbear_wbc.paths.usd_path``), policies in
``assets/policies/<run>/``, videos in ``assets/media/``, motion clips and datasets under ``data/``.

    python tools/fetch_assets.py                      # usd + policies (~470 MB): enough to run the demos
    python tools/fetch_assets.py --groups all         # + media, redistributable motions, GR00T dataset
    python tools/fetch_assets.py --groups media --list

Needs ``pip install huggingface_hub``. Existing files with the right checksum are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--groups", default="usd,policies", help="comma list of usd, policies, media, motions, datasets, or all")
    ap.add_argument("--list", action="store_true", help="only print what would be downloaded")
    ap.add_argument("--repo", default=None, help="override the Hugging Face repo id from the manifest")
    args = ap.parse_args()
    manifest = json.loads((REPO / "assets_manifest.json").read_text(encoding="utf-8"))
    groups = {g["group"] for g in manifest["files"]} if args.groups == "all" else set(args.groups.split(","))
    todo = [f for f in manifest["files"] if f["group"] in groups]
    total = sum(f["bytes"] for f in todo)
    print(f"{len(todo)} files, {total / 2**20:.0f} MB from https://huggingface.co/{args.repo or manifest['hf_repo']}")
    if args.list:
        for f in todo:
            print(f"  {f['bytes'] / 2**20:8.1f} MB  {f['dst']}")
        return 0
    from huggingface_hub import hf_hub_download

    bad = 0
    for i, f in enumerate(todo, 1):
        dst = REPO / f["dst"]
        if dst.is_file() and dst.stat().st_size == f["bytes"] and sha256(dst) == f["sha256"]:
            continue
        print(f"[{i}/{len(todo)}] {f['dst']} ({f['bytes'] / 2**20:.1f} MB)", flush=True)
        for attempt in range(4):  # the Hub occasionally drops a request mid-download; retry before giving up
            try:
                got = Path(hf_hub_download(args.repo or manifest["hf_repo"], f["hf"],
                                           repo_type=manifest.get("hf_repo_type", "model"),
                                           local_dir=REPO / ".cache" / "hf_assets"))  # no second copy in the HF cache
                break
            except Exception as exc:  # noqa: BLE001 - network errors come in many types
                if attempt == 3:
                    raise
                print(f"  retry {attempt + 1}/3 after {type(exc).__name__}", flush=True)
                time.sleep(3 * (attempt + 1))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(got), dst)
        if sha256(dst) != f["sha256"]:
            print(f"  CHECKSUM MISMATCH: {dst}")
            bad += 1
    print("done" if not bad else f"{bad} file(s) failed verification")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
