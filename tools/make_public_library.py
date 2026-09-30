"""Derive a ``<library>_public.json`` motion-library manifest that keeps only redistributable clips.

The full libraries (``accepted_v6ts``, ``accepted_v7gen``) include clips whose source licenses forbid redistribution
(LAFAN1 CC BY-NC-ND 4.0, NVIDIA BONES-SEED sample data, unitree_rl_lab dance mocap of unconfirmed provenance), so those
NPZs are not published. The public manifest lists the rest (synthetic, Kimodo-generated, Kimodo G1 examples, ASAP), all
of which ``tools/fetch_assets.py --groups motions`` downloads, with the same per-clip SHA-256 pins. Anyone can train or
evaluate on it without rebuilding anything; rebuild the excluded clips from your own downloads to use the full library
(docs/DATA.md).

    python tools/make_public_library.py data/motions_v6ts/libraries/accepted_v7gen.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PUBLIC_SOURCES = ("synthetic", "kimodo_gen", "kimodo_g1", "asap_g1")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()
    m = json.loads(args.manifest.read_text(encoding="utf-8"))
    keep = [c for c in m["clips"] if Path(c["npz"]).parent.name in PUBLIC_SOURCES]
    dropped = sorted({Path(c["npz"]).parent.name for c in m["clips"]} - set(PUBLIC_SOURCES))
    name = f"{m['name']}_public"
    out = {"schema": m["schema"], "name": name,
           "description": f"{m['name']} restricted to redistributable sources ({', '.join(PUBLIC_SOURCES)}); "
                          f"dropped: {', '.join(dropped)} (tools/make_public_library.py, docs/DATA.md)",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "created_by": "tools/make_public_library.py",
           "clips": keep, "derived_from": {"manifest": args.manifest.name, "num_clips": len(m["clips"]),
                                           "library_sha256": (m.get("library") or {}).get("sha256")}}
    dst = args.manifest.with_name(f"{name}.json")
    dst.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{dst}: {len(keep)}/{len(m['clips'])} clips (dropped sources: {', '.join(dropped)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
