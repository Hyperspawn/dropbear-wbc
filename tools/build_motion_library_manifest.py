"""Build a motion-library manifest (``dropbear-motion-library-v1``, docs/CONTRACTS.md 5.3) from the validator verdicts.

Regenerate it whenever ``tools/validate_motion_npz.py --write-verdicts`` accepts new clips (e.g. the foot-contact
track's dances/walks). CPU only (numpy), any Python with numpy (system Python 3.12)::

    python tools/build_motion_library_manifest.py                  # -> data/motions/libraries/accepted_v<N+1>.json
    python tools/build_motion_library_manifest.py --dry-run        # print the selection, write nothing
    python tools/build_motion_library_manifest.py --name dances_v0 --include "unitree_rl_lab_mimic/*,v4/**"

Selection (fail closed; every exclusion is listed with its reason in the manifest's ``selection`` block):

1. Every ``<clip>.validation.json`` under ``--root`` (default ``data/motions``), minus ``--exclude`` globs
   (default: ``smoke/**`` (duplicates ``synthetic/stand``), ``**/archive/**``, ``libraries/**``), optionally limited to
   ``--include`` globs. Globs match the posix path relative to ``--root``.
2. Verdict ``accepted`` and not stale (the verdict's ``npz_sha256`` equals the NPZ bytes).
3. Per clip, the library's own checks (``motion_library.build_library`` on the clip alone): contract self-checks,
   fps = policy rate (50), ``validate_provenance`` (schema, producer status, verdict, USD SHA, ankle variant = the
   contract plant's spherical tie rods unless ``--authored-ankle``), tracked bodies present.
4. Across clips: identical joint/body/motor names (order included) as the first selected clip; a known
   ``calibration_sha256`` must equal ``--calibration-sha`` (default: SHA-256 of the default calibration file
   ``data/calibration/dropbear_semantic_calibration.json``); clips that record none are kept and reported "unknown".
5. The final manifest is loaded once more with ``build_library`` (all cross-clip checks) before it is written.

Clip order: the base manifest's order (``--base``, default the newest ``accepted_v*.json``) for clips still selected,
then new clips sorted by path. Each clip entry pins its NPZ ``sha256`` (``build_library`` refuses a changed file).
When the selection equals the base manifest's (same names, files, bytes and weights) nothing is written.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "source", REPO / "third_party" / "pydeps"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dropbear_wbc.robots import dropbear_names as N  # noqa: E402
from dropbear_wbc.tasks.tracking import motion_library as ML  # noqa: E402
from dropbear_wbc.tasks.tracking.motion_npz import (  # noqa: E402
    MotionFormatError,
    expected_usd_sha256,
    load_validation_verdict,
)

DEFAULT_ROOT = REPO / "data" / "motions"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "libraries"
DEFAULT_EXCLUDE = ("smoke/**", "**/archive/**", "libraries/**")
DEFAULT_CALIBRATION = REPO / "data" / "calibration" / "dropbear_semantic_calibration.json"
POLICY_FPS = 50.0
TOOL = "tools/build_motion_library_manifest.py"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _match(rel: str, globs) -> bool:
    return any(fnmatch.fnmatchcase(rel, g) or fnmatch.fnmatchcase(rel, g.replace("**/", "")) for g in globs)


def _rel(path: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:  # Windows: different drives have no relative path; keep the absolute one
        return Path(path).resolve().as_posix()


def newest_accepted_manifest(out_dir: Path) -> Path | None:
    best = None
    for p in out_dir.glob("accepted_v*.json"):
        m = re.fullmatch(r"accepted_v(\d+)\.json", p.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    return best[1] if best else None


def next_accepted_name(out_dir: Path) -> str:
    newest = newest_accepted_manifest(out_dir)
    n = int(re.fullmatch(r"accepted_v(\d+)\.json", newest.name).group(1)) + 1 if newest else 0
    return f"accepted_v{n}"


def select_clips(root: Path, *, include=(), exclude=DEFAULT_EXCLUDE, authored_ankle: bool = False,
                 calibration_sha: str | None = None, weights: dict[str, float] | None = None) -> dict:
    """Candidates -> ``{"selected": [...], "excluded": [...]}`` (see the module docstring for the rules)."""
    weights = dict(weights or {})
    selected, excluded = [], []
    for vpath in sorted(root.rglob("*.validation.json")):
        rel_v = _rel(vpath, root)
        npz = vpath.with_name(vpath.name[: -len(".validation.json")] + ".npz")
        rel = _rel(npz, root)
        if _match(rel, exclude) or (include and not _match(rel, include)):
            continue
        row = {"npz": rel, "path": npz}
        if not npz.is_file():
            excluded.append({**row, "reason": f"no NPZ next to {rel_v}"})
            continue
        try:
            verdict = load_validation_verdict(npz)
        except MotionFormatError as exc:
            excluded.append({**row, "reason": f"unreadable verdict: {exc}"})
            continue
        if verdict.get("verdict") != "accepted":
            excluded.append({**row, "reason": f"verdict {verdict.get('verdict')!r}: " + "; ".join(verdict.get("reasons") or [])})
            continue
        if verdict.get("stale"):
            excluded.append({**row, "reason": "verdict is STALE (written for other NPZ bytes); re-run the validator"})
            continue
        single = ML.parse_manifest({"schema": ML.LIBRARY_SCHEMA, "name": "check", "clips": [{"npz": str(npz)}]})
        try:
            data = ML.build_library(single, keep_body_names=list(N.TRACKED_BODIES), expected_fps=POLICY_FPS,
                                    expected_usd_sha256=expected_usd_sha256(), expected_authored_ankle=authored_ankle)
        except MotionFormatError as exc:  # LibraryFormatError is a subclass
            excluded.append({**row, "reason": str(exc)})
            continue
        info = data.clips[0]
        selected.append({**row, "sha256": info.sha256, "num_frames": info.num_frames,
                         "duration_s": round(info.duration_s, 3), "calibration_sha256": info.calibration_sha256,
                         "verdict_created": verdict.get("created"),
                         "names": (tuple(data.joint_names), tuple(data.body_names), tuple(data.motor_names))})

    # cross-clip: names vs the first selected clip, calibration vs the target
    keep = []
    ref_names = selected[0]["names"] if selected else None
    for s in selected:
        if s["names"] != ref_names:
            excluded.append({"npz": s["npz"], "path": s["path"],
                             "reason": f"joint/body/motor names differ from {selected[0]['npz']} (order matters)"})
        elif calibration_sha and s["calibration_sha256"] and s["calibration_sha256"] != calibration_sha:
            excluded.append({"npz": s["npz"], "path": s["path"],
                             "reason": f"calibration {s['calibration_sha256'][:12]}... != target {calibration_sha[:12]}..."})
        else:
            keep.append(s)
    # unique clip names: the stem, or <parent>_<stem> where two stems collide
    stems = [Path(s["npz"]).stem for s in keep]
    for s, stem in zip(keep, stems):
        s["name"] = stem if stems.count(stem) == 1 else f"{Path(s['npz']).parent.name}_{stem}"
        s["weight"] = float(weights.pop(s["name"], 1.0))
    if weights:
        raise SystemExit(f"--weight given for clips that are not selected: {sorted(weights)}")
    return {"selected": keep, "excluded": excluded}


def order_like_base(selected: list[dict], base: ML.LibraryManifest | None) -> list[dict]:
    if base is None:
        return sorted(selected, key=lambda s: s["npz"])
    pos = {str(c.npz).lower(): i for i, c in enumerate(base.clips)}
    old = [s for s in selected if str(s["path"].resolve()).lower() in pos]
    new = [s for s in selected if str(s["path"].resolve()).lower() not in pos]
    old.sort(key=lambda s: pos[str(s["path"].resolve()).lower()])
    return old + sorted(new, key=lambda s: s["npz"])


def same_as_base(entries: list[dict], base: ML.LibraryManifest | None) -> bool:
    if base is None or len(base.clips) != len(entries):
        return False
    for e, c in zip(entries, base.clips):
        if (e["name"] != c.name or str(e["path"].resolve()).lower() != str(c.npz).lower() or e["weight"] != c.weight
                or not c.npz.is_file() or ML._sha256_file(c.npz) != e["sha256"]):
            return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="motion data root scanned for *.validation.json")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--name", default="auto", help="manifest name; 'auto' = accepted_v<N+1> after the newest accepted_v*")
    ap.add_argument("--base", type=Path, default=None, help="manifest whose clip order is kept (default: newest accepted_v*)")
    ap.add_argument("--include", default="", help="comma list of globs (relative to --root); default: everything")
    ap.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDE), help="comma list of globs (relative to --root)")
    ap.add_argument("--weight", action="append", default=[], metavar="NAME=W", help="per-clip prior weight (default 1)")
    ap.add_argument("--authored-ankle", action="store_true", help="select clips of the authored-ankle plant variant")
    ap.add_argument("--calibration-sha", default=None,
                    help="required calibration SHA-256 for clips that record one (default: the default calibration file)")
    ap.add_argument("--description", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest file of the same name")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    cal = args.calibration_sha or (_sha256(DEFAULT_CALIBRATION) if DEFAULT_CALIBRATION.is_file() else None)
    weights = {}
    for w in args.weight:
        k, _, v = w.partition("=")
        weights[k.strip()] = float(v)
    sel = select_clips(root, include=[g for g in args.include.split(",") if g.strip()],
                       exclude=[g for g in args.exclude.split(",") if g.strip()], authored_ankle=args.authored_ankle,
                       calibration_sha=cal, weights=weights)
    base_path = args.base or newest_accepted_manifest(out_dir)
    base = ML.load_manifest(base_path) if base_path and base_path.is_file() else None
    entries = order_like_base(sel["selected"], base)
    name = next_accepted_name(out_dir) if args.name == "auto" else args.name
    out = out_dir / f"{name}.json"

    for e in entries:
        print(f"  + {e['name']:<28} {e['npz']:<52} {e['num_frames']:>5} frames  cal "
              f"{(e['calibration_sha256'] or 'unknown')[:12]}", flush=True)
    for x in sel["excluded"]:
        print(f"  - {x['npz']:<52} {x['reason'][:160]}", flush=True)
    if not entries:
        print("no accepted clip selected; nothing written", flush=True)
        return 2
    if base is not None and args.name == "auto" and same_as_base(entries, base):
        print(f"selection unchanged vs {base_path.name} (same clips, bytes, weights, order): nothing written", flush=True)
        return 0

    rel_base = out_dir
    clips = [{"name": e["name"], "npz": _rel(e["path"], rel_base), "weight": e["weight"], "sha256": e["sha256"]}
             for e in entries]
    manifest = {
        "schema": ML.LIBRARY_SCHEMA,
        "name": name,
        "description": args.description or (
            f"Every clip under {_rel(root, REPO)} with a non-stale 'accepted' validator verdict that passes the library "
            f"checks on the {'authored-ankle' if args.authored_ankle else 'contract (spherical ankle)'} plant, generated "
            f"by {TOOL}. Excluded clips and reasons: 'selection.excluded'."),
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "created_by": TOOL,
        "clips": clips,
    }
    parsed = ML.parse_manifest(manifest, base_dir=out_dir, path=out)
    data = ML.build_library(parsed, keep_body_names=list(N.TRACKED_BODIES), expected_fps=POLICY_FPS,
                            expected_usd_sha256=expected_usd_sha256(), expected_authored_ankle=args.authored_ankle)
    rep = ML.library_report(data)
    manifest["selection"] = {
        "root": _rel(root, REPO), "include": args.include or None, "exclude": args.exclude,
        "authored_ankle_tierods": bool(args.authored_ankle), "calibration_sha256_target": cal,
        "base_manifest": base_path.name if base_path else None,
        "included": [{"name": e["name"], "npz": e["npz"], "num_frames": e["num_frames"], "duration_s": e["duration_s"],
                      "calibration_sha256": e["calibration_sha256"], "verdict_created": e["verdict_created"]}
                     for e in entries],
        "excluded": [{"npz": x["npz"], "reason": x["reason"]} for x in sel["excluded"]],
    }
    manifest["library"] = {k: rep[k] for k in ("sha256", "num_clips", "num_frames_total", "duration_s_total",
                                                "calibration_sha256", "calibration_unknown_clips", "all_accepted")}
    text = json.dumps(manifest, indent=1) + "\n"
    print(f"library {name!r}: {rep['num_clips']} clips, {rep['duration_s_total']} s, sha {rep['sha256'][:12]}", flush=True)
    if args.dry_run:
        print("dry run: nothing written", flush=True)
        return 0
    if out.exists() and not args.force:
        print(f"{out} exists; pass --force or another --name", flush=True)
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    ML.build_library(out, keep_body_names=list(N.TRACKED_BODIES))  # round trip from disk
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
