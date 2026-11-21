"""Derive a hardware-curated motion library from a manifest and the arm-duty screen (backlog item 167, notes/50).

``tools/motion_duty_screen.py`` flags reference clips whose arm motors (X8 Pro elbows and shoulder pitches first) would
run above rated torque if tracked: 27 of 122 clips in ``accepted_v7gen`` (jumps, boxing, jumping jacks). A policy
trained on those clips learns thermal duty the hardware cannot sustain. This tool writes a NEW manifest for ``hw_*``
training (the source manifest is never modified; sim-only training keeps the full set)::

    python tools/curate_hw_library.py data/motions_v6ts/libraries/accepted_v8push.json            # -> accepted_v8push_hw.json
    python tools/curate_hw_library.py LIB.json --high exclude --warn weight=0.5 --name my_hw_lib
    python tools/curate_hw_library.py LIB.json --screen-json screen.json     # reuse a saved motion_duty_screen --json

Rules (fail closed):

* The screen's ``sus`` thresholds depend on ``--ambient_C`` (default ``DESIGN_AMBIENT_C`` 45 C, ASSUMED; item 363).
  A ``--screen-json`` without ``ambient_C`` (a pre-363 fixed-rated screen) is refused, and so is one saved at another
  ambient than an explicit ``--ambient_C``. ``curation.screen`` records the ambient and the thresholds used.
* Every clip of the source manifest is screened (or matched in ``--screen-json`` by resolved NPZ path; the NPZ bytes
  must still hash to the manifest pin). Severity is the screen's per-clip ``worst`` (HIGH / warn / ok).
* ``--high`` (default ``exclude``) and ``--warn`` (default ``keep``): ``exclude``, ``keep`` or ``weight=W`` (W > 0
  multiplies the clip's prior weight; the library refuses weight 0).
* ``--peak-only warn`` downgrades a clip that is HIGH only through instantaneous ``peak`` findings (18 of 27 HIGH clips
  in accepted_v8push) to warn; sustained (``sus``) and static HIGH always stay HIGH.
* A clip the screen could not process (missing NPZ, load error, not in ``--screen-json``) stops the tool (exit 2)
  unless ``--unscreened exclude`` or ``--unscreened keep`` is given; the choice is recorded per clip.
* The result is validated with ``motion_library.parse_manifest`` and, unless ``--no-verify``, loaded with
  ``build_library`` (pins, provenance, names) before it is written. ``curation`` records the source manifest name and
  file SHA-256, the screen parameters and thresholds, and every clip's severity, findings and action.

Needs mujoco when it runs the screen itself (requirements/tests-cpu.txt).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "source", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TOOL = "tools/curate_hw_library.py"
SCHEMA = "dropbear-hw-library-curation-v1"
SEVERITIES = ("HIGH", "warn", "ok")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def parse_action(text: str) -> tuple[str, float]:
    """``exclude`` / ``keep`` / ``weight=W`` -> (kind, factor). W must be finite and > 0."""
    t = text.strip().lower()
    if t in ("exclude", "keep"):
        return t, 1.0
    if t.startswith("weight="):
        try:
            w = float(t.split("=", 1)[1])
        except ValueError:
            raise ValueError(f"bad action {text!r}: weight=W needs a number") from None
        if not (w > 0 and w != float("inf")):
            raise ValueError(f"bad action {text!r}: W must be finite and > 0 (use 'exclude' to drop a clip)")
        return "weight", w
    raise ValueError(f"bad action {text!r}: use exclude, keep or weight=W")


def resolve_clip_paths(manifest: dict, manifest_path: Path) -> list[Path]:
    out = []
    for c in manifest["clips"]:
        q = Path(c["npz"])
        out.append((q if q.is_absolute() else manifest_path.parent / q).resolve())
    return out


def is_peak_only_high(rec: dict) -> bool:
    """HIGH only because of a ``peak`` finding (one instant above the motor's peak), no sustained / static HIGH."""
    highs = [f for f in rec.get("findings", []) if f["severity"] == "HIGH"]
    return bool(highs) and all(f["metric"] == "peak" for f in highs)


def curate(manifest: dict, manifest_path: Path, screen_res: dict, *, high=("exclude", 1.0), warn=("keep", 1.0),
           unscreened: str | None = None, peak_only: str = "high") -> dict:
    """Pure selection: ``{"clips": [...kept entries...], "decisions": [...every clip...], "unscreened": [...]}``.

    ``screen_res`` is a ``motion_duty_screen.screen`` result (``clips[*].path``/``worst``/``findings``).
    ``peak_only="warn"`` treats a clip that is HIGH only through instantaneous ``peak`` findings as warn.
    Raises ``RuntimeError`` when a clip is unscreened (or its bytes differ from the pin) and ``unscreened`` is None.
    """
    by_path = {}
    for rec in screen_res.get("clips", []):
        by_path[str(Path(rec["path"]).resolve())] = rec
    actions = {"HIGH": high, "warn": warn, "ok": ("keep", 1.0)}
    kept, decisions, missing = [], [], []
    for entry, path in zip(manifest["clips"], resolve_clip_paths(manifest, manifest_path)):
        name = entry.get("name") or path.stem
        rec = by_path.get(str(path))
        reason = None
        if rec is None:
            reason = "screen error (see the screen's errors)" if path.is_file() else f"NPZ missing: {path}"
        elif entry.get("sha256") and path.is_file() and _sha256(path) != entry["sha256"]:
            reason = "NPZ bytes differ from the manifest pin"
            rec = None
        w0 = float(entry.get("weight", 1.0))
        if rec is None:
            missing.append({"name": name, "reason": reason})
            if unscreened is None:
                decisions.append({"name": name, "severity": "unscreened", "action": "error", "reason": reason})
                continue
            kind, factor = (unscreened, 1.0)
            sev, findings = "unscreened", []
        else:
            sev = rec["worst"]
            if sev not in SEVERITIES:
                raise ValueError(f"clip {name}: unknown screen severity {sev!r}")
            findings = [f"{f['motor']} {f['metric']} {f['value']} {f['severity']}" for f in rec.get("findings", [])]
            if sev == "HIGH" and peak_only == "warn" and is_peak_only_high(rec):
                sev = "warn"
                findings.append("peak-only HIGH treated as warn")
            kind, factor = actions[sev]
        d = {"name": name, "severity": sev, "action": kind, "weight_in": w0, "findings": findings}
        if reason:
            d["reason"] = reason
        if kind == "exclude":
            d["weight_out"] = None
        else:
            new = dict(entry)
            new["weight"] = w0 * factor if kind == "weight" else w0
            d["weight_out"] = new["weight"]
            kept.append(new)
        decisions.append(d)
    if missing and unscreened is None:
        raise RuntimeError(f"{len(missing)} clip(s) could not be screened (pass --unscreened exclude|keep): "
                           + "; ".join(f"{m['name']}: {m['reason']}" for m in missing[:3]))
    return {"clips": kept, "decisions": decisions, "unscreened": missing}


def rebase_npz(entries: list[dict], src_dir: Path, out_dir: Path) -> list[dict]:
    """Rewrite relative ``npz`` paths so they resolve from ``out_dir`` the same as from ``src_dir``."""
    import os
    out = []
    for e in entries:
        q = Path(e["npz"])
        if not q.is_absolute() and src_dir.resolve() != out_dir.resolve():
            absq = (src_dir / q).resolve()
            try:
                q = Path(os.path.relpath(absq, out_dir.resolve()))
            except ValueError:
                q = absq
        out.append({**e, "npz": q.as_posix()})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path, help="source library manifest (never modified)")
    ap.add_argument("--name", default=None, help="output manifest name (default <source name>_hw)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: the source manifest's directory")
    ap.add_argument("--high", default="exclude", help="exclude | keep | weight=W (default exclude)")
    ap.add_argument("--warn", default="keep", help="exclude | keep | weight=W (default keep)")
    ap.add_argument("--unscreened", choices=("exclude", "keep"), default=None,
                    help="what to do with clips the screen could not process (default: stop, exit 2)")
    ap.add_argument("--peak-only", choices=("high", "warn"), default="high",
                    help="clips HIGH only through instantaneous peak findings: keep as HIGH (default) or treat as warn")
    ap.add_argument("--screen-json", type=Path, default=None, help="reuse a motion_duty_screen.py --json result")
    ap.add_argument("--map", default="user_2026_09_25")
    ap.add_argument("--window", type=float, default=None, help="screen window [s] (default: the screen's 2.0)")
    ap.add_argument("--lp-hz", type=float, default=None)
    ap.add_argument("--ambient_C", type=float, default=None,
                    help="ambient for the screen's sus thresholds [C] (default: DESIGN_AMBIENT_C 45, ASSUMED; with "
                         "--screen-json: the saved screen's, which must match if both are given)")
    ap.add_argument("--no-verify", action="store_true", help="skip build_library on the result (parse only)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    a = ap.parse_args(argv)

    src = a.manifest.resolve()
    manifest = json.loads(src.read_text(encoding="utf-8"))
    try:
        high, warn = parse_action(a.high), parse_action(a.warn)
    except ValueError as exc:
        ap.error(str(exc))
    if a.screen_json:
        res = json.loads(a.screen_json.read_text(encoding="utf-8"))
        if res.get("ambient_C") is None:
            print(f"refused: {a.screen_json} has no ambient_C (fixed-rated screen from before the thermal model); "
                  "re-run tools/motion_duty_screen.py --json", flush=True)
            return 2
        if a.ambient_C is not None and float(res["ambient_C"]) != a.ambient_C:
            print(f"refused: {a.screen_json} was screened at {res['ambient_C']} C, --ambient_C is {a.ambient_C} C",
                  flush=True)
            return 2
    else:
        import motion_duty_screen as mds
        paths = [p for p in resolve_clip_paths(manifest, src) if p.is_file()]
        kw = {}
        if a.window is not None:
            kw["window_s"] = a.window
        if a.lp_hz is not None:
            kw["lp_hz"] = a.lp_hz or None
        if a.ambient_C is not None:
            kw["ambient_C"] = a.ambient_C
        res = mds.screen(paths, a.map, **kw)
    errs = res.get("errors", [])
    for e in errs[:3]:
        print(f"  screen error {e['clip']}: {e['error'][:160]}", flush=True)
    if len(errs) > 3:
        print(f"  ... {len(errs) - 3} more screen errors", flush=True)
    try:
        sel = curate(manifest, src, res, high=high, warn=warn, unscreened=a.unscreened,
                     peak_only=a.peak_only)
    except RuntimeError as exc:
        print(f"refused: {exc}", flush=True)
        return 2
    if not sel["clips"]:
        print("every clip was excluded; nothing written", flush=True)
        return 2

    from dropbear_wbc.tasks.tracking import motion_library as ML

    out_dir = (a.out_dir or src.parent).resolve()
    name = a.name or f"{manifest.get('name') or src.stem}_hw"
    out = out_dir / f"{name}.json"
    if out.resolve() == src:
        ap.error("output would overwrite the source manifest; pick another --name")
    counts = {s: sum(d["severity"] == s for d in sel["decisions"]) for s in (*SEVERITIES, "unscreened")}
    excluded = [d["name"] for d in sel["decisions"] if d["action"] == "exclude"]
    new = {
        "schema": manifest["schema"],
        "name": name,
        "description": (f"Hardware-curated {manifest.get('name')!r}: arm-duty screen HIGH -> {a.high}, warn -> {a.warn}"
                        f" ({len(excluded)} of {len(manifest['clips'])} clips excluded) at {res['ambient_C']:g} C "
                        f"ambient, generated by {TOOL}. "
                        "For hw_* training; sim-only training keeps the source manifest."),
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "created_by": TOOL,
        "clips": rebase_npz(sel["clips"], src.parent, out_dir),
        "curation": {
            "schema": SCHEMA,
            "source_manifest": manifest.get("name"),
            "source_manifest_file_sha256": _sha256(src),
            "screen": {k: res.get(k) for k in ("schema", "map", "window_s", "lp_hz", "basis", "payload_kg", "ambient_C",
                                                "thresh", "thermal")},
            "policy": {"HIGH": a.high, "warn": a.warn, "ok": "keep", "unscreened": a.unscreened or "error",
                       "peak_only_high": a.peak_only},
            "counts": counts,
            "excluded": excluded,
            "decisions": sel["decisions"],
        },
    }
    parsed = ML.parse_manifest(new, base_dir=out_dir, path=out)
    if not a.no_verify:
        from dropbear_wbc.robots import dropbear_names as N
        ML.build_library(parsed, keep_body_names=list(N.TRACKED_BODIES))
    for d in sel["decisions"]:
        if d["severity"] != "ok":
            print(f"  {d['action']:<8} {d['severity']:<10} {d['name']:<60} {'; '.join(d['findings'])[:120]}", flush=True)
    print(f"{name}: {len(new['clips'])} of {len(manifest['clips'])} clips kept "
          f"(HIGH {counts['HIGH']}, warn {counts['warn']}, ok {counts['ok']}, unscreened {counts['unscreened']})",
          flush=True)
    if a.dry_run:
        print("dry run: nothing written", flush=True)
        return 0
    if out.exists() and not a.force:
        print(f"{out} exists; pass --force or another --name", flush=True)
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(new, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
