"""Hardware tracking runs train on a hardware-curated motion library (item 349).

``tools/curate_hw_library.py`` writes ``<name>_hw.json`` with a ``curation`` block (schema
``dropbear-hw-library-curation-v1``): the clips whose reference torque exceeds the arm motors' peak or sustained
rating under the arm-duty screen are excluded or down-weighted. Nothing used to force an ``hw_*`` run onto such a
manifest, so a hardware-twin policy could be trained to track motions the real arm motors cannot produce.

``check_hw_library(...)`` runs before the env is built. For an hardware-armed profile (``hw_*`` and ``legacy_hwarm``)
and a library task it refuses, unless ``allow_uncurated``:

* a manifest without a ``curation`` block, or with another curation schema;
* a curation whose HIGH policy is ``keep`` (every HIGH clip trains unweighted: no curation);
* a manifest whose clips no longer match the recorded decisions (an excluded clip put back, a clip that was never
  screened, a weight that differs from ``weight_out``): the manifest was edited after curation.

A ``--continue_run`` chunk only warns: the run already trained on that library, and the continuity checks
(motion sha, run settings) decide whether it may continue. Single-clip tasks and sim-only profiles are not checked.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

CURATION_SCHEMA = "dropbear-hw-library-curation-v1"
CURATION_TOOL = "tools/curate_hw_library.py"


class UncuratedLibraryError(ValueError):
    """An hardware-armed tracking run would train on a motion library without a valid hardware curation."""


def is_hw_profile(profile: str | None) -> bool:
    """Profiles that put the real arm motors (datasheet armature / envelope) on the plant."""
    p = str(profile or "legacy")
    return p.startswith("hw_") or p == "legacy_hwarm"


def curation_problems(manifest: dict) -> list[str]:
    """Reasons ``manifest`` (a parsed dropbear-motion-library-v1 dict) is not a valid hardware-curated library."""
    cur = manifest.get("curation")
    if not isinstance(cur, dict):
        return [f"no 'curation' block (generate one with {CURATION_TOOL})"]
    if cur.get("schema") != CURATION_SCHEMA:
        return [f"curation schema {cur.get('schema')!r} != {CURATION_SCHEMA!r}"]
    out: list[str] = []
    high = str((cur.get("policy") or {}).get("HIGH", "")).strip().lower()
    if high in ("", "keep"):
        out.append(f"curation policy HIGH={high or None!r} keeps every HIGH clip unweighted")
    decisions = cur.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        return out + ["curation has no per-clip decisions"]
    by_name = {str(d.get("name")): d for d in decisions if isinstance(d, dict)}
    for i, e in enumerate(manifest.get("clips") or []):
        e = {"npz": e} if isinstance(e, str) else (e if isinstance(e, dict) else {})
        name = str(e.get("name") or Path(str(e.get("npz", f"clip{i}"))).stem)
        d = by_name.get(name)
        if d is None:
            out.append(f"clip {name!r} is not in the curation decisions (added after curation)")
            continue
        action = str(d.get("action"))
        if action in ("exclude", "error"):
            out.append(f"clip {name!r} was curated as {action!r} ({d.get('severity')}) but is in the manifest")
            continue
        w_out, w = d.get("weight_out"), float(e.get("weight", 1.0))
        if w_out is not None and not math.isclose(float(w_out), w, rel_tol=1e-9, abs_tol=1e-12):
            out.append(f"clip {name!r} weight {w:g} != curated weight_out {float(w_out):g}")
    return out


def curation_summary(manifest: dict) -> dict[str, Any] | None:
    """The run_info record of the curation (None for an uncurated manifest)."""
    cur = manifest.get("curation")
    if not isinstance(cur, dict):
        return None
    return {k: cur.get(k) for k in ("schema", "source_manifest", "source_manifest_file_sha256", "policy", "counts")}


def check_hw_library(manifest_path: str | Path | None, actuator_profile: str | None, *, allow_uncurated: bool = False,
                     continue_run: bool = False) -> tuple[dict[str, Any], list[str]]:
    """Gate an hardware-armed library run on a curated manifest.

    Returns ``(record, notes)``: ``record`` goes to run_info (``checked``, ``curated``, ``problems``, ``allowed``,
    ``curation``), ``notes`` are printable lines. Raises :class:`UncuratedLibraryError` for a fresh run that fails.
    """
    rec: dict[str, Any] = {"checked": False, "curated": None, "problems": [], "allowed": False, "curation": None}
    if manifest_path is None or not is_hw_profile(actuator_profile):
        return rec, []
    p = Path(manifest_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    problems = curation_problems(data)
    rec.update(checked=True, curated=not problems, problems=problems, curation=curation_summary(data))
    if not problems:
        c = rec["curation"] or {}
        return rec, [f"hw library curation OK: {p.name} from {c.get('source_manifest')!r}, policy {c.get('policy')}, "
                     f"counts {c.get('counts')}"]
    head = f"{p.name} is not a hardware-curated library for actuator profile {actuator_profile!r}: " + "; ".join(problems[:5])
    if len(problems) > 5:
        head += f"; ... {len(problems) - 5} more"
    if allow_uncurated:
        rec["allowed"] = True
        return rec, [f"WARNING {head} (allowed by --allow_uncurated)"]
    if continue_run:
        rec["allowed"] = True
        return rec, [f"WARNING {head} (--continue_run: kept; start new hw runs on a curated manifest)"]
    raise UncuratedLibraryError(f"{head}. Run {CURATION_TOOL} on the manifest and train on its *_hw.json, or pass "
                                "--allow_uncurated")
