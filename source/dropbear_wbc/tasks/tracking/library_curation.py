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

Screen ambient (item 367). Since 0101 the arm-duty screen's ``sus`` thresholds depend on the ambient (schema
``dropbear-motion-duty-screen-v2`` with ``ambient_C``; 0.934x rated warn at the 45 C design ambient instead of the
fixed 1.0x). :func:`screen_ambient_status` classifies ``curation.screen``:

* ``ok``: v2 screen at the design ambient (``dropbear_wbc.robots.thermal.DESIGN_AMBIENT_C``);
* ``no_ambient``: a pre-0101 fixed-rated screen (v1 / no ``ambient_C`` / no ``screen`` block);
* ``mismatch``: screened at another ambient than the design ambient;
* ``inconsistent``: ``screen.ambient_C`` differs from ``screen.thermal.ambient_C``;
* ``stale_thermal`` (item 368): the screen's recorded thermal constants (``screen.thermal``: rise at rated, reference
  ambient, split, tau_w / tau_h, trip, max continuous ratio, HIGH trip time) or its ``thresh.sus`` (warn, HIGH) differ
  from what ``dropbear_wbc.robots.thermal`` gives now at the screen's own ambient (a v2 screen made before the
  constants / the HIGH conversion changed). Checked before ``mismatch``; the per-key differences are recorded in
  ``thermal_check``.

Anything but ``ok`` only WARNS by default (an older ``*_hw.json`` on the PC / Brev keeps working until it is
re-curated); ``require_curation_ambient`` (train.py ``--require_curation_ambient``) refuses a fresh run instead.
``--allow_uncurated`` and ``--continue_run`` downgrade that refusal to a warning, as for the other curation checks.
The result goes to run_info ``hw_library_curation.ambient_status``.
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


class CurationAmbientError(UncuratedLibraryError):
    """``require_curation_ambient``: the curation's duty screen is not ambient-aware, not at the design ambient or was
    made with other thermal constants than the current model (item 368)."""


SCREEN_SCHEMA_V2 = "dropbear-motion-duty-screen-v2"
AMBIENT_TOL_C = 1e-6
# screen.thermal keys compared with the current model (item 368) and their absolute tolerance: the constants are
# copied verbatim (float noise only), max_continuous_ratio / thresh.sus are rounded to 1e-4 and sus_high_trip_s to 1e-3
# by the screen, so one unit in the last recorded place is allowed (bisection / platform rounding at a .5 boundary).
THERMAL_PARAM_TOL = {"rise_at_rated_K": 1e-9, "ref_ambient_C": 1e-9, "split_wh": 1e-9, "tau_w_s": 1e-9,
                     "tau_h_s": 1e-9, "t_trip_C": 1e-9, "max_continuous_ratio": 1.5e-4, "sus_high_trip_s": 1.5e-3}
SUS_THRESH_TOL = 1.5e-4


def _design_ambient_C() -> float:
    from dropbear_wbc.robots.thermal import DESIGN_AMBIENT_C

    return float(DESIGN_AMBIENT_C)


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if not isinstance(v, bool) else math.nan


def screen_thermal_check(screen: dict, ambient_C: float) -> dict[str, Any]:
    """Compare a v2 screen's ``thermal`` block and ``thresh.sus`` with the current thermal model at ``ambient_C``.

    Returns ``{"stale": bool, "diffs": [{"key", "saved", "current", "tol"}, ...]}``; a missing or non-numeric saved
    value counts as a difference (``saved`` None / the raw value)."""
    from dropbear_wbc.robots.thermal import duty_screen_thermal, duty_sus_thresholds

    try:
        cur_th = duty_screen_thermal(ambient_C)
        now_sus = duty_sus_thresholds(float(ambient_C))
    except (ValueError, OverflowError) as e:  # an absurd screen ambient (e.g. -273 C: nothing trips in the HIGH time)
        return {"stale": True, "diffs": [{"key": "thermal_model", "saved": ambient_C, "current": math.nan, "tol": None,
                                          "error": str(e)}]}
    th = screen.get("thermal")
    th = th if isinstance(th, dict) else {}
    diffs: list[dict[str, Any]] = []
    for key, tol in THERMAL_PARAM_TOL.items():
        saved, now = th.get(key), float(cur_th[key])
        v = _num(saved)
        if not (math.isfinite(v) and abs(v - now) <= tol):
            diffs.append({"key": f"thermal.{key}", "saved": saved, "current": now, "tol": tol})
    sus = (screen.get("thresh") or {}).get("sus") if isinstance(screen.get("thresh"), dict) else None
    for i, name in enumerate(("warn", "HIGH")):
        saved = sus[i] if isinstance(sus, (list, tuple)) and len(sus) == 2 else None
        v = _num(saved)
        if not (math.isfinite(v) and abs(v - now_sus[i]) <= SUS_THRESH_TOL):
            diffs.append({"key": f"thresh.sus.{name}", "saved": saved, "current": now_sus[i], "tol": SUS_THRESH_TOL})
    return {"stale": bool(diffs), "diffs": diffs}


def screen_ambient_status(manifest: dict, design_ambient_C: float | None = None) -> dict[str, Any] | None:
    """Classify the ambient of the duty screen a curated manifest was made with (item 367).

    Returns ``{"status", "screen_schema", "ambient_C", "design_ambient_C", "detail", "thermal_check"}``; ``status`` is
    ``ok``, ``no_ambient``, ``inconsistent``, ``stale_thermal`` or ``mismatch`` (module docstring; first match wins).
    ``thermal_check`` (:func:`screen_thermal_check`) is None when the ambient is unusable. ``None`` for a manifest
    without curation.
    """
    cur = manifest.get("curation")
    if not isinstance(cur, dict):
        return None
    design = _design_ambient_C() if design_ambient_C is None else float(design_ambient_C)
    scr = cur.get("screen")
    scr = scr if isinstance(scr, dict) else {}
    schema, amb = scr.get("schema"), scr.get("ambient_C")
    out = {"status": "ok", "screen_schema": schema, "ambient_C": amb, "design_ambient_C": design, "detail": "",
           "thermal_check": None}
    if not scr:
        out.update(status="no_ambient", detail="curation has no 'screen' block")
    elif schema != SCREEN_SCHEMA_V2 or amb is None:
        out.update(status="no_ambient", detail=f"screen schema {schema!r}, ambient_C {amb!r}: fixed-rated screen "
                                               "from before the thermal model (0101)")
    else:
        try:
            amb_f = float(amb)
        except (TypeError, ValueError):
            amb_f = math.nan
        th = scr.get("thermal")
        th_amb = th.get("ambient_C") if isinstance(th, dict) else None
        if not math.isfinite(amb_f):
            out.update(status="no_ambient", detail=f"screen ambient_C {amb!r} is not a finite number")
        elif th_amb is not None and abs(float(th_amb) - amb_f) > AMBIENT_TOL_C:
            out.update(status="inconsistent", detail=f"screen ambient_C {amb_f:g} C != screen.thermal.ambient_C "
                                                     f"{float(th_amb):g} C")
        else:
            tc = screen_thermal_check(scr, amb_f)
            out["thermal_check"] = tc
            if tc["stale"]:
                d = tc["diffs"]
                txt = ", ".join(f"{x['key']} {x['saved']!r} (now {x['current']:g})" for x in d[:4])
                more = f", ... {len(d) - 4} more" if len(d) > 4 else ""
                out.update(status="stale_thermal",
                           detail=f"screen thermal constants/thresholds differ from the current thermal model at "
                                  f"{amb_f:g} C: {txt}{more}")
            elif abs(amb_f - design) > AMBIENT_TOL_C:
                out.update(status="mismatch", detail=f"screened at {amb_f:g} C, design ambient is {design:g} C")
    return out


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
                     continue_run: bool = False, require_curation_ambient: bool = False,
                     design_ambient_C: float | None = None) -> tuple[dict[str, Any], list[str]]:
    """Gate an hardware-armed library run on a curated manifest.

    Returns ``(record, notes)``: ``record`` goes to run_info (``checked``, ``curated``, ``problems``, ``allowed``,
    ``curation``, ``ambient_status``), ``notes`` are printable lines. Raises :class:`UncuratedLibraryError` for a fresh
    run that fails, :class:`CurationAmbientError` when ``require_curation_ambient`` and the screen ambient is not ``ok``.
    """
    rec: dict[str, Any] = {"checked": False, "curated": None, "problems": [], "allowed": False, "curation": None,
                           "ambient_status": None}
    if manifest_path is None or not is_hw_profile(actuator_profile):
        return rec, []
    p = Path(manifest_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    problems = curation_problems(data)
    rec.update(checked=True, curated=not problems, problems=problems, curation=curation_summary(data))
    amb = screen_ambient_status(data, design_ambient_C)
    if amb is not None:
        amb["required"] = bool(require_curation_ambient)
    rec["ambient_status"] = amb
    if not problems:
        c = rec["curation"] or {}
        notes = [f"hw library curation OK: {p.name} from {c.get('source_manifest')!r}, policy {c.get('policy')}, "
                 f"counts {c.get('counts')}"]
        if amb is None or amb["status"] == "ok":
            return rec, notes
        why = ("sus thresholds come from older thermal constants" if amb["status"] == "stale_thermal" else
               f"sus thresholds differ from the {amb['design_ambient_C']:g} C design ambient")
        head = f"{p.name} curation screen ambient {amb['status']}: {amb['detail']} ({why}; re-run {CURATION_TOOL})"
        if not require_curation_ambient:
            return rec, notes + [f"WARNING {head}"]
        if allow_uncurated:
            rec["allowed"] = True
            return rec, notes + [f"WARNING {head} (--require_curation_ambient overridden by --allow_uncurated)"]
        if continue_run:
            rec["allowed"] = True
            return rec, notes + [f"WARNING {head} (--continue_run: kept)"]
        raise CurationAmbientError(f"{head}. --require_curation_ambient refuses it; re-curate, drop the flag or pass "
                                   "--allow_uncurated")
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
