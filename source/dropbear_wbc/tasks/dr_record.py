"""Record of the event / domain-randomization terms a run trains with (backlog item 190). Pure Python: no Isaac/torch.

``run_info.json`` recorded the actuator profile, plant mass and reward settings, but not the EventCfg: mass DR
(``add_base_mass``, ``scale_limb_mass``, ``motor_mass``), base CoM, friction, default-pos noise and pushes were
invisible. A run started before the mass-DR terms existed and resumed with ``--continue_run`` afterwards silently
trained its later chunks on a different plant, and ``run_settings.audit_run_dir`` could not see it.

* :func:`dr_event_record` -- JSON snapshot of every active term of an EventCfg, in declaration order (= the order
  the EventManager applies startup terms, which matters since 0078: ``motor_mass`` must run first).
* :func:`check_dr_continuity` -- for a ``--continue_run`` chunk: refuse a changed record unless the change is
  explicitly allowed; a previous chunk without a record is reported as UNRECORDED (cannot be verified).
* :func:`dr_drift` -- the drift record ``audit_run_dir`` emits between two chunks.
"""
from __future__ import annotations

import json
import math

DR_RECORD_KEY = "dr_events"
DR_RECORD_VERSION = 1
UNRECORDED = "UNRECORDED"
_ENTITY_FIELDS = ("name", "joint_names", "body_names", "fixed_tendon_names", "object_collection_names",
                  "preserve_order")
_TERM_FIELDS = ("interval_range_s", "is_global_time", "min_step_count_between_reset")


def _qualname(obj) -> str:
    mod = getattr(obj, "__module__", None) or type(obj).__module__
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or type(obj).__qualname__
    return f"{mod}:{name}"


def _jsonable(v):
    """JSON-native form of an event parameter (SceneEntityCfg -> its selection, callables -> module:qualname)."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else repr(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, slice):
        return {"slice": [v.start, v.stop, v.step]}
    if hasattr(v, "name") and hasattr(v, "joint_names") and hasattr(v, "body_names"):  # SceneEntityCfg
        return {"entity": {f: _jsonable(getattr(v, f)) for f in _ENTITY_FIELDS if hasattr(v, f)}}
    if callable(v):
        return {"callable": _qualname(v)}
    if hasattr(v, "tolist"):  # numpy / torch
        return _jsonable(v.tolist())
    if hasattr(v, "item") and not hasattr(v, "__len__"):
        return _jsonable(v.item())
    return {"repr": repr(v)}


def _is_term(v) -> bool:
    return v is not None and hasattr(v, "func") and hasattr(v, "mode")


def dr_event_record(events_cfg) -> dict:
    """``{"version", "terms": [{"name", "mode", "func", "params", ...}]}`` for every non-None term of ``events_cfg``,
    in attribute (declaration) order. ``None`` -> an empty record (task without events)."""
    terms = []
    items = list(vars(events_cfg).items()) if events_cfg is not None else []
    for name, term in items:
        if name.startswith("_") or not _is_term(term):
            continue
        rec = {"name": name, "mode": str(term.mode), "func": _qualname(term.func),
               "params": _jsonable(dict(getattr(term, "params", None) or {}))}
        for f in _TERM_FIELDS:
            if getattr(term, f, None) is not None:
                rec[f] = _jsonable(getattr(term, f))
        terms.append(rec)
    return {"version": DR_RECORD_VERSION, "terms": terms}


def _norm(rec):
    return json.loads(json.dumps(rec, sort_keys=True))


def dr_record_diff(before: dict, after: dict) -> list[str]:
    """Human-readable differences between two records (empty = same terms, params and order)."""
    a, b = _norm(before), _norm(after)
    if a == b:
        return []
    out: list[str] = []
    ta = {t["name"]: t for t in a.get("terms", [])}
    tb = {t["name"]: t for t in b.get("terms", [])}
    for n in ta.keys() - tb.keys():
        out.append(f"{n}: removed")
    for n in tb.keys() - ta.keys():
        out.append(f"{n}: added ({tb[n]['mode']}, {tb[n]['func']})")
    for n in sorted(ta.keys() & tb.keys()):
        for k in sorted(set(ta[n]) | set(tb[n])):
            va, vb = ta[n].get(k), tb[n].get(k)
            if va == vb:
                continue
            if isinstance(va, dict) and isinstance(vb, dict):  # params: one line per changed parameter
                for pk in sorted(set(va) | set(vb)):
                    if va.get(pk) != vb.get(pk):
                        out.append(f"{n}.{k}.{pk}: {json.dumps(va.get(pk), sort_keys=True)} -> "
                                   f"{json.dumps(vb.get(pk), sort_keys=True)}")
            else:
                out.append(f"{n}.{k}: {json.dumps(va, sort_keys=True)} -> {json.dumps(vb, sort_keys=True)}")
    common = [n for n in (t["name"] for t in a.get("terms", [])) if n in tb]
    common_b = [n for n in (t["name"] for t in b.get("terms", [])) if n in ta]
    if common != common_b:
        out.append(f"order: {common} -> {common_b}")
    if a.get("version") != b.get("version"):
        out.append(f"version: {a.get('version')} -> {b.get('version')}")
    return sorted(out) if out else ["record differs"]


def recorded_dr(info: dict):
    """The record a run_info carries, or :data:`UNRECORDED` (written before item 190)."""
    return info.get(DR_RECORD_KEY, UNRECORDED) if info is not None else UNRECORDED


def check_dr_continuity(prev_info: dict | None, current: dict, *, continue_run: bool,
                        allow_change: bool = False) -> list[str]:
    """Notes for this chunk. A ``--continue_run`` chunk whose record differs from the run's latest recorded one raises
    ``ValueError`` unless ``allow_change`` (then noted as CHANGED, which the drift audit treats as acknowledged)."""
    if not continue_run or prev_info is None:
        return []
    prev = recorded_dr(prev_info)
    if prev == UNRECORDED:
        return [f"{DR_RECORD_KEY}: {UNRECORDED} in the run's previous chunk (pre-item-190 run_info); this chunk's "
                f"{len(current['terms'])} event terms cannot be verified against it"]
    diff = dr_record_diff(prev, current)
    if not diff:
        return []
    if not allow_change:
        raise ValueError(f"--continue_run: event/DR terms differ from the run's recorded ones: {'; '.join(diff)}. "
                         f"Pass --allow_settings_change to change the plant randomization mid-run")
    return [f"{DR_RECORD_KEY}: CHANGED {'; '.join(diff)} (--allow_settings_change)"]


def dr_drift(prev_info: dict, cur_info: dict) -> dict | None:
    """Drift between consecutive chunks: None if both records match or neither chunk has one. A chunk with a record
    after one without is drift with ``before`` = UNRECORDED (fail closed: the earlier plant randomization is unknown)."""
    a, b = recorded_dr(prev_info), recorded_dr(cur_info)
    if a == UNRECORDED and b == UNRECORDED:
        return None
    if a == UNRECORDED or b == UNRECORDED:
        return {"key": DR_RECORD_KEY, "before": a if a == UNRECORDED else f"{len(a.get('terms', []))} terms",
                "after": b if b == UNRECORDED else f"{len(b.get('terms', []))} terms", "unverifiable": True}
    diff = dr_record_diff(a, b)
    return {"key": DR_RECORD_KEY, "before": "recorded", "after": diff} if diff else None
