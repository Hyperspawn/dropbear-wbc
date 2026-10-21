"""Push-case plan and summary of ``scripts/robustness_bench.py --mode push``. Pure Python (no Isaac / torch).

``repeats`` > 1 runs every (phase, direction, dv, offset) cell several times with the push moved by
``k * jitter_s`` (k = 0 .. repeats-1). A single deterministic case per cell flipped survive/fall in ~12 % of cells
between otherwise identical runs, so the per-cell outcome is reported as a fraction and the mixed cells are counted.
"""
from __future__ import annotations

from typing import Iterable, Sequence


def push_cases(phases: Sequence[tuple[str, tuple[float, float, float]]], directions: int, dvs: Sequence[float],
               offsets: Sequence[float], repeats: int = 1, jitter_s: float = 0.02) -> list[dict]:
    """One dict per env. ``offset_s`` is the push delay actually used; ``base_offset_s`` the requested one."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if repeats > 1 and jitter_s <= 0.0:
        raise ValueError("repeats > 1 needs jitter_s > 0 (identical cases add no information)")
    cases = []
    for ph, c in phases:
        for d in range(directions):
            for dv in dvs:
                for o in offsets:
                    for r in range(repeats):
                        cases.append({"phase": ph, "cmd": c, "dir_deg": 360.0 * d / directions, "dv": dv,
                                      "base_offset_s": o, "repeat": r, "offset_s": round(o + r * jitter_s, 6)})
    return cases


def case_key(c: dict) -> str:
    """``--only`` key 'phase:dir_deg:dv:offset_s' (the requested offset: every repeat of a cell matches)."""
    return f"{c['phase']}:{c['dir_deg']:g}:{c['dv']:g}:{c.get('base_offset_s', c['offset_s']):g}"


def push_summary(results: Iterable[dict], phases: Sequence[str], directions: int) -> dict:
    """Per phase: ``max_dv_survived_by_dir`` (every case of every lower dv survived, as before), ``max_dv_majority_by_dir``
    (>= half of the cases survived at every dv up to it), ``worst_dir_dv``, ``survival_rate`` and, over the
    (dir, dv, offset) cells, ``mixed_cells`` / ``mixed_cell_rate`` (cells whose repeats disagree)."""
    results = list(results)
    summary: dict = {}
    for ph in phases:
        rows_ph = [r for r in results if r["phase"] == ph]
        dvs = sorted({r["dv"] for r in rows_ph})
        strict, majority = {}, {}
        for d in range(directions):
            deg = 360.0 * d / directions
            best_s = best_m = 0.0
            ok_s = ok_m = True
            for dv in dvs:
                rows = [r for r in rows_ph if r["dir_deg"] == deg and r["dv"] == dv]
                if not rows:
                    break
                frac = sum(r["survived"] for r in rows) / len(rows)
                ok_s = ok_s and frac == 1.0
                ok_m = ok_m and frac >= 0.5
                best_s = dv if ok_s else best_s
                best_m = dv if ok_m else best_m
            strict[f"{deg:g}"] = best_s
            majority[f"{deg:g}"] = best_m
        cells: dict = {}
        for r in rows_ph:
            cells.setdefault((r["dir_deg"], r["dv"], r.get("base_offset_s", r["offset_s"])), []).append(r["survived"])
        mixed = sum(1 for v in cells.values() if 0 < sum(v) < len(v))
        summary[ph] = {"max_dv_survived_by_dir": strict, "worst_dir_dv": min(strict.values()) if strict else 0.0,
                       "max_dv_majority_by_dir": majority,
                       "worst_dir_dv_majority": min(majority.values()) if majority else 0.0,
                       "survival_rate": sum(r["survived"] for r in rows_ph) / max(1, len(rows_ph)),
                       "cells": len(cells), "mixed_cells": mixed, "mixed_cell_rate": mixed / max(1, len(cells))}
    return summary
