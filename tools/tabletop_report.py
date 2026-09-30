"""Summarize tabletop collection runs (``scripts/tabletop_collect.py --summary`` JSONs): success rate with a Wilson 95 %
interval, per arm side, terminations, time to success, and the failures (final block-to-zone distance, last phase,
re-approaches). CPU, stdlib + numpy.

    python tools/tabletop_report.py logs/tabletop/baseline_heldout_v1.json [more.json ...] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def report(paths: list[Path]) -> dict:
    res, prov = [], []
    for p in paths:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        res += s["results"]
        prov.append({"summary": str(p), "split": s["provenance"]["split"], "policy": s["provenance"]["policy"],
                     "episodes": s["episodes"], "incomplete": s.get("incomplete"), "wall_s": s.get("wall_s"),
                     "step_wall_ms": s.get("step_wall_ms"), "blank_frames": s.get("blank_frames")})
    pids = [r["pid"] for r in res]
    dup = sorted({p for p in pids if pids.count(p) > 1})
    n, k = len(res), sum(r["success"] for r in res)
    out = {"runs": prov, "episodes": n, "successes": k, "success_rate": k / n if n else None,
           "wilson95": wilson(k, n), "duplicate_pids": dup}
    out["by_side"] = {}
    for side in ("left", "right"):
        rs = [r for r in res if r["side"] == side]
        ks = sum(r["success"] for r in rs)
        out["by_side"][side] = {"n": len(rs), "success": ks, "rate": ks / len(rs) if rs else None}
    out["terminations"] = {}
    for r in res:
        out["terminations"][r["termination"]] = out["terminations"].get(r["termination"], 0) + 1
    ts = [r["episode_s"] for r in res if r["success"]]
    out["time_to_success_s"] = ({"mean": float(np.mean(ts)), "p50": float(np.median(ts)), "max": float(np.max(ts))}
                                if ts else None)
    out["failures"] = [{"pid": r["pid"], "side": r["side"], "termination": r["termination"],
                        "initial_d_m": round(r["initial_block_to_zone_m"], 3),
                        "final_d_m": round(r["final_block_to_zone_m"], 3), "final_phase": r.get("final_phase"),
                        "n_reapproach": r.get("n_reapproach")} for r in res if not r["success"]]
    fd = [r["final_block_to_zone_m"] for r in res]
    out["final_block_to_zone_m"] = {"p50": float(np.median(fd)), "p90": float(np.percentile(fd, 90))} if fd else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rep = report(a.summaries)
    lo, hi = rep["wilson95"]
    print(f"{rep['successes']}/{rep['episodes']} success = {100 * (rep['success_rate'] or 0):.1f} % "
          f"(Wilson 95 % {100 * lo:.1f}-{100 * hi:.1f} %); by side {json.dumps(rep['by_side'])}; "
          f"terminations {rep['terminations']}; time to success {rep['time_to_success_s']}")
    for f in rep["failures"]:
        print("  FAIL", json.dumps(f))
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
