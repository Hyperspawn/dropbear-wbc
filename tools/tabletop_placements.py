"""Generate the tabletop placement lists (train / held-out) with IK-verified push paths (CPU).

Each candidate (arm side, zone centre, block centre at 8.5-13 cm from it, block yaw +-45 deg; ``layout.candidate``)
is kept only if the scripted policy's whole path is kinematically feasible for that arm with the teleop IK on the
hand-tool chain (:func:`scripted.plan_check_points`): lift waypoint, pre-push point (hover and push height), the push
line every 2 cm, the push end, retreat and retreat hover. Per point: tool residual < 2.5 mm, hand lowest point within
3 mm of the request, elbow >= 5 cm above the table at push height, tool in front of the torso for table points, no
shoulder pitch / roll / elbow joint at its limit. Block and zone must lie on the table (margins 4 / 6 cm).

The two splits use disjoint seed streams (``layout.SPLIT_SEED_BASE``); placement ``k`` of a split is the first feasible
candidate of ``np.random.default_rng(base + k)`` sequences, so the lists are reproducible and never share a draw.

    python tools/tabletop_placements.py --train 400 --heldout 50 \
        --out data/groot/placements/tabletop_push_v1.json > logs/tabletop/placements_v1.log
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.tasks.tabletop.kinematics import HAND_RADIUS_M, TabletopArmIK  # noqa: E402
from dropbear_wbc.tasks.tabletop.layout import (  # noqa: E402
    BLOCK_DIST,
    LAYOUT,
    SPLIT_SEED_BASE,
    ZONE_REGION_LEFT,
    Placement,
    candidate,
    save_placements,
)
from dropbear_wbc.tasks.tabletop.scripted import PushParams, ScriptedPushPolicy, plan_check_points  # noqa: E402

TORSO_FRONT_X = 0.1405  # max torso collider x for root z 1.20-1.45 (logs/tabletop/geometry_probe.log)


def check_point(ik: TabletopArmIK, side: str, name: str, xy, clear: float, axis_pref) -> tuple[bool, dict]:
    zt = LAYOUT.table_top_z
    res, hp = ik.solve_lowest(side, np.asarray(xy, float), zt + clear, axis_pref, restarts="full")
    ch = ik.chains[side]
    el = ik.elbow_root(side, res.q)
    at_lim = ((res.q <= ch.lower + 1e-3) | (res.q >= ch.upper - 1e-3))[[0, 1, 3]]
    rec = {"err_mm": round(1e3 * res.pos_err_m, 2), "lowest_err_mm": round(1e3 * (hp.lowest_z - zt - clear), 2),
           "tilt_deg": round(math.degrees(hp.tilt_rad), 1), "elbow_clear_m": round(float(el[2] - zt), 3)}
    ok = res.pos_err_m < 0.0025 and abs(hp.lowest_z - zt - clear) < 0.003 and not at_lim.any()
    if name == "lift":
        # the lift waypoint may be short of hover height only while the hand is still behind the table edge
        ok = res.pos_err_m < 0.01 and (hp.lowest_z > zt + 0.02 or hp.tip[0] + HAND_RADIUS_M < LAYOUT.table_front_x)
    else:
        ok = ok and hp.tool[0] > TORSO_FRONT_X + HAND_RADIUS_M
        if clear < 0.03:
            ok = ok and el[2] - zt >= 0.05
    return ok, rec


def feasible(ik: TabletopArmIK, pol: ScriptedPushPolicy, cand: dict) -> tuple[bool, dict]:
    side = cand["side"]
    if not (LAYOUT.on_table(cand["block_xy"], 0.04) and LAYOUT.on_table(cand["zone_xy"], 0.06)):
        return False, {"reason": "off_table"}
    ik.q_prev[side] = ik.chains[side].q_rest.copy()
    checks = {}
    # warm-start along the path, as the policy does
    for name, xy, clear in plan_check_points(ik, pol, side, cand["block_xy"], cand["block_yaw"], cand["zone_xy"]):
        ok, rec = check_point(ik, side, name, xy, clear, pol.axis_pref)
        checks[name] = rec
        if not ok:
            return False, {"reason": f"ik:{name}", "checks": checks}
    return True, {"checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--heldout", type=int, default=50)
    ap.add_argument("--out", type=Path, default=REPO / "data/groot/placements/tabletop_push_v1.json")
    ap.add_argument("--max_tries", type=int, default=200)
    args = ap.parse_args()
    ik = TabletopArmIK()
    pol = ScriptedPushPolicy(ik)
    t0 = time.perf_counter()
    placements: list[Placement] = []
    stats = {}
    for split, n in (("heldout", args.heldout), ("train", args.train)):
        rej: dict[str, int] = {}
        for k in range(n):
            rng = np.random.default_rng(SPLIT_SEED_BASE[split] + k)
            for tries in range(1, args.max_tries + 1):
                c = candidate(rng)
                ok, info = feasible(ik, pol, c)
                if ok:
                    placements.append(Placement(
                        pid=f"{split}_{k:06d}", split=split, side=c["side"],
                        block_xy=tuple(round(float(v), 5) for v in c["block_xy"]), block_yaw=round(float(c["block_yaw"]), 5),
                        zone_xy=tuple(round(float(v), 5) for v in c["zone_xy"]), zone_yaw=0.0,
                        seed=SPLIT_SEED_BASE[split] + k, checks={"tries": tries}))
                    break
                r = info["reason"].split(":")[0] if info["reason"] == "off_table" else info["reason"]
                rej[r] = rej.get(r, 0) + 1
            else:
                raise RuntimeError(f"{split} {k}: no feasible candidate in {args.max_tries} tries")
            if (k + 1) % 25 == 0:
                print(f"{split}: {k + 1}/{n} ({time.perf_counter() - t0:.0f} s)", flush=True)
        stats[split] = {"n": n, "rejections": dict(sorted(rej.items(), key=lambda kv: -kv[1]))}
        print(split, "rejections:", stats[split]["rejections"], flush=True)
    sides = {s: {sp: sum(1 for p in placements if p.split == sp and p.side == s) for sp in ("train", "heldout")}
             for s in ("left", "right")}
    calib = ik.base.path
    meta = {"created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "tool": "tools/tabletop_placements.py", "layout": LAYOUT.to_dict(), "push_params": PushParams().to_dict(),
            "zone_region_left": ZONE_REGION_LEFT, "block_dist": BLOCK_DIST, "split_seed_base": SPLIT_SEED_BASE,
            "calibration": str(calib), "calibration_sha256": hashlib.sha256(Path(calib).read_bytes()).hexdigest(),
            "tool_d_m": ik.tool_d, "torso_front_x": TORSO_FRONT_X, "stats": stats, "sides": sides,
            "frame": "robot root frame (fixed 'world' body); env frame = root + layout.root_pos_w"}
    save_placements(args.out, placements, meta)
    print("sides:", sides)
    print(f"wrote {args.out} ({len(placements)} placements, {time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
