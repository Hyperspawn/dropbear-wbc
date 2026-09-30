"""Reachable tabletop workspace of Dropbear's hands, measured with the teleop arm IK (tool-point variant).

For each candidate table height and preferred hand-axis direction, every (x, y) grid point of the table plane
(ROOT frame, i.e. the fixed ``world`` body frame) is solved with :class:`TabletopArmIK.solve_lowest` (hand's lowest
point 8 mm above the table, the push height). A point counts as PUSHABLE for an arm when

* the tool-point residual is < 2 mm and the hand's lowest point is within 3 mm of the request,
* the hand axis is within ``--max_tilt_deg`` of vertical (default 92 deg: a horizontal hand pushes with its side or
  end face; it must not point up),
* the elbow axis point is >= 6 cm above the table (upper arm / forearm clear of the table),
* the tool point is in front of the torso (x > torso front + hand radius; torso front from the geometry probe),
* the solution is not at a semantic joint limit except wrist roll (margin for the push motion).

It also checks the HOVER height (lowest point 8 cm higher) at every pushable point. History: a first run with max tilt 45 deg, elbow clearance 8 cm and a vertical / 30 deg hand preference found only
0.01-0.03 m^2 per arm (``workspace_v0_strict_criteria.log``): a vertical forearm in front of the body needs ~150 deg of
shoulder yaw (at the 2.6 rad limit). Diagnosis per cell: the inner boundary is the shoulder-roll limit (-10 deg
adduction, ``LH_pitch`` motor limit), so each hand works only in front of / outside its own shoulder.

Output: JSON with per-arm
boolean grids, areas and the chosen layout recommendation, plus a PNG heat map (matplotlib if available).

    .venv-teleop/Scripts/python.exe tools/tabletop_workspace.py --out logs/tabletop/workspace.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.tasks.tabletop.kinematics import HAND_RADIUS_M, TabletopArmIK  # noqa: E402

PUSH_CLEAR_M = 0.008
HOVER_EXTRA_M = 0.08


def torso_front_x(probe: dict, z_lo: float, z_hi: float) -> float:
    xs = [b["max_x"] for b in probe["torso_front_extent"] if b["max_x"] is not None and b["z_hi"] > z_lo
          and b["z_lo"] < z_hi]
    return float(max(xs)) if xs else 0.0


def scan(ik: TabletopArmIK, side: str, z_table: float, axis_pref, xs, ys, front_x, max_tilt_deg) -> dict:
    ok = np.zeros((len(xs), len(ys)), dtype=bool)
    hover_ok = np.zeros_like(ok)
    tilt = np.full(ok.shape, np.nan)
    elbow_clear = np.full(ok.shape, np.nan)
    ch = ik.chains[side]
    lim_lo, lim_hi = ch.lower, ch.upper
    q_row = None
    for i, x in enumerate(xs):
        order = range(len(ys)) if i % 2 == 0 else range(len(ys) - 1, -1, -1)  # serpentine: warm starts stay close
        q_warm = q_row
        for j in order:
            y = ys[j]
            res, hp = ik.solve_lowest(side, np.array([x, y]), z_table + PUSH_CLEAR_M, axis_pref, q_init=q_warm,
                                      restarts="light")
            q_warm = res.q
            if j == (0 if i % 2 == 0 else len(ys) - 1):
                q_row = res.q
            el = ik.elbow_root(side, res.q)
            tilt[i, j] = np.degrees(hp.tilt_rad)
            elbow_clear[i, j] = el[2] - z_table
            at_lim = ((res.q <= lim_lo + 1e-3) | (res.q >= lim_hi - 1e-3))[:4]
            good = (res.pos_err_m < 0.002 and abs(hp.lowest_z - (z_table + PUSH_CLEAR_M)) < 0.003
                    and tilt[i, j] <= max_tilt_deg and elbow_clear[i, j] >= 0.06
                    and hp.tool[0] > front_x + HAND_RADIUS_M and not at_lim.any())
            ok[i, j] = good
            if good:
                r2, hp2 = ik.solve_lowest(side, np.array([x, y]), z_table + PUSH_CLEAR_M + HOVER_EXTRA_M, axis_pref,
                                          q_init=res.q, restarts="light")
                hover_ok[i, j] = r2.pos_err_m < 0.003 and abs(hp2.lowest_z - (z_table + PUSH_CLEAR_M + HOVER_EXTRA_M)) < 0.005
                ik.q_prev[side] = res.q.copy()
    return {"ok": ok, "hover_ok": hover_ok, "tilt_deg": tilt, "elbow_clear_m": elbow_clear}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "logs/tabletop/workspace.json")
    ap.add_argument("--probe", type=Path, default=REPO / "logs/tabletop/geometry_probe.json")
    ap.add_argument("--heights", type=float, nargs="+", default=[1.20, 1.25, 1.30],
                    help="table-top heights in the ROOT frame [m]")
    ap.add_argument("--forward_tilts_deg", type=float, nargs="+", default=[60.0, 75.0],
                    help="preferred hand axis: down, tilted forward by this angle (a soft, null-space objective)")
    ap.add_argument("--step", type=float, default=0.025)
    ap.add_argument("--max_tilt_deg", type=float, default=92.0)
    args = ap.parse_args()

    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    ik = TabletopArmIK()
    xs = np.round(np.arange(0.125, 0.5001, args.step), 4)
    ys = np.round(np.arange(-0.60, 0.4501, args.step), 4)
    out = {"schema": "dropbear-tabletop-workspace-v1", "frame": "root (fixed world body) frame [m]",
           "calibration": ik.base.info()["calibration"], "calibration_sha256": ik.base.sha256,
           "tool_d_m": ik.tool_d, "push_clearance_m": PUSH_CLEAR_M, "hover_extra_m": HOVER_EXTRA_M,
           "max_tilt_deg": args.max_tilt_deg, "xs": xs.tolist(), "ys": ys.tolist(),
           "shoulders_root": {s: ik.chains[s].shoulder.tolist() for s in ("left", "right")},
           "torso_origin_root": ik.base.torso_origin_root.tolist(), "configs": []}
    cell = args.step ** 2
    for zt in args.heights:
        front = torso_front_x(probe, zt - 0.05, zt + 0.25)
        for tdeg in args.forward_tilts_deg:
            a = np.array([np.sin(np.radians(tdeg)), 0.0, -np.cos(np.radians(tdeg))])
            t0 = time.perf_counter()
            per = {}
            for side in ("left", "right"):
                ik.q_prev[side] = ik.chains[side].q_rest.copy()
                per[side] = scan(ik, side, zt, a, xs, ys, front, args.max_tilt_deg)
            both = per["left"]["ok"] & per["left"]["hover_ok"], per["right"]["ok"] & per["right"]["hover_ok"]
            union = both[0] | both[1]
            cfg = {"z_table_root": zt, "forward_tilt_deg": tdeg, "torso_front_x": front,
                   "seconds": round(time.perf_counter() - t0, 1),
                   "area_m2": {"left": float(both[0].sum() * cell), "right": float(both[1].sum() * cell),
                               "union": float(union.sum() * cell), "overlap": float((both[0] & both[1]).sum() * cell)}}
            for side, b in zip(("left", "right"), both):
                if b.any():
                    ii, jj = np.nonzero(b)
                    cfg[f"{side}_bbox"] = {"x": [float(xs[ii].min()), float(xs[ii].max())],
                                           "y": [float(ys[jj].min()), float(ys[jj].max())]}
                    cfg[f"{side}_tilt_deg_p50"] = float(np.nanmedian(per[side]["tilt_deg"][b]))
                cfg[f"{side}_ok"] = b.astype(int).tolist()
            out["configs"].append(cfg)
            print(f"z_table {zt:.3f} tilt {tdeg:4.1f}: area L {cfg['area_m2']['left']:.4f} R {cfg['area_m2']['right']:.4f} "
                  f"union {cfg['area_m2']['union']:.4f} overlap {cfg['area_m2']['overlap']:.4f} m^2 "
                  f"L bbox {cfg.get('left_bbox')} R bbox {cfg.get('right_bbox')} ({cfg['seconds']} s)", flush=True)
    best = max(out["configs"], key=lambda c: c["area_m2"]["union"])
    out["best_union"] = {k: best[k] for k in ("z_table_root", "forward_tilt_deg", "area_m2")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print("best union:", out["best_union"])
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(out["configs"])
        fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 4.2), squeeze=False)
        for k, c in enumerate(out["configs"]):
            img = np.array(c["left_ok"]) * 1 + np.array(c["right_ok"]) * 2
            axs[0, k].imshow(img, origin="lower", extent=[ys[0], ys[-1], xs[0], xs[-1]], cmap="viridis", vmin=0, vmax=3)
            axs[0, k].set_title(f"z {c['z_table_root']:.2f} tilt {c['forward_tilt_deg']:.0f}", fontsize=8)
            axs[0, k].set_xlabel("y (root) [m]")
            axs[0, k].invert_xaxis()
        axs[0, 0].set_ylabel("x (root, forward) [m]")
        fig.tight_layout()
        fig.savefig(args.out.with_suffix(".png"), dpi=110)
        print("wrote", args.out.with_suffix(".png"))
    except ImportError:
        print("matplotlib not available: no PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
