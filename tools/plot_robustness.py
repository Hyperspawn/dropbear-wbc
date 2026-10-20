"""Charts for ``scripts/robustness_bench.py``: a push survival map (strongest push survived from each direction, per
phase) and a terrain grid (survived / progress per terrain type and difficulty band).

    python tools/plot_robustness.py --push logs/bench/push.json --terrain logs/bench/terrain.json --out logs/reports/bench.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
PHASE_COL = ["#2a78d6", "#eb6834", "#1f9e6e", "#8a5cc2"]
TERRAIN_LABEL = {"flat": "Flat", "rough": "Rough blocks", "uphill": "Uphill", "downhill": "Downhill",
                 "stairs_up": "Stairs up", "stairs_down": "Stairs down"}


def _style() -> None:
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 12.5,
                         "axes.titleweight": "semibold", "figure.facecolor": SURF, "axes.facecolor": SURF})


def push_panel(ax, res: dict) -> None:
    dvs = sorted({c["dv"] for c in res["cases"]})
    for k, (phase, s) in enumerate(res["summary"].items()):
        by = s["max_dv_survived_by_dir"]
        degs = sorted(float(d) for d in by)
        # bench angle = where the push comes from (0 = behind, 90 = right); plot it there with the robot facing up
        th = [math.radians(d) + 1.5 * math.pi for d in degs] + [math.radians(degs[0]) + 1.5 * math.pi]
        r = [by[f"{d:g}"] for d in degs] + [by[f"{degs[0]:g}"]]
        col = PHASE_COL[k % len(PHASE_COL)]
        ax.plot(th, r, color=col, lw=2.2, label=f"{phase}: weakest side {s['worst_dir_dv']:.1f} m/s")
        ax.fill(th, r, color=col, alpha=0.12)
    ax.set_ylim(0, max(dvs))
    ax.set_yticks(dvs, [f"{v:g}" for v in dvs], fontsize=8.5, color=INK2)
    ax.set_xticks([math.radians(a) for a in (90, 270, 0, 180)], ["from front", "from behind", "from right", "from left"])
    ax.set_title("Strongest push survived, by direction\n(whole-body speed change, m/s)", loc="left")
    ax.grid(color=GRID)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.04), frameon=False, fontsize=10)


def terrain_status(row: dict, goal_m: float) -> int:
    """0 = fell, 1 = stalled (no fall, but under half the commanded distance), 2 = walked."""
    if not row["survived"]:
        return 0
    return 2 if row["progress_m"] >= 0.5 * goal_m else 1


def terrain_panel(ax, res: dict) -> None:
    names = list(res["summary"])
    levels = max(len(v) for v in res["summary"].values())
    a = res.get("args", {})
    goal = float(a.get("speed", 0.5)) * float(a.get("walk_s", 8.0))
    grid = np.full((len(names), levels), np.nan)
    for i, n in enumerate(names):
        for row in res["summary"][n]:
            grid[i, row["level"]] = terrain_status(row, goal)
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#f0c9c3", "#f5e3b8", "#cfe3f5"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i, n in enumerate(names):
        for row in res["summary"][n]:
            band = row.get("band")
            if band and "stairs" in n:
                txt = f"{band[0] * 100:.0f}–{band[1] * 100:.0f} cm"
            elif band and "hill" in n:
                txt = f"{math.degrees(math.atan(band[0])):.0f}–{math.degrees(math.atan(band[1])):.0f}°"
            elif band:
                txt = f"≤{band[1] * 100:.0f} cm"
            else:
                txt = ""
            st = terrain_status(row, goal)
            ax.text(row["level"], i - 0.13, ("fell", "stalled", "walked")[st], ha="center", va="center",
                    fontsize=9.5, fontweight="semibold", color=("#8c2a1e", "#7a5a12", "#1d4f80")[st])
            ax.text(row["level"], i + 0.2, f"{txt}  {row['progress_m']:.1f} m".strip(), ha="center", va="center",
                    fontsize=8, color=INK2)
    ax.set_yticks(range(len(names)), [TERRAIN_LABEL.get(n, n) for n in names])
    ax.set_xticks(range(levels), [f"level {i + 1}" for i in range(levels)], fontsize=9.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Terrain: walk {goal:g} m from each tile's centre (distance walked; stalled = under half)", loc="left")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--push", type=Path, default=None)
    ap.add_argument("--terrain", type=Path, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    _style()
    push = json.loads(a.push.read_text(encoding="utf-8")) if a.push else None
    terr = json.loads(a.terrain.read_text(encoding="utf-8")) if a.terrain else None
    n = (push is not None) + (terr is not None)
    fig = plt.figure(figsize=(7.5 * n + (3 if terr else 0), 7.2), constrained_layout=True)
    gs = fig.add_gridspec(1, n, width_ratios=([1] if push else []) + ([1.6] if terr else []))
    k = 0
    if push:
        push_panel(fig.add_subplot(gs[0, k], projection="polar"), push)
        k += 1
    if terr:
        terrain_panel(fig.add_subplot(gs[0, k]), terr)
    src = push or terr
    fig.suptitle(a.title or f"Robustness benchmark  ·  {Path(src['checkpoint']).parent.name} / {Path(src['checkpoint']).stem}"
                 f"  ·  {src['actuator_profile']}  ·  simulation", x=0.01, ha="left", fontsize=13.5, fontweight="semibold")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
