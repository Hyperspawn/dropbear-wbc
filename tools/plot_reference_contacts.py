"""Plot a motion NPZ's foot-sole heights against its contact flags (CPU; demo_eval track, 2026-09-24).

For each foot: the lowest sole point (collision convex hulls, ``data/calibration/dropbear_foot_sole_hulls.json``,
the same model as ``tools/validate_motion_npz.py``) over time, with the frames where the NPZ says that foot is in
contact shaded and the +-15 mm contact gate drawn. Where a shaded (contact) span sits above the gate band, the
reference asks the robot to hold its foot in the air while calling it a stance foot -- the "feet float" failure.
Optionally overlays a policy rollout's measured foot heights (``--rollout`` JSON/NPZ with ``t``,
``left_sole_z``, ``right_sole_z``).

    python tools/plot_reference_contacts.py data/motions/unitree_rl_lab_mimic/G1_Take_102.npz \
        --out logs/demo_eval/media/G1_Take_102_reference_contacts.png --out-json logs/demo_eval/G1_Take_102_contacts.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.settle.ground import SoleModel  # noqa: E402
from dropbear_wbc.tasks.tracking.motion_npz import load_motion_npz  # noqa: E402


def spans(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--sole", type=Path, default=REPO / "data/calibration/dropbear_foot_sole_hulls.json")
    ap.add_argument("--gate-mm", type=float, default=15.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = load_motion_npz(args.npz)
    sole = SoleModel.load(args.sole)
    low = sole.lowest_z(np.asarray(m.body_pos_w, np.float64), np.asarray(m.body_quat_w, np.float64), list(m.body_names))
    with np.load(args.npz, allow_pickle=False) as d:
        contact = np.asarray(d["contact"], dtype=bool) if "contact" in d.files else None
    src = "npz contact"
    if contact is None:
        contact = np.stack([low["left"] <= low["right"], low["right"] < low["left"]], -1)
        src = "lower foot"
    t = np.arange(len(low["left"])) / float(m.fps)
    gate = args.gate_mm / 1e3
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True, constrained_layout=True)
    stats = {}
    for ax, (k, side) in zip(axes, enumerate(("left", "right"))):
        z = 1e3 * low[side]
        c = contact[:, k]
        for a, b in spans(c):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color="#9ecae1", alpha=0.35, lw=0)
        viol = c & (np.abs(low[side]) > gate)
        ax.axhspan(-args.gate_mm, args.gate_mm, color="#74c476", alpha=0.25, lw=0)
        ax.plot(t, z, color="#08306b", lw=1.3, label=f"{side} lowest sole point")
        ax.scatter(t[viol], z[viol], s=6, color="#cb181d", zorder=3,
                   label=f"contact frame with |z| > {args.gate_mm:.0f} mm ({100 * viol.mean():.0f} % of frames)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(f"{side} sole z [mm]")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3)
        zc = low[side][c]
        stats[side] = {"contact_frac": float(c.mean()), "violating_frac_of_frames": float(viol.mean()),
                       "contact_z_max_mm": 1e3 * float(zc.max()) if zc.size else None,
                       "contact_z_p95_mm": 1e3 * float(np.percentile(zc, 95)) if zc.size else None,
                       "longest_violation_s": max(((b - a) / float(m.fps) for a, b in spans(viol)), default=0.0)}
    axes[-1].set_xlabel("time [s]")
    axes[0].set_title(args.title or (f"{args.npz.stem}: reference foot heights (blue = frames flagged as contact, "
                                     f"green band = +-{args.gate_mm:.0f} mm contact gate)"), fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    rec = {"npz": str(args.npz).replace("\\", "/"), "contact_source": src, "gate_mm": args.gate_mm,
           "frames": int(len(t)), "fps": float(m.fps), "feet": stats,
           "either_foot_violating_frac": float(((contact[:, 0] & (np.abs(low["left"]) > gate)) |
                                                (contact[:, 1] & (np.abs(low["right"]) > gate))).mean()),
           "plot": str(args.out).replace("\\", "/")}
    if args.out_json:
        args.out_json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
