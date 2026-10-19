"""Did the robot actually DO the motion? Reference vs simulated rollout, per clip.

"No fall" is not enough: on 2026-10-16 a tracker that held 99 of 138 clips without falling turned out to keep both
feet planted through jumps, kicks, one-leg balance and squats (docs/RESULTS.md). This compares the features a viewer
sees, between the reference motion and the recorded rollout (``scripts/play.py --record_rollout``):

* flight: share of frames with both feet more than 2 cm up, and the longest flight [s];
* foot lift: the highest each foot's lowest sole point rises above the ground [cm];
* one-foot support: share of frames with exactly one foot up;
* squat depth: how far the torso drops (95th percentile height minus minimum) [cm];
* knee: the largest knee-crank angle [deg] (0-30 deg crank = knee 0-47 deg).

A feature counts as **present** in the reference above a threshold (flight 5 %, foot lift 8 cm, torso drop 8 cm) and
as **performed** when the rollout reaches at least half of the reference value. ``performed`` per clip = all present
features performed (clips with none present, e.g. arm gestures, count as performed); ``score`` = the mean over present
features of the reached share, capped at 1 (1.0 when none are present).

    python tools/motion_fidelity.py --pair <reference.npz> <rollout.npz> [--pair ...] [--json out.json]
    python tools/motion_fidelity.py --jobs <render_policy_clean jobs.json> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HULLS = REPO / "data/calibration/dropbear_foot_sole_hulls.json"
FEET = {"L": "LL_basis_left_1", "R": "RL_basis_left_1"}
TORSO = "torso_RMD_X10__1_Rotor_1"
KNEES = ("LL_knee_actuator_joint", "RL_knee_actuator_joint")
LIFT_M = 0.02
PRESENT = {"flight_frac": 0.05, "lift_cm": 8.0, "torso_drop_cm": 8.0}
PERFORMED_RATIO = 0.5


def _rot(q: np.ndarray) -> np.ndarray:
    """(T, 4) wxyz -> (T, 3, 3)."""
    w, x, y, z = (q[:, i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def features(npz: Path, hulls: dict | None = None) -> dict:
    """The visible features of one motion file (reference or rollout; both use body_pos_w / body_quat_w)."""
    if hulls is None:
        hulls = json.loads(HULLS.read_text(encoding="utf-8"))["feet"]
    z = np.load(npz, allow_pickle=True)
    names = [str(n) for n in z["body_names"]]
    pos, quat = np.asarray(z["body_pos_w"], np.float64), np.asarray(z["body_quat_w"], np.float64)
    sole = {}
    for k, body in FEET.items():
        i = names.index(body)
        v = np.asarray(hulls[body]["vertices_b"], np.float64)
        sole[k] = (np.einsum("tij,nj->tni", _rot(quat[:, i]), v)[..., 2] + pos[:, i, None, 2]).min(axis=1)
    ground = min(np.percentile(sole["L"], 2), np.percentile(sole["R"], 2))
    h = {k: s - ground for k, s in sole.items()}
    up_l, up_r = h["L"] > LIFT_M, h["R"] > LIFT_M
    flight = up_l & up_r
    run = best = 0
    for f in flight:
        run = run + 1 if f else 0
        best = max(best, run)
    fps = float(z["fps"])
    torso = pos[:, names.index(TORSO), 2]
    jn = [str(n) for n in z["joint_names"]]
    knee = np.degrees(np.asarray(z["joint_pos"])[:, [jn.index(k) for k in KNEES]])
    return {"flight_frac": float(flight.mean()), "flight_max_s": best / fps,
            "lift_cm": float(max(h["L"].max(), h["R"].max()) * 100),
            "lift_l_cm": float(h["L"].max() * 100), "lift_r_cm": float(h["R"].max() * 100),
            "one_foot_frac": float((up_l ^ up_r).mean()),
            "torso_drop_cm": float((np.percentile(torso, 95) - torso.min()) * 100),
            "knee_crank_max_deg": float(knee.max()), "seconds": len(flight) / fps}


def compare(ref: dict, sim: dict) -> dict:
    """Per present feature: the share of the reference value the rollout reached; ``performed`` = all >= half."""
    reached = {k: (sim[k] / ref[k] if ref[k] > 0 else 1.0) for k, thr in PRESENT.items() if ref[k] >= thr}
    score = float(np.mean([min(1.0, v) for v in reached.values()])) if reached else 1.0
    return {"present": sorted(reached), "reached": {k: round(v, 2) for k, v in reached.items()},
            "performed": all(v >= PERFORMED_RATIO for v in reached.values()), "score": round(score, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("REFERENCE", "ROLLOUT"))
    ap.add_argument("--jobs", type=Path, action="append", default=[], help="render_policy_clean job files")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    pairs = [(Path(a).stem, Path(a), Path(b)) for a, b in args.pair]
    for jf in args.jobs:
        for j in json.loads(jf.read_text(encoding="utf-8")):
            pairs.append((j["tag"], REPO / j["motion_file"], REPO / j["rollout"]))
    if not pairs:
        ap.error("give --pair or --jobs")
    hulls = json.loads(HULLS.read_text(encoding="utf-8"))["feet"]
    rows = []
    print(f"{'clip':28s} {'flight % ref/sim':>17s} {'foot lift cm':>14s} {'torso drop cm':>14s} {'knee crank deg':>15s}  score performed")
    for tag, ref_p, sim_p in pairs:
        ref, sim = features(ref_p, hulls), features(sim_p, hulls)
        c = compare(ref, sim)
        rows.append({"clip": tag, "reference": str(ref_p), "rollout": str(sim_p), "ref": ref, "sim": sim, **c})
        print(f"{tag[:28]:28s} {ref['flight_frac'] * 100:7.0f} /{sim['flight_frac'] * 100:5.0f}   "
              f"{ref['lift_cm']:6.1f} /{sim['lift_cm']:5.1f} {ref['torso_drop_cm']:6.1f} /{sim['torso_drop_cm']:5.1f} "
              f"{ref['knee_crank_max_deg']:6.1f} /{sim['knee_crank_max_deg']:5.1f}  "
              f"{c['score']:5.2f} {'yes' if c['performed'] else 'NO'} {c['reached'] if c['present'] else ''}")
    n = sum(r["performed"] for r in rows)
    mean = float(np.mean([r["score"] for r in rows]))
    print(f"performed: {n} of {len(rows)}, mean score {mean:.2f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"tool": "tools/motion_fidelity.py", "thresholds": PRESENT,
                                         "performed_ratio": PERFORMED_RATIO, "performed": n, "mean_score": round(mean, 3),
                                         "clips": rows},
                                        indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
