"""Sim2sim report: a tracking policy run in Newton (tools/newton_bridge.py + tools/policy_runner.py) vs its reference
motion and vs the Isaac Lab evaluation (scripts/play.py summary). CPU only.

Metrics during the runner's POLICY phase (reference frame k = policy step k, as in the runner):

* ``error_joint_pos``: ||q_motor - q_ref|| over the 22 motors, the same definition as the Isaac metric
  (``MotionCommand._update_metrics``), mean / p95 / max over the policy steps before a fall;
* per-motor RMS error;
* from the bridge trace (``--sim-bodies``): anchor z and hand z vs the reference (absolute z; ground at 0 in both), and
  the right-hand rise above its start;
* fall: first time the runner's torso tilt exceeded 1 rad or root z dropped below the summary threshold.

    python tools/sim2sim_report.py --runner-log run.jsonl --runner-summary run.json --bridge-trace trace.npz \
        --bridge-report bridge.json --npz data/motions/synthetic/wave_right.npz --isaac-play <run>/play_<stamp>.json \
        --out logs/gpu_pipeline/sim2sim/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.robots.dropbear_names import ANCHOR_BODY, HAND_BODIES, MOTOR_NAMES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner-log", type=Path, required=True)
    ap.add_argument("--runner-summary", type=Path, required=True)
    ap.add_argument("--bridge-trace", type=Path, default=None)
    ap.add_argument("--bridge-report", type=Path, default=None)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--isaac-play", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = [json.loads(line) for line in a.runner_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    summ = json.loads(a.runner_summary.read_text(encoding="utf-8"))
    with np.load(a.npz, allow_pickle=False) as d:
        jn = [str(n) for n in d["joint_names"]]
        bn = [str(n) for n in d["body_names"]]
        ref_q = np.asarray(d["joint_pos"], float)[:, [jn.index(n) for n in MOTOR_NAMES]]
        ref_pos = np.asarray(d["body_pos_w"], float)
    pol = [r for r in rows if r.get("fsm") == "policy"]
    rep: dict = {"runner_log": str(a.runner_log), "npz": str(a.npz), "policy_steps": len(pol),
                 "transitions": summ.get("transitions"), "fall": summ.get("fall"),
                 "privileged_terms": summ.get("privileged_terms")}
    if not pol:
        rep["error"] = "no POLICY steps in the runner log"
        a.out.write_text(json.dumps(rep, indent=1) + "\n")
        print(json.dumps(rep, indent=1))
        return 1
    t0 = pol[0]["t"]
    fall_t = (summ.get("fall") or {}).get("t")
    rep["policy_start_t"] = t0
    rep["time_in_policy_before_fall_s"] = (fall_t - t0) if fall_t is not None else None
    k = np.arange(len(pol))
    k = np.minimum(k, ref_q.shape[0] - 1)
    q = np.array([r["q"] for r in pol], float)
    alive = np.array([fall_t is None or r["t"] < fall_t for r in pol])
    err = np.linalg.norm(q - ref_q[k], axis=1)
    e = err[alive]
    rep["error_joint_pos"] = {"mean": float(e.mean()), "p95": float(np.percentile(e, 95)), "max": float(e.max()),
                              "steps": int(alive.sum()), "definition": "||q_motor - q_ref|| over 22 motors (Isaac metric)"}
    per = np.sqrt(((q - ref_q[k])[alive] ** 2).mean(0))
    rep["per_motor_rms_rad"] = {n: round(float(v), 4) for n, v in zip(MOTOR_NAMES, per)}
    windows = {}
    for w0, w1 in ((0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 10)):
        m = alive & (k >= w0 * 50) & (k < w1 * 50)
        if m.any():
            windows[f"{w0}-{w1}s"] = round(float(err[m].mean()), 4)
    rep["error_joint_pos_by_window"] = windows

    if a.bridge_trace is not None and a.bridge_trace.is_file():
        with np.load(a.bridge_trace, allow_pickle=False) as tr:
            names = [str(n) for n in tr["body_names"]] if "body_names" in tr.files else []
            ts = np.asarray(tr["time_s"], float)
            root = np.asarray(tr["root_pos_w"], float)
            bp = np.asarray(tr["body_pos_w"], float) if "body_pos_w" in tr.files else None
        sel = (ts >= t0) & ((ts < fall_t) if fall_t is not None else True)
        kk = np.clip(np.round((ts[sel] - t0) / 0.02).astype(int), 0, ref_pos.shape[0] - 1)
        out = {"samples": int(sel.sum()), "root_z_min": float(root[sel, 2].min()) if sel.any() else None}
        if bp is not None and names and sel.any():
            for label, body in (("anchor", ANCHOR_BODY), ("left_hand", HAND_BODIES[0]), ("right_hand", HAND_BODIES[1])):
                if body not in names:
                    continue
                z = bp[sel, names.index(body), 2]
                zr = ref_pos[kk, bn.index(body), 2]
                out[label] = {"z_err_mean_m": float(np.abs(z - zr).mean()), "z_err_max_m": float(np.abs(z - zr).max()),
                              "z_min": float(z.min()), "z_max": float(z.max()),
                              "ref_z_min": float(zr.min()), "ref_z_max": float(zr.max()),
                              "rise_above_start_m": float(z.max() - z[0]), "ref_rise_above_start_m": float(zr.max() - zr[0])}
        rep["bridge_trace"] = out
    if a.bridge_report is not None and a.bridge_report.is_file():
        br = json.loads(a.bridge_report.read_text(encoding="utf-8"))
        plant = br.get("plant", {})
        rep["bridge"] = {k: br.get(k) for k in ("status", "rtf", "ticks", "sim_time_s", "closure_residual_final_m")
                         if k in br}
        rep["bridge"]["presettle"] = {k: v for k, v in (plant.get("presettle") or {}).items() if k != "closure_trace_m"}
        rep["bridge"]["authored_ankle_tierods"] = plant.get("authored_ankle_tierods")
    if a.isaac_play is not None and a.isaac_play.is_file():
        ip = json.loads(a.isaac_play.read_text(encoding="utf-8"))
        rep["isaac"] = {"file": str(a.isaac_play), "mean_metrics": ip.get("mean_metrics"),
                        "termination_counts": ip.get("termination_counts"), "num_envs": ip.get("num_envs"),
                        "steps": ip.get("steps"), "solver_iterations": ip.get("solver_iterations"),
                        "closure_gap_m": ip.get("closure_gap_m")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
