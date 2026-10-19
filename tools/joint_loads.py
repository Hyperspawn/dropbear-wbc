"""Where could a part break? Joint and end-stop loads from motor telemetry (``scripts/play.py --telemetry --stop_model``).

Two measurements (``source/dropbear_wbc/isaac/telemetry.py``):

* **End stops** (``tau_stop``, with ``--stop_model``): the torque each motor joint's mechanical stop carries. For the
  knees this is converted to the force on the printed knee-lock blocks (``hw_motor_specs.KNEE_LOCK_ARM_M`` from the
  motor axis).
* **Joint forces** (``link_wrench``): the force and moment every articulation joint transmits, i.e. the load on that
  joint's pin, bearing seat or motor output. The loop-closure joints (the last pin of each closed linkage: knee
  bearing-block pins, the tie-rod upper ends, the elbow hangers) are outside the articulation and not measured.

    python tools/joint_loads.py logs/loads/onefoot_v4_stops.npz [more.npz ...] --md out.md --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.robots.hw_motor_specs import KNEE_LOCK_ARM_M  # noqa: E402

FRAMES = REPO / "logs/calibrate_settle/usd_joint_frames.json"
WEIGHT_N = 56.2 * 9.81
PART = {  # USD joint suffix -> part (left side; the right side mirrors with RL_/RH_/right)
    "PG_{s}_leg_pitch": "hip roll motor (pelvis)",
    "PG_{s}_leg_roll": "hip yaw motor",
    "{L}L_hip_joint": "hip pitch motor",
    "{L}L_knee_actuator_joint": "knee motor + 300 mm drive arm",
    "{L}L_Revolute28": "knee drive-arm lower pin (bearing 13)",
    "{L}L_Revolute29": "knee drive-arm lower pin (bearing 14)",
    "{L}L_Revolute119": "knee bearing block to shank pin",
    "{L}L_Revolute33": "thigh strap top pin (outer)",
    "{L}L_Revolute37": "thigh strap top pin (inner)",
    "{L}L_Revolute47": "thigh strap knee-end pin (bearing 20)",
    "{L}L_Revolute49": "thigh strap knee-end pin (bearing 16)",
    "{L}L_Revolute67": "ankle motor A",
    "{L}L_Revolute81": "ankle motor B",
    "{L}L_Revolute87": "ankle axle 1 (shank to ankle cross)",
    "{L}L_Revolute88": "ankle axle 2 (ankle cross to foot)",
    "{L}L_Revolute115": "tie-rod ball joint at foot (210 mm rod)",
    "{L}L_Revolute117": "tie-rod ball joint at foot (290 mm rod)",
    "{L}H_yaw": "shoulder pitch motor",
    "{L}H_pitch": "shoulder roll motor",
    "{L}H_roll": "upper-arm twist motor",
    "{L}H_elbow_joint": "elbow motor",
    "{L}H_Revolute32": "elbow linkage pin (bicep)",
    "{L}H_Revolute41": "elbow hanger pin (rear)",
    "{L}H_Revolute42": "elbow hanger pin (front)",
    "{L}H_Revolute44": "forearm pin",
    "{L}H_Revolute123": "elbow linkage pin (upper arm)",
    "{L}H_wrist_roll": "wrist motor",
}


def part_names() -> dict[str, str]:
    out = {}
    for pat, label in PART.items():
        out[pat.format(s="left", L="L")] = "left " + label
        out[pat.format(s="right", L="R")] = "right " + label
    return out


def analyse(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    motors = [str(n) for n in z["motor_names"]]
    out = {"file": str(path), "meta": meta, "seconds": float(len(z["q"]) * float(z["dt"])), "stops": [], "joints": []}
    if "tau_stop" in z.files and np.abs(z["tau_stop"]).max() > 0:
        st = np.abs(z["tau_stop"])
        for i, m in enumerate(motors):
            s = st[:, i]
            if s.max() < 1.0:
                continue
            row = {"motor": m, "loaded_share": float(np.mean(s > 1.0)), "p50_nm": float(np.median(s)),
                   "p90_nm": float(np.percentile(s, 90)), "p99_nm": float(np.percentile(s, 99)), "max_nm": float(s.max())}
            if "knee_actuator" in m:
                row.update({f"{k}_block_n": row[f"{k}_nm"] / KNEE_LOCK_ARM_M for k in ("p50", "p90", "p99", "max")})
            out["stops"].append(row)
    if "link_wrench" in z.files and np.isfinite(z["link_wrench"]).any():
        links = [str(n) for n in z["link_names"]]
        frames = json.loads(FRAMES.read_text(encoding="utf-8"))["joints"]
        incoming = {v["body1"]: n for n, v in frames.items() if not v["excluded"]}
        names = part_names()
        w = z["link_wrench"]
        f, mo = np.linalg.norm(w[..., :3], axis=-1), np.linalg.norm(w[..., 3:], axis=-1)
        for k, link in enumerate(links):
            joint = incoming.get(link)
            if joint is None or joint not in names:
                continue
            out["joints"].append({"joint": joint, "part": names[joint], "z_m": float(frames[joint]["anchor0_w"][2]),
                                  "p50_n": float(np.median(f[:, k])), "p99_n": float(np.percentile(f[:, k], 99)),
                                  "max_n": float(f[:, k].max()), "p99_moment_nm": float(np.percentile(mo[:, k], 99))})
        out["joints"].sort(key=lambda r: -r["p99_n"])
    return out


def to_md(results: list[dict]) -> str:
    lines = ["# Joint and end-stop loads", "",
             f"Body weight {WEIGHT_N:.0f} N. Knee-lock force = knee stop torque / {KNEE_LOCK_ARM_M} m. Loop-closure pins are not "
             "measured (see `tools/joint_loads.py`).", ""]
    for r in results:
        lines += [f"## {Path(r['file']).stem}", "", f"{r['seconds']:.1f} s, profile `{r['meta'].get('actuator_profile', '?')}`", ""]
        if r["stops"]:
            lines += ["| end stop | loaded | typical | p99 | max | knee-lock block p99 / max |", "|---|---:|---:|---:|---:|---:|"]
            for s in r["stops"]:
                block = f"{s['p99_block_n']:.0f} / {s['max_block_n']:.0f} N" if "p99_block_n" in s else "-"
                lines.append(f"| {s['motor']} | {s['loaded_share']:.0%} | {s['p50_nm']:.0f} N·m | {s['p99_nm']:.0f} N·m | "
                             f"{s['max_nm']:.0f} N·m | {block} |")
            lines.append("")
        if r["joints"]:
            lines += ["| joint (most loaded first) | typical force | p99 | max | × body weight (p99) | p99 moment |",
                      "|---|---:|---:|---:|---:|---:|"]
            for j in r["joints"][:16]:
                lines.append(f"| {j['part']} | {j['p50_n']:.0f} N | {j['p99_n']:.0f} N | {j['max_n']:.0f} N | "
                             f"{j['p99_n'] / WEIGHT_N:.1f} | {j['p99_moment_nm']:.0f} N·m |")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    res = [analyse(f) for f in a.files]
    md = to_md(res)
    print(md)
    if a.md:
        a.md.write_text(md + "\n", encoding="utf-8")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
