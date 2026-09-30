"""Render verification figures and tables of the semantic calibration (CPU only).

Inputs: the calibration JSON, the raw sweep NPZ and (optionally) the verify NPZ written by
``tools/calibrate_semantics.py``. Outputs (``--out-dir``, default ``logs/calibrate_settle/report``):

* ``skeleton_zero_standing.png`` -- side (x-z) and front (y-z) views of Dropbear key points at the
  semantic zero and standing poses (from the physics verify records) over the G1 zero-pose skeleton
  (``dropbear_wbc.motion.g1_model``) scaled to Dropbear's hip height;
* ``maps.png`` -- knee / elbow lut1d curves (left vs right), ankle feasible (pitch, roll) sets, linear-map
  residuals;
* ``tables.md`` -- key-point table (zero vs standing vs scaled G1), per-DOF map summary, findings.

Run with the system python (numpy, matplotlib)::

    python tools/report_calibration.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.kinematics.calib_fit import SIDES, Raw, seg_bodies  # noqa: E402
from dropbear_wbc.kinematics.semantic import MOTOR_NAMES, SEMANTIC_NAMES, SemanticMap  # noqa: E402

KEYPTS = ("hip", "knee", "ankle", "sole", "shoulder", "elbow", "wrist")


def dropbear_points(raw: Raw, i: int, calib: dict, sole: dict) -> dict[str, np.ndarray]:
    """Key points (root frame) of verify record ``i``: joint anchors on the segment bodies."""
    out = {}
    for side in SIDES:
        leg, _, arm = SIDES[side]
        sb = seg_bodies(side)

        def anchor(joint, which):
            b, lp = raw.anchor_local(joint, which)
            return raw.P(b, i) + raw.R(b, i) @ lp

        geo = calib["geometry"]["per_side"][side]
        out[f"{side}_hip"] = anchor(f"{leg}_hip_joint", 1)
        k_local = np.asarray(geo["knee_pivot_fit"]["knee_center_thigh_local"])
        out[f"{side}_knee"] = raw.P(sb["thigh"], i) + raw.R(sb["thigh"], i) @ k_local
        out[f"{side}_ankle"] = anchor(f"{leg}_Revolute87", 0)
        v = np.asarray(sole["feet"][sb["foot"]]["vertices_b"])
        pts = raw.P(sb["foot"], i) + v @ raw.R(sb["foot"], i).T
        out[f"{side}_sole"] = pts[np.argmin(pts[:, 2])]
        ap = geo["arm_points_rest_root"]
        out[f"{side}_shoulder"] = np.asarray(ap["shoulder_center"])  # on the shoulder axes, fixed in the root
        e_local = np.asarray(ap["elbow_center_upper_arm_local"])
        out[f"{side}_elbow"] = raw.P(sb["upper_arm"], i) + raw.R(sb["upper_arm"], i) @ e_local
        out[f"{side}_wrist"] = anchor(f"{arm}_wrist_roll", 0)
    return out


def g1_points(hip_height: float) -> dict[str, np.ndarray]:
    from dropbear_wbc.motion.g1_model import load_g1_kinematics

    g1 = load_g1_kinematics()
    fk = g1.zero_pose  # cached property on G1Kinematics
    pts = {}
    for side in SIDES:
        pts[f"{side}_hip"] = fk.body_pos(f"{side}_hip_roll_link")
        pts[f"{side}_knee"] = fk.body_pos(f"{side}_knee_link")
        pts[f"{side}_ankle"] = fk.body_pos(f"{side}_ankle_pitch_link")
        sp = np.asarray(fk.sole_points(side)).reshape(-1, 3)
        pts[f"{side}_sole"] = sp[np.argmin(sp[:, 2])] - np.array([0, 0, 0.005])
        pts[f"{side}_shoulder"] = fk.body_pos(f"{side}_shoulder_roll_link")
        pts[f"{side}_elbow"] = fk.body_pos(f"{side}_elbow_link")
        pts[f"{side}_wrist"] = fk.body_pos(f"{side}_wrist_roll_link")
    pts = {k: np.asarray(v).reshape(-1, 3)[0] for k, v in pts.items()}
    hip_mid = 0.5 * (pts["left_hip"] + pts["right_hip"])
    ground = min(pts["left_sole"][2], pts["right_sole"][2])
    scale = hip_height / (hip_mid[2] - ground)
    return {k: (v - np.array([hip_mid[0], hip_mid[1], ground])) * scale for k, v in pts.items()}, scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", type=Path, default=REPO / "data/calibration/dropbear_semantic_calibration.json")
    ap.add_argument("--out-dir", type=Path, default=REPO / "logs/calibrate_settle/report")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calib = json.loads(args.calibration.read_text())
    smap = SemanticMap(calib)
    sole = json.loads(Path(REPO / calib["provenance"]["sole_hulls"]).read_text()) \
        if not Path(calib["provenance"]["sole_hulls"]).is_absolute() else json.loads(Path(calib["provenance"]["sole_hulls"]).read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Dropbear semantic calibration report\n", f"calibration: `{args.calibration}`  ",
             f"plant: {calib.get('plant_variant')}  ", f"usd_sha256: `{calib['usd_sha256']}`\n"]

    # ---- skeleton figure ---------------------------------------------------------------------------
    ver = calib.get("verification")
    poses = {}
    if ver:
        vr = Raw(REPO / ver["verify_npz"] if not Path(ver["verify_npz"]).is_absolute() else ver["verify_npz"])
        for i in range(len(vr.program)):
            nm = vr.program_names[vr.program[i]]
            if nm in ("semantic_zero", "standing"):
                poses[nm] = dropbear_points(vr, i, calib, sole)
    hip_h = calib["segment_lengths"]["standing_hip_height"]
    g1, g1_scale = g1_points(hip_h)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    colors = {"semantic_zero": "tab:blue", "standing": "tab:orange"}
    table_rows = []
    for nm, pts in poses.items():
        hip_mid = 0.5 * (pts["left_hip"] + pts["right_hip"])
        ground = min(pts["left_sole"][2], pts["right_sole"][2])
        off = np.array([hip_mid[0], hip_mid[1], ground])
        for side in SIDES:
            chain = [pts[f"{side}_{k}"] - off for k in ("hip", "knee", "ankle", "sole")]
            arm = [pts[f"{side}_shoulder"] - off, pts[f"{side}_elbow"] - off, pts[f"{side}_wrist"] - off]
            for ax, (i, j) in zip(axes, ((0, 2), (1, 2))):
                c = np.array(chain)
                ax.plot(c[:, i], c[:, j], "-o", color=colors[nm], label=f"Dropbear {nm}" if side == "left" else None)
                a = np.array(arm)
                ax.plot(a[:, i], a[:, j], "--", color=colors[nm])
            for k in ("hip", "knee", "ankle", "sole", "shoulder", "elbow", "wrist"):
                p = pts[f"{side}_{k}"] - off
                table_rows.append((nm, f"{side}_{k}", p))
    for side in SIDES:
        chain = np.array([g1[f"{side}_{k}"] for k in ("hip", "knee", "ankle", "sole")])
        arm = np.array([g1[f"{side}_{k}"] for k in ("shoulder", "elbow", "wrist")])
        for ax, (i, j) in zip(axes, ((0, 2), (1, 2))):
            ax.plot(chain[:, i], chain[:, j], "-s", color="gray", alpha=0.6, label="G1 zero (scaled)" if side == "left" else None)
            ax.plot(arm[:, i], arm[:, j], ":", color="gray", alpha=0.6)
        for k in ("hip", "knee", "ankle", "sole", "shoulder", "elbow", "wrist"):
            table_rows.append(("g1_zero_scaled", f"{side}_{k}", g1[f"{side}_{k}"]))
    for ax, title in zip(axes, ("side view (x fwd, z up)", "front view (y left, z up)")):
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.5)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Dropbear key points (origin: hip midpoint x/y, ground z) vs G1 zero x{g1_scale:.2f}")
    fig.tight_layout()
    fig.savefig(args.out_dir / "skeleton_zero_standing.png", dpi=110)
    plt.close(fig)

    # ---- maps figure --------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, base in enumerate(("knee", "elbow")):
        ax = axes[0, col]
        for side, ls in (("left", "-"), ("right", "--")):
            d = calib["dofs"][f"{side}_{base}"]
            if d["type"] == "lut1d":
                ax.plot(np.degrees(d["motor_grid"]), np.degrees(d["semantic_values"]), ls, label=f"{side} (lut1d)")
            else:
                inf = d.get("info", {})
                if "raw_motor" in inf:
                    ax.plot(np.degrees(inf["raw_motor"]), np.degrees(inf["raw_semantic"]), ls + "x",
                            label=f"{side} ({d['type']})")
        ax.set_xlabel(f"{base} motor [deg]")
        ax.set_ylabel(f"semantic {base} [deg]")
        ax.grid(alpha=0.3)
        ax.legend()
    for col, side in enumerate(SIDES):
        ax = axes[1, col]
        pr = calib["ankle_pairs"][side]
        p = np.array(pr["pitch"], dtype=float)
        r = np.array(pr["roll"], dtype=float)
        v = np.array(pr["valid"], dtype=bool)
        ax.scatter(np.degrees(p[v]), np.degrees(r[v]), s=6, label="valid grid node")
        ax.scatter(np.degrees(p[~v]), np.degrees(r[~v]), s=6, c="r", marker="x", label="infeasible node")
        ax.set_xlabel("ankle pitch [deg]")
        ax.set_ylabel("ankle roll [deg]")
        ax.set_title(f"{side} ankle: (calf A, calf B) grid -> (pitch, roll)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax = axes[0, 2]
    lin = [(n, d) for n, d in calib["dofs"].items() if d["type"] in ("linear", "serial3")]
    ax.barh([n for n, _ in lin], [np.degrees(d["fit"]["max_abs_rad"]) for _, d in lin])
    ax.set_xlabel("linear fit max residual [deg]")
    ax.tick_params(axis="y", labelsize=7)
    axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(args.out_dir / "maps.png", dpi=110)
    plt.close(fig)

    # ---- tables -------------------------------------------------------------------------------------
    lines.append("## Key points (m; x fwd / y left relative to the hip midpoint, z above the lowest sole point)\n")
    lines.append("| pose | point | x | y | z |\n|---|---|---:|---:|---:|")
    for nm, k, p in table_rows:
        lines.append(f"| {nm} | {k} | {p[0]:+.3f} | {p[1]:+.3f} | {p[2]:+.3f} |")
    lines.append("\n## Per-DOF maps\n")
    lines.append("| semantic | type | motors | valid range [deg] | notes |\n|---|---|---|---|---|")
    for n in SEMANTIC_NAMES:
        d = calib["dofs"][n]
        note = ""
        if d["type"] in ("linear", "serial3"):
            note = f"scale {d['scale']:+.4f}, offset {np.degrees(d['offset']):+.2f} deg, fit max {np.degrees(d['fit']['max_abs_rad']):.2f} deg"
        elif d["type"] == "lut1d":
            note = f"gain {d['info']['mean_gain']:+.3f}, hysteresis {np.degrees(d['info'].get('hysteresis_max_rad', 0)):.2f} deg"
        elif d["type"] == "fixed":
            note = "LOCKED"
        vr = np.degrees(d["valid_range"])
        lines.append(f"| {n} | {d['type']} | {', '.join(d['motors'])} | [{vr[0]:+.1f}, {vr[1]:+.1f}] | {note} |")
    lines.append("\n## semantic_zero_motor_pos / standing_motor_pos [deg]\n")
    lines.append("| motor | zero | standing |\n|---|---:|---:|")
    for i, m in enumerate(MOTOR_NAMES):
        lines.append(f"| {m} | {np.degrees(smap.semantic_zero_motor_pos[i]):+.2f} | {np.degrees(smap.standing_motor_pos[i]):+.2f} |")
    lines.append("\n## Segment lengths\n")
    for k, v in calib["segment_lengths"].items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
    if ver:
        lines.append("\n## Verification (physics vs map)\n")
        for k in ("num_poses", "semantic_error_max_deg", "semantic_error_model_vs_physics_max_deg",
                  "fk_model_position_error_max_mm", "closure_gap_max_mm", "motor_tracking_err_max_deg"):
            lines.append(f"- {k}: {ver[k]}")
        lines.append("\n| semantic | max err [deg] | rms [deg] |\n|---|---:|---:|")
        for n, e in ver["semantic_error_map_vs_physics"].items():
            lines.append(f"| {n} | {e['max_abs_deg']:.3f} | {e['rms_deg']:.3f} |")
    lines.append("\n## Findings\n")
    lines += [f"- {f}" for f in calib.get("findings", [])]
    (args.out_dir / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_dir}/skeleton_zero_standing.png, maps.png, tables.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
