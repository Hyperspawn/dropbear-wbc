"""Static 3D actuator map of Dropbear: where every motor sits, what it turns, and its numbers, for the robot as it is
and for the two candidate builds (docs/ACTUATORS.md).

Everything comes from the files the simulation itself uses:

* ``logs/calibrate_settle/usd_joint_frames.json``: joint anchors, axes, limits and body poses of the plant USD
  (authored pose, root frame: x forward, y left, z up);
* ``source/dropbear_wbc/robots/hw_motor_specs.py``: motor models, gains, CAN ids and the build profiles;
* the semantic calibration: which human joint each motor moves, and its range;
* ``tools/export_dropbear_glb.py`` output: the decimated visual mesh (one node per body), embedded in the page.

    python tools/build_actuator_viewer.py [--glb logs/actuator_viewer/dropbear.glb]

Writes ``logs/actuator_viewer/index.html`` (git-ignored). Market prices are built in; Hyperspawn's own quotes are added
only from the git-ignored ``logs/actuator_viewer/prices_private.json`` when it exists (never commit them).
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.robots import hw_motor_specs as hw  # noqa: E402

CAL = REPO / "data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json"
FRAMES = REPO / "logs/calibrate_settle/usd_joint_frames.json"
TEMPLATE = REPO / "tools/actuator_viewer/template.html"
PRIVATE_PRICES = REPO / "logs/actuator_viewer/prices_private.json"

LABELS = {  # USD joint -> plain name, and the name on the user's actuator sheet
    "PG_{s}_leg_pitch": ("{S} hip roll", "Hip Spreader"),
    "PG_{s}_leg_roll": ("{S} hip yaw", "Leg Rotator"),
    "{p}L_hip_joint": ("{S} hip pitch", "Waist Pivot"),
    "{p}L_knee_actuator_joint": ("{S} knee motor", "Knee Bender"),
    "{p}L_Revolute67": ("{S} ankle motor A", "Calf-Foot Flexor"),
    "{p}L_Revolute81": ("{S} ankle motor B", "Calf-Foot Flexor"),
    "{p}H_yaw": ("{S} shoulder pitch", "Shoulder Rotator"),
    "{p}H_pitch": ("{S} shoulder roll", "Shoulder Extender"),
    "{p}H_roll": ("{S} upper-arm twist", "Arm Twister"),
    "{p}H_elbow_joint": ("{S} elbow motor", "Elbow"),
    "{p}H_wrist_roll": ("{S} wrist roll", "Wrist"),
}
LINKAGE = {  # docs/ACTUATORS.md section 5
    "knee": {"kind": "Four-bar linkage", "gain": 1.79,
             "note": "The motor sits at the top of the thigh and turns the knee through a four-bar that speeds the "
                     "knee up about 1.79× at standing (2.1–2.2× deeply bent). The knee joint gets the motor torque "
                     "divided by that. Crank 0–30° = knee 0–47°."},
    "elbow": {"kind": "Linkage", "gain": 4.8,
              "note": "The motor sits in the upper arm and turns the elbow through a linkage that speeds it up about "
                      "4.8×, so the elbow joint gets about a fifth of the motor torque. Crank 0–30° = elbow −33° to 88°."},
    "calf": {"kind": "Tie rods", "gain": None,
             "note": "The two shank motors push the foot through tie rods with ball ends: both the same way pitch the "
                     "foot, opposite ways roll it."},
}
MODEL = {  # model -> short name, rated speed [rpm], market price [USD] (docs/ACTUATORS.md section 2)
    "EPS-CEM-60-V2": ("CEM-60", 60, 1290),
    "RMD-X8-P20-120-V4": ("X8-120", 127, 645),
    "RMD-X10-P20-200-V4.1": ("X10-200", 74, 899),
    "RMD-X10-V3-1:7": ("X10 1:7", 165, 645),
    "RMD-X8-Pro-V2-1:9": ("X8 Pro 1:9", 110, 470),
    "RMD-X10-S2-V3-1:35": ("X10-S2 1:35", 50, None),
}
BUILDS = [  # id, title, profile (motor map), hip roll installed?
    ("robot", "On the robot", "hw_cem6", False),
    ("hw_cem6", "Build A · 6 × CEM-60", "hw_cem6", True),
    ("hw_v4", "Build B · X8-120 + X10-200", "hw_v4", True),
]
OWNED = {"EPS-CEM-60-V2": 4, "RMD-X10-V3-1:7": 4, "RMD-X8-Pro-V2-1:9": 12}
"""Motors bought in 2024 (user, 2026-10-16)."""
HEAVY = [f"PG_{s}_leg_pitch" for s in ("left", "right")] + ["LL_hip_joint", "RL_hip_joint",
                                                            "LL_knee_actuator_joint", "RL_knee_actuator_joint"]
SIDE = {"left": "Left", "right": "Right"}


def axis_class(a):
    k = max(range(3), key=lambda i: abs(a[i]))
    return ["roll", "pitch", "yaw"][k], math.degrees(math.acos(min(1.0, abs(a[k]))))


def motor_entry(name: str, role: str, joint: str, profile: str, quotes: dict) -> dict:
    m = hw.MOTOR_MODELS[name]
    short, rated_rpm, market = MODEL[name]
    kp, kd = hw.HW_PROFILE_OPTIONS.get(profile, {}).get("gain_overrides", {}).get(joint, hw.HW_GAINS[role])
    link = LINKAGE.get("knee" if role == "knee" else "elbow" if "elbow" in joint else "calf" if role == "calf" else "")
    gain = link["gain"] if link else None
    return {"model": name, "short": short, "status": m.status, "ratio": m.gear_ratio, "peak_nm": m.peak_torque,
            "rated_nm": m.rated_torque, "rated_rpm": rated_rpm,
            "no_load_rpm": round(m.no_load_speed * 60 / (2 * math.pi), 1), "armature": round(m.armature, 4),
            "mass_kg": m.mass_kg, "kp": kp, "kd": kd, "market_usd": market, "quote_usd": quotes.get(name),
            "joint_peak_nm": round(m.peak_torque / gain, 1) if gain else None,
            "joint_rated_nm": round(m.rated_torque / gain, 1) if gain else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "logs/actuator_viewer/index.html")
    ap.add_argument("--glb", type=Path, default=REPO / "logs/actuator_viewer/dropbear.glb")
    args = ap.parse_args()

    frames = json.loads(FRAMES.read_text(encoding="utf-8"))
    cal = json.loads(CAL.read_text(encoding="utf-8"))
    quotes = {}
    if PRIVATE_PRICES.is_file():
        quotes = {k: v for k, v in json.loads(PRIVATE_PRICES.read_text(encoding="utf-8")).items()
                  if not k.startswith("_")}
    J = frames["joints"]
    lims = dict(zip(cal["motor_names"], cal["motor_limits_rad"]))
    moves = {}
    for sname, d in cal["dofs"].items():
        rng = d["valid_range"]
        if d["type"] in ("serial3", "linear") and len(d["motors"]) == 1:  # direct drive: motor travel x fitted scale
            lo, hi = lims[d["motors"][0]]
            rng = sorted(d["scale"] * x + d["offset"] for x in (lo, hi))
        for m in d["motors"]:
            moves.setdefault(m, []).append({"name": sname.replace("_", " "),
                                            "range_deg": [round(math.degrees(x), 1) for x in rng]})

    motors = []
    for jn in hw.MOTOR_NAMES:
        j = J[jn]
        side = "left" if jn.startswith(("LL", "LH")) or "left" in jn else "right"
        label = user_name = None
        for pat, (lab, un) in LABELS.items():
            if pat.format(s=side, p=side[0].upper()) == jn:
                label, user_name = lab.format(S=SIDE[side]), un
        assert label and user_name, jn
        role = hw.JOINT_ROLES[jn]
        acls, aoff = axis_class(j["axis0_w"])
        notes = []
        if abs(j["upperLimit"] - j["lowerLimit"]) >= 359:
            notes.append("No end stop in the sim (±180°). Where does the real joint stop?")
        if aoff > 5:
            notes.append(f"Axis tilted {aoff:.0f}° by the CAD's 10° leg bracket.")
        tag = jn.split("_")[-1] if jn.startswith("PG_") else jn.split("_")[1] if jn[:2] in ("LH", "RH") else ""
        if tag in ("pitch", "roll", "yaw") and tag != acls:
            notes.append(f"The USD names it '{tag}', but it turns about the body's {acls} axis; the sim treats it "
                         f"as {label.split(' ', 1)[1]}.")
        per_build = {}
        for bid, _, profile, hip_roll_on in BUILDS:
            name = hw.JOINT_MOTOR_MAPS[hw.HW_PROFILE_MAPS[profile]][jn]
            per_build[bid] = None if (role == "hip_roll" and not hip_roll_on) else motor_entry(name, role, jn, profile, quotes)
        link_key = "knee" if role == "knee" else "elbow" if "elbow" in jn else "calf" if role == "calf" else None
        motors.append({
            "id": jn, "label": label, "user_name": user_name, "side": side, "heavy": jn in HEAVY,
            "group": f"{SIDE[side]} {'leg' if jn[:2] in ('PG', 'LL', 'RL') else 'arm'}",
            "pos": [round(x, 4) for x in j["anchor0_w"]], "axis": [round(x, 4) for x in j["axis0_w"]],
            "axis_class": acls, "axis_off_deg": round(aoff, 1),
            "range_deg": [round(j["lowerLimit"], 1), round(j["upperLimit"], 1)], "moves": moves.get(jn, []),
            "can_id": hw.MOTOR_CAN_IDS.get(jn), "linkage": LINKAGE[link_key] if link_key else None,
            "notes": notes, "builds": per_build,
        })

    cad_mass = sum(b.get("mass", 0) for b in frames["bodies"].values())
    summary = {}
    for bid, title, profile, hip_roll_on in BUILDS:
        need = {}
        for m in motors:
            e = m["builds"][bid]
            if e:
                need[e["model"]] = need.get(e["model"], 0) + 1
        buy = {k: max(0, n - OWNED.get(k, 0)) for k, n in need.items() if n > OWNED.get(k, 0)}
        spare = {k: n - need.get(k, 0) for k, n in OWNED.items() if n > need.get(k, 0)}
        heavy = [m["builds"][bid] for m in motors if m["heavy"]]
        def cost(field):
            vals = [(n, MODEL[k][2] if field == "market" else quotes.get(k)) for k, n in buy.items()]
            known = sum(n * v for n, v in vals if v)
            return {"usd": known, "complete": all(v for _, v in vals)}
        summary[bid] = {
            "title": title,
            "slots": {s: next((m["builds"][bid] or {}).get("short", "not installed") for m in motors
                              if m["id"] == j) for s, j in (("Hip roll", "PG_left_leg_pitch"),
                                                             ("Hip pitch", "LL_hip_joint"),
                                                             ("Knee", "LL_knee_actuator_joint"))},
            "buy": {MODEL[k][0]: n for k, n in buy.items()}, "spare": {MODEL[k][0]: n for k, n in spare.items()},
            "market": cost("market"), "quote": cost("quote") if quotes else None,
            "heavy_mass_kg": round(sum(e["mass_kg"] for e in heavy if e), 2),
            "plant_mass_kg": round(cad_mass + sum(hw.motor_mass_offsets(profile).values()), 1) if hip_roll_on else None,
            "simulated": hip_roll_on,
        }

    targets = {}  # the joints the linkage motors move, from the calibration geometry
    geo, seg = cal["geometry"]["per_side"], cal["segment_lengths"]
    for side in ("left", "right"):
        g = geo[side]["zero"]
        hip, ank = g["hip_center_root"], g["ankle_center_root"]
        f = seg["hip_to_knee"] / (seg["hip_to_knee"] + seg["knee_to_ankle"])
        targets[f"{side} knee"] = [round(hip[i] + f * (ank[i] - hip[i]), 4) for i in range(3)]
        targets[f"{side} ankle"] = [round(x, 4) for x in ank]
        sh = J[("LH" if side == "left" else "RH") + "_pitch"]["anchor0_w"]
        targets[f"{side} elbow"] = [round(sh[0], 4), round(sh[1], 4), round(sh[2] - seg["shoulder_to_elbow"], 4)]
    neck = [{"id": f"head_LeadScrew{i}", "pos": [round(x, 4) for x in J[f"head_LeadScrew{i}"]["anchor0_w"]]}
            for i in range(1, 7)]
    bodies = {n: {"p": [round(x, 5) for x in b["pos_w"]], "q": [round(x, 6) for x in b["quat_w_wxyz"]]}
              for n, b in frames["bodies"].items()}
    data = {"usd_sha256": frames["usd_sha256"], "builds": [b[:2] for b in BUILDS], "summary": summary,
            "motors": motors, "targets": targets, "neck": neck, "bodies": bodies, "has_quotes": bool(quotes),
            "cad_mass_kg": round(cad_mass, 1)}
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    html = html.replace('/*__GLB__*/""', json.dumps(base64.b64encode(args.glb.read_bytes()).decode("ascii")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(json.dumps({"out": str(args.out), "motors": len(motors), "summary": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
