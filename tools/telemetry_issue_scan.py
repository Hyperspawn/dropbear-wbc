"""Scan motor telemetry for the small issues that are easy to miss in a video (docs/ISSUES.md).

Input: the ``--telemetry`` npz of ``scripts/play.py`` / ``scripts/play_locomotion.py`` (``isaac/telemetry.py``), one env,
50 Hz rows plus 200 Hz substep torques. Checks, each with a threshold in ``THRESH``:

* ``thermal``        RMS torque over the rollout / rated (continuous) torque, per motor
* ``clipped``        share of policy steps where the PD asked for more than the torque-speed envelope gave
* ``torque_jump``    share of 5 ms steps whose torque change exceeds 25 % of peak (48 V bus ripple, gear shock)
* ``speed``          p99 |joint speed| / no-load speed (headroom for the torque-speed line)
* ``target_lag``     mean |position target - position| (the motor cannot follow the command)
* ``near_limit``     share of steps within 3 deg of a joint limit
* ``stop_load``      share of steps a joint sits on its hard stop while its motor pushes AWAY from it with > 20 % of
                     peak: the mechanical stop carries the load the motor cannot (on the robot: printed stops hammered
                     every step)
* ``stop_press``     share of steps a motor pushes INTO a stop it already sits on (wasted current, heat)
* ``target_beyond``  share of steps whose position target lies > 2 deg past a hard stop (a deploy runtime that clamps
                     p_des to the range behaves differently; ISSUES #25)
* ``jitter``         RMS of the second difference of the 50 Hz targets, per motor (shaky commands)
* ``asymmetry``      left/right RMS torque ratio of mirrored motors (policy braces one side)
* ``scuff``          ground touches shorter than 60 ms inside a swing (toe scuff), per second per foot
* ``bounce``         lift-offs shorter than 60 ms inside a stance (bouncing contact), per second per foot
* ``limp``           left/right difference in stance share
* ``touchdown_skid`` mean planar sole travel from touchdown until the foot settles (needs ``sole_pos_w``)
* ``foot_slip``      p95 planar sole speed in settled mid-stance (needs ``sole_pos_w``)
* ``crossed``        share of steps with the soles closer than 5 cm sideways (or crossed)
* ``sway``           RMS lateral base speed and RMS tilt (needs ``proj_grav``)
* ``fall``           the anchor drops below 55 % of its start height

usage: python tools/telemetry_issue_scan.py <telemetry.npz> [...] [--md out.md] [--json out.json]
Old files without the newer keys skip those checks; joint limits fall back to the USD dump ``--limits_json``. Pure numpy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

THRESH = {
    "thermal": (1.0, 1.3),           # RMS / rated: warn, high
    "clipped": (0.05, 0.20),         # share of policy steps
    "torque_jump": (0.05, 0.15),     # share of 5 ms steps with |d tau| > 25 % peak
    "speed": (0.8, 0.95),            # p99 |qd| / no-load
    "target_lag": (0.08, 0.15),      # rad, mean
    "near_limit": (0.05, 0.20),      # share of steps
    "stop_load": (0.02, 0.10),       # share of steps
    "stop_press": (0.05, 0.15),      # share of steps
    "target_beyond": (0.05, 0.20),   # share of steps
    "jitter": (0.02, 0.05),          # rad, RMS second difference of 50 Hz targets
    "asymmetry": (1.5, 2.0),         # RMS torque ratio (only if the larger side is > 30 % of rated)
    "contact_chatter": (0.3, 1.0),   # short segments per second per foot
    "limp": (0.05, 0.10),            # |stance share L - R|
    "foot_slip": (0.10, 0.20),       # m/s, p95 planar sole speed in settled mid-stance
    "touchdown_skid": (0.04, 0.08),  # m, mean planar sole travel from touchdown until it settles
    "crossed": (0.01, 0.05),         # share of steps with lateral sole distance < 5 cm
    "sway_vy": (0.15, 0.30),         # m/s RMS lateral base speed
    "tilt": (0.10, 0.20),            # RMS |projected gravity xy| (~ rad)
}
HW_GATE = ("fall", "stop_load", "thermal", "clipped", "speed", "touchdown_skid", "foot_slip", "scuff")
"""Checks whose HIGH findings fail the hardware gate: each one breaks or overheats a real part, or slips a foot. The
rest (jitter, lag, asymmetry, sway, ...) are quality warnings."""
JUMP_FRAC_OF_PEAK = 0.25
CHATTER_S = 0.06
SETTLE_V = 0.1  # m/s: a landed sole below this planar speed has settled
LIMIT_MARGIN = np.deg2rad(3.0)
STOP_MARGIN = np.deg2rad(1.0)
STOP_TORQUE = 0.2  # of peak
USD_LIMITS_JSON = Path(__file__).resolve().parents[1] / "logs" / "calibrate_settle" / "usd_joint_frames.json"


def usd_limits(names: list[str], path: Path = USD_LIMITS_JSON) -> np.ndarray | None:
    """(n, 2) joint limits [rad] of ``names`` from a ``tools/dump_usd_joint_frames.py`` dump (USD limits in deg)."""
    if not Path(path).is_file():
        return None
    joints = json.loads(Path(path).read_text(encoding="utf-8"))["joints"]
    if isinstance(joints, list):
        joints = {j.get("name"): j for j in joints}
    try:
        return np.deg2rad(np.array([[joints[n]["lowerLimit"], joints[n]["upperLimit"]] for n in names], dtype=float))
    except (KeyError, TypeError):
        return None


def _sev(value: float, key: str) -> str:
    warn, high = THRESH[key]
    if not np.isfinite(value):
        return ""
    if value >= high:
        return "HIGH"
    return "warn" if value >= warn else ""


def _mirror_pairs(names: list[str]) -> list[tuple[int, int]]:
    pairs = []
    for i, n in enumerate(names):
        for a, b in (("PG_left_", "PG_right_"), ("LL_", "RL_"), ("LH_", "RH_")):
            if n.startswith(a) and n.replace(a, b, 1) in names:
                pairs.append((i, names.index(n.replace(a, b, 1))))
    return pairs


def _runs(x: np.ndarray) -> list[tuple[bool, int, int]]:
    """Run-length encoding of a boolean series: [(value, start, length)], dropping the first and last (clipped) runs."""
    runs, start = [], 0
    for t in range(1, len(x) + 1):
        if t == len(x) or x[t] != x[start]:
            runs.append((bool(x[start]), start, t - start))
            start = t
    return runs[1:-1]


def _segments(x: np.ndarray) -> list[tuple[bool, int]]:
    """[(value, length)] of :func:`_runs`."""
    return [(on, L) for on, _, L in _runs(x)]


def scan(path: Path, limits_json: Path = USD_LIMITS_JSON) -> dict:
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["motor_names"]]
    dt = float(d["dt"]) if "dt" in d.files else 0.02
    T = len(d["q"])
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    peak = np.asarray(d["peak_torque"], dtype=float)
    rated = np.asarray(d["rated_torque"], dtype=float) if "rated_torque" in d.files else peak / 2.5
    nl = np.asarray(d["no_load_speed"], dtype=float) if "no_load_speed" in d.files else None
    tau, pd, q, qd, qt = (np.asarray(d[k], dtype=float) for k in ("tau", "tau_pd", "q", "qd", "q_target"))
    findings: list[dict] = []

    def add(check, value, key=None, motor=None, detail=""):
        sev = _sev(value, key or check)
        if sev:
            findings.append({"check": check, "motor": motor, "value": round(float(value), 4), "severity": sev,
                             "detail": detail})

    # ---- per motor ----
    rms = np.sqrt((tau ** 2).mean(0))
    for i, n in enumerate(names):
        add("thermal", rms[i] / rated[i], motor=n, detail=f"RMS {rms[i]:.1f} N*m vs rated {rated[i]:.0f}")
    clip = (np.abs(pd - tau) > 1e-3 * peak).mean(0)
    for i, n in enumerate(names):
        add("clipped", clip[i], motor=n, detail="PD demand above the torque-speed envelope")
    jump_any = np.nan
    if "tau_sub" in d.files:
        sub = np.asarray(d["tau_sub"], dtype=float).reshape(-1, len(names))
        sub = sub[~np.isnan(sub).any(1)]
        dj = np.abs(np.diff(sub, axis=0)) / peak
        jf = (dj > JUMP_FRAC_OF_PEAK).mean(0)
        jump_any = float((dj > JUMP_FRAC_OF_PEAK).any(1).mean())
        for i, n in enumerate(names):
            add("torque_jump", jf[i], motor=n, detail=f"p99 jump {np.percentile(dj[:, i], 99) * 100:.0f} % of peak")
    if nl is not None:
        sp = np.percentile(np.abs(qd), 99, axis=0) / nl
        for i, n in enumerate(names):
            add("speed", sp[i], motor=n, detail=f"p99 {np.percentile(np.abs(qd[:, i]), 99):.1f} of {nl[i]:.1f} rad/s")
    lag = np.abs(qt - q).mean(0)
    for i, n in enumerate(names):
        add("target_lag", lag[i], motor=n, detail=f"mean |q_target - q| = {np.rad2deg(lag[i]):.1f} deg")
    lim = np.asarray(d["joint_limits"], dtype=float) if "joint_limits" in d.files else usd_limits(names, limits_json)
    if lim is not None:
        near = ((q < lim[:, 0] + LIMIT_MARGIN) | (q > lim[:, 1] - LIMIT_MARGIN)).mean(0)
        at_lo, at_hi = q < lim[:, 0] + STOP_MARGIN, q > lim[:, 1] - STOP_MARGIN
        f = STOP_TORQUE * peak
        load = ((at_hi & (tau < -f)) | (at_lo & (tau > f))).mean(0)
        press = ((at_hi & (tau > f)) | (at_lo & (tau < -f))).mean(0)
        for i, n in enumerate(names):
            add("near_limit", near[i], motor=n, detail="within 3 deg of a joint stop")
            end = "upper" if at_hi[:, i].mean() >= at_lo[:, i].mean() else "lower"
            add("stop_load", load[i], motor=n,
                detail=f"on its {end} stop ({np.rad2deg(lim[i, 1 if end == 'upper' else 0]):.0f} deg) while the motor "
                       f"pushes away with > {STOP_TORQUE:.0%} of peak")
            add("stop_press", press[i], motor=n, detail=f"pushing into its {end} stop")
        past = np.maximum(lim[:, 0] - qt, qt - lim[:, 1])  # > 0: target past a stop [rad]
        for i, n in enumerate(names):
            add("target_beyond", float((past[:, i] > np.deg2rad(2.0)).mean()), motor=n,
                detail=f"target up to {np.rad2deg(past[:, i].max()):.0f} deg past a stop")
    jit = np.sqrt((np.diff(qt, n=2, axis=0) ** 2).mean(0))
    for i, n in enumerate(names):
        add("jitter", jit[i], motor=n, detail=f"RMS second difference of the 50 Hz target {np.rad2deg(jit[i]):.1f} deg")
    for i, j in _mirror_pairs(names):
        hi, lo = max(rms[i], rms[j]), min(rms[i], rms[j])
        if hi > 0.3 * rated[i] and lo > 1e-6:
            side = names[i] if rms[i] > rms[j] else names[j]
            add("asymmetry", hi / lo, motor=side, detail=f"{names[i]} {rms[i]:.1f} vs {names[j]} {rms[j]:.1f} N*m RMS")

    # ---- whole body ----
    body: dict = {"steps": T, "seconds": round(T * dt, 2), "torque_jump_any_motor": jump_any}
    if "contact" in d.files:
        c = np.asarray(d["contact"], dtype=bool)
        body["stance_share"] = [round(float(x), 3) for x in c.mean(0)]
        add("limp", abs(c[:, 0].mean() - c[:, 1].mean()), detail=f"stance L/R {c[:, 0].mean():.2f}/{c[:, 1].mean():.2f}")
        for k, side in enumerate(("left", "right")):
            segs = _segments(c[:, k])
            touch = sum(1 for on, L in segs if on and L * dt < CHATTER_S)   # brief touch in swing: toe scuff
            bounce = sum(1 for on, L in segs if not on and L * dt < CHATTER_S)  # brief lift in stance: bounce
            add("scuff", touch / (T * dt), key="contact_chatter", motor=f"{side} foot",
                detail=f"{touch} ground touches < 60 ms during swing (toe scuff / low clearance)")
            add("bounce", bounce / (T * dt), key="contact_chatter", motor=f"{side} foot",
                detail=f"{bounce} lift-offs < 60 ms during stance (bouncing / chattering contact)")
        if "sole_pos_w" in d.files:
            s = np.asarray(d["sole_pos_w"], dtype=float)
            v = np.linalg.norm(np.diff(s[:, :, :2], axis=0), axis=2) / dt  # (T-1, 2): step t -> t+1
            for k, side in enumerate(("left", "right")):
                # per stance (contact run >= 5 steps): the touchdown skid = planar sole travel from contact onset until
                # the sole first moves slower than SETTLE_V; mid-stance = settled .. 3 steps before lift-off (the sole
                # ORIGIN moves while the foot rolls over its toe at lift-off)
                skids, mid = [], []
                for on, t0, L in _runs(c[:, k]):
                    if on and L >= 5 and t0 + L <= len(v):
                        run = v[t0:t0 + L, k]
                        settle = int(np.argmax(run < SETTLE_V)) if (run < SETTLE_V).any() else L
                        skids.append(float(run[:settle].sum() * dt))
                        mid.extend(run[settle:max(settle, L - 3)])
                if skids:
                    add("touchdown_skid", float(np.mean(skids)), motor=f"{side} foot",
                        detail=f"mean {np.mean(skids) * 100:.1f} cm (max {np.max(skids) * 100:.1f}) of sliding after "
                               f"touchdown over {len(skids)} steps")
                if len(mid) > 10:
                    p95 = float(np.percentile(mid, 95))
                    add("foot_slip", p95, motor=f"{side} foot", detail=f"p95 {p95:.2f} m/s in settled mid-stance")
    if "feet_lateral" in d.files:
        lat = np.asarray(d["feet_lateral"], dtype=float)
        body["feet_lateral_min_m"] = round(float(lat.min()), 3)
        add("crossed", float((lat < 0.05).mean()), detail=f"min lateral sole distance {lat.min() * 100:.1f} cm")
    if "base_vel_b" in d.files:
        vy = float(np.sqrt((np.asarray(d["base_vel_b"])[:, 1] ** 2).mean()))
        add("sway", vy, key="sway_vy", detail=f"RMS lateral base speed {vy:.2f} m/s")
    if "proj_grav" in d.files:
        g = np.asarray(d["proj_grav"], dtype=float)
        tilt = float(np.sqrt((g[:, :2] ** 2).sum(1).mean()))
        pitch, roll = np.rad2deg(np.arcsin(np.clip(g[:, 0].mean(), -1, 1))), np.rad2deg(np.arcsin(np.clip(-g[:, 1].mean(), -1, 1)))
        add("tilt", tilt, detail=f"RMS tilt {np.rad2deg(tilt):.1f} deg (mean root pitch {pitch:+.1f}, roll {roll:+.1f}; "
                                 f"compare with the reference's own lean)")
    if "anchor_z" in d.files:
        z = np.asarray(d["anchor_z"], dtype=float)
        if z[0] > 0.1:
            falls = int(((z[1:] < 0.55 * z[0]) & (z[:-1] >= 0.55 * z[0])).sum())
            body["falls"] = falls
            if falls:
                findings.append({"check": "fall", "motor": None, "value": falls, "severity": "HIGH",
                                 "detail": "anchor below 55 % of its start height"})
    order = {"HIGH": 0, "warn": 1}
    findings.sort(key=lambda f: (order[f["severity"]], f["check"], -f["value"]))
    fails = sorted({f"{f['check']}:{f['motor'] or 'body'}" for f in findings if f["severity"] == "HIGH" and f["check"] in HW_GATE})
    gate = {"pass": not fails, "fails": fails}
    return {"file": str(path), "meta": meta, "body": body, "hw_gate": gate, "findings": findings}


def _label(r: dict) -> str:
    f = Path(r["file"])
    return f.parent.name if f.name == "telemetry.npz" else f.stem.replace("telemetry_", "")


def to_md(results: list[dict]) -> str:
    out = ["# Telemetry issue scan", "",
           "HW gate = FAIL on any HIGH " + " / ".join(HW_GATE) + " finding (`tools/telemetry_issue_scan.py`).", "",
           "| rollout | motors | s | HW gate | gate failures (worst first) | falls | 200 Hz jumps > 25 % |",
           "|---|---|---|---|---|---|---|"]
    for r in results:
        fails = [f for f in r["findings"] if f["severity"] == "HIGH" and f["check"] in HW_GATE]
        top = ", ".join(f"{f['check']} {f['motor'] or ''} {f['value']:.2g}".strip() for f in fails[:4])
        more = f" (+{len(fails) - 4})" if len(fails) > 4 else ""
        jump = r["body"].get("torque_jump_any_motor")
        out.append(f"| {_label(r)} | {r['meta'].get('actuator_profile', '?')} | {r['body']['seconds']:.0f} | "
                   f"{'PASS' if r['hw_gate']['pass'] else 'FAIL'} | {top}{more} | {r['body'].get('falls', '?')} | "
                   f"{'' if jump is None or not np.isfinite(jump) else f'{jump * 100:.0f} %'} |")
    out.append("")
    for r in results:
        m = r["meta"]
        out.append(f"### {Path(r['file']).parent.name}/{Path(r['file']).name}")
        out.append(f"profile `{m.get('actuator_profile', '?')}`, {r['body']['seconds']} s, "
                   f"checkpoint `{Path(str(m.get('checkpoint', '?'))).parent.name}/{Path(str(m.get('checkpoint', '?'))).name}`"
                   f"; body: {json.dumps({k: v for k, v in r['body'].items() if k not in ('steps', 'seconds')})}")
        g = r["hw_gate"]
        out.append(f"**HW gate: {'PASS' if g['pass'] else 'FAIL'}**" + ("" if g["pass"] else f" ({', '.join(g['fails'])})"))
        out.append("")
        if not r["findings"]:
            out.append("No findings.\n")
            continue
        out.append("| severity | check | where | value | detail |")
        out.append("|---|---|---|---|---|")
        for f in r["findings"]:
            out.append(f"| {f['severity']} | {f['check']} | {f['motor'] or '-'} | {f['value']} | {f['detail']} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--limits_json", default=str(USD_LIMITS_JSON), help="USD joint dump for files without joint_limits")
    args = ap.parse_args()
    results = [scan(Path(f), Path(args.limits_json)) for f in args.files]
    md = to_md(results)
    print(md)
    if args.md:
        Path(args.md).write_text(md + "\n", encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
