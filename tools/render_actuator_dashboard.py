"""Overlay a live actuator dashboard on a rollout video: the 22 motors as the CAN bus would see them.

Inputs: a clean rollout mp4 (``scripts/play_locomotion.py --clean_video``) and the telemetry NPZ of the SAME rollout
(``--telemetry``; one row per video frame at 50 Hz). Output: the video with a translucent panel on the right:

* a table row per motor: CAN ID (0x140 + ID), joint, motor model, position [deg] (knees: knee-joint angle through the
  four-bar LUT), speed [rad/s], torque [N*m] and a bar of |torque| / peak (green <= rated, amber <= peak, red = the
  actuator model clipped the command). A speed bar marks |speed| / no-load speed;
* scrolling curves (last ``--window_s``) of knee torque L/R, knee angle L/R, and the worst 200 Hz torque step of the
  frame (the "spikiness" that a 48 V drive has to follow), with rated/peak reference lines.

System Python (numpy + Pillow + ffmpeg); no Isaac needed.

    python tools/render_actuator_dashboard.py --video <run>/videos/clean/velocity_model_3200_track.mp4 \
        --telemetry logs/hw_twin/telemetry_x.npz --out logs/brev/media/x_dashboard.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
CAL = REPO / "data/calibration/dropbear_semantic_calibration.json"

SHORT = {
    "PG_left_leg_pitch": "L hip roll", "PG_left_leg_roll": "L hip yaw", "LL_hip_joint": "L hip pitch",
    "LL_knee_actuator_joint": "L knee", "LL_Revolute67": "L calf A", "LL_Revolute81": "L calf B",
    "PG_right_leg_pitch": "R hip roll", "PG_right_leg_roll": "R hip yaw", "RL_hip_joint": "R hip pitch",
    "RL_knee_actuator_joint": "R knee", "RL_Revolute67": "R calf A", "RL_Revolute81": "R calf B",
    "LH_yaw": "L sh pitch", "LH_pitch": "L sh out", "LH_roll": "L arm rot", "LH_elbow_joint": "L elbow",
    "LH_wrist_roll": "L wrist", "RH_yaw": "R sh pitch", "RH_pitch": "R sh out", "RH_roll": "R arm rot",
    "RH_elbow_joint": "R elbow", "RH_wrist_roll": "R wrist",
}
MODEL_SHORT = {"RMD-X10-S2-V3-1:35": "X10-S2", "RMD-X10-V3-1:7": "X10 1:7", "RMD-X8-Pro-V2-1:9": "X8 Pro",
               "EPS-CEM-60": "CEM-60*"}

GREEN, AMBER, RED, GREY = (70, 200, 110), (240, 180, 40), (235, 70, 60), (120, 120, 130)
BG, FG, DIM = (18, 20, 26, 205), (235, 238, 242), (150, 155, 165)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("consolab.ttf" if bold else "consola.ttf"), "DejaVuSansMono.ttf"):
        for base in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")):
            if (base / name).is_file():
                return ImageFont.truetype(str(base / name), size)
    return ImageFont.load_default()


def ffmpeg() -> str:
    for c in ("C:/ProgramData/chocoportable/bin/ffmpeg.exe", shutil.which("ffmpeg")):
        if c and Path(c).exists():
            return str(c)
    raise FileNotFoundError("ffmpeg")


def probe(video: Path) -> tuple[int, int, float, int]:
    fp = str(Path(ffmpeg()).with_name("ffprobe.exe")) if Path(ffmpeg()).with_name("ffprobe.exe").exists() else "ffprobe"
    out = subprocess.run([fp, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return int(s["width"]), int(s["height"]), num / den, int(s["nb_read_packets"])


def knee_luts() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not CAL.is_file():
        return {}
    d = json.loads(CAL.read_text(encoding="utf-8"))["dofs"]
    return {"LL_knee_actuator_joint": (np.array(d["left_knee"]["motor_grid"]), np.array(d["left_knee"]["semantic_values"])),
            "RL_knee_actuator_joint": (np.array(d["right_knee"]["motor_grid"]), np.array(d["right_knee"]["semantic_values"]))}


class Dashboard:
    def __init__(self, tel: np.lib.npyio.NpzFile, width: int, height: int, window_s: float, panel_w: int,
                 gate: dict | None = None):
        self.t = tel
        self.gate = gate  # tools/telemetry_issue_scan.py verdict of the whole rollout (None = not scanned)
        self.names = [str(n) for n in tel["motor_names"]]
        self.ids = [int(i) for i in tel["can_ids"]]
        self.peak = tel["peak_torque"]
        self.rated = tel["rated_torque"] if "rated_torque" in tel.files else np.full(len(self.names), np.nan)
        self.no_load = tel["no_load_speed"] if "no_load_speed" in tel.files else np.full(len(self.names), np.nan)
        self.models = [MODEL_SHORT.get(str(m), str(m)) for m in tel["model"]] if "model" in tel.files else ["legacy"] * 22
        self.meta = json.loads(str(tel["meta"]))
        self.dt = float(tel["dt"])
        self.q, self.qd, self.tau, self.tau_pd = tel["q"], tel["qd"], tel["tau"], tel["tau_pd"]
        self.tau_sub = tel["tau_sub"]
        self.contact, self.cmd, self.vel = tel["contact"], tel["cmd"], tel["base_vel_b"]
        self.feet_lat = tel["feet_lateral"] if "feet_lateral" in tel.files else None
        self.T = len(self.q)
        self.clipped = np.abs(self.tau_pd - self.tau) > 0.01 * self.peak[None]
        luts = knee_luts()
        self.knee_deg = {}
        for m, (g, s) in luts.items():
            i = self.names.index(m)
            self.knee_deg[m] = np.degrees(np.interp(self.q[:, i], g, s))
        # worst torque step between consecutive 200 Hz samples inside each frame (+ the step into the frame)
        sub = self.tau_sub
        flat = sub.reshape(-1, sub.shape[-1])
        flat = flat[~np.isnan(flat).any(axis=1)] if np.isnan(flat).any() else flat
        steps = np.abs(np.diff(sub, axis=1))
        self.frame_step = np.nanmax(np.nan_to_num(steps, nan=0.0), axis=(1, 2)) if sub.shape[1] > 1 else np.zeros(self.T)
        self.frame_step_rel = np.nanmax(np.nan_to_num(steps / self.peak[None, None], nan=0.0), axis=(1, 2)) \
            if sub.shape[1] > 1 else np.zeros(self.T)
        self.W, self.H, self.win = width, height, int(round(window_s / self.dt))
        self.pw = panel_w
        self.f_small, self.f_row, self.f_head = font(11), font(12), font(13, bold=True)

    def bar(self, d: ImageDraw.ImageDraw, x, y, w, h, frac, color):
        d.rectangle([x, y, x + w, y + h], outline=(70, 72, 80))
        d.rectangle([x + 1, y + 1, x + 1 + max(0, min(1.0, frac)) * (w - 2), y + h - 1], fill=color)

    def curve(self, d, x, y, w, h, series, colors, lo, hi, refs, label, k):
        d.rectangle([x, y, x + w, y + h], fill=(28, 30, 38, 220), outline=(70, 72, 80))
        a = max(0, k - self.win + 1)
        for val, col in refs:
            if lo < val < hi:
                yy = y + h - (val - lo) / (hi - lo) * h
                for xx in range(x, x + w, 6):
                    d.line([xx, yy, xx + 3, yy], fill=col)
        for s, col in zip(series, colors):
            seg = s[a:k + 1]
            if len(seg) < 2:
                continue
            xs = x + w - (len(seg) - 1 - np.arange(len(seg))) * (w / max(1, self.win - 1))
            ys = y + h - (np.clip(seg, lo, hi) - lo) / (hi - lo) * h
            d.line(list(zip(xs.tolist(), ys.tolist())), fill=col, width=2)
        # label last, on a backing box, so the curves never draw over it
        tw = d.textlength(label, font=self.f_small)
        d.rectangle([x + 2, y + 2, x + 6 + tw, y + 15], fill=(28, 30, 38, 235))
        d.text((x + 4, y + 2), label, font=self.f_small, fill=DIM)

    def draw(self, frame: np.ndarray, k: int) -> np.ndarray:
        k = min(k, self.T - 1)
        img = Image.fromarray(frame).convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        x0 = self.W - self.pw - 8
        rh = 16
        top = 8
        table_h = 34 + rh * 22
        d.rounded_rectangle([x0, top, self.W - 8, top + table_h], radius=6, fill=BG)
        prof = self.meta.get("actuator_profile", "")
        d.text((x0 + 8, top + 4), f"ACTUATORS  CAN 1 Mbit/s  |  {prof}  |  t = {k * self.dt:5.2f} s", font=self.f_head, fill=FG)
        d.text((x0 + 8, top + 20), " ID   joint        motor    pos°   rad/s   N·m   |τ|/peak   speed", font=self.f_small,
               fill=DIM)
        for r, m in enumerate(self.names):
            y = top + 34 + r * rh
            i = r
            tau, qd = self.tau[k, i], self.qd[k, i]
            pos = self.knee_deg[m][k] if m in self.knee_deg else np.degrees(self.q[k, i])
            frac = abs(tau) / self.peak[i] if self.peak[i] > 0 else 0.0
            col = RED if self.clipped[k, i] else (GREEN if not np.isfinite(self.rated[i]) or abs(tau) <= self.rated[i]
                                                  else AMBER)
            if r in (6, 12):
                d.line([x0 + 6, y - 2, self.W - 14, y - 2], fill=(60, 62, 70))
            d.text((x0 + 8, y), f"{0x140 + self.ids[i]:03X}", font=self.f_row, fill=DIM)
            d.text((x0 + 44, y), f"{SHORT.get(m, m)[:11]:<11}", font=self.f_row, fill=FG)
            d.text((x0 + 130, y), f"{self.models[i][:7]:<7}", font=self.f_small, fill=DIM)
            d.text((x0 + 186, y), f"{pos:6.1f} {qd:6.2f} {tau:6.1f}", font=self.f_row, fill=FG)
            self.bar(d, x0 + 330, y + 3, 64, 10, frac, col)
            sf = abs(qd) / self.no_load[i] if np.isfinite(self.no_load[i]) and self.no_load[i] > 0 else 0.0
            self.bar(d, x0 + 400, y + 3, 30, 10, sf, RED if sf > 1.0 else (120, 160, 230))
        # curves
        cy = top + table_h + 8
        ch = 54
        li, ri = self.names.index("LL_knee_actuator_joint"), self.names.index("RL_knee_actuator_joint")
        pk = float(max(self.peak[li], self.peak[ri]))
        rt = float(np.nanmax([self.rated[li], self.rated[ri]])) if np.isfinite(self.rated[[li, ri]]).any() else np.nan
        cw = self.pw
        self.curve(d, x0, cy, cw, ch, [self.tau[:, li], self.tau[:, ri]], [(110, 190, 255), (255, 140, 90)],
                   -pk * 1.05, pk * 1.05, [(pk, RED), (-pk, RED)] + ([(rt, AMBER), (-rt, AMBER)] if np.isfinite(rt) else []),
                   "knee motor torque N·m  (L blue / R orange; red = peak, amber = rated)", k)
        if self.knee_deg:
            self.curve(d, x0, cy + ch + 6, cw, ch,
                       [self.knee_deg["LL_knee_actuator_joint"], self.knee_deg["RL_knee_actuator_joint"]],
                       [(110, 190, 255), (255, 140, 90)], -8, 52, [(0, GREY)], "knee angle °  (0 = straight, max ~48)", k)
        self.curve(d, x0, cy + 2 * (ch + 6), cw, ch, [self.frame_step_rel * 100.0], [(200, 200, 90)], 0, 60,
                   [(25, AMBER)], "worst 200 Hz torque step, % of peak (spikiness)", k)
        # status line
        c, v = self.cmd[k], self.vel[k]
        ct = self.contact[k]
        sy = cy + 3 * (ch + 6)
        d.rounded_rectangle([x0, sy - 2, self.W - 8, sy + (46 if self.gate else 32)], radius=4, fill=BG)
        d.text((x0 + 8, cy + 3 * (ch + 6) + 2),
               (f"cmd {c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f} | " if np.any(np.abs(c) > 1e-6) else "")
               + f"v {v[0]:+.2f},{v[1]:+.2f} m/s | feet {'L' if ct[0] else '-'}{'R' if ct[1] else '-'}"
               + (f" | width {100 * self.feet_lat[k]:+.0f} cm" if self.feet_lat is not None else ""),
               font=self.f_small, fill=(RED if self.feet_lat is not None and self.feet_lat[k] < 0.05 else FG))
        d.text((x0 + 8, cy + 3 * (ch + 6) + 16),
               "CAN IDs: arms = README, legs = planned (unverified) | *CEM-60 provisional",
               font=self.f_small, fill=DIM)
        if self.gate:
            g = self.gate
            txt = "HW gate (whole run): PASS" if g["pass"] else "HW gate (whole run): FAIL - " + ", ".join(
                SHORT.get(f.split(":")[1], f.split(":")[1]).replace(" foot", "") + " " + f.split(":")[0]
                for f in g["fails"][:3]) + (" ..." if len(g["fails"]) > 3 else "")
            d.text((x0 + 8, cy + 3 * (ch + 6) + 30), txt[:78], font=self.f_small, fill=GREEN if g["pass"] else RED)
        out = Image.alpha_composite(img, ov).convert("RGB")
        return np.asarray(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--telemetry", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window_s", type=float, default=4.0)
    ap.add_argument("--panel_w", type=int, default=440)
    ap.add_argument("--layout", choices=("side", "overlay"), default="side",
                    help="side: the panel sits BESIDE the video (wider canvas, the robot is never covered); overlay: on it")
    ap.add_argument("--telemetry_offset", type=int, default=0,
                    help="video frame k shows telemetry row k - offset (1 for kinematic replays of --record_rollout, whose "
                         "frame 0 is the initial state)")
    args = ap.parse_args()
    W, H, fps, n = probe(args.video)
    tel = np.load(args.telemetry, allow_pickle=True)
    CW = W + args.panel_w + 16 if args.layout == "side" else W
    CW += CW % 2  # even width for yuv420p
    gate = None
    try:  # whole-run hardware verdict (docs/ISSUES.md #22); the dashboard renders without it
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from telemetry_issue_scan import scan

        gate = scan(args.telemetry)["hw_gate"]
    except Exception as exc:  # noqa: BLE001
        print(f"[dashboard] no HW gate line: {type(exc).__name__}: {exc}", flush=True)
    dash = Dashboard(tel, CW, H, args.window_s, args.panel_w, gate=gate)
    if abs(n - dash.T) > 2:
        print(f"warning: video has {n} frames, telemetry {dash.T} rows (aligning by index)")
    dec = subprocess.Popen([ffmpeg(), "-v", "error", "-i", str(args.video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen([ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CW}x{H}",
                            "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                            str(args.out)], stdin=subprocess.PIPE)
    k = 0
    size = W * H * 3
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        if CW != W:
            canvas = np.empty((H, CW, 3), np.uint8)
            canvas[:] = (14, 16, 22)
            canvas[:, :W] = frame
            frame = canvas
        enc.stdin.write(dash.draw(frame, max(0, k - args.telemetry_offset)).tobytes())
        k += 1
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(json.dumps({"out": str(args.out), "frames": k, "telemetry_rows": dash.T}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
