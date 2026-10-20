"""Pre-training arm-duty screen for reference motion clips (backlog item 50, ISSUES #10/#24/#28c, FINDINGS #7).

The X8 Pro elbow drives a four-bar with |G| = d elbow / d motor = 3.7-5.5, worst (5.3-5.5) at the straight arm, so a
straight arm held raised (forearm horizontal) costs ~1.9 N*m at the joint and up to ~10.5 N*m at the motor: the
X8 Pro's rated (continuous) torque, with no payload. Clips that hold that pose train policies into thermal duty the
hardware cannot sustain. This tool flags them BEFORE training, from the reference alone.

Per clip, per arm motor (shoulder pitch/roll/yaw, elbow, wrist roll), on the serial MJCF surrogate
(``data/robot/dropbear_serial.xml``) with the clip's own root orientation:

* ``static``   quasi-static gravity load: qvel = qacc = 0, tau = g(q), mapped to the motor by J^T (J = d sem / d motor
               from ``SemanticMap``, so the elbow LUT ratio is included). Robust to noisy clip derivatives.
* ``dynamic``  inverse dynamics with the clip's joint + root velocities/accelerations: finite differences of a
               ``--lp-hz`` (6 Hz) zero-phase Gaussian low-pass of the reference (raw 50 Hz differences of a clip that
               reaches a joint stop in one frame give >1000 rad/s^2 and a 280 N*m elbow "demand").
               Arm dofs see no contact force in a tree, so no GRF model is needed for these motors.

Metrics (motor side, per motor model of ``--map``): ``peak`` = max |tau| / peak, ``rms`` = whole-clip RMS / rated (the
duty if the clip is looped), ``sus`` = worst sliding ``--window`` s RMS / rated (a held pose inside a longer clip),
``over_s`` = longest run with |tau| > rated, ``over_frac`` = share of frames above rated. A clip shorter than the window
uses its whole-clip RMS for ``sus``.

Severity (``THRESH``, same convention as ``telemetry_issue_scan.thermal``): dynamic ``sus`` >= 1.0 warn, >= 1.3 HIGH;
dynamic ``peak`` >= 1.0 HIGH (the reference asks for more than the motor's peak: tracking error by construction);
static ``sus`` >= 0.8 warn, >= 1.0 HIGH (gravity-only pose hold; with zero payload the worst X8 Pro elbow pose is
~0.93x rated, so any payload or brace pushes a warned hold over rated). ``--fail-on high|warn`` exits 1 when any clip reaches that level. This is a REFERENCE screen: the
trained policy can brace differently (ISSUES #10), so ``telemetry_issue_scan`` (``thermal`` + ``sustained``) remains the
gate on rollouts. Payload is zero (no grasp point in any model).

usage: python tools/motion_duty_screen.py <clip.npz | dir | library.json> [...] [--map user_2026_09_25]
       [--window 2.0] [--md out.md] [--json out.json] [--fail-on high]
Needs mujoco (requirements/tests-cpu.txt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "source") not in sys.path:
    sys.path.insert(0, str(REPO / "source"))

ARM_SUFFIX = ("_yaw", "_pitch", "_roll", "_elbow_joint", "_wrist_roll")
THRESH = {"sus": (1.0, 1.3), "peak": (1.0, 1.0), "static_sus": (0.8, 1.0)}
DEFAULT_WINDOW_S = 2.0
DEFAULT_LP_HZ = 6.0  # zero-phase Gaussian low-pass before differentiating (50 Hz clips hit joint stops in one frame)
J_REFRESH = 5  # frames between motor-Jacobian refreshes (arm LUTs are smooth)


# ---------------------------------------------------------------- pure numpy helpers (unit-tested without mujoco)
def window_rms_max(x: np.ndarray, n: int) -> np.ndarray:
    """Max over sliding windows of length ``n`` frames of the RMS of ``x`` (T, k) -> (k,). n >= T -> whole RMS."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    T = len(x)
    if T == 0:
        return np.full(x.shape[1], np.nan)
    if n >= T or n < 1:
        return np.sqrt((x ** 2).mean(0))
    c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x ** 2, axis=0)])
    return np.sqrt(((c[n:] - c[:-n]) / n).max(0))


def gaussian_lp(x: np.ndarray, dt: float, fc_hz: float | None) -> np.ndarray:
    """Zero-phase Gaussian low-pass along axis 0, -3 dB at ``fc_hz`` (sigma = 0.1325 / fc). None/<=0 = passthrough."""
    x = np.asarray(x, dtype=float)
    if not fc_hz or fc_hz <= 0 or len(x) < 3:
        return x
    sig = 0.1325 / fc_hz / dt  # frames
    r = max(1, int(np.ceil(4 * sig)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.empty_like(x)
    flat = pad.reshape(len(pad), -1)
    o = out.reshape(len(x), -1)
    for j in range(flat.shape[1]):
        o[:, j] = np.convolve(flat[:, j], k, mode="valid")
    return out


def longest_run(mask: np.ndarray) -> np.ndarray:
    """Longest run of True along axis 0, (T, k) -> (k,) frames."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask[:, None]
    out = np.zeros(mask.shape[1], dtype=int)
    for j in range(mask.shape[1]):
        best = cur = 0
        for v in mask[:, j]:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        out[j] = best
    return out


def severity(metric: str, value: float) -> str:
    warn, high = THRESH[metric]
    if not np.isfinite(value):
        return ""
    if value >= high:
        return "HIGH"
    return "warn" if value >= warn else ""


def duty_metrics(tau: np.ndarray, dt: float, rated: np.ndarray, peak: np.ndarray, window_s: float) -> dict:
    """Per-motor duty metrics of a (T, k) motor-torque series."""
    a = np.abs(np.asarray(tau, dtype=float))
    over = a > rated[None, :]
    n = max(1, int(round(window_s / dt)))
    return {
        "peak": a.max(0) / peak,
        "rms": np.sqrt((a ** 2).mean(0)) / rated,
        "sus": window_rms_max(a, n) / rated,
        "over_s": longest_run(over) * dt,
        "over_frac": over.mean(0),
        "window_covered": len(a) > n,
    }


# ---------------------------------------------------------------- clip loading
def clip_paths(args: list[str]) -> list[Path]:
    out: list[Path] = []
    for a in args:
        p = Path(a)
        if p.suffix == ".json":
            lib = json.loads(p.read_text(encoding="utf-8"))
            for c in lib.get("clips", []):
                q = Path(c["npz"])
                q = q if q.is_absolute() else (p.parent / q)
                out.append(q.resolve())
        elif p.is_dir():
            out.extend(sorted(p.rglob("*.npz")))
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def load_clip(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    need = ("fps", "motor_target", "motor_names", "body_pos_w", "body_quat_w")
    miss = [k for k in need if k not in z.files]
    if miss:
        raise KeyError(f"{path.name}: missing {miss}")
    body = [str(b) for b in z["body_names"]] if "body_names" in z.files else []
    if body and body[0] != "world":
        raise ValueError(f"{path.name}: body 0 is {body[0]!r}, expected the USD 'world' (= MJCF pelvis) root")
    out = {
        "name": path.stem, "path": str(path), "fps": float(z["fps"]),
        "qm": np.asarray(z["motor_target"], dtype=float),
        "motor_names": [str(n) for n in z["motor_names"]],
        "root_pos": np.asarray(z["body_pos_w"][:, 0], dtype=float),
        "root_quat": np.asarray(z["body_quat_w"][:, 0], dtype=float),  # wxyz
    }
    for k in ("body_lin_vel_w", "body_ang_vel_w"):
        if k in z.files:
            out[k] = np.asarray(z[k][:, 0], dtype=float)
    return out


# ---------------------------------------------------------------- plant
class ArmDutyPlant:
    def __init__(self, xml: Path | None = None, calib: Path | None = None):
        import mujoco

        from dropbear_wbc.kinematics.semantic import MOTOR_NAMES, SEMANTIC_NAMES, SemanticMap

        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(xml or REPO / "data/robot/dropbear_serial.xml"))
        self.d = mujoco.MjData(self.m)
        self.sm = SemanticMap(json.loads(Path(calib or REPO / "data/calibration/dropbear_semantic_calibration.json")
                                         .read_text(encoding="utf-8")))
        self.motor_names = list(MOTOR_NAMES)
        self.sem_names = list(SEMANTIC_NAMES)
        self.arm_motors = [i for i, n in enumerate(self.motor_names)
                           if n.startswith(("LH_", "RH_")) and n.endswith(ARM_SUFFIX)]
        self.sem_qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in self.sem_names])
        self.sem_dadr = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in self.sem_names])
        # Clip body 0 is the USD 'world' body; in the MJCF it is the site ``usd_world`` on the pelvis (offset ~1.1 m,
        # same orientation). Convert clip root poses to the pelvis free joint through it.
        sid = self.m.site("usd_world").id
        if not np.allclose(self.m.site_quat[sid], [1.0, 0.0, 0.0, 0.0], atol=1e-6):
            raise ValueError("usd_world site is rotated in the pelvis frame; root conversion assumes identity")
        self.root_site_off = self.m.site_pos[sid].copy()

    def to_sem(self, qm: np.ndarray) -> np.ndarray:
        s = self.sm.motor_to_semantic(qm, clip=False)
        return np.asarray(s[0] if isinstance(s, tuple) else s, dtype=float)

    def jac_arm(self, qm: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """J[:, j] = d sem / d motor_j for the arm motors only, (n_sem, n_arm)."""
        J = np.zeros((len(self.sem_names), len(self.arm_motors)))
        for c, j in enumerate(self.arm_motors):
            a = qm.copy(); b = qm.copy()
            a[j] += h; b[j] -= h
            J[:, c] = (self.to_sem(a) - self.to_sem(b)) / (2 * h)
        return J

    def _set(self, root_pos, root_quat, qs, qd_s=None, root_v=None, root_w_local=None):
        d = self.d
        d.qpos[:] = 0.0
        qn = root_quat / np.linalg.norm(root_quat)
        R = np.zeros(9)
        self.mj.mju_quat2Mat(R, qn)
        d.qpos[0:3] = root_pos - R.reshape(3, 3) @ self.root_site_off
        d.qpos[3:7] = qn
        d.qpos[self.sem_qadr] = qs
        d.qvel[:] = 0.0
        if qd_s is not None:
            d.qvel[self.sem_dadr] = qd_s
        if root_v is not None:
            d.qvel[0:3] = root_v
        if root_w_local is not None:
            d.qvel[3:6] = root_w_local

    def screen_torques(self, clip: dict, dynamic: bool = True, lp_hz: float | None = DEFAULT_LP_HZ) -> dict:
        """(T, n_arm) motor torques: static gravity and (optionally) dynamic inverse dynamics."""
        mj, m, d = self.mj, self.m, self.d
        names = clip["motor_names"]
        if names != self.motor_names:
            idx = [names.index(n) for n in self.motor_names]
            qm_all = clip["qm"][:, idx]
        else:
            qm_all = clip["qm"]
        T, dt = len(qm_all), 1.0 / clip["fps"]
        qs = np.stack([self.to_sem(q) for q in qm_all])
        qsf = gaussian_lp(qs, dt, lp_hz)
        qd = np.gradient(qsf, dt, axis=0) if T > 2 else np.zeros_like(qs)
        qdd = np.gradient(gaussian_lp(qd, dt, lp_hz), dt, axis=0) if T > 2 else np.zeros_like(qs)
        v = clip.get("body_lin_vel_w")
        if v is None:
            v = np.gradient(gaussian_lp(clip["root_pos"], dt, lp_hz), dt, axis=0) if T > 2 else np.zeros((T, 3))
        v = gaussian_lp(v, dt, lp_hz)
        a_lin = np.gradient(v, dt, axis=0) if T > 2 else np.zeros((T, 3))
        w_world = clip.get("body_ang_vel_w")
        if w_world is None:
            w_world = np.zeros((T, 3))
        w_world = gaussian_lp(w_world, dt, lp_hz)
        a_ang = np.gradient(w_world, dt, axis=0) if T > 2 else np.zeros((T, 3))
        arm_d = self.sem_dadr  # all semantic dofs; the motor map picks the arm part
        tau_s = np.zeros((T, len(self.arm_motors)))
        tau_d = np.zeros_like(tau_s)
        J = None
        R = np.zeros(9)
        for i in range(T):
            if J is None or i % J_REFRESH == 0:
                J = self.jac_arm(qm_all[i])
            self._set(clip["root_pos"][i], clip["root_quat"][i], qs[i])
            d.qacc[:] = 0.0
            mj.mj_inverse(m, d)
            tau_s[i] = J.T @ d.qfrc_inverse[arm_d]
            if dynamic:
                mj.mju_quat2Mat(R, d.qpos[3:7] / np.linalg.norm(d.qpos[3:7]))
                Rm = R.reshape(3, 3)
                self._set(clip["root_pos"][i], clip["root_quat"][i], qsf[i], qd[i], v[i], Rm.T @ w_world[i])
                d.qacc[:] = 0.0
                d.qacc[0:3] = a_lin[i]
                d.qacc[3:6] = Rm.T @ a_ang[i]
                d.qacc[self.sem_dadr] = qdd[i]
                mj.mj_inverse(m, d)
                tau_d[i] = J.T @ d.qfrc_inverse[arm_d]
        out = {"dt": dt, "static": tau_s, "qs": qs}
        if dynamic:
            out["dynamic"] = tau_d
        return out

    def fk_check(self, clip: dict, z_path: Path, frames: int = 5) -> dict:
        """Max |MJCF ``usd_<body>`` site - clip body_pos_w| [m] per tracked body over a few frames (validates the
        root-orientation convention and the semantic map on this clip)."""
        mj, m, d = self.mj, self.m, self.d
        z = np.load(z_path, allow_pickle=True)
        body = [str(b) for b in z["body_names"]]
        sites = [(m.site(k).name, m.site(k).name[4:]) for k in range(m.nsite) if m.site(k).name.startswith("usd_")]
        worst = {b: 0.0 for _, b in sites if b in body}
        for i in np.linspace(0, len(clip["qm"]) - 1, frames).astype(int):
            self._set(clip["root_pos"][i], clip["root_quat"][i], self.to_sem(clip["qm"][i]))
            mj.mj_kinematics(m, d)
            for site, b in sites:
                if b in worst:
                    e = float(np.linalg.norm(d.site_xpos[m.site(site).id] - z["body_pos_w"][i, body.index(b)]))
                    worst[b] = max(worst[b], e)
        return worst


def screen(paths: list[Path], map_name: str = "user_2026_09_25", window_s: float = DEFAULT_WINDOW_S,
           dynamic: bool = True, plant: ArmDutyPlant | None = None, lp_hz: float | None = DEFAULT_LP_HZ) -> dict:
    from dropbear_wbc.robots.hw_motor_specs import JOINT_MOTOR_MAPS, MOTOR_MODELS

    plant = plant or ArmDutyPlant()
    mmap = JOINT_MOTOR_MAPS[map_name]
    arm = [plant.motor_names[j] for j in plant.arm_motors]
    models = [MOTOR_MODELS[mmap[n]] for n in arm]
    rated = np.array([mm.rated_torque for mm in models])
    peak = np.array([mm.peak_torque for mm in models])
    clips, errors = [], []
    for p in paths:
        try:
            c = load_clip(p)
            t = plant.screen_torques(c, dynamic=dynamic, lp_hz=lp_hz)
        except (OSError, KeyError, ValueError) as e:
            errors.append({"clip": str(p), "error": str(e)})
            continue
        rec = {"clip": c["name"], "path": c["path"], "frames": len(c["qm"]), "seconds": len(c["qm"]) * t["dt"],
               "motors": {}, "findings": []}
        for kind in ("static", "dynamic"):
            if kind not in t:
                continue
            mt = duty_metrics(t[kind], t["dt"], rated, peak, window_s)
            for k, n in enumerate(arm):
                r = rec["motors"].setdefault(n, {"model": models[k].name, "rated": rated[k], "peak": peak[k]})
                r[kind] = {key: (float(val[k]) if isinstance(val, np.ndarray) else bool(val)) for key, val in mt.items()}
        gate_kind = "dynamic" if dynamic else "static"
        for n, r in rec["motors"].items():
            g = r[gate_kind]
            for metric in ("sus", "peak"):
                sev = severity(metric, g[metric])
                if sev:
                    rec["findings"].append({"motor": n, "metric": metric, "value": round(g[metric], 3),
                                            "severity": sev, "basis": gate_kind})
            sev = severity("static_sus", r["static"]["sus"])  # gravity-only pose hold (straight arm raised)
            if sev:
                rec["findings"].append({"motor": n, "metric": "static_sus", "value": round(r["static"]["sus"], 3),
                                        "severity": sev, "basis": "static"})
        sevs = {f["severity"] for f in rec["findings"]}
        rec["worst"] = "HIGH" if "HIGH" in sevs else ("warn" if "warn" in sevs else "ok")
        clips.append(rec)
    return {"schema": "dropbear-motion-duty-screen-v1", "map": map_name, "window_s": window_s, "lp_hz": lp_hz,
            "basis": "dynamic" if dynamic else "static", "payload_kg": 0.0, "thresh": THRESH,
            "clips": clips, "errors": errors}


def to_markdown(res: dict) -> str:
    L = [f"# Arm duty screen (map {res['map']}, window {res['window_s']} s, basis {res['basis']}, payload 0 kg)", "",
         f"Dynamic = inverse dynamics on a {res.get('lp_hz')} Hz zero-phase low-passed reference; static = gravity only. "
         "Columns are motor-side ratios: sus = worst-window RMS / rated, rms = whole-clip RMS / rated, peak = max / peak.",
         "", "| clip | s | worst | elbow L dyn sus / rms / peak | elbow R dyn sus / rms / peak | elbow L/R static sus | other findings |",
         "|---|---|---|---|---|---|---|"]
    b = res["basis"]
    for c in sorted(res["clips"], key=lambda c: -max(c["motors"][n][b]["sus"] for n in c["motors"])):
        def el(n):
            r = c["motors"][n][b]
            return f"{r['sus']:.2f} / {r['rms']:.2f} / {r['peak']:.2f}"
        other = ", ".join(f"{f['motor']} {f['metric']} {f['value']:.2f} {f['severity']}" for f in c["findings"]
                          if "elbow" not in f["motor"])
        st = " / ".join(f"{c['motors'][n]['static']['sus']:.2f}" for n in ("LH_elbow_joint", "RH_elbow_joint"))
        L.append(f"| {c['clip']} | {c['seconds']:.1f} | {c['worst']} | {el('LH_elbow_joint')} | {el('RH_elbow_joint')} | "
                 f"{st} | {other or '-'} |")
    for e in res["errors"]:
        L.append(f"\nERROR {e['clip']}: {e['error']}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help="clip .npz, directory of clips, or a library manifest .json")
    ap.add_argument("--map", default="user_2026_09_25")
    ap.add_argument("--window", type=float, default=DEFAULT_WINDOW_S, help="sustained-RMS window [s]")
    ap.add_argument("--static-only", action="store_true", help="gravity only (skip finite-difference dynamics)")
    ap.add_argument("--lp-hz", type=float, default=DEFAULT_LP_HZ, help="reference low-pass before differentiating; 0 = raw")
    ap.add_argument("--md"); ap.add_argument("--json")
    ap.add_argument("--fail-on", choices=("high", "warn"), default=None)
    a = ap.parse_args(argv)
    res = screen(clip_paths(a.inputs), a.map, a.window, dynamic=not a.static_only, lp_hz=a.lp_hz)
    md = to_markdown(res)
    print(md)
    if a.md:
        Path(a.md).write_text(md, encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    worst = {c["worst"] for c in res["clips"]}
    if res["errors"]:
        return 2
    if a.fail_on == "high" and "HIGH" in worst:
        return 1
    if a.fail_on == "warn" and worst & {"HIGH", "warn"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
