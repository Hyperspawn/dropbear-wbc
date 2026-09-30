"""Compact a recorded motor-twin rollout (+ its motor telemetry) for the browser demo in ``site/`` (GitHub Pages).

Input: ``scripts/play.py --record_rollout <rollout.npz> --telemetry <telemetry.npz>`` output (every link pose of env 0
at 50 Hz, and per-motor torque / contact). Output: ``site/data/<name>.bin`` + ``<name>.json``, and an entry in
``site/data/index.json``. Binary layout (little-endian, all arrays C order, ``T`` frames at ``--fps``):

==============  ======================  ====================================================================
root_pos        float32 (T, 3)          root body position [m]
rel_pos         int16   (T, B, 3)       every body's position minus the root's, in millimetres
quat            int16   (T, B, 4)       every body's orientation (w, x, y, z) * 32767
load            uint8   (T, M)          |motor torque| / rated torque * 100 (clipped at 255 = 2.55x rated)
contact         uint8   (T,)            bit 0 left foot, bit 1 right foot in contact
==============  ======================  ====================================================================

    python tools/export_web_rollout.py logs/hw_twin/live_final --name live_text_to_motion \\
        --title "Live text to motion" --events "8.84:a person waves hello with the right hand;22.38:..."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("run_dir", type=Path, help="folder with rollout.npz (and telemetry.npz)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--notes", default="", help="what it shows / proves / does not prove (one paragraph)")
    ap.add_argument("--events", default="", help="'t_s:caption;t_s:caption' shown at those times")
    ap.add_argument("--facts", default="", help="'key=value;key=value' shown in the side card")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--out", type=Path, default=REPO / "site" / "data")
    args = ap.parse_args()

    r = np.load(args.run_dir / "rollout.npz", allow_pickle=False)
    src_fps = float(r["fps"])
    step = max(1, int(round(src_fps / args.fps)))
    bp, bq = r["body_pos_w"][::step].astype(np.float64), r["body_quat_w"][::step].astype(np.float64)
    names = [str(n) for n in r["body_names"]]
    root = bp[:, names.index("world")] if "world" in names else bp[:, 0]
    rel = np.clip(np.round((bp - root[:, None]) * 1000.0), -32767, 32767).astype("<i2")
    q = bq / np.linalg.norm(bq, axis=-1, keepdims=True)
    q = np.clip(np.round(q * 32767.0), -32767, 32767).astype("<i2")
    T = rel.shape[0]
    meta: dict = {"name": args.name, "title": args.title, "subtitle": args.subtitle, "notes": args.notes,
                  "fps": src_fps / step, "frames": T, "duration_s": round(T * step / src_fps, 2),
                  "body_names": names, "events": [], "facts": {}}
    tel = args.run_dir / "telemetry.npz"
    load = np.zeros((T, 0), dtype=np.uint8)
    contact = np.zeros(T, dtype=np.uint8)
    if tel.is_file():
        t = np.load(tel, allow_pickle=False)
        tau = np.abs(np.asarray(t["tau"], dtype=np.float64))
        rated = np.asarray(t["rated_torque"], dtype=np.float64)
        # telemetry row k is the state after policy step k (rollout frame k+1): align on the rollout frames
        idx = np.clip(np.arange(T) * step - 1, 0, tau.shape[0] - 1)
        load = np.clip(np.round(tau[idx] / rated * 100.0), 0, 255).astype(np.uint8)
        c = np.asarray(t["contact"], dtype=bool)[idx] if "contact" in t.files else np.zeros((T, 2), bool)
        contact = (c[:, 0].astype(np.uint8) | (c[:, 1].astype(np.uint8) << 1))
        meta["motors"] = [{"name": str(n), "model": str(m), "rated_nm": round(float(rt), 2), "peak_nm": round(float(pk), 2)}
                          for n, m, rt, pk in zip(t["motor_names"], t["model"], t["rated_torque"], t["peak_torque"])]
        tm = json.loads(str(t["meta"])) if "meta" in t.files else {}
        meta["facts"]["actuator profile"] = tm.get("actuator_profile", "")
        meta["facts"]["policy"] = Path(str(tm.get("checkpoint", ""))).parent.name.split("_", 2)[-1] + " / " + \
            Path(str(tm.get("checkpoint", ""))).stem.replace("model_", "iter ")
        over = (tau[:, :] / rated) > 1.0
        meta["facts"]["motor-steps over rated torque"] = f"{100 * over.mean():.1f} %"
    for part in filter(None, (x.strip() for x in args.facts.split(";"))):
        k, v = part.split("=", 1)
        meta["facts"][k.strip()] = v.strip()
    for part in filter(None, (x.strip() for x in args.events.split(";"))):
        ts, cap = part.split(":", 1)
        meta["events"].append({"t": float(ts), "text": cap.strip()})
    blobs = [("root_pos", root.astype("<f4")), ("rel_pos", rel), ("quat", q), ("load", load), ("contact", contact)]
    offsets, off, buf = {}, 0, bytearray()
    for key, arr in blobs:
        b = np.ascontiguousarray(arr).tobytes()
        offsets[key] = {"offset": off, "bytes": len(b), "dtype": arr.dtype.str, "shape": list(arr.shape)}
        buf += b
        off += len(b)
        pad = (-off) % 4  # keep every array 4-byte aligned for the typed-array views
        buf += b"\0" * pad
        off += pad
    meta["layout"] = offsets
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.name}.bin").write_bytes(bytes(buf))
    (args.out / f"{args.name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    index_p = args.out / "index.json"
    index = json.loads(index_p.read_text(encoding="utf-8")) if index_p.is_file() else []
    index = [e for e in index if e["name"] != args.name] + [{"name": args.name, "title": args.title,
                                                              "subtitle": args.subtitle, "duration_s": meta["duration_s"]}]
    index_p.write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{args.name}: {T} frames at {meta['fps']:g} fps, {len(buf) / 2**20:.2f} MB, {len(meta['events'])} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
