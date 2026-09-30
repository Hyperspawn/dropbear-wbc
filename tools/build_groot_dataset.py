"""Assemble raw tabletop episodes into a GR00T-flavoured LeRobot v2 dataset (CPU; numpy + pyarrow).

Input: raw episodes written by ``scripts/tabletop_collect.py --record <raw>`` (``<raw>/<split>/<pid>_<policy>/`` with
``meta.json``, ``lowdim.npz`` and one mp4 per camera). Output (Isaac-GR00T ``getting_started/data_preparation.md``)::

    <out>/meta/info.json            codebase_version v2.1, fps 20, features (float32 vectors, video, int64 indices)
    <out>/meta/episodes.jsonl       {"episode_index", "tasks": [instruction], "length", + dropbear extras}
    <out>/meta/tasks.jsonl          {"task_index": 0, "task": instruction}
    <out>/meta/modality.json        state/action = left_arm [0:5], right_arm [5:10]; video ego_view (+ wrist views);
                                    annotation human.task_description -> task_index
    <out>/meta/stats.json           GR00T's statistics (mean/std/min/max/q01/q99 of every float feature; same
                                    formula as gr00t/data/stats.py::calculate_dataset_statistics)
    <out>/meta/episodes_stats.jsonl LeRobot v2.1 per-episode statistics
    <out>/meta/dropbear_tabletop.json  provenance (raw episode dirs, placements, calibration / USD SHA, policy, layout)
    <out>/data/chunk-000/episode_XXXXXX.parquet
    <out>/videos/chunk-000/observation.images.<view>/episode_XXXXXX.mp4
    <out>/dropbear_tabletop_config.py  GR00T NEW_EMBODIMENT modality config (copy of source/dropbear_wbc/groot/)

By default only SUCCESSFUL episodes of the chosen split(s) are included (imitation data); ``--include_failed`` keeps
all (the per-episode ``success`` flag is in episodes.jsonl either way). ``timestamp = frame_index / fps`` (LeRobot's
1e-4 s check); the simulated clock is kept in ``time.sim_s``.

    python tools/build_groot_dataset.py --raw data/groot/_raw/dropbear_tabletop_push_v1 --split train \
        --out data/groot/dropbear_tabletop_push_v1
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CONFIG_SRC = REPO / "source" / "dropbear_wbc" / "groot" / "dropbear_tabletop_config.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
FFMPEG_CANDIDATES = [Path(_paths.ffmpeg()).parent, _paths.isaac_sim_root() / "kit/python/Lib/site-packages/imageio_ffmpeg/binaries"]

STATE_NAMES = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
               "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll"]
ARM_MOTORS = ["LH_yaw", "LH_pitch", "LH_roll", "LH_elbow_joint", "LH_wrist_roll",
              "RH_yaw", "RH_pitch", "RH_roll", "RH_elbow_joint", "RH_wrist_roll"]
POSE7 = ["x", "y", "z", "qw", "qx", "qy", "qz"]
VIEW_KEYS = {"head": "ego_view", "left_wrist": "left_wrist_view", "right_wrist": "right_wrist_view"}
# raw low-dim key -> (parquet column, names); policy.* diagnostics stay in the raw episodes only
EXTRA_COLUMNS = {
    "observation.motor_q": ("observation.motor_q", "motor22"),
    "observation.motor_dq": ("observation.motor_dq", "motor22"),
    "action.motor_q": ("action.motor_q", ARM_MOTORS),
    "action.tau_ff": ("action.tau_ff", ARM_MOTORS),
    "observation.block_pose": ("observation.block_pose", POSE7),
    "observation.zone_pos": ("observation.zone_pos", ["x", "y", "z"]),
    "observation.left_hand_pose": ("observation.left_hand_pose", POSE7),
    "observation.right_hand_pose": ("observation.right_hand_pose", POSE7),
    "task.success_now": ("task.success_now", None),
    "time.sim_s": ("time.sim_s", None),
}


def motor22_names() -> list[str]:
    sys.path.insert(0, str(REPO / "source"))
    from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES

    return list(MOTOR_NAMES)


def ffmpeg_exe() -> str | None:
    for d in FFMPEG_CANDIDATES:
        hits = sorted(d.glob("ffmpeg*.exe")) if d.is_dir() else []
        if hits:
            return str(hits[-1])
    return shutil.which("ffmpeg")


def probe_video(path: Path, ff: str | None) -> tuple[int | None, tuple[int, int] | None]:
    """(frame count, (height, width)) of an mp4 from the ffmpeg binary's stream log; (None, None) without ffmpeg."""
    import re

    if ff is None:
        return None, None
    r = subprocess.run([ff, "-hide_banner", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
                       capture_output=True, text=True)
    frames, hw = None, None
    for line in (r.stderr or "").splitlines():
        if "frame=" in line:
            try:
                frames = int(line.split("frame=")[1].split()[0])
            except (ValueError, IndexError):
                pass
        if hw is None and "Video:" in line:
            m = re.search(r"[ ,](\d{2,5})x(\d{2,5})[ ,\[]", line)
            if m:
                hw = (int(m.group(2)), int(m.group(1)))
    return frames, hw


def feature_stats(v: np.ndarray) -> dict:
    """gr00t/data/stats.py::calculate_dataset_statistics on one feature (float32 input)."""
    v = np.asarray(v, dtype=np.float32).reshape(len(v), -1)
    return {"mean": np.mean(v, axis=0).tolist(), "std": np.std(v, axis=0).tolist(), "min": np.min(v, axis=0).tolist(),
            "max": np.max(v, axis=0).tolist(), "q01": np.quantile(v, 0.01, axis=0).tolist(),
            "q99": np.quantile(v, 0.99, axis=0).tolist()}


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--split", nargs="+", default=["train"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include_failed", action="store_true")
    ap.add_argument("--views", nargs="+", default=["head", "left_wrist", "right_wrist"],
                    help="raw cameras to include (those missing in an episode abort the build)")
    ap.add_argument("--force", action="store_true", help="replace an existing output dataset")
    ap.add_argument("--max_zone_err_px", type=float, default=15.0,
                    help="drop episodes whose zone render check (meta zone_check.err_px) exceeds this or is missing")
    ap.add_argument("--allow_blank_frames", action="store_true",
                    help="keep episodes whose meta.json reports blank (uniform) camera frames in an included view")
    args = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    eps = []
    for sp in args.split:
        for d in sorted((args.raw / sp).glob("*")):
            if (d / "meta.json").is_file() and (d / "lowdim.npz").is_file():
                m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                if m.get("success") or args.include_failed:
                    eps.append((d, m))
    if not eps:
        raise SystemExit(f"no episodes under {args.raw} for splits {args.split}")
    views = [v for v in args.views if all((d / f"{v}.mp4").is_file() for d, _ in eps)]
    missing = sorted(set(args.views) - set(views))
    skipped_blank = []
    if not args.allow_blank_frames:
        keep = []
        for d, m in eps:
            bad = {v: n for v, n in (m.get("blank_frames") or {}).items() if v in views and n > 0}
            (skipped_blank.append({"raw_dir": str(d), "blank_frames": bad}) if bad else keep.append((d, m)))
        eps = keep
        if skipped_blank:
            print(f"skipped {len(skipped_blank)} episode(s) with blank camera frames: {skipped_blank}")
        if not eps:
            raise SystemExit("no episodes left after dropping those with blank frames")
    if "head" not in views:
        raise SystemExit("every episode needs head.mp4 (the ego view)")
    skipped_zone = []
    keep = []
    for d, m in eps:
        zc = m.get("zone_check") or {}
        err = zc.get("err_px")
        (keep.append((d, m)) if err is not None and err <= args.max_zone_err_px
         else skipped_zone.append({"raw_dir": str(d), "zone_check": zc or None}))
    eps = keep
    if skipped_zone:
        print(f"skipped {len(skipped_zone)} episode(s) failing the zone render check: {skipped_zone}")
    if not eps:
        raise SystemExit("no episodes left after the zone render check")
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists (use --force)")
        for sub in ("meta", "data", "videos"):
            shutil.rmtree(args.out / sub, ignore_errors=True)
    (args.out / "meta").mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()
    m22 = motor22_names()
    fps = int(eps[0][1]["fps"])
    instruction = eps[0][1]["instruction"]
    tasks = {instruction: 0}
    all_cols: dict[str, list] = {}
    episodes_meta, ep_stats_lines, prov_eps = [], [], []
    video_shapes = {}
    gidx = 0
    for ei, (d, m) in enumerate(eps):
        if int(m["fps"]) != fps:
            raise SystemExit(f"{d}: fps {m['fps']} != {fps}")
        z = np.load(d / "lowdim.npz")
        arr = {k.replace("__", "."): z[k] for k in z.files}
        n = int(arr["action"].shape[0])
        if n < 2:
            raise SystemExit(f"{d}: only {n} frames")
        ti = tasks.setdefault(m["instruction"], len(tasks))
        cols = {
            "observation.state": arr["observation.state"].astype(np.float32),
            "action": arr["action"].astype(np.float32),
            "timestamp": (np.arange(n, dtype=np.float64) / fps).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ei, dtype=np.int64),
            "index": np.arange(gidx, gidx + n, dtype=np.int64),
            "task_index": np.full(n, ti, dtype=np.int64),
            "annotation.human.task_description": np.full(n, ti, dtype=np.int64),
            "next.reward": np.zeros(n, dtype=np.float32),
            "next.done": np.zeros(n, dtype=bool),
        }
        cols["next.reward"][-1] = 1.0 if m.get("success") else 0.0
        cols["next.done"][-1] = True
        for raw_k, (col, _) in EXTRA_COLUMNS.items():
            v = np.nan_to_num(arr[raw_k].astype(np.float32), nan=0.0)
            cols[col] = v[:, 0] if v.shape[1] == 1 else v
        # videos
        vinfo = {}
        for v in views:
            key = f"observation.images.{VIEW_KEYS[v]}"
            dst = args.out / "videos" / "chunk-000" / key / f"episode_{ei:06d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(d / f"{v}.mp4", dst)
            nf, hw = probe_video(dst, ff)
            if nf is not None and nf != n:
                raise SystemExit(f"{d}/{v}.mp4 has {nf} frames, parquet has {n}")
            vinfo[key] = nf
            shape = list(hw) if hw else (m.get("camera_shapes", {}).get(v) or [None, None])[:2]
            if None in shape:
                raise SystemExit(f"{d}/{v}.mp4: resolution unknown (no ffmpeg and no camera_shapes in meta.json)")
            if key not in video_shapes:
                video_shapes[key] = shape
            elif list(video_shapes[key]) != list(shape):
                raise SystemExit(f"{d}/{v}.mp4 is {shape}, earlier episodes {video_shapes[key]} (one size per view)")
        table = {}
        for k, v in cols.items():
            if v.ndim == 2:
                table[k] = pa.array(v.tolist(), pa.list_(pa.float32()))
            else:
                table[k] = pa.array(v)
        path = args.out / "data" / "chunk-000" / f"episode_{ei:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(table), path)
        for k, v in cols.items():
            all_cols.setdefault(k, []).append(v)
        gidx += n
        episodes_meta.append({"episode_index": ei, "tasks": [m["instruction"]], "length": n,
                              "dropbear": {"pid": m["pid"], "split": m["split"], "side": m["side"],
                                           "success": bool(m["success"]), "termination": m["termination"],
                                           "policy": m["policy"]}})
        st = {}
        for k, v in cols.items():
            if v.dtype == bool:
                continue
            vv = np.asarray(v, dtype=np.float64).reshape(n, -1)
            st[k] = {"min": vv.min(0).tolist(), "max": vv.max(0).tolist(), "mean": vv.mean(0).tolist(),
                     "std": vv.std(0).tolist(), "count": [n]}
        ep_stats_lines.append({"episode_index": ei, "stats": st})
        prov_eps.append({"episode_index": ei, "raw_dir": str(d.relative_to(REPO)) if d.is_relative_to(REPO) else str(d),
                         "pid": m["pid"], "split": m["split"], "side": m["side"], "success": m["success"],
                         "termination": m["termination"], "frames": n, "video_frames": vinfo,
                         "placement": m.get("placement"), "policy": m["policy"]})
    total = gidx
    # stats (GR00T formula over all float features)
    stats = {}
    features = {}
    for k, parts in all_cols.items():
        v = np.concatenate(parts, axis=0)
        if v.dtype == bool or v.dtype == np.int64:
            continue
        stats[k] = feature_stats(v)
    names = {"observation.state": STATE_NAMES, "action": STATE_NAMES}
    for raw_k, (col, nm) in EXTRA_COLUMNS.items():
        names[col] = m22 if nm == "motor22" else nm
    for k, parts in all_cols.items():
        v = parts[0]
        if v.dtype == bool:
            features[k] = {"dtype": "bool", "shape": [1], "names": None}
        elif v.dtype == np.int64:
            features[k] = {"dtype": "int64", "shape": [1], "names": None}
        else:
            w = 1 if v.ndim == 1 else int(v.shape[1])
            features[k] = {"dtype": "float32", "shape": [w], "names": names.get(k)}
    for v in views:
        key = f"observation.images.{VIEW_KEYS[v]}"
        h, w = video_shapes[key]
        features[key] = {"dtype": "video", "shape": [int(h), int(w), 3], "names": ["height", "width", "channels"],
                         "info": {"video.height": int(h), "video.width": int(w), "video.codec": "h264",
                                  "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": fps,
                                  "video.channels": 3, "has_audio": False}}
    info = {
        "codebase_version": "v2.1", "robot_type": "dropbear_fixed_base_arms", "total_episodes": len(eps),
        "total_frames": total, "total_tasks": len(tasks), "total_videos": len(eps) * len(views), "total_chunks": 1,
        "chunks_size": 1000, "fps": fps, "splits": {"train": f"0:{len(eps)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    meta = args.out / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes_meta), encoding="utf-8")
    (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n"
                                              for t, i in sorted(tasks.items(), key=lambda kv: kv[1])), encoding="utf-8")
    modality = {
        "state": {"left_arm": {"start": 0, "end": 5}, "right_arm": {"start": 5, "end": 10}},
        "action": {"left_arm": {"start": 0, "end": 5}, "right_arm": {"start": 5, "end": 10}},
        "video": {VIEW_KEYS[v]: {"original_key": f"observation.images.{VIEW_KEYS[v]}"} for v in views},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=4) + "\n", encoding="utf-8")
    (meta / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ep_stats_lines), encoding="utf-8")
    prov0 = eps[0][1].get("provenance", {})
    prov = {"schema": "dropbear-groot-dataset-provenance-v1",
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "tool": "tools/build_groot_dataset.py", "raw": str(args.raw), "splits": args.split,
            "include_failed": args.include_failed, "views": views, "views_missing_in_some_episodes": missing,
            "skipped_blank_frame_episodes": skipped_blank, "skipped_zone_check_episodes": skipped_zone,
            "max_zone_err_px": args.max_zone_err_px,
            "state_action_space": "semantic (G1-named) arm joint angles [rad], CONTRACTS section 2; left 5 then right 5",
            "motor_map": "SemanticMap.semantic_to_motor of the calibration below (dropbear_wbc.kinematics.semantic)",
            "calibration": prov0.get("calibration"), "calibration_sha256": prov0.get("calibration_sha256"),
            "usd_sha256_contract": prov0.get("usd_sha256_contract"), "placements_file": prov0.get("placements_file"),
            "placements_sha256": prov0.get("placements_sha256"), "layout": prov0.get("layout"),
            "push_params": prov0.get("push_params"), "control_hz": fps, "episodes": prov_eps}
    (meta / "dropbear_tabletop.json").write_text(json.dumps(prov, indent=1, default=str) + "\n", encoding="utf-8")
    # the GR00T modality config, with VIDEO_KEYS = exactly the views this dataset has (fail if the template changed)
    src = CONFIG_SRC.read_text(encoding="utf-8")
    template = 'VIDEO_KEYS = ["ego_view", "left_wrist_view", "right_wrist_view"]'
    if template not in src:
        raise SystemExit(f"{CONFIG_SRC}: VIDEO_KEYS line not found (update build_groot_dataset.py)")
    video_keys = [VIEW_KEYS[v] for v in views]
    (args.out / CONFIG_SRC.name).write_text(src.replace(template, f"VIDEO_KEYS = {video_keys!r}"), encoding="utf-8")
    n_succ = sum(1 for _, m in eps if m.get("success"))
    print(f"wrote {args.out}: {len(eps)} episodes ({n_succ} successful), {total} frames, views {views}"
          + (f", views dropped (missing in some episodes): {missing}" if missing else ""))
    print("ffmpeg frame check:", "on" if ff else "OFF (no ffmpeg found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
