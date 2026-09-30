"""Validate a Dropbear GR00T dataset (LeRobot v2 + meta/modality.json) statically and with Isaac-GR00T's own loader.

Static checks (any Python with numpy + pyarrow): the layout of getting_started/data_preparation.md; info.json features vs
parquet columns / dtypes / widths; episodes.jsonl lengths == parquet rows; LeRobot timestamps (frame_index / fps,
1e-4 s); contiguous global ``index``; task indices resolve in tasks.jsonl; modality.json slices inside the state /
action widths, video original keys declared as ``video`` features, annotation original key present; one mp4 per
(episode, video key) whose decoded frame count equals the parquet rows (PyAV if installed, else an ffmpeg binary);
stats.json has the six statistics with the right widths for every float feature (what gr00t/data/stats.py writes).

``--groot_repo <Isaac-GR00T clone>`` additionally imports the upstream modules READ-ONLY (``sys.dont_write_bytecode``,
nothing is written into the clone) and runs, on a scratch COPY of the dataset:

* ``gr00t.data.dataset.lerobot_episode_loader.LeRobotEpisodeLoader`` with the NEW_EMBODIMENT modality config registered
  by the dataset's ``dropbear_tabletop_config.py``, for every episode (``df = loader[i]``), plus
  ``get_dataset_statistics()``. torchcodec (their video backend, needs torch) is NOT installed: its
  ``get_frames_by_indices`` is replaced by a PyAV decoder with the same contract (frames at indices, NHWC uint8);
* ``gr00t.data.stats.generate_stats`` + ``generate_rel_stats`` (GR00T's own statistics and the RELATIVE-action
  statistics), compared with the dataset's stats.json.

    .venv-groot-check/Scripts/python.exe tools/validate_groot_dataset.py data/groot/<dataset> \
        --groot_repo $DROPBEAR_UPSTREAM/Isaac-GR00T --out logs/tabletop/groot_validate_<dataset>.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
FFMPEG_DIR = Path(_paths.ffmpeg()).parent


def count_frames(path: Path) -> tuple[int | None, str]:
    try:
        import av

        with av.open(str(path)) as c:
            n = sum(1 for _ in c.decode(video=0))
        return n, "pyav"
    except ImportError:
        pass
    exe = sorted(FFMPEG_DIR.glob("ffmpeg*.exe")) if FFMPEG_DIR.is_dir() else []
    ff = str(exe[-1]) if exe else shutil.which("ffmpeg")
    if not ff:
        return None, "none"
    r = subprocess.run([ff, "-hide_banner", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"], capture_output=True,
                       text=True)
    n = None
    for line in r.stderr.splitlines():
        if "frame=" in line:
            try:
                n = int(line.split("frame=")[1].split()[0])
            except (ValueError, IndexError):
                pass
    return n, "ffmpeg"


def static_checks(root: Path) -> dict:  # noqa: C901
    import pyarrow.parquet as pq

    errors, notes = [], []
    meta = root / "meta"
    for f in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json", "stats.json"):
        if not (meta / f).is_file():
            errors.append(f"missing meta/{f}")
    if errors:
        return {"ok": False, "errors": errors}
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    eps = [json.loads(line) for line in (meta / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = {t["task_index"]: t["task"] for t in
             (json.loads(line) for line in (meta / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    modality = json.loads((meta / "modality.json").read_text(encoding="utf-8"))
    stats = json.loads((meta / "stats.json").read_text(encoding="utf-8"))
    feats = info["features"]
    fps = float(info["fps"])
    for k in ("codebase_version", "fps", "chunks_size", "data_path", "features"):
        if k not in info:
            errors.append(f"info.json lacks {k}")
    if info.get("total_episodes") != len(eps):
        errors.append(f"total_episodes {info.get('total_episodes')} != {len(eps)} lines in episodes.jsonl")
    video_keys = [k for k, v in feats.items() if v.get("dtype") == "video"]
    low_feats = {k: v for k, v in feats.items() if v.get("dtype") != "video"}
    # modality
    widths = {}
    for mod, col in (("state", "observation.state"), ("action", "action")):
        w = int(feats[col]["shape"][0])
        widths[mod] = w
        for key, sl in modality.get(mod, {}).items():
            if not (0 <= sl["start"] < sl["end"] <= w):
                errors.append(f"modality.{mod}.{key} slice {sl} outside width {w}")
    for key, v in modality.get("video", {}).items():
        ok_key = v.get("original_key", f"observation.images.{key}")
        if ok_key not in video_keys:
            errors.append(f"modality.video.{key} -> {ok_key} is not a video feature")
    for key, v in modality.get("annotation", {}).items():
        ok_key = v.get("original_key", f"annotation.{key}")
        if ok_key not in low_feats:
            errors.append(f"modality.annotation.{key} -> column {ok_key} not in features")
    total, gidx_expected = 0, 0
    frame_backend = None
    ep_reports = []
    for e in eps:
        ei = int(e["episode_index"])
        chunk = ei // int(info["chunks_size"])
        p = root / info["data_path"].format(episode_chunk=chunk, episode_index=ei)
        if not p.is_file():
            errors.append(f"missing {p}")
            continue
        t = pq.read_table(p)
        n = t.num_rows
        rep = {"episode_index": ei, "rows": n}
        if n != int(e["length"]):
            errors.append(f"episode {ei}: {n} rows != length {e['length']}")
        cols = set(t.column_names)
        for k, f in low_feats.items():
            if k not in cols:
                errors.append(f"episode {ei}: feature column {k} missing in parquet")
                continue
            col = t.column(k)
            if f["dtype"] == "float32" and f["shape"][0] > 1:
                arr = np.asarray(col.to_pylist(), dtype=np.float64)
                if arr.shape != (n, f["shape"][0]):
                    errors.append(f"episode {ei}: {k} shape {arr.shape} != ({n}, {f['shape'][0]})")
                if not np.isfinite(arr).all():
                    errors.append(f"episode {ei}: {k} has non-finite values")
        extra = cols - set(low_feats)
        if extra:
            errors.append(f"episode {ei}: parquet columns not in info.features: {sorted(extra)}")
        ts = np.asarray(t.column("timestamp").to_pylist(), dtype=np.float64)
        fi = np.asarray(t.column("frame_index").to_pylist(), dtype=np.int64)
        if not np.array_equal(fi, np.arange(n)):
            errors.append(f"episode {ei}: frame_index is not 0..n-1")
        if np.abs(ts - fi / fps).max() > 1e-4:
            errors.append(f"episode {ei}: timestamps deviate from frame_index/fps by {np.abs(ts - fi / fps).max():.2e} s")
        gi = np.asarray(t.column("index").to_pylist(), dtype=np.int64)
        if not np.array_equal(gi, np.arange(gidx_expected, gidx_expected + n)):
            errors.append(f"episode {ei}: global index not contiguous")
        gidx_expected += n
        epi = set(t.column("episode_index").to_pylist())
        if epi != {ei}:
            errors.append(f"episode {ei}: episode_index column {sorted(epi)}")
        for k in ("task_index", "annotation.human.task_description"):
            if k in cols:
                bad = set(t.column(k).to_pylist()) - set(tasks)
                if bad:
                    errors.append(f"episode {ei}: {k} values {sorted(bad)} not in tasks.jsonl")
        for vk in video_keys:
            vp = root / info["video_path"].format(episode_chunk=chunk, video_key=vk, episode_index=ei)
            if not vp.is_file():
                errors.append(f"missing video {vp}")
                continue
            nf, frame_backend = count_frames(vp)
            rep[vk] = nf
            if nf is not None and nf != n:
                errors.append(f"episode {ei}: {vk} has {nf} frames, parquet {n}")
        ep_reports.append(rep)
        total += n
    if info.get("total_frames") != total:
        errors.append(f"total_frames {info.get('total_frames')} != {total}")
    for k, f in low_feats.items():
        if "float" not in f["dtype"]:
            continue
        s = stats.get(k)
        if s is None:
            errors.append(f"stats.json lacks float feature {k}")
            continue
        for st in ("mean", "std", "min", "max", "q01", "q99"):
            if st not in s or len(np.atleast_1d(s[st])) != int(f["shape"][0]):
                errors.append(f"stats.json {k}.{st} missing or wrong width")
    if frame_backend in (None, "none"):
        notes.append("video frame counts NOT checked (neither PyAV nor ffmpeg available)")
    return {"ok": not errors, "errors": errors, "notes": notes, "episodes": len(eps), "frames": total,
            "video_keys": video_keys, "frame_count_backend": frame_backend, "state_width": widths.get("state"),
            "action_width": widths.get("action"), "tasks": tasks, "per_episode": ep_reports}


def groot_checks(root: Path, groot_repo: Path) -> dict:
    """Run Isaac-GR00T's loader and stats code on a scratch copy of the dataset."""
    sys.dont_write_bytecode = True  # never write __pycache__ into the read-only upstream clone
    sys.path.insert(0, str(groot_repo))
    import importlib.util

    import av  # noqa: F401  (required: replaces torchcodec)

    out: dict = {"groot_repo": str(groot_repo)}
    try:
        out["groot_commit"] = subprocess.run(["git", "-C", str(groot_repo), "rev-parse", "HEAD"], capture_output=True,
                                             text=True).stdout.strip()
    except OSError:
        pass
    import gr00t.utils.video_utils as vu

    def pyav_frames(video_path, indices, decoder_kwargs=None):
        import av as _av

        want = [int(i) for i in np.asarray(indices).tolist()]
        frames = []
        with _av.open(str(video_path)) as c:
            for fr in c.decode(video=0):
                frames.append(fr.to_ndarray(format="rgb24"))
        arr = np.stack(frames)  # (T, H, W, 3) NHWC, as torchcodec with dimension_order="NHWC"
        return arr[want]

    vu.get_frames_by_indices = pyav_frames
    import gr00t.data.dataset.lerobot_episode_loader as lel

    lel.get_frames_by_indices = pyav_frames  # the loader imported the name directly
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag

    cfg_path = root / "dropbear_tabletop_config.py"
    modality = json.loads((root / "meta" / "modality.json").read_text(encoding="utf-8"))
    src = cfg_path.read_text(encoding="utf-8")
    views = list(modality["video"].keys())
    # datasets built before the builder wrote VIDEO_KEYS carry the 3-view template: register the views they have
    src = src.replace('VIDEO_KEYS = ["ego_view", "left_wrist_view", "right_wrist_view"]', f"VIDEO_KEYS = {views!r}")
    tmp = Path(tempfile.mkdtemp(prefix="groot_check_", dir=str(REPO / ".cache" / "tmp")))
    mod_file = tmp / "dropbear_tabletop_config_check.py"
    mod_file.write_text(src, encoding="utf-8")
    MODALITY_CONFIGS.pop(EmbodimentTag.NEW_EMBODIMENT.value, None)
    spec = importlib.util.spec_from_file_location("dropbear_tabletop_config_check", mod_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mcfg = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
    out["registered_modality"] = {k: {"keys": v.modality_keys, "delta_indices": v.delta_indices} for k, v in mcfg.items()}
    out["config_video_keys_match_modality_json"] = list(mcfg["video"].modality_keys) == views
    copy = tmp / root.name
    shutil.copytree(root, copy)
    loader = lel.LeRobotEpisodeLoader(copy, mcfg)
    per = []
    for i in range(len(loader)):
        df = loader[i]
        row = df.iloc[0]
        rep = {"episode": i, "rows": len(df), "columns": sorted(df.columns.tolist())}
        for k in ("state.left_arm", "state.right_arm", "action.left_arm", "action.right_arm"):
            rep[k] = list(np.asarray(row[k]).shape)
        for v in views:
            rep[f"video.{v}"] = list(np.asarray(row[f"video.{v}"]).shape)
        rep["language"] = row.get("language.annotation.human.task_description")
        per.append(rep)
    out["loader_episodes"] = per
    ds = loader.get_dataset_statistics()
    out["loader_statistics_keys"] = {k: sorted(v.keys()) for k, v in ds.items()}
    # GR00T's own stats generation (overwrites stats.json in the COPY) and relative-action statistics
    import gr00t.data.stats as gst

    ours = json.loads((copy / "meta" / "stats.json").read_text(encoding="utf-8"))
    (copy / "meta" / "stats.json").unlink()
    gst.generate_stats(copy)
    gst.generate_rel_stats(copy, EmbodimentTag.NEW_EMBODIMENT)
    theirs = json.loads((copy / "meta" / "stats.json").read_text(encoding="utf-8"))
    diffs = {}
    for k, v in theirs.items():
        if k.startswith("__") or k not in ours:
            continue
        diffs[k] = max(float(np.max(np.abs(np.asarray(v[s], float) - np.asarray(ours[k][s], float))))
                       for s in ("mean", "std", "min", "max", "q01", "q99"))
    out["stats_max_abs_diff_vs_groot"] = diffs
    rel = json.loads((copy / "meta" / "relative_stats.json").read_text(encoding="utf-8"))
    out["relative_stats"] = {k: {"shape": list(np.asarray(v["mean"]).shape)} for k, v in rel.items()
                             if not k.startswith("__")}
    out["relative_stats_file"] = str(copy / "meta" / "relative_stats.json")
    out["scratch_copy"] = str(copy)
    out["ok"] = (bool(per) and all(r["rows"] > 0 for r in per) and max(diffs.values(), default=0.0) < 1e-4
                 and out["config_video_keys_match_modality_json"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--groot_repo", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = {"dataset": str(args.dataset), "static": static_checks(args.dataset)}
    print("static:", "OK" if rep["static"]["ok"] else "FAIL", json.dumps(rep["static"].get("errors", [])[:20]))
    if args.groot_repo is not None:
        try:
            rep["groot"] = groot_checks(args.dataset, args.groot_repo)
            print("groot loader:", "OK" if rep["groot"]["ok"] else "FAIL",
                  "stats diff", rep["groot"]["stats_max_abs_diff_vs_groot"])
        except Exception as e:  # noqa: BLE001 - the report must say what failed
            import traceback

            rep["groot"] = {"ok": False, "error": repr(e), "traceback": traceback.format_exc()}
            print("groot loader: FAIL", repr(e))
    rep["ok"] = rep["static"]["ok"] and rep.get("groot", {"ok": True})["ok"]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
        print("wrote", args.out)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
