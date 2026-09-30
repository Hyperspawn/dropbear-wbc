"""Maintainer tool: build ``assets_manifest.json`` (checksums) and upload the large files to the Hugging Face Hub.

Everything too big or too binary for git (the 421 MB plant USD, trained policies, dashboard videos, redistributable
motion clips, the GR00T tabletop dataset) is listed in ``ASSETS`` below, hashed into ``assets_manifest.json``
(committed), and uploaded in one commit to ``--repo``. Users fetch it with ``tools/fetch_assets.py``.

    python tools/publish_assets.py --manifest-only            # hash + write assets_manifest.json
    python tools/publish_assets.py --repo Hyperspawn/dropbear-wbc --private   # + create the repo and upload
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402

R = "logs/brev/remote"
LIB = f"{R}/dbw3/logs/rsl_rl/dropbear_tracking_library"
# (group, path in the HF repo, local source file or glob, local destination for fetch_assets.py)
POLICIES = {
    "lib_v7gen_v1ie_smooth": (f"{LIB}/2026-09-26_07-52-45_lib_v7gen_v1ie_smooth", 13600),
    "lib_v6ts_v1i_fix2": (f"{R}/dbw1/logs/rsl_rl/dropbear_tracking_library/2026-09-26_05-31-13_lib_v6ts_v1i_fix2", 13700),
    "lib_v6ts_nostate_v1i_fix": (f"{R}/dbw2/logs/rsl_rl/dropbear_tracking_library/2026-09-25_21-25-05_lib_v6ts_nostate_v1i_fix", 10492),
    "kimodo_ts119_allfix_smooth": (f"{R}/dbw3/logs/rsl_rl/dropbear_tracking/2026-09-25_22-52-53_kimodo_ts119_allfix_smooth", 5800),
    "lafan_nostate_smooth_elbow": (f"{R}/dbw2/logs/rsl_rl/dropbear_tracking/2026-09-26_05-31-33_lafan_nostate_smooth_elbow", 8000),
    "lafan_cyclic_nostate_v1i_kstop_feet": (f"{R}/dbw1/logs/rsl_rl/dropbear_tracking/*_lafan_cyclic_nostate_v1i_kstop_feet", 3100),
    "vel_rough_v1i_b": (f"{R}/dbw0/logs/rsl_rl/dropbear_velocity/2026-09-26_08-52-39_vel_rough_v1i_b", 5500),
    "vel_hw_v1_gait3b_ft": (f"{R}/dbw0/logs/rsl_rl/dropbear_velocity/2026-09-25_09-46-22_vel_hw_v1_gait3b_ft", 1300),
}
MEDIA = ["live_final_dashboard", "live_kimodo1_dashboard", "live_text_demo3_dashboard", "kimodo_gen_wave_cem60_dashboard",
         "kimodo_walk_froude_allfix_smooth_5800_dashboard", "kimodo_walk_hwpass_3184_dashboard",
         "lafan_walk_smooth_elbow_8000_dashboard", "lafan_walk_deployable_kstop_feet_3100_dashboard",
         "lafan_walk_4way_compare", "rough_terrain_b_5500_dashboard", "gait3b_ft_1200_dashboard",
         "asap_walk_hw_reg_4700_dashboard", "take102_reference_vs_cloud_policy"]
MOTIONS = ["data/motions_v6ts/kimodo_gen/*", "data/motions_v6ts/kimodo_g1/*", "data/motions_v6ts/asap_g1/*",
           "data/motions_cyclic/kimodo_walk*", "data/motions_gen/raw/**/*", "data/motions_gen/kimodo_g1/*"]
DATASETS = ["data/groot/dropbear_tabletop_push_v1/**/*"]


def entries() -> list[dict]:
    out = [{"group": "usd", "hf": "robot/dropbear.usd", "src": str(_paths.usd_path()), "dst": "assets/dropbear.usd"}]
    for name, (run_glob, it) in POLICIES.items():
        run = Path(sorted(glob.glob(str(REPO / run_glob)))[-1])
        # the checkpoint + what play.py / play_locomotion.py read (run_info*.json: calibration, actuator profile, target
        # clamp, terrain) + the training record (chunks, warm start, params/*.yaml)
        infos = [run / "run_info.json"] if (run / "run_info.json").is_file() else sorted(run.glob("run_info_resume_*.json"))[-1:]
        files = [run / f"model_{it}.pt", *infos, run / "chunks.json", run / "warm_start.json",
                 run / "params" / "env.yaml", run / "params" / "agent.yaml"]  # no pickles, no per-resume copies
        for f in files:
            if f.is_file():
                rel = f.relative_to(run).as_posix()
                out.append({"group": "policies", "hf": f"policies/{name}/{rel}", "src": str(f),
                            "dst": f"assets/policies/{name}/{rel}"})
    for m in MEDIA:
        f = REPO / "logs/brev/media" / f"{m}.mp4"
        out.append({"group": "media", "hf": f"media/{f.name}", "src": str(f), "dst": f"assets/media/{f.name}"})
    for pattern, group, prefix in [(p, "motions", "motions") for p in MOTIONS] + [(p, "datasets", "datasets") for p in DATASETS]:
        for f in sorted(glob.glob(str(REPO / pattern), recursive=True)):
            f = Path(f)
            if f.is_file():
                rel = f.relative_to(REPO / "data").as_posix()
                out.append({"group": group, "hf": f"{prefix}/{rel}", "src": str(f), "dst": f"data/{rel}"})
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--repo", default="Hyperspawn/dropbear-wbc", help="Hugging Face repo id (model repo)")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--private", action="store_true", help="create the HF repo private")
    ap.add_argument("--groups", default="usd,policies,media,motions,datasets")
    args = ap.parse_args()
    groups = set(args.groups.split(","))
    items = [e for e in entries() if e["group"] in groups]
    missing = [e["src"] for e in items if not Path(e["src"]).is_file()]
    if missing:
        raise SystemExit(f"missing sources: {missing[:5]}")
    t0 = time.time()
    for e in items:
        p = Path(e["src"])
        e["bytes"] = p.stat().st_size
        e["sha256"] = sha256(p)
    if items[0]["group"] == "usd" and items[0]["sha256"] != _paths.USD_SHA256:
        raise SystemExit(f"USD {items[0]['src']} sha256 {items[0]['sha256'][:12]} is not the contract plant")
    manifest = {"schema": "dropbear-wbc-assets-v1", "hf_repo": args.repo, "hf_repo_type": "model",
                "files": [{k: e[k] for k in ("group", "hf", "dst", "bytes", "sha256")} for e in items]}
    (REPO / "assets_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    by = {}
    for e in items:
        by.setdefault(e["group"], [0, 0])
        by[e["group"]][0] += 1
        by[e["group"]][1] += e["bytes"]
    print(json.dumps({g: f"{n} files, {b / 2**20:.1f} MB" for g, (n, b) in by.items()}), f"hashed in {time.time() - t0:.0f} s")
    if args.manifest_only:
        return 0
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    card = REPO / "docs" / "HF_MODEL_CARD.md"
    ops = [CommitOperationAdd(path_in_repo=e["hf"], path_or_fileobj=e["src"]) for e in items]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card)))
    ops.append(CommitOperationAdd(path_in_repo="assets_manifest.json", path_or_fileobj=str(REPO / "assets_manifest.json")))
    api.create_commit(args.repo, repo_type="model", operations=ops,
                      commit_message=f"dropbear-wbc assets ({', '.join(sorted(by))})")
    print(f"uploaded {len(ops)} files to https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
