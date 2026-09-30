"""Frame grids and episode clips from tabletop raw episodes (CPU; system Python + PIL + the kit ffmpeg binary).

    python tools/tabletop_media.py grid <raw_episode_dir> [<raw_episode_dir> ...] --out logs/tabletop/media/x.png \
        [--cams head left_wrist right_wrist scene] [--times 0 3 6 9]
    python tools/tabletop_media.py clip <raw_episode_dir> --cam head --out logs/tabletop/media/x.mp4

``grid``: rows = (episode, camera), columns = sample times [s] (default: 6 evenly spaced frames); each tile is labelled
with the episode pid, camera, time, scripted phase and block-to-zone distance from ``lowdim.npz``.
``clip``: copies one camera's mp4 (already H.264) next to the grid, optionally side by side with a second camera.
"""
from __future__ import annotations

import sys

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
FFMPEG = Path(_paths.ffmpeg())
PHASES = ("LIFT", "TRANSIT", "DESCEND", "PUSH", "RETREAT", "RISE", "BACK", "HOME", "IDLE")


def ffmpeg() -> str:
    return str(FFMPEG) if FFMPEG.is_file() else (shutil.which("ffmpeg") or "ffmpeg")


def frame_at(mp4: Path, idx: int, tmp: Path):
    from PIL import Image

    out = tmp / f"{mp4.parent.name}_{mp4.stem}_{idx}.png"
    subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4), "-vf",
                    f"select=eq(n\\,{idx})", "-vframes", "1", str(out)], check=True)
    return Image.open(out).convert("RGB")


def grid(eps: list[Path], cams: list[str], times: list[float] | None, out: Path, tile_w: int = 320) -> dict:
    from PIL import Image, ImageDraw

    rows, info = [], []
    with tempfile.TemporaryDirectory(dir=str(out.parent)) as td:
        tmp = Path(td)
        for ep in eps:
            meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
            z = np.load(ep / "lowdim.npz")
            n = int(meta["frames"])
            fps = float(meta["fps"])
            idxs = ([min(n - 1, int(round(t * fps))) for t in times] if times
                    else np.linspace(0, n - 1, 6).round().astype(int).tolist())
            bz = z["observation__block_pose"][:, :2] - z["observation__zone_pos"][:, :2]
            dist = np.linalg.norm(bz, axis=1)
            ph = z["policy__phase"][:, 0]
            for cam in cams:
                mp4 = ep / f"{cam}.mp4"
                if not mp4.is_file():
                    continue
                tiles = []
                for i in idxs:
                    im = frame_at(mp4, i, tmp)
                    im = im.resize((tile_w, int(round(im.height * tile_w / im.width))))
                    d = ImageDraw.Draw(im)
                    p = int(ph[i])
                    label = (f"{meta['pid']} {cam} t={i / fps:.1f}s {PHASES[p] if 0 <= p < len(PHASES) else '-'} "
                             f"d={100 * dist[i]:.1f}cm")
                    d.rectangle([0, 0, tile_w, 14], fill=(0, 0, 0))
                    d.text((3, 1), label, fill=(255, 255, 255))
                    tiles.append(im)
                h = max(t.height for t in tiles)
                row = Image.new("RGB", (tile_w * len(tiles), h), (40, 40, 40))
                for k, t in enumerate(tiles):
                    row.paste(t, (k * tile_w, 0))
                rows.append(row)
            info.append({"episode": str(ep), "pid": meta["pid"], "success": meta.get("success"), "frames": n,
                         "frame_indices": idxs})
    w = max(r.width for r in rows)
    img = Image.new("RGB", (w, sum(r.height for r in rows)), (40, 40, 40))
    y = 0
    for r in rows:
        img.paste(r, (0, y))
        y += r.height
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return {"out": str(out), "episodes": info, "cams": cams, "size": list(img.size)}


def clip(ep: Path, cam: str, out: Path, cam2: str | None = None) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    if cam2 is None:
        shutil.copy2(ep / f"{cam}.mp4", out)
    else:  # side by side, both scaled to 480 px height
        subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(ep / f"{cam}.mp4"), "-i",
                        str(ep / f"{cam2}.mp4"), "-filter_complex",
                        "[0:v]scale=-2:480[a];[1:v]scale=-2:480[b];[a][b]hstack=inputs=2", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)
    return {"out": str(out), "bytes": out.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("episodes", nargs="+", type=Path)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--cams", nargs="+", default=["head", "left_wrist", "right_wrist", "scene"])
    g.add_argument("--times", nargs="+", type=float, default=None)
    g.add_argument("--tile_w", type=int, default=320)
    c = sub.add_parser("clip")
    c.add_argument("episode", type=Path)
    c.add_argument("--cam", default="head")
    c.add_argument("--cam2", default=None)
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rep = grid(a.episodes, a.cams, a.times, a.out, a.tile_w) if a.cmd == "grid" else clip(a.episode, a.cam, a.out, a.cam2)
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
