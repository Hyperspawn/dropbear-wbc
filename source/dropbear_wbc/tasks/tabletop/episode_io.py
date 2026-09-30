"""Raw episode writer for the tabletop collector (runs inside kit python: numpy + PIL + imageio_ffmpeg only).

Layout written per attempt (``<raw_root>/<split>/<pid>_a<attempt>/``)::

    meta.json         placement, outcome, policy, timing, calibration / USD provenance, camera list
    lowdim.npz        per-frame arrays (see ``LOWDIM_KEYS``), float32/float64
    <camera>.mp4      H.264 (libx264, yuv420p, CRF 18), exactly one video frame per recorded low-dim frame

Frames are JPEG-compressed (quality 95, 4:4:4) in memory while the episode runs (about 50-100 KB per 640 x 480 frame)
and encoded to mp4 by ONE background thread running one ffmpeg process at a time (the team limit is <= 4 worker
processes, and RAM is shared with training runs). ``tools/build_groot_dataset.py`` turns these raw episodes into the
GR00T LeRobot v2 dataset.
"""
from __future__ import annotations

import io
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

LOWDIM_KEYS = {
    # key: (width, description)
    "observation.state": (10, "measured semantic arm angles (G1 names; left 5 then right 5) [rad]"),
    "action": (10, "commanded semantic arm angles [rad]"),
    "observation.motor_q": (22, "measured motor angles, motor-contract order [rad]"),
    "observation.motor_dq": (22, "measured motor velocities [rad/s]"),
    "action.motor_q": (10, "arm motor position targets sent to the env (motor slots 12..21) [rad]"),
    "action.tau_ff": (10, "arm gravity feed-forward effort targets [N*m]"),
    "observation.block_pose": (7, "block pose, robot root frame: xyz + quat wxyz (privileged)"),
    "observation.zone_pos": (3, "target-zone centre, robot root frame (privileged)"),
    "observation.left_hand_pose": (7, "left hand body pose, root frame: xyz + wxyz (privileged)"),
    "observation.right_hand_pose": (7, "right hand body pose, root frame: xyz + wxyz (privileged)"),
    "policy.tool_cmd": (3, "scripted policy: commanded tool point, root frame (NaN for teleop)"),
    "policy.phase": (1, "scripted policy phase id (scripted.PHASES; -1 for teleop)"),
    "policy.ik_err": (1, "IK position residual of the commanded target [m]"),
    "task.success_now": (1, "instantaneous success condition (before the 0.5 s hold)"),
    "time.sim_s": (1, "simulated time since the episode's first recorded frame [s]"),
}


class JpegFrames:
    """In-memory JPEG frame store for one camera of one episode."""

    def __init__(self, quality: int = 95):
        self.quality = int(quality)
        self.frames: list[bytes] = []
        self.shape: tuple | None = None

    def add(self, rgb: np.ndarray) -> None:
        from PIL import Image

        a = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        self.shape = a.shape
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="JPEG", quality=self.quality, subsampling=0)
        self.frames.append(buf.getvalue())

    def __len__(self) -> int:
        return len(self.frames)

    def nbytes(self) -> int:
        return sum(len(f) for f in self.frames)


def encode_mp4(frames: JpegFrames, path: Path, fps: int, crf: int = 18) -> dict:
    """Decode the JPEGs and pipe them to ffmpeg (libx264, yuv420p). Returns a small report."""
    import imageio_ffmpeg
    from PIL import Image

    t0 = time.perf_counter()
    h, w = frames.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path), (w, h), fps=fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=None,
        macro_block_size=16, ffmpeg_log_level="error",
        output_params=["-crf", str(crf), "-preset", "veryfast", "-g", str(fps), "-threads", "2"])
    writer.send(None)
    for jb in frames.frames:
        writer.send(np.asarray(Image.open(io.BytesIO(jb)).convert("RGB")))
    writer.close()
    return {"path": str(path), "frames": len(frames), "width": w, "height": h, "fps": fps,
            "bytes": path.stat().st_size, "encode_s": round(time.perf_counter() - t0, 2)}


class VideoWriterThread:
    """Single background thread encoding finished episodes' cameras one after another."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.reports: list[dict] = []
        self.errors: list[str] = []
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            frames, path, fps, done_cb = item
            try:
                rep = encode_mp4(frames, path, fps)
                self.reports.append(rep)
                if done_cb:
                    done_cb(rep)
            except Exception as e:  # noqa: BLE001 - reported in the summary, never silently
                self.errors.append(f"{path}: {e!r}")
            finally:
                self.q.task_done()

    def submit(self, frames: JpegFrames, path: Path, fps: int, done_cb=None) -> None:
        self.q.put((frames, path, fps, done_cb))

    def pending(self) -> int:
        return self.q.qsize()

    def close(self, timeout: float = 600.0) -> None:
        self.q.join()
        self.q.put(None)
        self._t.join(timeout)


class EpisodeBuffer:
    """Per-env episode data (low-dim rows + JPEG frames per camera)."""

    def __init__(self, cams: list[str], jpeg_quality: int = 95):
        self.rows: dict[str, list] = {k: [] for k in LOWDIM_KEYS}
        self.cams = {c: JpegFrames(jpeg_quality) for c in cams}

    def add(self, row: dict, images: dict | None = None) -> None:
        for k, (w, _) in LOWDIM_KEYS.items():
            v = np.asarray(row[k], dtype=np.float64).reshape(-1)
            if v.shape[0] != w:
                raise ValueError(f"{k}: width {v.shape[0]} != {w}")
            self.rows[k].append(v)
        for c, fr in self.cams.items():
            fr.add(images[c])

    def __len__(self) -> int:
        return len(self.rows["action"])

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: (np.stack(v) if v else np.zeros((0, LOWDIM_KEYS[k][0]))) for k, v in self.rows.items()}


def write_episode(ep_dir: Path, buf: EpisodeBuffer, meta: dict, fps: int, video: VideoWriterThread | None) -> dict:
    ep_dir.mkdir(parents=True, exist_ok=True)
    arr = buf.arrays()
    np.savez_compressed(ep_dir / "lowdim.npz", **{k.replace(".", "__"): v for k, v in arr.items()})
    meta = dict(meta)
    meta["frames"] = len(buf)
    meta["lowdim_keys"] = {k: {"width": w, "description": d} for k, (w, d) in LOWDIM_KEYS.items()}
    meta["videos"] = {c: f"{c}.mp4" for c in buf.cams}
    meta["camera_shapes"] = {c: list(fr.shape) for c, fr in buf.cams.items() if fr.shape is not None}  # (H, W, 3)
    meta["jpeg_bytes"] = {c: fr.nbytes() for c, fr in buf.cams.items()}
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8")
    if video is not None:
        for c, fr in buf.cams.items():
            if len(fr):
                video.submit(fr, ep_dir / f"{c}.mp4", fps)
    return meta
