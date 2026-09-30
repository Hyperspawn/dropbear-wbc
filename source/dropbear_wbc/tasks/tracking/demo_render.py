"""Clean demo rendering for the Dropbear tracking task (demo_eval track, 2026-09-24).

Two pieces, both used by ``scripts/play.py --clean_video`` and ``scripts/render_reference.py``:

* :func:`apply_clean_render` edits an env cfg BEFORE ``gym.make``: every debug-visualisation marker off (motion command
  frame markers, contact-sensor debug_vis), neutral white lighting (angled key light + white dome), a mid-grey grid
  floor (the grid stays: it is the scale/slip reference), and ``env_spacing`` wide enough that neighbouring envs stay
  out of frame. Physics is untouched: the same config renders and evaluates.
* :class:`DemoRecorder` owns its own USD cameras + replicator render products (NOT the gym ``RecordVideo`` viewport),
  so several views are captured from ONE rollout:

  - ``track``: follows env 0's root (``world`` body) xy with a low-pass filter (tau 0.6 s) at a fixed height, from a
    world-fixed front-right 3/4 azimuth (-35 deg), 4.4 m away, eye 1.55 m, look-at 1.0 m (12 mm lens; see the framing
    note in :class:`DemoRecorder`);
  - ``front``: fixed, straight in front of the clip's mean root xy (robot heading +x at frame 0), far enough to keep
    the clip's whole xy footprint in frame;
  - ``side``: fixed, from the robot's right (-y) side;
  - ``feet``: tracking, low (eye 0.30 m) from the front-right, framing the lower legs and the ground line (to show
    floating or sliding feet).

  Frames (RGB, 50 fps = real time at the 50 Hz policy rate) are piped to ffmpeg (libx264, yuv420p, crf 18); an
  optional caption and a running clock are burned in with ``drawtext``.

Rendering needs ``--enable_cameras`` (headless RTX). Nothing here changes observations, actions or rewards.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FFMPEG_DEFAULT = "C:/ProgramData/chocoportable/bin/ffmpeg.exe"
FONT_DIR = "C:/Windows/Fonts"
FONT_FILE = "arial.ttf"

__all__ = ["apply_clean_render", "DemoRecorder", "CameraSpec", "frame_is_valid", "ffmpeg_exe", "contact_sheet", "parse_resolution"]


def ffmpeg_exe() -> str:
    """ffmpeg binary: ``$DROPBEAR_FFMPEG``, the chocolatey-portable build, or ``ffmpeg`` on PATH."""
    for cand in (os.environ.get("DROPBEAR_FFMPEG"), FFMPEG_DEFAULT, shutil.which("ffmpeg")):
        if cand and Path(cand).is_file():
            return str(cand)
    raise FileNotFoundError("ffmpeg not found (set $DROPBEAR_FFMPEG)")


def parse_resolution(text: str) -> tuple[int, int]:
    w, h = (int(v) for v in str(text).lower().split("x"))
    if w % 2 or h % 2:
        raise ValueError(f"resolution {text!r}: width and height must be even (yuv420p)")
    return w, h


def _quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """wxyz quaternion rotating unit vector ``a`` onto ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(a, b))
    if s < 1e-9:
        return (1.0, 0.0, 0.0, 0.0) if c > 0 else (0.0, 1.0, 0.0, 0.0)
    axis /= s
    ang = math.atan2(s, c)
    return (math.cos(ang / 2), *(float(v) for v in axis * math.sin(ang / 2)))


def apply_clean_render(env_cfg, *, floor_rgb=(0.42, 0.43, 0.45), key_intensity: float = 2600.0,
                       dome_intensity: float = 900.0, env_spacing: float = 12.0) -> dict:
    """Configure ``env_cfg`` (a ``TrackingEnvCfg``) for clean demo rendering; returns a record of what changed."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    changed = {}
    motion = env_cfg.commands.motion
    changed["motion_command_debug_vis"] = [bool(motion.debug_vis), False]
    motion.debug_vis = False
    cs = env_cfg.scene.contact_forces
    changed["contact_sensor_debug_vis"] = [bool(cs.debug_vis), False]
    cs.debug_vis = False
    # any other sensor / command term with a debug_vis flag
    for holder in (env_cfg.scene, env_cfg.commands):
        for name in dir(holder):
            if name.startswith("_"):
                continue
            obj = getattr(holder, name, None)
            if obj is not None and hasattr(obj, "debug_vis") and getattr(obj, "debug_vis") is True:
                obj.debug_vis = False
                changed[f"{name}.debug_vis"] = [True, False]
    # floor: grid tint (the default is near-black); lights: white dome + an angled white key light (from the
    # front-left, above), replacing the grey 0.75 distant light and the dim 0.13 dome
    env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(floor_rgb))
    key_dir = np.array([-0.55, -0.35, -0.76])  # direction the light travels
    env_cfg.scene.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=float(key_intensity), angle=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=_quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), key_dir)),
    )
    env_cfg.scene.sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(color=(0.92, 0.93, 0.95), intensity=float(dome_intensity))
    )
    changed.update(floor_rgb=list(floor_rgb), key_light=[float(key_intensity), key_dir.tolist()],
                   dome_light=float(dome_intensity))
    if env_cfg.scene.num_envs > 1:
        changed["env_spacing"] = [float(env_cfg.scene.env_spacing), float(env_spacing)]
        env_cfg.scene.env_spacing = float(env_spacing)
    return changed


@dataclass
class CameraSpec:
    name: str
    focal_mm: float = 18.0  # horizontal aperture 20.955 mm -> 60 deg horizontal / 36 deg vertical FOV at 16:9


@dataclass
class _Cam:
    spec: CameraSpec
    path: str
    op: object
    annot: object
    proc: subprocess.Popen | None = None
    out: Path | None = None
    frames: int = 0
    empty: int = 0
    retried: int = 0  # captures that needed extra renders before the frame was valid
    invalid: int = 0  # frames still invalid after all retries (written anyway)


def frame_is_valid(arr: np.ndarray) -> bool:
    """Render-glitch check (demo_eval, 2026-09-24). The RTX annotator sometimes returns a full-size but BLANK image:
    all black, or a washed-out uniform grey without floor or robot (seen in 25-61 % of the frames of two play renders;
    ``empty_frames`` stayed 0). Real frames of this scene contain the coloured grid floor and the robot, so they have
    colour (mean max-min over RGB >> 4) and luma contrast (p90 - p10 >> 15); blank frames have neither."""
    if arr.ndim != 3 or arr.size == 0:
        return False
    a = arr[::8, ::8, :3].astype(np.int16)
    chroma = float((a.max(axis=2) - a.min(axis=2)).mean())
    luma = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    p10, p90 = np.percentile(luma, [10, 90])
    return chroma > 4.0 and (p90 - p10) > 15.0


@dataclass
class DemoRecorder:
    """Record env 0 from several cameras of one rollout (call :meth:`capture` after every ``env.step``)."""

    env: object  # the unwrapped ManagerBasedRLEnv
    out_dir: Path
    tag: str
    cams: tuple[str, ...] = ("track", "front")
    resolution: tuple[int, int] = (1280, 720)
    fps: float = 50.0
    caption: str = ""
    crf: int = 18
    # Framing (demo_eval, 2026-09-24 16:20): the first clean renders (18 mm, track 3.3 m / front 4.2 m) cropped the head
    # and feet: projecting the reference bodies into the rendered frames gives an EFFECTIVE focal length ~1.3-1.5x the
    # nominal one (cause not identified). 12 mm at 4.4 m (track) / >= 5 m (front, side) keeps a 1.75 m robot plus a
    # 3-line caption in frame for any factor 1.2-1.6; the low feet camera keeps 18 mm.
    track_distance: float = 4.4
    track_azimuth_deg: float = -35.0
    track_eye_height: float = 1.55
    lookat_height: float = 1.0
    focal_mm: float = 12.0
    feet_focal_mm: float = 18.0
    smooth_tau_s: float = 0.6
    reference_xy: np.ndarray | None = None  # (T, 2) env-local root xy of the clip (fixed cameras frame it)
    warmup_renders: int = 4
    max_retry_renders: int = 8  # extra sim.render() calls per capture while any camera frame is invalid
    root_xy_fn: object = None  # optional callable -> env-local root xy (kinematic replays); default: live robot root
    _cams: list = field(default_factory=list)
    _xy: np.ndarray | None = None

    def __post_init__(self):
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._Gf = Gf
        stage = omni.usd.get_context().get_stage()
        self._origin = self.env.scene.env_origins[0].detach().cpu().numpy().astype(float)
        self._robot = self.env.scene["robot"]
        self._root_idx = 0  # articulation root 'world'
        ref = self.reference_xy if self.reference_xy is not None else np.zeros((1, 2))
        self._ref_center = 0.5 * (ref.min(0) + ref.max(0))
        self._ref_extent = float(np.linalg.norm(ref.max(0) - ref.min(0)))
        for name in self.cams:
            spec = CameraSpec(name, focal_mm=self.feet_focal_mm if name == "feet" else self.focal_mm)
            path = f"/World/DemoCams/{name}"
            cam = UsdGeom.Camera.Define(stage, path)
            cam.GetFocalLengthAttr().Set(spec.focal_mm)
            cam.GetHorizontalApertureAttr().Set(20.955)
            cam.GetVerticalApertureAttr().Set(20.955 * self.resolution[1] / self.resolution[0])
            cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e4))
            xf = UsdGeom.Xformable(cam.GetPrim())
            xf.ClearXformOpOrder()
            op = xf.AddTransformOp()
            rp = rep.create.render_product(path, tuple(self.resolution))
            annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            annot.attach([rp])
            self._cams.append(_Cam(spec=spec, path=path, op=op, annot=annot))
        self._place_cameras(initial=True)
        for c in self._cams:
            self._open_writer(c)

    # ------------------------------------------------------------------ camera poses
    def _look_at(self, eye: np.ndarray, target: np.ndarray) -> object:
        Gf = self._Gf
        fwd = target - eye
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        m = Gf.Matrix4d(
            right[0], right[1], right[2], 0.0,
            up[0], up[1], up[2], 0.0,
            -fwd[0], -fwd[1], -fwd[2], 0.0,
            eye[0], eye[1], eye[2], 1.0,
        )
        return m

    def _root_xy_local(self) -> np.ndarray:
        if self.root_xy_fn is not None:
            return np.asarray(self.root_xy_fn(), dtype=float)
        p = self._robot.data.root_link_pos_w[0, :2].detach().cpu().numpy().astype(float)
        return p - self._origin[:2]

    def _place_cameras(self, initial: bool = False) -> None:
        xy = self._root_xy_local()
        if initial or self._xy is None:
            self._xy = xy.copy()
        else:
            dt = 1.0 / self.fps
            a = dt / (self.smooth_tau_s + dt)
            self._xy = (1 - a) * self._xy + a * xy
        o = self._origin
        for c in self._cams:
            if c.spec.name in ("track", "feet"):
                if c.spec.name == "track":
                    az, dist, eye_h, look_h = (math.radians(self.track_azimuth_deg), self.track_distance,
                                               self.track_eye_height, self.lookat_height)
                else:  # low camera on the feet / ground line (shows floating or sliding feet)
                    az, dist, eye_h, look_h = math.radians(-60.0), 2.4, 0.30, 0.32
                tgt = np.array([self._xy[0], self._xy[1], look_h])
                eye = tgt + np.array([dist * math.cos(az), dist * math.sin(az), eye_h - look_h])
            elif c.spec.name in ("front", "side"):
                if not initial:
                    continue
                d = max(5.0, self._ref_extent + 4.0)
                ctr = self._ref_center
                tgt = np.array([ctr[0], ctr[1], self.lookat_height])
                offs = np.array([d, 0.0, 0.4]) if c.spec.name == "front" else np.array([0.0, -d, 0.4])
                eye = tgt + offs
            else:
                raise ValueError(f"unknown camera {c.spec.name!r} (track, front, side, feet)")
            c.op.Set(self._look_at(eye + o, tgt + o))

    # ------------------------------------------------------------------ encoding
    def _drawtext(self) -> str:
        def esc(s: str) -> str:  # caption is literal text inside '...': drop characters the filter parser treats specially
            # commas are safe inside the single-quoted text (verified with ffmpeg 2026-09-24, demo_eval), so they stay
            return s.replace("\\", "").replace("'", "").replace(":", " -").replace("%", " pct")

        fs = max(16, self.resolution[1] // 30)
        box = f"fontsize={fs}:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=8"
        parts = []
        if self.caption:
            for i, line in enumerate(self.caption.split("\n")[:3]):
                parts.append(f"drawtext=fontfile={FONT_FILE}:text='{esc(line)}':x=20:y={18 + i * int(fs * 1.9)}:{box}")
        # running clock (video time = sim time: one frame per 50 Hz policy step), e.g. "t = 3.4 s"
        parts.append(f"drawtext=fontfile={FONT_FILE}:text='t = %{{eif\\:trunc(t)\\:d}}.%{{eif\\:trunc(mod(t*10\\,10))\\:d}} s'"
                     f":x=w-tw-20:y=h-th-18:{box}")
        return ",".join(parts)

    def _open_writer(self, c: _Cam) -> None:
        w, h = self.resolution
        c.out = (self.out_dir / f"{self.tag}_{c.spec.name}.mp4").resolve()
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", f"{self.fps:g}", "-i", "-", "-vf", self._drawtext(), "-c:v", "libx264",
               "-preset", "medium", "-crf", str(self.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(c.out)]
        # cwd = the fonts dir so drawtext's fontfile needs no drive-letter colon escaping
        c.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, cwd=FONT_DIR if Path(FONT_DIR).is_dir() else None)

    # ------------------------------------------------------------------ per step
    def capture(self, force_render: bool = False) -> None:
        """Update the tracking camera, render and write one frame per camera.

        ``env.step`` renders by itself only when the scene has RTX sensors (``sim.has_rtx_sensors()``); otherwise (or
        with ``force_render``, e.g. a kinematic replay that never steps) this renders once.
        """
        self._place_cameras()
        sim = self.env.sim
        if force_render or not sim.has_rtx_sensors():
            sim.render()
        if self.warmup_renders > 0:  # the first RTX frames of a new render product are often empty/black
            for _ in range(self.warmup_renders):
                sim.render()
            self.warmup_renders = 0
        w, h = self.resolution
        frames = {c.spec.name: np.asarray(c.annot.get_data()) for c in self._cams}
        # re-render (no physics step: sim.render() only) until every camera returns a valid frame, or give up
        bad = [c for c in self._cams if not frame_is_valid(frames[c.spec.name])]
        for _ in range(self.max_retry_renders if bad else 0):
            sim.render()
            for c in bad:
                frames[c.spec.name] = np.asarray(c.annot.get_data())
            still = [c for c in bad if not frame_is_valid(frames[c.spec.name])]
            for c in bad:
                if c not in still:
                    c.retried += 1
            bad = still
            if not bad:
                break
        for c in bad:
            c.invalid += 1
        for c in self._cams:
            arr = frames[c.spec.name]
            if arr.size == 0 or arr.ndim != 3:
                c.empty += 1
                arr = np.zeros((h, w, 3), dtype=np.uint8)
            arr = np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
            if arr.shape[0] != h or arr.shape[1] != w:
                raise RuntimeError(f"camera {c.spec.name}: frame {arr.shape} != {h}x{w}")
            c.proc.stdin.write(arr.tobytes())
            c.frames += 1

    def close(self) -> dict:
        out = {}
        for c in self._cams:
            if c.proc is not None:
                c.proc.stdin.close()
                rc = c.proc.wait(timeout=600)
                out[c.spec.name] = {"mp4": str(c.out), "frames": c.frames, "empty_frames": c.empty,
                                    "retried_frames": c.retried, "invalid_frames": c.invalid, "ffmpeg_rc": rc}
                c.proc = None
        return out

    def describe(self) -> dict:
        return {
            "cameras": list(self.cams), "resolution": list(self.resolution), "fps": self.fps,
            "focal_mm": {c.spec.name: c.spec.focal_mm for c in self._cams},
            "track": {"distance_m": self.track_distance, "azimuth_deg": self.track_azimuth_deg,
                      "eye_height_m": self.track_eye_height, "lookat_height_m": self.lookat_height,
                      "smooth_tau_s": self.smooth_tau_s, "follows": "env-0 root 'world' xy (low-pass)"},
            "fixed_cameras": {"center_xy_env": self._ref_center.tolist(), "extent_m": self._ref_extent},
            "caption": self.caption, "encoder": "ffmpeg libx264 crf %d yuv420p" % self.crf,
        }


def contact_sheet(mp4: Path, out_png: Path, fps: float = 2.0, cols: int = 6, width: int = 1920) -> Path:
    """ffmpeg ``fps=<fps>,scale,tile=<cols>x<rows>`` contact sheet of a whole video (rows from its duration)."""
    mp4, out_png = Path(mp4), Path(out_png)
    probe = subprocess.run([str(Path(ffmpeg_exe()).with_name("ffprobe.exe")) if Path(ffmpeg_exe()).with_name(
        "ffprobe.exe").is_file() else "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
        "default=nw=1:nk=1", str(mp4)], capture_output=True, text=True, check=True)
    dur = float(probe.stdout.strip())
    n = max(1, int(math.floor(dur * fps + 1e-6)))
    rows = int(math.ceil(n / cols))
    tile_w = width // cols
    vf = f"fps={fps:g},scale={tile_w}:-2,tile={cols}x{rows}:padding=4:margin=4:color=white"
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4), "-vf", vf,
                    "-frames:v", "1", "-update", "1", str(out_png)], check=True)
    return out_png
