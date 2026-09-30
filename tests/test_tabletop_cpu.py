"""CPU tests of the tabletop track (no Isaac): layout / success rule, scripted push policy (kinematic rollout),
GR00T dataset builder + static validator on synthetic raw episodes.

    python -m pytest tests/test_tabletop_cpu.py -q
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))
sys.path.insert(0, str(REPO / "tools"))

from dropbear_wbc.tasks.tabletop.layout import LAYOUT, Placement, in_zone  # noqa: E402

FFMPEG = Path("C:/isaac-sim/kit/python/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe")


def test_in_zone_margin_and_yaw():
    z = np.array([0.3, 0.3])
    h = 0.5 * LAYOUT.zone_size - LAYOUT.success_margin
    assert in_zone(z + [h - 1e-4, 0.0], z, 0.0)
    assert not in_zone(z + [h + 1e-4, 0.0], z, 0.0)
    # rotated zone: a point on the diagonal at |h*sqrt(2)| is inside a 45-deg zone along x
    assert in_zone(z + [h * math.sqrt(2) - 1e-4, 0.0], z, math.pi / 4)
    assert not in_zone(z + [h * math.sqrt(2) - 1e-4, 0.0], z, 0.0)


def test_layout_frames():
    p = np.array([0.3, 0.2, 1.25])
    assert np.allclose(LAYOUT.w_to_root(LAYOUT.root_to_w(p)), p)
    assert LAYOUT.on_table((LAYOUT.table_front_x + 0.1, LAYOUT.table_center_y))
    assert not LAYOUT.on_table((LAYOUT.table_front_x - 0.01, LAYOUT.table_center_y))


@pytest.mark.parametrize("pl", [
    Placement("t_left", "heldout", "left", (0.30, 0.36), 0.3, (0.27, 0.25)),
    Placement("t_right", "heldout", "right", (0.25, -0.40), -0.2, (0.30, -0.50)),
])
def test_scripted_policy_kinematic_rollout(pl):
    """Perfect joint tracking + a block pushed by contact along the push direction: the state machine must run
    LIFT -> ... -> IDLE and end with the block inside the zone (no physics; logic and IK only)."""
    from dropbear_wbc.tasks.tabletop.kinematics import TabletopArmIK
    from dropbear_wbc.tasks.tabletop.scripted import ScriptedPushPolicy

    ik = TabletopArmIK()
    pol = ScriptedPushPolicy(ik)
    q = ik.rest_q().copy()
    pol.reset(pl.side, pl.zone_xy, q)
    b = np.array([pl.block_xy[0], pl.block_xy[1], LAYOUT.block_rest_z])
    for _ in range(int(LAYOUT.episode_s * LAYOUT.control_hz)):
        q, info = pol.act(b, pl.block_yaw, q)
        hp = ik.hand(pl.side, q[pol.sl])
        if pol.u is not None and hp.lowest_z < LAYOUT.table_top_z + LAYOUT.block_size:
            u = pol.u
            c_off = pol.contact_offset(pl.side, q[pol.sl], u, pl.block_yaw)
            along = float((b[:2] - hp.tool[:2]) @ u)
            lat = abs(float((b[0] - hp.tool[0]) * u[1] - (b[1] - hp.tool[1]) * u[0]))
            if -0.02 < along < c_off and lat < 0.05:
                b[:2] += u * (c_off - along)
        if pol.phase == "IDLE":
            break
    phases = [e[1] for e in pol.events]
    assert phases[:4] == ["LIFT", "TRANSIT", "DESCEND", "PUSH"], phases
    assert pol.phase == "IDLE"
    assert in_zone(b[:2], pl.zone_xy, pl.zone_yaw)
    # the inactive arm never moved
    other = slice(5, 10) if pl.side == "left" else slice(0, 5)
    assert np.allclose(q[other], ik.rest_q()[other])


# ------------------------------------------------------------------------------------------------ dataset chain
def _write_raw_episode(root: Path, pid: str, n: int, success: bool, cams: dict, zone_err_px: float | None = 0.5) -> None:
    from dropbear_wbc.tasks.tabletop.episode_io import LOWDIM_KEYS

    d = root / "train" / f"{pid}_scripted"
    d.mkdir(parents=True)
    rng = np.random.default_rng(abs(hash(pid)) % 2**32)
    arr = {k.replace(".", "__"): rng.normal(size=(n, w)) * 0.1 for k, (w, _) in LOWDIM_KEYS.items()}
    arr["policy__tool_cmd"][:3] = np.nan  # NaN diagnostics must not reach the parquet
    arr["time__sim_s"] = (np.arange(n) * 0.05)[:, None]
    np.savez_compressed(d / "lowdim.npz", **arr)
    meta = {"schema": "dropbear-tabletop-raw-episode-v1", "pid": pid, "split": "train", "side": "left",
            "success": success, "termination": "success" if success else "time_out", "frames": n, "fps": 20,
            "instruction": LAYOUT.instruction, "policy": "scripted", "placement": {"pid": pid},
            "provenance": {"calibration_sha256": "x" * 64, "layout": LAYOUT.to_dict()},
            "zone_check": {"err_px": zone_err_px}, "blank_frames": {}}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for cam, (w, h) in cams.items():
        subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"testsrc=size={w}x{h}:rate=20", "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(d / f"{cam}.mp4")], check=True)


@pytest.mark.skipif(not FFMPEG.is_file(), reason="kit ffmpeg binary not found")
def test_build_and_static_validate(tmp_path):
    pytest.importorskip("pyarrow")
    cams = {"head": (64, 48), "left_wrist": (32, 24), "right_wrist": (32, 24)}
    raw = tmp_path / "_raw"
    _write_raw_episode(raw, "train_000001", 23, True, cams)
    _write_raw_episode(raw, "train_000002", 17, False, cams)
    _write_raw_episode(raw, "train_000003", 31, True, cams)
    _write_raw_episode(raw, "train_000004", 19, True, cams, zone_err_px=180.0)  # zone rendered in the wrong place
    out = tmp_path / "ds"
    r = subprocess.run([sys.executable, str(REPO / "tools" / "build_groot_dataset.py"), "--raw", str(raw),
                        "--split", "train", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2 and info["total_frames"] == 23 + 31  # failed episode left out by default
    eps = [json.loads(x) for x in (out / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [e["length"] for e in eps] == [23, 31] and all(e["dropbear"]["success"] for e in eps)
    mod = json.loads((out / "meta" / "modality.json").read_text())
    assert set(mod["video"]) == {"ego_view", "left_wrist_view", "right_wrist_view"}
    assert info["features"]["observation.images.ego_view"]["shape"] == [48, 64, 3]
    import validate_groot_dataset as vg

    rep = vg.static_checks(out)
    assert rep["ok"], rep["errors"]
    # --include_failed keeps the failed attempt and flags it
    out2 = tmp_path / "ds_all"
    r = subprocess.run([sys.executable, str(REPO / "tools" / "build_groot_dataset.py"), "--raw", str(raw),
                        "--split", "train", "--out", str(out2), "--include_failed"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    eps2 = [json.loads(x) for x in (out2 / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [e["dropbear"]["success"] for e in eps2] == [True, False, True]  # train_000004 dropped (zone check)
    assert vg.static_checks(out2)["ok"]


def test_head_projection_and_green_centroid():
    """The zone-render check: a green square drawn where project_head puts a root-frame point is found there."""
    from dropbear_wbc.tasks.tabletop.cameras import HEAD_CAM_RES, green_centroid, project_head

    for p in [(0.27, 0.24, 1.2515), (0.30, -0.45, 1.2515)]:
        uv = project_head(p)
        assert uv is not None and 0 <= uv[0] < HEAD_CAM_RES[0] and 0 <= uv[1] < HEAD_CAM_RES[1]
        img = np.full((HEAD_CAM_RES[1], HEAD_CAM_RES[0], 3), 200, np.uint8)
        u, v = int(round(uv[0])), int(round(uv[1]))
        img[v - 8:v + 9, u - 8:u + 9] = (150, 230, 190)  # rendered zone colour
        got, n = green_centroid(img)
        assert n == 17 * 17 and np.linalg.norm(got - uv) < 1.0
    # left-arm workspace projects to the left half of the image, right-arm workspace to the right half
    assert project_head((0.3, 0.3, 1.25))[0] < HEAD_CAM_RES[0] / 2 < project_head((0.3, -0.45, 1.25))[0]
    assert green_centroid(np.full((48, 64, 3), 200, np.uint8))[0] is None
