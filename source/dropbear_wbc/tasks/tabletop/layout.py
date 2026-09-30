"""Tabletop layout, placement sampling (train / held-out split) and the success rule (pure Python + numpy).

Frames: everything here is in the ROBOT ROOT frame (the fixed ``world`` body; x forward, y left, z up) unless a name
ends in ``_w`` (Isaac env frame: env origin, ground at z = 0). The root is fixed at :attr:`TabletopLayout.root_pos_w`
with identity rotation, so ``p_w = p_root + root_pos_w``.

Geometry choices (evidence in ``logs/tabletop/``):

* root height: the calibration's standing root height is -0.159 m (``hip_height.root_z_at_standing``: lowest sole
  point at z = 0); the fixed root sits 4 cm higher (-0.12 m) so the hanging legs never touch the ground;
* table top at root z 1.25 m (world 1.13 m): ``tools/tabletop_workspace.py`` (``workspace.{log,json,png}``) measured
  the pushable area (tool IK residual < 2 mm, hand lowest point 8 mm above the table, elbow >= 6 cm above it, not at a
  shoulder / elbow limit, hover reachable) for table heights 1.20 / 1.25 / 1.30 m;
* table front edge 2.4 cm in front of the torso collider (x 0.141 m at those heights, ``geometry_probe.log``);
* each hand can only work in front of / outside its own shoulder: the inner boundary is the shoulder-roll limit
  (-10 deg adduction). A placement therefore belongs to ONE arm (``side``), chosen at random.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
PLACEMENTS_DIR = REPO / "data" / "groot" / "placements"


@dataclass(frozen=True)
class TabletopLayout:
    """All task geometry and the success rule (numbers in metres / kg / seconds)."""

    name: str = "dropbear_tabletop_push_v1"
    root_pos_w: tuple = (0.0, 0.0, -0.12)
    # table: a slab; top at table_top_z (root frame); x from table_front_x to table_front_x + table_depth
    table_top_z: float = 1.25
    table_front_x: float = 0.165
    table_depth: float = 0.45
    table_center_y: float = -0.0697     # torso midline (pelvis_in_root y)
    table_width: float = 1.30
    table_thickness: float = 0.04
    table_static_friction: float = 0.5
    table_dynamic_friction: float = 0.4
    # block
    block_size: float = 0.05
    block_mass: float = 0.10
    block_static_friction: float = 0.5
    block_dynamic_friction: float = 0.4
    block_color: tuple = (0.85, 0.08, 0.06)
    # target zone ("tray"): a flat, visual-only square on the table (no collider, so pushing never has to lift the block)
    zone_size: float = 0.10
    zone_thickness: float = 0.002
    zone_color: tuple = (0.10, 0.70, 0.20)
    # success: block centre inside the zone (margin), resting on the table, upright, at rest for hold_s
    success_margin: float = 0.01
    success_z_tol: float = 0.01
    success_max_tilt_deg: float = 10.0
    success_max_lin_vel: float = 0.03
    success_max_ang_vel: float = 0.5
    success_hold_s: float = 0.5
    # failure: block below the table top (fell off) or left the table area
    drop_margin: float = 0.05
    # timing
    control_hz: int = 20
    episode_s: float = 20.0
    instruction: str = "push the red block onto the green target"

    # ---------------------------------------------------------------------------------------------- derived
    @property
    def block_rest_z(self) -> float:
        return self.table_top_z + 0.5 * self.block_size

    @property
    def table_center(self) -> np.ndarray:
        return np.array([self.table_front_x + 0.5 * self.table_depth, self.table_center_y,
                         self.table_top_z - 0.5 * self.table_thickness])

    def root_to_w(self, p_root) -> np.ndarray:
        return np.asarray(p_root, float) + np.asarray(self.root_pos_w, float)

    def w_to_root(self, p_w) -> np.ndarray:
        return np.asarray(p_w, float) - np.asarray(self.root_pos_w, float)

    def on_table(self, xy, margin: float = 0.0) -> bool:
        x, y = float(xy[0]), float(xy[1])
        return (self.table_front_x + margin <= x <= self.table_front_x + self.table_depth - margin
                and abs(y - self.table_center_y) <= 0.5 * self.table_width - margin)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["block_rest_z"] = self.block_rest_z
        return d


LAYOUT = TabletopLayout()


def in_zone(block_xy_root, zone_xy_root, zone_yaw: float, layout: TabletopLayout = LAYOUT):
    """Whether the block centre is inside the zone shrunk by ``success_margin`` (numpy, broadcasts over rows)."""
    d = np.asarray(block_xy_root, float) - np.asarray(zone_xy_root, float)
    c, s = np.cos(zone_yaw), np.sin(zone_yaw)
    lx = c * d[..., 0] + s * d[..., 1]
    ly = -s * d[..., 0] + c * d[..., 1]
    h = 0.5 * layout.zone_size - layout.success_margin
    return (np.abs(lx) <= h) & (np.abs(ly) <= h)


# -------------------------------------------------------------------------------------------------- placements
@dataclass
class Placement:
    """One episode's initial condition (root frame)."""

    pid: str                    # e.g. "train_000123" / "heldout_000007"
    split: str
    side: str                   # arm that can do the push
    block_xy: tuple
    block_yaw: float
    zone_xy: tuple
    zone_yaw: float = 0.0
    seed: int = 0
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def block_half_extent_along(u: np.ndarray, yaw: float, size: float) -> float:
    """Half extent of a square block (edge ``size``, yaw) along unit direction ``u`` (xy)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return 0.5 * size * (abs(u[0] * c + u[1] * s) + abs(-u[0] * s + u[1] * c))


# Sampling regions for the zone centre per arm (root frame): the measured pushable bounding boxes shrunk so a zone and a
# push path fit; the IK checks below decide. Right = mirror of left about the torso midline.
ZONE_REGION_LEFT = {"x": (0.20, 0.34), "y": (0.17, 0.40)}
BLOCK_DIST = (0.085, 0.13)       # block-centre -> zone-centre distance (block starts fully outside the zone)
SPLIT_SEED_BASE = {"train": 1_000_000, "heldout": 9_000_000}


def zone_region(side: str, layout: TabletopLayout = LAYOUT) -> dict:
    if side == "left":
        return ZONE_REGION_LEFT
    ym = layout.table_center_y
    lo, hi = ZONE_REGION_LEFT["y"]
    return {"x": ZONE_REGION_LEFT["x"], "y": (2 * ym - hi, 2 * ym - lo)}


def candidate(rng: np.random.Generator, layout: TabletopLayout = LAYOUT) -> dict:
    side = "left" if rng.random() < 0.5 else "right"
    reg = zone_region(side, layout)
    zone = np.array([rng.uniform(*reg["x"]), rng.uniform(*reg["y"])])
    d = rng.uniform(*BLOCK_DIST)
    th = rng.uniform(-math.pi, math.pi)
    block = zone + d * np.array([math.cos(th), math.sin(th)])
    yaw = rng.uniform(-math.pi / 4, math.pi / 4)
    return {"side": side, "zone_xy": zone, "block_xy": block, "block_yaw": yaw}


def save_placements(path: Path, placements: list[Placement], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "dropbear-tabletop-placements-v1", "meta": meta,
                                "placements": [p.to_dict() for p in placements]}, indent=1) + "\n", encoding="utf-8")


def load_placements(path: Path, split: str | None = None) -> tuple[list[Placement], dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("schema") != "dropbear-tabletop-placements-v1":
        raise ValueError(f"{path}: unexpected schema {d.get('schema')!r}")
    out = [Placement(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in p.items()}) for p in d["placements"]]
    if split is not None:
        out = [p for p in out if p.split == split]
    return out, d["meta"]
