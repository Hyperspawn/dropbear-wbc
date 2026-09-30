"""Kinematic dry run of the scripted push policy on a placements file (CPU; no physics): perfect joint tracking and a
"block" pushed along the push direction when the hand reaches it. Checks the state machine + IK, not contact physics.

    python tools/tabletop_dryrun.py data/groot/placements/tabletop_push_v1.json heldout 50 > logs/tabletop/dryrun_x.log
"""
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc.tasks.tabletop.kinematics import TabletopArmIK  # noqa: E402
from dropbear_wbc.tasks.tabletop.layout import LAYOUT, in_zone, load_placements  # noqa: E402
from dropbear_wbc.tasks.tabletop.scripted import ScriptedPushPolicy  # noqa: E402

path = sys.argv[1]
split = sys.argv[2] if len(sys.argv) > 2 else "heldout"
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 5
pl, _ = load_placements(path, split)
ik = TabletopArmIK()
pol = ScriptedPushPolicy(ik)
out = []
for p in pl[:nmax]:
    t0 = time.perf_counter()
    q = ik.rest_q().copy()
    pol.reset(p.side, p.zone_xy, q)
    b = np.array([p.block_xy[0], p.block_xy[1], LAYOUT.block_rest_z])
    yaw = p.block_yaw
    sl = pol.sl
    ok = False
    for k in range(int(LAYOUT.episode_s * LAYOUT.control_hz)):
        q_cmd, info = pol.act(b, yaw, q)
        q = q_cmd  # perfect tracking
        hp = ik.hand(p.side, q[sl])
        # crude contact: if the hand's lowest point is below the block top, push the block along the hand's
        # motion direction so the tool stays c_off behind the block centre along u
        if pol.u is not None and hp.lowest_z < LAYOUT.table_top_z + LAYOUT.block_size:
            u = pol.u
            c_off = pol.contact_offset(p.side, q[sl], u, yaw)
            along = float((b[:2] - hp.tool[:2]) @ u)
            lat = abs(float((b[0] - hp.tool[0]) * u[1] - (b[1] - hp.tool[1]) * u[0]))
            if along < c_off and along > -0.02 and lat < 0.05:
                b[:2] += u * (c_off - along)
        if pol.phase == "IDLE":
            break
    ok = bool(in_zone(b[:2], p.zone_xy, p.zone_yaw))
    out.append({"pid": p.pid, "side": p.side, "ok": ok, "final_d": float(np.linalg.norm(b[:2] - np.asarray(p.zone_xy))),
                "phase": pol.phase, "steps": k + 1, "events": pol.events, "ms": round(1e3 * (time.perf_counter() - t0))})
    print(json.dumps(out[-1]))
print("kinematic success", sum(o["ok"] for o in out), "/", len(out))
