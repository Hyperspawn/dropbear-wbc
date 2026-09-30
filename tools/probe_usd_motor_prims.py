"""List the MyActuator CAD components merged into each rigid body of the plant USD (CPU, pxr, READ-ONLY).

Why: the motor model of each joint is not written anywhere authoritative. The Fusion export keeps the CAD component
names (``RMD_X10_S2_*`` = RMD-X10-S2 1:35, ``RMD_X10_V3*`` = RMD-X10 1:7 V3, ``RMD_X8_Pro_*`` = RMD-X8 Pro) as Xform
prims inside the rigid body they were merged into. A motor joint is driven by the motor whose stator sits in one of
its two bodies and whose rotor sits in the other, so this listing identifies the motor model per joint
(docs/ACTUATORS.md section 1).

Run with an interpreter that has ``pxr`` (usd-core), e.g.::

    .venv-newton/Scripts/python.exe tools/probe_usd_motor_prims.py \
        --out logs/actuators/usd_motor_prims.json > logs/actuators/usd_motor_prims.log

The USD is opened read-only; nothing is written back.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES, USD_SHA256, resolve_usd_path  # noqa: E402

KEYWORDS = ("rmd", "cem", "stator", "rotor", "rotot")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--usd", default=resolve_usd_path())
    ap.add_argument("--tree", default=str(REPO / "docs" / "usd_tree_45586414.json"))
    ap.add_argument("--out", default=str(REPO / "logs" / "actuators" / "usd_motor_prims.json"))
    ap.add_argument("--skip-sha", action="store_true", help="do not hash the 420 MB USD")
    args = ap.parse_args()

    from pxr import Usd  # noqa: WPS433 (optional dependency, CPU only)

    usd_sha = None if args.skip_sha else sha256(args.usd)
    print(f"usd: {args.usd}")
    print(f"usd sha256: {usd_sha} (contract {USD_SHA256}; match={usd_sha == USD_SHA256 if usd_sha else 'skipped'})")
    stage = Usd.Stage.Open(args.usd, Usd.Stage.LoadAll)
    tree = json.loads(Path(args.tree).read_text(encoding="utf-8"))
    bodies = [b[0] for b in tree["bodies"]]

    per_body: dict[str, list[str]] = {}
    for body in bodies:
        prim = stage.GetPrimAtPath(f"/humanoid/{body}")
        if not prim:
            per_body[body] = ["<missing>"]
            continue
        names = sorted({d.GetName() for d in Usd.PrimRange(prim) if d != prim
                        and any(k in d.GetName().lower() for k in KEYWORDS) and "rmd" in d.GetName().lower()})
        per_body[body] = names

    joints = {j["name"]: j for j in tree["joints"]}
    per_motor = {}
    print("\n# motor joint: body0 [RMD CAD prims inside] -> body1 [RMD CAD prims inside]")
    for m in MOTOR_NAMES:
        j = joints[m]
        b0, b1 = j["b0"][0], j["b1"][0]
        per_motor[m] = {"body0": b0, "body0_rmd_prims": per_body.get(b0, []),
                        "body1": b1, "body1_rmd_prims": per_body.get(b1, [])}
        own1 = [b1] if "RMD" in b1 else []
        print(f"{m:24s} {b0} {per_body.get(b0, [])} -> {b1} (body name itself: {own1}) {per_body.get(b1, [])}")

    print("\n# every body that contains RMD CAD components")
    for body, names in per_body.items():
        if names:
            print(f"{body:45s} {names}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "dropbear-usd-motor-prims-v1", "usd_path": args.usd, "usd_sha256": usd_sha,
                               "script": "tools/probe_usd_motor_prims.py", "per_motor_joint": per_motor,
                               "per_body_rmd_prims": {k: v for k, v in per_body.items() if v}}, indent=1),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
