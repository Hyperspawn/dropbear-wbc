"""Write the hardware-surrogate variants of the DERIVED serial model (see dropbear_wbc.kinematics.serial_hw_actuators).

    python tools/export_serial_hw.py [--map user_2026_09_25] [--out data/robot]

Outputs: ``dropbear_serial_<profile>.xml`` (same kinematics/masses as dropbear_serial.xml, hw armature/effort/servos),
``dropbear_serial_<profile>.urdf`` (meshes/ relative) and ``dropbear_serial_<profile>.json`` (per-joint hw and legacy
surrogates + J). Needs ``mujoco`` for the URDF; CPU only. Rerun after tools/build_serial_mjcf.py or a motor-map change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.kinematics import serial_hw_actuators as sh  # noqa: E402
from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS  # noqa: E402

PROFILE_FOR_MAP: dict[str, str] = {}
for _p, _m in HW_PROFILE_MAPS.items():
    PROFILE_FOR_MAP.setdefault(_m, _p)  # first profile per map (hw_v1 for the user map)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map", default="user_2026_09_25", choices=sorted(PROFILE_FOR_MAP))
    ap.add_argument("--out", type=Path, default=REPO / "data/robot")
    ap.add_argument("--no-urdf", action="store_true")
    a = ap.parse_args(argv)
    prof = PROFILE_FOR_MAP[a.map]
    si = sh.semantic_index()
    jac = sh.standing_jacobian()
    sur = sh.hw_surrogates(jac, si, a.map)
    xml = sh.SERIAL_XML.read_text()
    sha = hashlib.sha256(sh.SERIAL_XML.read_bytes()).hexdigest()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"dropbear_serial_{prof}.xml").write_text(sh.hw_mjcf_text(xml, sur, a.map))
    meta = {"schema": "dropbear-serial-hw-surrogate-v1", "source_xml_sha256": sha, "map": a.map, "profile": prof,
            "method": "J = dm/ds at standing_semantic_pos; effort = sum|J| x peak|rated; velocity = min no_load/|J|; "
                      "armature = sum J^2 rotor N^2; kp/kd = sum J^2 HW_GAINS",
            "joints": sur, "legacy": sh.legacy_surrogates(jac, si)}
    (a.out / f"dropbear_serial_{prof}.json").write_text(json.dumps(meta, indent=1))
    if not a.no_urdf:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(sh.SERIAL_XML))
        (a.out / f"dropbear_serial_{prof}.urdf").write_text(sh.urdf_text(model, sur, a.map, sha))
    worst = max(si, key=lambda j: meta["legacy"][j]["effort"] / sur[j]["effort_peak"])
    print(f"[export_serial_hw] {prof}: wrote xml/json{'' if a.no_urdf else '/urdf'} -> {a.out}; worst legacy/hw "
          f"effort {worst} {meta['legacy'][worst]['effort']:.1f}/{sur[worst]['effort_peak']:.1f} N*m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
