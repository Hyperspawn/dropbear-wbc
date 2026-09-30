"""Patch a GMR checkout for Dropbear (idempotent) and record the patch.

GMR (https://github.com/YanjieZe/GMR, MIT) is used from a fresh shallow clone at
``$DROPBEAR_UPSTREAM/GMR-dropbear`` (never the read-only reference clone ``$DROPBEAR_UPSTREAM/GMR``).
This script applies exactly two changes and writes ``third_party/gmr_dropbear/gmr_dropbear.patch``
(``git diff`` of the clone) plus a license notice:

1. ``general_motion_retargeting/motion_retarget.py``: GMR calls
   ``mink.solve_ik(configuration, tasks, dt, solver, damping, limits)`` positionally. In mink >= 0.0.x the
   6th positional parameter is ``safety_break``, so the configuration limits were silently NOT applied
   (and a non-empty list was passed as ``safety_break``). Patched to ``damping=..., limits=...`` keywords.
2. ``general_motion_retargeting/params.py``: registers robot ``dropbear`` (MJCF
   ``<DROPBEAR_WBC_ROOT>/data/robot/dropbear_serial.xml``; IK configs
   ``data/robot/gmr/{bvh_lafan1,smplx}_to_dropbear.json``; base body ``pelvis``). ``DROPBEAR_WBC_ROOT``
   defaults to ``<repo>``. Also adds ``dropbear`` to ``scripts/bvh_to_robot.py`` /
   ``scripts/smplx_to_robot.py`` ``--robot`` choices.

Usage (any Python)::

    python tools/setup_gmr.py [--gmr $DROPBEAR_UPSTREAM/GMR-dropbear]

Environment used for GMR itself: ``.venv-gmr`` (see docs/SERIAL_MODEL_AND_GMR.md).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402

DEFAULT_GMR = _paths.upstream_dir() / "GMR-dropbear"
MARK = "# --- dropbear-wbc registration (tools/setup_gmr.py) ---"

REGISTRATION = f'''

{MARK}
import os as _os

_DROPBEAR_WBC = pathlib.Path(_os.environ.get("DROPBEAR_WBC_ROOT", r"{REPO.as_posix()}"))
# DERIVED serial output-space model of the Dropbear humanoid (joint values = dropbear-semantic-v1 angles);
# the plant is the USD. For viewing with a floor use data/robot/dropbear_serial_scene.xml.
ROBOT_XML_DICT["dropbear"] = _DROPBEAR_WBC / "data" / "robot" / "dropbear_serial.xml"
IK_CONFIG_DICT["bvh_lafan1"]["dropbear"] = _DROPBEAR_WBC / "data" / "robot" / "gmr" / "bvh_lafan1_to_dropbear.json"
IK_CONFIG_DICT["smplx"]["dropbear"] = _DROPBEAR_WBC / "data" / "robot" / "gmr" / "smplx_to_dropbear.json"
ROBOT_BASE_DICT["dropbear"] = "pelvis"
VIEWER_CAM_DISTANCE_DICT["dropbear"] = 3.0
'''

LICENSE_NOTICE = """GMR (General Motion Retargeting), https://github.com/YanjieZe/GMR
Copyright (c) Yanjie Ze and contributors. MIT License (see LICENSE in the GMR repository).

dropbear-wbc does not vendor GMR source. `gmr_dropbear.patch` is the diff applied by tools/setup_gmr.py to a
fresh shallow clone at $DROPBEAR_UPSTREAM/GMR-dropbear (commit recorded in gmr_commit.txt):
  1. mink.solve_ik limits passed by keyword (the positional call bound them to `safety_break`);
  2. 'dropbear' robot registration (params.py) + --robot choices in scripts/{bvh,smplx}_to_robot.py.
"""


def patch_solve_ik(path: Path) -> bool:
    s = path.read_text(encoding="utf-8")
    pat = re.compile(r"mink\.solve_ik\(\s*self\.configuration,\s*(self\.tasks[12]),\s*dt,\s*self\.solver,\s*"
                     r"self\.damping,\s*self\.ik_limits\s*\)")
    new, n = pat.subn(r"mink.solve_ik(self.configuration, \1, dt, self.solver, damping=self.damping, "
                      r"limits=self.ik_limits)", s)
    if n:
        path.write_text(new, encoding="utf-8")
    return n > 0


def patch_params(path: Path) -> bool:
    s = path.read_text(encoding="utf-8")
    if MARK in s:
        return False
    path.write_text(s.rstrip("\n") + "\n" + REGISTRATION, encoding="utf-8")
    return True


def patch_choices(path: Path) -> bool:
    if not path.is_file():
        return False
    s = path.read_text(encoding="utf-8")
    if '"dropbear"' in s:
        return False
    new, n = re.subn(r'(choices=\[)("unitree_g1",)', r'\1"dropbear", \2', s, count=1)
    if n:
        path.write_text(new, encoding="utf-8")
    return n > 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gmr", type=Path, default=DEFAULT_GMR)
    args = ap.parse_args(argv)
    gmr = args.gmr
    if gmr.resolve() == (_paths.upstream_dir() / "GMR").resolve():
        print("refusing to patch the read-only reference clone $DROPBEAR_UPSTREAM/GMR", file=sys.stderr)
        return 2
    pkg = gmr / "general_motion_retargeting"
    done = {
        "solve_ik_keywords": patch_solve_ik(pkg / "motion_retarget.py"),
        "params_registration": patch_params(pkg / "params.py"),
        "bvh_to_robot_choices": patch_choices(gmr / "scripts" / "bvh_to_robot.py"),
        "smplx_to_robot_choices": patch_choices(gmr / "scripts" / "smplx_to_robot.py"),
    }
    s = (pkg / "motion_retarget.py").read_text(encoding="utf-8")
    if "limits=self.ik_limits" not in s or "self.damping, self.ik_limits" in s:
        print("solve_ik patch not in place", file=sys.stderr)
        return 1
    out = REPO / "third_party" / "gmr_dropbear"
    out.mkdir(parents=True, exist_ok=True)
    diff = subprocess.run(["git", "-C", str(gmr), "diff"], capture_output=True, text=True, check=True).stdout
    (out / "gmr_dropbear.patch").write_text(diff, encoding="utf-8")
    commit = subprocess.run(["git", "-C", str(gmr), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
    (out / "gmr_commit.txt").write_text(f"{commit.strip()}  (https://github.com/YanjieZe/GMR, shallow clone)\n", encoding="utf-8")
    (out / "NOTICE.txt").write_text(LICENSE_NOTICE, encoding="utf-8")
    print(f"[setup_gmr] {gmr}: {done}; patch -> {out / 'gmr_dropbear.patch'} ({len(diff.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
