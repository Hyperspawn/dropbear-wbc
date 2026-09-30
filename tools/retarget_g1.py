"""Retarget one Unitree-G1 motion clip (any supported format) to a Dropbear motion CSV + sidecar.

Writes ``<out-root>/<source>/<clip>.csv``, ``<clip>.json`` and ``semantic/<clip>.semantic.csv``
(contract ``dropbear-motion-csv-v1``, docs/CONTRACTS.md section 3).

Examples (system Python 3.12, CPU only)::

    python tools/retarget_g1.py $DROPBEAR_UPSTREAM/ProtoMotions/data/g1-kimodo-generated/output_wave.csv
    python tools/retarget_g1.py some_lafan1_clip.csv --source lafan1_g1
    # before the real calibration exists (outputs go to data/motions_mock/ and are labelled MOCK):
    python tools/retarget_g1.py clip.csv --calibration tests/fixtures/mock_semantic_calibration.json --allow-mock

Formats / sources: see ``dropbear_wbc.motion.g1_sources.SOURCES`` (``--list-sources``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "source", REPO / "third_party" / "pydeps"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dropbear_wbc.motion.calibration_view import DEFAULT_CALIBRATION, load_calibration  # noqa: E402
from dropbear_wbc.motion.contacts import ContactParams  # noqa: E402
from dropbear_wbc.motion.g1_sources import SOURCES, source_summary  # noqa: E402
from dropbear_wbc.motion.g1_to_dropbear import RetargetOptions  # noqa: E402
from dropbear_wbc.motion.pipeline import retarget_file  # noqa: E402

DEFAULT_OUT = REPO / "data" / "motions"
MOCK_OUT = REPO / "data" / "motions_mock"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="G1 motion file (.csv / .pkl) or SONIC reference folder")
    ap.add_argument("--source", choices=sorted(SOURCES), help="source key (format, fps, license); inferred if omitted")
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    ap.add_argument("--allow-mock", action="store_true", help="accept a calibration flagged MOCK")
    ap.add_argument("--out-root", type=Path, default=None, help=f"default {DEFAULT_OUT} ({MOCK_OUT} for MOCK)")
    ap.add_argument("--waist-mode", choices=("auto", "fold", "drop"), default="auto")
    ap.add_argument("--joint-mapping", choices=("anatomical", "by_name"), default="anatomical")
    ap.add_argument("--no-ground-fix", action="store_true")
    ap.add_argument("--no-foot-lock", action="store_true")
    ap.add_argument("--output-fps", type=float, default=None,
                    help="resample (default: 50 with the foot-contact stage, else keep source fps)")
    ap.add_argument("--foot-contact", action=argparse.BooleanOptionalAction, default=None,
                    help="foot-contact stage (dropbear_wbc.motion.foot_contact; default ON except with a MOCK "
                         "calibration; --no-foot-contact = legacy v3 route)")
    ap.add_argument("--max-motor-step", type=float, default=None,
                    help="motor-step projection [rad/frame] (default 0.18 with the foot stage, else off; <= 0 = off)")
    ap.add_argument("--contact-height", type=float, default=ContactParams.height_thresh)
    ap.add_argument("--contact-speed", type=float, default=ContactParams.speed_thresh)
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="stretch the clip in time by this factor (Froude-consistent G1 -> Dropbear: ~1.19 = "
                         "sqrt(hip-height ratio)); the output clip name gets a _ts<k> suffix")
    ap.add_argument("--list-sources", action="store_true")
    return ap


def options_from_args(args: argparse.Namespace, mock: bool = False) -> RetargetOptions:
    from dropbear_wbc.motion.foot_contact import FootContactParams

    foot = (not mock) if getattr(args, "foot_contact", None) is None else bool(args.foot_contact)
    step = getattr(args, "max_motor_step", None)
    step = (0.18 if foot else None) if step is None else (step if step > 0 else None)
    fps = args.output_fps if args.output_fps is not None else (50.0 if foot else None)
    return RetargetOptions(
        waist_mode=args.waist_mode,
        joint_mapping=args.joint_mapping,
        ground_fix=not args.no_ground_fix,
        foot_lock_xy=not args.no_foot_lock,
        output_fps=fps,
        contact=ContactParams(height_thresh=args.contact_height, speed_thresh=args.contact_speed),
        foot_contact=FootContactParams() if foot else None,
        max_motor_step_rad=step,
        time_scale=float(getattr(args, "time_scale", 1.0) or 1.0),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_sources:
        print(source_summary())
        return 0
    if not args.input:
        print("error: input required", file=sys.stderr)
        return 2
    cal = load_calibration(args.calibration, allow_mock=args.allow_mock)
    out_root = args.out_root or (MOCK_OUT if cal.is_mock else DEFAULT_OUT)
    if cal.is_mock and out_root.resolve() == DEFAULT_OUT.resolve():
        print("error: refusing to write MOCK-calibrated clips into data/motions", file=sys.stderr)
        return 2
    entry = retarget_file(args.input, out_root, source=args.source, cal=cal, opts=options_from_args(args, mock=cal.is_mock))
    print(json.dumps(entry, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
