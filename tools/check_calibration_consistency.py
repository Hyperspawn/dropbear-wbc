"""Post-recalibration gate: serial MJCF provenance + fresh hw MJCF export + the sign / symmetry test set (item 371).

Run after any ``tools/calibrate_semantics.py`` re-fit or USD re-author, before training::

    python tools/check_calibration_consistency.py            # check only; exit 1 on a stale model or a failed test
    python tools/check_calibration_consistency.py --write    # regenerate the stale serial / hw models, then check
    python tools/check_calibration_consistency.py --no-tests # provenance + MJCF only (seconds)

Stages (dropbear-backlog notes/342 re-run list, notes/371):

1. provenance (``dropbear_wbc.kinematics.serial_provenance``): ``dropbear_serial.json`` inputs (calibration, USD,
   USD body properties, raw sweep) and its own ``xml_sha256`` re-hashed; every ``dropbear_serial_hw_*.json``
   ``source_xml_sha256`` against the current ``dropbear_serial.xml``. Fix: ``tools/build_serial_mjcf.py`` then
   ``tools/export_serial_hw.py --map M``.
2. hw MJCF: each committed ``dropbear_serial_hw_<profile>.xml`` (``dropbear_serial_hw_v1.xml`` is the one the FK sign
   tests read) equals what ``export_serial_hw`` writes from the current calibration + ``hw_motor_specs`` (newlines
   normalized). Fix: ``tools/export_serial_hw.py --map M``.
3. tests: ``RERUN_TESTS`` + ``STALENESS_TESTS`` in one pytest process (venv with mujoco, PYTHONPATH=source). Every
   file must exist and run >= 1 test; any failure, error, or skip outside ``ALLOWED_SKIPS`` fails the gate (a missing
   mujoco would otherwise skip the FK sign tests and pass vacuously). Skipped when stages 1-2 are stale (the tests
   would read the stale model) unless ``--keep-going``.

``--write`` (only on this checkout): rebuilds the serial model with ``tools/build_serial_mjcf.py`` when stage 1 finds a
serial-level break (needs the raw sweep and the body properties), re-exports every hw profile whose files are stale
(all of them after a rebuild), then re-checks. Exit 0 = consistent and green, 1 = stale or failing, 2 = usage.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.kinematics import serial_provenance as sp  # noqa: E402

# notes/342 "Re-run list after tools/calibrate_semantics.py or a USD change" (the FK tests read dropbear_serial_hw_v1.xml)
RERUN_TESTS: tuple[str, ...] = (
    "tests/test_motor_sign_constants.py",
    "tests/test_arm_swing_sign.py",
    "tests/test_locomotion_symmetry.py",
    "tests/test_clamp_mirror.py",
    "tests/test_semantic_map.py",
    "tests/test_teleop_arm_ik.py",
)
# the existing staleness / parity gates of the serial model and its hw surrogates
STALENESS_TESTS: tuple[str, ...] = (
    "tests/test_serial_mjcf.py",
    "tests/test_serial_hw_actuators.py",
)
# skips that do not depend on the calibration (external clones); anything else skipped fails the gate
ALLOWED_SKIPS: dict[str, str] = {
    "tests/test_serial_mjcf.py::test_sonic_humanoid_batch_fk": "GR00T SONIC reference clone (external, optional)",
}


def mjcf_findings(root: Path = REPO) -> list[dict[str, str]]:
    """Committed hw MJCFs vs a fresh in-memory export (same code path as tools/export_serial_hw.py)."""
    from dropbear_wbc.kinematics import serial_hw_actuators as sh

    root = Path(root)
    xml_text = (root / sp.SERIAL_XML_REL).read_text(encoding="utf-8")
    jac = sh.standing_jacobian(root / sp.CALIBRATION_REL)
    si = sh.semantic_index(root / sp.SERIAL_META_REL)
    out = []
    for p in sp.hw_surrogate_jsons(root):
        prof, xml_p = p.stem.removeprefix("dropbear_serial_"), p.with_suffix(".xml")
        map_name = json.loads(p.read_text(encoding="utf-8"))["map"]
        if not xml_p.is_file():
            continue  # reported by provenance_findings
        fresh = sh.hw_mjcf_text(xml_text, sh.hw_surrogates(jac, si, map_name), map_name)
        committed = xml_p.read_text(encoding="utf-8")  # universal newlines: a CRLF working tree is not stale
        if committed != fresh:
            a, b = committed.splitlines(), fresh.splitlines()
            n = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            out.append({"stage": f"hw:{prof}", "what": "mjcf",
                        "detail": f"{xml_p.name} differs from a fresh export in {n} lines (first line {first + 1})",
                        "fix": f"run {sp.EXPORT_TOOL} --map {map_name}"})
    return out


def missing_test_files(root: Path = REPO, files=RERUN_TESTS + STALENESS_TESTS) -> list[str]:
    return [f for f in files if not (Path(root) / f).is_file()]


def parse_junit(path: Path, files) -> dict:
    """Outcome counts + per-file executed counts from a pytest --junitxml report."""
    res = {"passed": 0, "failed": [], "skipped": [], "per_file": {f: 0 for f in files}}
    for tc in ET.parse(path).getroot().iter("testcase"):
        cls, name = tc.get("classname", ""), tc.get("name", "")
        rel = next((x for x in files if any(cls == m or cls.startswith(m + ".")
                                            for m in (x[:-3].replace("/", "."), Path(x).stem))), None)
        nid = f"{rel or cls}::{name.split('[')[0]}"
        kids = {c.tag for c in tc}
        if kids & {"failure", "error"}:
            res["failed"].append(f"{rel or cls}::{name}")
        elif "skipped" in kids:
            msg = next(c.get("message", "") for c in tc if c.tag == "skipped")
            res["skipped"].append({"id": nid, "full": f"{rel or cls}::{name}", "message": msg})
            continue
        else:
            res["passed"] += 1
        if rel:
            res["per_file"][rel] += 1
    return res


def test_problems(res: dict, rc: int, allowed=ALLOWED_SKIPS, allow_skips: bool = False) -> list[str]:
    out = [f"FAILED {t}" for t in res["failed"]]
    out += [f"SKIPPED (not allowed) {s['full']}: {s['message']}" for s in res["skipped"]
            if not allow_skips and s["id"] not in allowed]
    out += [f"NO TESTS RAN in {f}" for f, n in res["per_file"].items() if n == 0]
    if rc not in (0, 1) or (rc == 1 and not res["failed"]):
        out.append(f"pytest exit code {rc}")
    return out


def run_tests(root: Path = REPO, files=RERUN_TESTS + STALENESS_TESTS, allow_skips: bool = False,
              extra_args=()) -> tuple[list[str], dict]:
    root = Path(root)
    with tempfile.TemporaryDirectory() as td:
        junit = Path(td) / "junit.xml"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(root / "source")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", *extra_args, *files]
        rc = subprocess.run(cmd, cwd=str(root), env=env).returncode
        if not junit.is_file():
            return [f"pytest produced no report (exit code {rc})"], {}
        res = parse_junit(junit, files)
    return test_problems(res, rc, allow_skips=allow_skips), res


def _run(cmd: list[str], root: Path) -> int:
    print("[check_calibration_consistency] $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(root)).returncode


def regenerate(findings: list[dict[str, str]], root: Path = REPO, runner=_run) -> int:
    """Rebuild what ``findings`` say is stale: serial model first (if any serial-level break), then hw exports."""
    root = Path(root)
    maps: list[str] = []
    serial = any(f["stage"] == "serial" for f in findings)
    if serial:
        rc = runner([sys.executable, str(root / sp.BUILD_TOOL)], root)
        if rc:
            return rc
    for p in sp.hw_surrogate_jsons(root):
        prof = p.stem.removeprefix("dropbear_serial_")
        m = json.loads(p.read_text(encoding="utf-8"))["map"]
        if (serial or any(f["stage"] == f"hw:{prof}" for f in findings)) and m not in maps:
            maps.append(m)
    for m in maps:
        rc = runner([sys.executable, str(root / sp.EXPORT_TOOL), "--map", m], root)
        if rc:
            return rc
    return 0


def check_models(root: Path = REPO, check_raw_sweep: bool = True) -> list[dict[str, str]]:
    f = sp.provenance_findings(root, check_raw_sweep=check_raw_sweep)
    if not any(x["stage"] == "serial" for x in f):  # a stale serial model makes every export stale anyway
        f += mjcf_findings(root)
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO, help=argparse.SUPPRESS)
    ap.add_argument("--write", action="store_true", help="regenerate stale serial / hw models, then check")
    ap.add_argument("--no-tests", action="store_true", help="provenance + MJCF only")
    ap.add_argument("--keep-going", action="store_true", help="run the tests even when the models are stale")
    ap.add_argument("--allow-skips", action="store_true", help="do not fail on skipped tests (reported)")
    ap.add_argument("--no-raw-sweep", action="store_true", help="do not hash the raw sweep (bundle without logs/)")
    ap.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    if a.write and root != REPO:
        ap.error("--write regenerates this checkout only")
    findings = check_models(root, not a.no_raw_sweep)
    if findings and a.write:
        rc = regenerate(findings, root)
        if rc:
            print(f"[check_calibration_consistency] regeneration failed (exit {rc})", flush=True)
            return 1
        findings = check_models(root, not a.no_raw_sweep)
    for f in findings:
        print(f"[check_calibration_consistency] STALE {f['stage']} {f['what']}: {f['detail']} -> {f['fix']}", flush=True)
    report: dict = {"models": "stale" if findings else "ok", "findings": findings, "tests": None, "test_problems": []}
    missing = missing_test_files(root)
    problems = [f"MISSING test file {m}" for m in missing]
    if not a.no_tests and not missing and (not findings or a.keep_going):
        tp, res = run_tests(root, allow_skips=a.allow_skips)
        problems += tp
        report["tests"] = {k: res.get(k) for k in ("passed", "failed", "skipped", "per_file")} if res else None
    elif findings and not a.no_tests:
        problems.append("tests not run: models stale (regenerate first, or --keep-going)")
    report["test_problems"] = problems
    for p in problems:
        print(f"[check_calibration_consistency] {p}", flush=True)
    ok = not findings and not problems
    t = report["tests"]
    summary = (f"models {report['models']}; tests " +
               (f"{t['passed']} passed, {len(t['failed'])} failed, {len(t['skipped'])} skipped" if t else "not run"))
    print(f"[check_calibration_consistency] {'OK' if ok else 'FAIL'}: {summary}", flush=True)
    if a.json:
        a.json.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
