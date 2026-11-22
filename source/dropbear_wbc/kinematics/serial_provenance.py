"""Provenance chain calibration / USD -> serial MJCF -> hw surrogate MJCFs (item 371, dropbear-backlog notes/371).

``data/robot/dropbear_serial.json`` records the SHA-256 of every input of ``tools/build_serial_mjcf.py`` (semantic
calibration, raw physics sweep, USD, USD body properties) and of the MJCF it wrote; every
``dropbear_serial_hw_<profile>.json`` records the SHA-256 of the ``dropbear_serial.xml`` it was exported from
(``tools/export_serial_hw.py``). :func:`provenance_findings` re-hashes the files and lists every broken link.

Stdlib only (hashlib + json): ``scripts/train.py`` records :func:`run_info_record` in run_info without importing
MuJoCo. The full check (fresh hw MJCF export + the sign / symmetry test set) is ``tools/check_calibration_consistency.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SERIAL_META_REL = "data/robot/dropbear_serial.json"
SERIAL_XML_REL = "data/robot/dropbear_serial.xml"
CALIBRATION_REL = "data/calibration/dropbear_semantic_calibration.json"
BUILD_TOOL = "tools/build_serial_mjcf.py"
EXPORT_TOOL = "tools/export_serial_hw.py"
CHECK_TOOL = "tools/check_calibration_consistency.py"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_path(p: str | Path, root: Path = REPO) -> Path:
    """The serial meta records absolute build paths (``H:/dropbear/dropbear-wbc/...``): resolve them inside ``root``."""
    s = str(p).replace("\\", "/")
    if "dropbear-wbc/" in s:
        return Path(root) / s.split("dropbear-wbc/", 1)[1]
    q = Path(p)
    return q if q.is_absolute() else Path(root) / q


def hw_surrogate_jsons(root: Path = REPO) -> list[Path]:
    return sorted((Path(root) / "data/robot").glob("dropbear_serial_hw_*.json"))


def _finding(stage: str, what: str, detail: str, fix: str) -> dict[str, str]:
    return {"stage": stage, "what": what, "detail": detail, "fix": fix}


def provenance_findings(root: str | Path = REPO, calibration: str | Path | None = None, *,
                        check_raw_sweep: bool = True) -> list[dict[str, str]]:
    """Every broken link of calibration/USD -> dropbear_serial.* -> dropbear_serial_hw_*.json (empty list = consistent).

    ``calibration`` defaults to ``<root>/data/calibration/dropbear_semantic_calibration.json`` (pass the training
    calibration, e.g. ``$DROPBEAR_CALIBRATION_JSON``, to check that the serial model was built from it).
    Findings are dicts ``stage`` ("serial" or "hw:<profile>"), ``what``, ``detail``, ``fix``.
    """
    root = Path(root)
    cal = Path(calibration) if calibration is not None else root / CALIBRATION_REL
    meta_p, xml_p = root / SERIAL_META_REL, root / SERIAL_XML_REL
    build = f"run {BUILD_TOOL} (then {EXPORT_TOOL} per hw map)"
    out: list[dict[str, str]] = []
    for p in (cal, meta_p, xml_p):
        if not p.is_file():
            out.append(_finding("serial", "missing", str(p), build))
    if out:
        return out
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    inp = meta.get("inputs") or {}
    cal_sha = sha256_file(cal)
    if inp.get("calibration_sha256") != cal_sha:
        out.append(_finding("serial", "calibration_sha256",
                            f"serial model built from {str(inp.get('calibration_sha256'))[:16]}, calibration "
                            f"{cal.name} is {cal_sha[:16]}", build))
    cal_usd = json.loads(cal.read_text(encoding="utf-8")).get("usd_sha256")
    if inp.get("usd_sha256") != cal_usd:
        out.append(_finding("serial", "usd_sha256", f"serial model USD {str(inp.get('usd_sha256'))[:16]}, calibration "
                            f"USD {str(cal_usd)[:16]} (USD re-authored: re-fit tools/calibrate_semantics.py first)", build))
    links = [("body_properties", "body_properties_sha256")]
    if check_raw_sweep:
        links.append(("raw_sweep", "raw_sweep_sha256"))
    for key, sha_key in links:
        if not inp.get(key):
            out.append(_finding("serial", sha_key, f"inputs.{key} not recorded", build))
            continue
        p = repo_path(inp[key], root)
        if not p.is_file():
            out.append(_finding("serial", sha_key, f"{p} missing (cannot verify)", f"restore {p.name} or {build}"))
        elif sha256_file(p) != inp.get(sha_key):
            out.append(_finding("serial", sha_key, f"{p.name} changed since the build "
                                f"({str(inp.get(sha_key))[:16]} -> {sha256_file(p)[:16]})", build))
    xml_sha = sha256_file(xml_p)
    if meta.get("xml_sha256") != xml_sha:
        out.append(_finding("serial", "xml_sha256", f"dropbear_serial.xml is {xml_sha[:16]}, meta records "
                            f"{str(meta.get('xml_sha256'))[:16]} (hand edit or partial rebuild)", build))
    hws = hw_surrogate_jsons(root)
    if not hws:
        out.append(_finding("hw", "missing", "no data/robot/dropbear_serial_hw_*.json", f"run {EXPORT_TOOL}"))
    for p in hws:
        prof = p.stem.removeprefix("dropbear_serial_")
        hm = json.loads(p.read_text(encoding="utf-8"))
        fix = f"run {EXPORT_TOOL} --map {hm.get('map')}"
        if hm.get("source_xml_sha256") != xml_sha:
            out.append(_finding(f"hw:{prof}", "source_xml_sha256", f"exported from dropbear_serial.xml "
                                f"{str(hm.get('source_xml_sha256'))[:16]}, current is {xml_sha[:16]}", fix))
        if not p.with_suffix(".xml").is_file():
            out.append(_finding(f"hw:{prof}", "missing", f"{p.with_suffix('.xml').name} missing", fix))
    return out


def run_info_record(root: str | Path = REPO, calibration: str | Path | None = None) -> dict[str, Any]:
    """Cheap, never-raising record for run_info: does the serial model (and so the sign / FK tests of the
    notes/342 re-run list) describe the calibration this run trains with? Raw sweep not hashed (may be absent on
    cloud bundles). ``status``: ok / stale / error."""
    if calibration is None:
        env = os.environ.get("DROPBEAR_CALIBRATION_JSON", "").strip()
        calibration = None if not env or env.lower() == "none" else env
    rec: dict[str, Any] = {"status": "error", "calibration": str(calibration or CALIBRATION_REL), "findings": [],
                           "check_tool": CHECK_TOOL}
    try:
        f = provenance_findings(root, calibration, check_raw_sweep=False)
    except Exception as e:  # noqa: BLE001 - a provenance record must never stop training
        rec["findings"] = [f"{type(e).__name__}: {e}"]
        return rec
    rec["status"] = "stale" if f else "ok"
    rec["findings"] = [f"{x['stage']} {x['what']}: {x['detail']}" for x in f]
    return rec
