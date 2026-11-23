"""Newline-insensitive serial-model sha pins (item 374, dropbear-backlog notes/374; closes item 372).

Text pins (dropbear_serial.xml, dropbear_serial_motionlib.xml, USD body properties; hw ``source_xml_sha256``) hash
CRLF->LF-normalised bytes and carry ``"sha256_newlines": "lf"``; JSONs without the flag keep their raw-bytes pins;
the calibration (``-text``) and the binary raw sweep stay raw-bytes pins. Build metadata paths are repo-relative.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.kinematics import serial_model as sm
from dropbear_wbc.kinematics import serial_provenance as sp

ROOT = Path(__file__).resolve().parents[1]
CRLF_FIXED = ("data/robot/actuators_datasheet_v1.json", "data/robot/dropbear_serial.json",
              "data/robot/dropbear_serial.xml", "data/robot/dropbear_serial_motionlib.xml",
              "data/robot/dropbear_serial_scene.xml", "data/robot/usd_body_properties_45586414.json")


def _raw(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _crlf(p: Path) -> None:
    b = p.read_bytes().replace(b"\r\n", b"\n")
    p.write_bytes(b.replace(b"\n", b"\r\n"))


def _whats(f):
    return sorted((x["stage"], x["what"]) for x in f)


def _meta(root=ROOT) -> dict:
    return json.loads((Path(root) / sp.SERIAL_META_REL).read_text(encoding="utf-8"))


@pytest.fixture()
def tree(tmp_path):
    dst = tmp_path / "repo"
    inp = _meta()["inputs"]
    files = [sp.SERIAL_META_REL, sp.SERIAL_XML_REL, sp.MOTIONLIB_XML_REL, sp.CALIBRATION_REL]
    files += [str(sp.repo_path(inp[k], ROOT).relative_to(ROOT)) for k in ("body_properties", "raw_sweep")]
    for p in sp.hw_surrogate_jsons(ROOT):
        files += [str(p.relative_to(ROOT)), str(p.with_suffix(".xml").relative_to(ROOT))]
    for f in files:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / f, dst / f)
    return dst


# ---------------------------------------------------------------- hashing

def test_text_sha256_is_newline_insensitive(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_bytes(b"<x>\n  <y/>\n</x>\n")
    b.write_bytes(b"<x>\r\n  <y/>\r\n</x>\r\n")
    c.write_bytes(b"<x>\n  <z/>\n</x>\n")
    assert sp.text_sha256(a) == sp.text_sha256(b) == _raw(a)
    assert _raw(b) != _raw(a) and sp.text_sha256(c) != sp.text_sha256(a)
    assert sp.pin_matches(b, _raw(a), True) and not sp.pin_matches(b, _raw(a), False)
    assert sp.pin_matches(b, _raw(b), False) and not sp.pin_matches(b, _raw(b), True)
    assert not sp.pin_matches(a, None, True) and not sp.pin_matches(a, 5, False)
    assert sp.lf_pins({"sha256_newlines": "lf"}) and not sp.lf_pins({}) and not sp.lf_pins({"sha256_newlines": "crlf"})


# ---------------------------------------------------------------- committed state (372 closed)

def test_committed_pins_are_lf_text_pins():
    meta = _meta()
    assert meta["sha256_newlines"] == "lf"
    assert meta["xml_sha256"] == sp.text_sha256(ROOT / sp.SERIAL_XML_REL)
    assert meta["motionlib_xml_sha256"] == sp.text_sha256(ROOT / sp.MOTIONLIB_XML_REL)
    inp = meta["inputs"]
    assert inp["body_properties_sha256"] == sp.text_sha256(sp.repo_path(inp["body_properties"]))
    assert inp["calibration_sha256"] == _raw(ROOT / sp.CALIBRATION_REL)  # -text, raw identity
    for p in sp.hw_surrogate_jsons(ROOT):
        hm = json.loads(p.read_text(encoding="utf-8"))
        assert hm["sha256_newlines"] == "lf", p.name
        assert hm["source_xml_sha256"] == meta["xml_sha256"], p.name
        urdf = p.with_suffix(".urdf")
        if urdf.is_file():
            assert f"(sha256 {meta['xml_sha256'][:16]})" in urdf.read_text(encoding="utf-8").splitlines()[1], urdf.name


def test_build_meta_paths_are_repo_relative():
    meta = _meta()
    paths = [meta["inputs"][k] for k in ("calibration", "raw_sweep", "body_properties", "sole_hulls")]
    paths += [v for v in meta["files"].values() if v] + [meta["visual_meshes"]["dir"]]
    for s in paths:
        assert not Path(s).is_absolute() and ":" not in s and "\\" not in s, s
        assert (ROOT / s).exists(), s


def test_committed_data_robot_files_are_lf_in_the_index():
    try:
        out = subprocess.run(["git", "ls-files", "--eol", "--", *CRLF_FIXED], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("no git checkout")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        pytest.skip("files not tracked here")
    assert len(lines) == len(CRLF_FIXED)
    assert all(ln.split()[0] == "i/lf" for ln in lines), out


# ---------------------------------------------------------------- provenance checks

def test_crlf_checkout_of_every_text_pin_is_consistent(tree):
    assert sp.provenance_findings(tree) == []
    for rel in (sp.SERIAL_XML_REL, sp.MOTIONLIB_XML_REL, sp.repo_path(_meta(tree)["inputs"]["body_properties"], tree)):
        _crlf(tree / rel)
    assert sp.provenance_findings(tree) == []


def test_content_edit_is_still_caught_through_crlf(tree):
    x = tree / sp.SERIAL_XML_REL
    x.write_bytes(x.read_bytes().replace(b'<mujoco', b'<mujoco ', 1))
    f = sp.provenance_findings(tree)
    assert ("serial", "xml_sha256") in _whats(f)
    assert sum(w == "source_xml_sha256" for _, w in _whats(f)) == len(sp.hw_surrogate_jsons(tree))


def test_motionlib_pin_is_checked(tree):
    ml = tree / sp.MOTIONLIB_XML_REL
    ml.write_bytes(ml.read_bytes() + b"<!-- edit -->")
    f = sp.provenance_findings(tree)
    assert _whats(f) == [("serial", "motionlib_xml_sha256")]
    assert sp.text_sha256(ml)[:16] in f[0]["detail"]
    ml.unlink()
    f = sp.provenance_findings(tree)
    assert _whats(f) == [("serial", "motionlib_xml_sha256")] and "missing" in f[0]["detail"]


def test_body_properties_edit_is_caught(tree):
    bp = sp.repo_path(_meta(tree)["inputs"]["body_properties"], tree)
    bp.write_bytes(bp.read_bytes() + b"\n ")
    assert _whats(sp.provenance_findings(tree)) == [("serial", "body_properties_sha256")]


def _set(p: Path, **kv):
    d = json.loads(p.read_text(encoding="utf-8"))
    for k, v in kv.items():
        if v is None:
            d.pop(k, None)
        else:
            d[k] = v
    p.write_text(json.dumps(d), encoding="utf-8")


def test_legacy_hw_json_raw_pin_accepted_and_strict(tree):
    x = tree / sp.SERIAL_XML_REL
    _crlf(x)  # as committed before item 374
    hw = tree / "data/robot/dropbear_serial_hw_v1.json"
    _set(hw, sha256_newlines=None, source_xml_sha256=_raw(x))  # legacy: raw CRLF bytes
    assert sp.provenance_findings(tree) == []
    _set(hw, source_xml_sha256=sp.text_sha256(x))  # no flag -> raw compare -> text sha is not accepted
    assert _whats(sp.provenance_findings(tree)) == [("hw:hw_v1", "source_xml_sha256")]
    _set(hw, sha256_newlines="lf", source_xml_sha256=_raw(x))  # flagged -> raw CRLF sha is not accepted
    assert _whats(sp.provenance_findings(tree)) == [("hw:hw_v1", "source_xml_sha256")]


def test_legacy_serial_meta_raw_pins_accepted(tree):
    x, ml = tree / sp.SERIAL_XML_REL, tree / sp.MOTIONLIB_XML_REL
    bp = sp.repo_path(_meta(tree)["inputs"]["body_properties"], tree)
    for p in (x, ml, bp):
        _crlf(p)
    meta_p = tree / sp.SERIAL_META_REL
    d = _meta(tree)
    d.pop("sha256_newlines")
    d["xml_sha256"], d["motionlib_xml_sha256"] = _raw(x), _raw(ml)
    d["inputs"]["body_properties_sha256"] = _raw(bp)
    meta_p.write_text(json.dumps(d), encoding="utf-8")
    for p in sp.hw_surrogate_jsons(tree):
        _set(p, sha256_newlines=None, source_xml_sha256=_raw(x))
    assert sp.provenance_findings(tree) == []
    x.write_bytes(x.read_bytes().replace(b"\r\n", b"\n"))  # LF re-checkout of a legacy CRLF pin: stale (regenerate)
    whats = _whats(sp.provenance_findings(tree))
    assert ("serial", "xml_sha256") in whats and ("hw:hw_v1", "source_xml_sha256") in whats


def test_raw_sweep_and_calibration_stay_raw_pins(tree):
    meta_p = tree / sp.SERIAL_META_REL
    d = _meta(tree)
    raw = sp.repo_path(d["inputs"]["raw_sweep"], tree)
    assert b"\r\n" in raw.read_bytes()  # binary npz: normalising would alias different bytes
    d["inputs"]["raw_sweep_sha256"] = sp.text_sha256(raw)
    meta_p.write_text(json.dumps(d), encoding="utf-8")
    assert _whats(sp.provenance_findings(tree)) == [("serial", "raw_sweep_sha256")]
    cal = tree / sp.CALIBRATION_REL
    assert b"\r\n" in cal.read_bytes()
    d["inputs"]["raw_sweep_sha256"] = _raw(raw)
    d["inputs"]["calibration_sha256"] = sp.text_sha256(cal)
    meta_p.write_text(json.dumps(d), encoding="utf-8")
    assert _whats(sp.provenance_findings(tree)) == [("serial", "calibration_sha256")]


# ---------------------------------------------------------------- writers

def test_serial_model_paths_portable():
    assert sm.resolve_path("H:/dropbear/dropbear-wbc/data/robot/dropbear_serial.xml") == ROOT / sp.SERIAL_XML_REL
    assert sm.resolve_path("H:\\dropbear\\dropbear-wbc\\data\\robot\\x.json") == ROOT / "data/robot/x.json"
    assert sm.resolve_path("data/robot/x.json") == ROOT / "data/robot/x.json"
    assert sm.resolve_path("/nonexistent/dropbear-wbc/data/robot/dropbear_serial.xml") == ROOT / sp.SERIAL_XML_REL
    assert sm.resolve_path("/nonexistent/other/x.json") == Path("/nonexistent/other/x.json")
    assert sm.rel_path(ROOT / sp.SERIAL_XML_REL) == sp.SERIAL_XML_REL
    assert sm.rel_path("/nonexistent/other/x.json") == "/nonexistent/other/x.json"


def test_export_serial_hw_writes_lf_pin(tmp_path, monkeypatch):
    from dropbear_wbc.kinematics import serial_hw_actuators as sh
    spec = importlib.util.spec_from_file_location("export_serial_hw", ROOT / "tools/export_serial_hw.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--map", "user_2026_09_25", "--out", str(tmp_path), "--no-urdf"])
    hm = json.loads((tmp_path / "dropbear_serial_hw_v1.json").read_text(encoding="utf-8"))
    assert hm["sha256_newlines"] == "lf"
    assert hm["source_xml_sha256"] == sp.text_sha256(ROOT / sp.SERIAL_XML_REL)
    assert hm == json.loads((ROOT / "data/robot/dropbear_serial_hw_v1.json").read_text(encoding="utf-8"))
    crlf = tmp_path / "src" / "dropbear_serial.xml"  # a Windows (CRLF) working tree exports the same pin
    crlf.parent.mkdir()
    shutil.copyfile(ROOT / sp.SERIAL_XML_REL, crlf)
    _crlf(crlf)
    monkeypatch.setattr(sh, "SERIAL_XML", crlf)
    mod.main(["--map", "user_2026_09_25", "--out", str(tmp_path / "w"), "--no-urdf"])
    assert json.loads((tmp_path / "w/dropbear_serial_hw_v1.json").read_text(encoding="utf-8")) == hm


def test_gmr_serial_provenance_uses_the_text_pin(tmp_path):
    from dropbear_wbc.motion import gmr_serial
    rec = gmr_serial.serial_provenance(ROOT / sp.SERIAL_XML_REL)
    assert rec["serial_model_xml_sha256"] == _meta()["xml_sha256"]
    shutil.copyfile(ROOT / sp.SERIAL_XML_REL, tmp_path / "dropbear_serial.xml")
    shutil.copyfile(ROOT / sp.SERIAL_META_REL, tmp_path / "dropbear_serial.json")
    _crlf(tmp_path / "dropbear_serial.xml")
    assert gmr_serial.serial_provenance(tmp_path / "dropbear_serial.xml")["serial_model_xml_sha256"] == _meta()["xml_sha256"]


def test_rebuild_reproduces_committed_pins_and_paths(tmp_path, monkeypatch):
    """tools/build_serial_mjcf.py code path into a temp dir: same text pins and repo-relative inputs as committed,
    also from a CRLF copy of the body properties; the MJCF pins never use the raw-bytes hash (on Windows
    write_text emits CRLF, so a raw pin would differ between platforms)."""
    pytest.importorskip("mujoco")
    meta = _meta()
    raw = sp.repo_path(meta["inputs"]["raw_sweep"])
    if not raw.is_file():
        pytest.skip(f"raw sweep {raw} not in this checkout")
    bp = tmp_path / "props" / Path(meta["inputs"]["body_properties"]).name
    bp.parent.mkdir()
    shutil.copyfile(sp.repo_path(meta["inputs"]["body_properties"]), bp)
    _crlf(bp)
    spec = sm.fit_serial_model(ROOT / sp.CALIBRATION_REL, body_props=bp, log=lambda *a, **k: None)

    def raw_hash(p):
        raise AssertionError(f"raw-bytes sha256 of {p} in write_mjcf")
    monkeypatch.setattr(sm, "sha256_file", raw_hash)
    out = tmp_path / "out"
    out.mkdir()
    fresh = sm.write_mjcf(spec, out / "dropbear_serial.xml", out / "dropbear_serial.json", out / "dropbear_serial_scene.xml")
    assert fresh["sha256_newlines"] == "lf"
    assert fresh["xml_sha256"] == sp.text_sha256(out / "dropbear_serial.xml") == meta["xml_sha256"]
    assert fresh["motionlib_xml_sha256"] == sp.text_sha256(out / "dropbear_serial_motionlib.xml") \
        == meta["motionlib_xml_sha256"]
    assert fresh["inputs"]["body_properties_sha256"] == meta["inputs"]["body_properties_sha256"]
    for k in ("calibration", "calibration_sha256", "raw_sweep", "raw_sweep_sha256", "sole_hulls"):
        assert fresh["inputs"][k] == meta["inputs"][k], k
