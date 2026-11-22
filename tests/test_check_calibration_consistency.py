"""Post-recalibration gate tools/check_calibration_consistency.py + serial_provenance (item 371, dropbear-backlog notes/371).

Model stage on a temp copy of the serial model inputs/outputs (stale MJCF, changed calibration / USD / body props /
raw sweep, hand-edited serial XML, stale hw JSON, missing files, CRLF-only); pytest stage on tiny fake test trees
(fail, skip, allowed skip, empty file, missing file, class-based ids); --write planning; the MIRROR_SIGN single source;
the train.py run_info record.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.kinematics import serial_provenance as sp

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_calibration_consistency",
                                               ROOT / "tools/check_calibration_consistency.py")
ccc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ccc)

NOTES_342_RERUN = ("tests/test_motor_sign_constants.py", "tests/test_arm_swing_sign.py",
                   "tests/test_locomotion_symmetry.py", "tests/test_clamp_mirror.py", "tests/test_semantic_map.py",
                   "tests/test_teleop_arm_ik.py")


# ---------------------------------------------------------------- lists

def test_rerun_list_is_the_notes_342_list_and_exists():
    assert ccc.RERUN_TESTS == NOTES_342_RERUN
    assert set(ccc.STALENESS_TESTS) == {"tests/test_serial_mjcf.py", "tests/test_serial_hw_actuators.py"}
    assert ccc.missing_test_files(ROOT) == []
    assert ccc.missing_test_files(ROOT, ccc.RERUN_TESTS + ("tests/test_nope.py",)) == ["tests/test_nope.py"]
    assert set(ccc.ALLOWED_SKIPS) == {"tests/test_serial_mjcf.py::test_sonic_humanoid_batch_fk"}


# ---------------------------------------------------------------- model stage

def _copy_tree(dst: Path) -> Path:
    meta = json.loads((ROOT / sp.SERIAL_META_REL).read_text(encoding="utf-8"))["inputs"]
    files = [sp.SERIAL_META_REL, sp.SERIAL_XML_REL, sp.CALIBRATION_REL]
    files += [str(sp.repo_path(meta[k], ROOT).relative_to(ROOT)) for k in ("body_properties", "raw_sweep")]
    files += [str(p.relative_to(ROOT)) for p in sp.hw_surrogate_jsons(ROOT)]
    files += [str(p.with_suffix(".xml").relative_to(ROOT)) for p in sp.hw_surrogate_jsons(ROOT)]
    for f in files:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / f, dst / f)
    return dst


@pytest.fixture()
def tree(tmp_path):
    return _copy_tree(tmp_path / "repo")


def _whats(findings):
    return sorted((f["stage"], f["what"]) for f in findings)


def test_checkout_is_consistent():
    assert sp.provenance_findings(ROOT) == []
    assert ccc.mjcf_findings(ROOT) == []
    assert ccc.check_models(ROOT) == []


def test_copy_is_consistent_and_crlf_is_not_stale(tree):
    assert ccc.check_models(tree) == []
    hw = tree / "data/robot/dropbear_serial_hw_v1.xml"
    hw.write_bytes(hw.read_bytes().replace(b"\n", b"\r\n"))
    assert ccc.check_models(tree) == []


def test_stale_hw_v1_mjcf(tree):
    hw = tree / "data/robot/dropbear_serial_hw_v1.xml"
    txt = hw.read_text(encoding="utf-8")
    new, n = re.subn(r'armature="([0-9.eE+-]+)"', lambda m: f'armature="{float(m.group(1)) * 1.01:.6g}"', txt, count=1)
    assert n == 1
    hw.write_text(new, encoding="utf-8")
    f = ccc.check_models(tree)
    assert _whats(f) == [("hw:hw_v1", "mjcf")]
    assert "1 lines" in f[0]["detail"] and "export_serial_hw.py --map user_2026_09_25" in f[0]["fix"]
    assert ccc.main(["--root", str(tree), "--no-tests"]) == 1


def test_hw_mjcf_missing(tree):
    (tree / "data/robot/dropbear_serial_hw_v4.xml").unlink()
    assert _whats(ccc.check_models(tree)) == [("hw:hw_v4", "missing")]


def test_recalibrated_calibration_is_stale(tree):
    cal = tree / sp.CALIBRATION_REL
    d = json.loads(cal.read_text(encoding="utf-8"))
    d["created"] = "2026-11-22T18:00:00+05:30"
    cal.write_text(json.dumps(d), encoding="utf-8")
    f = ccc.check_models(tree)
    assert _whats(f) == [("serial", "calibration_sha256")]  # mjcf stage skipped: every export is stale anyway
    assert sp.BUILD_TOOL in f[0]["fix"]
    assert ccc.main(["--root", str(tree), "--no-tests"]) == 1


def test_usd_reauthor_is_stale(tree):
    cal = tree / sp.CALIBRATION_REL
    d = json.loads(cal.read_text(encoding="utf-8"))
    d["usd_sha256"] = "0" * 64
    cal.write_text(json.dumps(d), encoding="utf-8")
    assert ("serial", "usd_sha256") in _whats(sp.provenance_findings(tree))


@pytest.mark.parametrize("key", ["body_properties", "raw_sweep"])
def test_changed_build_input_is_stale(tree, key):
    meta = json.loads((tree / sp.SERIAL_META_REL).read_text(encoding="utf-8"))["inputs"]
    p = sp.repo_path(meta[key], tree)
    p.write_bytes(p.read_bytes() + b" ")
    assert _whats(sp.provenance_findings(tree)) == [("serial", f"{key}_sha256")]
    if key == "raw_sweep":
        assert sp.provenance_findings(tree, check_raw_sweep=False) == []
    p.unlink()
    f = sp.provenance_findings(tree)
    assert _whats(f) == [("serial", f"{key}_sha256")] and "missing" in f[0]["detail"]


def test_hand_edited_serial_xml_breaks_every_hw_link(tree):
    x = tree / sp.SERIAL_XML_REL
    x.write_bytes(x.read_bytes() + b"<!-- edit -->")
    f = sp.provenance_findings(tree)
    n_hw = len(sp.hw_surrogate_jsons(tree))
    assert n_hw >= 6
    assert _whats(f).count(("serial", "xml_sha256")) == 1
    assert sum(w == "source_xml_sha256" for _, w in _whats(f)) == n_hw


def test_hw_json_exported_from_another_serial_xml(tree):
    p = tree / "data/robot/dropbear_serial_hw_cem6.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["source_xml_sha256"] = "f" * 64
    p.write_text(json.dumps(d), encoding="utf-8")
    assert _whats(ccc.check_models(tree)) == [("hw:hw_cem6", "source_xml_sha256")]


def test_missing_serial_meta_and_calibration_override(tree, tmp_path):
    other = tmp_path / "other_cal.json"
    d = json.loads((tree / sp.CALIBRATION_REL).read_text(encoding="utf-8"))
    d["note"] = "re-fit"
    other.write_text(json.dumps(d), encoding="utf-8")
    assert ("serial", "calibration_sha256") in _whats(sp.provenance_findings(tree, other))
    (tree / sp.SERIAL_META_REL).unlink()
    assert _whats(sp.provenance_findings(tree)) == [("serial", "missing")]


def test_stale_models_skip_tests_unless_keep_going(tree, monkeypatch):
    (tree / "data/robot/dropbear_serial_hw_v4.xml").unlink()
    calls = []
    monkeypatch.setattr(ccc, "missing_test_files", lambda root: [])
    monkeypatch.setattr(ccc, "run_tests", lambda root, allow_skips=False: calls.append(root) or ([], {}))
    assert ccc.main(["--root", str(tree)]) == 1 and calls == []
    assert ccc.main(["--root", str(tree), "--keep-going"]) == 1 and len(calls) == 1
    assert ccc.main(["--root", str(_copy_tree(tree.parent / "fresh"))]) == 0 and len(calls) == 2


def test_failed_tests_fail_the_gate(tree, monkeypatch):
    monkeypatch.setattr(ccc, "missing_test_files", lambda root: [])
    monkeypatch.setattr(ccc, "run_tests", lambda root, allow_skips=False: (["FAILED x"], {}))
    assert ccc.main(["--root", str(tree)]) == 1
    monkeypatch.setattr(ccc, "missing_test_files", lambda root: ["tests/test_x.py"])
    monkeypatch.setattr(ccc, "run_tests", lambda root, allow_skips=False: ([], {}))
    assert ccc.main(["--root", str(tree)]) == 1


def test_write_only_on_this_checkout(tree):
    with pytest.raises(SystemExit) as e:
        ccc.main(["--root", str(tree), "--write"])
    assert e.value.code == 2


def test_regenerate_plan(tree):
    cmds = []

    def runner(cmd, root):
        cmds.append([Path(c).name if c.endswith(".py") else c for c in cmd[1:]])
        return 0

    assert ccc.regenerate([{"stage": "hw:hw_v1", "what": "mjcf"}], tree, runner) == 0
    assert cmds == [["export_serial_hw.py", "--map", "user_2026_09_25"]]
    cmds.clear()
    assert ccc.regenerate([{"stage": "serial", "what": "calibration_sha256"}], tree, runner) == 0
    maps = {json.loads(p.read_text())["map"] for p in sp.hw_surrogate_jsons(tree)}
    assert cmds[0] == ["build_serial_mjcf.py"]
    assert sorted(c[2] for c in cmds[1:]) == sorted(maps) and all(c[0] == "export_serial_hw.py" for c in cmds[1:])
    cmds.clear()
    assert ccc.regenerate([{"stage": "serial", "what": "x"}], tree, lambda c, r: cmds.append(c) or 3) == 3
    assert len(cmds) == 1  # build failed: no export on top of a stale serial model


# ---------------------------------------------------------------- pytest stage

FAKE = {
    "tests/test_ok.py": "def test_a():\n    pass\n\n\nclass TestK:\n    def test_b(self):\n        pass\n",
    "tests/test_fail.py": "def test_a():\n    assert 1 == 2\n",
    "tests/test_skip.py": "import pytest\n\n\ndef test_a():\n    pass\n\n\ndef test_s():\n    pytest.skip('clone missing')\n",
    "tests/test_empty.py": "X = 1\n",
    "tests/test_err.py": "import pytest\n\n\n@pytest.fixture\ndef boom():\n    raise RuntimeError\n\n\n"
                         "def test_e(boom):\n    pass\n",
}


@pytest.fixture()
def fake(tmp_path):
    for f, src in FAKE.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(src)
    return tmp_path


def test_pytest_stage_pass_and_class_ids(fake):
    probs, res = ccc.run_tests(fake, ("tests/test_ok.py",))
    assert probs == [] and res["passed"] == 2 and res["per_file"] == {"tests/test_ok.py": 2}


def test_pytest_stage_failure(fake):
    probs, res = ccc.run_tests(fake, ("tests/test_ok.py", "tests/test_fail.py"))
    assert probs == ["FAILED tests/test_fail.py::test_a"] and res["passed"] == 2


def test_pytest_stage_fixture_error(fake):
    probs, _ = ccc.run_tests(fake, ("tests/test_err.py",))
    assert "FAILED tests/test_err.py::test_e" in probs


def test_pytest_stage_skip_policy(fake):
    probs, res = ccc.run_tests(fake, ("tests/test_skip.py",))
    assert probs == ["SKIPPED (not allowed) tests/test_skip.py::test_s: clone missing"]
    assert ccc.run_tests(fake, ("tests/test_skip.py",), allow_skips=True)[0] == []
    assert ccc.test_problems(res, 0, allowed={"tests/test_skip.py::test_s": "x"}) == []


def test_pytest_stage_file_without_tests(fake):
    probs, _ = ccc.run_tests(fake, ("tests/test_ok.py", "tests/test_empty.py"))
    assert probs == ["NO TESTS RAN in tests/test_empty.py"]
    only_skips = {"passed": 0, "failed": [], "skipped": [{"id": "a::b", "full": "a::b", "message": ""}],
                  "per_file": {"a": 0}}
    assert "NO TESTS RAN in a" in ccc.test_problems(only_skips, 0, allow_skips=True)


def test_pytest_stage_missing_file(fake):
    probs, _ = ccc.run_tests(fake, ("tests/test_ok.py", "tests/test_gone.py"))
    assert probs  # pytest exit 4, no report or a usage error
    assert ccc.missing_test_files(fake, ("tests/test_ok.py", "tests/test_gone.py")) == ["tests/test_gone.py"]


def test_bad_exit_code_without_failures_is_a_problem():
    res = {"passed": 1, "failed": [], "skipped": [], "per_file": {"t": 1}}
    assert ccc.test_problems(res, 0) == []
    assert ccc.test_problems(res, 1) == ["pytest exit code 1"]
    assert ccc.test_problems(res, 2) == ["pytest exit code 2"]


# ---------------------------------------------------------------- MIRROR_SIGN single source

def test_mirror_sign_has_one_source_in_tests():
    from dropbear_wbc.kinematics.calib_build import MIRROR_SIGN
    from dropbear_wbc.teleop.arm_ik import ARM_JOINTS
    from dropbear_wbc.teleop.arm_ik import MIRROR_SIGN as ARM_MIRROR

    for p in sorted((ROOT / "tests").glob("*.py")):
        for node in ast.parse(p.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
                assert not any(isinstance(t, ast.Name) and t.id == "MIRROR_SIGN" for t in tgts), p.name
    tree = ast.parse((ROOT / "tests/test_semantic_map.py").read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.ImportFrom) and n.module == "dropbear_wbc.kinematics.calib_build"
               and any(a.name == "MIRROR_SIGN" for a in n.names) for n in tree.body)
    # the teleop arm copy (source) equals the calib_build convention joint by joint
    assert list(ARM_MIRROR) == [MIRROR_SIGN[j] for j in ARM_JOINTS]


# ---------------------------------------------------------------- train.py record

def test_train_records_calibration_consistency():
    src = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    assert '"calibration_consistency": calib_consistency' in src
    assert "from dropbear_wbc.kinematics.serial_provenance import run_info_record" in src


def test_run_info_record(tree, tmp_path, monkeypatch):
    monkeypatch.delenv("DROPBEAR_CALIBRATION_JSON", raising=False)
    assert sp.run_info_record(ROOT)["status"] == "ok"
    rec = sp.run_info_record(tree)
    assert rec["status"] == "ok" and rec["findings"] == [] and rec["check_tool"] == sp.CHECK_TOOL
    other = tmp_path / "cal.json"
    other.write_text((tree / sp.CALIBRATION_REL).read_text(encoding="utf-8") + " ", encoding="utf-8")
    monkeypatch.setenv("DROPBEAR_CALIBRATION_JSON", str(other))
    rec = sp.run_info_record(tree)
    assert rec["status"] == "stale" and rec["calibration"] == str(other)
    assert rec["findings"][0].startswith("serial calibration_sha256")
    monkeypatch.setenv("DROPBEAR_CALIBRATION_JSON", "none")
    assert sp.run_info_record(tree)["status"] == "ok"
    (tree / sp.SERIAL_META_REL).write_text("{not json", encoding="utf-8")
    rec = sp.run_info_record(tree)
    assert rec["status"] == "error" and rec["findings"][0].startswith("JSONDecodeError")
