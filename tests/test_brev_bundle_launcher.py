"""pack_brev_bundle --launcher: a job script that the bundle cannot run must fail at pack time, not on the GPU box.

Session 11 (tools/brev/run_session11.sh) turns on --amp_weight; the AMP discriminator loads
tasks/locomotion/amp.py DEFAULT_LIBRARY at env build, so a bundle without that manifest dies on the instance after
setup. The default-argument data paths of the chunked drivers were outside 0025's source/ + scripts/ scan.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("pbb_launcher", ROOT / "tools" / "pack_brev_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _default_shipped(tool) -> set[str]:
    out = set(tool.DATA_FILES)
    for d in tool.DATA_DIRS:
        out |= {p.relative_to(ROOT).as_posix() for p in (ROOT / d).rglob("*") if p.is_file()}
    return out


def test_amp_default_library_matches_amp_module_source():
    tool = _tool()
    tree = ast.parse((ROOT / "source/dropbear_wbc/tasks/locomotion/amp.py").read_text(encoding="utf-8"))
    consts = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
              and any(getattr(t, "id", None) == "DEFAULT_LIBRARY" for t in n.targets)]
    assert consts == [tool.amp_default_library()]


def test_session11_needs_the_amp_manifest():
    tool = _tool()
    text = (ROOT / "tools/brev/run_session11.sh").read_text(encoding="utf-8")
    lib = tool.amp_default_library()
    bad = tool.check_launcher(text, _default_shipped(tool), set(), lib)
    assert len(bad) == 1 and "--amp_weight" in bad[0] and lib in bad[0], bad
    assert tool.check_launcher(text, _default_shipped(tool) | {lib}, {lib}, lib) == []


@pytest.mark.parametrize("flag,on", [("--amp_weight=1.0", True), ("--amp_weight 2", True), ("--amp_weight=0", False),
                                     ("--amp_weight=0.0", False), ("--amp_weight=$W", True)])
def test_amp_flag_values(flag, on):
    tool = _tool()
    bad = tool.check_launcher(f"python -u tools/run_chunked_locomotion.py {flag} --symmetry\n", set(), set(), "data/x.json")
    assert bool(bad) is on, bad


def test_comments_do_not_count_as_references_or_flags():
    tool = _tool()
    text = "# pack with --manifest data/motions/libraries/a.json and --amp_weight=1.0\necho ok  # data/b.npz\n"
    assert tool.check_launcher(text, set(), set(), "data/x.json") == []


def test_unshipped_data_file_is_reported_and_dirs_or_vars_are_not():
    tool = _tool()
    text = ("M=data/motions\nCAL=data/calibration/snapshots/c_1234abcd.json\n"
            "python scripts/train.py --motion_file data/motions_cyclic/walk.npz --x $M/a.npz data/motions/${C}.npz\n"
            "ls data/m/a.np* data/m/b.json$SUF data/m/c.v{1,2}\n")
    bad = tool.check_launcher(text, {"data/calibration/snapshots/c_1234abcd.json"}, set(), "data/x.json")
    assert bad == ["data/motions_cyclic/walk.npz: named by the launcher, not in the bundle (add --manifest or --extra)"]


def test_driver_fstring_passthrough_is_not_amp_on():
    tool = _tool()
    text = (ROOT / "tools/run_chunked_locomotion.py").read_text(encoding="utf-8")
    assert tool.check_launcher(text, _default_shipped(tool), set(), tool.amp_default_library()) == []


@pytest.mark.parametrize("driver", ["tools/run_chunked_training.py", "tools/run_chunked_locomotion.py"])
def test_driver_default_data_args_are_shipped(driver):
    """argparse defaults naming a repo data/ path (0025 scanned only source/ and scripts/)."""
    tool = _tool()
    shipped = _default_shipped(tool)
    tree = ast.parse((ROOT / driver).read_text(encoding="utf-8"))
    defaults = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_argument":
            for kw in n.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str) \
                        and kw.value.value.startswith("data/"):
                    defaults.append(kw.value.value)
    assert all(d in shipped or (ROOT / d).is_dir() for d in defaults), (driver, defaults)


def test_pack_refuses_launcher_without_amp_manifest(tmp_path):
    tool = _tool()
    sh = ROOT / "tools/brev/run_session11.sh"
    with pytest.raises(SystemExit, match="AMP reference library"):
        tool.main(["--out", str(tmp_path / "b.tgz"), "--no-code", "--launcher", str(sh)])
    assert not (tmp_path / "b.tgz").exists()
    with pytest.raises(SystemExit, match="does not exist"):
        tool.main(["--out", str(tmp_path / "c.tgz"), "--no-code", "--launcher", str(tmp_path / "nope.sh")])


def test_pack_accepts_clean_launcher(tmp_path):
    tool = _tool()
    sh = tmp_path / "job.sh"
    sh.write_text("python -u tools/run_chunked_training.py --motion_file data/motions/smoke/dropbear_static_stand.npz\n")
    assert tool.main(["--out", str(tmp_path / "b.tgz"), "--no-code", "--launcher", str(sh)]) == 0
