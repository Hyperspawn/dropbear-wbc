"""pack_brev_bundle --launcher: a job script that the bundle cannot run must fail at pack time, not on the GPU box.

Session 11 (tools/brev/run_session11.sh) turns on --amp_weight; the AMP discriminator loads
tasks/locomotion/amp.py DEFAULT_LIBRARY at env build, so a bundle without that manifest dies on the instance after
setup. The default-argument data paths of the chunked drivers were outside 0025's source/ + scripts/ scan.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
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


def _miss(ref):
    return f"{ref}: named by the launcher, not in the bundle (add --manifest or --extra)"


def test_unshipped_data_file_is_reported_and_dirs_or_unknown_vars_are_not():
    tool = _tool()
    text = ("M=data/motions\nCAL=data/calibration/snapshots/c_1234abcd.json\n"
            "python scripts/train.py --motion_file data/motions_cyclic/walk.npz --x $M/a.npz data/motions/${C}.npz\n"
            "ls data/m/a.np* data/m/b.json$SUF data/m/c.v{1,2} $D/telemetry.npz\n")
    bad = tool.check_launcher(text, {"data/calibration/snapshots/c_1234abcd.json"}, set(), "data/x.json")
    assert bad == [_miss("data/motions/a.npz"), _miss("data/motions_cyclic/walk.npz")]
    assert tool.check_launcher(text, {"data/calibration/snapshots/c_1234abcd.json", "data/motions/a.npz",
                                      "data/motions_cyclic/walk.npz"}, set(), "data/x.json") == []


@pytest.mark.parametrize("line", [
    "clip_job a $M/sub/a.npz 2500 11", "clip_job a ${M}/sub/a.npz 2500 11", "x --motion_file=$M/sub/a.npz",
    "declare -A NPZ=( [a]=$M/sub/a.npz )", "x \"$M/sub/a.npz\"", "x ${N:-data/motions}/sub/a.npz",
    "x $SUB/a.npz", "x $LIB2",
])
def test_composed_paths_resolve(line):
    tool = _tool()
    text = f"export M=data/motions\nSUB=$M/sub\nLIB2=\"$SUB/a.npz\"\n{line}\n"
    assert tool.composed_data_refs(text) == {"data/motions/sub/a.npz"}
    assert tool.check_launcher(text, set(), set(), "data/x.json") == [_miss("data/motions/sub/a.npz")]


def test_every_value_of_a_reassigned_name_is_checked_and_unresolvable_values_skipped():
    tool = _tool()
    text = ("M=data/a\nf() { local M=data/b; run $M/c.npz; }\nM=$(ls -d data/*)\nG=data/*\n"
            "LIB=\"--task T --motion_library data/l.json\"\nrun $G/x.npz $LIB # $M/comment.npz\n# $M/z.npz\n")
    assert tool.shell_assignments(text)["M"] == {"data/a", "data/b"}
    assert "G" not in tool.shell_assignments(text) and "LIB" not in tool.shell_assignments(text)
    assert tool.composed_data_refs(text) == {"data/a/c.npz", "data/b/c.npz"}


def test_real_launchers_composed_clip_refs():
    """run_all.sh / eval_watcher.sh name their single-clip jobs as $M/<set>/<clip>.npz (unchecked before 0094)."""
    tool = _tool()
    want = {"data/motions/unitree_rl_lab_mimic/G1_Take_102_v4.npz",
            "data/motions/unitree_rl_lab_mimic/G1_gangnam_style_V01_v4.npz",
            "data/motions/gmr_lafan1/lafan1_dance2_subject1_v4.npz", "data/motions/gmr_lafan1/lafan1_walk1_subject1_v4.npz",
            "data/motions/kimodo_g1/output_walk_v4.npz", "data/motions/synthetic/wave_right_v2.npz"}
    for name in ("run_all.sh", "eval_watcher.sh"):
        text = (ROOT / "tools/brev" / name).read_text(encoding="utf-8")
        assert tool.composed_data_refs(text) >= want, name
        assert set(tool.check_launcher(text, _default_shipped(tool), set(), "data/x.json")) >= {_miss(r) for r in want}


def test_expansion_is_bounded():
    tool = _tool()
    text = "".join(f"A=data/v{i}\n" for i in range(20)) + "".join(f"B=b{i}\n" for i in range(20)) + "x $A/$B/c.npz\n"
    assert 0 < len(tool.composed_data_refs(text)) <= tool._MAX_EXPANSIONS
    lim = sys.getrecursionlimit()
    sys.setrecursionlimit(250)  # a self-referential value must stop at the depth bound, not at the interpreter's
    try:
        assert tool.expand_word("$A/$A/$A/x", {"A": {"$A"}}) == set()
    finally:
        sys.setrecursionlimit(lim)
    chain = "".join(f"V{i}=$V{i - 1}/d{i}\n" for i in range(1, 6))
    assert tool.composed_data_refs("V0=data\n" + chain + "x $V5/c.npz\n") == {"data/d1/d2/d3/d4/d5/c.npz"}


def test_default_value_expansion():
    tool = _tool()
    assert tool.composed_data_refs("x ${N:-data/motions}/a.npz ${N-data/m2}/b.npz\n") == {
        "data/motions/a.npz", "data/m2/b.npz"}
    assert tool.composed_data_refs("N=data/set\nx ${N:-data/motions}/a.npz\n") == {"data/set/a.npz"}


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
