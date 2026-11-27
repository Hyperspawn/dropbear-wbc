"""Every hardware-armed Brev launcher trains / evaluates on a hardware-curated motion library (item 399).

Since 0092 scripts/train.py refuses a fresh ``hw_*`` / ``legacy_hwarm`` library run on a manifest without a valid
``curation`` block (dropbear_wbc.tasks.tracking.library_curation.check_hw_library). The tools/brev launchers of
sessions 3-10 still named the uncurated accepted_v7gen.json / lowbody_v1.json, so re-running any of them died at
env build. They now name accepted_v7gen_hw.json (0119, 100/138 at the 45 C design ambient) and lowbody_v1_hw.json
(0093). Launchers whose profiles are all sim-only (run_all.sh, preflight.sh: stiff_knee_hw) are not checked.

``--motion_library`` arguments are resolved with pack_brev_bundle's static shell-variable expansion (``V7=...``
then ``--motion_library $V7``); a reference that does not resolve to a literal path fails the test instead of being
skipped.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
from pathlib import Path

import pytest

from dropbear_wbc.tasks.tracking.library_curation import UncuratedLibraryError, check_hw_library, is_hw_profile

ROOT = Path(__file__).resolve().parents[1]
BREV = ROOT / "tools" / "brev"
HW_TOKEN = re.compile(r"(?<![\w$])(hw_[a-z0-9_]+|legacy_hwarm)\b")
LIB_ARG = re.compile(r"--motion_library[ =]+(\S+)")
# launchers with an hw profile and a library job (sessions 3-10); a new one is picked up by the glob, this set only
# guards against the parser silently finding nothing
EXPECTED_HW_LIBRARY_LAUNCHERS = {
    "eval_session3.sh", "run_session3.sh", "run_session4.sh", "run_session5b.sh", "run_session6.sh",
    "run_session7.sh", "run_session8.sh", "run_session9.sh", "run_session10.sh", "swap_session6_height.sh",
}


def _pbb():
    spec = importlib.util.spec_from_file_location("pbb_hw_launcher", ROOT / "tools" / "pack_brev_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _body(text: str) -> str:
    """Non-comment text with line continuations joined."""
    lines = [ln for ln in text.replace("\\\n", " ").splitlines() if not ln.lstrip().startswith("#")]
    return "\n".join(ln.split(" #", 1)[0] for ln in lines)


def launcher_libraries(path: Path, pbb) -> tuple[set[str], dict[str, set[str]]]:
    """(hw profiles named by the launcher, {raw --motion_library word: resolved repo paths})."""
    text = path.read_text(encoding="utf-8")
    body = _body(text)
    env = pbb.shell_assignments(text)
    profiles = {m.group(1) for m in HW_TOKEN.finditer(body) if is_hw_profile(m.group(1))}
    libs: dict[str, set[str]] = {}
    for m in LIB_ARG.finditer(body):
        word = m.group(1).strip("\"'\\")
        libs[word] = {w.strip("\"'\\") for w in pbb.expand_word(word, env)}
    return profiles, libs


def _hw_library_launchers(pbb):
    out = {}
    for p in sorted(BREV.glob("*.sh")):
        profiles, libs = launcher_libraries(p, pbb)
        if profiles and libs:
            out[p.name] = (profiles, libs)
    return out


@pytest.fixture(scope="module")
def pbb():
    return _pbb()


def test_expected_launchers_found(pbb):
    assert set(_hw_library_launchers(pbb)) == EXPECTED_HW_LIBRARY_LAUNCHERS


@pytest.mark.parametrize("name", sorted(EXPECTED_HW_LIBRARY_LAUNCHERS))
def test_hw_launcher_library_passes_train_gate(name, pbb):
    profiles, libs = launcher_libraries(BREV / name, pbb)
    assert profiles and libs, name
    for word, paths in libs.items():
        assert paths and all("$" not in p for p in paths), f"{name}: --motion_library {word} does not resolve"
        for rel in sorted(paths):
            assert (ROOT / rel).is_file(), f"{name}: {rel} missing"
            for prof in sorted(profiles):
                # strict: the fresh-run gate with the 45 C design-ambient requirement, no --allow_uncurated
                rec, _ = check_hw_library(ROOT / rel, prof, require_curation_ambient=True)
                assert rec["checked"] and rec["curated"], (name, rel, prof, rec["problems"])
                assert rec["ambient_status"]["status"] == "ok", (name, rel, prof, rec["ambient_status"])


def test_old_uncurated_manifests_are_refused():
    """The manifests the launchers used to name really are refused (the test above would catch a revert)."""
    for rel in ("data/motions_v6ts/libraries/accepted_v7gen.json", "data/motions_v6ts/libraries/lowbody_v1.json"):
        with pytest.raises(UncuratedLibraryError):
            check_hw_library(ROOT / rel, "hw_v4")


def test_sim_only_launchers_not_hw(pbb):
    """run_all.sh / preflight.sh run stiff_knee_hw (sim-only); they keep their uncurated library."""
    for name in ("run_all.sh", "preflight.sh"):
        profiles, libs = launcher_libraries(BREV / name, pbb)
        assert libs and not profiles, (name, profiles)
        assert not is_hw_profile("stiff_knee_hw")


def test_eval_session3_one_env_per_clip(pbb):
    """eval_session3's --continuous_loop library rollout used --num_envs 138 (one env per accepted_v7gen clip)."""
    body = _body((BREV / "eval_session3.sh").read_text(encoding="utf-8"))
    _, libs = launcher_libraries(BREV / "eval_session3.sh", pbb)
    (rel,) = {p for v in libs.values() for p in v}
    n_clips = len(json.loads((ROOT / rel).read_text(encoding="utf-8"))["clips"])
    cmd = next(ln for ln in body.splitlines() if "--continuous_loop" in ln)
    toks = shlex.split(cmd.replace('"$ck"', "CK"))
    assert int(toks[toks.index("--num_envs") + 1]) == n_clips == 100
