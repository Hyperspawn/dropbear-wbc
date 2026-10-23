"""Every repo ``data/`` path the training code reads must reach the Brev instance.

The instance gets the bundle (tools/pack_brev_bundle.py), not a git clone. 0017 read
``data/robot/usd_body_properties_*.json`` before ``data/robot`` was in DATA_DIRS and lost a 4x L40S start. This test
scans ``source/`` and ``scripts/`` for repo-rooted ``data/...`` references (string literals and ``"data" / "x"`` path
joins) and requires each one to be shipped by default (DATA_DIRS / DATA_FILES) or listed in EXEMPT with the reason it
need not be. A new data reference fails here until it is classified.
"""

from __future__ import annotations

import importlib.util
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# prefix -> why the default bundle need not contain it
EXEMPT = {
    "data/motions_v6ts/libraries/": "AMP / tracking clips: shipped with --manifest (manifest + pinned NPZs)",
    "data/motions/libraries/": "tracking clips: shipped with --manifest",
    "data/motions/smoke/dropbear_static_stand_try1.npz": "archived attempt, not read by training",
    "data/teleop": "teleop recordings and arm inertia (local, not a Brev training job)",
    "data/motions_v6ts/synthetic/stand.npz": "scripts/live_viewer.py scene default (local viewer)",
    "data/groot/": "GR00T tabletop data collection (local)",
    "data/chunk-": "LeRobot dataset layout under a recording root, not the repo data/ dir",
    "data/LAFAN1_Retargeting_Dataset/": "outside the repo (REPO_ROOT.parent), retargeting only",
}

_LIT = re.compile(r"""["']((?:\.\./)*data/[A-Za-z0-9_./{}*-]+)""")
_JOIN = re.compile(r"""["']data["']((?:\s*/\s*(?:["'][A-Za-z0-9_.{}*-]+["']|f["'][A-Za-z0-9_.{}*\[\]:-]+["']))+)""")


def _tool():
    spec = importlib.util.spec_from_file_location("pbb_data", ROOT / "tools" / "pack_brev_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _refs() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for top in ("source", "scripts"):
        for py in sorted((ROOT / top).rglob("*.py")):
            text = py.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                found = [m.group(1).lstrip("./") for m in _LIT.finditer(line)]
                for m in _JOIN.finditer(line):
                    parts = re.findall(r"""f?["']([^"']+)["']""", m.group(1))
                    found.append("data/" + "/".join(parts))
                for ref in found:
                    ref = ref if ref.startswith("data/") else ref[ref.index("data/"):]
                    out.setdefault(ref, []).append(f"{py.relative_to(ROOT).as_posix()}:{i}")
    return out


def _shipped(ref: str, tool) -> bool:
    return any(ref == d or ref.startswith(d + "/") for d in tool.DATA_DIRS) or ref in tool.DATA_FILES


def test_every_training_data_reference_is_shipped_or_exempt():
    tool = _tool()
    refs = _refs()
    assert len(refs) >= 10, f"scanner found too little ({len(refs)}); regex broken?"
    missing = {r: w for r, w in refs.items() if not _shipped(r, tool) and not any(r.startswith(p) for p in EXEMPT)}
    assert not missing, "data/ paths read by source/ or scripts/ that the Brev bundle does not ship:\n" + "\n".join(
        f"  {r}  ({', '.join(w[:3])})" for r, w in sorted(missing.items()))


def test_default_stand_npz_is_shipped():
    from dropbear_wbc.tasks.locomotion.stand import DEFAULT_STAND_NPZ  # noqa: PLC0415

    tool = _tool()
    assert DEFAULT_STAND_NPZ.relative_to(ROOT).as_posix() in tool.DATA_FILES


def test_bundle_without_flags_has_stand_and_data_dirs(tmp_path):
    tool = _tool()
    out = tmp_path / "b.tgz"
    assert tool.main(["--out", str(out), "--no-code"]) == 0
    with tarfile.open(out) as tar:
        names = set(tar.getnames())
    for f in tool.DATA_FILES:
        assert f in names
    for d in tool.DATA_DIRS:
        assert any(n.startswith(d + "/") for n in names), d
