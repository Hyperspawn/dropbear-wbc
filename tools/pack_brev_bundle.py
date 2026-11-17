"""Pack what a cloud (Brev) training box needs into one .tgz with repo-relative paths (docs/BREV.md section 4).

Stdlib only (any Python 3.10+, Windows or WSL). Contents:

* code: ``source/ scripts/ tools/ third_party/pydeps/ docs/ requirements/`` (no ``__pycache__``/``*.pyc``);
* ``data/calibration/`` (the default calibration and its pinned snapshots) and ``data/robot/`` (USD body
  properties for ``--robot_mass_kg``, serial surrogates);
* for every ``--manifest`` (``dropbear-motion-library-v1``): the manifest, every clip NPZ it lists and the clip's
  ``<clip>.validation.json`` verdict (the library refuses clips without an accepted verdict, CONTRACTS 5.3);
* ``DATA_FILES`` (always): the locomotion/tabletop reset pose ``data/motions/smoke/dropbear_static_stand.npz``;
* ``--extra`` files/dirs;
* ``--launcher`` (optional, the job script or driver the instance will run, e.g. ``tools/brev/run_session11.sh``):
  checked, not packed separately. Every repo ``data/<file>`` it names (literally or as ``$VAR/<file>`` with a
  literal ``VAR=data/...`` assignment) must be in the bundle, and a job with a
  non-zero ``--amp_weight`` needs the AMP reference library (``tasks/locomotion/amp.py`` ``DEFAULT_LIBRARY``) as a
  ``--manifest``: otherwise the pack fails here instead of the GPU job failing on the instance;
* ``BUNDLE.json``: file list with SHA-256, the manifests' clip pins, the source host and time.

The plant USD (421 MB, read-only on P:) is NOT packed: copy it separately and point ``$DROPBEAR_USD`` at it.

    python tools/pack_brev_bundle.py --manifest data/motions/libraries/accepted_v1.json --out ../dropbear-wbc-brev.tgz
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import os
import platform
import re
import shlex
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE_DIRS = ("source", "scripts", "tools", "third_party/pydeps", "docs", "requirements")
DATA_DIRS = ("data/calibration", "data/robot")  # data/robot: USD body masses (--robot_mass_kg), serial surrogates
# Single files every training job reads whatever its flags: the locomotion/tabletop reset pose
# (DropbearFlatEnvCfg.__post_init__ -> set_stand_npz(DEFAULT_STAND_NPZ), tabletop mdp.DEFAULT_STAND_NPZ).
DATA_FILES = ("data/motions/smoke/dropbear_static_stand.npz", "data/motions/smoke/dropbear_static_stand.validation.json")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git"}
AMP_PY = REPO / "source" / "dropbear_wbc" / "tasks" / "locomotion" / "amp.py"
# repo data/ file tokens in a launcher (needs a file suffix; directories such as M=data/motions are composed later)
_DATA_TOKEN = re.compile(r"(?<![\w./$-])(data/[A-Za-z0-9_.\-/]+\.[A-Za-z0-9]+)(?![\w$*{])")
_AMP_FLAG = re.compile(r"--amp_weight(?:=|\s+)([^\s\\\"';|&)]+)")
# simple shell assignments NAME=value (optionally after export/readonly/local/declare); value must be one literal word
_ASSIGN_PREFIX = {"export", "readonly", "local", "declare", "typeset"}
_CMD_SEP = {"{", "}", "then", "do", "else", "elif", "&&", "||", "|", "!", ";", "(", ")"}
_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
# a word that expands one or more variables ($NAME, ${NAME}, ${NAME:-default}, ${NAME-default})
_VAR_WORD = re.compile(r"[^\s\"'=()\[\]`;|&<>]*\$(?:\{[A-Za-z_][A-Za-z0-9_]*(?::?-[^}]*)?\}|[A-Za-z_][A-Za-z0-9_]*)"
                       r"[^\s\"'()`;|&<>]*")
_VAR_REF = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}|([A-Za-z_][A-Za-z0-9_]*))")
_MAX_EXPANSIONS = 64


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError as exc:
        raise SystemExit(f"{path} is outside the repo {REPO}; the bundle keeps repo-relative paths") from exc


def _walk(root: Path):
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_PARTS)
        for fn in sorted(filenames):
            if not fn.endswith((".pyc", ".pyo")):
                yield Path(dirpath) / fn


def manifest_files(manifest: Path) -> tuple[list[Path], dict]:
    """The manifest, its clip NPZs and their verdicts (fail closed on missing files)."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != "dropbear-motion-library-v1":
        raise SystemExit(f"{manifest}: not a dropbear-motion-library-v1 manifest")
    files, pins = [manifest], {}
    for c in data.get("clips", []):
        npz = Path(c["npz"] if isinstance(c, dict) else c)
        npz = (npz if npz.is_absolute() else manifest.parent / npz).resolve()
        verdict = npz.with_name(npz.stem + ".validation.json")
        missing = [str(p) for p in (npz, verdict) if not p.is_file()]
        if missing:
            raise SystemExit(f"{manifest.name}: missing {missing}")
        files += [npz, verdict]
        pins[_rel(npz)] = c.get("sha256") if isinstance(c, dict) else None
    return files, pins


def amp_default_library() -> str:
    """``DEFAULT_LIBRARY`` of tasks/locomotion/amp.py, read from source (this tool stays stdlib-only, no torch)."""
    m = re.search(r'^DEFAULT_LIBRARY\s*=\s*["\']([^"\']+)["\']', AMP_PY.read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit(f"{AMP_PY}: DEFAULT_LIBRARY not found; update pack_brev_bundle.amp_default_library")
    return m.group(1)


def _strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        try:
            toks = shlex.split(line, comments=True, posix=True)
        except ValueError:  # unbalanced quotes (heredoc / multi-line string): keep the line as is
            out.append(line)
            continue
        out.append(" ".join(toks) if not line.lstrip().startswith("#") else "")
    return "\n".join(out)


def shell_assignments(text: str) -> dict[str, set[str]]:
    """Literal one-word values of ``NAME=value`` assignments (every value a name takes anywhere in the script).

    Values holding command substitution, globbing or whitespace are skipped (not resolvable statically); a name
    assigned in several places keeps every value, so each composed path is checked under each of them.
    """
    out: dict[str, set[str]] = {}
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        try:
            toks = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        start = True  # assignments only lead a simple command (A=x B=y cmd ...); `;`, `{`, `then`, `do` start one
        for t in toks:
            if t in _CMD_SEP:
                start = True
                continue
            if not start:
                if t.endswith(";"):
                    start = True
                continue
            if t in _ASSIGN_PREFIX or (t.startswith("-") and t != "-"):  # export / local -r / declare -x flags
                continue
            end = t.endswith(";")
            m = _ASSIGN.match(t.rstrip(";"))
            if m:
                name, val = m.groups()
                if val and not any(c in val for c in " \t`*?()") and "$(" not in val:
                    out.setdefault(name, set()).add(val)
            start = end or bool(m)
    return out


def expand_word(word: str, env: dict[str, set[str]], depth: int = 0) -> set[str]:
    """Every expansion of ``word`` under ``env``; a word with an unknown variable expands to nothing."""
    m = _VAR_REF.search(word)
    if not m:
        return {word}
    if depth > 8:
        return set()
    name, default = m.group(1) or m.group(3), m.group(2)
    vals = env.get(name) or ({default} if default is not None else set())
    res: set[str] = set()
    for v in sorted(vals):
        for tail in expand_word(word[m.end():], env, depth + 1):
            for head in expand_word(v, env, depth + 1):  # a value may itself compose another variable
                res.add(word[:m.start()] + head + tail)
                if len(res) >= _MAX_EXPANSIONS:
                    return res
    return res


def composed_data_refs(text: str) -> set[str]:
    """Repo ``data/<file>`` paths built from simple variables (``M=data/motions`` ... ``$M/clip.npz``)."""
    env = shell_assignments(text)
    refs: set[str] = set()
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for word in _VAR_WORD.findall(line.split(" #", 1)[0]):
            for w in expand_word(word, env):
                refs.update(r for r in _DATA_TOKEN.findall(w) if w.startswith(r) or w.startswith("./" + r))
    return refs


def check_launcher(text: str, shipped: set[str], manifests: set[str], amp_library: str) -> list[str]:
    """Problems that would make the jobs in ``text`` fail on the instance with this bundle (empty = OK).

    ``shipped``: repo-relative files in the bundle; ``manifests``: repo-relative ``--manifest`` paths.
    Comments are ignored, so a usage note naming a manifest does not count as a reference. Paths composed from
    simple literal variables (``M=data/motions`` then ``$M/clip.npz``) are resolved; a word with a variable that is
    never assigned a literal value (``$D/x.npz``, ``${C}``, command substitution) stays unchecked.
    """
    body = _strip_comments(text)
    problems = []
    for ref in sorted(set(_DATA_TOKEN.findall(body)) | composed_data_refs(text)):
        if ref not in shipped:
            problems.append(f"{ref}: named by the launcher, not in the bundle (add --manifest or --extra)")
    amp_on = []
    for v in _AMP_FLAG.findall(body):
        if v.startswith("{"):  # a driver's f-string pass-through (run_chunked_locomotion): only on if its caller sets it
            continue
        try:
            on = float(v) != 0.0
        except ValueError:
            on = True  # $W or similar: assume on, the bundle must carry the library
        if on:
            amp_on.append(v)
    if amp_on and amp_library not in manifests:
        problems.append(f"--amp_weight {'/'.join(sorted(set(amp_on)))}: the AMP reference library {amp_library} "
                        f"(tasks/locomotion/amp.py DEFAULT_LIBRARY) is not a --manifest of this bundle")
    return problems


def build_bundle(out: Path, manifests: list[Path], extra: list[Path], include_code: bool = True,
                 launchers: list[Path] = ()) -> dict:
    roots = ([REPO / d for d in CODE_DIRS] if include_code else []) + [REPO / d for d in DATA_DIRS]
    files: dict[str, Path] = {}
    for r in roots:
        if r.exists():
            for p in _walk(r):
                files[_rel(p)] = p
    for f in DATA_FILES:
        p = REPO / f
        if not p.is_file():
            raise SystemExit(f"{f}: required by every training job and missing")
        files[_rel(p)] = p
    info_manifests = {}
    for m in manifests:
        mf, pins = manifest_files(m.resolve())
        for p in mf:
            files[_rel(p)] = p
        info_manifests[_rel(m)] = {"clips": pins}
    for e in extra:
        if not e.exists():
            raise SystemExit(f"--extra {e} does not exist")
        for p in _walk(e.resolve()):
            files[_rel(p)] = p
    if launchers:
        shipped, amp_lib = set(files), amp_default_library()
        for ln in launchers:
            if not ln.is_file():
                raise SystemExit(f"--launcher {ln} does not exist")
            bad = check_launcher(ln.read_text(encoding="utf-8", errors="replace"), shipped, set(info_manifests), amp_lib)
            if bad:
                raise SystemExit(f"--launcher {ln.name}: the bundle would not run it:\n  " + "\n  ".join(bad))
    listing = {k: {"sha256": _sha256(p), "bytes": p.stat().st_size} for k, p in sorted(files.items())}
    for m in info_manifests.values():  # a pinned clip must be packed with the pinned bytes
        for rel, pin in m["clips"].items():
            if pin and listing[rel]["sha256"] != pin:
                raise SystemExit(f"{rel}: sha256 differs from its manifest pin; regenerate the manifest")
    info = {"schema": "dropbear-brev-bundle-v1", "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "host": platform.node(), "repo": str(REPO), "manifests": info_manifests,
            "usd_note": "plant USD not included: copy it and set $DROPBEAR_USD (sha256 must be 45586414...)",
            "files": listing}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for rel, p in sorted(files.items()):
            tar.add(str(p), arcname=rel, recursive=False)
        blob = (json.dumps(info, indent=1) + "\n").encode()
        ti = tarfile.TarInfo("BUNDLE.json")
        ti.size = len(blob)
        tar.addfile(ti, io.BytesIO(blob))
    return {"out": str(out), "sha256": _sha256(out), "bytes": out.stat().st_size, "num_files": len(listing),
            "manifests": list(info_manifests)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, action="append", default=[], help="library manifest(s) to ship")
    ap.add_argument("--extra", type=Path, action="append", default=[], help="additional repo files/dirs")
    ap.add_argument("--launcher", type=Path, action="append", default=[],
                    help="job script/driver the instance will run (tools/brev/run_session*.sh): fail if it names a "
                         "data/ file the bundle lacks or turns on --amp_weight without the AMP library manifest")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-code", action="store_true", help="data only (tests)")
    a = ap.parse_args(argv)
    rep = build_bundle(a.out.resolve(), a.manifest, a.extra, include_code=not a.no_code,
                       launchers=a.launcher)
    print(json.dumps(rep, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
