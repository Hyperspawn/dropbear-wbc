"""Every machine-specific location in one place (Isaac-free, stdlib only).

Resolution order for each setting: the environment variable, then ``<repo>/.dropbear.env`` (optional, untracked,
``KEY=VALUE`` lines; copy ``dropbear.env.example``), then a default relative to the repository. So a fresh clone works
once ``python tools/fetch_assets.py`` has put the robot USD in ``assets/``, and a machine with a different layout only
edits ``.dropbear.env``.

=====================  =========================================================================================
``DROPBEAR_USD``        plant USD (default ``assets/dropbear.usd``; must match ``CONTRACTS.md`` SHA-256)
``DROPBEAR_ASSETS``     downloaded assets (USD, policies, media; default ``<repo>/assets``)
``ISAAC_SIM_ROOT``      Isaac Sim install (default ``C:/isaac-sim`` on Windows, ``~/isaacsim`` on Linux)
``ISAACSIM_PYTHON``     Isaac Sim's python launcher (default ``<ISAAC_SIM_ROOT>/python.bat|python.sh``)
``DROPBEAR_UPSTREAM``   external clones: GMR, unitree_mujoco, lafan1, GR00T-WholeBodyControl (default ``../upstream``)
``DROPBEAR_CONTROL_DIR`` clone of github.com/robit-man/dropbear_control (Newton plant adapter; ``../dropbear_control``)
``DROPBEAR_HF_CACHE``   Hugging Face cache for Kimodo / LLM2Vec (default ``$HF_HOME`` or ``~/.cache/huggingface``)
``DROPBEAR_FFMPEG``     ffmpeg executable (default: on PATH, else the one bundled with Isaac Sim)
=====================  =========================================================================================
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
ENV_FILE: Path = REPO_ROOT / ".dropbear.env"
USD_SHA256 = "45586414b065cd982d487cbd868fe982108b3b8ccec64d3dfcf629652ed8db0f"
"""SHA-256 of the plant USD this repository is calibrated and trained against (``docs/CONTRACTS.md``)."""


def _env_file() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


_FILE = _env_file()


def setting(name: str, default: str | None = None) -> str | None:
    """``$name``, else ``.dropbear.env``'s value, else ``default``."""
    return os.environ.get(name) or _FILE.get(name) or default


def _path(name: str, default: Path) -> Path:
    return Path(setting(name) or default).expanduser()


def assets_dir() -> Path:
    return _path("DROPBEAR_ASSETS", REPO_ROOT / "assets")


def usd_path() -> Path:
    """The plant USD: ``$DROPBEAR_USD`` / ``.dropbear.env``, else ``assets/dropbear.usd``."""
    return _path("DROPBEAR_USD", assets_dir() / "dropbear.usd")


def isaac_sim_root() -> Path:
    return _path("ISAAC_SIM_ROOT", Path("C:/isaac-sim") if sys.platform == "win32" else Path.home() / "isaacsim")


def isaac_python() -> str:
    """Isaac Sim's python launcher (``python.bat`` / ``python.sh``) as a string for subprocess command lines."""
    exe = setting("ISAACSIM_PYTHON")
    if exe:
        return exe
    return str(isaac_sim_root() / ("python.bat" if sys.platform == "win32" else "python.sh")).replace("\\", "/")


def upstream_dir() -> Path:
    return _path("DROPBEAR_UPSTREAM", REPO_ROOT.parent / "upstream")


def dropbear_control_dir() -> Path:
    return _path("DROPBEAR_CONTROL_DIR", REPO_ROOT.parent / "dropbear_control")


def hf_cache() -> Path:
    return _path("DROPBEAR_HF_CACHE", Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"))


def ffmpeg() -> str:
    """An ffmpeg executable: ``$DROPBEAR_FFMPEG``, else PATH, else the imageio-ffmpeg binary bundled with Isaac Sim."""
    exe = setting("DROPBEAR_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    pattern = isaac_sim_root() / "kit" / "python" / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries" / "ffmpeg*"
    hits = sorted(glob.glob(str(pattern)))
    return hits[-1] if hits else "ffmpeg"


def describe() -> dict[str, str]:
    """All resolved settings (``python -m dropbear_wbc.paths`` prints them)."""
    return {"repo": str(REPO_ROOT), "env_file": str(ENV_FILE) + ("" if ENV_FILE.is_file() else " (absent)"),
            "DROPBEAR_USD": str(usd_path()), "DROPBEAR_ASSETS": str(assets_dir()),
            "ISAAC_SIM_ROOT": str(isaac_sim_root()), "ISAACSIM_PYTHON": isaac_python(),
            "DROPBEAR_UPSTREAM": str(upstream_dir()), "DROPBEAR_CONTROL_DIR": str(dropbear_control_dir()),
            "DROPBEAR_HF_CACHE": str(hf_cache()), "DROPBEAR_FFMPEG": ffmpeg()}


if __name__ == "__main__":
    for k, v in describe().items():
        exists = Path(v).exists() if k not in ("repo", "env_file") else True
        print(f"{k:22s} {v}{'' if exists else '   (missing)'}")
