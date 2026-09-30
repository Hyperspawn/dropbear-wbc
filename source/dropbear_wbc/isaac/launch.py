"""sys.path preparation for scripts run with Isaac Sim's python (``python.bat`` / ``python.sh``, kit python 3.11).

Kit python ships ``rsl-rl-lib 5.0.1`` in its site-packages, which is API-incompatible with Isaac Lab 2.2.
We vendor ``rsl-rl-lib 2.3.3`` in ``third_party/pydeps`` and keep it first on ``sys.path``. We also prefer
the Warp bundled with Isaac Sim (``extscache/omni.warp.core-*``) over the separately installed
``warp-lang`` wheel, which reports a CUDA driver entry-point error on this machine (observed by the
``dropbear_control`` evaluation tool).
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys
from pathlib import Path

from .. import paths as _paths

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
PYDEPS: Path = REPO_ROOT / "third_party" / "pydeps"
SOURCE: Path = REPO_ROOT / "source"
ISAAC_SIM_ROOT: Path = _paths.isaac_sim_root()  # $ISAAC_SIM_ROOT / .dropbear.env (dropbear_wbc.paths)


def _prepend(path: Path) -> None:
    s = str(path)
    while s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


def prepare_kit_python(prefer_bundled_warp: bool = True) -> None:
    """Put vendored deps, our source tree and (optionally) the bundled Warp first on ``sys.path``."""
    if prefer_bundled_warp:
        candidates = sorted(glob.glob(str(ISAAC_SIM_ROOT / "extscache" / "omni.warp.core-*")))
        if candidates and "warp" not in sys.modules:
            _prepend(Path(candidates[-1]))
    _prepend(SOURCE)
    _prepend(PYDEPS)


def assert_vendored_rsl_rl() -> str:
    """Re-assert ``third_party/pydeps`` precedence (Kit may reorder sys.path) and check rsl_rl resolves there.

    Returns:
        The imported ``rsl_rl`` package directory.

    Raises:
        RuntimeError: if ``rsl_rl`` is imported from anywhere else (e.g. kit's 5.0.1).
    """
    _prepend(SOURCE)
    _prepend(PYDEPS)
    import importlib.metadata as md

    import rsl_rl

    location = Path(rsl_rl.__file__).resolve().parent
    if PYDEPS.resolve() not in location.parents:
        raise RuntimeError(f"rsl_rl imported from {location}, expected vendored copy under {PYDEPS}")
    version = md.version("rsl-rl-lib")
    if version != "2.3.3":
        raise RuntimeError(f"rsl-rl-lib metadata version {version} != 2.3.3")
    return str(location)


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    """SHA-256 hex digest of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def close_app_and_exit(app, rc: int, timeout_s: float = 45.0) -> None:
    """Close the Kit app, but never hang: force-exit the process after ``timeout_s``.

    On this machine ``SimulationApp.close()`` was observed to hang indefinitely after a headless run had
    finished (``logs/robot_task/inspect_articulation_v1_usdfriction_orphans.log``), which would keep the
    GPU lock held. All outputs must be flushed/written before calling this.
    """
    import os
    import threading

    sys.stdout.flush()
    sys.stderr.flush()
    watchdog = threading.Timer(timeout_s, lambda: os._exit(rc))
    watchdog.daemon = True
    watchdog.start()
    try:
        app.close(wait_for_replicator=False)
    except TypeError:
        app.close()
    except Exception:  # noqa: BLE001
        pass
    sys.stdout.flush()
    os._exit(rc)


def prefer_performance_cores() -> dict:
    """Windows: opt THIS process out of EcoQoS execution-speed throttling and raise it to ABOVE_NORMAL priority.

    On the i9-13980HX laptop (8 P + 16 E cores) a headless physics process ran 2.3x slower while a GUI window
    (``scripts/live_viewer.py``) had focus: Windows 11 treats the unfocused console process as background work and
    schedules it on the efficiency cores. Both hints are per-process and end with the process; no system setting is
    changed. Returns what was applied (all False off Windows).
    """
    if sys.platform != "win32":
        return {"eco_qos_off": False, "above_normal": False}
    import ctypes
    from ctypes import wintypes

    class _PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    k32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    h = k32.GetCurrentProcess()
    # ProcessPowerThrottling = 4; version 1; control EXECUTION_SPEED (0x1) with state 0 = never throttle
    st = _PowerThrottlingState(1, 0x1, 0)
    eco = bool(k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st)))
    prio = bool(k32.SetPriorityClass(h, 0x8000))  # ABOVE_NORMAL_PRIORITY_CLASS
    return {"eco_qos_off": eco, "above_normal": prio}
