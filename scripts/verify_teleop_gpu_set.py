r"""GPU (MuJoCo Warp) teleop verification set, run once under ``tools/gpu_lock_run.py`` (the caller holds the lock).

Runs ``scripts/verify_teleop.py --backend gpu --gpu-lock-held`` twice, on dedicated ZMQ ports (5565/5566):
the contract plant (bridge default passive damping 50) and the diagnostic plant (passive damping 0.5). About 2-3 min
of lock time. Launch detached so it survives the session (PowerShell)::

    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine =
      'python.exe <repo>\tools\gpu_lock_run.py
       --owner teleop --log <repo>\logs\teleop\gpu_verify_set.wrapper.log --timeout 900
       --wait-minutes 600 -- <repo>\.venv-teleop\Scripts\python.exe -u
       <repo>\scripts\verify_teleop_gpu_set.py'; CurrentDirectory = '<repo>' }
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    day = f"{dt.datetime.now():%Y-%m-%d}"
    # v4: final code (measured elbow wrist-path table, restart gating) with the final default reference ("forward",
    # -0.7/0.15/0/0/0) and amplitude (2/6/5 cm). Earlier sets kept: gpu_fig8_* (11:00, "forward_extended"
    # reference, pivot elbow) and gpu_fig8v3_* (11:19, final reference, pivot elbow).
    runs = [("gpu_fig8v4_contract", []), ("gpu_fig8v4_pd0.5", ["--bridge-extra=--passive-damping 0.5"])]
    rcs = []
    for tag, extra in runs:
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / "verify_teleop.py"), "--backend", "gpu", "--gpu-lock-held",
               "--tag", tag, "--duration", "25", "--session", f"{day}_{tag}", "--ports", "5565", "5566"] + extra
        print("$ " + " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        print(f"[gpu_set] {tag} rc={rc}", flush=True)
        rcs.append(rc)
    return 0 if all(r == 0 for r in rcs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
