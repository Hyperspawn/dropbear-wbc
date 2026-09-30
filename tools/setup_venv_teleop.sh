#!/usr/bin/env bash
# Creates .venv-teleop for XR / keyboard arm teleoperation (docs/TELEOP.md): vuer 0.0.60 as pinned by televuer 4.0.0.
# usage: bash tools/setup_venv_teleop.sh [python3.12 executable]   (pip cache kept inside the repo: .cache/pip)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:-python3.12}
export PIP_CACHE_DIR=$PWD/.cache/pip
"$PY" -m venv .venv-teleop
VPY=$( [ -x .venv-teleop/Scripts/python.exe ] && echo .venv-teleop/Scripts/python.exe || echo .venv-teleop/bin/python )
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install "vuer[all]==0.0.60" "numpy<2.0.0" pyzmq msgpack pyarrow pytest scipy "params-proto<3"  # vuer 0.0.60 breaks with params-proto 3.x (ImportError: Proto)
"$VPY" -m pip freeze
