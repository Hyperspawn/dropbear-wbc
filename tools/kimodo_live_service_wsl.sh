#!/bin/bash
# WSL side of tools/live_session.sh: the warm Kimodo text -> Dropbear clip service (tools/kimodo_live_service.py).
# Needs the Windows text-embed server running (tools/kimodo_text_embed_server.py; TEXT_ENCODER_MODE=file) and the
# WSL venv .venv-kimodo-wsl (docs/INSTALL.md). Paths: $VAR, then .dropbear.env (Windows paths are converted with
# wslpath), then repo-relative defaults, as in source/dropbear_wbc/paths.py.
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PY=.venv-kimodo-wsl/bin/python
setting() { PYTHONPATH=source "$PY" -c "from dropbear_wbc import paths; print(paths.setting('$1', '') or '')"; }
tolinux() { case "$1" in [A-Za-z]:/*|[A-Za-z]:\\*) wslpath -a "$1" ;; *) echo "$1" ;; esac; }
HF=$(tolinux "$(setting DROPBEAR_HF_CACHE)"); HF=${HF:-${HF_HOME:-$HOME/.cache/huggingface}}
UP=$(tolinux "$(setting DROPBEAR_UPSTREAM)"); UP=${UP:-$REPO/../upstream}
TMP=$(tolinux "$(setting DROPBEAR_TMP)")
export G1_MJCF=${G1_MJCF:-$UP/unitree_mujoco/unitree_robots/g1/g1_29dof.xml}
export HF_HOME=$HF HF_HUB_CACHE=$HF/hub HF_HUB_OFFLINE=1 TEXT_ENCODER_MODE=file KIMODO_EMBED_BRIDGE=$HF/embed_bridge
[ -n "$TMP" ] && export TMPDIR=$TMP  # e.g. keep large temp files off a small system drive
export PYTHONPATH=source
export TORCHINDUCTOR_COMPILE_THREADS=1  # no idle compile-worker pool (~0.35 GB RSS each)
export DROPBEAR_CALIBRATION_JSON=data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json
exec nice -n 10 "$PY" -u tools/kimodo_live_service.py "$@"  # below the physics
