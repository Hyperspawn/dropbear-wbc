#!/bin/bash
# WSL side of batch text -> G1 motion generation (tools/kimodo_batch_generate.py), same environment as
# tools/kimodo_live_service_wsl.sh. Needs the Windows text-embed server running (tools/kimodo_text_embed_server.py;
# TEXT_ENCODER_MODE=file) and the WSL venv .venv-kimodo-wsl (docs/INSTALL.md).
#   wsl.exe -d Ubuntu --cd <repo> -- bash tools/kimodo_batch_generate_wsl.sh prompts.json out_dir [--steps N --seed S]
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
[ -n "$TMP" ] && export TMPDIR=$TMP
export PYTHONPATH=source
export TORCHINDUCTOR_COMPILE_THREADS=1
exec nice -n 10 "$PY" -u tools/kimodo_batch_generate.py "$@"
