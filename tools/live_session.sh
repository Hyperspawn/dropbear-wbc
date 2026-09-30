#!/bin/bash
# Live text -> Dropbear motion with motor-twin physics, in real time, on the laptop (docs/TEXT_TO_MOTION.md).
#
#   text encoder (Windows CPU) -> Kimodo-G1 + retarget (WSL GPU) -> logs/live/inbox/*.npz
#   -> scripts/play.py --live_dir (Isaac PhysX on the CPU pipeline, datasheet motors, 27 closures, --realtime, headless)
#   -> logs/live/state.bin -> tools/live_web.py (browser page: robot + prompt box, http://127.0.0.1:8765)
#
# Memory (31.6 GB laptop, pagefile on a nearly full C:, committed memory is the limit): text encoder int8 memory-mapped
# ~1.2 GB (bf16: 16.4), physics ~4.8 GB, Kimodo WSL VM ~8 GB, web viewer ~0.1 GB + a browser tab. The Newton window
# (VIEWER=gl, ~5 GB) or a second Isaac process (scripts/live_viewer.py, ~13 GB) are heavier alternatives.
#
# Type prompts in the page, or in a terminal:  python tools/live_prompt.py
#
# usage (Git Bash, repo root):  bash tools/live_session.sh [minutes=10]
#   env: CKPT=<policy .pt>  PROFILE=<actuator profile>  FIRST=<first clip npz>
cd "$(dirname "$0")/.."
MIN=${1:-10}
WSL_DISTRO=${WSL_DISTRO:-Ubuntu}
# machine paths come from source/dropbear_wbc/paths.py ($VAR, then .dropbear.env, then repo-relative defaults)
pathcfg() { PYTHONPATH=source python -c "from dropbear_wbc import paths as p; print($1)"; }
ISAACPY=$(pathcfg "p.isaac_python()")
HFC=$(pathcfg "p.hf_cache().as_posix()")
# default tracker: lib_v7gen (library + 60 Kimodo everyday motions, hw_v1ie) at its newest pulled checkpoint; on the
# five clips of the 17:01 session it was 0 falls / HW gate PASS with fewer warnings than lib_v6ts_v1i_fix (no elbow
# lag, 0 % 200 Hz jumps; logs/hw_twin/live_checks). Fallback: lib_v6ts_v1i_fix/10492 on hw_v1i.
# published policies (python tools/fetch_assets.py) live in assets/policies/; your own Brev pulls in logs/brev/remote
V7DIR=$(ls -d assets/policies/lib_v7gen_v1ie_smooth logs/brev/remote/dbw3/logs/rsl_rl/dropbear_tracking_library/*_lib_v7gen_v1ie_smooth 2>/dev/null | head -1)
V7=$([ -n "$V7DIR" ] && ls "$V7DIR"/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt$/ &/' | sort -n | tail -1 | cut -d' ' -f2)
if [ -z "$CKPT" ] && [ -n "$V7" ]; then CKPT=$V7; PROFILE=${PROFILE:-hw_v1ie}; fi
CKPT=${CKPT:-$(ls assets/policies/lib_v6ts_nostate_v1i_fix/model_10492.pt logs/brev/remote/dbw2/logs/rsl_rl/dropbear_tracking_library/2026-09-25_21-25-05_lib_v6ts_nostate_v1i_fix/model_10492.pt 2>/dev/null | head -1)}
PROFILE=${PROFILE:-hw_v1i}
FIRST=${FIRST:-data/motions_v6ts/synthetic/stand.npz}
export DROPBEAR_CALIBRATION_JSON=data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json
mkdir -p logs/live/inbox logs/live/prompts
rm -f logs/live/prompts/*.txt logs/live/state.bin logs/live/state.bin.json logs/live/physics.log logs/live/viewer.log

alive_age() { python -c "import json,time;print(int(time.time()-json.load(open('$HFC/embed_bridge/alive.json'))['time']))" 2>/dev/null || echo 9999; }
if [ "$(alive_age)" -gt 30 ]; then
  W8=$HFC/llm2vec_llama3_8b_sup_w8  # tools/build_llm2vec_w8.py --to_dir (cosine 0.9996 of bf16, 60 prompts)
  if [ -d "$W8" ]; then EMB=(--w8 "$W8"); else EMB=(); echo "[session] no int8 encoder at $W8: bf16 (16.4 GB)"; fi
  echo "[session] starting the text-embed server (Windows CPU)"
  .venv-embed/Scripts/python.exe -u tools/kimodo_text_embed_server.py "${EMB[@]}" --low_priority \
    > logs/live/embed_server.log 2>&1 &
  until [ "$(alive_age)" -lt 30 ]; do sleep 2; done
fi
echo "[session] text-embed server up"

if ! MSYS_NO_PATHCONV=1 wsl.exe -d "$WSL_DISTRO" -- pgrep -f kimodo_live_service.py > /dev/null 2>&1; then
  echo "[session] starting the Kimodo service (WSL GPU)"
  : > logs/live/service.log
  MSYS_NO_PATHCONV=1 wsl.exe -d "$WSL_DISTRO" --cd "$(pwd -W)" -- bash tools/kimodo_live_service_wsl.sh \
    > logs/live/service.log 2>&1 &
  until grep -q "\[live\] ready" logs/live/service.log 2>/dev/null; do
    grep -q "Traceback" logs/live/service.log 2>/dev/null && { echo "[session] Kimodo service failed:"; tail -20 logs/live/service.log; exit 1; }
    sleep 2
  done
fi
echo "[session] Kimodo service up"

if [ "${VIEWER:-web}" = gl ]; then
  echo "[session] opening the Newton viewer window"
  .venv-newton/Scripts/python.exe -u tools/live_viewer_gl.py --state logs/live/state.bin --idle_exit_s 10     ${SNAP:+--snapshot "$SNAP"} > logs/live/viewer.log 2>&1 &
  VIEWER_PID=$!
  until grep -q "model ready" logs/live/viewer.log 2>/dev/null; do
    grep -q "Traceback" logs/live/viewer.log 2>/dev/null && { echo "[session] viewer failed:"; tail -20 logs/live/viewer.log; exit 1; }
    sleep 1
  done
else
  if ! python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/events', timeout=1)" 2>/dev/null; then
    python -u tools/live_web.py > logs/live/web.log 2>&1 &
    until grep -q "\[web\]" logs/live/web.log 2>/dev/null; do sleep 0.5; done
  fi
  echo "[session] open http://127.0.0.1:8765 (robot view + prompt box)"
fi

echo "[session] physics: $(basename "$CKPT") on $PROFILE for $MIN min"
"$ISAACPY" -u scripts/play.py --task Dropbear-Tracking-Flat-NoState-Play-v0 --motion_file "$FIRST" \
  --checkpoint "$CKPT" --actuator_profile "$PROFILE" --solver_iters 32 4 --num_envs 1 --steps $(python -c "print(int(float('$MIN') * 3000))") \
  --headless --device cpu --realtime --live_dir logs/live/inbox --state_out logs/live/state.bin \
  > logs/live/physics.log 2>&1
grep -a '^{"timing"' logs/live/physics.log
[ -n "$VIEWER_PID" ] && wait $VIEWER_PID 2>/dev/null
echo "[session] done. The text-embed server and the Kimodo service stay warm for the next session; to stop them:"
echo "  wsl.exe -d $WSL_DISTRO -- pkill -f kimodo_live_service.py ; the embed server and tools/live_web.py are Windows python processes"
