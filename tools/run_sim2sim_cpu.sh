#!/usr/bin/env bash
# Newton sim2sim of an exported tracking policy (demo_eval track, 2026-09-24). Same setup as
# logs/gpu_pipeline/sim2sim/run_sim2sim_wave_cpu.sh and logs/review_fixes/sim2sim/run_wave_ablation.sh: MuJoCo C backend on
# the CPU (no GPU lock), bridge defaults (passive damping 0 = sdk.motors.PASSIVE_DAMPING, 2 ms step), free base
# pre-settled 3 s at the clip's frame 0 (root pinned), lockstep 10 ticks (deterministic 50 Hz policy), private ports;
# runner in policy mode with the play.py --export sidecar (sim-only terms allowed), policy from t = 0.04 s, then
# hold(motion_end). Extra args go to the bridge (e.g. --diag-eq-solref 0.004 1).
# Usage: bash tools/run_sim2sim_cpu.sh <tag> <run dir or run name (needs <run>/exported/policy.json: play.py --export)> <npz> <runner duration s> <cmd port> <state port> [bridge args...]
set -u
cd "$(dirname "$0")/.."
TAG=$1; RUNNAME=$2; NPZ=$3; DUR=$4; CP=$5; SP=$6; shift 6
RUN=$RUNNAME; [ -d "$RUN" ] || RUN=logs/rsl_rl/dropbear_tracking/$RUNNAME  # a run folder, or a run name under logs/rsl_rl
OUT=logs/demo_eval/sim2sim; mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=-1
VPY=$( [ -x .venv-newton/Scripts/python.exe ] && echo .venv-newton/Scripts/python.exe || echo .venv-newton/bin/python )
BDUR=$(python -c "print(float('$DUR') + 2.0)")
BODIES=head_5mm_ujoint_base__5__1,LH_shoulder_ex_al_interface_1,RH_shoulder_ex_al_interface_1,LL_skateboard_bearing_left_2,RL_skateboard_bearing_left_2
echo "tag=$TAG run=$RUN npz=$NPZ runner_duration=$DUR bridge_duration=$BDUR bridge_extra=$*"
$VPY -u tools/newton_bridge.py --device cpu --mujoco-cpu --no-gpu-lock --cmd-port $CP --state-port $SP \
    --lockstep-ticks 10 --duration $BDUR --start-npz $NPZ --presettle-s 3 \
    --sim-bodies $BODIES --trace $OUT/${TAG}_trace.npz --trace-every 5 --report $OUT/${TAG}_bridge.json "$@" \
    > $OUT/${TAG}_bridge.log 2>&1 &
BPID=$!
$VPY -u tools/policy_runner.py --mode policy --sidecar $RUN/exported/policy.json --allow-privileged \
    --passive-s 0 --move-s 0.02 --hold-s 0 --duration $DUR --cmd-port $CP --state-port $SP \
    --log $OUT/${TAG}_runner.jsonl --summary $OUT/${TAG}_runner.json > $OUT/${TAG}_runner.log 2>&1
RRC=$?
echo "runner rc=$RRC"
wait $BPID
BRC=$?
echo "bridge rc=$BRC"
# the nominal Isaac evaluation (16 envs, no randomization, reset_at_wrap) of this run, newest
PLAY=$(python - "$RUN" << 'PYEOF'
import json, sys
from pathlib import Path
best = None
for p in sorted(Path(sys.argv[1]).glob("play_*.json")):
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    e = d.get("eval_design") or {}
    if d.get("num_envs") == 16 and str(e.get("randomization", "")).startswith("none") and str(e.get("loop_mode", "")).startswith("reset_at_wrap"):
        best = p
print(best or "")
PYEOF
)
echo "isaac play summary: $PLAY"
$VPY -u tools/sim2sim_report.py --runner-log $OUT/${TAG}_runner.jsonl --runner-summary $OUT/${TAG}_runner.json \
    --bridge-trace $OUT/${TAG}_trace.npz --bridge-report $OUT/${TAG}_bridge.json \
    --npz $NPZ ${PLAY:+--isaac-play $PLAY} --out $OUT/${TAG}_report.json > $OUT/${TAG}_report.log 2>&1
PRC=$?
echo "report rc=$PRC"
exit $(( RRC != 0 || PRC != 0 ))
