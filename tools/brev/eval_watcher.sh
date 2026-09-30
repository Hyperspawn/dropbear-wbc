#!/usr/bin/env bash
# Runs on the Brev box next to the training queues. As soon as a single-clip job ENDs, evaluate its final checkpoint
# at the strict 32/4 solver (nominal from frame 0, continuous loop, perturbed continuous), export ONNX + sidecar, and
# record the nominal rollout (for clean kinematic rendering at home). At LIB_EVAL_AT (UTC) evaluate the latest library
# checkpoints the same way (one env per clip) while they keep training. Results go under ~/dbw<g>/logs/brev/eval/
# and are pulled home by the watchdog.
source ~/runs/env.sh
LIB_EVAL_AT=${LIB_EVAL_AT:-2026-09-25T10:00:00+00:00}
M=data/motions
declare -A NPZ=( [take102_v4]=$M/unitree_rl_lab_mimic/G1_Take_102_v4.npz [gangnam_v4]=$M/unitree_rl_lab_mimic/G1_gangnam_style_V01_v4.npz
  [lafan_dance2_v4]=$M/gmr_lafan1/lafan1_dance2_subject1_v4.npz [lafan_walk1_v4]=$M/gmr_lafan1/lafan1_walk1_subject1_v4.npz
  [kimodo_walk_v4]=$M/kimodo_g1/output_walk_v4.npz [wave_v2_s16]=$M/synthetic/wave_right_v2.npz )
declare -A GPU=( [take102_v4]=2 [gangnam_v4]=2 [lafan_dance2_v4]=2 [lafan_walk1_v4]=3 [kimodo_walk_v4]=3 [wave_v2_s16]=3 )
declare -A STEPS=( [take102_v4]=2920 [gangnam_v4]=3240 [lafan_dance2_v4]=2000 [lafan_walk1_v4]=1500 [kimodo_walk_v4]=1000 [wave_v2_s16]=1110 )
log() { echo "$(date -Is) $*" >> ~/runs/eval.log; }

eval_clip() {  # job
  local job=$1 g=${GPU[$1]} npz=${NPZ[$1]} steps=${STEPS[$1]}
  cd ~/dbw$g || return
  local run; run=$(ls -td logs/rsl_rl/dropbear_tracking/*_$job 2>/dev/null | head -1)
  [ -z "$run" ] && { log "no run dir for $job"; return; }
  local rn; rn=$(basename "$run"); mkdir -p logs/brev/eval
  export CUDA_VISIBLE_DEVICES=$g
  local P="python -u scripts/play.py --task Dropbear-Tracking-Flat-Play-v0 --motion_file $npz --load_run $rn --solver_iters 32 4 --headless"
  log "eval start $job ($rn) gpu$g"
  $P --num_envs 16 --steps $steps --export > logs/brev/eval/${job}_nominal_32_4.log 2>&1; log "$job nominal rc=$?"
  $P --num_envs 16 --steps $steps --continuous_loop > logs/brev/eval/${job}_continuous_32_4.log 2>&1; log "$job continuous rc=$?"
  $P --num_envs 16 --steps $steps --continuous_loop --perturbed --seed 7 > logs/brev/eval/${job}_perturbed_32_4.log 2>&1; log "$job perturbed rc=$?"
  $P --num_envs 1 --steps $(( steps / 2 + 25 )) --record_rollout logs/brev/eval/rollout_${job}.npz > logs/brev/eval/${job}_record.log 2>&1; log "$job record rc=$?"
  # the 8/4 training-setting check (wave v2 lesson: a policy can depend on the looser solver)
  python -u scripts/play.py --task Dropbear-Tracking-Flat-Play-v0 --motion_file $npz --load_run $rn --solver_iters 16 4 --headless \
     --num_envs 16 --steps $steps > logs/brev/eval/${job}_nominal_16_4.log 2>&1; log "$job nominal16 rc=$?"
}

eval_lib() {  # g task_play run_name
  local g=$1 task=$2 job=$3
  cd ~/dbw$g || return
  local run; run=$(ls -td logs/rsl_rl/dropbear_tracking_library/*_$job 2>/dev/null | head -1)
  [ -z "$run" ] && { log "no run dir for $job"; return; }
  local rn; rn=$(basename "$run"); mkdir -p logs/brev/eval
  export CUDA_VISIBLE_DEVICES=$g
  log "lib eval start $job ($rn) gpu$g"
  python -u scripts/play.py --task $task --motion_library data/motions/libraries/accepted_v2.json --load_run $rn \
     --num_envs 78 --steps 1700 --solver_iters 32 4 --continuous_loop --export --headless > logs/brev/eval/${job}_lib_32_4.log 2>&1
  log "$job lib eval rc=$?"
}

lib_done=0
while true; do
  for job in "${!NPZ[@]}"; do
    g=${GPU[$job]}
    if grep -q "END $job" ~/runs/gpu$g.log 2>/dev/null && [ ! -f ~/runs/evaldone_$job ]; then
      eval_clip "$job"; touch ~/runs/evaldone_$job
    fi
  done
  if [ $lib_done = 0 ] && [ "$(date -u +%s)" -ge "$(date -u -d "$LIB_EVAL_AT" +%s)" ]; then
    eval_lib 0 Dropbear-Tracking-Library-Play-v0 lib_v2_main &
    eval_lib 1 Dropbear-Tracking-Library-NoState-Play-v0 lib_v2_nostate &
    wait; lib_done=1
  fi
  n=$(ls ~/runs/evaldone_* 2>/dev/null | wc -l)
  if [ "$n" -ge "${#NPZ[@]}" ] && [ $lib_done = 1 ]; then log "EVAL_ALL_DONE"; exit 0; fi
  sleep 120
done
