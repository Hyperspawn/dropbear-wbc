#!/usr/bin/env bash
# Cloud session 4 (2026-10-15, 4x L40S): keep the knees off their hard stops (docs/RESULTS.md, session 3: the terrain
# walkers rested the knee on its flexion stop in 91-97 % of stance; the squat policy in 19-20 %). Vendor CEM-60 knee
# (hw_v2*), mass randomization. Usage on the box: bash run_session4.sh [LIB_ENVS] [CLIP_ENVS] [SOLVER_POS]
#   GPU0  vel_rough_v2i_kstop10   terrain walker + knee-stop term -10, warm start vel_rough_v2i/11460
#   GPU1  lowbody_v2ie_kstop3     squat set + light knee-stop term -3, warm start lowbody_v2ie/17800
#   GPU2  lib_v8_nokstop_kstop3   the 97/138 library tracker + light knee-stop term -3, warm start lib_v8_v2ie_nokstop/17381
#   GPU3  vel_rough_v2i_kstop3    terrain walker + knee-stop term -3 (weight A/B with GPU0)
set -u
LIB_ENVS=${1:-8192}
CLIP_ENVS=${2:-4096}
SPOS=${3:-16}
mkdir -p ~/runs
cat > ~/runs/env.sh <<EOF
source ~/isaac/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export DROPBEAR_USD=\$HOME/assets/dropbear.usd
export PYTHONUNBUFFERED=1
CAL=data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json
LIB_ENVS=$LIB_ENVS
CLIP_ENVS=$CLIP_ENVS
SPOS=$SPOS
EOF

HW="--hw_regularizers --feet_slide_penalty=-1 --target_clamp_margin_deg=0 --action_rate_weight=-0.3"
COMMON="--save_interval 200 --timeout 7200 --gap_s 5 --log_dir logs/brev --isaac_python python --calibration \$CAL"
LOCO="--chunks 40 --iters_per_chunk 150 --num_envs \$CLIP_ENVS --solver_iters 8 4 --save_interval 150 --timeout 7200 \
  --first_wait_minutes 5 --wait_minutes 5 --gap_s 5 --log_dir logs/brev --isaac_python python --calibration \$CAL \
  --actuator_profile hw_v2i --gait_v3 --terrain rough --feet_width_weight=-5 --action_rate_weight=-0.05 \
  --target_clamp_margin_deg=0 --thermal_penalty=-2e-4"

job() {  # name, then the driver arguments
  local name=$1; shift
  echo "cd ~/dbw\$G && echo \"\$(date -Is) START $name\" >> ~/runs/gpu\$G.log && python -u $* --owner brev_gpu\$G > logs/brev/${name}_driver.log 2>&1; echo \"\$(date -Is) END $name rc=\$?\" >> ~/runs/gpu\$G.log"
}
write_queue() {  # gpu, then job lines on stdin
  local g=$1 f=~/runs/queue$1.sh
  { echo "#!/usr/bin/env bash"; echo "source ~/runs/env.sh"; echo "export CUDA_VISIBLE_DEVICES=$g G=$g"; echo "mkdir -p ~/dbw$g/logs/brev"; cat; echo "echo \"\$(date -Is) QUEUE_DONE\" >> ~/runs/gpu$g.log"; } > $f
  chmod +x $f
}

T=tools/run_chunked_training.py
V7=data/motions_v6ts/libraries/accepted_v7gen.json
LOW=data/motions_v6ts/libraries/lowbody_v1.json

job vel_rough_v2i_kstop10 tools/run_chunked_locomotion.py --run_name vel_rough_v2i_kstop10 \
  --resume_run 2026-10-15_20-00-00_vel_rough_v2i_kstop10 --seed 401 $LOCO --knee_stop_penalty=-10 | write_queue 0

job lowbody_v2ie_kstop3 $T --task Dropbear-Tracking-Library-NoState-v0 --motion_library $LOW --run_name lowbody_v2ie_kstop3 \
  --init_run 2026-10-14_14-00-28_lowbody_v2ie --init_checkpoint model_17800.pt --chunks 30 --iters_per_chunk 200 \
  --num_envs \$CLIP_ENVS --solver_iters \$SPOS 4 --seed 411 $COMMON --actuator_profile hw_v2ie $HW --knee_stop_penalty=-3 \
  | write_queue 1

job lib_v8_nokstop_kstop3 $T --task Dropbear-Tracking-Library-NoState-v0 --motion_library $V7 --run_name lib_v8_nokstop_kstop3 \
  --init_run 2026-10-14_14-00-28_lib_v8_v2ie_nokstop --init_checkpoint model_17381.pt --chunks 30 --iters_per_chunk 200 \
  --num_envs \$LIB_ENVS --solver_iters \$SPOS 4 --seed 421 $COMMON --actuator_profile hw_v2ie $HW --knee_stop_penalty=-3 \
  | write_queue 2

job vel_rough_v2i_kstop3 tools/run_chunked_locomotion.py --run_name vel_rough_v2i_kstop3 \
  --resume_run 2026-10-15_20-00-01_vel_rough_v2i_kstop3 --seed 431 $LOCO --knee_stop_penalty=-3 | write_queue 3

for g in 0 1 2 3; do
  tmux kill-session -t gpu$g 2>/dev/null || true
  tmux new-session -d -s gpu$g "bash ~/runs/queue$g.sh"
done
tmux ls
echo "launched $(date -Is) LIB_ENVS=$LIB_ENVS CLIP_ENVS=$CLIP_ENVS SOLVER=$SPOS/4"
