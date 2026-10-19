#!/usr/bin/env bash
# Cloud session 3 (2026-10-14, 4x RTX 6000 Ada): retrain on the vendor CEM-60 knee (hw_v2*, docs/archive/ACTUATORS_HISTORY.md 14)
# with mass domain randomization (0..+6 kg on the base, limbs x0.9-1.1). One checkout per GPU (~/dbw0..3), every job
# a chunked, resumable driver. Usage on the box: bash run_session3.sh [LIB_ENVS] [CLIP_ENVS] [SOLVER_POS]
#   GPU0  lib_v8_v2ie          library tracker, 138 clips, warm start lib_v7gen_v1ie_smooth/13600, all hw terms
#   GPU1  lowbody_v2ie         squats / lunge / pick up / reach down (10 clips), no knee-stop term, then
#         kimodo_walk_v2ie     the deployable walk, warm start kimodo_ts119_allfix_smooth/5800
#   GPU2  vel_rough_v2i        blind rough-terrain walker, warm start vel_rough_v1i_b/5500
#   GPU3  lib_v8_v2ie_nokstop  GPU0 without the knee-stop term (A/B: does it hold the knees back?)
# Status: ~/runs/gpu<N>.log + each driver's logs/brev/*_driver.log.
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
LIB_INIT="--init_run 2026-09-26_07-52-45_lib_v7gen_v1ie_smooth --init_checkpoint model_13600.pt"

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

job lib_v8_v2ie $T --task Dropbear-Tracking-Library-NoState-v0 --motion_library $V7 --run_name lib_v8_v2ie $LIB_INIT \
  --chunks 30 --iters_per_chunk 200 --num_envs \$LIB_ENVS --solver_iters \$SPOS 4 --seed 301 $COMMON \
  --actuator_profile hw_v2ie $HW --knee_stop_penalty=-10 | write_queue 0

{ job lowbody_v2ie $T --task Dropbear-Tracking-Library-NoState-v0 --motion_library $LOW --run_name lowbody_v2ie $LIB_INIT \
    --restore_train_state off --chunks 10 --iters_per_chunk 200 --num_envs \$CLIP_ENVS --solver_iters \$SPOS 4 --seed 311 \
    $COMMON --actuator_profile hw_v2ie $HW
  job kimodo_walk_v2ie $T --task Dropbear-Tracking-Flat-NoState-v0 --motion_file data/motions_cyclic/kimodo_walk_ts119_cyclic.npz \
    --run_name kimodo_walk_v2ie --init_run 2026-09-25_22-52-53_kimodo_ts119_allfix_smooth --init_checkpoint model_5800.pt \
    --chunks 8 --iters_per_chunk 250 --num_envs \$CLIP_ENVS --solver_iters \$SPOS 4 --seed 312 $COMMON \
    --actuator_profile hw_v2ie $HW --knee_stop_penalty=-10
  job lowbody_v2ie_more $T --task Dropbear-Tracking-Library-NoState-v0 --motion_library $LOW --run_name lowbody_v2ie \
    --resume_run LOWBODY_RUN --chunks 30 --iters_per_chunk 200 --num_envs \$CLIP_ENVS --solver_iters \$SPOS 4 --seed 313 \
    $COMMON --actuator_profile hw_v2ie $HW; } | sed "s#--resume_run LOWBODY_RUN#--resume_run \$(basename \$(ls -d ~/dbw\$G/logs/rsl_rl/dropbear_tracking_library/*_lowbody_v2ie | tail -1))#" | write_queue 1

job vel_rough_v2i tools/run_chunked_locomotion.py --run_name vel_rough_v2i --resume_run 2026-10-14_20-00-00_vel_rough_v2i \
  --chunks 40 --iters_per_chunk 150 --num_envs \$CLIP_ENVS --solver_iters 8 4 --seed 321 --save_interval 150 \
  --timeout 7200 --first_wait_minutes 5 --wait_minutes 5 --gap_s 5 --log_dir logs/brev --isaac_python python \
  --calibration \$CAL --actuator_profile hw_v2i --gait_v3 --terrain rough --feet_width_weight=-5 \
  --action_rate_weight=-0.05 --target_clamp_margin_deg=0 --thermal_penalty=-2e-4 | write_queue 2

job lib_v8_v2ie_nokstop $T --task Dropbear-Tracking-Library-NoState-v0 --motion_library $V7 --run_name lib_v8_v2ie_nokstop \
  $LIB_INIT --chunks 30 --iters_per_chunk 200 --num_envs \$LIB_ENVS --solver_iters \$SPOS 4 --seed 302 $COMMON \
  --actuator_profile hw_v2ie $HW | write_queue 3

for g in 0 1 2 3; do
  tmux kill-session -t gpu$g 2>/dev/null || true
  tmux new-session -d -s gpu$g "bash ~/runs/queue$g.sh"
done
tmux ls
echo "launched $(date -Is) LIB_ENVS=$LIB_ENVS CLIP_ENVS=$CLIP_ENVS SOLVER=$SPOS/4"
