#!/usr/bin/env bash
# Cloud session 5b (2026-10-16, the session-5 box, 1x RTX 6000 Ada): which motors to buy for the hips and knees.
# The real robot (user, 2026-10-16) has CEM-60 at hip pitch and knee and NO hip roll motor yet; two candidate builds,
# each with its motor masses (robots/hw_motor_specs.py, twin v3):
#   ~/dbw0  lib_cem6   6 x CEM-60 at hip roll, hip pitch and knee                     (actuator profile hw_cem6)
#   ~/dbw1  lib_v4     X8-120 (V4) at hip roll + hip pitch, X10-200 (V4.1) at the knee (actuator profile hw_v4)
# Both fine-tune the 99/138 library tracker (lib_v8_nokstop_kstop3/19800, trained on hw_v2ie) with identical settings,
# in parallel on the one GPU. One queue log (gpu0.log) gets QUEUE_DONE only after BOTH finish (the watchdog deletes
# the box on QUEUE_DONE). Usage on the box, after unpacking v3_upload.tgz into ~/dbw0 and ~/dbw1:
#   bash run_session5b.sh [ENVS_PER_JOB]
set -u
ENVS=${1:-4096}
INIT=2026-10-15_00-00-00_lib99_init
tmux kill-session -t gpu0 2>/dev/null || true
pkill -f run_chunked 2>/dev/null || true
sleep 5
pkill -f "scripts/train" 2>/dev/null || true
# keep the terrain run's queue log, start a fresh one (a stale QUEUE_DONE would make the watchdog delete the box)
[ -f ~/runs/gpu0.log ] && mv ~/runs/gpu0.log ~/runs/gpu0_s5_terrain.log
cat > ~/runs/env.sh <<EOF2
source ~/isaac/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export DROPBEAR_USD=\$HOME/assets/dropbear.usd
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
CAL=data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json
EOF2
HW="--hw_regularizers --feet_slide_penalty=-1 --target_clamp_margin_deg=0 --action_rate_weight=-0.3 --knee_stop_penalty=-3"
COMMON="--task Dropbear-Tracking-Library-NoState-v0 --motion_library data/motions_v6ts/libraries/accepted_v7gen.json \
  --init_run $INIT --init_checkpoint model_19800.pt --chunks 30 --iters_per_chunk 200 --num_envs $ENVS --solver_iters 16 4 \
  --save_interval 200 --timeout 7200 --gap_s 5 --log_dir logs/brev --isaac_python python --calibration \$CAL $HW"
job() {  # dir, run name, profile, seed
  echo "( cd ~/$1 && mkdir -p logs/brev && echo \"\$(date -Is) START $2\" >> ~/runs/gpu0.log && python -u tools/run_chunked_training.py \
    --run_name $2 --actuator_profile $3 --seed $4 $COMMON --owner brev_$1 > logs/brev/$2_driver.log 2>&1; \
    echo \"\$(date -Is) END $2 rc=\$?\" >> ~/runs/gpu0.log ) &"
}
{
  echo "#!/usr/bin/env bash"
  echo "source ~/runs/env.sh"
  job dbw0 lib_cem6 hw_cem6 531
  job dbw1 lib_v4 hw_v4 531
  echo "wait"
  echo "echo \"\$(date -Is) QUEUE_DONE\" >> ~/runs/gpu0.log"
} > ~/runs/queue0.sh
chmod +x ~/runs/queue0.sh
tmux new-session -d -s gpu0 "bash ~/runs/queue0.sh"
tmux ls
echo "launched $(date -Is) ENVS=$ENVS per job"
