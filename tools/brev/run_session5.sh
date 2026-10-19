#!/usr/bin/env bash
# Cloud session 5 (2026-10-16, 1x RTX 6000 Ada): a rough-terrain walker trained FROM SCRATCH with the knee-stop term
# from iteration 0. Every earlier velocity walker (flat and rough, scratch or fine-tuned) rested its knees on the
# flexion stop in ~95-99 % of stance, and a knee-stop fine-tune could not leave that optimum (docs/RESULTS.md, session 4).
# Usage on the box: bash run_session5.sh [ENVS]
set -u
ENVS=${1:-4096}
mkdir -p ~/runs
cat > ~/runs/env.sh <<EOF2
source ~/isaac/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export DROPBEAR_USD=\$HOME/assets/dropbear.usd
export PYTHONUNBUFFERED=1
CAL=data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json
EOF2
cat > ~/runs/queue0.sh <<EOF2
#!/usr/bin/env bash
source ~/runs/env.sh
export CUDA_VISIBLE_DEVICES=0 G=0
mkdir -p ~/dbw0/logs/brev
cd ~/dbw0 && echo "\$(date -Is) START vel_rough_v2i_ks0" >> ~/runs/gpu0.log && python -u tools/run_chunked_locomotion.py \
  --run_name vel_rough_v2i_ks0 --chunks 60 --iters_per_chunk 150 --num_envs $ENVS --solver_iters 8 4 --seed 501 \
  --save_interval 150 --timeout 7200 --first_wait_minutes 5 --wait_minutes 5 --gap_s 5 --log_dir logs/brev \
  --isaac_python python --calibration \$CAL --actuator_profile hw_v2i --gait_v3 --terrain rough --feet_width_weight=-5 \
  --action_rate_weight=-0.05 --target_clamp_margin_deg=0 --thermal_penalty=-2e-4 --knee_stop_penalty=-10 \
  --owner brev_gpu0 > logs/brev/vel_rough_v2i_ks0_driver.log 2>&1; echo "\$(date -Is) END vel_rough_v2i_ks0 rc=\$?" >> ~/runs/gpu0.log
echo "\$(date -Is) QUEUE_DONE" >> ~/runs/gpu0.log
EOF2
chmod +x ~/runs/queue0.sh
tmux kill-session -t gpu0 2>/dev/null || true
tmux new-session -d -s gpu0 "bash ~/runs/queue0.sh"
tmux ls
echo "launched $(date -Is) ENVS=$ENVS"
