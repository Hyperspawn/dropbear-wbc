#!/usr/bin/env bash
# Overnight Brev training plan (2026-09-25, 4x RTX 6000 Ada). One checkout per GPU (~/dbw0..3) so each has its own
# gpu.lock; every job is a chunked, resumable driver (tools/run_chunked_training.py / run_chunked_locomotion.py).
# Usage:  bash run_all.sh [LIB_ENVS] [CLIP_ENVS] [SOLVER_POS]      (defaults 8192 4096 16)
# Status: ~/runs/gpu<N>.log (one line per job start/end) + each driver's logs/brev/*_driver.log.
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

# clip job: name npz iters seed
clip_job() {
  echo "cd ~/dbw\$G && echo \"\$(date -Is) START $1\" >> ~/runs/gpu\$G.log && python -u tools/run_chunked_training.py --task Dropbear-Tracking-Flat-v0 --motion_file $2 --run_name $1 --chunks $(( ($3 + 249) / 250 )) --iters_per_chunk 250 --num_envs \$CLIP_ENVS --solver_iters \$SPOS 4 --save_interval 250 --seed $4 --timeout 7200 --gap_s 5 --log_dir logs/brev --owner brev_gpu\$G --isaac_python python --calibration \$CAL > logs/brev/$1_driver.log 2>&1; echo \"\$(date -Is) END $1 rc=\$?\" >> ~/runs/gpu\$G.log"
}
lib_job() {  # name task chunks seed
  echo "cd ~/dbw\$G && echo \"\$(date -Is) START $1\" >> ~/runs/gpu\$G.log && python -u tools/run_chunked_training.py --task $2 --motion_library data/motions/libraries/accepted_v2.json --run_name $1 --chunks $3 --iters_per_chunk 200 --num_envs \$LIB_ENVS --solver_iters \$SPOS 4 --save_interval 200 --seed $4 --timeout 7200 --gap_s 5 --log_dir logs/brev --owner brev_gpu\$G --isaac_python python --calibration \$CAL > logs/brev/$1_driver.log 2>&1; echo \"\$(date -Is) END $1 rc=\$?\" >> ~/runs/gpu\$G.log"
}
loco_job() {  # name iters
  echo "cd ~/dbw\$G && echo \"\$(date -Is) START $1\" >> ~/runs/gpu\$G.log && python -u tools/run_chunked_locomotion.py --run_name $1 --chunks 80 --iters_per_chunk 200 --until_iteration $2 --num_envs \$CLIP_ENVS --solver_iters \$SPOS 4 --save_interval 100 --timeout 7200 --first_wait_minutes 5 --wait_minutes 5 --gap_s 5 --actuator_profile stiff_knee_hw --isaac_python python --log_dir logs/brev --owner brev_gpu\$G > logs/brev/$1_driver.log 2>&1; echo \"\$(date -Is) END $1 rc=\$?\" >> ~/runs/gpu\$G.log"
}

M=data/motions
write_queue() {  # gpu, then job lines on stdin
  local g=$1 f=~/runs/queue$1.sh
  { echo "#!/usr/bin/env bash"; echo "source ~/runs/env.sh"; echo "export CUDA_VISIBLE_DEVICES=$g G=$g"; echo "mkdir -p ~/dbw$g/logs/brev"; cat; echo "echo \"\$(date -Is) QUEUE_DONE\" >> ~/runs/gpu$g.log"; } > $f
  chmod +x $f
}

# GPU0: universal library tracker (78 clips, 512 s), runs until the budget deadline
lib_job lib_v2_main Dropbear-Tracking-Library-v0 90 101 | write_queue 0
# GPU1: same, policy WITHOUT simulator-only state (deployable without a state estimator)
lib_job lib_v2_nostate Dropbear-Tracking-Library-NoState-v0 90 101 | write_queue 1
# GPU2: faithful single-clip dances, then a stronger walking (velocity) policy until the deadline
{ clip_job take102_v4 $M/unitree_rl_lab_mimic/G1_Take_102_v4.npz 2500 11
  clip_job gangnam_v4 $M/unitree_rl_lab_mimic/G1_gangnam_style_V01_v4.npz 2500 12
  clip_job lafan_dance2_v4 $M/gmr_lafan1/lafan1_dance2_subject1_v4.npz 2500 13
  loco_job vel_flat_kneehw_cloud 12000; } | write_queue 2
# GPU3: walks + wave v2 at the stricter solver, then the library future-window A/B until the deadline
{ clip_job lafan_walk1_v4 $M/gmr_lafan1/lafan1_walk1_subject1_v4.npz 2500 21
  clip_job kimodo_walk_v4 $M/kimodo_g1/output_walk_v4.npz 2500 22
  clip_job wave_v2_s16 $M/synthetic/wave_right_v2.npz 1500 23
  lib_job lib_v2_future Dropbear-Tracking-Library-Future-v0 90 101; } | write_queue 3

for g in 0 1 2 3; do
  tmux kill-session -t gpu$g 2>/dev/null || true
  tmux new-session -d -s gpu$g "bash ~/runs/queue$g.sh"
done
tmux ls
echo "launched $(date -Is) LIB_ENVS=$LIB_ENVS CLIP_ENVS=$CLIP_ENVS SOLVER=$SPOS/4"
