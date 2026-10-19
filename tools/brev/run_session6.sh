#!/usr/bin/env bash
# Cloud session 6 (2026-10-16, 4x RTX 6000 Ada): which motors to buy for the hips and knees, tested on both skills.
# The real robot (user, 2026-10-16) has CEM-60 at hip pitch and knee and no hip roll motor yet. Two candidate builds,
# each with its motor masses (robots/hw_motor_specs.py, twin v3; docs/ACTUATORS.md):
#   hw_cem6  6 x CEM-60 at hip roll, hip pitch and knee
#   hw_v4    X8-120 (V4) at hip roll + hip pitch, X10-200 (V4.1) at the knee
# Identical settings per build; only the actuator profile differs.
#   GPU0  lib_cem6_s6       the 99/138 library tracker (lib_v8_nokstop_kstop3/19800) fine-tuned on hw_cem6
#   GPU1  lib_v4_s6         the same on hw_v4
#   GPU2  vel_rough_cem6    the session-5 rough-terrain walker (vel_rough_v2i_ks0/3725) continued on hw_cem6
#   GPU3  vel_rough_v4      the same on hw_v4
# Usage on the box, after setup_instance.sh and unpacking s6_init.tgz into every ~/dbw<g>:
#   bash run_session6.sh [LIB_ENVS] [CLIP_ENVS]
set -u
LIB_ENVS=${1:-8192}
CLIP_ENVS=${2:-4096}
mkdir -p ~/runs
cat > ~/runs/env.sh <<EOF
source ~/isaac/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export DROPBEAR_USD=\$HOME/assets/dropbear.usd
export PYTHONUNBUFFERED=1
CAL=data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json
EOF

LIB="--task Dropbear-Tracking-Library-NoState-v0 --motion_library data/motions_v6ts/libraries/accepted_v7gen.json \
  --init_run 2026-10-15_00-00-00_lib99_init --init_checkpoint model_19800.pt --chunks 40 --iters_per_chunk 200 \
  --num_envs $LIB_ENVS --solver_iters 16 4 --save_interval 200 --timeout 7200 --gap_s 5 --log_dir logs/brev \
  --isaac_python python --calibration \$CAL \
  --hw_regularizers --feet_slide_penalty=-1 --target_clamp_margin_deg=0 --action_rate_weight=-0.3 --knee_stop_penalty=-3"
LOCO="--chunks 60 --iters_per_chunk 150 --num_envs $CLIP_ENVS --solver_iters 8 4 --save_interval 150 --timeout 7200 \
  --first_wait_minutes 5 --wait_minutes 5 --gap_s 5 --log_dir logs/brev --isaac_python python --calibration \$CAL \
  --gait_v3 --terrain rough --feet_width_weight=-5 --action_rate_weight=-0.05 --target_clamp_margin_deg=0 \
  --thermal_penalty=-2e-4 --knee_stop_penalty=-10"

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
L=tools/run_chunked_locomotion.py
job lib_cem6_s6 $T --run_name lib_cem6_s6 --actuator_profile hw_cem6 --seed 601 $LIB | write_queue 0
job lib_v4_s6 $T --run_name lib_v4_s6 --actuator_profile hw_v4 --seed 601 $LIB | write_queue 1
job vel_rough_cem6 $L --run_name vel_rough_cem6 --resume_run 2026-10-16_17-00-00_vel_rough_cem6 --actuator_profile hw_cem6 \
  --seed 611 $LOCO | write_queue 2
job vel_rough_v4 $L --run_name vel_rough_v4 --resume_run 2026-10-16_17-00-01_vel_rough_v4 --actuator_profile hw_v4 \
  --seed 611 $LOCO | write_queue 3

for g in 0 1 2 3; do
  tmux kill-session -t gpu$g 2>/dev/null || true
  tmux new-session -d -s gpu$g "bash ~/runs/queue$g.sh"
done
tmux ls
echo "launched $(date -Is) LIB_ENVS=$LIB_ENVS CLIP_ENVS=$CLIP_ENVS"
