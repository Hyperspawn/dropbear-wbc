#!/usr/bin/env bash
# Cloud session 10 (2026-10-21): natural arm swing. Free arms (scale 0.3, pull to stand -0.02), pushes up to 0.6 m/s,
# terrain as session 8, stop model + safety terms; the swing term rewards each shoulder pitch following 0.45 x the
# opposite hip pitch while walking (mdp.arm_swing_counter_phase) and pulls the shoulder rolls in (no tightrope arm):
#   GPU0  vel_swing_x8        vel_arms_x8_v4/22200 (hw_v4_arms, X8-120 shoulders) + swing 0.5, abduction -0.3
#   GPU1  vel_swing_x8_lite   the same with abduction -0.1 (keeps more of the arm's push help)
#   GPU2  vel_swing_x8_heavy  GPU0 at --robot_mass_kg 65.2 (the user's ~60 kg estimate + Build B / shoulder swaps)
#   GPU3  vel_swing_v4        stock shoulders (vel_arms_v4/22200, hw_v4) + the GPU0 swing settings
# Usage on the box, after setup_instance.sh and unpacking s10_init.tgz into every ~/dbw<g>:  bash run_session10.sh
set -u
mkdir -p ~/runs
cat > ~/runs/env.sh <<EOF
source ~/isaac/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export DROPBEAR_USD=\$HOME/assets/dropbear.usd
export PYTHONUNBUFFERED=1
CAL=data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json
EOF
SAFE="--stop_load_penalty=-5 --impact_penalty=-0.005"
LOCO="--chunks 100 --iters_per_chunk 150 --num_envs 4096 --solver_iters 8 4 --save_interval 150 --timeout 7200   --first_wait_minutes 5 --wait_minutes 5 --gap_s 5 --log_dir logs/brev --isaac_python python --calibration \$CAL   --actuator_profile hw_v4 --gait_v3 --terrain rough --feet_width_weight=-5 --action_rate_weight=-0.05   --target_clamp_margin_deg=0 --stair_max_m=0.15 --slope_max=0.35 --push_interval 4 8"
TRK="--save_interval 200 --timeout 7200 --gap_s 5 --log_dir logs/brev --isaac_python python --calibration \$CAL   --actuator_profile hw_v4 --hw_regularizers --feet_slide_penalty=-1 --target_clamp_margin_deg=0 --solver_iters 16 4"
LIB="--task Dropbear-Tracking-Library-NoState-v0 --motion_library data/motions_v6ts/libraries/accepted_v7gen.json   --chunks 60 --iters_per_chunk 200 --num_envs 8192"
OBJ="--anchor_height_weight=1.0 --feet_height_weight=1.0 --body_pos_std=0.2 --joint_pos_weight=0.5"
T=tools/run_chunked_training.py
L=tools/run_chunked_locomotion.py

line() {  # dir name driver-args...: one job, logged to the queue's gpu log
  local d=$1 name=$2; shift 2
  echo "( cd ~/$d && mkdir -p logs/brev && echo \"\$(date -Is) START $name\" >> ~/runs/gpu\$G.log && python -u $* --owner brev_$d > logs/brev/${name}_driver.log 2>&1; echo \"\$(date -Is) END $name rc=\$?\" >> ~/runs/gpu\$G.log )"
}
queue() {  # gpu, then job lines on stdin; several jobs on one GPU run in parallel, QUEUE_DONE after all of them
  local g=$1 f=~/runs/queue$1.sh
  { echo "#!/usr/bin/env bash"; echo "source ~/runs/env.sh"; echo "export CUDA_VISIBLE_DEVICES=$g G=$g";
    sed 's/$/ \&/'; echo "wait"; echo "echo \"\$(date -Is) QUEUE_DONE\" >> ~/runs/gpu$g.log"; } > $f
  chmod +x $f
}
ARMS="--arm_action_scale=0.3 --arm_deviation_weight=-0.02 --arm_swing_weight=0.5"
X8LOCO=$(echo $LOCO | sed 's/--actuator_profile hw_v4/--actuator_profile hw_v4_arms/')
line dbw0 vel_swing_x8 $L --run_name vel_swing_x8 --resume_run 2026-10-21_00-00-00_vel_swing_x8 --seed 1001 $X8LOCO $SAFE   --thermal_penalty=-2e-4 --push_velocity=0.6 $ARMS --arm_abduction_weight=-0.3 | queue 0
line dbw1 vel_swing_x8_lite $L --run_name vel_swing_x8_lite --resume_run 2026-10-21_00-00-01_vel_swing_x8_lite --seed 1011   $X8LOCO $SAFE --thermal_penalty=-2e-4 --push_velocity=0.6 $ARMS --arm_abduction_weight=-0.1 | queue 1
line dbw2 vel_swing_x8_heavy $L --run_name vel_swing_x8_heavy --resume_run 2026-10-21_00-00-02_vel_swing_x8_heavy --seed 1021   $X8LOCO $SAFE --thermal_penalty=-2e-4 --push_velocity=0.6 $ARMS --arm_abduction_weight=-0.3 --robot_mass_kg=65.2 | queue 2
line dbw3 vel_swing_v4 $L --run_name vel_swing_v4 --resume_run 2026-10-21_00-00-03_vel_swing_v4 --seed 1031 $LOCO $SAFE   --thermal_penalty=-2e-4 --push_velocity=0.6 $ARMS --arm_abduction_weight=-0.3 | queue 3
for g in 0 1 2 3; do
  tmux kill-session -t gpu$g 2>/dev/null || true
  tmux new-session -d -s gpu$g "bash ~/runs/queue$g.sh"
done
tmux ls
echo "launched $(date -Is)"
