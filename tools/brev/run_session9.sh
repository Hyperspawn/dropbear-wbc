#!/usr/bin/env bash
# Cloud session 9 (2026-10-20): arms for balance. Every walker continues the session-8 default walker
# (vel_push_v4/16050: pushes up to 0.6 m/s every 4-8 s, steps to 15 cm, slopes to 19 deg), stop model + safety terms:
#   GPU0  vel_arms_v4      arms free: action scale 0.1 -> 0.3 rad, pull to the standing pose -0.2 -> -0.02
#   GPU1  vel_ctrl_v4      control: the same training with the arms held as before (is any gain the arms or more time?)
#   GPU2  vel_arms_x8_v4   arms free on hw_v4_arms: X8-120 at shoulder pitch and roll (the user's idea)
#   GPU3  lib_v4_cool3     motion tracker (lib_v4_cool2/30600) with a lighter motor-heat penalty (-2e-3 -> -5e-4)
# Usage on the box, after setup_instance.sh and unpacking s9_init.tgz into every ~/dbw<g>:  bash run_session9.sh
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
ARMS="--arm_action_scale=0.3 --arm_deviation_weight=-0.02"
line dbw0 vel_arms_v4 $L --run_name vel_arms_v4 --resume_run 2026-10-20_00-00-00_vel_arms_v4 --seed 901 $LOCO $SAFE   --thermal_penalty=-2e-4 --push_velocity=0.6 $ARMS | queue 0
line dbw1 vel_ctrl_v4 $L --run_name vel_ctrl_v4 --resume_run 2026-10-20_00-00-01_vel_ctrl_v4 --seed 911 $LOCO $SAFE   --thermal_penalty=-2e-4 --push_velocity=0.6 | queue 1
line dbw2 vel_arms_x8_v4 $L --run_name vel_arms_x8_v4 --resume_run 2026-10-20_00-00-02_vel_arms_x8_v4 --seed 921   $(echo $LOCO | sed 's/--actuator_profile hw_v4/--actuator_profile hw_v4_arms/') $SAFE --thermal_penalty=-2e-4   --push_velocity=0.6 $ARMS | queue 2
line dbw3 lib_v4_cool3 $T --run_name lib_v4_cool3 --seed 931 --init_run 2026-10-19_19-01-55_lib_v4_cool2   --init_checkpoint model_30600.pt $LIB $TRK $OBJ $SAFE --thermal_penalty=-0.0005 | queue 3
for g in 0 1 2 3; do
  tmux kill-session -t gpu$g 2>/dev/null || true
  tmux new-session -d -s gpu$g "bash ~/runs/queue$g.sh"
done
tmux ls
echo "launched $(date -Is)"
