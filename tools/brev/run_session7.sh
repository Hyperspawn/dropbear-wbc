#!/usr/bin/env bash
# Cloud session 7 (2026-10-17, 4x RTX 6000 Ada): train for the real parts. Everything on Build B (hw_v4) with the
# explicit stop model, a stop-force penalty (knee-lock blocks) and a landing-impact penalty (2-3 kN at the ankle and
# knee pins, tools/joint_loads.py), plus the knee-elastic (theraband) A/B:
#   GPU0  lib_v4_scratch    universal tracker FROM SCRATCH: height rewards, stricter tracking, safety penalties
#   GPU1  lib_v4_safe       today's Build B tracker (lib_v4_h/23800) fine-tuned with the same objective
#   GPU2  spec_squat_safe   squat from scratch, safety penalties            + vel_rough_v4_safe (walker, same penalties)
#   GPU3  spec_squat_band   the same squat + knee elastic 150 N*m/rad from 15 deg crank + vel_rough_v4_band (walker)
# GPU2 and GPU3 carry the same load, so the elastic is the only difference between them.
# Usage on the box, after setup_instance.sh and unpacking s7_init.tgz into every ~/dbw<g>:  bash run_session7.sh
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
BAND="--knee_band 150,15"
OBJ="--anchor_height_weight=1.0 --feet_height_weight=1.0 --body_pos_std=0.2 --joint_pos_weight=0.5"
TRK="--save_interval 200 --timeout 7200 --gap_s 5 --log_dir logs/brev --isaac_python python --calibration \$CAL \
  --actuator_profile hw_v4 --hw_regularizers --feet_slide_penalty=-1 --target_clamp_margin_deg=0 --solver_iters 16 4"
LIB="--task Dropbear-Tracking-Library-NoState-v0 --motion_library data/motions_v6ts/libraries/accepted_v7gen.json \
  --chunks 60 --iters_per_chunk 200 --num_envs 8192"
SQUAT="--task Dropbear-Tracking-Flat-NoState-v0 --motion_file data/motions_v6ts/kimodo_gen/squat_ts1.19_v6.npz \
  --chunks 60 --iters_per_chunk 200 --num_envs 4096 --anchor_height_weight=1.0 --feet_height_weight=1.0"
LOCO="--chunks 100 --iters_per_chunk 150 --num_envs 2048 --solver_iters 8 4 --save_interval 150 --timeout 7200 \
  --first_wait_minutes 5 --wait_minutes 5 --gap_s 5 --log_dir logs/brev --isaac_python python --calibration \$CAL \
  --actuator_profile hw_v4 --gait_v3 --terrain rough --feet_width_weight=-5 --action_rate_weight=-0.05 \
  --target_clamp_margin_deg=0 --thermal_penalty=-2e-4"
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
line dbw0 lib_v4_scratch $T --run_name lib_v4_scratch --seed 701 $LIB $TRK $OBJ $SAFE | queue 0
line dbw1 lib_v4_safe $T --run_name lib_v4_safe --seed 711 --init_run 2026-10-17_10-00-00_lib_v4_h_init \
  --init_checkpoint model_23800.pt $LIB $TRK $OBJ $SAFE | queue 1
{ line dbw2 spec_squat_safe $T --run_name spec_squat_safe --seed 721 $SQUAT $TRK $SAFE
  line dbw4 vel_rough_v4_safe $L --run_name vel_rough_v4_safe --resume_run 2026-10-17_10-00-00_vel_rough_v4_safe \
    --seed 731 $LOCO $SAFE; } | queue 2
{ line dbw3 spec_squat_band $T --run_name spec_squat_band --seed 721 $SQUAT $TRK $SAFE $BAND
  line dbw5 vel_rough_v4_band $L --run_name vel_rough_v4_band --resume_run 2026-10-17_10-00-01_vel_rough_v4_band \
    --seed 731 $LOCO $SAFE $BAND; } | queue 3
for g in 0 1 2 3; do
  tmux kill-session -t gpu$g 2>/dev/null || true
  tmux new-session -d -s gpu$g "bash ~/runs/queue$g.sh"
done
tmux ls
echo "launched $(date -Is)"
