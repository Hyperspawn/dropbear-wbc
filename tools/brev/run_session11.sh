#!/usr/bin/env bash
# Cloud session 11 (planned; Brev balance permitting): natural motion from data. The hand-written arm-swing rules are
# off; an adversarial motion prior (--amp_weight, tasks/locomotion/amp.py) rewards whole-body motion the discriminator
# takes for the retargeted human walking / stepping / balance clips. Free arms, pushes up to 0.6 m/s, terrain, stop
# model + safety terms as session 10. Every job starts from a session-10 walker whose arms already hang at the sides
# (from the session-9 tightrope arm the discriminator wins on arm posture alone and the style reward goes to ~0):
# --symmetry mirrors every PPO batch left/right (tasks/locomotion/symmetry.py): locally only one arm moved toward the
# clips, and the session-10 walker's rsl-rl "symmetry loss" is ~4.1.
#   GPU0  amp_x8_sym     vel_swing_x8/25329 (hw_v4_arms, X8-120 shoulders), style weight 1.0, symmetry
#   GPU1  amp_x8         the same without symmetry, same seed (the A/B)
#   GPU2  amp_x8_w2_sym  style weight 2.0 (style vs velocity tracking), symmetry
#   GPU3  amp_v4_sym     stock shoulders (vel_swing_v4 final, hw_v4), style weight 1.0, symmetry
# The bundle must ship the clips: pack_brev_bundle.py --manifest data/motions_v6ts/libraries/accepted_v8push.json (42 AMP clips incl. 12 push-recovery).
# Usage on the box, after setup_instance.sh and unpacking s11_init.tgz into every ~/dbw<g>:  bash run_session11.sh
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
X8LOCO=$(echo $LOCO | sed 's/--actuator_profile hw_v4/--actuator_profile hw_v4_arms/')
COMMON="$SAFE --thermal_penalty=-2e-4 --push_velocity=0.6 $ARMS"
line dbw0 amp_x8_sym $L --run_name amp_x8_sym --resume_run 2026-10-22_00-00-00_amp_x8_sym --seed 1101 $X8LOCO $COMMON --amp_weight=1.0 --symmetry | queue 0
line dbw1 amp_x8 $L --run_name amp_x8 --resume_run 2026-10-22_00-00-01_amp_x8 --seed 1101 $X8LOCO $COMMON --amp_weight=1.0 | queue 1
line dbw2 amp_x8_w2_sym $L --run_name amp_x8_w2_sym --resume_run 2026-10-22_00-00-02_amp_x8_w2_sym --seed 1121 $X8LOCO $COMMON --amp_weight=2.0 --symmetry | queue 2
line dbw3 amp_v4_sym $L --run_name amp_v4_sym --resume_run 2026-10-22_00-00-03_amp_v4_sym --seed 1131 $LOCO $COMMON --amp_weight=1.0 --symmetry | queue 3
for g in 0 1 2 3; do
  tmux kill-session -t gpu$g 2>/dev/null || true
  tmux new-session -d -s gpu$g "bash ~/runs/queue$g.sh"
done
tmux ls
echo "launched $(date -Is)"
