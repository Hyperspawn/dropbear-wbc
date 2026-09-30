#!/usr/bin/env bash
# Pre-flight on the Brev box (~10 min, all 4 GPUs in parallel) before the overnight runs:
#  GPU0: library env probe (78-clip manifest) + 12-it library throughput at 8192 envs, 16/4
#  GPU1: NoState library task builds + 12 its at 16384 envs, 16/4 (throughput ceiling)
#  GPU2: single-clip (Take_102_v4) 12 its at 4096 envs, 16/4
#  GPU3: locomotion 1 chunk x 10 its at 2048 envs through the Linux launcher (run_chunked_locomotion --isaac_python)
source ~/isaac/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES DROPBEAR_USD=$HOME/assets/dropbear.usd PYTHONUNBUFFERED=1
CAL=data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json
LIB=data/motions/libraries/accepted_v2.json
mkdir -p ~/runs/preflight
( cd ~/dbw0 && mkdir -p logs/brev && export CUDA_VISIBLE_DEVICES=0 DROPBEAR_CALIBRATION_JSON=$PWD/$CAL &&
  python -u tools/library_env_probe.py --task Dropbear-Tracking-Library-v0 --motion_library $LIB --num_envs 256 --steps 150 \
     --out logs/brev/library_env_probe.json --headless > ~/runs/preflight/gpu0_probe.log 2>&1; echo "probe rc=$?" >> ~/runs/preflight/summary.txt
  python -u scripts/train.py --task Dropbear-Tracking-Library-v0 --motion_library $LIB --num_envs 8192 --max_iterations 12 \
     --save_interval 1000 --solver_iters 16 4 --run_name pf_lib8192 --headless > ~/runs/preflight/gpu0_tp.log 2>&1; echo "lib8192 rc=$?" >> ~/runs/preflight/summary.txt ) &
( cd ~/dbw1 && mkdir -p logs/brev && export CUDA_VISIBLE_DEVICES=1 DROPBEAR_CALIBRATION_JSON=$PWD/$CAL &&
  python -u scripts/train.py --task Dropbear-Tracking-Library-NoState-v0 --motion_library $LIB --num_envs 16384 --max_iterations 12 \
     --save_interval 1000 --solver_iters 16 4 --run_name pf_nostate16k --headless > ~/runs/preflight/gpu1_tp.log 2>&1; echo "nostate16k rc=$?" >> ~/runs/preflight/summary.txt ) &
( cd ~/dbw2 && mkdir -p logs/brev && export CUDA_VISIBLE_DEVICES=2 DROPBEAR_CALIBRATION_JSON=$PWD/$CAL &&
  python -u scripts/train.py --task Dropbear-Tracking-Flat-v0 --motion_file data/motions/unitree_rl_lab_mimic/G1_Take_102_v4.npz --num_envs 4096 \
     --max_iterations 12 --save_interval 1000 --solver_iters 16 4 --run_name pf_clip4096 --headless > ~/runs/preflight/gpu2_tp.log 2>&1; echo "clip4096 rc=$?" >> ~/runs/preflight/summary.txt ) &
( cd ~/dbw3 && mkdir -p logs/brev && export CUDA_VISIBLE_DEVICES=3 &&
  python -u tools/run_chunked_locomotion.py --run_name pf_loco --chunks 1 --iters_per_chunk 10 --num_envs 2048 --solver_iters 16 4 \
     --save_interval 100 --timeout 1800 --first_wait_minutes 2 --wait_minutes 2 --gap_s 1 --actuator_profile stiff_knee_hw \
     --isaac_python python --log_dir logs/brev --owner pf > ~/runs/preflight/gpu3_loco.log 2>&1; echo "loco rc=$?" >> ~/runs/preflight/summary.txt ) &
wait
echo "PREFLIGHT_DONE $(date -Is)" >> ~/runs/preflight/summary.txt
