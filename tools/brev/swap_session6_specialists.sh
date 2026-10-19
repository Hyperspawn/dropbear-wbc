#!/usr/bin/env bash
# Session 6, feasibility test (2026-10-16): replace the two terrain jobs (GPU2, GPU3) with single-motion policies trained
# FROM SCRATCH on Build B (hw_v4): can the hardware squat and balance on one foot at all once no inherited habit holds
# it back? Height rewards on, smoothness penalty at its default (-0.1), no knee-stop term (the hardware gate reports a
# knee resting on its stop). GPU0/GPU1 (library trackers with height rewards) are not touched.
# Run on the box:  bash swap_session6_specialists.sh
set -u
source ~/runs/env.sh
for g in 2 3; do
  tmux kill-session -t gpu$g 2>/dev/null || true
  for p in /proc/[0-9]*; do [ "$(readlink $p/cwd 2>/dev/null)" = "$HOME/dbw$g" ] && kill "${p#/proc/}" 2>/dev/null; done
done
sleep 8
for g in 2 3; do
  for p in /proc/[0-9]*; do [ "$(readlink $p/cwd 2>/dev/null)" = "$HOME/dbw$g" ] && kill -9 "${p#/proc/}" 2>/dev/null; done
  rm -f ~/dbw$g/.locks/gpu.lock
done
SPEC="--task Dropbear-Tracking-Flat-NoState-v0 --chunks 40 --iters_per_chunk 200 --num_envs 4096 --solver_iters 16 4 \
  --save_interval 200 --timeout 7200 --gap_s 5 --log_dir logs/brev --isaac_python python --calibration \$CAL \
  --actuator_profile hw_v4 --hw_regularizers --feet_slide_penalty=-1 --target_clamp_margin_deg=0 \
  --anchor_height_weight=1.0 --feet_height_weight=1.0"
for spec in "2 spec_squat_v4 data/motions_v6ts/kimodo_gen/squat_ts1.19_v6.npz" \
            "3 spec_onefoot_v4 data/motions_v6ts/asap_g1/motions_raw_tairantestbed_smpl_video_single_foot_balance_level1_filter_amass_ts1.19_v6.npz"; do
  set -- $spec; g=$1; name=$2; clip=$3
  [ -f ~/dbw$g/$clip ] || { echo "missing $clip in dbw$g"; continue; }
  {
    echo "#!/usr/bin/env bash"; echo "source ~/runs/env.sh"; echo "export CUDA_VISIBLE_DEVICES=$g G=$g"
    echo "cd ~/dbw$g && echo \"\$(date -Is) START $name\" >> ~/runs/gpu$g.log && python -u tools/run_chunked_training.py --run_name $name --motion_file $clip --seed 631 $SPEC --owner brev_gpu$g > logs/brev/${name}_driver.log 2>&1; echo \"\$(date -Is) END $name rc=\$?\" >> ~/runs/gpu$g.log"
    echo "echo \"\$(date -Is) QUEUE_DONE\" >> ~/runs/gpu$g.log"
  } > ~/runs/queue$g.sh
  chmod +x ~/runs/queue$g.sh
  sed -i "/QUEUE_DONE/d" ~/runs/gpu$g.log
  tmux new-session -d -s gpu$g "bash ~/runs/queue$g.sh"
  echo "gpu$g: $name ($clip)"
done
tmux ls
