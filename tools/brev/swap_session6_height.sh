#!/usr/bin/env bash
# Session 6, second half (2026-10-16): restart the two library-tracker jobs (GPU0 hw_cem6, GPU1 hw_v4) WITH the head and
# foot height rewards (scripts/train.py --anchor_height_weight / --feet_height_weight). The motors-only fine-tunes kept
# both feet planted on either build (tools/motion_fidelity.py), so the build comparison needs a policy that is asked to
# lift its feet. Each job continues from its own latest checkpoint; the terrain jobs on GPU2/GPU3 are not touched.
# Run on the box after copying the updated source files into ~/dbw0 and ~/dbw1:  bash swap_session6_height.sh
set -u
source ~/runs/env.sh
for g in 0 1; do
  tmux kill-session -t gpu$g 2>/dev/null || true
  for p in /proc/[0-9]*; do  # every process working in ~/dbw$g (driver + Isaac child)
    [ "$(readlink $p/cwd 2>/dev/null)" = "$HOME/dbw$g" ] && kill "${p#/proc/}" 2>/dev/null
  done
done
sleep 8
for g in 0 1; do
  for p in /proc/[0-9]*; do [ "$(readlink $p/cwd 2>/dev/null)" = "$HOME/dbw$g" ] && kill -9 "${p#/proc/}" 2>/dev/null; done
  rm -f ~/dbw$g/.locks/gpu.lock
done

LIB="--task Dropbear-Tracking-Library-NoState-v0 --motion_library data/motions_v6ts/libraries/accepted_v7gen.json \
  --chunks 40 --iters_per_chunk 200 --num_envs 8192 --solver_iters 16 4 --save_interval 200 --timeout 7200 --gap_s 5 \
  --log_dir logs/brev --isaac_python python --calibration \$CAL \
  --hw_regularizers --feet_slide_penalty=-1 --target_clamp_margin_deg=0 --action_rate_weight=-0.3 --knee_stop_penalty=-3 \
  --anchor_height_weight=1.0 --feet_height_weight=1.0"
for spec in "0 cem6 hw_cem6" "1 v4 hw_v4"; do
  set -- $spec; g=$1; b=$2; prof=$3
  run=$(ls -d ~/dbw$g/logs/rsl_rl/dropbear_tracking_library/*_lib_${b}_s6 | tail -1)
  ck=$(ls $run/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt$/\1/' | sort -n | tail -1)
  name=lib_${b}_h
  {
    echo "#!/usr/bin/env bash"; echo "source ~/runs/env.sh"; echo "export CUDA_VISIBLE_DEVICES=$g G=$g"
    echo "cd ~/dbw$g && echo \"\$(date -Is) START $name (from $(basename $run)/model_$ck.pt)\" >> ~/runs/gpu$g.log && python -u tools/run_chunked_training.py --run_name $name --actuator_profile $prof --seed 621 --init_run $(basename $run) --init_checkpoint model_$ck.pt $LIB --owner brev_gpu$g > logs/brev/${name}_driver.log 2>&1; echo \"\$(date -Is) END $name rc=\$?\" >> ~/runs/gpu$g.log"
    echo "echo \"\$(date -Is) QUEUE_DONE\" >> ~/runs/gpu$g.log"
  } > ~/runs/queue$g.sh
  chmod +x ~/runs/queue$g.sh
  sed -i "/QUEUE_DONE/d" ~/runs/gpu$g.log
  tmux new-session -d -s gpu$g "bash ~/runs/queue$g.sh"
  echo "gpu$g: $name from $(basename $run)/model_$ck.pt"
done
tmux ls
