#!/usr/bin/env bash
# Runs ON the Brev instance (2026-09-25 midday): stop GPU2 (vel_flat_kneehw_cloud, converged) and GPU3 (lib_v2_future,
# young A/B), install the hw-twin patch (~/hw_patch.tgz) into every checkout, and fine-tune the cloud walker on the
# real-actuator profiles: GPU2 hw_v1 (user's motor map), GPU3 hw_v1_cad (CAD motor map). See docs/ACTUATORS.md.
set -u
source ~/runs/env.sh
EXPREL=logs/rsl_rl/dropbear_velocity
SRC_RUN=$(ls -d ~/dbw2/$EXPREL/*_vel_flat_kneehw_cloud | tail -1)
SRC_CK=$(ls "$SRC_RUN"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
[ -f "$SRC_CK" ] || { echo "no source checkpoint"; exit 1; }
echo "source checkpoint: $SRC_CK"
RERUN=$(ls -d ~/dbw2/$EXPREL/*_vel_hw_v1_ft 2>/dev/null | head -1)

for g in 2 3; do tmux kill-session -t gpu$g 2>/dev/null; done
sleep 3
for sig in TERM KILL; do
  for p in $(pgrep -f python); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in */dbw2|*/dbw3) kill -$sig $p 2>/dev/null;; esac
  done
  sleep 8
done
rm -f ~/dbw2/.locks/gpu.lock ~/dbw3/.locks/gpu.lock
if [ -n "$RERUN" ]; then
  # a previous swap already ran: drop its warm-start run dirs and restart the hw jobs
  rm -rf ~/dbw2/$EXPREL/*_vel_hw_v1_ft ~/dbw3/$EXPREL/*_vel_hw_v1_cad_ft
  for g in 2 3; do echo "$(date -Is) END hw_twin rc=restarted_by_swap" >> ~/runs/gpu$g.log; done
else
  echo "$(date -Is) END vel_flat_kneehw_cloud rc=stopped_for_hw_twin" >> ~/runs/gpu2.log
  echo "$(date -Is) END lib_v2_future rc=stopped_for_hw_twin" >> ~/runs/gpu3.log
fi

for g in 0 1 2 3; do tar -xzf ~/hw_patch.tgz -C ~/dbw$g; done

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
for spec in "2 vel_hw_v1_ft hw_v1" "3 vel_hw_v1_cad_ft hw_v1_cad"; do
  set -- $spec; g=$1; name=$2; prof=$3
  dst=~/dbw$g/$EXPREL/${STAMP}_$name
  mkdir -p "$dst"
  cp "$SRC_CK" "$dst/"
  echo "{\"warm_start_from\": \"$SRC_CK\", \"actuator_profile\": \"$prof\"}" > "$dst/warm_start.json"
  cat > ~/runs/queue$g.sh <<EOF
#!/usr/bin/env bash
source ~/runs/env.sh
export CUDA_VISIBLE_DEVICES=$g G=$g
mkdir -p ~/dbw$g/logs/brev
cd ~/dbw$g && echo "\$(date -Is) START $name" >> ~/runs/gpu$g.log && python -u tools/run_chunked_locomotion.py --run_name $name --resume_run ${STAMP}_$name --chunks 80 --iters_per_chunk 200 --until_iteration 12000 --num_envs \$CLIP_ENVS --solver_iters \$SPOS 4 --save_interval 100 --timeout 7200 --first_wait_minutes 5 --wait_minutes 5 --gap_s 5 --actuator_profile $prof --isaac_python python --log_dir logs/brev --owner brev_gpu$g > logs/brev/${name}_driver.log 2>&1; echo "\$(date -Is) END $name rc=\$?" >> ~/runs/gpu$g.log
echo "\$(date -Is) QUEUE_DONE" >> ~/runs/gpu$g.log
EOF
  chmod +x ~/runs/queue$g.sh
  tmux new-session -d -s gpu$g "bash ~/runs/queue$g.sh"
done
sleep 2
tmux ls
grep -E "CLIP_ENVS|SPOS" ~/runs/env.sh
echo "swapped $(date -Is)"
