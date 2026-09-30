#!/usr/bin/env bash
# Runs ON the Brev instance. Usage: bash swap_gait.sh <gpu> <run_name> <warm_from_run_suffix|none> [gait flags] [profile]
# Stops GPU <gpu>'s queue, installs ~/hw_patch.tgz into ~/dbw<gpu>, and trains hw_v1 + thermal -5e-4 + gait shaping
# (target clamp, knee flexion in swing, swing clearance, torque rate, one-leg-hopping penalty; ACTUATORS.md 12).
set -u
source ~/runs/env.sh
G=$1; name=$2; warm=$3; GAIT=${4:---gait_shaping}; PROF=${5:-hw_v1}
EXPREL=logs/rsl_rl/dropbear_velocity
TMP=""
if [ "$warm" != "none" ]; then
  SRC_RUN=$(ls -d ~/dbw*/$EXPREL/*_$warm | tail -1)
  SRC_CK=$(ls "$SRC_RUN"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
  [ -f "$SRC_CK" ] || { echo "no source checkpoint"; exit 1; }
  TMP=~/warm_$name.pt; cp "$SRC_CK" "$TMP"
  echo "source checkpoint: $SRC_CK"
fi
old=$(grep -a START ~/runs/gpu$G.log | tail -1 | awk '{print $3}')
tmux kill-session -t gpu$G 2>/dev/null
sleep 3
for sig in TERM KILL; do
  for p in $(pgrep -f python); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in */dbw$G) kill -$sig $p 2>/dev/null;; esac
  done
  sleep 8
done
rm -f ~/dbw$G/.locks/gpu.lock
echo "$(date -Is) END $old rc=stopped_for_$name" >> ~/runs/gpu$G.log
tar -xzf ~/hw_patch.tgz -C ~/dbw$G
RESUME=""
if [ -n "$TMP" ]; then
  STAMP=$(date +%Y-%m-%d_%H-%M-%S)
  dst=~/dbw$G/$EXPREL/${STAMP}_$name
  mkdir -p "$dst"; mv "$TMP" "$dst/$(basename $SRC_CK)"
  echo "{\"warm_start_from\": \"$SRC_CK\"}" > "$dst/warm_start.json"
  RESUME="--resume_run ${STAMP}_$name"
fi
cat > ~/runs/queue$G.sh <<EOQ
#!/usr/bin/env bash
source ~/runs/env.sh
export CUDA_VISIBLE_DEVICES=$G G=$G
mkdir -p ~/dbw$G/logs/brev
cd ~/dbw$G && echo "\$(date -Is) START $name" >> ~/runs/gpu$G.log && python -u tools/run_chunked_locomotion.py --run_name $name $RESUME --chunks 80 --iters_per_chunk 200 --until_iteration 20000 --num_envs \$CLIP_ENVS --solver_iters \$SPOS 4 --save_interval 100 --timeout 7200 --first_wait_minutes 5 --wait_minutes 5 --gap_s 5 --actuator_profile $PROF --thermal_penalty=-5e-4 $GAIT --isaac_python python --log_dir logs/brev --owner brev_gpu$G > logs/brev/${name}_driver.log 2>&1; echo "\$(date -Is) END $name rc=\$?" >> ~/runs/gpu$G.log
echo "\$(date -Is) QUEUE_DONE" >> ~/runs/gpu$G.log
EOQ
chmod +x ~/runs/queue$G.sh
tmux new-session -d -s gpu$G "bash ~/runs/queue$G.sh"
sleep 2; tmux ls; echo "swapped gpu$G -> $name (warm: $warm) $(date -Is)"
