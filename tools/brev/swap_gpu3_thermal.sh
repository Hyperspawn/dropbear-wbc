#!/usr/bin/env bash
# Runs ON the Brev instance (2026-09-25 ~12:50 IST): stop GPU3 (vel_hw_v1_cad_ft; the CAD A/B answered: it walks, and
# its knee is pinned at peak too), install the updated hw patch (~/hw_patch.tgz: thermal penalty) into every checkout,
# and fine-tune the hw_v1 walker WITH the over-rated-torque penalty on GPU3 (warm start = GPU2's latest vel_hw_v1_ft).
set -u
source ~/runs/env.sh
EXPREL=logs/rsl_rl/dropbear_velocity
SRC_RUN=$(ls -d ~/dbw2/$EXPREL/*_vel_hw_v1_ft | tail -1)
SRC_CK=$(ls "$SRC_RUN"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
[ -f "$SRC_CK" ] || { echo "no source checkpoint"; exit 1; }
echo "source checkpoint: $SRC_CK"

tmux kill-session -t gpu3 2>/dev/null
sleep 3
for sig in TERM KILL; do
  for p in $(pgrep -f python); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in */dbw3) kill -$sig $p 2>/dev/null;; esac
  done
  sleep 8
done
rm -f ~/dbw3/.locks/gpu.lock
rm -rf ~/dbw3/$EXPREL/*_vel_hw_v1_thermal_ft
echo "$(date -Is) END vel_hw_v1_cad_ft rc=stopped_for_thermal" >> ~/runs/gpu3.log

for g in 2 3; do tar -xzf ~/hw_patch.tgz -C ~/dbw$g; done

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
name=vel_hw_v1_thermal_ft
dst=~/dbw3/$EXPREL/${STAMP}_$name
mkdir -p "$dst"
cp "$SRC_CK" "$dst/"
echo "{\"warm_start_from\": \"$SRC_CK\", \"actuator_profile\": \"hw_v1\", \"thermal_penalty\": -2e-4}" > "$dst/warm_start.json"
cat > ~/runs/queue3.sh <<EOF
#!/usr/bin/env bash
source ~/runs/env.sh
export CUDA_VISIBLE_DEVICES=3 G=3
mkdir -p ~/dbw3/logs/brev
cd ~/dbw3 && echo "\$(date -Is) START $name" >> ~/runs/gpu3.log && python -u tools/run_chunked_locomotion.py --run_name $name --resume_run ${STAMP}_$name --chunks 80 --iters_per_chunk 200 --until_iteration 12000 --num_envs \$CLIP_ENVS --solver_iters \$SPOS 4 --save_interval 100 --timeout 7200 --first_wait_minutes 5 --wait_minutes 5 --gap_s 5 --actuator_profile hw_v1 --thermal_penalty=-2e-4 --isaac_python python --log_dir logs/brev --owner brev_gpu3 > logs/brev/${name}_driver.log 2>&1; echo "\$(date -Is) END $name rc=\$?" >> ~/runs/gpu3.log
echo "\$(date -Is) QUEUE_DONE" >> ~/runs/gpu3.log
EOF
chmod +x ~/runs/queue3.sh
tmux new-session -d -s gpu3 "bash ~/runs/queue3.sh"
sleep 2
tmux ls
echo "swapped gpu3 $(date -Is)"
