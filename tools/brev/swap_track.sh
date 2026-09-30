#!/usr/bin/env bash
# Runs ON the Brev instance. Fine-tune a trained TRACKING policy on the real-actuator profile hw_v1.
# Usage: bash swap_track.sh <gpu> <run_name> <src_box> <src_exp> <src_run_dir> <task> <motion_flag> <motion_path> <envs>
#   e.g. swap_track.sh 0 kimodo_walk_hw 3 dropbear_tracking 2026-09-24_22-44-20_kimodo_walk_v4 \
#        Dropbear-Tracking-Flat-v0 --motion_file data/motions/kimodo_g1/output_walk_v4.npz 4096
set -u
source ~/runs/env.sh
G=$1; name=$2; SB=$3; SEXP=$4; SRUN=$5; TASK=$6; MFLAG=$7; MPATH=$8; ENVS=$9; PROF=${10:-hw_v1}; EXTRA=${11:-}
if [ "$SRUN" != "none" ]; then
  SRC=~/dbw$SB/logs/rsl_rl/$SEXP/$SRUN
  CK=$(ls $SRC/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
  [ -f "$CK" ] || { echo "no checkpoint in $SRC"; exit 1; }
  STAGE=~/stage_$name; rm -rf $STAGE; mkdir -p $STAGE/$SRUN
  rsync -a --exclude 'model_*.pt' --exclude 'videos' $SRC/ $STAGE/$SRUN/ && cp "$CK" $STAGE/$SRUN/
  echo "source: $CK"
else
  echo "source: none (from scratch)"
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
DEXP=~/dbw$G/logs/rsl_rl/$SEXP; mkdir -p $DEXP
INITARG=""
if [ "$SRUN" != "none" ]; then
  INIT=${SRUN}_init4hw
  rm -rf $DEXP/$INIT; mv $STAGE/$SRUN $DEXP/$INIT; rm -rf $STAGE
  INITARG="--init_run $INIT"
fi
cat > ~/runs/queue$G.sh <<EOQ
#!/usr/bin/env bash
source ~/runs/env.sh
export CUDA_VISIBLE_DEVICES=$G G=$G
mkdir -p ~/dbw$G/logs/brev
cd ~/dbw$G && echo "\$(date -Is) START $name" >> ~/runs/gpu$G.log && python -u tools/run_chunked_training.py --task $TASK $MFLAG $MPATH --run_name $name $INITARG --chunks 40 --iters_per_chunk 200 --num_envs $ENVS --solver_iters \$SPOS 4 --save_interval 100 --seed 31 --timeout 7200 --gap_s 5 --log_dir logs/brev --owner brev_gpu$G --isaac_python python --calibration \$CAL --actuator_profile $PROF $EXTRA --restore_train_state off > logs/brev/${name}_driver.log 2>&1; echo "\$(date -Is) END $name rc=\$?" >> ~/runs/gpu$G.log
echo "\$(date -Is) QUEUE_DONE" >> ~/runs/gpu$G.log
EOQ
chmod +x ~/runs/queue$G.sh
tmux new-session -d -s gpu$G "bash ~/runs/queue$G.sh"
sleep 2; tmux ls; echo "swapped gpu$G -> $name ($INITARG) $(date -Is)"
