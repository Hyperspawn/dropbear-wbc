#!/bin/bash
# Evaluate cloud session 3 (vendor CEM-60 knee, mass DR) locally once the watchdog has pulled the runs into
# logs/brev/remote_s3/. Latest checkpoint of each run, motor twin hw_v2ie / hw_v2i, 32/4 solver, telemetry + gate.
# Output: logs/hw_twin/session3/<tag>/ (+ library per-clip summaries in each rollout.log).
set -u
cd "$(dirname "$0")/../.."
ISAACPY=$(PYTHONPATH=source python -c "from dropbear_wbc import paths; print(paths.isaac_python())")
export DROPBEAR_CALIBRATION_JSON=data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json
R=logs/brev/remote_s3
OUT=logs/hw_twin/session3
latest() {  # run-name suffix -> newest model_*.pt of the newest matching run dir
  local d; d=$(ls -d $R/dbw*/logs/rsl_rl/*/*_"$1" 2>/dev/null | tail -1)
  [ -n "$d" ] && ls "$d"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2
}
run() {  # tag, then play.py arguments
  local tag=$1; shift; local D=$OUT/$tag; mkdir -p "$D"
  [ -f "$D/rollout.log" ] && grep -q play_summary "$D/rollout.log" && { echo "== $tag: done"; return; }
  echo "== $(date -Is) START $tag"
  python tools/gpu_lock_run.py --owner session3_eval --log "$D/rollout.log" --timeout 3600 -- $ISAACPY -u "$@" --headless
  [ -f "$D/telemetry.npz" ] && python tools/telemetry_issue_scan.py "$D/telemetry.npz" --md "$D/issue_scan.md" | tail -1
}
V7=data/motions_v6ts/libraries/accepted_v7gen.json
for name in lib_v8_v2ie lib_v8_v2ie_nokstop; do
  ck=$(latest $name); [ -z "$ck" ] && { echo "no $name"; continue; }
  run ${name}_library scripts/play.py --task Dropbear-Tracking-Library-NoState-Play-v0 --motion_library $V7 --checkpoint "$ck" \
    --actuator_profile hw_v2ie --num_envs 138 --steps 1700 --solver_iters 32 4 --continuous_loop
done
for clip in kimodo_gen/squat_ts1.19_v6 kimodo_gen/half_squat_ts1.19_v6 kimodo_gen/lunge_ts1.19_v6 \
            asap_g1/motions_raw_tairantestbed_smpl_video_squat_level1_filter_amass_ts1.19_v6; do
  for name in lowbody_v2ie lib_v8_v2ie lib_v8_v2ie_nokstop; do
    ck=$(latest $name); [ -z "$ck" ] && continue
    tag=$(basename $clip | cut -c1-24)_$name
    run $tag scripts/play.py --task Dropbear-Tracking-Flat-NoState-Play-v0 --motion_file data/motions_v6ts/$clip.npz \
      --checkpoint "$ck" --actuator_profile hw_v2ie --num_envs 1 --steps 600 --solver_iters 32 4 \
      --telemetry $OUT/$tag/telemetry.npz --record_rollout $OUT/$tag/rollout.npz
  done
done
ck=$(latest kimodo_walk_v2ie) && [ -n "$ck" ] && run kimodo_walk_v2ie scripts/play.py --task Dropbear-Tracking-Flat-NoState-Play-v0 \
  --motion_file data/motions_cyclic/kimodo_walk_ts119_cyclic.npz --checkpoint "$ck" --actuator_profile hw_v2ie \
  --num_envs 1 --steps 1000 --solver_iters 32 4 --telemetry $OUT/kimodo_walk_v2ie/telemetry.npz \
  --record_rollout $OUT/kimodo_walk_v2ie/rollout.npz
ck=$(latest vel_rough_v2i) && [ -n "$ck" ] && run vel_rough_v2i scripts/play_locomotion.py --checkpoint "$ck" --steps 1000 \
  --telemetry $OUT/vel_rough_v2i/telemetry.npz
echo "== SESSION3_EVAL_DONE $(date -Is)"
