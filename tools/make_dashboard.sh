#!/bin/bash
# Glitch-free dashboard video of a tracking policy (docs/ISSUES.md #8, #9):
#   1. physics rollout on the motor twin, recording every link pose + per-motor telemetry (play.py)
#   2. kinematic re-render of the recorded state (render_reference.py: no RTX annotator glitch)
#   3. per-motor torque / speed / contact panel beside the video (render_actuator_dashboard.py)
#   4. the hardware gate on the telemetry (telemetry_issue_scan.py)
# usage: bash tools/make_dashboard.sh <checkpoint.pt> <motion.npz> <actuator profile> <tag> [steps=1000] [task]
#   e.g. bash tools/make_dashboard.sh assets/policies/kimodo_ts119_allfix_smooth/model_5800.pt \
#          data/motions_cyclic/kimodo_walk_ts119_cyclic.npz hw_v1ie kimodo_walk 1000
# Output: logs/dashboards/<tag>/ and logs/dashboards/<tag>_dashboard.mp4. Needs Isaac Sim (docs/INSTALL.md tier 2).
set -u
cd "$(dirname "$0")/.."
CK=$1; NPZ=$2; P=$3; TAG=$4; STEPS=${5:-1000}; TASK=${6:-Dropbear-Tracking-Flat-NoState-Play-v0}
ISAACPY=$(PYTHONPATH=source python -c "from dropbear_wbc import paths; print(paths.isaac_python())")
export DROPBEAR_CALIBRATION_JSON=${DROPBEAR_CALIBRATION_JSON:-data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json}
D=logs/dashboards/$TAG; mkdir -p "$D"
python tools/gpu_lock_run.py --owner dashboard --log "$D/rollout.log" --timeout 2400 -- \
  $ISAACPY -u scripts/play.py --task "$TASK" --motion_file "$NPZ" --checkpoint "$CK" --solver_iters 32 4 \
  --actuator_profile "$P" --num_envs 1 --steps "$STEPS" --record_rollout "$D/rollout.npz" --telemetry "$D/telemetry.npz" --headless
echo "== rollout rc=$?"
python tools/gpu_lock_run.py --owner dashboard --log "$D/render.log" --timeout 2400 -- \
  $ISAACPY -u scripts/render_reference.py --motion_file "$D/rollout.npz" --video_cams track --video_dir "$D" \
  --video_tag "$TAG" --video_caption "Dropbear | $(basename "$NPZ" .npz) | motors: $P | $(basename "$CK")" --headless
echo "== render rc=$?"
python tools/render_actuator_dashboard.py --video "$D/${TAG}_track.mp4" --telemetry "$D/telemetry.npz" \
  --telemetry_offset 1 --layout side --out "logs/dashboards/${TAG}_dashboard.mp4"
echo "== dashboard rc=$? -> logs/dashboards/${TAG}_dashboard.mp4"
python tools/telemetry_issue_scan.py "$D/telemetry.npz" --md "$D/issue_scan.md" | tail -3
echo "== hardware gate: $D/issue_scan.md"
