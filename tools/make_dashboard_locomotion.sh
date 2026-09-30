#!/bin/bash
# Dashboard video of a velocity / terrain walker: clean video + per-motor telemetry (play_locomotion.py), then the
# dashboard panel and the hardware gate. The terrain the run was trained on is replayed (run_info 'terrain').
# usage: bash tools/make_dashboard_locomotion.sh <checkpoint.pt> <actuator profile> <tag> [steps=1000]
#   e.g. bash tools/make_dashboard_locomotion.sh assets/policies/vel_rough_v1i_b/model_5500.pt hw_v1i rough_terrain
# Output: logs/dashboards/<tag>*. Needs Isaac Sim (docs/INSTALL.md tier 2).
set -u
cd "$(dirname "$0")/.."
CK=$1; P=$2; TAG=$3; STEPS=${4:-1000}
ISAACPY=$(PYTHONPATH=source python -c "from dropbear_wbc import paths; print(paths.isaac_python())")
D=logs/dashboards; mkdir -p "$D"
python tools/gpu_lock_run.py --owner dashboard --log "$D/${TAG}_sim.log" --timeout 2400 -- \
  $ISAACPY -u scripts/play_locomotion.py --headless --checkpoint "$CK" --actuator_profile "$P" \
  --solver_iters 32 4 --clean_video --video_cams track --steps "$STEPS" --telemetry "$D/${TAG}_telemetry.npz" \
  --video_caption "Dropbear | walking on datasheet motors ($P) | $TAG" --out "$D/${TAG}_sim.json"
echo "== sim rc=$?"
MP4=$(python -c "import json;print(json.load(open('$D/${TAG}_sim.json'))['clean_video']['outputs']['track']['mp4'])")
python tools/render_actuator_dashboard.py --video "$MP4" --telemetry "$D/${TAG}_telemetry.npz" --out "$D/${TAG}_dashboard.mp4"
echo "== dashboard rc=$? -> $D/${TAG}_dashboard.mp4"
python tools/telemetry_issue_scan.py "$D/${TAG}_telemetry.npz" --md "$D/${TAG}_issue_scan.md" | tail -3
