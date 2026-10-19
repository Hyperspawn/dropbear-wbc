#!/bin/bash
# Score a tracking policy on the fidelity clip set (tools/fidelity_set_v1.json): one rollout per clip on the motor
# twin, then tools/motion_fidelity.py (did it do the motion?) and tools/telemetry_issue_scan.py (hardware gate).
# usage (Git Bash, repo root): bash tools/eval_fidelity_set.sh <model_*.pt> <actuator profile> <out_dir>
#   env PLAY_EXTRA: extra scripts/play.py arguments, e.g. PLAY_EXTRA="--stop_model" for policies trained with it
#   e.g. bash tools/eval_fidelity_set.sh logs/brev/remote_s6/dbw0/.../model_21000.pt hw_cem6 logs/fidelity/cem6_21000
set -u
cd "$(dirname "$0")/.."
CK=$1; P=$2; OUT=$3
mkdir -p "$OUT"
ISAACPY=$(PYTHONPATH=source python -c "from dropbear_wbc import paths; print(paths.isaac_python())")
export DROPBEAR_CALIBRATION_JSON=${DROPBEAR_CALIBRATION_JSON:-data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json}
PAIRS=()
while IFS=$'\t' read -r TAG NPZ STEPS; do
  if [ ! -f "$OUT/$TAG.npz" ]; then
    python tools/gpu_lock_run.py --owner fidelity --log "$OUT/$TAG.log" --timeout 900 -- \
      "$ISAACPY" -u scripts/play.py --task Dropbear-Tracking-Flat-NoState-Play-v0 --motion_file "$NPZ" --checkpoint "$CK" \
      --actuator_profile "$P" --solver_iters 32 4 --num_envs 1 --steps "$STEPS" --record_rollout "$OUT/$TAG.npz" \
      --telemetry "$OUT/${TAG}_telemetry.npz" ${PLAY_EXTRA:-} --headless > /dev/null 2>&1
    echo "$TAG rc=$?"
  fi
  PAIRS+=(--pair "$NPZ" "$OUT/$TAG.npz")
done < <(python -c "import json; [print(c['tag'], c['npz'], c['steps'], sep='\t') for c in json.load(open('tools/fidelity_set_v1.json'))['clips']]")
python tools/motion_fidelity.py "${PAIRS[@]}" --json "$OUT/fidelity.json" | tee "$OUT/fidelity.txt"
python tools/telemetry_issue_scan.py "$OUT"/*_telemetry.npz --md "$OUT/issue_scan.md" --json "$OUT/issue_scan.json" > /dev/null
python - "$OUT" <<'PY'
import json, sys
o = sys.argv[1]
scan = json.load(open(f"{o}/issue_scan.json"))
fails = {}
for r in scan:
    for f in r["hw_gate"]["fails"]:
        fails[f] = fails.get(f, 0) + 1
print(f"hardware gate: {sum(r['hw_gate']['pass'] for r in scan)} of {len(scan)} pass; falls {sum(r['body'].get('falls', 0) for r in scan)}")
print("most common gate failures:", sorted(fails.items(), key=lambda x: -x[1])[:8])
PY
