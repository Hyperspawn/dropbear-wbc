# Dropbear tabletop -> Isaac-GR00T N1.7 (NEW_EMBODIMENT)

The "autonomous useful task" path: a base-fixed Dropbear at a table in Isaac Lab 2.2, a scripted baseline and teleop
data, a dataset in Isaac-GR00T's LeRobot v2 layout, and the plan for a GR00T N1.7 NEW_EMBODIMENT fine-tune on Brev
A100s followed by closed-loop evaluation in the same simulated scene. Evidence lives in `logs/tabletop/` (progress:
`logs/tabletop/PROGRESS.md`); every claim below cites a log or is marked **UNVERIFIED**. Task choice: docs/DECISIONS.md
(2026-09-24, push without a gripper).

Reference code (read-only): Isaac-GR00T @ `51d4c89f` (`$DROPBEAR_UPSTREAM/Isaac-GR00T`:
`getting_started/{data_preparation,data_config,finetune_new_embodiment,policy,hardware_recommendation}.md`,
`gr00t/data/dataset/lerobot_episode_loader.py`, `gr00t/data/stats.py`, `gr00t/configs/data/embodiment_configs.py`,
`examples/SO100`, `demo_data/cube_to_bowl_5`) and unitree_sim_isaaclab (`tasks/g1_tasks/pick_place_redblock_g1_29dof_dex1`:
the base-fixed G1 tabletop pattern this scene follows).

## 0. Status at a glance (2026-09-24)

| item | status | evidence (`logs/tabletop/`) |
|---|---|---|
| scene builds (fixed base, table, block, zone, 20 Hz, closure-consistent NPZ reset, 3 cameras) | **VERIFIED** | `smoke_v2.*`, `record_train_v2_0.*` (`reset_info`: closure 0.18 mm, verdict accepted) |
| ego + wrist cameras produce correct images (no blank frames; zone rendered where physics has it) | **VERIFIED** after two fixes (camera inside the visor; zone render) | `camera_probe_v1/`, `record_train_v2_0.json` (0 blank; zone check 0.5-1.4 px), `media/push_v1_frames_grid.png` (looked at) |
| scripted baseline, 50 held-out placements | **VERIFIED: 42/50 = 84.0 %** (Wilson 95 % 71.5-91.7 %) | `baseline_heldout_v2.{log,json}`, `baseline_heldout_v2_report.json` |
| dataset `data/groot/dropbear_tabletop_push_v1/`: 34 successful scripted episodes (5303 frames, 20 Hz; train placements 0-39, 34/40 succeeded), ego 640x480 + 2 wrist views 320x240, per-episode instruction + success flag, 34 MB | **VERIFIED** (static checks + Isaac-GR00T's own loader/stats code @ 51d4c89f in a torch-free venv) | `logs/tabletop/groot_validate_dropbear_tabletop_push_v1.{log,json}`, `build_dataset_push_v1.log`, `record_train_v2_0.json`, `record_train_v3_16.json` |
| teleop path into the same env, controller and recorder (`--policy teleop_keys`: `KeyboardSource` driven by a fixed key script -> torso-frame wrist targets -> `DropbearArmIK` -> semantic arm angles) | **VERIFIED as plumbing** (not a task solution): the left wrist moved (+12.6, +6.2, +11.7) cm for keys asking (+12, +6, +12) cm, IK residual 0, steady-state tracking ~4 mm, other arm held, head video + zone check (1.5 px) recorded in the same raw format | `teleop_keys_v1.{log,json}`, `data/groot/_raw/teleop_keys_plumbing_v1/` |
| human teleop episodes, domain randomization, language variety | NOT DONE | section 7 |
| GR00T N1.7 fine-tune on Brev, open-loop and closed-loop evaluation | **UNVERIFIED** (plan only; the sim client `--policy groot` is not written) | sections 5, 6 |


## 1. Task and scene (`source/dropbear_wbc/tasks/tabletop/`, gym id `Dropbear-Tabletop-Push-v0`)

**Task.** Push a red 5 cm block into a green 10 cm target square with the hand body. Instruction (one task):
`"push the red block onto the green target"`. Dropbear has no gripper, and none is simulated (DECISIONS 2026-09-24).

| item | value | source |
|---|---|---|
| robot | contract plant (`make_dropbear_cfg`, CONTRACTS 0.1-0.3), `fix_root_link`, root at env (0, 0, -0.12) m: the standing root height (-0.159 m) + 4 cm, so the legs hang clear of the ground | `layout.py` |
| reset state | FULL settled joint row of `data/motions/smoke/dropbear_static_stand.npz` frame 0 (motors + 63 passive DOFs + neck), fail-closed checks as the velocity task (CONTRACTS 1: never write torn closures) | `mdp.reset_tabletop` |
| legs, neck | held at that row by their PD (not in the action space) | |
| arm controller | per-motor PD with the teleop SDK gains (shoulders 200/5, elbow motor 600/10, wrist 60/2; 40 N*m), plus the teleop gravity feed-forward (`teleop.gravity.ArmGravity`) as an effort target | `tabletop_env_cfg.py`, `scripts/tabletop_collect.py` |
| control | 20 Hz (sim dt 5 ms, decimation 10), PhysX 32/4 iterations, episode 20 s | |
| table | static slab 0.45 x 1.30 x 0.04 m, top at root z 1.25 m (env z 1.13 m), front edge root x 0.165 m (2.4 cm in front of the torso collider), centred on the torso midline (root y -0.0697); friction 0.5 / 0.4 | `workspace.log`, `geometry_probe.log` |
| block | dynamic cube 5 cm, 0.10 kg, friction 0.5 / 0.4, red | |
| target zone | visual-only 10 x 10 cm green square (2 mm, no physics: the block never has to climb onto it), moved per reset by USD xform ops; its pose is `env.tabletop_zone` | `mdp._author_zone_prims` |
| success | block centre inside the zone shrunk by 1 cm, on the table (z within 1 cm), upright (< 10 deg), still (< 3 cm/s, < 0.5 rad/s), held 0.5 s -> termination `success` | `mdp.block_in_zone_stable` |
| failure | block below the table top - 5 cm or outside the table (+2 cm) -> `block_dropped`; 20 s -> `time_out` | `mdp.block_dropped` |

**Placements** (`tools/tabletop_placements.py` -> `data/groot/placements/tabletop_push_v1.json`, `logs/tabletop/placements_v1.log`).
A placement = arm side, zone centre, block centre 8.5-13 cm from it in a random direction, block yaw +-45 deg. The two
arms' pushable regions are disjoint, so each placement belongs to one arm (side drawn 50/50). A candidate is kept only
if the scripted policy's whole path is kinematically feasible for that arm (tool IK residual < 2.5 mm, hand lowest point
within 3 mm of the request, elbow >= 5 cm above the table while pushing, no shoulder/elbow joint at a limit, block and
zone on the table). **Splits:** 400 `train` and 50 `heldout` placements from disjoint seed streams
(`np.random.default_rng(1_000_000 + k)` vs `9_000_000 + k`); the held-out split is never used for data collection.

**Cameras** (`cameras.py`; Isaac Lab `TiledCamera`, RGB):

| key | mount | resolution | notes |
|---|---|---|---|
| `head` -> GR00T `ego_view` | rigid child of the anchor body `head_5mm_ujoint_base__5__1` (fixed to the torso, CONTRACTS 5.1), 1.2 cm in front of the visor at eye height: root (0.135, -0.07, 1.88) m, pitched 68 deg down, 92 deg horizontal FOV | 640 x 480 | DECISIONS 2026-09-24; the head link itself sits on the PD-held Stewart neck, the anchor gives a steadier view |
| `left_wrist` / `right_wrist` (optional) | child of the hand body, 6 cm off the hand axis, looking along the hand | rendered 640 x 480, stored 320 x 240 | rendering them at 320 x 240 coincided with intermittent blank frames |
| `scene` (demo only) | fixed third-person | 640 x 480 | never in the dataset |

## 2. Scripted baseline (`tasks/tabletop/scripted.py`)

Privileged (true block and zone poses), one arm. Cartesian goals for a TOOL point on the hand axis 13 cm from the wrist
(`kinematics.TabletopArmIK`: the teleop calibrated arm chains with the end effector moved along the hand, solved by the
teleop priority IK `solve_chain_robust`; hand axis preferred 60 deg forward of vertical). Phases: LIFT -> TRANSIT (hand's
lowest point 7.5 cm above the table = 2.5 cm above the block top, 4 cm behind the block's trailing face opposite the
zone) -> DESCEND (lowest point 8 mm above the table) -> PUSH (closed loop: the target leads the modelled contact by 8 mm
along the block->zone direction, re-aimed every step, 7 cm/s; the lead grows up to 4 cm while the block does not move;
re-approach from above if the hand leaves the push line, max 3) -> RETREAT -> RISE -> BACK -> HOME. TRANSIT / DESCEND /
PUSH start only when the MEASURED tool point has reached the commanded one (12 mm, or after a timeout). A horizontal
outer-loop integrator on the measured tool point removes position-control sag. Semantic arm angles are mapped to the 10
arm motors by the calibration `SemanticMap` (`DropbearArmIK.to_motor_fast`).

**Result (VERIFIED, the bar GR00T must beat): 42 / 50 held-out placements = 84.0 % (Wilson 95 % 71.5-91.7 %)**;
left arm 24 / 29, right arm 18 / 21; time to success mean 8.4 s, median 7.5 s, max 15.7 s (20 s budget). Command:
`logs/tabletop/run_record_then_baseline_v2.sh` (second job): 8 envs, no cameras, PhysX 32/4, 20 Hz, all 50 held-out
placements once; evidence `logs/tabletop/baseline_heldout_v2.{log,json}`, `baseline_heldout_v2_report.json`
(`tools/tabletop_report.py`), raw low-dim episodes `data/groot/_raw/baseline_heldout_v2/`.

- **Failure mode.** All 8 failures time out stalled in PUSH, and 7 of them push OUTWARD (block->zone direction pointing
  away from the body midline, often also back toward the robot): outward pushes (outward component > 0.5) succeed
  5 / 12, all other directions 37 / 38. The hand has to work on the block's inner side there, near the shoulder-roll
  (adduction) limit, and stalls a few cm short of the modelled contact; the kinematic placement check does not see it.
- **Train split** (with cameras): placements 0-15 14 / 16 (`record_train_v2_0.json`; identical episode by episode to the
  pre-zone-fix run `record_train_v1_0.json`: the sim is deterministic, the zone is visual only), 16-39 20 / 24
  (`record_train_v3_16.json`); 34 / 40 = 85 % in total, consistent with the held-out 84 %.
- **Not the baseline:** the kinematic dry run (`tools/tabletop_dryrun.py`, perfect tracking, no contact physics) solves
  50 / 50 held-out placements (`logs/tabletop/dryrun_kinematic_heldout_v2.log`); physics costs 16 points.
- **The baseline is privileged**: it reads the true block and zone poses. GR00T gets the cameras, the arm state and the
  instruction only.
- **How it got there** (all on TRAIN placements or the 2 hand-made smoke placements, never tuned on the held-out split;
  `logs/tabletop/PROGRESS.md`): the first GPU smoke went 0 / 2 because phase changes used the commanded instead of the
  measured tool point (the lagging hand landed on the block) and the outer-loop integrator acted in z (it pressed the
  hand onto the table); fixed by measured-convergence gates, a horizontal-only integrator, a 4 cm stand-off, a push
  lead that grows while the block is stalled, and the teleop controller's dq* velocity feed-forward.
- Speed: 0.30 s wall per 20 Hz step at 8 envs without cameras (CPU IK-bound); 0.42 s per step at 4 envs with the three
  640 x 480 cameras rendered and JPEG-buffered.

## 3. Embodiment / data contract `dropbear-groot-tabletop-v1`

Dataset: `data/groot/<name>/` (first: `data/groot/dropbear_tabletop_push_v1/`), built by `tools/build_groot_dataset.py`
from raw episodes (`data/groot/_raw/<name>/<split>/<pid>_<policy>/`: `meta.json`, `lowdim.npz`, one mp4 per camera;
written by `scripts/tabletop_collect.py --record`, format `tasks/tabletop/episode_io.py`). Kit python has no
pandas/pyarrow, so Isaac writes the raw episodes and system Python 3.12 (pandas 2.2, pyarrow 21) builds the dataset.

Layout (Isaac-GR00T `getting_started/data_preparation.md`, "GR00T-flavoured LeRobot v2"):

| path | content |
|---|---|
| `meta/info.json` | `codebase_version` v2.1, `fps` 20, `chunks_size` 1000, `data_path`, `video_path`, `features` (every parquet column + one `video` feature per view) |
| `meta/episodes.jsonl` | `{"episode_index", "tasks": [instruction], "length", "dropbear": {pid, split, side, success, termination, policy}}` |
| `meta/tasks.jsonl` | `{"task_index": 0, "task": "push the red block onto the green target"}` |
| `meta/modality.json` | `state` / `action`: `left_arm` [0:5], `right_arm` [5:10]; `video`: `ego_view` (+ `left_wrist_view`, `right_wrist_view`), `original_key` `observation.images.<view>`; `annotation`: `human.task_description` -> column `task_index` |
| `meta/stats.json` | mean/std/min/max/q01/q99 of every float feature (the formula of `gr00t/data/stats.py`; GR00T recomputes and fingerprints it at fine-tune time anyway) |
| `meta/episodes_stats.jsonl` | LeRobot v2.1 per-episode statistics |
| `meta/dropbear_tabletop.json` | provenance: raw episode dirs, placements file + SHA-256, calibration path + SHA-256, USD SHA, layout, push parameters |
| `data/chunk-000/episode_NNNNNN.parquet` | one row per 20 Hz control step |
| `videos/chunk-000/observation.images.<view>/episode_NNNNNN.mp4` | H.264 (libx264, yuv420p, CRF 18; torchcodec-decodable), 20 fps, exactly one frame per parquet row (checked by the builder and the validator) |
| `dropbear_tabletop_config.py` | the GR00T modality config below (copy of `source/dropbear_wbc/groot/dropbear_tabletop_config.py` with `VIDEO_KEYS` = the views this dataset has) |

Parquet columns:

| column | shape | meaning |
|---|---|---|
| `observation.state` | 10 float32 | MEASURED semantic arm angles [rad], CONTRACTS 2 names and signs: `left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll`, then `right_*` (elbow = G1 convention) |
| `action` | 10 float32 | COMMANDED absolute semantic arm angles [rad] for this step (same order); the env maps them to the 10 arm motors with the calibration `SemanticMap` |
| `timestamp` | float32 | `frame_index / fps` exactly (LeRobot's 1e-4 s check); the sim clock is `time.sim_s` |
| `frame_index`, `episode_index`, `index`, `task_index` | int64 | LeRobot indices (`index` global and contiguous) |
| `annotation.human.task_description` | int64 | = `task_index` (resolved through `tasks.jsonl`) |
| `next.reward`, `next.done` | float32, bool | 1.0 on the last frame of a successful episode; `done` on the last frame |
| extras (not used by the GR00T config) | | `observation.motor_q` / `motor_dq` (22, motor contract order), `action.motor_q` (10 arm motor targets), `action.tau_ff` (10, gravity feed-forward), `observation.block_pose` (7: xyz + wxyz, robot root frame, privileged), `observation.zone_pos` (3), `observation.{left,right}_hand_pose` (7), `task.success_now`, `time.sim_s` |

Frame alignment: row `t` holds the images and state observed at control step `t` and the action commanded at step `t`
(applied from `t` to `t+1`). The first 6 steps after a reset (0.3 s settle + re-render) are not recorded. The inactive
arm's action is the calibrated standing pose, so the policy has to infer which arm to use from the images.

Dataset `dropbear_tabletop_push_v1` = two collection runs: train placements 0-15 (`record_train_v2_0`, 14/16, env
spacing 3 m) and 16-39 (`record_train_v3_16`, 20/24, 8 m). The wrist views can show a neighbouring env's robot in the
background (small at 8 m); a collection with `--num_envs 1` or a larger spacing avoids it.

GR00T modality config (`dropbear_tabletop_config.py`, registered under `EmbodimentTag.NEW_EMBODIMENT`, modelled on
`examples/SO100/so100_config.py`):

| modality | keys | delta indices | notes |
|---|---|---|---|
| video | `ego_view` (+ `left_wrist_view`, `right_wrist_view` when built with wrist views) | [0] | uint8 RGB |
| state | `left_arm`, `right_arm` | [0] | |
| action | `left_arm`, `right_arm` | 0..15 (16 steps = 0.8 s at 20 Hz) | `RELATIVE`, `NON_EEF`, `DEFAULT` for both arms (delta from the current state; GR00T converts back to absolute targets at inference, `state_action_processor.unapply_action`) |
| language | `annotation.human.task_description` | [0] | |

Validation: `tools/validate_groot_dataset.py <dataset> [--groot_repo $DROPBEAR_UPSTREAM/Isaac-GR00T]`: static checks of
the layout, then (in `.venv-groot-check`: numpy 1.26.4, pandas 2.2.3, scipy 1.15.3, pyarrow, PyAV; no torch) Isaac-GR00T's
own `LeRobotEpisodeLoader` on every episode with this modality config, `get_dataset_statistics`, and GR00T's
`generate_stats` + `generate_rel_stats` on a scratch copy. torchcodec needs torch, so the loader's
`get_frames_by_indices` is replaced by a PyAV decoder with the same contract (frames at indices, NHWC uint8). Nothing
is written into the upstream clone (`sys.dont_write_bytecode`, `git status` clean).

## 4. Collecting more data

All GPU commands go through the team lock (`tools/gpu_lock_run.py`, CONTRACTS 0; keep one job <= 15 min, so split
long collections with `--start` / `--episodes`). Raw episodes of one dataset go under `data/groot/_raw/<name>/`.

**Scripted (privileged) demonstrations** on the TRAIN placements (never the held-out split):

```bash
python tools/gpu_lock_run.py --owner tabletop --log logs/tabletop/collect_train_000.log --timeout 900 --wait-minutes 120 -- \
  C:/isaac-sim/python.bat -u scripts/tabletop_collect.py --headless --split train --start 0 --episodes 24 --num_envs 4 \
  --record data/groot/_raw/dropbear_tabletop_push_v1 --wrist_cams --summary logs/tabletop/collect_train_000.json
# next chunks: --start 24, 48, ... (400 train placements); every attempt is written, failed ones too (success flag)
```

Then build (system Python; successful episodes only unless `--include_failed`), validate and look at it:

```bash
python tools/build_groot_dataset.py --raw data/groot/_raw/dropbear_tabletop_push_v1 --split train \
    --out data/groot/dropbear_tabletop_push_v1 --force
.venv-groot-check/Scripts/python.exe -B tools/validate_groot_dataset.py data/groot/dropbear_tabletop_push_v1 \
    --groot_repo $DROPBEAR_UPSTREAM/Isaac-GR00T --out logs/tabletop/groot_validate_dropbear_tabletop_push_v1.json
python tools/tabletop_media.py grid data/groot/_raw/dropbear_tabletop_push_v1/train/<pid>_scripted \
    --out logs/tabletop/media/<x>.png
```

More placements: `python tools/tabletop_placements.py --train N --heldout 50 --out data/groot/placements/tabletop_push_v2.json`
(CPU; the held-out seeds do not depend on `--train`).

**Teleop demonstrations.** `scripts/tabletop_collect.py --policy teleop_keyboard` drives the same env through the teleop
stack (`teleop.devices.KeyboardSource` -> wrist targets in the torso frame -> `DropbearArmIK` -> semantic arm angles ->
the same recording path), so teleop and scripted episodes share one format and one controller. `--policy teleop_keys`
replays a fixed key sequence headless as a plumbing check (not a task solution). **UNVERIFIED with a human:** the
msvcrt keyboard needs an interactive console and the Isaac run is headless here; a live operator view needs the WebXR
path of docs/TELEOP.md (Vuer) plus a camera stream to the operator, which does not exist yet (section 7). Teleop
episodes must use TRAIN placements, so the held-out comparison stays clean.

## 5. Fine-tuning GR00T N1.7 on Brev (PLAN, UNVERIFIED: nothing has run on Brev)

Why Brev: GR00T N1.7 is a ~3B-parameter model; Isaac-GR00T `getting_started/hardware_recommendation.md` asks for >= 40 GB
of VRAM for fine-tuning (the default recipe tunes the projector + diffusion action head and stays under ~35 GB per GPU;
`--tune-llm` / `--tune-visual` need 80 GB+). The local RTX 4080 Laptop has 12 GB. Agents never run brev commands; the
user runs `brev login` and creates the instance (docs/BREV.md).

1. **Instance.** 1x A100 80 GB (first choice: headroom above the ~35-40 GB peak at batch 32 plus video-decode workers).
   An A100 40 GB is marginal (lower `--global-batch-size`, e.g. 16, with `--gradient-accumulation-steps 2`).
   Ubuntu 22.04, CUDA 12.8 driver, FFmpeg 4-7 (torchcodec 0.8.0 cannot load FFmpeg 8), >= 100 GB disk.
2. **Isaac-GR00T** at the commit the dataset was validated against:

   ```bash
   sudo apt-get install -y git-lfs ffmpeg && git lfs install
   git clone https://github.com/NVIDIA/Isaac-GR00T.git && cd Isaac-GR00T && git checkout 51d4c89f
   curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync --python 3.12   # includes flash-attn
   uv run python -c "import gr00t; print('ok')"
   ```
3. **Data** (WSL, the user runs it; docs/BREV.md section 4 conventions; `<instance>` e.g. `dropbear-training`):

   ```bash
   cd data/groot && tar czf ../groot_push_v1.tgz dropbear_tabletop_push_v1
   brev copy ../groot_push_v1.tgz <instance>:~/groot_push_v1.tgz
   # on the instance: mkdir -p ~/data && tar xzf ~/groot_push_v1.tgz -C ~/data
   ```
4. **Smoke** (proves the config, loader and stats on the box in a few minutes):

   ```bash
   uv run python gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B \
     --dataset-path ~/data/dropbear_tabletop_push_v1 --embodiment-tag NEW_EMBODIMENT \
     --modality-config-path ~/data/dropbear_tabletop_push_v1/dropbear_tabletop_config.py \
     --num-gpus 1 --output-dir ~/runs/push_v1_smoke --max-steps 50 --save-steps 50 --global-batch-size 8 \
     --dataloader-num-workers 4
   ```
   The dataset's copy of the config lists exactly the views it was built with (the builder writes `VIDEO_KEYS` from
   `meta/modality.json`).
5. **Fine-tune** (the upstream SO100 recipe; 16-step RELATIVE joint chunks = 0.8 s at 20 Hz):

   ```bash
   uv run python gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B \
     --dataset-path ~/data/dropbear_tabletop_push_v1 --embodiment-tag NEW_EMBODIMENT \
     --modality-config-path ~/data/dropbear_tabletop_push_v1/dropbear_tabletop_config.py \
     --num-gpus 1 --output-dir ~/runs/push_v1 --max-steps 10000 --save-steps 1000 --save-total-limit 5 \
     --global-batch-size 32 --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
     --dataloader-num-workers 8
   ```
   GR00T writes `meta/stats.json` fingerprints and `meta/relative_stats.json` on first use (rank 0), so the dataset
   directory must be writable. Budget (ESTIMATE, not measured): plan ~2-4 A100-hours for 10k steps at batch 32.
6. **Open-loop check**:

   ```bash
   uv run python gr00t/eval/open_loop_eval.py --dataset-path ~/data/dropbear_tabletop_push_v1 \
     --embodiment-tag NEW_EMBODIMENT --model-path ~/runs/push_v1/checkpoint-10000 --traj-ids 0 1 2 \
     --execution-horizon 16 --steps 400 --modality-keys left_arm right_arm --save-plot-path ~/runs/push_v1/ol
   ```
   Pass criteria (upstream "Is my fine-tune working?"): predicted curves track the ground truth on training
   trajectories, and the MSE falls across checkpoints. Open-loop error does not predict task success; section 6 does.

## 6. Closed-loop evaluation in sim via the GR00T policy server (PLAN, UNVERIFIED)

The bar: the scripted baseline's held-out success rate (section 2). GR00T must beat it on the same 50 held-out
placements, same scene, controller and success rule, **without privileged state** (cameras + arm state + instruction).

- **Server** (GPU box: the Brev instance; GR00T inference needs >= 16 GB and the laptop GPU runs Isaac):
  `uv run python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT --model-path <ckpt> --device cuda:0
  --host 0.0.0.0 --port 5555` (ZeroMQ REP + msgpack; `getting_started/policy.md`).
- **Sim client** (TO BE WRITTEN: `scripts/tabletop_collect.py --policy groot --groot_host <h> --groot_port 5555`, same
  loop as `scripted`): per control step build `{"video": {"ego_view": uint8 (B, 1, 480, 640, 3), ...}, "state":
  {"left_arm": float32 (B, 1, 5), "right_arm": ...}, "language": {<key>: [[instruction]] * B}}` from the env cameras and
  `motor_to_semantic_arms_fast` of the measured motors (check the language key with `PolicyClient.get_modality_config()`),
  call `PolicyClient.get_action`, and execute the returned ABSOLUTE semantic chunk (`left_arm` / `right_arm`, (B, 16, 5);
  GR00T converts its RELATIVE output back with the observed state) for the first 8 steps (re-plan at 2.5 Hz) through the
  same `SemanticMap` motor map, gravity feed-forward and dq* feed-forward as the demonstrations. The sim is synchronous
  (it waits for the server), so latency costs wall time, not success.
- **Transport:** an SSH tunnel from the laptop (`ssh -L 5555:localhost:5555 <instance>`), or Isaac on the instance too
  (Isaac Sim 5.0 + Lab 2.2 per docs/BREV.md section 3; an L40S has RT cores for the cameras).
- **Protocol:** all 50 held-out placements, 20 s budget; report the success rate with a Wilson 95 % interval next to
  the baseline's, the termination breakdown, per-side rates and the first 10 episodes' mp4s (`--record`). Then
  ablations: without wrist views; and ego view only vs ego + wrists.

## 7. What is missing

- **The GR00T sim client** (`--policy groot`, section 6) and every Brev step (fine-tune, open-loop, closed-loop):
  UNVERIFIED, nothing has run.
- **More, more diverse data.** One task, one instruction, one block, one table, fixed lighting. Domain randomization
  (block colour/size/mass/friction, table colour, light intensity/direction, camera pose jitter) is not implemented.
- **Human teleop episodes** (section 4): the path exists, a human-operated recording does not.
- **Language variety.** With one instruction GR00T cannot learn to use language; a second phrasing or task is needed.
- **Whole-body.** Base fixed, legs held: a standing/WBC policy is needed before the robot can work at a real table
  (docs/TELEOP.md section 9). **Hardware:** the head-camera mount is a simulation choice (section 1), to be matched to
  the real robot's sensor and calibrated.
- **Gripper / hand.** None (DECISIONS 2026-09-24): only non-prehensile tasks until Dropbear gets an end effector.

## 8. Files

| path | role |
|---|---|
| `source/dropbear_wbc/tasks/tabletop/layout.py` | geometry, success-rule constants, placements (train / held-out seeds) |
| `source/dropbear_wbc/tasks/tabletop/tabletop_env_cfg.py`, `mdp.py`, `config/__init__.py` | Isaac Lab scene / MDP, gym id `Dropbear-Tabletop-Push-v0` |
| `source/dropbear_wbc/tasks/tabletop/cameras.py` | camera mounts, head-image projection + zone render check helpers |
| `source/dropbear_wbc/tasks/tabletop/kinematics.py`, `scripted.py` | hand-tool IK on the teleop arm chains; scripted pusher |
| `source/dropbear_wbc/tasks/tabletop/episode_io.py` | raw episode writer (npz + JPEG-buffered mp4, kit python) |
| `source/dropbear_wbc/groot/dropbear_tabletop_config.py` | GR00T NEW_EMBODIMENT modality config (copied into each dataset with its views) |
| `scripts/tabletop_collect.py` | baseline evaluation / recording / teleop-device runs |
| `tools/tabletop_placements.py`, `tabletop_dryrun.py`, `tabletop_workspace.py`, `tabletop_geometry_probe.py` | CPU: placements, kinematic dry run, workspace and geometry measurements |
| `tools/tabletop_camera_probe.py` | GPU camera pose / blank-frame probe |
| `tools/build_groot_dataset.py`, `validate_groot_dataset.py` | dataset build (system Python) and validation (+ Isaac-GR00T loader in `.venv-groot-check`) |
| `tools/tabletop_report.py`, `tabletop_media.py` | success report (Wilson CI), frame grids / clips |
| `tests/test_tabletop_cpu.py` | CPU tests (success rule, kinematic rollout, build + validate, zone render check) |
| `data/groot/placements/tabletop_push_v1.json` | 400 train + 50 held-out placements |
| `data/groot/_raw/<dataset>/`, `data/groot/<dataset>/` | raw episodes, GR00T datasets; `data/groot/_rejected/` = kept evidence of the zone-render bug |
| `logs/tabletop/` | every log, summary and media file cited here; `PROGRESS.md` |
