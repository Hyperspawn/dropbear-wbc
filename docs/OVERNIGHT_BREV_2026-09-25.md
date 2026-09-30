# Overnight cloud training, 2026-09-25 (Brev)

## Instance

| | |
|---|---|
| Name | `dropbear-train` |
| Hardware | 4× NVIDIA RTX 6000 Ada 48 GB, 52 vCPU, 283 GB RAM (massedcompute) |
| Price | $4.66/h |
| Created | 00:15 IST |
| Software | Ubuntu 22.04, driver 580.126, Isaac Sim 5.0.0 + Isaac Lab 2.2.0 (pip, `~/isaac` venv), vendored rsl-rl 2.3.3 |
| EULA | Isaac Sim EULA accepted on this box, with the user's approval |

**Plant.** The P: USD (sha 45586414… verified on the box) with the same in-memory spawn fixes as local. Calibration
snapshot `e51033d4`. Bundle `logs/brev/dropbear-wbc-brev.tgz`, sha f5685d97…, 401 files, verified on the box.

## Budget guard (runs on this PC)

`tools/brev/watchdog.py` is detached. It keeps the laptop from idle sleep, but a **closed lid still sleeps**.

- **Every 30 min** it pulls logs, checkpoints, evals and rollouts into `logs/brev/remote/dbw{0..3}/` and
  `logs/brev/remote/runs/`.
- **At 17:40 IST** (about 17.4 h, about $81), or if the spend estimate passes $86, it does two final pulls and then
  runs **`brev delete dropbear-train`**.
- **Actions log:** `logs/brev/watchdog.log`.
- **Stop early:** create `logs/brev/STOP_AND_DELETE`.
- **Move the deadline:** write an ISO time into `logs/brev/deadline.txt`.

## What runs (one checkout per GPU, `~/dbw<g>`; queues `~/runs/queue<g>.sh` in tmux `gpu<g>`)

Solver 16/4 is stricter than the laptop's 8/4, which follows the wave-v2 lesson. Every job is chunked and resumable.

| GPU | Jobs, in order | Envs |
|---|---|---|
| 0 | `lib_v2_main`: universal tracker on the 78-clip library `accepted_v2` (512 s: G1 dances, walks, LAFAN, Kimodo, SOMA, synthetic). Runs until the deadline. | 16384 |
| 1 | `lib_v2_nostate`: the same, but the policy gets no simulator-only state (no `motion_anchor_pos_b`, no `base_lin_vel`). This is the deployable variant. | 16384 |
| 2 | `take102_v4` (faithful G1 dance_102, 2500 it), then `gangnam_v4` (2500), then `lafan_dance2_v4` (2500), then `vel_flat_kneehw_cloud` (walking velocity policy, stiff_knee_hw) | 4096 |
| 3 | `lafan_walk1_v4` (2500), then `kimodo_walk_v4` (2500), then `wave_v2_s16` (1500), then `lib_v2_future` (library + future frames, the A/B) | 4096 / 16384 |

**Preflight** (all passed, 00:55). Library probe ok (78 clips, command err 0.0). Throughput on the RTX 6000 Ada at
16/4: library 16384 envs 9.7 s/it (**40.6k env-steps/s**), 8192 envs 33.8k, single clip 4096 envs 25.7k. The walking
driver works through the Linux launcher. Warp prints a harmless `cuDeviceGetUuid` warning.

**On-box evaluator** (`~/runs/eval_watcher.sh`, tmux `eval`).
- **Each finished single-clip job** is evaluated at 32/4: nominal from frame 0, continuous loop, and perturbed
  (seed 7). It also gets a 16/4 check, an ONNX export, and one recorded rollout in
  `logs/brev/eval/rollout_<job>.npz`.
- **Library runs** are evaluated at 10:00 UTC (15:30 IST): one env per clip, continuous, with export.

## Morning checklist

1. `logs/brev/watchdog.log`: the last events show pulls and spend. After 17:40 you should see `deleted`.
2. **Per-job training curves.** Each run is a directory under
   `logs/brev/remote/dbw<g>/logs/rsl_rl/dropbear_tracking*/<stamp>_<job>/`.
   Run `python tools/summarize_training.py <run dir>` on it.
3. **Eval numbers:** `logs/brev/remote/dbw<g>/logs/brev/eval/*_32_4.log` plus the `play_*.json` files inside each run dir.
4. **Clean videos** (local GPU, two-pass renderer) from the recorded rollouts:
   `tools/render_policy_clean.py` with jobs pointing at `logs/brev/remote/dbw<g>/logs/brev/eval/rollout_<job>.npz`.
5. **Library policies** are ONNX with a runtime reference. Any accepted clip can drive them through `tools/policy_runner.py`.

## Changes during the night

- **02:12 IST, GPU2 re-queued.** `take102_v4` collapsed: the v4 foot-pinned reference is untrackable; see the DECISIONS
  entry of 2026-09-25. Two diagnostics settled it: v4 at 8/4 collapsed as well, while v3 at 16/4 kept learning.
  - The new GPU2 queue is `lafan_dance2_v4`, then `take102_v3_s16` (3000 iterations), then `vel_flat_kneehw_cloud`.
  - `gangnam_v4` was dropped (same route). The evaluator now tracks `take102_v3_s16` instead of the dropped jobs.
- **04:25 IST: `lafan_walk1_v4` done (2500 iterations), first cloud result.**
  - Nominal eval at 32/4 (16 envs, frame 0, reset at wrap): **0 falls, body 2.8 cm, joint 0.35 rad, anchor 5.2 cm.**
    This is a human LAFAN walk retargeted through GMR and tracked by Dropbear.
  - The continuous-loop eval "falls" (16, all at step ~766) are an artifact. The walk ends metres from its start, so
    wrapping the reference clock teleports the target. Continuous mode only means something for cyclic or in-place
    clips.
  - Note for reading the curves: the "mean episode length" dips at chunk starts (e.g. 494 -> 18 -> 209) are logging
    artifacts after each resume, not collapses. The joint error keeps improving through them.
- **05:25 IST: `lafan_dance2_v4` done (2500 iterations). NOT learned well; do not treat it as a demo.**
  - Training episode length only reached about 80-100 of 500.
  - The nominal eval from frame 0 terminates at step 6 every time (`ee_body_pos`; 5328 terminations over 2000 steps x
    16 envs). Body error 3.3 cm and anchor 1.2 cm while it holds.
  - The reference is clean: at most 4.5 cm per frame near the start. But the clip opens with both hands overhead,
    left hand at z 1.71-1.79 m against a head at 1.47 m, moving fast, and the policy cannot hold that. Overhead arms
    against gravity are where the elbow four-bar is weakest.
  - Next: more iterations, or check arm authority overhead (elbow motor torque vs the gravity load on the four-bar).
- **07:30 IST: `kimodo_walk_v4` done (2500 iterations; training episode length about 497/500).**
  - Nominal eval at 32/4: **0 falls, body 3.7 cm, joint 0.48 rad.**
  - The continuous-loop falls (step ~311) come right after the 5 s clip wraps: the non-cyclic artifact again.
  - This is a Kimodo-generated (text-to-motion) G1 walk, retargeted through the G1 route with the v4 foot stage. So
    v4 G1-route clips are not broken in general; Take_102 v4 is specifically at the limits of the ankle and knee.
- **06:36 IST snapshot:**
  - `take102_v3_s16` at iteration 944 with episode length 480/500, learning much faster than on the laptop.
  - Library main: 150-190 at iteration 1590. Library no-state: 175-238. Both still climbing slowly.
  - Spend so far: $29.37.
- **08:20 IST: local laptop walking policy (`vel_flat_kneehw`, 2998 iterations, H1-style velocity commands) evaluated.**
  - Setup: 32/4 solver, profile `stiff_knee_hw`, knee effort capped at the RMD-X10 100 N*m peak.
  - **0 falls in all 9 scenarios**; worst closure gap 0.8 mm.
  - Commanded -> measured:
    - stand: 0.00;
    - fwd 0.3: 0.34 and fwd 0.5: 0.53 m/s;
    - turn L/R 0.5: +0.47 / -0.47 rad/s;
    - fwd 0.3 + turn 0.3: 0.32 m/s + 0.29 rad/s;
    - left 0.2: 0.20.
  - Weaknesses:
    - right 0.2 gives only -0.07 (left/right asymmetry);
    - back 0.2 gives -0.10;
    - it marches in place when commanded to stand (about 100 % single support).
  - Export: `logs/rsl_rl/dropbear_velocity/2026-09-24_21-50-46_vel_flat_kneehw/exported/`.
  - Video: `.../videos/clean/velocity_model_2998_{track,front}.mp4`.
  - Eval JSON: `logs/locomotion/play_eval_final_2998.json`.
- **Clean videos of the cloud walks** (rendered locally from the synced rollouts): `logs/brev/media/{lafan_walk1_v4,kimodo_walk_v4}_policy_{track,front}.mp4`.
- **09:15 IST: `take102_v3_s16` done (3000 iterations, the G1 dance_102 v3 reference at 16/4 with 4096 envs).**
  - Nominal eval at 32/4: **0 falls, body 4.6 cm, joint 0.60 rad, anchor 8.4 cm.**
  - The laptop's `model_2300` scored body 6.9 cm, joint 0.81 rad, anchor 23 cm, so the anchor error is about 3x lower.
  - It is still the v3 reference, whose feet float in places.
- **09:50 IST: `wave_v2_s16` done (1500 iterations at 16/4).**
  - Nominal eval at 32/4: **0 falls, body 5.6 cm, anchor 2.4 cm.**
  - The earlier laptop version, trained at 8/4, terminated at 3.64 s in every episode at 32/4. Training at the stricter
    solver fixed it, which confirms the solver-dependence lesson.
- **12:00 IST: GPU2 and GPU3 moved to real-actuator walking** (`tools/brev/swap_to_hw_twin.sh`; docs/ACTUATORS.md
  section 11).
  - **Why:** in a local zero-shot test the walker falls on the datasheet motors (18/18 within 1.2 s), so a better
    idealized walker has no hardware value.
  - **Stopped jobs:**
    - `vel_flat_kneehw_cloud` (GPU2), at iteration 2189 with episode length 1000/1000, so it had converged;
    - `lib_v2_future` (GPU3), at iteration 648: the A/B could not have finished by 17:40.
  - **Both new runs are warm-started from `vel_flat_kneehw_cloud/model_2189`:**
    - GPU2 `vel_hw_v1_ft`: the user's motor map (hip roll X10 1:7, knee CEM-60 provisional);
    - GPU3 `vel_hw_v1_cad_ft`: the CAD motor map (hip roll and knee X10-S2).
  - **First iterations:** episode length for `hw_v1` went 258 -> 350; for `hw_v1_cad` it went 617 -> 691.
- **12:45 IST: the real-motor walkers walk.** Local evaluation at 32/4:
  - `vel_hw_v1_ft` `model_2600`: 0/18 falls; forward 0.3/0.5 m/s gives 0.24/0.45; turning ±0.5 gives +0.41/-0.46.
  - `vel_hw_v1_cad_ft` `model_2700`: 0/18 falls.
  - Both keep some motors above their RATED torque: the knee is pinned at peak, plus the left hip roll, right calf B and
    left elbow run at 2-2.5x rated RMS. See ACTUATORS.md section 11.
- **12:51 IST: GPU3 switched to `vel_hw_v1_thermal_ft`** (`tools/brev/swap_gpu3_thermal.sh`).
  - This is `hw_v1` plus an over-rated-torque penalty of -2e-4, warm-started from `vel_hw_v1_ft/model_2786`.
  - The CAD A/B was stopped because it had answered its question: the CAD motors walk too and show the same pinned knee.
  - The new code went into `dbw2` and `dbw3` only, so the library runs (`dbw0`, `dbw1`) and their 15:30 evaluation are
    untouched.
- **13:29 IST: GPU2 switched to `vel_hw_v1_thermal5_ft`** (`tools/brev/swap_thermal.sh 2 vel_hw_v1_thermal5_ft -1e-3`).
  - This is the thermal penalty at 5x the weight (-1e-3), warm-started from `vel_hw_v1_ft/model_3200`.
  - **Why:** the plain hw walker had converged (reward about 30, episode length 1000).
  - **Thermal run at -2e-4, iterations 2786-3184:** the over-rated term went -0.55 -> -0.35 per episode and then
    plateaued. Tracking stayed at 0.92 and episode length at about 995. The penalty reduced the over-rated torque but
    did not remove it.
  - The episode-length dips at iterations 2985/3184 (to about 13) are the known chunk-start logging artifact.
- **13:55 IST: GPU2 and GPU3 switched to the gait fix** (docs/ACTUATORS.md section 12).
  - **Why:** the user spotted a locked right knee. Telemetry shows the policy hops on its left leg, with the right knee
    crank commanded to -300 deg into its stop.
  - GPU2 `vel_hw_v1_gait_ft`: warm-started from `vel_hw_v1_thermal_ft/model_3500`.
  - GPU3 `vel_hw_v1_gait_scratch`: from scratch.
  - Both use hw_v1 + thermal -5e-4 + `--gait_shaping`.
  - **Stopped jobs:**
    - `vel_hw_v1_thermal5_ft` (-1e-3), after about 400 iterations;
    - `vel_hw_v1_thermal_ft` (-2e-4), at `model_3500`. At `model_3400` it had fixed the calf and elbow over-rating, but
      the knees were still pinned.
- **14:59 IST: GPU2 switched to `vel_hw_v1_gait2_ft`.**
  - This is gait v2 (the stance-knee extension term), warm-started from `vel_hw_v1_gait_scratch/model_700`.
  - **Why:** `gait_scratch` fixed the hopping and is thermally within rating, but it walks crouched on the knee stops.
  - It replaces `vel_hw_v1_gait_ft`, whose right knee still swung straight at iteration 3700.
- **15:31-15:40 IST: library evaluations.** Both trackers were run at 32/4 with one env per clip over the 78 clips of
  `accepted_v2`, in continuous mode, for 1700 steps. The logs are `logs/brev/remote/dbw{0,1}/logs/brev/eval/lib_*`.
  - **`lib_v2_main`:** 51 clips with 0 falls. Body error 7.1 cm, joint error 0.89 rad.
  - **`lib_v2_nostate`** (deployable: no simulator-only state): **57 clips with 0 falls**. Body error 6.9 cm.
  - 51 clips have 0 falls in both.
  - **Clips that fall:** gangnam (242/154 falls), LAFAN dance2 (nostate only, 189), high jump, CR7 / Kobe / Bolt / kick
    / jump-degree / single-foot-balance level 2-5 (TairanTestbed), throw ball, `output_dance`, armraise dance.
  - Walks, stands and waves do not fall in `nostate`.
  - The anchor-position error (0.5 m / 1.6 m) is the known continuous-loop artifact of non-cyclic clips.
  - **Limit:** these trackers use the IDEALIZED (legacy) motors. They are not hardware evidence until fine-tuned on
    `hw_v1` (`train.py --actuator_profile hw_v1`).
- **16:04 IST: GPU2 switched to `vel_hw_v1_gait3b_scratch_s7`**, v3b from scratch with seed 7.
  - **Why:** the warm-started v3b run (`gait3b_ft`) kept the crossed legs, with feet crossed in 88 % of frames at
    iteration 1200.
  - **`gait3b_scratch` at iteration 400:** no crossing (width 6.6-11 cm), all motors at or below 0.77x rated, and torque
    spikes of more than 25 % in only 2.6 % of 5 ms steps. It is not yet walking: 0.07 m/s for a 0.37 command, with
    straight knees.
- **16:20 IST: the switch to reference-motion walking on the real motors.**
  - **Context:** the user noted that the tracking walks (LAFAN, Kimodo) bend the knees naturally, while every
    velocity walker locks them. A reference motion dictates the knees; reference-free RL finds stiff-leg optima on
    Dropbear's weak, short-range knee.
  - **Zero-shot `kimodo_walk_v4` on `hw_v1`** (`logs/brev/media/kimodo_walk_hw_v1_zeroshot_dashboard.mp4`):
    - the knees bend (21-48 deg);
    - but it has 7 falls in 10 s: the motors clip 20-80 % of the time and run at 1.5-2.3x rated RMS, with torque jumps
      of more than 25 % in 43 % of 5 ms steps.
  - **The Kimodo reference itself crosses the feet:** they are less than 5 cm apart in 66 % of frames, a G1 -> Dropbear
    retarget artifact (the pelvis is narrower). Six of the 78 library clips have feet under 5 cm apart in more than
    20 % of frames: kimodo walk, SOMA A057 walk, lafan run1, and ASAP walk level 1 / 2 / step_forward level 2.
    - Clean walk references: ASAP `walk_level4` (4.5 %) and `lafan1_walk1` (8 %).
  - **New jobs** (`tools/brev/swap_track.sh`: warm start via `--init_run`, `--actuator_profile hw_v1`):

| GPU | run | clip / library | warm start | envs |
|---|---|---|---|---|
| 0 | `asap_walk4_hw` | ASAP `walk_level4` | the LAFAN walk tracker | 4096 |
| 1 | `lafan_walk1_hw` | `lafan1_walk1` | `lafan_walk1_v4` `model_2490` | 4096 |
| 2 | `lib_nostate_hw` | 78-clip library | `lib_v2_nostate` `model_4200` | 8192 |
| 3 | `vel_hw_v1_gait3b_scratch` | velocity walker | from scratch | 4096 |

- **16:24 IST: the deadline moved to 18:35 IST** (`logs/brev/deadline.txt`), an estimated $85.4. That is within the
  user's approved ~$90, and the watchdog's $86 hard cap still applies.
- **16:34 IST: the user added $50 of credit, so the budget is now about $140 total.**
  - The watchdog was restarted (PID 36708) with a hard cap of $130 and a deadline of **2026-09-26 04:00 IST** (about
    $129).
  - Both can be overridden without a restart through `logs/brev/budget.txt` and `logs/brev/deadline.txt`.
  - The laptop must stay awake and plugged in with the lid open. The watchdog is what deletes the instance.
- **16:42 IST: GPU3 switched to `kimodo_walk_w10_hw`.**
  - This is the Kimodo walk tracker fine-tuned on `hw_v1` on the width-fixed reference `output_walk_v4w10` (see the
    DECISIONS entry of 2026-09-25 16:40).
  - It replaces the reference-free `vel_hw_v1_gait3b_scratch`. At iteration 796 that run walked with 0 falls, no
    crossing, all motors within rating and 0.30 of a 0.37 m/s command, but with straight knees (a compass gait).
  - All four GPUs now run real-motor tracking fine-tunes.
- **17:08 IST: GPU2 switched to `lib_nostate_hw_v5`.**
  - This is the no-state library tracker on `hw_v1`, warm-started from `lib_nostate_hw/model_4500`, trained on
    library `accepted_v5`.
  - `accepted_v5` (`data/motions_v5/libraries/accepted_v5.json`) has 78 clips and 512 s. Its G1-route clips were
    rebuilt with the two-pass 10 cm minimum stance width and re-settled.
  - Crossing, v2 -> v5: clips with any crossed frame 20 -> 5; mean crossed frames 2.72 % -> 0.50 %; clips with more
    than 20 % narrow frames 6 -> 1.
- **17:44 IST: GPU3 switched to `lafan_walk1_kneeX10S2`**, the knee-motor A/B against GPU1's `lafan_walk1_hw`
  (ACTUATORS.md section 13). It replaces `kimodo_walk_w10_hw`, which was at iteration about 3200 with episode length
  389 and body error 15 cm.
  - The first real-motor bent-knee walk (`lafan_walk_hw_3300`) had 0 falls, bent knees and uncrossed feet, but the
    CEM-60 knees were clipped 82-85 % of the time.
- **17:49 IST: GPU0 switched to `asap_walk4_hw_reg`.**
  - This is the ASAP walk tracker on `hw_v1` warm-started from `asap_walk4_hw/model_3485`, now with `--hw_regularizers`
    (thermal -2e-4 plus torque-rate -0.02; tracking `enable_hw_regularizers`).
  - **`asap_walk_hw_3300`, the best real-motor walk so far:**
    - 0 falls, body error 5.4 cm;
    - the knees cycle: L -5..47 deg, R -5..30 deg;
    - the feet never cross (mean 33 cm);
    - knee cranks clipped 40-52 % of the time, knees and calves at 1.2-1.4x rated RMS, hip roll at 1.7x;
    - torque jumps in 13 % of 5 ms steps.
- **Runs from 17:49:**

| GPU | run |
|---|---|
| 0 | `asap_walk4_hw_reg` |
| 1 | `lafan_walk1_hw` (the knee A/B baseline, CEM-60) |
| 2 | `lib_nostate_hw_v5` |
| 3 | `lafan_walk1_kneeX10S2` (knee A/B) |
- **18:55 IST: the knee A/B gave a clear result** (ACTUATORS.md section 13; DECISIONS 2026-09-25 18:55).
  - CEM-60 knee: clipped 82-85 % of the time, 1.65x rated.
  - X10-S2 knee: clipped 14 %, 1.03-1.15x rated, body error 5.1 vs 10.0 cm.
  - Dashboard: `logs/brev/media/lafan_walk_kneeX10S2_3100_dashboard.mp4`.
- **18:48 IST: GPU1 switched to `lib_nostate_kneeX10S2_v5`**, the library A/B. It is the library-v5 no-state tracker
  with the X10-S2 knee (`hw_v1_knee_x10s2`), warm-started from `lib_nostate_hw_v5/model_5200`. GPU2 continues the
  CEM-60 version from the same checkpoint.
  - The LAFAN CEM-60 baseline (`lafan_walk1_hw`) was stopped at iteration about 4150.
  - **Planned for the end of the night** (after the last sync, locally): the per-clip library evaluation of both
    library runs on their profiles, and dashboards of the final ASAP-reg and X10-S2-knee walks.
- **19:48 IST: `asap_walk_hw_reg_4700` is the best real-motor walk so far.**
  - CEM-60 knee, 0 falls, 4.3 cm body error.
  - Only one motor is above rated (right elbow, 1.24x).
  - Dashboard: `logs/brev/media/asap_walk_hw_reg_4700_dashboard.mp4`.
- **19:48 IST: GPU3 switched to `lafan_walk1_hw_reg`**, the LAFAN walk on the CEM-60 with regularizers, warm-started
  from `lafan_walk1_hw/model_4100`. It tests whether the crouched-gait knee overload is fundamental. It replaces
  `lafan_walk1_kneeX10S2`, whose A/B is done.

| GPU | run |
|---|---|
| 0 | `asap_walk4_hw_reg` |
| 1 | `lib_nostate_kneeX10S2_v5` |
| 2 | `lib_nostate_hw_v5` |
| 3 | `lafan_walk1_hw_reg` |
- **20:50 IST: `lafan_walk_hw_reg_4700`** (LAFAN walk, CEM-60, regularizers).
  - The knees are still over rating: 1.30 / 1.57x, clipped 41 / 72 %.
  - The crouched gait needs more knee torque than the CEM-60 has. The upright ASAP gait fits (ACTUATORS.md section 13).
- **20:53 IST: GPU3 switched to `asap_walk_nostate_hw_reg`, trained from scratch.**
  - This is the ASAP walk on the new single-clip no-state task `Dropbear-Tracking-Flat-NoState-v0`. The policy obs
    are 119, without `motion_anchor_pos_b` and `base_lin_vel`; the critic keeps 251.
  - It runs on `hw_v1` with `--hw_regularizers`.
  - **Why:** this is the deployable counterpart of the best walk (`asap_walk_hw_reg`, which needs simulator state).
  - `tools/brev/swap_track.sh` now takes the source `none` to train from scratch.
- **21:42 IST: GPU0 switched to `kimodo_walk_w10_nostate_hw_reg`, trained from scratch.**
  - This is the width-fixed Kimodo (text-to-motion) walk on the no-state task, on `hw_v1` with regularizers: the
    deployable text-to-motion walk.
  - `asap_walk4_hw_reg` had converged: episode length 465-487, body error about 8 cm, at iteration about 6150. Its
    checkpoints are synced in `logs/brev/remote/dbw0/...`.

| GPU | run |
|---|---|
| 0 | `kimodo_walk_w10_nostate_hw_reg` |
| 1 | `lib_nostate_kneeX10S2_v5` |
| 2 | `lib_nostate_hw_v5` |
| 3 | `asap_walk_nostate_hw_reg` |
- **21:43 IST: the end-of-night job was launched, detached** (`logs/hw_twin/end_of_night.sh`, PID 41604, output in
  `logs/hw_twin/end_of_night.out`).
  - It waits for the watchdog's `deleted` event, which comes after the final pulls. It can be forced with
    `logs/hw_twin/RUN_END_OF_NIGHT_NOW`.
  - **It then runs locally:**
    - the per-clip library evaluation (78 clips, continuous, 1700 steps, 32/4) of `lib_nostate_hw_v5` (CEM-60) and
      `lib_nostate_kneeX10S2_v5`;
    - dashboards of the deployable walks `asap_walk_nostate` and `kimodo_walk_nostate`, and of the final
      `asap_walk_hw_reg`.
  - Outputs go to `logs/hw_twin/libeval_*.log`, `logs/brev/media/*_dashboard.mp4` and `logs/hw_twin/telemetry_*.npz`.

## 2026-09-26 night (user: "keep building, fix the small issues"; about $32 of credit left at 01:20)

- **01:22 IST: the deadline moved to 06:30 IST** (`deadline.txt`) and the budget cap to $142 (`budget.txt`). That is
  an estimated $141 total, which leaves about $8 of the user's real balance.
- **01:27 IST: continuous walking.** `tools/make_cyclic_clip.py` built looped walks (docs/ISSUES.md #7):
  `data/motions_cyclic/lafan1_walk1_cyclic.npz` (30.8 s) and `data/motions_cyclic/kimodo_walk_cyclic.npz` (26 s).

| GPU | run | task | start |
|---|---|---|---|
| 0 | `lafan_cyclic_hw_reg` | Flat, hw_v1 + regularizers | from `lafan_walk1_hw_reg/model_4800` |
| 1 | `lafan_cyclic_nostate_scratch` (from 01:36) | NoState | from scratch; the warm start from the Kimodo NoState policy did not transfer (episode length 47-63) |
| 3 | `kimodo_cyclic_nostate_hw_reg` | NoState | from `kimodo_walk_w10_nostate_hw_reg/model_2587` |

  - GPU1 replaced the library X10-S2 run and GPU3 replaced the ASAP NoState run. The ASAP "walk" clip is really a
    stand-sprint-stop (issue #18).
  - GPU2 `lib_nostate_hw_v5` continues.
- **01:30 IST: fair comparison** `logs/brev/media/lafan_walk_4way_compare.mp4` (same LAFAN clip, same kinematic
  renderer). Key-body tracking error, root-relative:

| policy | tracking error | lag behind the reference at the end |
|---|---|---|
| idealized motors | 6.6 cm | 6 cm |
| **real motors, CEM-60** | **14.7 cm** | **1.35 m** |
| real motors, X10-S2 knee | 8.9 cm | 21 cm |

- **01:34 IST: glitch-free side-panel dashboard** (`tools/make_dashboard.sh`):
  - `lafan_cyclic_test_4800_dashboard.mp4` is a 20 s continuous walk, 15 m, 0 falls, 0 % crossing.
  - Remaining issues: knee cranks at 1.07 / 1.45x rated, the right elbow at 1.13x, torque jumps in 26 % of 5 ms steps
    (issue #11 test pending).
- **The end-of-night job was replaced with v2** (`logs/hw_twin/end_of_night_v2.sh`, output in `end_of_night_v2.out`):
  - the library evaluations as before;
  - glitch-free 20 s dashboards of the three continuous-walk runs.
- **01:40 IST: target interpolation A/B (issue #11)**. Same policy (`lafan_walk1_hw_reg/model_4800`), cyclic LAFAN, 15 s:

| motor profile | falls | tracking error | jumps > 25 % of peak | p99 jump | over rated |
|---|---|---|---|---|---|
| `hw_v1` (step targets) | 0 | 0.092 m | 27.0 % of steps | 34.6 % | RL knee 1.40x, RH elbow 1.11x, LL knee 1.09x |
| `hw_v1i` (4-step ramp) | 0 | 0.091 m | **18.3 %** | **25.6 %** | RL knee 1.46x, LL knee 1.18x, RH elbow 1.11x |

  - The ramp cuts the big 200 Hz torque jumps by about a third and leaves tracking alone. The policy never saw the ramp,
    so the knees run a bit hotter.
  - Export sidecars (`policy.json`) now record `target_interp_steps`, so the deploy runtime / firmware knows to ramp.
- **01:44 IST: GPU0 -> `kimodo_cyclic_nostate_hw_v1i`**. It warm-starts from the same checkpoint as GPU3
  (`kimodo_cyclic_nostate_hw_reg/model_2700`) with the `hw_v1i` profile, which gives a clean trained-in A/B against
  GPU3. `lafan_cyclic_hw_reg` stopped at about 4900 iterations (converged, episode length about 480, and it is not
  deployable because it uses state observations); its checkpoints stay on dbw0 for the end-of-night dashboard.
- **01:50-02:05 IST: the telemetry issue scanner** (`tools/telemetry_issue_scan.py`, ISSUES #22) found two hardware
  problems that the videos hide:
  - **#19 knees resting on their flexion stop.** Every LAFAN policy on the CEM-60 rests the stance knee on the crank's
    30 deg stop 30-50 % of the time, with the motor saturated at -60 N*m. The X10-S2 knee does it 2 % of the time,
    ASAP 4-7 %, and the deployable Kimodo 6 % / 2 %.
  - **#20 right foot on the deployable Kimodo walk.** It lands at 2.7 m/s and skids 18 cm; its toe scuffs mid-swing.
    Cause: the policy, not the reference (which lifts both feet 9.6 cm). The robot lifts the right foot 12.5 cm vs
    16.5 cm on the left, keeps the right toe down in swing, and its right ankle-rod motor runs 1.55x rated and is
    clipped 29 % of the time.
  - New training terms, both smoke-tested locally: `--knee_stop_penalty=-10` and `--feet_slide_penalty=-1`.

| GPU | run | profile / terms | start |
|---|---|---|---|
| 0 | `kimodo_cyclic_nostate_hw_v1i` | hw_v1i + regularizers (control) | 01:44, from `kimodo_cyclic_nostate_hw_reg/model_2700` |
| 1 | `lafan_cyclic_nostate_v1i_kstop` | hw_v1i + regularizers + knee stop | 01:52, from scratch |
| 2 | `lib_nostate_hw_v5` | unchanged | |
| 3 | `kimodo_cyclic_nostate_v1i_feet` | hw_v1i + regularizers + knee stop + feet slide | 02:02, from `kimodo_cyclic_nostate_hw_reg` latest (about 2900) |

- The end-of-night job is now `logs/hw_twin/end_of_night_v4.sh`. It renders dashboards for all four walk runs (plus
  the hw_v1 Kimodo baseline) with each run's own motor profile, scans them into
  `logs/hw_twin/issue_scan_end_of_night.md`, then runs the library evaluation.
- **02:10 IST: hardware gate** (`telemetry_issue_scan.py`: FAIL on any HIGH fall / stop_load / thermal / clipped /
  speed / touchdown_skid / foot_slip / scuff). All 16 recorded rollouts are in `logs/hw_twin/issue_scan_all_2026-09-26.md`.

| rollout | gate | why |
|---|---|---|
| ASAP upright walk + regularizers (CEM-60 knee) | **PASS** | |
| LAFAN walk on the X10-S2 knee | **PASS** | |
| velocity walkers (gait / gait3b) | PASS | they squeeze a knee stop at 0.77x rated (#23, fixed for new runs) |
| every LAFAN walk on the CEM-60 | FAIL | knees on the flexion stop 30-50 %, knee clipped, 1.4x rated |
| deployable Kimodo (`kimodo_cyclic_nostate_hw_reg/2800`) | FAIL | right ankle-rod motor 1.55x rated and clipped 29 %; right touchdown skid 17.7 cm |
| early fine-tunes / zero-shot | FAIL | many motors |

  - Hardware reading: on the CEM-60 knee only an upright gait passes. A crouched gait (LAFAN) needs the X10-S2 knee.
- **02:10 IST: library evaluation of `lib_nostate_kneeX10S2_v5/model_8200`** (NoState, real motors with the X10-S2
  knee, 78 clips looped continuously for 34 s, locally): **41/78 clips with 0 falls** (idealized-motor NoState: 57).
  - The falling clips are the dynamic ones: gangnam style, high jump, chaines turns, CR7, degree jumps, kicks, shots,
    single-foot balance and jumps, run.
  - Walks, dances at walking pace and upper-body clips hold.
  - Per clip: `logs/hw_twin/libeval_lib_nostate_kneeX10S2_v5.json`. The CEM-60 library run (GPU2) is evaluated by
    the end-of-night job.
- **Tracking target clamp (ISSUES #25):** `train.py --target_clamp_margin_deg=0` (smoke-tested); `play.py` replays it
  from run_info; export `target_clip`. The scanner's `target_beyond` check shows that unclamped policies command past
  a stop in 20-84 % of steps.
- **02:20-02:30 IST: A/B verdict (same start `kimodo_cyclic_nostate_hw_reg`, same clip, scanned at 15 s):**

| run | iterations of fine-tune | HW gate | right touchdown skid | right ankle rod | 200 Hz jumps | body err |
|---|---|---|---|---|---|---|
| baseline hw_v1 (`hw_reg/2800`) | - | FAIL | 17.7 cm | 1.55x rated, clipped 29 % | 23 % | 6.97 cm |
| control hw_v1i (`hw_v1i/3100`) | 400 | FAIL | 14 cm (left 11) | ok | 9 % | |
| **hw_v1i + knee stop + feet slide (`v1i_feet/3184`)** | 284 | **PASS** | none | 1.01x | 13 % | 7.02 cm |

  - This is the first deployable (NoState) walk that passes the hardware gate. Tracking is unchanged, and the joint
    error even improved (0.92 -> 0.85).
  - The feet-slide term is what removes the skid. Interpolation alone fixed the ankle-rod overload and the jumps.
- **02:30 IST: GPU0 -> `kimodo_ts119_nostate_allfix`.** Froude-timed Kimodo walk (ISSUES #27), warm-started from the
  passing GPU3 policy.
  - Settings: `hw_v1ie` (hw_v1i + elbow kp 200/4, #24), knee stop, feet slide, target clamp 0 (#25).
  - The GPU0 control had served its purpose.
  - The end-of-night job is now `end_of_night_v5.sh`, which renders this run first.
- **02:36 IST: dashboard of the first passing walk.** `logs/brev/media/kimodo_walk_hwpass_3184_dashboard.mp4` (sent):
  20 s, the checker verdict PASS, 0 falls, 1.18 m/s.
  - The dashboard now prints the whole-run hardware verdict (`telemetry_issue_scan`) in its panel.
  - The curve labels are now drawn on top of the plots (they were overdrawn).
  - `tests/test_telemetry_issue_scan.py` covers the checker: 4 synthetic defects.
- **02:34 IST: Froude library v6 build started locally** (`logs/hw_twin/build_v6_froude.sh`): the 68 G1-route clips of
  `accepted_v5` re-retargeted with `--time-scale 1.19`, settled and validated; the 10 others are copied. The manifest
  goes to `data/motions_v6ts/libraries/accepted_v6ts.json`.
- **02:46 IST: Froude library v6 built** (`data/motions_v6ts/libraries/accepted_v6ts.json`): 78 clips, all accepted,
  585 s (v5: 512 s).

| | v5 | v6 (Froude) |
|---|---|---|
| clips within every motor's no-load speed | 70 / 78 | 71 / 78 |
| motor-clip pairs over no-load | 18 | 16 |
| motor-clip pairs over the peak-torque speed | 39 | 34 |

  - Modest gains: the fast clips are jumps and flights, and the settle's step projection had already capped speeds.
  - A zero-shot closed-loop evaluation of the X10-S2 library policy on v6 is running
    (`logs/hw_twin/libeval_kneeX10S2_on_v6ts.json`).
- **02:58 IST: zero-shot on the Froude library.** The same X10-S2-knee library policy, trained on v5 timing, ran on
  the v6 clips.

| | v5 | v6 (Froude) |
|---|---|---|
| clips with 0 falls | 41 / 78 | **43 / 78** |
| total falls in 34 s continuous loops | 626 | **574 (-8 %)** |

  - Clips that now hold: jump_forward_level3, side_jump_level4, single_foot_balance_level3, G1_Take_102.
  - Clips that now fall: SpiderMan_level2, shoot_level1.
  - This is without retraining on the new timing. Next session: train the library on `accepted_v6ts`.
- **02:55 IST: GPU2 -> `lib_v6ts_nostate_v1i_fix`.** The library tracker (15 h on v5, hw_v1, episode length about 170)
  is warm-started on the Froude library v6 with hw_v1i, knee stop, feet slide and target clamp 0 (8192 envs).
  - The `lib_nostate_hw_v5` checkpoints stay on dbw2.
  - The end-of-night job is now `end_of_night_v6.sh`: it evaluates both library runs, each on its own library.
- **02:53 status:**

| GPU | run | episode length | body error | note |
|---|---|---|---|---|
| 0 | Froude all-fix | 406 | 0.108 | |
| 1 | LAFAN knee-stop, from scratch | 283 | | iteration 682, climbing fast |
| 3 | Kimodo feet | 437 | | |
- **04:05 IST: the knee-stop penalty works (ISSUES #19).** Scan of `lafan_cyclic_nostate_v1i_kstop/model_1400`
  (LAFAN walk from scratch, CEM-60 knee, hw_v1i):

| | knee on its flexion stop under load (L / R) | knee RMS vs rated (L / R) |
|---|---|---|
| before (`lafan_walk1_hw_reg/4800`, hw_v1i) | 32 % / 45 % | 1.18x / 1.46x |
| with the penalty | **0.9 % / 0 %** | **0.88x / 0.81x** |

  - The crouched walk now fits the CEM-60 thermally.
  - It still fails the gate on touchdown skid (left 37 cm) and the ankle rods (1.3-1.4x): that run had no feet-slide
    term.
  - 04:07: GPU1 -> `lafan_cyclic_nostate_v1i_kstop_feet`, warm-started, adding feet slide and target clamp 0.
  - The end-of-night job is now `end_of_night_v7.sh` and renders this run first.
- **04:15 IST: the all-fixes candidate passes.** `kimodo_ts119_nostate_allfix/model_4400`: Froude-timed Kimodo walk,
  hw_v1ie (elbow kp 200), knee stop, feet slide, target clamp 0, NoState.
  - HW gate **PASS**, 0 falls, body error 7.1 cm, 200 Hz jumps 12 %.
  - Global anchor error after 15 s: 0.52 m (the untimed walk: 2.04 m). It keeps pace with the reference.
  - Warnings left: ankle-rod target jitter (3.1 deg), elbows near their straight stop, 12 deg lean (the reference's
    own is 9.3).
  - This is the best deployable walk so far: every deploy-relevant setting is trained in.
- **04:22 IST: GPU3 -> `kimodo_ts119_allfix_smooth`.** It warm-starts from the passing GPU0 all-fixes run with
  `action_rate_l2` -0.1 -> -0.3 (new flag `train.py --action_rate_weight`), an A/B against the ankle-rod target
  jitter (ISSUES #28a).
  - The end-of-night job is now `end_of_night_v8.sh`: 9 dashboards + 2 library evaluations, about 1.3 h after the
    06:30 delete.

| GPU | run | since |
|---|---|---|
| 0 | `kimodo_ts119_nostate_allfix` (the candidate) | |
| 1 | `lafan_cyclic_nostate_v1i_kstop_feet` | 04:07 |
| 2 | `lib_v6ts_nostate_v1i_fix` | |
| 3 | `kimodo_ts119_allfix_smooth` | 04:22 |
- **05:54 IST: watch item.** GPU2 `lib_v6ts_nostate_v1i_fix` episode length 170 (v5) -> 139 -> 104 while body error
  improves (0.211 -> 0.194).
  - Possible causes: adaptive sampling concentrating on the hardest clips, or the knee-stop / feet-slide penalties
    conflicting with deep-squat and dance references.
  - The end-of-night per-clip evaluation on v6 decides. If specific clips collapse, exempt them from the knee-stop
    term (or scale it by the reference knee angle) before the next library run.
- **06:30-06:40 IST: the auto-delete FAILED.** The brev CLI's login token expired, and every `brev delete` from the
  watchdog asks for an interactive login (EOF).
  - The final pulls did complete (06:31-06:32), so all checkpoints are local.
  - 06:40: all training processes on the instance were stopped (GPUs idle) and the VM was powered off over SSH
    (`sudo shutdown -h`). Whether a powered-off VM still bills depends on the provider.
  - **The instance still has to be deleted:** the user runs `brev login` (the watchdog keeps retrying) or deletes
    `dropbear-train` in the Brev console.
  - The end-of-night job was started by hand (`RUN_END_OF_NIGHT_NOW`).
- **06:45 IST: the LAFAN walk passes on the CEM-60 knee.** `lafan_cyclic_nostate_v1i_kstop_feet/model_3100`
  (NoState, hw_v1i, knee stop, feet slide, clamp 0), 20 s clean dashboard
  (`logs/brev/media/lafan_walk_deployable_kstop_feet_3100_dashboard.mp4`):
  - HW gate **PASS**, 0 falls, 200 Hz jumps 9 %.
  - Knee on its stop 0.2 % / 0 % (was 32 % / 45 %); knee 0.94x / 0.82x rated (was 1.18x / 1.46x).
  - Warnings: right-elbow asymmetry and hot elbow (1.06x; this run has kp-50 elbows), hip-pitch target lead about
    9 deg.
  - Both reference walks now pass the gate on the real motor twin. The crouched LAFAN gait no longer needs the X10-S2
    knee.
- **06:55 IST: smoothing A/B at equal training.**

| run | target jitter mean | worst rod | 200 Hz jumps | body error | gate |
|---|---|---|---|---|---|
| `allfix/5900` | 1.35 deg | 2.8 deg | 12 % | 6.3 cm | PASS |
| `allfix_smooth/5800` (action rate -0.3) | **1.01 deg** | **1.9 deg** | **10 %** | 6.2 cm | PASS |

  - **Best deployable walk: `kimodo_ts119_allfix_smooth/model_5800`** (Froude-timed Kimodo walk, NoState, hw_v1ie,
    knee stop, feet slide, clamp 0, action rate -0.3).
- **07:13 IST: end-of-night scan** (`logs/hw_twin/issue_scan_end_of_night.md`, 8 clean 20 s rollouts, 0 falls in all):

| walk | policy | HW gate |
|---|---|---|
| LAFAN | `lafan_cyclic_nostate_v1i_kstop_feet/3100` | **PASS** |
| Kimodo Froude | `kimodo_ts119_allfix_smooth/5800` | **PASS** |
| Kimodo Froude | `kimodo_ts119_nostate_allfix/5900` | **PASS** |
| Kimodo | `kimodo_cyclic_nostate_v1i_feet/4577` | **PASS** |
| LAFAN, knee stop only | `.../kstop/1500` | FAIL (skid) |
| Kimodo, hw_v1i only | `.../hw_v1i/3200` | FAIL (skid) |
| Kimodo, hw_v1 baseline | `.../hw_reg/2985` | FAIL (skid) |
| LAFAN, state obs | `lafan_cyclic_hw_reg/4900` | FAIL (knee stops, thermal) |

  - Conclusion: trained with the feet-slide + knee-stop penalties -> PASS; without -> FAIL.
- **07:22 IST: END_OF_NIGHT_DONE.** Library evaluations, CEM-60 knee, 78 clips, 34 s continuous loops:

| policy | library | profile | 0-fall clips | total falls |
|---|---|---|---|---|
| `lib_nostate_hw_v5/8900` | v5 | hw_v1, no hw penalties | 38 / 78 | 700 |
| **`lib_v6ts_nostate_v1i_fix/10492`** | v6 (Froude) | hw_v1i + knee stop + feet slide + clamp, 3.5 h fine-tune | **41 / 78** | 727 |
| (reference) `lib_nostate_kneeX10S2_v5/8200` | v6 zero-shot | X10-S2 knee, no penalties | 43 / 78 | 574 |

  - With the hardware constraints trained in, the library tracker holds more clips than before (38 -> 41).
  - Brev: the instance is powered off but NOT deleted. 45+ delete attempts fail on the expired brev login; the user
    must run `brev login` or delete it in the console.

## Night 2 summary (2026-09-26, 01:20-07:22 IST)

- **Deployable walks that pass the hardware gate on the real-motor twin:**
  - Kimodo Froude + all fixes + smoothing (`kimodo_ts119_allfix_smooth/5800`, the best one);
  - Kimodo Froude + all fixes;
  - Kimodo + feet slide;
  - LAFAN + knee stop + feet slide (**on the CEM-60 knee**).
- **What made them pass:** trained-in target interpolation, the knee-stop penalty, the feet-slide penalty, target
  clamp 0, stiffer elbows, action rate -0.3 and Froude timing.
- **Tools:**
  - `tools/telemetry_issue_scan.py` (the gate and 17 checks);
  - live text -> motion (`tools/text_to_motion.py`, `MotionStream`, `play.py --live_*`, `policy_runner --live_dir`);
  - the training motor law in the Newton bridge;
  - blind rough terrain (`--terrain rough`);
  - Froude retarget (`--time-scale`) and library v6.
- **Issues found and fixed:** ISSUES #19-#28.
- **Incident:** the brev CLI login expired, so the auto-delete failed. Before the next cloud run, check that the
  watchdog can delete: run `brev ls </dev/null` and make sure it does not ask for a login.
- **08:11 IST:** the watchdog gave up at 07:38 (`DELETE_FAILED_GIVE_UP` after 60 attempts).
  `tools/brev/delete_when_authed.sh` now runs detached in WSL. It polls the brev auth non-interactively every 2 min
  for 24 h and finishes the authorized delete as soon as the user has run `brev login` (log:
  `logs/brev/delete_retry.log`).

## Session 2 (2026-09-26, the user added $50)

- **The old instance was already gone** when the user re-logged in at about 10:30.
- **New `dropbear-train`**, massedcompute 4x RTX 6000 Ada, created at 10:43 IST.
- **Watchdog:** cap $35, deadline 18:10 IST. The real balance is uncertain, because the powered-off session-1 VM may
  have billed until it disappeared.
  - New: after 3 failed deletes it kills the jobs and powers the VM off over SSH.
  - Old `QUEUE_DONE` logs were moved to `remote/runs_2026-09-25`, and the stale deadline/budget files were reset.
- **Fresh bundle** (166 MB: code, v6 library, cyclic clips, stand).
- **`setup_instance.sh` now includes the flatdict / setuptools / pins fixes.** The first attempt failed on flatdict;
  the rerun succeeded.
- **Runs, launched 11:01 IST** (`launch_session2.sh`):

| GPU | run | profile / terms | warm start |
|---|---|---|---|
| 0 | `vel_rough_v1i` | velocity, `--terrain rough`, gait v3, clamp 0, thermal -5e-4 | `gait3b_ft/1300` |
| 1 | `lib_v6ts_v1i_fix2` | library v6, knee stop, feet slide, clamp 0 | `lib_v6ts/10492` |
| 2 | `lafan_nostate_smooth_elbow` | hw_v1ie, action rate -0.3 + all | `lafan kstop_feet/3100` |
| 3 | `lib_v6ts_v1ie_smooth` | library v6, hw_v1ie, action rate -0.3 + all | `lib_v6ts/10492` |

- **Kimodo:**
  - The weights (G1-RP-v1 1.1 GB, LLM2Vec adapters) are in the local HF cache (`$DROPBEAR_HF_CACHE`).
  - The Windows build failed (no C++ compiler), so it is being installed in WSL with the venv on H:.
  - Llama-3 access is still pending on the user's HF account (401 gated).
- **14:25 IST: terrain walker check** (`vel_rough_v1i/model_3200`, local replay on rough terrain; `play_locomotion.py`
  now replays the trained terrain).
  - 0 falls, velocity error 0.14 m/s.
  - But: legs crossed 39 %, target jitter 6-8 deg, 90 % single support (ISSUES #29). It keeps training until 18:10;
    the next round needs stronger feet-width and action-rate weights.
- **13:00-13:40: Kimodo end to end** (docs/TEXT_TO_MOTION.md).
  - Windows CPU text encoder + WSL GPU Kimodo.
  - 60 generated everyday clips -> library `accepted_v7gen` (138 clips) -> GPU3 `lib_v7gen_v1ie_smooth`.
  - Live service: prompt -> spliceable clip in about 8 s; live splice demo PASS.
- **14:40: GPU0 restarted as `vel_rough_v1i_b`** (feet width -5, action rate -0.05, ISSUES #29). It did not collapse:
  episode length 920-980 throughout, reward -31 -> +2.9 by 17:57.
- **GPU3 at 13:22: `lib_v7gen_v1ie_smooth`** (library + 60 Kimodo clips). Its checkpoint 13000 tracked the five live
  clips of the 17:01 session with 0 falls and HW gate PASS, cleaner than `lib_v6ts_v1i_fix/10492`
  (docs/TEXT_TO_MOTION.md); `tools/live_session.sh` now defaults to it.
- **End (18:10 deadline):** two final pulls at 18:11-18:12, `brev delete` at 18:12, **deleted 18:13:18**, `brev ls`:
  no instances. Estimated spend $34.92 (cap $35). The watchdog's auth was checked beforehand (`brev ls </dev/null`).
- **Wrap-up** (`logs/hw_twin/end_of_session2.sh`, started 18:13 after the deleted event): clean dashboard + scan of
  the final LAFAN smooth-elbow walk, the terrain walker replay + scan, library evals of `lib_v7gen` (138 clips) and
  `lib_v6ts_v1i_fix2` (78 clips). Results below when done.
- **Wrap-up results (18:13-18:45)** (note: the wrap-up ran twice in parallel by mistake: a `nohup` launch at 15:00 was
  alive although `ps` did not list it, and a second launch followed. They serialized on the GPU lock; the duplicate
  only repeats the same evals):

| run | checkpoint | eval | result |
|---|---|---|---|
| `lafan_nostate_smooth_elbow` | 8000 | cyclic LAFAN walk, hw_v1ie, 20 s | **HW gate PASS**, 0 falls, 200 Hz jumps 7 %. Warnings: right elbow 1.25x rated (the stiffer hw_v1ie elbow costs torque), hip target lag 9.6 deg, left-foot mid-stance slip 0.19 m/s, tilt 7.1 deg |
| `vel_rough_v1i_b` | 5500 | rough terrain replay, hw_v1i, 20 s | **HW gate PASS**, 0 falls. vs `vel_rough_v1i/3200`: legs crossed 39 % -> 2.2 %, ankle-rod target jitter 6-8 -> 2.9-3.9 deg. New: a limp (stance L/R 0.73/0.42) and knees within 3 deg of a stop 89 % of the time (straight-leg gait, #13) |
| `lib_v6ts_v1i_fix2` | 13700 | library v6 (78 clips), hw_v1i | **46/78 clips with 0 falls** (baseline `lib_v6ts_nostate_v1i_fix/10492`: 41/78), total falls 665 vs 727, body error 8.3 cm |
| `lib_v7gen_v1ie_smooth` | 13600 | library v7gen (138 clips), hw_v1ie | **94/138**: 43/78 on the original library, **51/60 on the Kimodo-generated everyday clips**; body error 7.7 cm. The 9 generated clips that fall: arms up, bounce, boxing, clap, knee raise, reach down, turn around, walk circle, walk turn right |

- Media: `logs/brev/media/lafan_walk_smooth_elbow_8000_dashboard.mp4`, `rough_terrain_b_5500_dashboard.mp4`,
  `live_final_dashboard.mp4`. Scan: `logs/hw_twin/issue_scan_session2.md`.
- The duplicate wrap-up doubled as a repeatability check (PhysX is not bit-exact run to run): `lib_v7gen` 94/138
  zero-fall clips in both runs (851 vs 852 falls), `lib_v6ts_v1i_fix2` 46/78 in both (665 vs 670).
