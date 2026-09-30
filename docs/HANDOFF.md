# dropbear-wbc handoff (started 2026-09-25 ~10:45 IST, updated through 2026-09-26)

> For the public overview start at the repository README and docs/README.md. This file is the working handoff
> between development sessions: dense, chronological, and written for someone continuing the work.

Read this first in a new session. Then read `CONTRACTS.md`, `DECISIONS.md`, `ACTUATORS.md`,
`OVERNIGHT_BREV_2026-09-25.md`, `DEMOS.md` and the `logs/*/PROGRESS.md` files.

## What exists (all simulation; the plant is still a CAD-dynamics twin)

- **Plant.** The user's P: USD (sha 45586414…) with in-memory spawn fixes:
  - orphan bodies removed;
  - hip joint friction set to 0;
  - `LL_Revolute121` axis Z (the old value locked the left knee);
  - bicep inertia fixed;
  - ankle tie rods retyped to spherical (the user confirmed ball-and-socket joints);
  - passive damping 0 in Isaac and Newton.
- **Stack.** Isaac Lab 2.2, Isaac Sim 5.0 and rsl-rl 2.3.3, locally on Windows and on Brev Linux (`docs/BREV.md`,
  `tools/brev/*`).
- **Tools:**
  - calibration of the semantic (PR) joint space;
  - G1 / GMR / Kimodo / SOMA / LAFAN motion pipeline with a foot-contact stage (v4);
  - settle and quality gates;
  - BeyondMimic tracking (single clip and 78-clip library, plus NoState and Future variants);
  - H1-style velocity walking;
  - `dropbear_hg` SDK, Newton bridge and Unitree-style runner;
  - arm teleop (WebXR/keyboard);
  - GR00T tabletop push task and dataset (34 episodes, scripted baseline 84 %);
  - serial MJCF, GMR registration and SONIC motion_lib compatibility.

## Results (verified in sim; see OVERNIGHT_BREV and DEMOS for numbers and media)

| policy | where | result |
|---|---|---|
| wave_right v1 | laptop | 0 falls, body 2.8 cm |
| G1 dance_102 v3 (`take102_v3_s16`) | Brev | 0 falls, body 4.6 cm, anchor 8.4 cm; video `logs/brev/media/take102_reference_vs_cloud_policy.mp4` |
| LAFAN walk (`lafan_walk1_v4`) | Brev | 0 falls, body 2.8 cm; video `logs/brev/media/lafan_walk1_v4_policy_*.mp4` |
| Kimodo walk (`kimodo_walk_v4`) | Brev | 0 falls, body 3.7 cm; video `logs/brev/media/kimodo_walk_v4_policy_*.mp4` |
| wave v2 at 16/4 (`wave_v2_s16`) | Brev | 0 falls, body 5.6 cm (the 8/4-trained version failed at 32/4) |
| velocity walking (`vel_flat_kneehw`, 2998 it) | laptop | 0 falls in 9 command scenarios: fwd 0.3/0.5, turn ±0.5 and left 0.2 track; right and back under-track |
| LAFAN dance2 v4 | Brev | NOT learned: fails at frame 0 because it opens with the arms overhead |
| G1 dance_102 v4 (foot-pinned) | Brev | untrackable (calf motors at their limits); stopped |

Lessons:
- Train at 16/4 and also evaluate at the training setting.
- Episode-length dips at chunk starts are logging artifacts.
- Continuous-loop falls on non-cyclic clips (walks) are artifacts.
- G1-route v4 clips can put the ankle at its limits.

## Brev (still running at handoff)

- **Instance:** `dropbear-train`, 4× RTX 6000 Ada, $4.66/h, created 00:15 IST.
- **Watchdog:** `tools/brev/watchdog.py` (a detached Windows process) pulls results to `logs/brev/remote/` every
  30 min and **deletes the instance at 04:00 IST on 2026-09-26** (the user added $50; cap $130 in
  `logs/brev/budget.txt`, deadline in `logs/brev/deadline.txt`; watchdog restarted 16:34, PID 36708). Log: `logs/brev/watchdog.log`. Stop early by creating
  `logs/brev/STOP_AND_DELETE`.
- **Running (since 17:49; details in OVERNIGHT):**
  - GPU0 `kimodo_walk_w10_nostate_hw_reg` (from 21:42, from scratch): the deployable Kimodo text-to-motion walk.
    The previous job, `asap_walk4_hw_reg`, has converged: best walk on today's motors, 21/22 motors within rating,
    dashboard `asap_walk_hw_reg_4700`.
  - GPU1 `lib_nostate_kneeX10S2_v5` (from 18:48): the library with an X10-S2 knee. It is the A/B partner of
    GPU2, warm-started from the same `model_5200`.
  - GPU2 `lib_nostate_hw_v5`: the library tracker on `hw_v1`, using the width-fixed library v5.
  - GPU3 `asap_walk_nostate_hw_reg` (from 20:53, from scratch): the DEPLOYABLE ASAP walk (task
    `Dropbear-Tracking-Flat-NoState-v0`, `hw_v1` + regularizers).
  - The launch scripts are `tools/brev/swap_track.sh` (args: gpu name src_box src_exp src_run task motion_flag
    motion_path envs [profile] [extra flags]) and `swap_gait.sh`.
- **Library evaluations, done at 15:40:** see OVERNIGHT. `lib_v2_nostate` has 57/78 clips with 0 falls, on the
  idealized motors.
- **Access (WSL):** `ssh -F ~/.brev/ssh_config dropbear-train`. The brev CLI is `~/.local/bin/brev`; always pass
  `--no-check-latest` and `</dev/null`.

## The honest gap: why none of this works on the real robot yet

Details are in `docs/ACTUATORS.md` and `data/robot/actuators_datasheet_v1.json`.

The sim's motors are too strong:
- hips 200 N·m (real: X10-S2 hip pitch 100 N·m, X10 1:7 hip roll and yaw 40 N·m);
- ankles 80 N·m (real X8 Pro about 25 N·m);
- armature about 70× too small;
- no friction;
- motor speed up to 10 rad/s (the real X10-S2 manages 5.6 rad/s).

Masses are CAD values: 56.2 kg vs the user's estimate of about 60 kg without battery. The printed parts are PLA
gyroid at 35-50 % infill, so the CAD solid densities are likely too high for printed parts, while the motors dominate
the mass.

**User answers (2026-09-25).** The real actuator map is in `myactuator-can/AttemptToExplainActuators.md`:
- **the knee is a CEM60** (specs unknown, need the datasheet);
- hip roll is an X10 1:7, not the X10-S2 in the CAD;
- "Waist Pivot" means the hip-pitch X10-S2 motors;
- 48 V;
- CAN layout undecided (one ESP32 per subassembly under an Orin, or one bus; EtherCAT possible);
- the bench leg is not ready yet.

⚠️ The ESP32 sketch sets the comms-timeout in the wrong bytes (a safety bug). Ask the user before touching firmware.

## Phase 1 progress (2026-09-25, 11:45-12:40 IST)

Details are in ACTUATORS.md section 11 and DECISIONS.md.

- **Knee motor = EPS-CEM-60, provisional.** No datasheet exists; the vendor only publishes the CEM-15/25/45 (30:1,
  48 V). The model assumes 60 N*m peak, 34 rated, 60 rpm.
- **Speed-up linkages:** the knee joint gets only about 33 N*m and the elbow about 5 N*m.
- **Real-motor twin implemented:**
  - `robots/hw_motor_specs.py` + `robots/hw_actuators.py` (the `DatasheetMotor` explicit model);
  - profiles `hw_v1` (the user's map) and `hw_v1_cad`;
  - `--actuator_profile` on `train_locomotion`/`play_locomotion`, and on `train.py`/`play.py`/`run_chunked_training`
    for tracking;
  - both exports read the explicit models' own gains;
  - `play_locomotion` now logs torque statistics.
- **Zero-shot results:**
  - The idealized walker on the real motors: 18/18 falls in about 1.2 s. Its calves need 2× the real peak and its knee
    is pinned at the limit.
  - The wave tracker on `hw_v1`: 0 falls, the same accuracy.
  - Standing with plain PD falls in every profile, legacy included (a balance problem, not motor strength).
- **Brev GPUs 2-3, since 12:00:**
  - `vel_hw_v1_ft` and `vel_hw_v1_cad_ft`, warm-started from `vel_flat_kneehw_cloud/model_2189`;
  - launched with `tools/brev/swap_to_hw_twin.sh`;
  - the poll script is `scratchpad/hw_poll.sh`.
- **12:45 evaluation:**
  - `vel_hw_v1_ft` `model_2600` walks with 0/18 falls, but it is thermally over-limit: the knee is pinned at 60 N*m, and
    the left hip roll, right calf B and left elbow run at 2-2.5x their rated RMS.
  - The CAD A/B walks too and has the same pinned knee, so it was stopped.
- **From 12:51, GPU3 runs `vel_hw_v1_thermal_ft`** (`--thermal_penalty=-2e-4`, warm-started from `model_2786`;
  `tools/brev/swap_gpu3_thermal.sh`). GPU2 keeps running `vel_hw_v1_ft`.
- **14:00, the locked right knee** (ACTUATORS.md section 12). The policy hops on its left leg and commands the right
  knee crank to -300 deg into its stop.
  - Fix: `--gait_shaping` (target clamp, knee flexion in swing, swing clearance, hop penalty, torque rate).
  - Running on Brev: GPU2 `vel_hw_v1_gait_ft` (warm-started from `thermal_ft/model_3500`) and GPU3
    `vel_hw_v1_gait_scratch`.
  - Actuator dashboard: `play_locomotion --telemetry`, then `tools/render_actuator_dashboard.py`. The pipeline script
    is `tools/make_dashboard_locomotion.sh <ckpt> <profile> <tag>`.
- **16:20:** switched to reference-motion walking on the real motors: GPU0 `asap_walk4_hw`, GPU1
  `lafan_walk1_hw`, GPU2 `lib_nostate_hw`, GPU3 the velocity `gait3b_scratch`. The deadline moved to **18:35 IST**
  (about $85.4). The Kimodo walk reference crosses the feet (retarget artifact); see OVERNIGHT.
- **After 04:00 (2026-09-26):** `logs/hw_twin/end_of_night.sh` (detached) runs the library per-clip evals and the
  final dashboards automatically once the watchdog logs `deleted`; read `logs/hw_twin/end_of_night.out`. Then:
  - pull all four runs; the tracking dashboards come from `logs/hw_twin/run_dash_track.sh <run_dir> <npz> hw_v1 <tag>`;
  - pull both gait runs (the watchdog syncs `logs/rsl_rl`);
  - run `tools/make_dashboard_locomotion.sh` on the latest checkpoint of each. Check that the right knee angle curve moves and
    that the right foot contact is about 50 %;
  - compare the thermal run against the plain one on RMS / rated per motor;
  - run `play_locomotion --actuator_profile hw_v1` with torque statistics on the latest checkpoints;
  - render a video.
- **New tool:** `tools/hw_motion_feasibility.py`. 70/78 library clips are within the real motor speeds; the knee range
  is the bigger constraint.
- **2026-09-26 night (superseded the item above):** the end-of-night job is now `logs/hw_twin/end_of_night_v4.sh`
  (output `end_of_night_v4.out`). After the watchdog deletes the instance (06:30 IST), it renders clean dashboards of
  the four walk runs with each run's own motor profile, scans them into `logs/hw_twin/issue_scan_end_of_night.md`, then
  evaluates the library policy. Check first:
  - `docs/ISSUES.md` #19: did `--knee_stop_penalty` bring the knee `stop_load` below 5 % (LAFAN was 30-50 %)?
  - #20: did `--feet_slide_penalty` remove the right-foot touchdown skid (17.7 cm) and the scuffs? Compare GPU3
    `kimodo_cyclic_nostate_v1i_feet` against the GPU0 control `kimodo_cyclic_nostate_hw_v1i`.
  - #11: `torque_jump_any_motor` of the `hw_v1i` runs against the hw_v1 baseline (27 %).
  - Run the scanner on any new rollout: `python tools/telemetry_issue_scan.py <telemetry.npz> --md out.md`. Record
    rollouts with `play.py --telemetry` or `play_locomotion.py --telemetry`.
- **New velocity runs:** pass `--target_clamp_margin_deg=0` (ISSUES #23).
- **Next gains A/B:** elbow kp 150-300 (ISSUES #24).
- **Night 2 outcome (07:22 IST):** 4 deployable walks pass the HW gate on the real-motor twin. The best is
  `logs/brev/remote/dbw3/.../kimodo_ts119_allfix_smooth/model_5800`; the LAFAN one is
  `dbw1/.../lafan_cyclic_nostate_v1i_kstop_feet/model_3100` (CEM-60 knee). The CEM-60 library tracker holds 41/78
  clips (v6). The Brev instance still has to be DELETED (the brev login expired): run `brev login` or use the console.
- **Before any next cloud run:** check the brev auth non-interactively (`brev ls </dev/null`); the watchdog cannot
  delete without it.
- **Text -> motion, live (2026-09-26 night):** see `docs/TEXT_TO_MOTION.md`.
  - `tools/text_to_motion.py` does retrieval with the local MiniLM, hardware-safe clips only.
  - Live splicing: `play.py --live_script / --live_dir` (Isaac) and `policy_runner --live_dir` (deploy).
  - The Newton bridge takes `--motor-profile` (training motor law) and `--target-ramp-ms`.
  - Demo: `logs/brev/media/live_text_demo3_dashboard.mp4`.
  - Kimodo-G1 + Llama-3 downloaded (the user accepted the license on 2026-09-26); generation works (6.1-6.5 s).
  - **Real time, 2026-09-26 17:01:** `bash tools/live_session.sh [minutes]`, then http://127.0.0.1:8765: type a
    motion, the motor-twin robot performs it 8.5-10.6 s later; physics 1.0x real time between prompts (Isaac CPU
    pipeline), 0 falls on 5 prompts. Memory is the constraint on this laptop (ISSUES #30-#34). Video of that session:
    `logs/brev/media/live_final_dashboard.mp4`.
- **Terrain:** `train_locomotion.py --terrain rough` (docs/TERRAIN.md). It is blind, uses a curriculum of stairs up
  to 10 cm, and is smoke-tested but not trained.
- **Manipulation:** the existing fixed-base tabletop PUSH task (tasks/tabletop) still uses idealized arm motors and an
  elbow kp of 600 (above the MIT limit of 500). Next: an hw_v1 arm profile; walking plus carrying later. Grasping needs
  a gripper on the robot.
- **Deploy path (ISSUES #26):** the deploy FSM now applies the sidecar's `target_clip`. For a policy trained with
  target interpolation (`target_interp_steps` > 0, e.g. `hw_v1i`), run the sim2sim bridge with
  `--target-ramp-ms 20`. The ESP32 firmware must implement the same ramp (`source/dropbear_wbc/sdk/target_ramp.py` is
  the reference); ask the user before touching firmware.
- **New tracking runs:** add `--target_clamp_margin_deg=0` (ISSUES #25). Add `--knee_stop_penalty=-10` /
  `--feet_slide_penalty=-1` if the night's A/B confirms them.

## Next (Phase 1: digital twin v2)

1. Get the CEM60 knee datasheet and model from the user.
2. Correct `actuators_datasheet_v1.json` with the user's map: hip roll is an X10 1:7, the knee is a CEM60.
3. Implement the actuator profile `hw_datasheet_v1`:
   - real peak torque and torque-speed curve, armature and friction;
   - gains deployable within the MIT-mode limits kp ≤ 500, kd ≤ 5, |τ| ≤ 24 N·m, or an ESP32-side PD;
   - latency randomization.
4. Mass pass: scale the printed-part masses for PLA gyroid infill, keep the motor masses, and match the user's
   ~60 kg (weigh when possible).
5. Per-joint motor-speed gates in `validate_motion_npz` from the datasheet, plus a motor-at-limit fraction gate.
6. Retrain walking and the tracker on twin v2 and see what survives.

Then:
- **Phase 2:** bench identification (one leg).
- **Phase 3:** firmware impedance mode (fix the timeout bug first).
- **Phase 4:** hardware-ready policies (NoState, randomization, sim2sim gate) and staged bring-up.
- **Phase 5:** text-to-motion, teleop and VLA on top.
