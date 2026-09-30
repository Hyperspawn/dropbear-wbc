# dropbear-wbc decision log

Newest first. Each entry gives the decision, the evidence, who made it, and what would reverse it.

## 2026-09-25: Knee actuation is undersized for a human-style bent-knee walk (sim evidence; lead, 18:55)

- **Evidence:** ACTUATORS.md section 13. The same LAFAN walk tracker was trained with the same init and seed:
  - with the CEM-60 knee: clipped 82-85 % of the time, RMS 1.65x rated;
  - with an X10-S2 knee: clipped 14 %, RMS 1.03-1.15x rated, tracking twice as accurate.
  - About 30 N*m RMS is needed at the knee joint. The four-bar is a 1.79x speed-up, so the crank needs about 55 N*m RMS.
- **Recommendation to the user:**
  - a knee motor with at least about 50 N*m continuous at the crank (X10-S2 class), or a four-bar with less speed-up;
  - until then, expect straighter-legged gaits on the real robot.
- **Caveat:** the CEM-60 rated torque (34 N*m) is extrapolated; no datasheet exists. Confirm on a bench.
- **Refined at 19:45:** the ASAP walk with the thermal and torque-rate regularizers runs on the CEM-60 knee with
  all leg motors within their ratings and 4.3 cm tracking. It is an upright gait (mean knee 14 deg; LAFAN's is
  23 deg). So the recommendation applies to crouched or deep-knee motions. Upright walking fits the current
  motors.

## 2026-09-25: The retarget foot-contact stage enforces a 10 cm minimum stance width (lead, 16:40)

- **Evidence:** Dropbear's pelvis is narrow: the soles are 0.138 m apart at rest. G1-route references therefore land
  the feet on top of each other.
  - The Kimodo walk v4 has feet less than 5 cm apart in 66 % of frames and crossed in 44 %.
  - Across `accepted_v2`, 6 of 78 clips have feet less than 5 cm apart in more than 20 % of frames (Kimodo walk, SOMA
    A057 walk, LAFAN run1, ASAP walk level 1/2 and step_forward level 2).
  - Trackers reproduce the crossing faithfully, and self-collision is off in the sim.
- **Decision:** `FootContactParams.min_stance_width_m = 0.10` is the default.
  - `build_targets` widens the retargeted sole-centre paths symmetrically along the pelvis-heading lateral axis before
    the stance targets are built.
  - The report is `foot_contact_stage.targets.min_width`. Set the value to 0 to reproduce the old behaviour.
- **Result on the Kimodo walk** (`data/motions_w10/kimodo_g1/output_walk_v4w10.npz`):
  - crossed 44 % -> 0 %; width mean 2.2 -> 12.4 cm, min 3.7 cm;
  - stance residual after IK at most 3.6 mm; settle closure 0.47 mm; the validator accepts it;
  - frames over a motor's no-load speed went 2.4 % -> 0.8 % (right knee crank).
- **Second pass, 17:05 (`_widen_swing_targets`).** A stance target is locked where its phase started, so the
  first pass, which runs on the retargeted paths, cannot see it.
  - The new pass pushes a SWING foot that comes closer than 10 cm to the other foot's final target out by the whole
    deficit (half each when both swing; smoothed over 5 frames). The report is `targets.min_width_targets`.
  - Settled results, crossed frames v4 -> v5:
    - ASAP walk level 1: 11.9 % -> 0 %; level 2: 15.7 % -> 0 %;
    - Kimodo walk: 43.8 % -> 0 %;
    - SOMA A057 walk: 27.4 % -> 6.2 %. The remainder is double support with both feet locked.
  - Mean width is now 12.6-16.5 cm, up from 2-13.
- **Library v5:** `data/motions_v5`, all G1-route clips rebuilt, re-settled (`_v5`) and validated. The LAFAN / GMR
  clips keep their v4 files.

## 2026-09-25: Real-actuator twin (`hw_v1`) is the new target for walking; idealized-sim policies are not hardware evidence (lead, midday)

- **Evidence:**
  - The `stiff_knee_hw` walker is 0/18 falls on its own sim motors.
    - On the datasheet motors (`hw_v1`) it is 18/18 falls, median 1.2 s.
    - It relies on calf torque of 41-51 N*m p95 (the real peak is 25) and a knee crank pinned at its 100 N*m limit.
    - Logs: `logs/hw_twin/zeroshot_*.json`.
  - The knee four-bar (G 1.79) and the elbow linkage (G 4.8) are speed-ups. With a CEM-60 the knee joint has about
    33 N*m; the elbow has about 5 N*m.
- **Decision:**
  - The velocity task gets opt-in profiles `hw_v1` (the user's map) and `hw_v1_cad`.
  - The Brev GPUs 2-3 fine-tune the walker on both.
  - Walking and tracking claims about the real robot are made only on an `hw_*` profile from now on.
- **Open:**
  - the CEM-60 datasheet (provisional values in `hw_motor_specs.py`);
  - bench tests B1-B5, the real weight and motor thermal limits;
  - a tracking-task `hw_*` profile.
- **Reverses if:** bench identification shows the datasheet values are far off. In that case, re-fit
  `MOTOR_MODELS`.

## 2026-09-25: G1-route v4 (foot-pinned) Take_102 is untrackable; the cloud dance run uses the v3 reference (lead, overnight)

- **Evidence (Brev, 4096 envs):**
  - `take102_v4` at 16/4 peaked at a mean episode length of about 91 near iteration 220, then collapsed to about 10-18 by
    iteration 844. Nearly every termination was `ee_body_pos`, joint error was about 3-3.8 rad, and action std was 0.29.
  - The same v4 clip at 8/4 (`diag_take102v4_s8`) showed the same rise and collapse (109 -> 37), so the solver setting
    is not the cause.
  - The v3 reference at 16/4 (`diag_take102v3_s16`) kept learning: 7 -> 124 -> 194 by iteration 256.
- **Likely cause (INFERRED):** the stance-leg IK pinned the soles flat. Because the knee only reaches 0-48 deg, it
  drove a calf motor within about 1 deg of its stop on about 26 % of the frames (foot_contact PROGRESS). The ankle
  then has no authority left to balance.
- **Decision:**
  - GPU2 stopped `take102_v4` and skipped `gangnam_v4`, which uses the same route.
  - It now trains `lafan_dance2_v4` (GMR route; the GMR-route walk learns to about 490/500), then `take102_v3_s16`,
    then walking.
  - The v4 G1-route clips stay in the library run, where the per-clip eval will show which are trackable.
- **Fix to do:** in `motion/foot_contact.py`, let the heel lift (toe pivot) in late stance, and keep a calf-motor
  margin of 5 deg or more from its limits. Then re-settle and re-validate. Add a "motor-at-limit fraction" gate to
  `validate_motion_npz`.

## 2026-09-24: Retargeted clips get a foot-contact stage; stance = a PLANTED source foot (foot_contact)

- **Decision.** Every retargeted clip (G1-intermediate and GMR routes) runs `dropbear_wbc.motion.foot_contact` by
  default (library v4; CLI `--no-foot-contact` = the old route). The stage re-solves the pelvis (height, small xy /
  roll / pitch) and the 12 leg motors per frame, jointly over the clip, so that every stance sole is flat on z = 0 at
  a world-locked pose (xy from the phase's first frame, yaw from the source foot clamped to the hip-yaw band) with the
  PLANT forward model (MeasuredFK; foot position error <= 3.2 mm against physics, where the serial MJCF has up to 37 mm
  and the semantic leg model up to 112 mm of sole-z error). Swing soles keep >= 1 cm clearance; motors stay in the
  motor box and under 0.18 rad per 50 Hz frame. Output at 50 Hz. Settled NPZs of v4 clips are `<clip>_v4.npz`; the
  previous CSVs are kept as `<clip>_v3.csv`.
- **Contacts must mean "planted".** The stage world-locks every stance foot, so the contact rule needs a speed term:
  the GMR route's height-only human rule merged LAFAN's low-clearance walking steps (85-90 % stance, source ankle
  drifting 0.4-0.6 m per "stance"), which forced 0.6 m pelvis corrections. Now: 3 cm / 5 cm height AND 0.3 / 0.6 m/s
  hysteresis on the slower of toe and ankle (drift per phase 1-2 cm).
- **Why.** Floating (3-14 cm) and sliding stance feet failed the settle's contact gate on every whole-body clip; the
  cause was geometric (angle copying / serial-surrogate IK onto a leg with different proportions and a 0-48 deg knee),
  so it has to be fixed kinematically before the settle. Evidence: `logs/foot_contact/summary.md`.
- **Reversal.** A physics-based retarget (e.g. tracking-policy rollouts replacing kinematic references) or evidence that
  policies track better from the unmodified (floating) references.

## 2026-09-24: Velocity task knee gains = crank kp 600 / kd 20, effort 100 N*m (`stiff_knee_hw`) (locomotion)

- **Decision.** `Dropbear-Velocity-Flat-v0/-Play-v0` use their own actuator profile: `flat_env_cfg.ACTUATOR_PROFILES`,
  default `stiff_knee_hw`.
  - The knee crank gets kp 600, kd 20 and effort 100 N*m.
  - Everything else keeps the legacy values: hips 150/5/200, ankles 80/4/80, arms 50/2/40.
  - The action interface is unchanged (standing-pose offset, knee scale 0.375 rad).
  - The tracking task and its trained policies keep the legacy knee (200/12/300).
  - The export writes the live gains into the sidecar (`joint_stiffness`, `joint_damping`, `dropbear_velocity.effort_limit`,
    `.actuator_profile`), like Unitree's per-joint `deploy.yaml` stiffness.
- **Why.**
  1. *Reflected stiffness.* The knee is a four-bar driven by a crank. The calibration LUT slope dknee/dcrank is 1.79 (left)
     and 1.82 (right) at the stand, and 1.36-2.21 over the range. The legacy crank 200 / 12 is therefore only 62 / 3.7 at
     the knee, where H1's knee has 200 / 5. 600 / 20 on the crank gives ~186 / ~6.2 at the knee, which is H1-equivalent.
  2. *Static demand.* Virtual work over the calibrated leg-length table (`geometry.per_side.*.hip_ankle_distance_vs_knee`)
     with half the 557 N weight per leg gives, at the 20 deg standing knee:
     - 42 N*m at the knee, or 75 N*m on the crank per leg, in double support; about 2x that (~150 N*m crank) on one leg;
     - at 10 deg knee: 36 / 72 N*m crank.
     Gravity then acts as a negative knee stiffness of ~105 N*m/rad, which exceeds the legacy reflected 62: the legacy
     PD cannot even statically stabilize the pose, so the policy must supply all of the stiffness at 50 Hz.
  3. *Probe* (`logs/locomotion/probe_knee_authority_v1.json`; zero action, 8/4). Stiffer knee PD alone only delays the
     zero-action collapse: 0.72 -> 1.02 s from crank kp 200 to 1200. It never saturates, and hip pitch kp 300 adds
     nothing. The reset is not a whole-body equilibrium, so a policy is needed with any gains.
  4. *A/B* (60 PPO iterations each, seed 42, 2048 envs; `logs/locomotion/ab_summary_v1.json`). Mean episode length
     (last 10 iterations):
     - legacy 99;
     - 600/20/300: 165;
     - 600/20/100: 92.
  5. *Effort must be physically plausible.* The knee crank is driven directly by an RMD-X10-S2 V3 (MyActuator X10-100,
     1:35), with 50 N*m rated and 100 N*m peak. Sources:
     - `github.com/Hyperspawn/myactuator-can/X10-100/.../X10-100(RMD-X10-S2 V3).pdf`;
     - `myactuator-can/Actuators1.csv:16` "Knee Bender, RMD-X10-S2 1:35";
     - `dropbear_control/integrations/gr00t_wbc/config/dropbear_embodiment.json:104,138`;
     - USD knee joint = `LL_RMD_X10_S2_MIR4__3_Stator_1` -> `..._Rotor_1`.

     The legacy 300 N*m is 3x that peak. 600/20/300 learns fastest, but it would teach the policy torques the motor
     cannot produce. With the 100 N*m cap, learning was not worse than legacy after 60 iterations. The cap forces a
     straighter stance knee (<= ~10 deg in single support), which is how a person walks too.
- **Assumptions (UNVERIFIED on hardware).**
  - The X10-S2 drives the crank with no extra reduction. `dropbear_docs` legs/actuators.md instead calls the knee
    motor a "CEM-60" (60 N*m, 400 N).
  - The peak (not the 50 N*m rated) torque is the sim limit.
  - kp 600 / kd 20 are sim gains, not system-identified: the embodiment file says `implicitPdGains`
    "unverified-requires-system-identification".
- **Not changed, flagged.** The legacy hip and ankle efforts also exceed the datasheet peaks:
  - Hips 200 N*m. The CAD and CSV give hip roll = X10-S2 (100 peak) and hip yaw = X10 1:7 V3 (40 peak). The hip-pitch
    motor is ambiguous across the docs (X10 / X10-S2 / CEM-60).
  - Ankle motors 80 N*m on RMD-X8 Pro 1:9 (25 N*m peak). That is 46 N*m ankle pitch / 61 N*m roll at the stand
    (calibration Jacobian), against 147 / 196 N*m in sim.

  A `hw_all` profile is the next A/B. The Newton bridge still clips at `sdk.motors.EFFORT_LIMIT` (knee 300), so it is
  more lenient than training.
- **Reversal.**
  - A hardware measurement of the knee transmission or torque (e.g. a CEM-60 linear actuator with a lever).
  - The `stiff_knee_hw` run failing to learn walking while `stiff_knee` succeeds: then report the knee as
    torque-limited for walking and revisit the hardware.

## 2026-09-24: First "useful task" for GR00T = PUSH a block into a target zone, no gripper (tabletop)

- **Decision.** The first autonomous tabletop task is **non-prehensile pushing**: push a 5 cm, 100 g block into a
  10 cm target square on a table with the hand body. Robot base fixed (`fix_root_link`), legs held at the calibrated
  standing pose, one arm per episode (the arm whose workspace contains the zone). Gym id `Dropbear-Tabletop-Push-v0`,
  docs/GROOT.md. **No sim-only gripper.**
- **Why not a sim-only gripper.**
  - Dropbear has no hand or gripper, and none is designed. A gripper added in simulation would produce data and a
    policy for a robot that does not exist; nothing learned about grasping would transfer.
  - It would change the plant (new bodies/joints on the hand link). The USD is the plant authority (CONTRACTS 0), and
    every in-memory spawn fix so far corrects a defect toward the real robot; an invented end effector would not.
  - A 5-DoF arm plus a parallel gripper also needs grasp-pose planning and closure detection, which multiply the
    failure modes of the scripted baseline. Pushing tests the same pipeline (cameras, language, state/action, dataset,
    fine-tune, closed-loop eval) with fewer invented parts.
- **Why pushing works on this arm.** The hand body is a rigid cylinder (radius ~4.3 cm, length 16.75 cm along the
  forearm axis; `logs/tabletop/geometry_probe.log`), which is a usable pusher. With the table top at root z 1.25 m
  (37 cm below the shoulder centres, the best of 1.20 / 1.25 / 1.30 m in `tools/tabletop_workspace.py`) and the hand
  axis tilted 60 deg forward from vertical, each arm has ~0.06 m^2 of pushable table (`logs/tabletop/workspace.log`).
- **Consequences.**
  - The two arms' pushable regions are disjoint (each arm works in front of / outside its own shoulder: shoulder-roll
    adduction limit), so every placement is single-arm, with the side drawn at random.
  - Pushing is 2-D and the block can rotate or slip off the hand, so the scripted baseline is closed-loop on the block
    pose (re-aim every step, re-approach up to 3 times).
  - GR00T sees the head camera (anchor-mounted, torso-rigid) and optional wrist cameras. State and action are the 10
    semantic arm angles; the legs are not in the action space.
- **Reversal.** A real end effector (hand or gripper) designed for Dropbear and added to the USD: then add a pick-and-place
  task next to this one. A WBC policy that can stand while the arms work: then drop `fix_root_link`.
- **Head camera mount (same day, after the GPU smoke).** The ego camera is a rigid child of the anchor body
  (torso-fixed), 1.2 cm in front of the visor's front face at eye height (root (0.135, -0.07, 1.88) m), pitched 68 deg
  down, 92 deg horizontal FOV, 640 x 480. The first pose (0.10, 1.78) was inside the visor mesh (uniform frames,
  `logs/tabletop/smoke_v2.json`). Of the three candidates rendered (`logs/tabletop/camera_probe_v1`,
  `media/camera_probe_v1_head_candidates.png`), a mount 9 cm in front of the face gave the cleanest view, but it has
  no physical counterpart; the visor-front pose is where a real head camera would sit, and the block, zone and pushing
  hand stay visible for both arms (the chest handle covers only the lower centre). Reversal: the real robot's head
  sensor pose, once it exists.

## 2026-09-24: Multi-clip tracker = motion-library command + runtime-reference export (multiclip)

- **Decision.** The step from one policy per clip (BeyondMimic) towards a SONIC-style general tracker is a
  motion-LIBRARY command (`mdp/library_commands.MotionLibraryCommand`, CONTRACTS 5.3).
  - **MDP.** It keeps the single-clip MDP and command layout (44 values; the policy observation is still 125).
  - **Sampling.** RSI draws per (clip, 1 s bin): BeyondMimic's failure bins inside each clip, plus a clip-level
    mixture of 0.5 x a duration-weighted prior and 0.5 x the measured failure hazard.
  - **Future frames.** The command with +0.1/+0.2 s future frames is a separate task id (`-Future`, 132 values;
    observation 213). It is an A/B, not the default.
  - **Export.** Library policies bake no reference (`policy.onnx` + sidecar `motion.source: runtime`). The runner feeds
    any NPZ/CSV that meets the sidecar's `motion.requirements`.
  - **Manifests** (added 21:35). A manifest is generated from the validator verdicts
    (`tools/build_motion_library_manifest.py`), not written by hand. Each clip is pinned by the SHA-256 of its NPZ, so a
    re-settled clip cannot silently change a library that is already training; new clips go into a new `accepted_vN`.
- **Why.**
  - **Unchanged layout.** The command layout and MDP stay as they are, so single-clip checkpoints, the deploy
    observation builder and the sim2sim tooling apply unchanged.
  - **The single-clip path stays untouched.** It is a new module, not an edit of `commands.py`. A one-clip library
    reproduces the single-clip sampler exactly (test).
  - **Why the hazard.** Failure counts alone would favour clips that are sampled often. The clip-level hazard (failures
    per env-step) measures difficulty independently of how often a clip is sampled. The 0.5 prior share keeps every
    clip in the mix.
  - **Why runtime.** A baked reference would tie the export to one clip. The runtime reference is the interface a
    general tracker (and later teleop/VLA streams) needs.
- **Reversal.**
  - Evidence that per-clip failure counts, uniform clip weighting or a different future window train better on a
    bigger library.
  - SONIC-style body-keypoint commands instead of motor-space commands (a new task id, not a change of this one).

## 2026-09-24: Passive-joint damping of the contract plant is 0 in every simulator (review_fixes)

- **Decision.** The 63 passive DOFs have no damping (armature 0.01 only): Isaac `make_dropbear_cfg(passive_damping=0)`,
  Newton `sdk.motors.PASSIVE_DAMPING = 0` (bridge `--passive-damping` default 0, was 50).
- **Why.** Measured, not assumed: Isaac never applied the legacy 50 N*m*s/rad (a drive parameter on joints without an
  authored DriveAPI; dynamics identical for 0 / 0.5 / 5 / 50, `logs/review_fixes/passive_damping/`), while Newton did,
  which made its four-bar knees and elbows ~10^3 N*m*s/rad stiffer than the plant every policy was trained on. Aligning
  Newton to Isaac (instead of making Isaac apply 50) keeps every trained policy, the calibration and the settled clips
  valid. With it the wave policy's sim2sim completes at the default 2 ms step (with stiffer closures), CONTRACTS 0.3.
- **Reversal.** A hardware measurement of real passive-joint friction/damping; then author it in BOTH simulators
  (`make_dropbear_cfg(passive_drive_api=True, passive_damping=d)` in Isaac covers the revolute passive joints only).

## 2026-09-24: Ankle tie rods are ball-and-socket (spherical) at both ends. CONFIRMED BY USER.

- **Decision.** Retype `LL/RL_Revolute111` and `LL/RL_Revolute112` (crank → tie-rod loop closures) to
  `PhysicsSphericalJoint` in memory at spawn. This is a default-on plant fix in Isaac and Newton,
  and the USD on P: is not modified.
- **Why.** As authored, those joints are revolute joints parallel to the calf-motor axis. That keeps the rods
  planar and over-constrains the parallel ankle (Grübler mobility 0):
  - ankle roll is limited to about ±8°;
  - closure gaps open by 8–21 mm under differential calf-motor motion.

  With spherical ends, the entire calf-motor grid closes to under 0.01 mm, and roll reaches about −33..+23°.
  Evidence: `CONTRACTS.md` §0.2, `logs/calibrate_settle/diag_spherical_tierods_*`.
- **User confirmation (2026-09-24).** "yes - ball and socket joint". The real hardware uses ball-and-socket
  joints at the tie rods, so this fix moves the simulation plant closer to the robot. It is not a
  simulation convenience.
- **Reversal.** Only if a hardware measurement shows otherwise. The opt-out flag remains for A/B comparison.

## 2026-09-24: Brev cloud training only after local results

- **Decision.** The user wants quick local training results first, on the RTX 4080 with small env counts. Brev A100s come
  only after the pipeline has been shown to work locally.
- **Setup.** The Brev CLI v0.6.323 is installed in WSL Ubuntu (`/usr/local/bin/brev`), but its session is logged out.
  The user will run `brev login` themselves. Claude never handles or stores API keys.

## 2026-09-23: Plant = user's P: USD (sha 45586414…) with in-memory spawn fixes. Stack = Isaac Lab 2.2 + rsl-rl 2.3.3.

See `CONTRACTS.md` §0, §0.1 and the project memory. Codex's control-v1 plant is a different spanning-tree cut and
elbow motor assignment. It is not interchangeable with this one.
