# dropbear-wbc interface contracts (v1)

`dropbear-wbc` brings the Unitree H1/G1 whole-body-control ecosystem to Dropbear. It covers
BeyondMimic/unitree_rl_lab motion tracking, G1 motion libraries, a Unitree-style
low-level SDK, Newton sim2sim and, later, GMR, SONIC and teleop. Every component
implements against the contracts below. When a contract needs to change, edit this
file and describe the reason. Do not diverge silently.

## 0. Ground rules

- **Plant authority.** The user's verified USD is
  `$DROPBEAR_USD`,
  SHA-256 `45586414b065cd982d487cbd868fe982108b3b8ccec64d3dfcf629652ed8db0f`.
  - It is read-only. Never modify it.
  - Code may override the path through the environment variable `DROPBEAR_USD`.
  - Its structure: 93 rigid bodies, 117 joints, 28 authored drives (22 body motors plus 6 neck lead
    screws), and 27 loop closures excluded from the articulation. PhysX solves the
    closures as maximal-coordinate joints.
  - Joint and body dump: `docs/usd_tree_45586414.json` (limits in degrees, as authored).
- **Isaac runtime.** Isaac Sim 5.0 (`C:/isaac-sim`) with Isaac Lab 2.2.0 (`the Isaac Lab 2.2.0 checkout`).
  - Launch with `C:/isaac-sim/python.bat -u <script> --headless ...`.
  - Data accessors are torch tensors, which is the Isaac Lab 2.x API.
- **RSL-RL.** Vendor rsl-rl-lib **2.3.3** into `third_party/pydeps`. Our scripts prepend it to `sys.path`.
  - Never pip-install into `C:/isaac-sim/kit/python`. It contains 5.0.1, which is API-incompatible.
- **Disk.** C: has about 11 GB free.
  - Never write datasets, logs or caches to C:.
  - Everything goes under `<repo>` (logs in `logs/`, data in `data/`).
- **Concurrency.** A separate Codex agent works in `$DROPBEAR_CONTROL_DIR` and
  `dropbear-research` (a separate, read-only research checkout).
  - Treat both as **read-only**. Import code from them by path if useful, but never edit them.
  - `$DROPBEAR_UPSTREAM/*` holds read-only reference clones of upstream projects.
- **GPU lock.** Only one Isaac Sim / Newton GPU process may run at a time.
  - Acquire the lock by creating `.locks/gpu.lock` exclusively. It holds the owner name, PID and start time.
    PowerShell: `New-Item -ItemType File .locks/gpu.lock -ErrorAction Stop`.
  - If the lock exists, poll every 30 s for up to 60 min.
  - Always delete the lock in a `finally` block.
  - For tests, use headless mode and small `num_envs` (≤ 64 unless measuring throughput).
- **Evidence.** Every claim of "works" needs a command and its output log saved under `logs/`.
  Keep rejected results. Never present finite simulation as a successful behavior.

### 0.1 Plant defects and spawn-time fixes (added by the robot/task component, 2026-09-24)

Reason: as authored, the USD cannot move the way the robot does. The left knee is locked, the hip
motors stick, and three free bodies sit inside the knees. Each simulator must apply the same
fixes **in memory, at load time**, or its results are not comparable. The USD file on P: stays
untouched. The Isaac implementation is `dropbear_wbc.robots.spawn.apply_stage_fixes`, selected
by `make_dropbear_cfg(...)`. Evidence (before and after each fix):
`logs/robot_task/inspect_articulation_v1_usdfriction_orphans.*`, `..._v2.*`, `..._v3.*`,
`usd_joint_attrs_probe.log`, `usd_inertia_probe.log`.

- **Orphan bodies.** `LL_skateboard_bearing__10__1`, `LL_skateboard_bearing__11__1` and
  `RL_skateboard_bearing__11__1` have colliders but no joints. PhysX simulates them as free
  bodies inside the knee, and self-collision filtering does not cover them. **Deactivate them.**
  - The articulation therefore has **90 links**; the other 3 of the 93 USD rigid bodies are these
    orphans. NPZ `body_names` has B = 90.
  - It has 91 DOFs: 22 motors, 6 neck screws, and 63 passive DOFs. `*_Revolute115/117` are
    spherical joints and appear as `:0`, `:1` and `:2`.
- **Joint friction.** The USD sets `physxJoint:jointFriction` to 0.5 on
  `PG_left/right_leg_pitch/roll` and to 0.1 on `LL/RL_Revolute87`. PhysX scales this by the joint
  constraint force, so the loaded hips barely move: about 2% of a 0.2 rad step in 2 s (v2).
  After the fix they reach 96–99% in 0.25 s (v3). **Set it to 0.**
  - Isaac Lab 2.2's `ImplicitActuatorCfg.friction` does nothing on Isaac Sim 5.x.
    `write_joint_friction_coefficient_to_sim` edits a copy of `get_dof_friction_properties()`
    and never calls the setter. So override the USD attribute instead.
- **Left-knee closure axis.** `LL_Revolute121` is authored with axis **X**. Its mirror
  `RL_Revolute121` and every other knee four-bar joint use **Z**, with identical local frames.
  The X axis locks the left knee: a 0.3 rad step reaches 0.3% (v1, v2). **Override it to Z.**
  The left knee then behaves like the right one (v3).
- **Zero inertia.** `LH_bicep_1` and `RH_bicep_1` have diagonal inertia (2e-6, 0, 2e-6). PhysX
  rejects zero components for articulation links. **Raise them to 2e-6 kg·m².**
- **Kept as authored.** `physxJoint:maxJointVelocity` is 572.96 deg/s (10 rad/s) on the motors.
  The disabled `debugging_joint` produces a harmless warning.
- **Legacy config note.** The legacy `DROPBEAR_CFG` sets `max_angular_velocity=50.0`, and Isaac
  Lab reads that in **deg/s**. A spun body's speed decays from 3 to 0.3–0.5 rad/s within 1 s,
  against about 3.6 rad/s with 50 rad/s ≈ 2865 deg/s (inspect v1–v3, `angular_velocity_cap`).
  Our config uses 2865 deg/s.

### 0.2 Over-constrained parallel ankle: spherical tie-rod closures, ADOPTED DEFAULT (added by the calibration/settle component; adopted by the GPU pipeline component on the lead's decision, 2026-09-24)

**Status: adopted default, pending user hardware confirmation.** Every simulator retypes `LL/RL_Revolute111` and
`LL/RL_Revolute112` to `PhysicsSphericalJoint` **in memory at load time**, in the same mechanism as the 0.1 fixes.
The USD file on P: stays untouched.

- **Reason.** As authored, the ankle has Grübler mobility 0 instead of 2 (details below): PhysX binds on any foot
  roll, which limits ankle roll to about ±8 deg and stalls a single calf motor at about 50 % of a step. The real
  hardware almost certainly has rod-end (spherical) bearings there. **The user should confirm this on the robot.**
- **Implementation.**
  - Isaac: `dropbear_wbc.robots.spawn.DropbearUsdFileCfg.spherical_joint_overrides` (`retype_closure_to_spherical`
    fails closed on anything that is not an excluded loop closure), set by `make_dropbear_cfg(authored_ankle_tierods=None)`
    and by `isaac.quasistatic.QuasiStaticCfg` (calibration and settle).
  - Newton: `newton_sim.usd_fixes.apply_contract_fixes(..., authored_ankle_tierods=None)` via `PlantConfig`. SolverMuJoCo
    then maps each of the four closures to 1 CONNECT equality instead of the revolute's 2
    (`logs/gpu_pipeline/newton_ankle_probe.json`: 47 -> 43 equality constraints).
  - The joint list is `dropbear_names.SPHERICAL_JOINT_FIXES`. The articulation is unchanged (91 DOFs, 90 bodies),
    because the four joints are excluded from it.
- **Opt-out.** `DROPBEAR_AUTHORED_ANKLE=1` in the environment, `make_dropbear_cfg(authored_ankle_tierods=True)`,
  `QuasiStaticCfg(authored_ankle_tierods=True)`, `PlantConfig(authored_ankle_tierods=True)`, or `--authored-ankle` on
  `tools/calibrate_semantics.py`, `tools/settle_motion.py` and `tools/newton_bridge.py`. The opt-out plant is
  a variant, not the contract plant.
- **Provenance and fail-closed checks.**
  - The calibration records `authored_ankle_tierods` and a `plant_variant` string.
  - Settled NPZs record `meta.authored_ankle_tierods`. NPZs without the key predate the adoption and count as
    authored (`motion_npz.npz_authored_ankle`).
  - `motion_npz.validate_provenance(expected_authored_ankle=...)` refuses an NPZ from the other variant, because its
    settled passive ankle joints violate the other plant's closures. The tracking env passes the live spawn config.
- **Evidence** (all under `logs/gpu_pipeline/`):
  - Sweep `calib_sweep_v3.log`: closure gap max 1.19 mm, 0 samples > 3 mm. On the authored plant the max was
    21.3 mm, with 2410 samples > 3 mm.
  - Verify `calib_verify_v3.log`, 82 poses: gap max 1.17 mm (authored: 2.50 mm); map vs physics max 0.40 deg.
  - Ankle grid: 100 % valid (authored: 38 % / 34 %).
  - Comparison: `calib_v3_vs_v2_report.log`.
  - The old calibration is kept as `data/calibration/snapshots/dropbear_semantic_calibration_1a7161d3.json`.
- **Side effect: idle rod spin.** With both rod ends spherical, each tie rod can spin freely about its own axis (an
  S-S link). This does not move the foot, but the rod-end joints `*_Revolute115/117` can drift to arbitrary angles
  when settled under gravity (±1.8-2.8 rad in `data/motions/smoke/dropbear_static_stand.npz`). Consumers must not
  read meaning into those passive DOF values. *Added 2026-09-24 (review fix):* the spin is **undamped**, not a slow
  drift: Isaac applies no passive-joint damping (0.3), so a rod given 3 rad/s keeps 2.84 rad/s for 1 s with any
  configured damping (`logs/review_fixes/passive_damping/passive_damping_nodrive.json`).
- **Newton note.** MuJoCo's soft equalities never bound the authored ankle the way PhysX does: a single calf-motor
  step reaches 84 % (authored) vs 83 % (spherical) in Newton, against 53 % in PhysX (`newton_ankle_probe.json`).

The original analysis (calibration/settle component), kept for reference, follows.

- **Cause.** The ankle's crank-to-tie-rod joints `LL/RL_Revolute111` and `LL/RL_Revolute112` (loop closures) are
  authored as revolute joints whose axis is parallel to the calf-motor axis. Each tie rod therefore stays in a
  plane. The rod-to-foot ends (`*_Revolute115/117`) are spherical, but the foot attachment points sit about 16 mm
  above the ankle-roll axis. Any foot roll moves them sideways, which the planar rods cannot follow. In Grübler
  terms the ankle has mobility 0 instead of 2.
- **Effect in PhysX.** The effect does not depend on solver iterations (8, 16 and 32 give the same result:
  `logs/calibrate_settle/probe/`). Pure pitch, where both calf motors move together, closes. Differential motion
  (roll) opens those two closures: by up to 8-10 mm when one calf motor moves alone, and up to 21 mm at the grid
  corners.
  - With the calibration's 3 mm validity threshold, ankle roll is limited to about ±8 deg (pitch -40..+44 deg).
    Evidence: `logs/calibrate_settle/sweep_v2.log`, `diag_spherical_tierods_ankle_ranges.log`.
  - Moving `*_Revolute81` alone binds after about 50 % of a 0.3 rad step even with gravity off. A common-mode
    `67+81` step reaches 93 % (`logs/calibrate_settle/step_response/`). This explains the `Revolute81` stall
    in `logs/robot_task/inspect_articulation_v3.log`.
- **Diagnostic.** With the four joints retyped to spherical joints in memory, the whole 41x41 calf-motor grid closes
  to < 0.01 mm. Ankle roll then reaches -33..+23 deg at neutral pitch (left; right mirrored). Evidence:
  `logs/calibrate_settle/diag_spherical_tierods_sweep.log`, `raw/diag_spherical_tierods_sweep.npz`.
  - The real hardware almost certainly uses rod-end (spherical) bearings there.
  - Proposed 0.1 fix, for the plant owner to decide: retype `*_Revolute111/112` to `PhysicsSphericalJoint` at
    spawn. Adopting it requires re-running `tools/calibrate_semantics.py` (about 3 GPU minutes) and re-settling
    the motions. *(Adopted 2026-09-24, see the status block above. The calibration was re-run as v3 and the demo
    clips were re-settled.)*

### 0.3 Passive-joint damping: Isaac applies NONE; every simulator uses 0 (MEASURED; rewritten by review_fixes, 2026-09-24)

Reason: the legacy config's passive damping (50 N·m·s/rad) never acted in Isaac, while Newton applied it. That made
Newton's knee and elbow four-bars ~10³ N·m·s/rad stiffer than the plant the policies were trained on.

- **Measurement** (`tools/probe_passive_damping.py` -> `logs/review_fixes/probe_passive_damping.log`,
  `passive_damping/passive_damping_{nodrive,driveapi}.json`; root fixed, gravity off, 32/4, damping 0/0.5/5/50 written
  per env). On the contract spawn the PhysX read-back (`get_dof_dampings`) shows the written values, but a freed elbow
  and knee four-bar and a spinning tie rod evolve **identically to 4 decimals** for all four values. Cause: Isaac Lab
  writes damping as a drive parameter, and none of the 55 movable passive tree joints has an authored
  `UsdPhysics.DriveAPI` (the 28 drives are motors and neck screws). With a DriveAPI added in memory the damping acts
  on the passive revolute joints (d=5 stops a freed elbow within 20 ms) but still not on the spherical rod-end DOFs.
- **Contract (changed).** The contract plant's passive-joint damping is **0** (armature 0.01 only).
  - Isaac: `make_dropbear_cfg(passive_damping=0)` by default (`dropbear_names.EFFECTIVE_PASSIVE_DAMPING`), physically
    identical to the legacy request. `passive_drive_api=True` is an OPT-IN variant that authors drives so a non-zero
    value acts (revolute passive joints only); it is not the contract plant. The calibration/settle `QuasiStaticCfg`
    passive damping (0.5) is equally inert.
  - Newton: `sdk.motors.PASSIVE_DAMPING = 0` (was 50; `LEGACY_PASSIVE_DAMPING = 50` keeps the old value), used by
    `newton_sim.plant.PlantConfig` and `tools/newton_bridge.py --passive-damping`. Runs before 2026-09-24 13:00 used 50.
- **Sim2sim consequence** (wave policy, CPU MuJoCo C, `logs/review_fixes/sim2sim/`, one deterministic run each): at
  the bridge's default 2 ms step, passive damping 50 falls after 1.88 s of policy, 5 after 2.42 s, 0.5 completes the
  clip (joint error 0.58 rad), 0 falls at 7.74 s; **0 with stiffer closures (`--diag-eq-solref 0.004 1`) completes the
  clip at 2 ms with joint error 0.285 rad** (Isaac 0.31). The earlier conclusion that Newton needs a 0.5 ms step was
  confounded by this mismatch (docs/SDK.md 8.6).
- **Previous teleop observation, kept.** With damping 50 the elbow motor saturated at 40 N·m and reached about half of a
  0.3 rad step in 2 s; with 0.5 it settled to 0.004 rad (`logs/teleop/probe_elbow_*_analysis.log`). Newton's forearm
  angle differs from the Isaac-calibrated elbow table by about 0.05 rad on the 0.5 plant
  (`tools/analyze_teleop_probe.py` "actual" column).

The original note (teleop component), superseded by the measurement above, follows.

- **Setting.** Every passive DOF gets the legacy `parasitic_*` damping of 50 N·m·s/rad: `robots.dropbear` actuator
  groups in Isaac, and `newton_sim.plant` / `tools/newton_bridge.py --passive-damping` (default 50) in Newton.
- **Newton.** MuJoCo applies the damping. The elbow four-bar has five passive joints (`*_Revolute32/41/42/44/123`),
  and its ~4.5:1 ratio reflects the damping to the elbow motor on the order of 10³ N·m·s/rad. Fixed base, gravity
  feed-forward, elbow motor kp 150-600:
  - the motor saturates at its 40 N·m limit;
  - it reaches only about half of a 0.3 rad step in 2 s;
  - with passive damping 0.5 it settles to 0.004 rad.

  Evidence: `logs/teleop/probe_elbow_pd50_analysis.log`, `probe_elbow_pd0.5_analysis.log`,
  `probe_elbow_contract_final_analysis.log`, `verify_gpu_fig8v3_{contract,pd0.5}_summary.json`.
- **Isaac.** The calibration's elbow step responses are identical for passive damping 50 and 0.5
  (`logs/calibrate_settle/probe_step_response.log`), so the damping apparently has no effect there. The semantic
  calibration sweep itself used 0.5.
- **Consequence.** The knee four-bars carry the same damping. A policy trained in Isaac will meet much stiffer
  (slower) four-bar knees and elbows in the Newton bridge. Measure or align before judging sim2sim.
- **Also observed** (diagnostic 0.5 plant, under gravity): Newton's forearm angle differs from the Isaac-calibrated
  elbow table by about 0.05 rad (`tools/analyze_teleop_probe.py` "actual" column).

## 1. Motor contract `dropbear-wbc-motors-v1` (22 body motors)

The order follows the G1 SDK: legs first, then arms. The index is the SDK slot.

| idx | USD joint | semantic role (to be confirmed by calibration) |
|---:|---|---|
| 0 | PG_left_leg_pitch | left hip (roll-like) |
| 1 | PG_left_leg_roll | left hip (yaw-like) |
| 2 | LL_hip_joint | left hip pitch |
| 3 | LL_knee_actuator_joint | left knee crank (four-bar input) |
| 4 | LL_Revolute67 | left calf motor A (parallel ankle) |
| 5 | LL_Revolute81 | left calf motor B (parallel ankle) |
| 6 | PG_right_leg_pitch | right hip (roll-like) |
| 7 | PG_right_leg_roll | right hip (yaw-like) |
| 8 | RL_hip_joint | right hip pitch |
| 9 | RL_knee_actuator_joint | right knee crank |
| 10 | RL_Revolute67 | right calf motor A |
| 11 | RL_Revolute81 | right calf motor B |
| 12 | LH_yaw | left shoulder (pitch-like) |
| 13 | LH_pitch | left shoulder (yaw/abduction-like) |
| 14 | LH_roll | left upper-arm roll |
| 15 | LH_elbow_joint | left elbow (four-bar input; this USD drives it) |
| 16 | LH_wrist_roll | left wrist roll |
| 17–21 | RH_yaw, RH_pitch, RH_roll, RH_elbow_joint, RH_wrist_roll | right arm |

- **Neck.** `head_LeadScrew1..6` are prismatic joints held by PD at their init value. They are
  never part of the locomotion/tracking action.
- **Passive joints.** Every other articulation joint is passive: no stiffness and **no damping** (armature 0.01).
  *Corrected 2026-09-24:* this used to say "damping as in the legacy `DROPBEAR_CFG`" (50), a value PhysX never applied
  (0.3). Never command or randomize passive joints independently. A reset writes the full, physically settled joint
  state recorded in the motion NPZ.
- **Python constant.** `dropbear_wbc.robots.dropbear.MOTOR_NAMES` (tuple of 22). Pass `preserve_order=True`
  to every `find_joints` call.

## 2. Semantic ("PR"/serial output) joint space `dropbear-semantic-v1`

The semantic DOFs use G1 names and sign conventions, so G1/H1 motion maps by name:

```
left_hip_yaw left_hip_roll left_hip_pitch left_knee left_ankle_pitch left_ankle_roll
right_hip_yaw right_hip_roll right_hip_pitch right_knee right_ankle_pitch right_ankle_roll
left_shoulder_pitch left_shoulder_roll left_shoulder_yaw left_elbow left_wrist_roll
right_shoulder_pitch right_shoulder_roll right_shoulder_yaw right_elbow right_wrist_roll
```

- **World frame.** x forward, y left, z up. Each semantic angle is a rotation about the G1
  axis for that joint:
  - pitch about +y;
  - roll about +x;
  - yaw about +z;
  - knee flexion positive (G1 convention);
  - elbow: the exact G1 `*_elbow_joint` convention, which is **not** flexion-positive. 0 means the forearm
    points forward and +pi/2 means a straight arm (G1 shoulder-yaw and wrist-roll axes parallel). Positive
    is extension.
    - *Reason for change (motion pipeline, 2026-09-24):* the earlier wording said "elbow flexion
      positive", which contradicts G1. This was verified with MuJoCo and the numpy FK
      (`logs/motion_pipeline/probe_g1_elbow_convention.log`,
      `logs/motion_pipeline/probe_g1_elbow_straight_definitions.log`). The calibration uses the same
      convention (`calib_build.py` `conventions.elbow`).
- **Mapping G1 motion.** The semantic axes are idealised: exact +x/+y/+z, and hips and shoulders are
  YXZ Euler angles of zero-referenced segment frames. G1's own hip-roll axis is tilted 10 deg and its
  shoulder-pitch axis 16 deg, so a literal by-name copy of G1 joint values is off by up to ~15 deg at
  the shoulder.
  - `dropbear_wbc.motion.g1_to_dropbear` therefore converts G1 joint values into these segment-frame
    semantics (`joint_mapping="anatomical"`, the default). Knee, ankle, elbow and wrist roll still map
    by value.
  - `joint_mapping="by_name"` keeps the literal copy. Each clip's sidecar reports the per-joint
    difference between the two.
- **Semantic zero.** The Dropbear configuration closest to the G1 zero pose: legs straight
  vertical, feet flat, arms hanging down beside the body. Calibration measures the motor values
  at this pose.
- **Calibration output.** `data/calibration/dropbear_semantic_calibration.json`, schema
  `dropbear-semantic-calibration-v1`. It contains:
  - For each semantic DOF: the motors used, the map type (`linear` with sign/offset/scale, `lut1d`,
    or `lut2d` for the 2-motor ankle), forward tables (motor → semantic), inverse tables
    (semantic → motor), and the valid range.
  - `semantic_zero_motor_pos` (22).
  - `standing_motor_pos` (22): the pose used as the default and reset pose.
  - `key_bodies`: `root` (`world`), `anchor` (a chest-level body rigidly attached to the torso),
    plus pelvis, thigh, shank, foot, upper_arm, forearm, hand and head for each side, where
    applicable.
  - Rest-pose transforms, and measured segment lengths (hip→knee, knee→ankle, shoulder→elbow,
    elbow→wrist, standing hip height).
  - The provenance of every measurement: script, log and USD SHA.
- **Python API.** `dropbear_wbc.kinematics.semantic.SemanticMap.load(path)` provides
  `semantic_to_motor(q_sem[...,22]) -> q_motor[...,22]` and `motor_to_semantic(...)`. It is
  vectorized with numpy and clips to valid ranges while reporting saturation.

### 2.1 Calibration details (added by the calibration/settle component, 2026-09-24)

Reason: the measured plant needs one more map type, and consumers need the validity rules. Everything here is
additive, and the SemanticMap API is unchanged.

- **Where it comes from.** `tools/calibrate_semantics.py`: `--stage sweep` then `fit`, then `verify` and `fit`
  again. Logs of the current calibration (v3, CONTRACTS 0.2 plant): `logs/gpu_pipeline/calib_sweep_v3.log`,
  `calib_fit_v3_final.log`, `calib_verify_v3.log`. Logs of v2 (authored ankle): `logs/calibrate_settle/sweep_v2.log`,
  `fit_v2_final.log`, `verify_v2.log`.
  - The sweep runs on a gravity-free, root-fixed plant with the 0.1 fixes, 16/4 iterations and 128 parallel envs.
    It uses fixed ramp/hold stages and judges convergence by position and closure gap.
  - Samples with a loop-closure gap > 3 mm are outside the valid range: a `lut1d` is truncated and a `lut2d`
    node is marked invalid.
  - *Added 2026-09-24 (review fix).* `--raw` / `--verify-raw` have no hard-coded defaults any more (they pointed at the
    v2 authored-ankle evidence); omitted, they come from the provenance of the calibration at `--out`. GPU stages
    refuse to overwrite an existing raw without `--force`, and `fit` fails when the sweep and verify raws differ in
    plant variant (`calib_build.plant_variant_mismatch`: `authored_ankle_tierods`, `contract_fixes`, diagnostics).
    `robots.defaults.load_standing_motor_pos` fails closed on a calibration of the other ankle variant (a missing
    `authored_ankle_tierods` counts as authored, as for NPZs) or a DIAGNOSTIC `plant_variant`.
- **Map type `serial3`** (hips and shoulders).
  - It maps the three motors of a serial chain to (pitch, roll, yaw), the YXZ Euler angles of the segment's
    zero-referenced orientation. The mapping is exact, through the measured screw axes.
  - The `serial_groups.<side>_<hip|shoulder>` block holds `chain_motors`, `axes_root`, `slopes`, `R_rest`,
    `R_ref`, the linear guess (`role_motors`, `linear_scale`, `linear_offset`) and `semantic_box`.
  - Why it exists: Dropbear's hip chain runs roll -> yaw -> pitch, where G1 runs pitch -> roll -> yaw, and the
    hip yaw/pitch axes are tilted 10 deg. On the 82 physics verify poses, per-DOF linear hip maps were off by up
    to 22 deg; `serial3` stays within 0.23 deg (`logs/calibrate_settle/serial3_vs_linear_on_verify.log`).
  - The per-DOF `scale`/`offset` stay in the file as the linear approximation.
  - The `valid_range` of a `serial3` DOF is the bounding box over motor-limit combinations. `single_dof_range`
    is that DOF moved alone.
  - A request inside the box that needs out-of-limit motors is clipped in motor space. `SaturationReport.used`
    is then the orientation actually reached.
- **Semantic zero.** "Feet flat" means the sole bottom plane (a convex-hull fit) is horizontal. The authored rest
  pose has the soles pitched 3.3 deg toe-up.
  - At the semantic zero the knee crank is at 3.75 deg (straightest leg), and the elbow motor at 21 deg (G1
    elbow 0, forearm forward).
  - `standing_motor_pos` has knee 20 deg, hip pitch -6.3 deg, ankle pitch -13.5 deg (soles flat) and elbow
    72.8 deg (G1 convention).
- **Knee.** The knee is a polycentric four-bar: long rockers from the hip area carry the shank. The semantic
  knee is the rotation of the shank body relative to the thigh body.
  - `segment_lengths.hip_to_knee` and `knee_to_ankle` form a pivot model with the knee on the hip-ankle line.
    It reproduces the real ankle centre only to about 3 cm RMS over the full flexion range.
  - The exact leg length vs knee angle is in `geometry.per_side.<side>.hip_ankle_distance_vs_knee`.
- **Measured ranges** (deg; left side, right side mirrored):

  | joint | range |
  |---|---|
  | knee | -4.9..47.2, from the 0-30 deg crank; mean gain 1.76 |
  | elbow (G1 convention) | -32.8..88.5, i.e. 1.5..123 deg of flexion; gain -4.1 |
  | ankle pitch | -40.5..44.0 (right -44.2..42.7) |
  | ankle roll | -33.3..+23.0 at neutral pitch (\|pitch\| < 3 deg); -44.9..+32.7 over the whole grid. Right: -23.3..+37.3 and -30.3..+48.1 |
  | hip pitch / roll / yaw (single DOF) | -50..28 / -15..30 / -31..28 |

  *Updated 2026-09-24 (calibration v3, `logs/gpu_pipeline/calib_v3_vs_v2_report.log`).* The ankle rows are for the
  0.2 default plant with spherical tie rods. On the authored ankle (v2, snapshot `1a7161d3`), roll was about ±8 deg.
  The standing pose changed by at most 0.3 mrad.

  The G1 motion that exceeds these ranges is clipped by `semantic_to_motor` and reported per clip.

### 2.2 Derived serial model `dropbear_serial` (added by the serial-model/GMR component, 2026-09-24)

Reason: GMR, SONIC motion_lib, mjlab and ProtoMotions need a serial kinematic tree whose joints form the command
space. This section is additive. It gives downstream code a stable set of guarantees and does not change sections 0–2.

- **Status: DERIVED, NOT CANONICAL.** The model is `data/robot/dropbear_serial.xml`, plus `_motionlib.xml`, `_scene.xml`
  and the metadata file `dropbear_serial.json`. The USD remains the plant, and motors are reached only through
  `SemanticMap.semantic_to_motor`.
- **Regeneration.** Run `tools/build_serial_mjcf.py`. It builds the model from the current calibration and the raw
  sweep named in the calibration's provenance.
  - `dropbear_serial.json` records the calibration SHA-256 the model was built from.
  - `tests/test_serial_mjcf.py` fails when that SHA-256 no longer matches the current calibration.
- **Guarantees.**
  - The 22 hinge joints have the same names as the semantic DOFs, and their values are semantic angles.
  - The joint ranges are the SemanticMap valid ranges. For `serial3` and `lut2d` joints these are bounding boxes.
  - The free root is `pelvis`, placed at `rest_transforms.pelvis_in_root`. The USD root pose is recovered as
    `world = pelvis * inv(pelvis_in_root)`.
  - `usd_<body>` sites mark the key and tracked USD bodies of 5.1.
  - Every joint axis is an integer unit vector.
- **Accuracy** (docs/SERIAL_MODEL_AND_GMR.md section 4).
  - Orientations are exact to within 0.6 deg.
  - The pelvis, anchor, head and upper arms are exact to within 1.1 mm.
  - Hand positions are within 8 mm at p95.
  - Foot positions are 8–11 mm at p50 and up to about 47 mm at the maximum. The cause is structural: the hip chain
    order differs and the knee four-bar is polycentric.
  - Do not use serial-model FK where plant-accurate foot positions matter. Use `CalibrationFK` (MeasuredFK) instead.
- **GMR clips.** Clips retargeted by GMR on this model use `retarget_method` `gmr-serial-v1` and follow section 3
  unchanged. The shoulder comfort bounds and motor-limit constraints that the GMR runner adds are retargeting choices,
  recorded in each sidecar. They are not contract ranges.

## 3. Dropbear motion CSV `dropbear-motion-csv-v1` (retarget output → settle input)

- **File.** `data/motions/<source>/<clip>.csv`, with a header row.
- **Columns.** `root_x, root_y, root_z, root_qx, root_qy, root_qz, root_qw`, then the 22 motor angles
  (radians) in motor-contract order, with column names equal to the USD joint names.
  - Root = the pose of articulation root body `world` in the world frame, in meters, with an xyzw quaternion.
  - The settle tool corrects `root_z` against the ground, so it may be approximate.
- **Sidecar.** `<clip>.json` holds:
  - `fps`;
  - `source` and `source_file`;
  - `source_license`;
  - `retarget_method`;
  - `semantic_names` plus an optional `semantic_trajectory` path;
  - `contact_hint` (per-frame left/right foot contact booleans, if known);
  - `notes`;
  - *added 2026-09-24:* `calibration` provenance with `calibration_sha256`, `authored_ankle_tierods` and `plant_variant`
    of the calibration whose SemanticMap produced the motor columns. `tools/settle_motion.py` (v2.2) refuses a clip
    from the other ankle variant and records `meta.source_calibration` (sidecars without the fields: variant unknown).

### 3.1 Library v4 file versions (added by the foot_contact track, 2026-09-24)

Reason: the library was rebuilt with the foot-contact stage while a finished training run references a settled NPZ of
the previous CSV. Additive; the file formats are unchanged.

- `data/motions/<source>/<clip>.csv` / `.json` are the CURRENT (v4) retarget: foot-contact stage
  (`dropbear_wbc.motion.foot_contact`, DECISIONS 2026-09-24), 50 Hz, motor-step projection 0.18 rad/frame. The sidecar
  records the stage report (`g1_mapping.foot_contact_stage` or `metrics.foot_contact_stage`) and
  `metrics.plant_gates_predicted`.
- `<clip>_v3.csv` / `.json` are the previous CSVs (`tools/archive_motion_version.py`; sidecar `archived` block). A legacy
  `<clip>.npz` (e.g. `unitree_rl_lab_mimic/G1_Take_102.npz`) was settled from the v3 CSV, not from the current one.
- The v4 settle output is `<clip>_v4.npz` (`tools/settle_motion.py --out-suffix _v4`) with `<clip>_v4.validation.json`.
- `data/motions/catalog.json` (v4) lists per clip `npz`, `status` / `verdict` (validator), physics and predicted gates;
  the v3 catalog is `catalog_v3.json`.

## 4. Motion NPZ `dropbear-motion-npz-v1` (settle output → tracking env input)

This is a superset of the BeyondMimic NPZ. All frames are at the policy rate (50 Hz).

| key | shape | meaning |
|---|---|---|
| `fps` | () | 50 |
| `joint_pos`, `joint_vel` | (T, J) | ALL articulation joints (motors + passive + neck), Isaac joint order, physically settled |
| `body_pos_w` | (T, B, 3) | world frame, ground at z = 0, env origin at (0,0) |
| `body_quat_w` | (T, B, 4) | wxyz |
| `body_lin_vel_w`, `body_ang_vel_w` | (T, B, 3) | |
| `joint_names` | (J,) str | Isaac articulation order |
| `body_names` | (B,) str | Isaac articulation order |
| `motor_names` | (22,) str | motor contract |
| `closure_residual_m` | (T,) | worst loop-closure anchor gap after settle |
| `meta` | () str | JSON: source sidecar + settle parameters + USD SHA + tool versions |

The tracking env must verify `joint_names` and `body_names` against the live articulation and fail closed.

## 5. Tracking task `dropbear-tracking-v1` (BeyondMimic MDP port)

- **Gym ids.** `Dropbear-Tracking-Flat-v0` for training and `Dropbear-Tracking-Flat-Play-v0` for playback.
- **Actions.** `JointPositionAction` on the 22 motors only, with `preserve_order=True` and
  `use_default_offset=True`. The default is `standing_motor_pos` when available.
- **Observations.** Joint observations cover the 22 motors only. The command is the reference
  motor `joint_pos` + `joint_vel` (22 + 22).
- **Reset (RSI).** Write the FULL settled `joint_pos`/`joint_vel` row plus the root state from the NPZ.
  - Additive noise goes on the 22 motors only.
  - The `randomize_joint_default_pos` startup event is limited to the 22 motors.
- **Bodies.** The anchor is the chest-level torso body, not `world`, whose frame origin sits
  about 12.5 cm below the soles.
  - Terminations use the anchor and end-effector bodies (feet and hands).
  - Undesired contacts cover everything except the foot bodies
    (`LL/RL_skateboard_bearing_left_2`, `LL/RL_basis_left_1`) and the hands.
- **Timing.** dt 0.005, decimation 4 (50 Hz).
- **Solver.** 32 position / 4 velocity iterations by default. This is configurable, and its
  throughput impact must be measured.
  - *Measured (robot/task component, 2026-09-24; `logs/robot_task/throughput_matrix_run.log`).* Env-steps/s
    for the training config with zero actions, at 8/4, 16/4 and 32/4 iterations: 2118/2024/1605 at 256 envs,
    7501/6302/4693 at 1024 envs, 13493/11126/8066 at 2048 envs. Total GPU memory was 4.2, 5.1 and 6.4 GB.
    Loop-closure gaps while holding the stand had p95/max of 0.99/1.05 mm at 8/4, 0.40/0.55 mm at 16/4 and
    0.30/0.32 mm at 32/4. The worst closure is the ankle tie rod `*_Revolute112`.
  - *Recommendation.* Train at **8/4 with 2048 envs**. It is the cheapest setting and its gaps stay far
    below 3 mm. Replay and evaluate at 32/4 as a solver-slack check. The 32/4 default of `make_dropbear_cfg`
    is unchanged. Pass `--solver_iters 8 4` to `scripts/train.py`.

### 5.1 Tracking implementation details (added by the robot/task component)

Reason: deploy code and NPZ producers need the exact conventions the trained policies assume.
All of this is additive.

- **Code.**
  - Robot: `source/dropbear_wbc/robots/` (`dropbear_names.py` is pure Python).
  - Task: `source/dropbear_wbc/tasks/tracking/`.
  - Scripts: `scripts/train.py` and `scripts/play.py`. Evidence is in `logs/robot_task/`.
- **Anchor.** The anchor is `head_5mm_ujoint_base__5__1`, attached to `world` by the fixed joint
  `head_Platform_Base_Joint1` and sitting at z ≈ 1.64 m at rest. Verified rigid:
  < 0.002 mm / 5e-5° while the robot spins at 3.9 rad/s.
  - Observations and rewards use its **raw link frame**, as in BeyondMimic. Its rest orientation
    in the root frame is (0.5, 0.5, 0.5, 0.5), so the link's z axis points forward.
  - Only `bad_anchor_ori` uses the world-aligned frame `q_link * (0.5, -0.5, -0.5, -0.5)`, so that
    projected-gravity z detects roll as well as pitch.
- **Tracked bodies (14, in order).**
  1. Anchor.
  2. Left leg: `LL_RMD_X10_S2_MIR4__3_Stator_1` (thigh), `LL_double_bracket_10deg_MIR_MIR_MIR_1` (shank),
     `LL_skateboard_bearing_left_2` (foot).
  3. Right leg: the same bodies with the `RL_` prefix.
  4. Left arm: `LH_RMD_X8_Pro_MIR8_MIR1__3__1` (upper arm), `LH_6mm_bearing__4__1` (forearm),
     `LH_shoulder_ex_al_interface_1` (hand).
  5. Right arm: the same bodies with the `RH_` prefix.
  6. `head_u_joint_center__8__1` (head).

  A motor sweep confirmed which motor drives each body (`EXPECTED_BODY_DRIVERS`,
  `inspect_articulation_v3.json`).
- **End effectors (terminations).** Feet: `LL/RL_skateboard_bearing_left_2`. Hands:
  `LH/RH_shoulder_ex_al_interface_1`.
- **NPZ velocity convention.** `body_pos_w` and `body_quat_w` are link-frame poses.
  `body_lin_vel_w` is the **centre-of-mass** linear velocity of each body (Isaac Lab's
  `body_lin_vel_w`). This matters for `world`, whose frame origin is about 1.4 m below its CoM.
  - Reset writes the `world` row with `write_root_state_to_sim`: link pose plus CoM velocity.
  - Producers that differentiate link positions must convert: `v_com = v_link + w × r_com`.
- **Actions.** Per motor, `q* = default + scale * a`, with `scale = 0.25 * effort_limit / kp`
  (BeyondMimic's rule): arms 0.2, hips 0.333, knees 0.375, ankles 0.25.
- **Policy observations (125).**
  - `command` 44: reference motor positions (22), then velocities (22).
  - `motion_anchor_pos_b` 3 and `base_lin_vel` 3. Both are sim-only.
  - `motion_anchor_ori_b` 6 and `base_ang_vel` 3.
  - `joint_pos_rel` 22, `joint_vel_rel` 22 and `last_action` 22.
- **Export (`scripts/play.py --export`).**
  - `policy.json` in the `dropbear-policy-sidecar-v1` schema (6.1). It also has a
    `dropbear_tracking` block with the layout, normalizer, provenance and parity results.
  - `policy_motion.onnx` with BeyondMimic I/O (`obs`, `time_step`). The embedded reference holds
    the 22 motor columns and the 14 tracked bodies.
  - `policy.onnx` and `policy.pt`, which map obs to actions. The observation normalizer is baked in.
- **Fail-closed inputs (added 2026-09-24, robot/task continuation).** Reason: the calibration and settle
  components now write real files, and a stale or rejected file must not silently train a policy.
  - Motion NPZ (`motion_npz.validate_provenance`, called by the motion command): `meta.schema`, if present,
    must be `dropbear-motion-npz-v1`. `meta.status == "rejected"` (the `tools/settle_motion.py` quality
    gate) is refused unless `MotionCommandCfg.allow_rejected_motion`. `meta.usd_sha256`, if present, must be
    the contract SHA; the check is skipped when `$DROPBEAR_USD` overrides the plant. The name/order checks of
    section 4 run first. Extra keys (`frame_flags`, `motor_target`, ...) are ignored.
    *Added 2026-09-24 (0.2 adoption):* `meta.authored_ankle_tierods` (missing = authored) must match the env's
    plant variant (the spawn config's `spherical_joint_overrides`), otherwise the NPZ is refused.
  - Calibration (`robots.defaults.load_standing_motor_pos`): besides schema, completeness, finiteness and
    hard limits, `motor_names` (if present) must equal the motor contract **in order**, and `usd_sha256`
    (if present) must be the contract SHA (skipped under `$DROPBEAR_USD`). `$DROPBEAR_CALIBRATION_JSON`
    overrides the path. The calibration path, SHA-256, `created` and `plant_variant` are recorded in
    `run_info.json` (`default_pose_info`) and in the export sidecar.
  - *Added 2026-09-24 (review fixes).*
    - Validation verdict: a sibling `<clip>.validation.json` (`dropbear-motion-validation-v1`, written by
      `tools/validate_motion_npz.py --write-verdicts`, with the NPZ's `npz_sha256`) whose `verdict` is `rejected` is
      refused like `meta.status == "rejected"` (also when stale). An exploratory run needs
      `scripts/train.py --allow_rejected_motion` (recorded in `<run>/motion_acceptance.json`, inherited by resumed
      chunks, checked against the motion SHA-256); `run_info.json`, the play summary and the export sidecar
      provenance carry `motion_validation` and `allow_rejected_motion`.
    - Reference dynamics gate (`settle.quality.motor_dynamics`, in the validator and in `tools/settle_motion.py` v2.2
      `meta.status_reasons`): max |motor joint_vel| <= 10 rad/s (the USD motor cap) and max motor step <= 0.3 rad per
      50 Hz frame.
    - Chunked training state: `DropbearOnPolicyRunner` checkpoints `alg.learning_rate` and the motion command's
      `bin_failed_count`; `scripts/train.py --restore_train_state auto` restores them on resume (runs whose first
      chunk predates this keep their original restart-from-scratch behaviour).
  - A clip built by `tools/make_static_npz.py` records the default pose it was built for
    (`meta.default_motor_pos`). The env logs `npz_default_pose_max_abs_diff` and warns above 0.02 rad, which
    means the clip is stale relative to the current calibration.

### 5.2 Evaluation modes (added by review_fixes, 2026-09-24)

Reason: the Play config removes all randomization and the policy is deterministic, so every env of a play run follows
the same trajectory (16 envs are N = 1), and at the clip end the default resample wrote the NPZ frame-0 state back
into the sim, so "loop 2" restarted from the reference.

- `scripts/play.py --continuous_loop` (`MotionCommandCfg.continuous_loop`): at the clip end only the reference clock
  wraps; the robot keeps its state. Episode resets after a termination still write the reference.
- `scripts/play.py --perturbed [--seed N]`: the training randomization (pushes, friction, base CoM, default-pose
  offsets, observation noise) in play; envs then differ through their randomization draws.
- The play summary reports `eval_design` (randomization, loop mode, the reference jump at the wrap, the N = 1 note) and
  `falls` (per env, per loop, first fall step, 1-step episodes).
- Evidence: `logs/review_fixes/wave_eval_modes.json` (wave policy: nominal continuous and perturbed continuous, 16 envs x
  2 loops each, 0 falls).

### 5.3 Motion-library tracking `dropbear-tracking-library-v1` (added by the multiclip track, 2026-09-24)

Reason: BeyondMimic trains one policy per clip, which does not scale to a motion library. A library policy is the step
towards a SONIC-style general tracker, and its export has to accept any reference at runtime. Everything here is
additive: the single-clip task, its checkpoints and its exports are unchanged. Progress and evidence:
`logs/multiclip/PROGRESS.md`.

- **Manifest `dropbear-motion-library-v1`.** A JSON file: `{"schema", "name", "clips": [{"name", "npz", "weight"}]}`.
  Relative `npz` paths resolve against the manifest's directory. The first manifest is
  `data/motions/libraries/accepted_v0.json`: the 5 synthetic clips (stand, wave_right, arm_swing, weight_shift,
  squat_lite, 62 s in total). Kimodo `output_wave` is left out because the v4 motor-dynamics gate rejects it.
  - *Added 2026-09-24 21:35.* A clip entry may carry `"sha256"` (the NPZ bytes). `build_library` and
    `manifest_fingerprint` then refuse a file that changed since the manifest was built. Pins do not change the
    library identity.
  - *Generator.* `tools/build_motion_library_manifest.py` builds manifests from the validator verdicts:
    - every `*.validation.json` under `data/motions` (default excludes `smoke/**`, `**/archive/**`);
    - the verdict must be `accepted` and not stale, and the clip must pass this section's per-clip checks;
    - across clips, names must match, and a known calibration must equal the default calibration's SHA;
    - excluded clips are listed with reasons in `selection.excluded`, and every clip is pinned;
    - the base manifest's order is kept, and new clips are appended;
    - an unchanged selection writes nothing.

    The current manifest is `accepted_v1.json`, regenerated by the tool: 6 synthetic clips (the v0 set plus
    wave_right_v2), 73 s, library sha256 `a089f262ab48...`.
- **Loading fails closed** (`tasks/tracking/motion_library.build_library`).
  - Every clip passes the section 4 / 5.1 checks: names and order against the live articulation, fps equal to the
    policy rate, `validate_provenance` (schema, producer status, validator verdict, USD SHA, ankle variant).
  - Every clip needs a sibling `<clip>.validation.json` whose verdict is `accepted` and not stale.
    `allow_rejected_motion` lifts this (exploratory runs only; it is recorded as for single-clip).
  - Across clips, these must match: joint, body and motor names (with order), fps, `usd_sha256` and ankle variant.
    `calibration_sha256` must also match where a clip records it. Clips settled before v2.2 record none, and the
    library report lists them as "unknown".
- **Identity.** The library `sha256` hashes the clip names, NPZ SHA-256s and weights, in order
  (`library_fingerprint` / `manifest_fingerprint`). It does not depend on file paths. It is the run's
  `motion_sha256` for `motion_acceptance.json` and for resume checks.
- **Memory.** Clips are concatenated into one tensor per quantity, with per-clip `starts`/`lengths`. Frame `t` of
  clip `c` is row `starts[c] + t`. Only the root body and the 14 tracked bodies are kept: about 1.5 KB per frame,
  against about 5 KB with all 90 bodies.
- **Gym ids.** `Dropbear-Tracking-Library-v0` / `-Play-v0`, and `Dropbear-Tracking-Library-Future-v0` / `-Play-v0`.
  - The config is `config/dropbear/library_env_cfg.py`, and the command is `mdp/library_commands.MotionLibraryCommand`.
  - The PPO config is identical to single-clip. Its experiment directory is `logs/rsl_rl/dropbear_tracking_library`.
  - The MDP is section 5 / 5.1 unchanged: actions, rewards, terminations, randomization, timing and solver.
- **Command / observations.**
  - Library: the command is the 44-value single-clip layout, so the policy observation is the same 125-value vector
    as 5.1.
  - `-Future`: the command is `[q_t, dq_t, q_{t+5}, dq_{t+5}, q_{t+10}, dq_{t+10}]` (132 values; +0.1 and +0.2 s).
    Every `q`/`dq` is 22 motor values. Future frames come from the same clip and are clamped to its last frame.
    The policy observation has 213 values, and the critic carries the same command.
- **Sampling (RSI).** Every env carries `(clip_ids, time_steps)`. A reset draws `(clip, time-bin)` from
  `p(c) * p(b | c)`:
  - `p(b | c)`: BeyondMimic's rule per clip. Bins are `T_c // 50 + 1` (about 1 s). Each bin has a failure EMA
    (alpha 0.001), plus the uniform floor `0.1 / nb_c` and the optional look-ahead kernel.
  - `p(c) = (1 - rho) * prior_c + rho * h_c / sum(h)`, with `rho = clip_adaptive_ratio = 0.5`.
  - `prior_c` is proportional to `weight x frames` (`clip_weighting="duration"`, the default, which is
    time-uniform) or to `weight` alone (`"uniform"`).
  - `h_c` is the clip's failure hazard: the EMA of failures on the clip divided by the EMA of envs on it, per env
    step.
  - A one-clip library reproduces the single-clip sampler exactly (`tests/test_motion_library.py`).
  - At a clip end the env resamples `(clip, t)`, as upstream does at a clip end.
  - Play (`start_at_zero`): env `i` plays clip `i % N` (or `play_clips`) from frame 0. `continuous_loop` wraps that
    clip's clock.
- **Metrics** (`Metrics/motion/...`).
  - The single-clip error metrics, plus `sampling_entropy`, `sampling_top1_prob` and `sampling_top1_clip`.
  - Aggregates: `clip_prob_min`, `clip_fail_per_s_max` and `clip_err_joint_max`.
  - Per clip, for the first 32 clips: `clip_prob/<name>`, `clip_fail_per_s/<name>` (hazard x fps) and
    `clip_err_joint/<name>` (EMA of the mean motor-joint error of the envs on that clip).
  - `scripts/play.py` on a library task adds `per_clip` results to the summary: mean errors, falls, first fall.
- **Chunked training.** `runner.DropbearOnPolicyRunner` also checkpoints `command_extra`: the library SHA plus the
  clip EMAs. They are restored only for the same library and bin layout.
  - *Added 2026-09-24 21:35.* On a library task, the per-bin `bin_failed_count` is restored only together with a
    matching `command_extra`. So a warm start from another library, or from a single-clip run, with the same bin count
    starts the sampler fresh. `train_state_restore.bin_failed_count_skipped` records it.
- **CLI.** `scripts/train.py --motion_library <manifest>` and `scripts/play.py --motion_library <manifest>
  [--play_clips a,b]` take the manifest; `--motion_file` stays for single-clip tasks, and exactly one of the two is
  required. `tools/run_chunked_training.py --motion_library <manifest> --task Dropbear-Tracking-Library-v0`
  (optionally `--experiment_name`).
- **Export with a runtime reference** (`scripts/play.py --export` on a library task calls
  `tasks/tracking/export_library.export_library_policy`).
  - It writes `policy.onnx` (`obs -> actions`, normalizer baked in), `policy.pt` and `policy.json`.
    **No `policy_motion.onnx`**, so no reference is baked in.
  - Sidecar fields: `policy_onnx: policy.onnx`, `onnx_inputs.time_step: null`, `motion.source: "runtime"`,
    `motion.fps` and `motion.future_steps`.
  - `motion.requirements` lists what a reference fed at runtime must meet: `fps`, `usd_sha256`,
    `authored_ankle_tierods`, `motor_names`, `anchor_body_name`.
  - The command observation entry has `params.future_steps` for `-Future` policies.
  - `dropbear_tracking.export_type` is `"library_runtime_reference"`, and it carries the library report.
- **Deploy** (`tools/policy_runner.py --motion <clip.npz|clip.csv>`, REQUIRED for a `runtime` export).
  - `deploy.motion.load_runtime_motion` fails closed on:
    - another plant USD or ankle variant;
    - an NPZ fps other than the policy rate (a CSV is sampled at the nearest frame);
    - a clip that is `rejected` by its validator verdict or `meta.status` (unless `--allow-rejected-motion`).
  - Clips without a verdict are used and reported as `unvalidated`.
  - The export's anchor offset and alignment are kept. `motion_command` with `params.future_steps` reads the
    reference `k` policy steps ahead, clamped at the clip end like training.
  - Tests: `tests/test_library_export_runner.py`, on the real `export_library.py` via
    `scripts/emulate_tracking_export.py --library`.
- **GPU evidence (2026-09-24, `logs/multiclip/PROGRESS.md`).**
  - Env probe: the command, the future slices and the tracked bodies equal the NPZ at the sampled (clip, t), with
    error 0.0, and the play assignment is correct.
  - 20-iteration PPO smokes of both task ids: finite, with 11.9k / 12.0k env-steps/s at 2048 envs and 8/4, the same
    as single-clip.
  - Play and export: `tools/check_library_export.py` ok. The deploy runner fed an NPZ reproduces the Isaac command
    exactly.
  - Nothing yet on tracking quality.

### 5.3 Demo rendering, reference replay and warm start (added by demo_eval, 2026-09-24)

Reason: demo videos need to show the robot, not debug frames, and several views of one rollout; a policy for a new clip
of the same task should be able to start from an existing one. Everything here is additive; physics, observations,
actions and rewards are unchanged.

- **Clean render** (`dropbear_wbc.tasks.tracking.demo_render`). `apply_clean_render(env_cfg)` turns off every debug marker
  (motion-command frames, contact-sensor `debug_vis`, any other `debug_vis` flag on scene/command terms), replaces the
  lights (white dome 900 + white distant key light 2600 from the front-left) and tints the grid floor mid-grey.
  `DemoRecorder` records env 0 with its own USD cameras + replicator `rgb` annotators (not the gym `RecordVideo`
  viewport): `track` (front-right 3/4 view, 3.3 m, eye 1.45 m, look-at 0.95 m, low-pass root following), `front` and
  `side` (fixed, framing the clip's root xy extent), `feet` (low, tracking). One frame per 50 Hz policy step (50 fps =
  real time), ffmpeg libx264 crf 18, caption + clock burned in. Needs `--enable_cameras`.
- **CLI.** `scripts/play.py --clean_video [--video_cams track,front] [--video_res 1280x720] [--video_dir] [--video_tag]
  [--video_caption]` (exclusive with `--video`); the play summary JSON gets a `clean_video` block (cameras, outputs,
  render cfg). `scripts/render_reference.py --motion_file <npz>`: kinematic replay of a settled NPZ (full joint row +
  root state written per frame, `sim.render()` only, physics never stepped) with the same renderer, to show what a
  policy is asked to track (including floating reference feet). `tools/plot_reference_contacts.py`: per-foot lowest
  sole height vs the NPZ contact flags and the 15 mm gate.
- **Warm start.** `tools/run_chunked_training.py --init_run <run dir> [--init_checkpoint model_N.pt]`: chunk 1 of a NEW
  run calls `scripts/train.py --resume --load_run <init_run>` without `--continue_run` (new run dir; weights, optimizer
  and iteration counter from the checkpoint; the train-state restore skips a mismatched adaptive-sampling bin count).
  Valid only between runs with the identical 125-D observation / 22-D action layout of 5.1 (every clip of this task).
  `chunks.json` records `warm_start` on chunk 1; the train log prints the loaded checkpoint.
- **Media locations.** Demo media of the demo_eval track live in `logs/demo_eval/media/`; the gallery is `docs/DEMOS.md`.
- *Changed 2026-09-24 22:30 (demo_eval), reason: the first renders cropped head and feet, and 3 of 4 physics play renders
  returned 25-61 % blank frames (black or washed-out grey) while `empty_frames` stayed 0; every kinematic replay was
  clean (`logs/demo_eval/media/rejected_render_glitch/README.txt`).* Framing: 12 mm lens, `track` 4.4 m / eye 1.55 m /
  look-at 1.0 m, `front`/`side` at max(5 m, root extent + 4 m), `feet` keeps 18 mm. `demo_render.frame_is_valid`
  (colour + contrast) checks every captured frame; `DemoRecorder` re-renders invalid ones and reports `retried_frames` /
  `invalid_frames`. **Policy videos are rendered from the recorded rollout**: `scripts/play.py --record_rollout <npz>`
  (env 0's simulated state per policy step, contract NPZ with `meta.not_a_reference`, stored under `logs/`, never used for
  training) then `scripts/render_reference.py` on it; `tools/render_policy_clean.py --jobs <json> | --job <json>` chains
  both and writes `<tag>_frame_check.json`. Physics, observations and actions are unchanged.

## 6. Low-level SDK `dropbear_hg-v1` (Unitree LowCmd/LowState equivalent)

Unitree's `unitree_hg` has `MotorCmd{mode,q,dq,tau,kp,kd}` and
`MotorState{q,dq,ddq,tau_est,temperature}`. Dropbear uses the same shape:

- **Commands.** 22 motor slots in motor-contract order. Slots 22–27 are reserved for the neck.
- **LowState.** `{tick, imu{quat_wxyz, gyro, acc} on the torso body, motor_state[22]}`.
- **Motor law** (every simulator bridge and, later, the ESP32 gateway):
  `tau = tau_ff + kp*(q* - q) + kd*(dq* - dq)`, clipped to the effort limit.
- **Transport.** ZMQ + msgpack. State goes on `tcp://127.0.0.1:5556` (PUB). Commands go on
  `tcp://127.0.0.1:5555` (SUB). Rates: 500 Hz state, command rate set by the client.
- **Future.** CycloneDDS on Linux/Jetson, wire-compatible with `unitree_hg` IDL, is planned later.

### 6.1 Wire details and policy sidecar (added by the SDK/bridge component)

Reason: section 6 fixes the shape but not the bytes. The tracking export and any other client
needs these details to interoperate. They are additive; nothing above changes.

- **Implementation.** `source/dropbear_wbc/sdk/` (`types.py`, `transport.py`, `crc.py`, `motors.py`).
  The authoritative field lists are in the `types.py` docstrings.
- **Frames.** One ZMQ frame per message: `b"rt/lowstate\0"` or `b"rt/lowcmd\0"` followed by a
  msgpack map. The robot side (bridge, later the gateway) binds both sockets; clients connect.
- **Map keys.** `{"schema": "dropbear_hg-v1", "type": "LowCmd"|"LowState", "tick", "stamp_ns", ..., "crc"}`.
  Arrays are raw little-endian bytes: float32 values and uint8 `mode`.
- **LowCmd.** Carries `motor{mode,q,dq,tau,kp,kd}` and an optional `neck` block (slots 22–27; `null`
  keeps the current neck targets).
  - `mode` 1 enables the motor and 0 disables it (zero torque).
  - `tick` echoes the LowState tick the command was computed from.
- **LowState.** Carries `imu{quat_wxyz,gyro,acc,rpy}` of the root body `world`, `motor{mode,q,dq,ddq,tau_est,temperature}`,
  an optional `neck` block, and an optional `sim` block.
  - `gyro` is the body-frame angular velocity, in rad/s.
  - `acc` is the body-frame specific force, in m/s². It reads +9.81 on z when the robot is upright at rest.
  - `sim` holds privileged ground truth: the root pose and velocity in world, plus the poses of any requested bodies.
    Hardware never has it. A policy that needs it is sim-only and must say so.
- **CRC.** Unitree's `crc32_core` is applied to the canonical bytes (`<I tick`, `<q stamp_ns`, then every
  array in field order), read as little-endian uint32 words.
- **Safety semantics.** The robot side holds the last command.
  - If no command arrives for more than 100 ms (wall clock), it switches to damping (`kp=0`, `kd=damping`).
  - The Newton bridge stays paused until the first command arrives.
- **Policy sidecar `dropbear-policy-sidecar-v1`.** A JSON file next to an exported ONNX, consumed by
  `tools/policy_runner.py --sidecar` (parser: `dropbear_wbc/deploy/config.py:load_sidecar`).
  - Keys: `schema`, `policy_onnx` (relative path), `onnx_inputs{obs, time_step|null}`, `step_dt`.
  - `joint_names`: the policy and action order, 22 motor-contract names.
  - `default_joint_pos`, `joint_stiffness`, `joint_damping`: in policy order.
  - `action_scale` (a scalar or 22 values). `action_offset` is optional and defaults to `default_joint_pos`.
    `action_clip` is optional.
  - `observations`: an ordered list of `{name, func, dim, scale, clip, history_length, params}`. `func` is an
    observation function name, for example `motion_command`, `motion_anchor_ori_b`, `motion_anchor_pos_b`,
    `base_lin_vel`, `base_ang_vel`, `projected_gravity`, `joint_pos_rel`, `joint_vel_rel` or `last_action`.
    Give `func` explicitly, because BeyondMimic's term *name* `joint_pos` means `joint_pos_rel`.
  - `obs_dim`.
  - `motion{source: "npz"|"onnx", npz, anchor_body_name, align, reference_joint_names, body_names, num_frames}`.
    - *Added by the SDK/bridge component (2026-09-24):* `anchor_offset_pos` (3) and `anchor_offset_quat`
      (4, wxyz). They give the rigid pose of the anchor body in the root (`world`) link frame. The tracking
      export writes them. The runner rebuilds the robot anchor as root pose (IMU quat + root position)
      composed with this offset. They are **required** when `body_names` does not contain `world`, which is
      the case for the export's 14 tracked bodies. Without them the runner now fails closed. Before this
      change it silently used the root frame.
      Evidence: `logs/sdk_bridge/export_emulated_negative_controls.log`, `tests/test_export_sidecar_runner.py`.
    - *Added by the multiclip track (2026-09-24; section 5.3):* `source: "runtime"` means that the export embeds no
      reference: `policy_onnx` is the plain `policy.onnx` and `onnx_inputs.time_step` is `null`. The runner then
      requires `--motion <npz|csv>`, which is checked against the new optional `motion.requirements`
      (`fps`, `usd_sha256`, `authored_ankle_tierods`, `motor_names`, `anchor_body_name`).
    - Also optional: `motion.fps`, `motion.future_steps`, `motion.library` (name, sha256, num_clips), and
      `params.future_steps` on a `motion_command` observation entry (future reference frames appended, see 5.3).
    - `--motion` also works with the other sources: it then replaces the configured or embedded reference.
  - Optional provenance: `task`, `usd_sha256`, `run_path`.
  - BeyondMimic ONNX metadata (`attach_onnx_metadata`) fills any key the sidecar omits.

## 7. Velocity locomotion task `dropbear-velocity-v1` (H1 flat port; added by the locomotion component, 2026-09-24)

Reason: the classic Unitree H1/G1 capability (Isaac Lab `Isaac-Velocity-Flat-H1-v0`, unitree_rl_lab
`Unitree-H1-Velocity`) needs a task, a training/eval path and a deployable export on the contract plant. Additive:
nothing in sections 0-6 changes. Design, reasons and commands: `docs/LOCOMOTION.md`; evidence: `logs/locomotion/`.

- **Gym ids.** `Dropbear-Velocity-Flat-v0` (training: 2048 envs, PhysX 8/4) and `Dropbear-Velocity-Flat-Play-v0`
  (16 envs, 32/4, no randomization, curriculum off). Code: `source/dropbear_wbc/tasks/locomotion/`; registered by
  `dropbear_wbc.tasks.register()` after the tracking ids (guarded: a velocity import error cannot break tracking tools).
- **Plant / timing.** `make_dropbear_cfg` (sections 0.1-0.3), calibration `standing_motor_pos` as the default pose;
  dt 0.005, decimation 4 (50 Hz), 20 s episodes.
- **Actuator profile (added 2026-09-24, locomotion).** Reason: the legacy knee PD is too soft at the knee joint
  (62 N*m/rad reflected through the four-bar) and its 300 N*m crank effort is 3x the RMD-X10-S2 peak (docs/DECISIONS.md).
  - The velocity task applies `flat_env_cfg.ACTUATOR_PROFILES[<profile>]` on top of the legacy groups. The default is
    `stiff_knee_hw`: knee crank kp 600, kd 20, effort 100 N*m; every other motor is as in section 1 and `ACTUATOR_PARAMS`.
  - Every run records its profile in `run_info.json` (`actuator_profile`, `actuator_gains_sim`).
  - The export writes the live gains: sidecar `joint_stiffness` / `joint_damping`, plus `dropbear_velocity.effort_limit` /
    `.actuator_profile`.
  - Deploy code must use the sidecar gains, not `sdk.motors.DEFAULT_KP`, and should clip the knee at the sidecar effort
    (the Newton bridge still clips at `sdk.motors.EFFORT_LIMIT`, 300 N*m for the knees).
  - The tracking task (section 5) is unchanged.
- **Actions.** Same interface as section 5: `JointPositionAction` on the 22 motors, contract order, `use_default_offset`.
  Scale: legs `0.25 * effort / kp` (hips 0.333, knees 0.375, ankles 0.25 rad), arms **0.1 rad**.
- **Command.** `base_velocity` = `(vx, vy, wz)`: planar CoM velocity of the root body `world` in its yaw frame [m/s]
  and yaw rate [rad/s] (no heading control). Training ranges start at vx 0..0.5, vy ±0.2, wz ±0.5 and are widened by
  0.1 per curriculum success up to vx −0.3..1.0, vy ±0.4, wz ±1.0; the current ranges are stored in every checkpoint
  (`infos["dropbear_train_state"]["velocity_command_ranges"]`) and restored on a chunked resume.
- **Policy observations (78, in order).** `base_lin_vel` 3 (root CoM, root frame; **sim-only**), `base_ang_vel` 3,
  `projected_gravity` 3 (root frame), `velocity_commands` 3, `joint_pos` 22 (`joint_pos_rel`, motors), `joint_vel` 22
  (`joint_vel_rel`, motors), `actions` 22. Critic: the same terms without noise.
- **Reset.** Full joint row (all 91 DOFs) + root state of a settled standing contract NPZ (default
  `data/motions/smoke/dropbear_static_stand.npz`; fail-closed checks of section 5.1 incl. the validator verdict), a
  rigid yaw/x/y move, and noise on the 14 serial motors only (legs ±0.02, arms ±0.1 rad). Passive and closure-coupled
  joints are never perturbed. Measured: the zero-action robot collapses at the knees in ~0.7 s from this state (legacy
  knee PD), so the reset is grounded and closure-consistent but not a zero-action equilibrium (`probe_smoke_v1.json`).
- **Torso signals.** Height and orientation come from the anchor body (`ANCHOR_BODY`, world-aligned with
  `ANCHOR_FRAME_OFFSET_WXYZ`); never from the root frame height. Targets/thresholds come from the reset NPZ
  (anchor height 1.498 m; fall below NPZ − 0.30 m; tilt > 0.8 rad). Any ground contact of a non-foot body terminates.
- **Feet.** `FOOT_BODIES` grouped per foot (sole plate + ankle cross); a foot is in contact when either body is.
- **Export (`scripts/play_locomotion.py --export`).** `policy.onnx` (input `obs` (1, 78) -> `actions` (1, 22), normalizer
  baked in), `policy.pt`, and `policy.json` in `dropbear-policy-sidecar-v1` with `motion: null`,
  `onnx_inputs.time_step: null`, observation funcs `base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
  joint_pos_rel, joint_vel_rel, last_action`, plus a `dropbear_velocity` block (layout, normalizer, command ranges at
  export, reset NPZ, provenance, parity). `tools/policy_runner.py --sidecar ... --allow-privileged --velocity-cmd VX VY WZ`
  consumes it (`tests/test_locomotion_cpu.py`).

## 8. Tabletop task and GR00T dataset `dropbear-groot-tabletop-v1` (added by the tabletop track, 2026-09-24)

Reason: the first "useful task" and its imitation dataset are consumed by later GR00T fine-tunes and a closed-loop
evaluation client; their interface must not drift silently. Additive: nothing in sections 0-7 changes. Details, commands
and evidence: `docs/GROOT.md`, `logs/tabletop/`.

- **Gym id.** `Dropbear-Tabletop-Push-v0` (`source/dropbear_wbc/tasks/tabletop/`, registered by importing
  `dropbear_wbc.tasks.tabletop.config`). Contract plant (0.1-0.3) with `fix_root_link`, root at env (0, 0, -0.12) m;
  reset = the full settled joint row of `data/motions/smoke/dropbear_static_stand.npz` (section 1 rule, fail-closed checks
  as in section 7). 20 Hz control (dt 5 ms, decimation 10), PhysX 32/4.
- **Arm control.** Absolute position targets of the 10 arm motors (motor-contract order) from semantic arm angles via
  the calibration `SemanticMap`; teleop gains (shoulders 200/5, elbow motor 600/10, wrist 60/2), teleop gravity
  feed-forward as effort target, dq* = finite difference of the targets. Legs and neck hold the reset row.
- **Dataset.** Isaac-GR00T LeRobot v2 layout; `observation.state` / `action` = 10 semantic arm angles (section 2 names,
  left 5 then right 5; action ABSOLUTE in the file, RELATIVE in the GR00T config); `timestamp = frame_index / 20`;
  views `ego_view` (head camera on the anchor body, 640x480) and optional `left/right_wrist_view` (320x240); one
  instruction per episode via `task_index`; per-episode `success` in `meta/episodes.jsonl`. Only successful episodes by
  default; episodes failing the blank-frame or zone-render checks are dropped by the builder.
- **Splits.** `data/groot/placements/tabletop_push_v1.json`: 400 `train` + 50 `heldout` placements from disjoint seed
  streams. Data is collected on `train` only; `heldout` is for success rates (scripted baseline 42/50).

