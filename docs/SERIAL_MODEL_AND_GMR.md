# Dropbear serial ("output-space") model and GMR integration

> **DERIVED, NOT CANONICAL.** The plant is the user's USD (`docs/CONTRACTS.md` section 0, SHA `45586414...`).
> The serial model is a surrogate whose joints are the 22 `dropbear-semantic-v1` DOFs (CONTRACTS section 2). Its
> closed loops (knee and elbow four-bars, parallel ankle, head Stewart platform) are collapsed into those joints.
> Motors are always reached through `SemanticMap.semantic_to_motor`. Do not train a deployable policy on the serial
> model alone, and never use it to replace the USD in the Isaac or Newton pipelines.

G1 and H1 tools such as GMR, SONIC motion_lib, mjlab and ProtoMotions assume the robot is a serial kinematic tree
whose joint values are the command space. For Dropbear that command space is the semantic space. This document
covers four things:

- what the serial model contains and how accurately it reproduces the plant;
- how to regenerate it from whatever calibration is current;
- how GMR was set up to retarget human motion onto it;
- how motion_lib and mjlab tools would consume it next.

Every number below comes from a log under `logs/serial_mjcf_gmr/`. Anything without a log is marked **UNVERIFIED**.

## 1. Status at a glance

| Item | Status | Evidence |
|---|---|---|
| Serial MJCF built from the final v3 calibration (`e51033d4`, spherical ankle tie rods) | VERIFIED: compiles in MuJoCo 3.3 (system Python), 3.12 (`.venv-newton`) and 3.14 (`.venv-gmr`) | `build_serial_mjcf.log`, `pytest_serial_gmr_*.log` |
| Joint values equal semantic values; ranges equal semantic valid ranges; every joint axis an integer unit vector | VERIFIED | `tests/test_serial_mjcf.py::test_structure`, `::test_integer_axes_and_motionlib_variant` |
| Orientation parity with the calibration forward model | VERIFIED: < 0.6 deg on every key body (the residual is the ankle's measured yaw) | `fk_parity_report.json` |
| Position parity with a < 1 cm target | **Met** for head, anchor, pelvis, upper arms (< 1.1 mm), forearms and hands (p95 <= 8 mm). **Not met** for thighs, shanks and feet at large angles: feet p50 8–11 mm, p95 17–27 mm, max 47 mm. The causes are structural (section 4). | `fk_parity_report.json` |
| Masses and inertias lumped from the USD | VERIFIED: total 56.175 kg equals the USD minus the three orphan bodies | `test_structure`, `extract_usd_body_properties.log` |
| motion_lib variant loads in GR00T SONIC's real `Humanoid_Batch` (read-only clone imported by path). Its `fk_batch` matches MuJoCo on all 5 clips; `mesh_fk` works. | VERIFIED: bodies and 15 extend points within 1.2e-6 m (float32). The PKL exporter's numpy FK is within 1e-15 m. A full SONIC motion_lib / training run is **not** done. | `check_sonic_humanoid_batch.log`, `sonic_humanoid_batch_check.json`, `export_serial_motionlib.log`, `tests/test_serial_mjcf.py::test_sonic_humanoid_batch_fk` |
| GMR (fresh clone, patched) retargets LAFAN1 onto Dropbear | VERIFIED: 5 clips, contract-valid CSVs | `gmr_lafan1_batch.log`, `tests/test_gmr_serial.py` |
| GMR `smplx_to_dropbear.json` | **UNVERIFIED**: generated and checked structurally only; there are no SMPL-X body models or data on disk | `make_gmr_configs.log` |
| Retargeted clips settled in Isaac or tracked by a policy | **Not done**: this track is CPU-only. Next step: `tools/settle_motion.py` | none |

## 2. What the serial model is

The builder writes these files to `data/robot/`:

| File | Purpose |
|---|---|
| `dropbear_serial.xml` | Main model for GMR, MuJoCo and mjlab. Free root `pelvis`, 22 hinges, welded `torso_link` (anchor) and `head_link` bodies, surrogate position servos. |
| `dropbear_serial_motionlib.xml` | Same kinematics and dynamics for PHC-style `Humanoid_Batch` parsers (SONIC motion_lib, ProtoMotions). Every non-root body has exactly one hinge; there are no jointless bodies (torso and head become sites); actuators are `<motor>`. |
| `dropbear_serial_scene.xml` | `dropbear_serial.xml` plus a floor and a light, for viewers. |
| `dropbear_serial.json` | Metadata: input paths and SHA-256s, frames, per-joint fit residuals, segment assignment of all 90 USD bodies, surrogate actuator parameters, `motionlib_extend_config`, self-check. |
| `meshes/mesh_<body>.stl` | One visual mesh per serial body: the union of the convex hulls of its USD members' collision hulls, baked into the body frame. Visual only (`contype 0`). SONIC's `mesh_fk` height fix needs them. |
| `usd_body_properties_45586414.json` | Per-body mass, COM, inertia and collision-hull subsample, read from the USD with pxr. |
| `gmr/bvh_lafan1_to_dropbear.json`, `gmr/smplx_to_dropbear.json` | GMR IK configs (section 6). |

### 2.1 Tree, names, frames

```
pelvis (free root; USD 'world' body frame = pelvis * inv(pelvis_in_root))
├─ {side}_hip_pitch_link (+y) ─ {side}_hip_roll_link (+x) ─ {side}_hip_yaw_link (+z, thigh)
│    └─ {side}_knee_link (measured axis, tilted ~10 deg; the link frame is rotated so the axis is +y)
│         └─ {side}_ankle_pitch_link (+y, U-joint cross) ─ {side}_ankle_roll_link (+x, foot)
├─ {side}_shoulder_pitch_link (+y) ─ {side}_shoulder_roll_link (+x) ─ {side}_shoulder_yaw_link (+z, upper arm)
│    └─ {side}_elbow_link (measured axis ~+y, forearm) ─ {side}_wrist_roll_link (measured screw ~+x, hand)
├─ torso_link (welded, at the anchor head_5mm_ujoint_base__5__1)   [full variant only]
└─ head_link (welded, at head_u_joint_center__8__1)                [full variant only]
```

- **Names.** Link names follow G1, so G1 GMR templates map by name. Joint names are exactly the semantic DOF names
  (`left_hip_pitch`, ..., `right_wrist_roll`). MuJoCo `qpos` order is the tree order, not the semantic order, so
  always map by name: `SerialModel.sem_qadr`, `load_gmr_npz`.
- **World frame.** The USD root frame at the semantic zero, with x forward, y left, z up.
- **Zero pose.** All 22 joints at 0 is the semantic zero: legs straight, soles level, arms hanging, forearms pointing
  forward (G1 elbow 0).
- **Body frames.** They are world-aligned at the zero pose, except the knee, elbow and wrist-roll links. Those are
  rotated so their measured joint axis becomes a principal axis. This is G1's tilted-link pattern, and it is required
  because `Humanoid_Batch` reads axes with `int()`.
- **Root.** The root body `pelvis` sits at the calibration's `rest_transforms.pelvis_in_root` (the hip-centre midpoint).
  USD root = pelvis * inv(`pelvis_in_root`) (`SerialModel.root_from_pelvis`). Its default height is 0.986 m, the sole
  height at the zero pose.
- **Sites.** One site per tracked USD body (CONTRACTS 5.1), named `usd_<USD body name>` and placed at that body's frame,
  plus `usd_world` (the USD root frame) and `{side}_sole`. A consumer can read the 14 BeyondMimic tracked-body poses
  directly from serial FK.

### 2.2 What is exact and what is fitted

The builder is `source/dropbear_wbc/kinematics/serial_model.py::fit_serial_model`. The reference is the calibration's
own forward model, `CalibrationFK`: `SemanticMap.semantic_to_motor` followed by `MeasuredFK`, built from the raw physics
sweep named in the calibration provenance.

| Joint group | Axes | Anchors | Residual (rms / p95 / max) |
|---|---|---|---|
| Hips (`serial3`) | Exact YXZ semantic axes (the definition) | 3 anchors, LS over 3000 uniform and 303 single-DOF samples, exact at zero | 11.7 / 23.8 / 35.9 mm thigh translation; orientation exact |
| Knee (four-bar) | Measured relative-rotation axis (9.9 deg tilt from y) | LS fixed pivot plus free zero position over the valid range | 8.3 / 15.6 / 18.6 mm at the ankle centre; zero-pose shift 7.6 mm |
| Ankle (`lut2d`) | Pitch +y, then roll +x (the definition) | 2 anchors (pitch axis 40 mm above the roll axis), LS over the calf-motor grid | 0.17 / 0.33 / 0.92 mm; measured yaw residual <= 0.69 deg |
| Shoulders (`serial3`) | Exact YXZ, the same order as the physical chain | LS | <= 0.81 mm |
| Elbow (four-bar) | Measured axis | LS fixed pivot plus free zero over the valid range | 3.1 / 6.0 / 7.4 mm at the wrist; zero shift 1.1–1.6 mm |
| Wrist roll | Measured screw (forearm frame) | Screw point | 0 |

The fits are recorded in the `fit` block of `dropbear_serial.json` and in `build_serial_mjcf.log`.

The model makes two structural approximations. No set of 22 semantic-valued hinges can avoid them:

1. **Hip chain order.** The physical hip is roll -> yaw -> pitch, and the roll axis sits about 8 cm above the pitch
   axis. The semantic (G1) hip is pitch -> roll -> yaw about idealised axes, so for combined hip angles the thigh
   translation cannot be reproduced exactly. A "physical-order" hip would be exact to about 1 mm, but its joint values
   would no longer be the contract's semantic angles (the linear hip map is off by up to 22 deg,
   CONTRACTS 2.1). This model keeps the contract.
2. **Polycentric four-bars.** The knee's instantaneous centre moves along its long rockers. A fixed hinge is off by
   about 8 mm rms at the ankle. The elbow is off by about 3 mm rms at the wrist.

### 2.3 Mass, inertia, collision, actuators

- **Mass lumping.** All 90 articulation bodies are lumped; the three orphan bodies from CONTRACTS 0.1 are excluded.
  Each body goes to the physical segment it moves least relative to, measured by the COM spread over all 5023 sweep
  samples plus 0.02 m/rad of rotation spread. Placement uses the verify-stage semantic-zero physics record. The 0.1
  minimum-inertia fix is applied.
  - Resulting masses: pelvis 25.61 kg (torso, head, hip-motor rotors), thigh 5.77, shank 3.50, ankle cross 0.10,
    foot 0.94, shoulder links 0.34 / 1.20, upper arm 1.84, forearm 0.92, hand 0.65.
  - `hip_pitch_link` and `hip_roll_link` are virtual links (the physical hip rotors ride on the pelvis, with their
    COM on the roll axis), so each gets a 0.01 kg placeholder taken from the pelvis.
  - The per-body assignment is in `segment_assignment`.
- **Collision.** Foot-sole boxes come from the sole hulls (26.5 x 10.1 cm, bottom at the sole plane). There are also a
  hand box and a torso box from the USD collision hulls. Robot geoms use `contype=1 conaffinity=0`, so they collide
  with a floor but not with each other. Capsules are visual only.
- **Actuators (surrogate).**
  - Semantic-space PD values come from the motor PD through `J = dm/ds` at the standing pose: effort =
    sum |J| x effort, kp = sum J^2 x kp.
  - Examples: knee 167 N m / kp 62; elbow 8.3 N m (the four-bar trades torque for its ~4x range); ankle pitch/roll
    179 / 235 N m.
  - They are only good enough for kinematic tools and rough simulation. The plant's torque limits live in motor space.

## 3. How to regenerate

A single command rebuilds everything from whatever `data/calibration/dropbear_semantic_calibration.json` is current.
It runs on the CPU in about 6 s:

```bash
.venv-newton/Scripts/python.exe tools/build_serial_mjcf.py
```

- The command reads the calibration, the raw sweep named in its `provenance.raw_sweep`, and its verify NPZ if present.
- If `data/robot/usd_body_properties_<sha8>.json` is missing, it re-extracts it from the USD with
  `tools/extract_usd_body_properties.py` (pxr, `.venv-newton`, 8 s).
- It then compiles the model and runs a quick FK self-check.

After a calibration change, also rerun the dependants:

```bash
.venv-gmr/Scripts/python.exe tools/make_gmr_configs.py      # GMR offsets/scale
.venv-gmr/Scripts/python.exe tools/gmr_lafan1_batch.py      # clips + comparison (~2.5 min)
.venv-gmr/Scripts/python.exe tools/export_serial_motionlib.py data/motions/gmr_lafan1
.venv-gmr/Scripts/python.exe -m pytest tests/test_serial_mjcf.py tests/test_gmr_serial.py -q
```

Stale outputs fail the tests:

- `test_matches_current_calibration` compares the calibration and raw-sweep SHA-256s.
- `test_gmr_configs` checks the config provenance.
- `test_gmr_lafan1_clips` checks the clip provenance.

## 4. FK parity (target < 1 cm)

`tests/test_serial_mjcf.py` checks the model against three references and writes
`logs/serial_mjcf_gmr/fk_parity_report.json`:

- **Random.** 256 uniform random semantic poses inside the valid ranges, compared with `CalibrationFK`.
- **Motion library.** 3878 frames sampled from `data/motions/*/semantic` (a realistic joint distribution), compared with
  `CalibrationFK`.
- **Verify physics.** The 82 settled Isaac poses of `logs/gpu_pipeline/calib/raw/semantic_verify_v3.npz`, using
  semantic angles measured from the physics link rotations.

Positions are in mm, as p50 / p95 / max. The left and right sides are within about 1 mm of each other; the table shows
the left side.

| Body | Random | Motion library | Verify physics | Meets < 1 cm at p95? |
|---|---|---|---|---|
| pelvis / anchor (torso) / head | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | yes |
| upper arm | 0 / 0 / 0.2 (right: max 1.1) | 0 / 0 / 0.05 | 0 / 0 / 0.01 | yes |
| forearm, hand | 2.6 / 6.2 / 7.4 | 2.1 / 5.1 / 7.4 | 3.1 / 7.4 / 7.5 | yes |
| thigh | 7.7 / 24.7 / 35.7 | 2.7 / 15.3 / 28.6 | 5.0 / 15.7 / 22.1 | no |
| shank, foot | 11.4 / 27.3 / 43.3 | 8.2 / 24.0 / 39.6 (right max 46.8) | 9.2 / 17.1 / 25.1 | no |

- **Orientation.** Errors are below 0.6 deg everywhere. The only residual is the ankle's measured yaw, which a
  pitch/roll U-joint cannot represent.
- **Zero pose.** The model is exact at the semantic zero, except that the knee and elbow subtrees sit at their
  least-squares zero position: 7.6–7.7 mm and 1.1–1.6 mm respectively (`test_zero_pose_exact`).
- **Foot error.** The foot is below 1 cm in 70 % of motion-library frames and 40 % of uniform random poses. The two
  causes are the hip chain order (thigh) and the polycentric knee (shank and foot); see section 2.2.
- **Settle.** Contract clips made through this model are settled on the USD plant by `tools/settle_motion.py`. The
  GMR converter also uses the plant forward model, not the serial model, for ground contact and contact hints
  (section 6.3).

## 5. GMR setup

- **Clone.** GMR is a fresh shallow clone at `$DROPBEAR_UPSTREAM/GMR-dropbear`, commit `bb1bbe40`. The same commit
  is at `$DROPBEAR_UPSTREAM/GMR`, which stays untouched.
- **Environment.** `.venv-gmr` is built with uv and CPython 3.11.11, with `UV_CACHE_DIR=.uv-cache`.
  - Packages: numpy 2.4.6, scipy 1.17, mujoco 3.14.0, mink 1.3.0, qpsolvers 4.13 with daqp and quadprog,
    torch 2.14.0+cpu, loop-rate-limiters, rich, imageio[ffmpeg], pytest, and GMR installed editable with `--no-deps`.
  - Windows extras skipped: proxsuite (`qpsolvers[proxqp]`), smplx, redis and opencv. GMR imports without them.
  - `setup.py` reads the README with the locale codec, so the editable install needs `PYTHONUTF8=1`.
  - Logs: `venv_gmr_create.log`, `venv_gmr_install_{core,torch,gmr}.log`.
- **Patch.** `tools/setup_gmr.py` is idempotent and refuses to touch the reference clone. It records its diff and a
  NOTICE in `third_party/gmr_dropbear/` (`setup_gmr.log`). It makes two changes:
  1. **mink fix.** GMR called `mink.solve_ik(cfg, tasks, dt, solver, damping, limits)` positionally. In mink 1.3 the
     sixth positional parameter is `safety_break`, so the joint limits were silently not applied. The call now passes
     `damping=` and `limits=` by keyword.
  2. **Registration.** `params.py` registers `dropbear`: MJCF `data/robot/dropbear_serial.xml`, IK configs
     `data/robot/gmr/{bvh_lafan1,smplx}_to_dropbear.json`, base `pelvis`. The root can be moved with
     `DROPBEAR_WBC_ROOT`. `dropbear` is also added to the `--robot` choices of `scripts/{bvh,smplx}_to_robot.py`.
- **Configs.** `tools/make_gmr_configs.py` derives them from the G1 LAFAN1 template and the H1 SMPL-X template:
  - **Body mapping.** By name. The G1 hand `wrist_yaw_link` maps to `wrist_roll_link`. The H1 `hip_roll_link` maps to
    the thigh, and `shoulder_roll_link` maps to the upper arm.
  - **Rotation offsets.** `offset_db = offset_template * R_template(q=0)^T * R_dropbear(q=0)`, with both zero-pose
    rotations from MuJoCo FK. All three robots share the anatomical zero pose (arms hanging, forearms forward).
  - **LAFAN1 scale.** A hip-height ratio: `0.9857 / (0.9233 * 1.75/1.8) = 1.0981`. Here 0.9233 m is the p95 Hips
    height over walk1_subject1 5–60 s, and 1.75/1.8 is GMR's fixed LAFAN height ratio.
  - **SMPL-X scale.** The H1 scale times the pelvis-to-ankle ratio, 1.0337.
  - **Weights.** Hip-joint and knee position weights are set to 0. Dropbear's thigh:shank ratio is 0.59:0.32 m and its
    knee is a fitted pivot, so human knee positions are not meaningful targets. Arm position weights are also set to 0
    in the file; see the direction mode in section 6.

## 6. Retargeting LAFAN1 onto Dropbear

- **Data.** LAFAN1 is a shallow clone of `ubisoft-laforge-animation-dataset` at `$DROPBEAR_UPSTREAM/lafan1`
  (commit `94084601`; clone log `lafan1_clone.log`). `lafan1.zip` was unpacked to `$DROPBEAR_UPSTREAM/lafan1/bvh`
  (77 clips).
- **License.** LAFAN1 is **CC BY-NC-ND 4.0**. Retargeted clips are Adapted Material for **internal non-commercial R&D
  only**. They must never be shared or redistributed. Every sidecar records the license with `redistributable: false`.

### 6.1 `tools/gmr_retarget.py` (headless GMR, `.venv-gmr`)

The runner runs stock GMR on the Dropbear model and adds the following. Each addition is recorded in the output NPZ
`meta`.

- **Warm start.** It makes 50 IK calls on the first frame. Without them, a mid-clip window starts with a convergence
  transient from the default pose.
- **`DropbearMotorLimit`** (`source/dropbear_wbc/motion/gmr_limits.py`). This is a mink `Limit`. At each IK iteration
  it adds `m_lo <= m(s) + J ds <= m_hi` for the chain motors of the four `serial3` groups and the two calf-motor pairs.
  It uses the calibration's own unclipped inverse maps with finite-difference Jacobians.
  - Why: the model's hip and shoulder ranges are bounding boxes, so without it GMR produced motor-infeasible hip
    combinations. SemanticMap then clipped them by up to 32.5 deg on the dance clip, which moves the feet.
  - With it, the clipping left after SemanticMap is at most 2.1 deg. That residual is the ankle, where SemanticMap's
    41 x 41 feasibility grid is coarse.
  - The QP stayed feasible on every frame: the fallback list is empty.
  - Cost: IK goes from about 4 ms/frame to 26–44 ms/frame.
- **Arm mode `direction`** (the default). Dropbear's forearm is 0.105 m against about 0.25 m for a human.
  - The elbow and wrist targets are rebuilt along the human upper-arm and forearm directions, with Dropbear's segment
    lengths divided by the scale. They are driven by position (cost 20), with weak orientation terms (cost 1).
  - *Corrected 2026-09-24 (review fix, `--arm-anchor robot`, the new default):* the rebuilt targets used to start at
    the scaled HUMAN shoulder. Dropbear's shoulder centres are 0.477 m apart against LAFAN's 0.367 m (scaled), so the
    arms were pulled 7-14 deg (p50) toward the midline, which contradicted "follows the human directions". They now
    start at the ROBOT shoulder (`shoulder_yaw_link` origin; pelvis target + previous pelvis orientation) and are mapped
    back through GMR's own per-body affine transform, and `l_ua` is measured from `shoulder_yaw_link` (0.2985 m, was
    0.3035 m from the pitch link). The solve records the robot-vs-human direction error (`arm_dir_err`): upper arm p50
    1.2-4.3 deg on every clip except the right arm of the arm-raise (10.1 deg p50, raised overhead against the comfort
    bounds); p95 2.4-14.3 deg; forearm p50 11-14 deg (logs/review_fixes/gmr_v2/gmr_lafan1_batch.log).
  - Why: with the template's orientation tasks (upper-arm rotation weight 100), LAFAN's upper-arm twist drove the
    full-range YXZ shoulder onto its flipped branch `(p + pi, pi - r, y + pi)`, or to 180 deg yaw. *(The former claim
    that "G1 has the same problem on that clip, but its joint limits stop it" is withdrawn: GMR's G1 solve of that
    window had diverged, see 6.3.)*
- **Root policy (`tools/make_gmr_configs.py --root-policy`, added 2026-09-24).** G1 has a 3-DoF waist; the serial
  model welds `torso_link` to the pelvis, so the template's chest (`Spine2`, rotation weight 100) and pelvis (`Hips`,
  weight 10) tasks act on one rigid body, and the pelvis followed the human CHEST (Hips error p50 13-20 deg, pelvis
  pitched +9..+13 deg on walk/run). Default now `blend`: both tasks get the pelvis weight, so the rigid torso takes the
  least-squares compromise. Result: Hips / Spine2 rotation error p50 7.8-10.4 / 9.0-15.7 deg (was 13-20 / 1.6-4.7).
  `chest` restores the template weights. Recorded in the config's `_dropbear.root_policy`.
- **Pre-roll (`--preroll`, batch default 1 s, added 2026-09-24).** The solve starts 1 s before the window (not
  recorded), for both robots, so the window starts from a tracked configuration.
- **Shoulder comfort bounds and analytic arm seed.** The comfort bounds are G1's anatomical ranges, applied to the IK
  only; the model ranges stay the full motor range (`gmr_limits.comfort_limit`). The seed is
  `source/dropbear_wbc/motion/gmr_arm.py`:
  - It computes the exact shoulder and elbow angles for the target directions.
  - On the first frame it picks the branch inside the comfort bounds. After that it picks by continuity, unwrapping
    each angle to its nearest value.
  - A re-seed is skipped if it would move a joint more than 0.35 rad, unless the arm is stuck (more than 0.12 m error).
  - Residual joint-speed spikes remain near 90 deg abduction. There the physical shoulder has a gimbal singularity
    (the pitch and yaw axes align), which the plant shares, so the runner reports these spikes rather than hiding them.

### 6.2 `tools/gmr_to_dropbear_csv.py` (+ `source/dropbear_wbc/motion/gmr_serial.py`)

The converter turns the GMR output into a contract clip:

1. **Semantic trajectory.** The GMR joint vector, mapped by name, is the requested semantic trajectory.
2. **Motors.** `SemanticMap.semantic_to_motor` produces the motors. Saturation statistics match the G1 route
   (`saturation_stats`). `ik_at_limit` lists the frames where the IK sits at a model limit.
3. **Root.** The USD `world` body = pelvis * inv(`pelvis_in_root`).
4. **Ground and contacts** (*changed 2026-09-24, review fix*). The contact hints come from the HUMAN source feet: raw
   LAFAN `LeftFoot/LeftToe` (and right), a foot is in contact when its sole proxy (lower of toe height above the clip's
   toe ground and ankle height above its standing height) is within 3 cm of the ground. The ground is then set PER
   FRAME: the plant contact-foot sole (MeasuredFK + sole hulls) goes to z = 0 on source-contact frames, interpolated over
   flight and smoothed (the settle tool's `ground_correction`). Before, one clip-wide 5th-percentile shift was used and
   the hints came from the IK robot feet: frames where the knee limit pushed the IK feet through the floor lifted the
   whole clip, and dance / run / jump had 49 / 86 / 81 % "no foot in contact" frames against 3 / 11 / 49 % in the
   source. The sidecar records `metrics.contact_agreement` (source vs plant-foot contacts after grounding, same rule;
   flagged above 10 %: every clip except jump is flagged, 14-32 % disagreement, i.e. the IK feet do not reproduce the
   human footfalls where the knee range binds). The xy position is re-centred on the first frame.

   *Library v4 (foot_contact track, 2026-09-24), now the default of `tools/gmr_to_dropbear_csv.py`.* The clip is
   resampled to 50 Hz and passed to the foot-contact stage (`dropbear_wbc.motion.foot_contact`), which replaces the
   per-frame grounding: stance soles are pinned flat on z = 0 at world-locked poses with the plant forward model, and
   the pelvis and leg motors are re-solved.
   - The stage's contacts need a SPEED term (`gmr_serial.HUMAN_HYSTERESIS`): touch-down below 3 cm AND 0.3 m/s,
     lift-off above 5 cm OR 0.6 m/s. The speed is that of the slower of toe and ankle.
   - The height-only rule above merged LAFAN walking steps. On walk1 it gave 85-90 % stance, with the source ankle
     drifting 0.4-0.6 m per phase.
   - The agreement check then uses a 5 mm plant band.
   - `--no-foot-contact` = the route described above. Results: `logs/foot_contact/PROGRESS.md`.
5. **Output.** The files are CONTRACTS section 3 CSV + sidecar + `semantic/<clip>.semantic.csv`. The semantic file
   holds the angles the motors actually realise. The sidecar records `retarget_method: gmr-serial-v1 ...`, the LAFAN1
   license, the serial-model and calibration SHAs, the GMR commit and meta, and the metrics (joint speeds, plant-FK
   foot slip, penetration and float, GMR task errors).
6. **G1 comparison.** `--g1-npz` sends the same clip, retargeted by GMR onto G1, through the existing G1-intermediate
   route (`g1_to_dropbear`), for comparison.

### 6.3 Clips and comparison with the G1-intermediate route

`tools/gmr_lafan1_batch.py` produces the clips in about 2.3 min on the CPU. The outputs are:

- `data/motions/gmr_lafan1/*.csv` plus `catalog_gmr_lafan1.json`;
- the G1-route counterparts in `data/motions/gmr_lafan1_via_g1/`;
- `logs/serial_mjcf_gmr/route_comparison.{md,json}`, per-clip reports in `route_compare/`, and montages (serial model
  plus scaled human targets) in `renders/`.

LAFAN1 has no wave clip. The wave-like clip is a right-arm raise from dance2_subject1.

"Clipped" means SemanticMap had to change the requested semantic value. For the G1 route this includes G1 joint values
outside Dropbear's ranges. The slip, float and velocity columns are computed identically for both routes, on the plant
forward model.

*Restated 2026-09-24 (review fixes).* The first comparison (kept below, struck) counted a FAILED G1 solve as evidence:
on the arm-raise window GMR's G1 IK had diverged (waist yaw 80 deg, Spine2 rotation error p50 135 deg, hands p95
0.44-0.79 m, left shoulder stuck at the `g1_mocap_29dof` limit corner roll -0.6 / yaw 2.0 rad on 100 % of frames), and
`g1_to_dropbear` did not wrap its Euler outputs, so a requested yaw of 290 deg (= -70 deg) was clipped to 180 deg. The
"100 % clipped / shoulder roll 122.6 deg" row and the "G1 route clips the shoulder by 123 deg" headline are
**withdrawn**. Fixes: Euler outputs wrapped to (-pi, pi] (`g1_to_dropbear.wrap_to_pi`), a 1 s solve pre-roll for both
robots (from 7.0 s the G1 solve tracks: Hips 4.8 deg, Spine2 3.4 deg, hands p95 0.07/0.16 m), and an IK-HEALTH gate
(`gmr_lafan1_batch.ik_health`: root/chest rotation error p50 <= 15 deg, hand/forearm position error p95 <= 0.15 m, no
joint at a limit on > 50 % of frames) that marks a failed route INVALID. Current table (all review fixes applied:
robot-shoulder arm anchor, blend root policy, source-foot contacts; `logs/review_fixes/gmr_v2/route_comparison.md`):

| Clip (window) | Frames | IK health serial / G1 | GMR-serial clipped: frames / worst | G1 route clipped: frames / worst | Stance slip p95 (serial / G1) | Stance float p95 (serial / G1) | Frames over motor velocity limit (serial / G1) |
|---|---:|---|---|---|---|---|---|
| walk1_subject1 (13–28 s) | 450 | ok / ok | 0.4 % / 0.5 deg | 41.6 % / knee 25.3 deg | 0.36 / 0.39 m/s | 0.020 / 0.045 m | 0 / 0 |
| arm-raise, dance2_subject1 (8–17 s) | 270 | ok / **FAIL** (right hand p95 0.17 m) | 0.0 % / 0.0 deg | INVALID | 0.33 / 0.30 | 0.035 / 0.055 | 0 / 20 |
| dance2_subject1 (18–38 s) | 600 | ok / **FAIL** (hands p95 0.18 m) | 12.8 % / 1.9 deg (ankle) | INVALID | 0.32 / 0.39 | 0.018 / 0.044 | 61 / 82 |
| run1_subject2 (28–40 s) | 360 | ok / **FAIL** (Hips p50 97.5 deg) | 50.6 % / 1.8 deg (ankle) | INVALID | 0.55 / 0.27 | 0.014 / 0.035 | 102 / 25 |
| jumps1_subject1 (8–20 s) | 360 | **FAIL** (Spine2 p50 15.7 deg) / ok | 22.2 % / 1.8 deg (ankle) | 56.7 % / knee 42.4 deg | 0.43 / 0.35 | 0.023 / 0.090 | 73 / 85 |

With the health gate, only walk and jumps have a valid G1 baseline; the knee rows there (25 and 42 deg beyond
Dropbear's 47 deg knee) are genuine. The GMR-serial jump solve fails the gate narrowly (chest rotation error p50
15.7 deg > 15): the blend root policy's compromise on a clip with deep crouches, with the knees at their limit on 42-44 %
of frames (the knee-range finding below).

First comparison (2026-09-24 11:00), **superseded** (see above):

| Clip (window) | Frames | GMR-serial clipped: frames / worst | G1 route clipped: frames / worst | Stance slip p95 (serial / G1) | Stance float p95 (serial / G1) | Frames over motor velocity limit (serial / G1) |
|---|---:|---|---|---|---|---|
| walk1_subject1 (13–28 s) | 450 | 1.6 % / 0.7 deg | 41.6 % / knee 25.3 deg | 0.35 / 0.39 m/s | 0.036 / 0.045 m | 0 / 0 |
| arm-raise, dance2_subject1 (8–17 s) | 270 | 2.6 % / 0.1 deg | ~~100 % / shoulder roll 122.6 deg~~ (G1 IK had failed) | 0.35 / 0.36 | 0.034 / 0.049 | 3 / 6 |
| dance2_subject1 (18–38 s) | 600 | 16.2 % / 2.1 deg (ankle) | 49.3 % / knee 80.9 deg | 0.34 / 0.39 | 0.105 / 0.044 | 48 / 83 |
| run1_subject2 (28–40 s, 1.6–1.8 m/s) | 360 | 50.0 % / 1.9 deg (ankle) | 83.3 % / knee 65.3 deg | 0.49 / 0.46 | 0.055 / 0.067 | 110 / 92 |
| jumps1_subject1 (8–20 s) | 360 | 21.1 % / 1.9 deg (ankle) | 56.7 % / knee 42.4 deg | 0.34 / 0.35 | 0.047 / 0.090 | 68 / 85 |

**What the comparison shows.** This is a qualitative comparison. It is not evidence that either route can be tracked.

- **Clipping.** The G1 route clips G1's knee flexion (up to 81 deg beyond Dropbear's 47.2 deg knee) and its shoulder
  representation after the fact, so the feet and hands end up where the source did not put them. GMR on the serial
  model solves the IK against Dropbear's own joint and motor limits, so the requested and realised poses agree to
  within 2 deg.
- **Knee range.** Dropbear's knee range (-4.9..47.2 deg, from the 0–30 deg knee crank) is the binding limit for
  running, jumping and dancing. GMR hits it on 38–44 % of the frames of run and jumps. There the IK trades foot
  position for root height and hip pitch: the GMR foot task error is p95 0.17 m for run and 0.15 m for dance, against
  0.03 m for walk. Neither route can make Dropbear crouch like a human.
- **Joint agreement.** On the walk clip the leg joints of the two routes agree within p50 2–8 deg. The knee differs
  most, because the G1 route clips it (`route_compare/lafan1_walk1_subject1.json`, `semantic_difference_deg`). The arm
  joints differ by p50 2–17 deg, mostly in shoulder yaw and wrist roll. The direction mode deliberately ignores the
  human upper-arm twist, which the G1 route follows through G1's orientation tasks.
- **Remaining problems.** Stance slip p95 of about 0.35 m/s (at the contact speed threshold) and joint-speed spikes
  appear in both routes. They come from the source motion, from contact detection, and from the shoulder singularity.
  Settle and tracking still have to handle them.

## 7. Using the model in SONIC motion_lib, mjlab and ProtoMotions

- **Contract clips.** `data/motions/gmr_lafan1/*.csv` are ordinary contract clips. The existing pipeline goes
  `tools/settle_motion.py` -> motion NPZ -> `Dropbear-Tracking-Flat-v0`, with no serial model involved.
- **SONIC motion_lib (GR00T-WholeBodyControl `gear_sonic`).**
  - Robot config: `asset.assetFileName = dropbear_serial_motionlib.xml`, which satisfies `Humanoid_Batch`: one hinge
    per non-root body, integer axes, a `range` on every joint, named actuators.
  - Take `extend_config` from `dropbear_serial.json["motionlib_extend_config"]`: the anchor, head, soles and USD
    tracked bodies as fixed offsets.
  - Motion data: `tools/export_serial_motionlib.py` writes `{clip: {root_trans_offset, pose_aa, dof, root_rot, fps}}`
    (`logs/serial_mjcf_gmr/motionlib/dropbear_motionlib.pkl`, plain pickle, readable by `joblib.load`). It is built
    the same way `convert_soma_csv_to_motion_lib.py` builds G1 data.
  - Verified with SONIC's real `Humanoid_Batch` (`tools/check_sonic_humanoid_batch.py`, imported read-only from
    `$DROPBEAR_UPSTREAM/GR00T-WholeBodyControl`):
    - it loads the MJCF;
    - `fk_batch` on the exported `pose_aa` matches MuJoCo within 1.2e-6 m, bodies and extend points alike;
    - `mesh_fk` runs.
  - Two incompatibilities surfaced along the way and were fixed in the writer: comments inside `<actuator>` (lxml
    `getchildren()` returns them), and several mesh geoms per body (`mesh_fk` duplicates them quadratically).
  - SONIC's import dependencies were added to `.venv-gmr`: easydict, loguru, lxml, omegaconf, hydra-core, open3d
    (`venv_gmr_install_sonic_deps.log`).
  - Next steps: a SONIC robot and motion config for Dropbear (joint groups, default pose `standing_semantic_pos`,
    surrogate gains from `surrogate_actuators`), then a motion_lib load and a short training smoke run.
    **UNVERIFIED.**
- **mjlab.** Create `asset_zoo/robots/dropbear/` in a copy, modelled on `unitree_g1/g1_constants.py`:
  - the spec from `data/robot/dropbear_serial.xml`;
  - `BuiltinPositionActuatorCfg` per joint group, with effort and kp from `surrogate_actuators`;
  - a standing keyframe from `standing_semantic_pos`.

  The tracking task needs `anchor_body_name="torso_link"` and body names drawn from the serial links. The
  `csv_to_npz.py` input is `logs/serial_mjcf_gmr/motionlib/mjlab_csv/<clip>.csv`: pelvis pos, quat xyzw, then the
  22 joints in `joint_names.txt` order. **UNVERIFIED** (not run).
- **ProtoMotions.** It uses the same PHC `Humanoid_Batch` MJCF conventions as the motion_lib variant. It also needs a
  robot config. **UNVERIFIED.**
- **Warning.** A policy trained purely on the serial model (mjlab or SONIC) learns the surrogate's dynamics: fitted
  kinematics, lumped masses, semantic-space PD. Transfer to the plant goes semantic -> SemanticMap -> motors and must be
  validated sim2sim on the USD (Isaac, or Newton with `tools/newton_bridge.py`).

## 8. Known gaps

- The feet miss the < 1 cm parity target at large hip and knee angles. The causes are structural (section 2.2).
- SemanticMap's ankle feasibility grid clips up to about 2 deg near the boundary of the ankle roll range.
- Surrogate actuators are linearised at the standing pose. The elbow's effective torque varies about 4x over its
  range.
- `smplx_to_dropbear.json` and the SONIC, mjlab and ProtoMotions consumption paths are **UNVERIFIED** (section 7).
- The GMR outputs have not been settled or tracked, so there is no physics evidence yet that any of these clips is
  feasible on the plant.

## 9. Files

| Path | Role |
|---|---|
| `tools/extract_usd_body_properties.py` | USD mass properties and collision hulls (pxr) |
| `tools/build_serial_mjcf.py` | Single-command build and self-check |
| `source/dropbear_wbc/kinematics/serial_model.py` | `CalibrationFK`, `fit_serial_model`, MJCF writer, `SerialModel`, `evaluate_parity` |
| `tests/test_serial_mjcf.py` | Structure, zero-pose and FK parity tests (3 references), variants |
| `tools/setup_gmr.py`, `third_party/gmr_dropbear/` | GMR patch and registration, diff, NOTICE, commit |
| `tools/make_gmr_configs.py`, `data/robot/gmr/*.json` | IK configs from the G1 and H1 templates |
| `tools/gmr_retarget.py` | Headless GMR with the motor limits, arm direction mode, comfort bounds and seed |
| `source/dropbear_wbc/motion/gmr_limits.py`, `gmr_arm.py` | mink motor-limit constraint, comfort bounds, analytic arm seed |
| `tools/gmr_to_dropbear_csv.py`, `source/dropbear_wbc/motion/gmr_serial.py` | GMR -> contract CSV; G1-route comparison |
| `tools/gmr_lafan1_batch.py` | The 5 LAFAN1 clips, comparison and renders |
| `tools/export_serial_motionlib.py` | SONIC/PHC motion_lib PKL and mjlab CSV export with the FK check |
| `tools/check_sonic_humanoid_batch.py` | Loads the model in SONIC's real `Humanoid_Batch` and compares FK and `mesh_fk` |
| `tests/test_gmr_serial.py` | Configs, converter, clips, motor-limit constraint |
| `logs/serial_mjcf_gmr/PROGRESS.md` | Checkpoints |
