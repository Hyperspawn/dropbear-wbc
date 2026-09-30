# Dropbear arm teleop (xr_teleoperate pattern, simulation)

This is the Dropbear equivalent of Unitree's `xr_teleoperate`: a WebXR headset (or a keyboard, or a script) drives the
robot's wrists; an arm IK turns wrist targets into joint targets; the robot follows through the low-level SDK. It runs
on the `dropbear_hg-v1` SDK and the Newton bridge (docs/SDK.md) with the robot **hanging** (`--fixed-base`, Unitree's
bench mode). Hardware is out of scope. Dropbear arms have 5 DoF each (shoulder pitch / roll / yaw, elbow, wrist roll)
and no hands.

Every number below cites a log under `logs/teleop/` (paths relative to it unless they start elsewhere). Anything
without a log is marked **UNVERIFIED**.

## 1. Status at a glance

| Item | Status | Evidence |
|---|---|---|
| Arm IK in the semantic space (calibration-derived chain, priority solver, limits, posture bias, warm start, reload) | VERIFIED (unit tests + benchmark) | `pytest_teleop_venv_teleop.log`, `ik_benchmark.json` |
| Gravity feed-forward (xr_teleoperate `pin.rnea` equivalent) | *Corrected 2026-09-24 (review):* the old claim covered only the elbow motor at one shoulder pose, and its worst error was 0.76-0.80 N·m, not ~0.6. Now: **shoulder motors VERIFIED** at 6 poses x 2 arms (`PROBE_SETS['shoulder']`, passive damping 0) within **0.06 N·m** (loads 0.9-8.6 N·m); wrist roll 0.02; **elbow motor max 0.45-0.59 N·m, RMS 0.22-0.23 N·m** with the new measured-path forearm model (default; the old pivot model: max 0.72-0.80, RMS 0.35-0.46). The residual elbow error is large relative to small loads (4-7x at pose 0 of the elbow set). | `logs/review_fixes/gravity_forearm_model_compare.log`, `logs/review_fixes/probe_shoulder_pd0_analysis.log`, `probe_elbow_final_pd0.5_analysis.log`, `extract_arm_inertia.log` |
| Scripted source -> IK -> LowCmd -> Newton bridge (GPU, fixed base, 25 s) with recording | VERIFIED, both plant variants below | `verify_gpu_fig8v3_{pd0.5,contract}_summary.json` |
| Wrist tracking, bridge passive damping 50 (the bridge default until 2026-09-24 13:00, called "contract" in the logs) | **Poor: 31 mm mean / 42 mm RMS** (elbow cannot follow). *Obsolete as a plant:* Isaac never applied that damping (CONTRACTS 0.3); the bridge default is now 0 | `verify_gpu_fig8v3_contract_summary.json` |
| Wrist tracking, **diagnostic plant** (passive damping 0.5) | **1.8 mm mean / 1.9 mm RMS / 3.6 mm max** (FK of measured joints vs target) | `verify_gpu_fig8v3_pd0.5_summary.json` |
| Simulated wrist (ground-truth hand body) vs target, diagnostic plant | 5.3-5.9 mm mean with the measured elbow model (v4, CPU); 8.2-8.4 mm with the best-fit pivot (v3, GPU). The rest is an Isaac-vs-Newton elbow offset, section 5.3 | `verify_cpu_fig8v4_pd0.5_summary.json`, `verify_gpu_fig8v3_pd0.5_summary.json` |
| WebXR via Vuer: server on 127.0.0.1, simulated headset (hands and controllers), calibrate / start / stop via controller buttons or terminal keys, end-to-end with the bridge | **Events and data flow VERIFIED** with a **simulated** headset client. **Tracking accuracy with XR input UNVERIFIED** (*corrected 2026-09-24*, the earlier "robot tracks" was not supported): wrist FK-vs-target 20.8 mm mean / 53.6 mm max (xr5, controllers) and 19.6 / 53.2 mm (xr6, hands), of which the **IK residual** is 17.8 / 17.1 mm mean (53 mm max): the simulated operator's targets leave Dropbear's 0.28-0.41 m reachable shell after the 0.70 scale mapping, so these runs do not measure servo tracking | `test_teleop_devices.py`, `verify_cpu_xr5_controllers_pd0.5_summary.json`, `verify_cpu_xr6_hands_keys_pd0.5_summary.json` |
| Keyboard device end-to-end (stdin backend, scripted keys) | VERIFIED (path and events) | `teleop_cpu_keyboard3_pd0.5.json` |
| A real headset (Quest / Pico / Vision Pro), HTTPS certificates, Wi-Fi latency | **UNVERIFIED** (no device here) | none |
| `msvcrt` / `pynput` keyboard backends with a human | **UNVERIFIED** (key mapping unit-tested) | `test_teleop_devices.py` |
| LeRobot-v2.1 recording + GR00T `modality.json` draft | *Corrected 2026-09-24:* the sessions recorded before 13:00 are **NOT valid LeRobot v2.1** (loop-clock `timestamp` drifts from `frame_index/fps` by up to 5.8 s, frames skipped by an overrunning loop, no `meta/episodes_stats.jsonl`); only the GPU v3/v4 sessions have uniform timestamps. Since then: `timestamp = frame_index / fps`, real clock in `time.sim_s`, missed control periods in `time.skipped_steps`, the sim-clock loop stays on a fixed tick grid, `meta/episodes_stats.jsonl` + `meta/timing.json` written. **VERIFIED** on new sessions (LeRobot 1e-4 s check passes; `review_reload_live`: sim-time drift 8 ms, 0 skipped) | `logs/review_fixes/teleop_sessions_timing_audit.log`, `tests/test_teleop_recorder_timing.py`, `logs/review_fixes/review_reload_live.json` |
| Calibration hot reload during a live session | *Fixed 2026-09-24:* the rebuild (0.5-1.3 s) ran inside the 50 Hz loop and stalled LowCmd past the 100 ms watchdog. Now built on a background thread and swapped between steps: **VERIFIED live** (rebuild 1.30 s off-loop, worst loop period 31.5 ms, no watchdog event before the session end) | `logs/review_fixes/review_reload_live.json`, `tests/test_teleop_recorder_timing.py` |
| HOME phase termination | *Fixed 2026-09-24:* HOME now ends on LowState silence (`--state-timeout-s`), a wall deadline (`home_s + state_timeout_s`) or a second Ctrl-C (`home_aborted` transition); the recording and summary are still written | `tools/teleop_arm.py` (code review; no bridge-kill run yet) |
| Whole-body teleop (standing / walking while teleoperating) | Not done. Tracking / WBC policies exist since 2026-09-24 (docs/DEMOS.md); wiring teleop into one is open (section 9) | none |

## 2. Architecture

```
 operator                         tools/teleop_arm.py  (client, .venv-teleop, one loop at 50 Hz)
 ---------                        ------------------------------------------------------------------------
 headset browser --WebXR/wss-->  VuerXRSource (own process, shared memory)   \
 keyboard ------------------->   KeyboardSource                               > wrist targets, torso frame
 script (figure-8 / poses) -->   ScriptedSource                              /
                                    |  (XR: head_yaw reference, OpenXR->robot basis, OperatorMapping scale+offset)
                                    v
                                 DropbearArmIK.solve   priority IK in the semantic space, warm-started
                                    |  joint-velocity limit (4 rad/s semantic)
                                    v
                                 SemanticMap (calibration) -> 10 arm motor targets     ArmGravity -> tau_ff
                                    v
                                 LowCmd: arms {q*, dq*, tau_ff, kp, kd}; legs held at standing_motor_pos (2x legacy gains)
                                    |  rt/lowcmd  (ZMQ, tcp://127.0.0.1:5555)
                                    v
                                 tools/newton_bridge.py --fixed-base --realtime (500 Hz, motor PD per physics step)
                                    |  rt/lowstate (500 Hz; privileged sim block with the hand-plate poses)
                                    +--> recorder: data/teleop/<session>/ (LeRobot v2.1-like + modality.json draft)
```

Phases (`tools/teleop_arm.py`): MOVE_IN (all 22 motors ramp from the measured pose to the calibrated standing pose,
2 s) -> SETTLE (1 s) -> TELEOP -> HOME (arms ramp back, 1.5 s) -> exit with a damping LowCmd (Unitree `Passive`).
Operator events are handled in every phase: `start` / `toggle` (keyboard `r`, right controller A), `calibrate`
(keyboard `c`, left controller X), `stop` (keyboard `x` / Esc, both thumbsticks pressed: xr_teleoperate's soft
e-stop gesture), `home` (keyboard `h`). Scripted runs start tracking at TELEOP. In WebXR mode the terminal keys work
too (xr_teleoperate's r / q / s pattern). They are the only controls in hand-tracking mode, which has no buttons.

### 2.1 What mirrors xr_teleoperate and what differs

Reference: `$DROPBEAR_UPSTREAM/xr_teleoperate` @ 817fb00c (`teleop/robot_control/robot_arm_ik.py` `R1_A5_ArmIK`,
`robot_arm.py` `R1_A5_ArmController`, `teleop_hand_and_arm.py`) and `$DROPBEAR_UPSTREAM/televuer` @ 766de45e
(new shallow clone, MIT; notice `third_party/televuer_NOTICE.md`).

| Aspect | xr_teleoperate (R1_A5, 5-DoF arm) | Dropbear |
|---|---|---|
| Robot model | URDF + pinocchio, reduced model (waist / head locked) | Serial chain built from `data/calibration/dropbear_semantic_calibration.json` (no URDF: closed loops), `teleop/arm_ik.py` |
| IK solver | casadi + ipopt, `50·trans + 0.5·rot + 0.02·|q|² + 0.1·|q - q_last|²`, joint bounds, max 30 iterations | Same terms and weights, but as a **strict priority**: position first; orientation + posture (toward the calibrated standing pose) + smoothing in the exact position null space. Bounded damped Gauss-Newton with active set, singularity-robust damping, closed-form restart seeds. The single weighted cost is kept as `--ik-mode weighted` (section 5.1 says why it is not the default). |
| End-effector frame | `L_ee` 0.20 m along the wrist-roll x axis | The wrist point (wrist-roll anchor = hand-plate body origin); Unitree arm convention (identity with the forearm forward, palm in, thumb up) |
| Output smoothing | `WeightedMovingFilter([0.4, 0.3, 0.2, 0.1])` on the IK solution | Smoothing term in the null space + joint-velocity limit (no filter lag) |
| Command thread | 250 Hz, clips arm targets to 30 rad/s (not in sim), `dq* = 0` | 50 Hz loop (default; 100 Hz supported), `dq*` = finite difference of the targets (option `--no-dq-ff` = xr_teleoperate) |
| Gains | kp 50 / 40 / 30 (wrist), kd 2 | kp 200 (shoulder) / 600 (elbow **motor**) / 60 (wrist), kd 5 / 10 / 2 (section 5.3) |
| Gravity feed-forward | `pin.rnea(q, 0, 0)` from the URDF | Arm inertials extracted from the Newton model of the contract USD, on the semantic chain, mapped to motors by the calibration's derivative (`teleop/gravity.py`) |
| XR input | televuer: Vuer in a separate process, hand (25 joints) or controller tracking, `head_yaw` reference, head -> waist shift (+0.15, 0, +0.45) m | Same Vuer event handling and conventions (`teleop/frames.py`, `devices.py`); the head -> torso step is an `OperatorMapping` (scale 0.70 + per-hand offsets set by a calibration button) |
| Hands | dex-retargeting / Dex1-3 / Inspire / BrainCo | None (Dropbear has no hands); pinch / trigger values are recorded by nothing |
| Image to headset | teleimager (ZMQ / WebRTC), immersive / ego / pass-through | None: the Newton bridge renders no camera (pass-through-equivalent only) |
| Recording | EpisodeWriter JSON + images | LeRobot-v2.1-like parquet + npz + GR00T `modality.json` draft (section 7) |

IsaacTeleop (`$DROPBEAR_UPSTREAM/IsaacTeleop`, 1.6.x) was reviewed: CloudXR/OpenXR device IO, a retargeting graph,
mcap recording and LeRobot interop, but it targets Isaac Lab 3.0; this stack is Isaac Lab 2.2 + Newton, so it was not
used.

## 3. Frames and conventions

- **Robot / root frame**: the articulation root body `world` (rigid torso + pelvis), x forward, y left, z up.
- **Torso frame** (all device targets): root axes, origin at the pelvis centre `rest_transforms.pelvis_in_root`
  (0.041, -0.070, 1.112 m in the root frame). It lies on the shoulder midline, so left/right mirroring is `y -> -y`.
  It is the same origin and axes as the free root `pelvis` of the derived serial model (CONTRACTS 2.2), so poses can be
  exchanged with GMR / motion-lib tooling without conversion.
- **Semantic arm joints** (CONTRACTS 2): `shoulder_pitch` (+y), `shoulder_roll` (+x), `shoulder_yaw` (+z), `elbow`
  (G1 convention: 0 = forearm forward, +pi/2 = straight arm, positive = extension), `wrist_roll` (about the forearm).
- **End-effector orientation**: `R = Ry(pitch) Rx(roll) Rz(yaw) Ry(elbow) Rx(wrist_roll)`, identity at the semantic
  zero. It equals the "Unitree humanoid arm URDF convention" that televuer outputs, so XR wrist orientations pass
  through unchanged.
- **XR**: Vuer streams OpenXR-basis matrices (y up, z back) column-major. `frames.xr_to_head_relative` converts them to
  the robot basis (`T_ROBOT_OPENXR`), applies the hand-tracking initial-pose change (`T_TO_UNITREE_HUMANOID_*_ARM`),
  and expresses the wrists relative to the head position and head **yaw** (pitch / roll ignored, so looking around does
  not move the hands). `OperatorMapping.apply`: `p_torso = offset[side] + 0.70 · p_head_rel`, `R_torso = R_head_rel`.
  `calibrate` (left controller X or `c`) sets the offsets so that the operator's current wrists land on the robot's
  current wrists. The inverse (`head_relative_to_xr`) drives the simulated headset.

## 4. Kinematic model and IK (`source/dropbear_wbc/teleop/arm_ik.py`)

Product-of-exponentials chain per arm, root frame, at the semantic zero, all from the calibration JSON:

| # | joint | axis | point on the axis (left arm, m) |
|---|---|---|---|
| 1 | shoulder pitch | +y | shoulder centre `arm_points_rest_root.shoulder_center` (-0.035, 0.155, 1.621) |
| 2 | shoulder roll | +x | same |
| 3 | shoulder yaw | +z | measured screw `serial_screw_fits.LH_roll` (x, y = -0.035, 0.169): 14 mm lateral offset kept |
| 4 | elbow | +y | rotation: exact (it defines the semantic angle); wrist **position**: measured wrist path vs elbow angle (below) |
| 5 | wrist roll | +x | wrist point, rotated from the authored straight-arm rest to the semantic zero |

The elbow is a polycentric four-bar, so the wrist does not move on a circle. **Default elbow model (`elbow_model="table"`)**:
the wrist position at zero shoulder angles as a function of the semantic elbow angle (129 samples over the valid range).
It comes from the calibration's own measured forward model (`kinematics.serial_model.CalibrationFK`: the raw sweep
named in the calibration's provenance, interpolated four-bar tables), and the shoulder transform is applied on top.
Fallback (raw sweep missing): the best-fit pivot `arm_points_rest_root.elbow_center` (-0.017, 0.165, 1.321).

Accuracy of the wrist position against `CalibrationFK` over 500 random arm configurations per side:
- **table: 0.01 mm p50, 0.13 mm max** (`fk_vs_calibration_table.json`);
- pivot: 4.5 mm p50, 11-12 mm p95, 14 mm max, worst at full flexion (`fk_vs_calibration.log`).

Upper arm 0.301 m. Because pitch and roll act about the shoulder centre and wrist roll does not move the wrist, the
shoulder-to-wrist distance depends only on (yaw, elbow). The reachable wrist positions therefore form a thin shell:
0.287-0.405 m (left) and 0.283-0.405 m (right) (`ArmChain.shell`). The model reloads when the calibration file
changes (mtime + SHA-256). *Since 2026-09-24 the teleop loop does not call `reload_if_changed` itself:*
`dropbear_wbc.teleop.reload.BackgroundReloader` rebuilds `DropbearArmIK` + `ArmGravity` on a thread and the loop swaps
them in between two steps (the old in-loop reload stalled the loop 0.5 s; see the status table). The unit test was
written with the pivot model, whose
parameters all come from the JSON).

IK limits = calibration semantic valid ranges intersected with teleop soft limits (left arm; right mirrored):
pitch [-3.0, 1.0], roll [-0.175 (motor limit), 2.6], yaw [-2.6, 2.6], elbow [-0.572, 1.544] (four-bar range),
wrist roll [-2.6, 2.6] rad. The semantic solution goes through `SemanticMap.semantic_to_motor` to the 10 arm motors
(`to_motor_fast` evaluates only the arm entries of the same map and is tested bit-identical).

### 4.1 Solver and why position is strictly first

`solve_chain` (per arm, per step): damped least squares on the wrist position (singularity-robust damping near the
straight arm), then, in the SVD null space of the position task (2-D for 5 joints),
`0.5 |e_rot|² + 0.02 |q - q_stand|² + 0.1 |q - q_prev|²`. Box limits via an active set, backtracking with
lexicographic acceptance, 5 iterations per control step (warm start; the rest converges over steps). When the wrist
error exceeds 2 mm for a target inside the reachable shell, the solver retries from closed-form seeds (elbow from the
shoulder-target distance, shoulder from the minimal rotation, both YXZ branches) and switches branch only if that at
least halves the error; a failed restart is not retried until the target moves by 1 cm.

## 5. Measured results

### 5.1 IK (CPU, `ik_benchmark.json`, `pytest_teleop_venv_teleop.log`)

300 random configurations per arm inside the limits -> FK pose (reachable, orientation-consistent) -> IK:

| Solver | warm start (0.1 rad away): max error | cold start (standing pose): max error / fraction > 5 mm |
|---|---|---|
| priority (default) | < 0.4 mm (left 0.22, right 0.39) | 2.1 mm / 0 % |
| weighted (xr_teleoperate R1_A5 single cost) | 8.0 mm (3-5 % > 5 mm) | 32.3 mm / ~50 % |

The weighted cost trades wrist position for orientation / posture at the 100:1 weight ratio: with a 5-DoF arm most
wrist orientations are unreachable, and the residual moves the wrist. In closed loop (CPU, same figure-8) the
weighted mode tracked at 15.1 mm mean vs 3.7 mm (`teleop_cpu_fig8_pd0.5_weighted.json` vs `teleop_cpu_fig8_pd0.5.json`)
and its loop compute was 20.9 ms p50 vs 6.9 ms.

Unit tests (32 in `.venv-teleop`, numpy 1.26; 29 + 3 Vuer skips in `.venv-newton` (numpy 2.5) and the system Python;
`pytest_teleop_{venv_teleop,venv_newton,systempython}.log`):
reachability (cold multi-start >= 97 %, measured 100 %), joint limits (80 random arbitrary targets: semantic and motor
limits respected), IK error on FK targets < 5 mm (max 0.4 mm warm), left/right mirror symmetry (exact to 1e-6 rad for a
mirrored chain; calibrated arms within 6 mm / median 0.02 rad), continuity (10 s figure-8 at 100 Hz: max 0.03 rad per
step, error < 1 mm), unreachable targets (arm points at the target), straight-arm trap regression (section 5.4),
calibration reload, semantic <-> motor round trip, measured elbow table vs pivot. Whole suite with the final code:
193 passed, 9 skipped, 1 failed. The failure is `test_export_parity` on the gpu_pipeline's wave_right export, which
was being written during the run; it is not teleop code (`pytest_all_tests_systempython.log`).

Loop costs (CPU, `ik_benchmark.json`, final code, while an Isaac job shared the CPU): IK both arms 3.8 ms p50
(figure-8, 5 iterations per step, residual < 0.001 mm, max 0.019 rad per step); `semantic_to_motor` 4.4 ms,
`to_motor_fast` 1.05 ms; gravity feed-forward 0.6 ms.

### 5.2 Closed loop on the Newton bridge (GPU, hanging robot, 25 s scripted figure-8)

Setup (`verify_gpu_fig8v3_*`): `scripts/verify_teleop_gpu_set.py` under `tools/gpu_lock_run.py` (lock 11:19:39-11:21:31,
`gpu_verify_set_v3.wrapper.log`); MuJoCo Warp on cuda:0, `--fixed-base --realtime`, 500 Hz; spherical ankle tie rods
(the current plant default, irrelevant for the arms). Targets: both wrists on a 3-D Lissajous (+-2 / 6 / 5 cm,
6 s period) around the "forward" pose (upper arm 40 deg forward, forearm forward), always 0.296-0.389 m from the
shoulder. Stats over TELEOP minus the first 2.5 s (1126 frames at 50 Hz). Wrist errors are in the torso frame.

| Metric | contract plant (passive damping 50) | diagnostic plant (passive damping 0.5) |
|---|---|---|
| **Wrist, FK of measured joints vs target** mean / RMS / p95 / max | 31.0 / 42.3 / 84.2 / 96.0 mm (left) | **1.8 / 1.9 / 2.9 / 3.3 mm** (left), 1.8 / 2.0 / 3.2 / 3.6 (right) |
| Wrist, simulated hand body vs target (ground truth) | 50.7 mm mean | 8.2 mm mean, 14.0 mm max |
| IK residual (commanded FK vs target) | 0.0 mm | 0.0 mm |
| Elbow semantic tracking RMS | 0.41 rad | 0.013 rad |
| Best-fit lag, measured wrist vs target | 0.06 s | 0.02 s |
| Loop compute (LowState in -> LowCmd out) p50 / p95 / max | 6.2 / 9.3 / 11.4 ms | 5.4 / 7.7 / 10.2 ms |
| State age (bridge publish -> client) p50 | 0.55 ms | 0.49 ms |
| LowCmd transport p50 / p99 | 1.28 / 2.44 ms | 1.37 / 2.47 ms |
| Command applied at bridge tick - tick it was computed from: p50 / max | 3 / 6 ticks (6 / 12 ms) | 3 / 5 ticks |
| Bridge rate, RTF, physics step p50 | 500 Hz, 1.00, 0.75 ms | 500 Hz, 1.00, 0.75 ms |

These GPU runs used the best-fit pivot elbow model (v3). CPU (MuJoCo C) gives the same numbers within 0.3 mm
(`verify_cpu_fig8v3_*`: 2.1 / 2.2 / 3.4 / 4.0 mm and 31.1 mm mean). The first GPU set (11:00, `verify_gpu_fig8_*`)
used a figure-8 that leaves the reachable shell by up to 2.4 cm twice per 12 s ("forward_extended" reference,
+-4 / 6 / 5 cm): 3.4 mm mean / 5.1 mm RMS (diagnostic) and 34.1 mm mean (contract).

**Final code (v4: measured elbow table, restart gating), CPU bridge, same figure-8**
(`verify_cpu_fig8v4_{pd0.5,contract}_summary.json`):

| Metric | contract plant | diagnostic plant |
|---|---|---|
| Wrist, FK of measured joints vs target, mean / RMS / p95 / max | 30.5 / 41.0 / 82.8 / 93.6 mm (left) | 2.3 / 2.5 / 4.0 / 5.0 mm (left), 2.4 / 2.6 / 4.2 / 5.2 (right) |
| **Wrist, simulated hand body vs target** | 50.8 mm mean | **5.3 / 6.2 / 10.7 / 13.7 mm** (left), 5.9 / 6.7 / 11.6 / 15.0 (right) |
| Model error, simulated wrist vs FK | 21.4 mm mean (transients) | 4.9 mm mean, 9.8 mm max (was 7.9 / 13.1 with the pivot) |
| Loop compute p50 / lag | 8.8 ms / 0.06 s | 8.5 ms / 0.02 s |

A GPU run of v4 was queued at 11:39 (`gpu_verify_set_v4.wrapper.log`, tags `gpu_fig8v4_*`); see
logs/teleop/PROGRESS.md for whether it has run.

End-to-end latency budget (diagnostic plant, GPU): target sampled at the step start -> LowCmd out 5.3 ms p50 ->
transport 1.4 ms -> applied on the next physics tick(s) (3 ticks = 6 ms after the state it used) -> servo response. The
measured wrist lags the target by 0.02 s (best-fit shift). At 50 Hz the targets are also zero-order held for 20 ms.
XR adds the headset -> Vuer transport (about 1 ms locally with the simulated client, `counts.last_event_delay_s`; a
real headset over Wi-Fi is UNVERIFIED).

### 5.3 What limits accuracy

1. **Passive damping (contract plant)**. The Newton bridge applies the legacy `parasitic_*` damping of 50 N·m·s/rad to
   every passive DOF. The elbow four-bar has five passive joints (`*_Revolute32/41/42/44/123`), and its ~4.5:1 gear
   ratio reflects that damping to the elbow motor as the order of 10³ N·m·s/rad. The elbow motor saturates at its
   40 N·m limit and reaches only about half of a 0.3 rad step in 2 s (`probe_elbow_pd50_analysis.log`,
   `probe_elbow_contract_final_analysis.log`: 40 N·m at poses 2-5, wrist up to 9 cm off after 2 s holds). With 0.5 the
   same motor settles to 0.004 rad (`probe_elbow_pd0.5_analysis.log`). Isaac apparently ignores this damping: the
   calibration's Isaac step responses are identical for 50 and 0.5 (`logs/calibrate_settle/probe_step_response.log`).
   This is an **Isaac-vs-Newton plant parity problem** that also concerns the knee four-bars (sim2sim of any policy).
   It is reported here, not fixed: the plant owner decides.
2. **Model error**, simulated wrist vs FK of the measured joints (diagnostic plant): 4.9 mm mean with the measured
   elbow table (v4), 7.9 mm with the best-fit pivot (v3). Two sources:
   - about 0.05 rad between the Isaac-calibrated elbow table (gravity-free sweep) and Newton's forearm angle under
     load. It was measured from the hand-plate orientation (`analyze_teleop_probe.py`, "actual" column), is about
     5 mm at the wrist, and remains;
   - the best-fit pivot's own error (2-13 mm vs the calibration's measured FK), which the table model removes.

   The FK-based metric therefore understates the true error by that much. The recording keeps both
   (`fk.*` and `sim.*` columns).
3. **Gains**. The four-bar makes the forearm about 20x softer than the elbow motor gain (semantic stiffness ≈ kp /
   ratio²), hence elbow motor kp 600. Ablations on the diagnostic CPU plant (older "forward_extended" figure-8, baseline
   3.7 mm mean / 5.3 RMS, lag 0.04 s, `teleop_cpu_fig8_pd0.5.json`):

   | change | mean / RMS | lag | log |
   |---|---|---|---|
   | xr_teleoperate-like kp 150 elbow/shoulder, kd 3, dq* = 0 | 11.0 / 15.4 mm | 0.10 s | `data/teleop/cpu_fig8_pd0.5/meta/teleop_session.json` |
   | `--no-dq-ff` | 5.0 / 6.1 mm | 0.06 s | `teleop_cpu_fig8_pd0.5_nodqff.json` |
   | `--gravity-ff off` | 12.9 / 13.8 mm | 0.04 s | `teleop_cpu_fig8_pd0.5_nograv.json` |
   | `--rate 100` | 3.3 / 5.3 mm | 0.02 s | `teleop_cpu_fig8_pd0.5_rate100.json` (compute 9.7 ms p50: the 10 ms budget is tight) |
   | `--ik-mode weighted` | 15.1 / 17.9 mm | 0.02 s | `teleop_cpu_fig8_pd0.5_weighted.json` |

### 5.4 Issues found and fixed during verification

- **Straight-arm IK trap**. Moving from a nearly straight arm (elbow 1.4) to elbow 1.0, the warm-started solver rode
  onto the straight-elbow limit, where d|wrist - shoulder| / d elbow ≈ 0. It stayed 3.9 mm off for about 3 s under a
  10 mm restart threshold (`probe_elbow_contract_final_analysis.log`, pose 1). Fix: 2 mm threshold, restart gating,
  halve-and-gain branch acceptance. Replay of the recorded targets: 1.9 mm max; live CPU re-run of the probe with the
  final code: IK residual max 2.0 mm and the elbow follows the requested 1.0 (`probe_elbow_final_pd0.5_analysis.log`).
  Regression test `test_straight_arm_trap_escaped_by_restarts`. It runs on the pivot model: with the measured elbow
  table the trap does not occur on this path, because the real wrist path does not peak at the elbow limit.
- **Vuer in-process** made the loop compute 22 ms p50 in XR mode (GIL contention). Now in its own process, like
  televuer: about 10 ms (`teleop_cpu_xr2_*` vs `teleop_cpu_xr3/4_*`).
- **Operator events before TELEOP were dropped** (the simulated operator pressed calibrate / start during MOVE_IN:
  `teleop_cpu_xr_controllers_pd0.5.json`, robot never tracked). Now handled in every phase.

## 6. Devices and how to run

Interpreters: the teleop client and the simulated headset use `.venv-teleop`, created by
`tools/setup_venv_teleop.sh`. It holds Python 3.12, vuer 0.0.60 (televuer's pin), params-proto < 3 (3.x breaks
vuer 0.0.60), numpy 1.26, pyzmq, msgpack, pyarrow and pytest. The bridge uses `.venv-newton` (docs/SDK.md).

```bash
# 0) bridge (robot side), hanging, real time. GPU: it takes .locks/gpu.lock itself (or --gpu-lock-held under
#    tools/gpu_lock_run.py). CPU alternative without the GPU lock: add --device cpu --mujoco-cpu --no-gpu-lock and set
#    CUDA_VISIBLE_DEVICES=-1. The bridge's passive damping now defaults to 0, the value Isaac actually simulates
#    (CONTRACTS 0.3, 2026-09-24); the logs' "contract" runs used the old default 50 and "pd0.5" runs used 0.5.
.venv-newton/Scripts/python.exe tools/newton_bridge.py --fixed-base --realtime --duration 0 \
    --sim-bodies LH_shoulder_ex_al_interface_1,RH_shoulder_ex_al_interface_1

# 1a) scripted (no operator): 25 s figure-8, recorded
.venv-teleop/Scripts/python.exe tools/teleop_arm.py --device scripted --duration 25 \
    --record data/teleop/my_session --summary logs/teleop/my_session.json
# 1b) keyboard (interactive Windows console, msvcrt): r = start, w/s a/d q/e = left hand +-x/+-y/+-z (1 cm per key,
#     hold to repeat), i/k j/l u/o = right hand, [ ] and ; ' = wrist roll, h = home, x or Esc = stop
.venv-teleop/Scripts/python.exe tools/teleop_arm.py --device keyboard --clock wall --duration 0
# 1c) WebXR headset (see 6.1): hand tracking, or add --xr-controllers
.venv-teleop/Scripts/python.exe tools/teleop_arm.py --device webxr --clock wall --duration 0 \
    --xr-host 127.0.0.1 --xr-port 8012

# End-to-end harness (bridge + client [+ simulated headset / scripted keys]), merged summary in logs/teleop/:
.venv-teleop/Scripts/python.exe scripts/verify_teleop.py --backend cpu --tag my_test --duration 25
.venv-teleop/Scripts/python.exe scripts/verify_teleop.py --backend cpu --tag my_xr --duration 20 --xr-sim controllers
.venv-teleop/Scripts/python.exe scripts/verify_teleop.py --backend cpu --tag my_kb --duration 20 --kb-sim
# GPU under the team lock (detached WMI launch shown in scripts/verify_teleop_gpu_set.py):
python tools/gpu_lock_run.py --owner teleop --log logs/teleop/x.wrapper.log --timeout 900 -- \
    .venv-teleop/Scripts/python.exe -u scripts/verify_teleop_gpu_set.py
```

Key options of `tools/teleop_arm.py`: `--rate` (50), `--clock sim|wall`, `--ref-pose` / `--amplitude` / `--period`
(scripted), `--scripted-kind poses --probe-set elbow|shoulder` (probe sequences, analysed by
`tools/analyze_teleop_probe.py`), `--kp-*` / `--kd-*`, `--gravity-ff off`, `--no-dq-ff`, `--ik-mode weighted`,
`--max-joint-vel`, `--xr-scale`, `--xr-reference head_yaw|head_position`, `--auto-start`.

### 6.1 Connecting a real headset (UNVERIFIED: no device was available)

WebXR only runs in a **secure context**: `https://` or `http://localhost`. The Vuer server serves both the web client
(the vuer package ships its build; no internet needed) and the websocket on one port. Televuer's documented flow,
adapted:

1. **Certificates** (on the PC, into a directory outside the repo):
   - Quest 3 / Pico 4: a self-signed certificate is enough (the browser warns once):
     `openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem`
   - Apple Vision Pro: Safari needs a trusted CA. Create a root CA, sign a server certificate whose
     `subjectAltName` contains the PC's LAN IP, AirDrop `rootCA.pem` to the device, install and trust it
     (Settings > General > About > Certificate Trust Settings). Enable WebXR in Safari's feature flags (visionOS).
     Commands: `$DROPBEAR_UPSTREAM/televuer/README.md` section 2.2.
2. **Bind to the LAN explicitly** (the default is 127.0.0.1):
   `--xr-host <PC LAN IP> --xr-cert cert.pem --xr-key key.pem`.
3. **Firewall**: allow inbound TCP 8012 on the **Private** network profile only. Headset and PC must be on the same
   trusted Wi-Fi.
4. In the headset browser open `https://<PC LAN IP>:8012/?ws=wss://<PC LAN IP>:8012`, accept the certificate, and
   press "Enter VR" / "pass-through". The teleop log prints the URL.
5. **Quest over USB without certificates** (not tested): `adb reverse tcp:8012 tcp:8012`, then open
   `http://localhost:8012` in the Quest browser. localhost is a secure context.

Do **not** use ngrok or other tunnels, and do not port-forward. They would put the robot's control endpoint on the
public internet (this repo never starts one).

Operator procedure: stand with the arms relaxed at your sides (the robot's arms hang too). Press **left X** to
calibrate: this maps your current wrists onto the robot's current wrists. Press **right A** to start or pause
following. Press **both thumbsticks** to stop (arms go home, then damping). With hand tracking (no buttons), use
the terminal keys `c` (calibrate), `r` (start), `x` (stop) of the teleop client, or `--auto-start`. Scale (`--xr-scale`, default 0.70 ≈
Dropbear's 0.41 m reach / a 0.58 m human reach) maps human motion onto Dropbear's much smaller workspace.

## 7. Recording format (`source/dropbear_wbc/teleop/recorder.py`)

`data/teleop/<session>/` (one episode per run):

- `meta/info.json`: LeRobot `codebase_version` v2.1, fps 50, features, data path template.
- `meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/stats.json` (mean/std/min/max/q01/q99 of state and action).
- `meta/modality.json`: a **GR00T draft**. `state` and `action` = `left_arm` [0:5] and `right_arm` [5:10], `video` {},
  and `annotation.human.task_description` from `task_index`. Companion registration draft (not executed):
  `data/teleop/gr00t/dropbear_arms_config_draft.py` (NEW_EMBODIMENT, RELATIVE joint actions, 16-step chunks, like the
  upstream SO100 example).
- `meta/teleop_session.json`: full provenance (calibration path + SHA-256, IK limits / weights / model residuals,
  gains, gravity model, XR mapping, analysis, loop timing).
- `data/chunk-000/episode_000000.parquet` (pyarrow) and `extras/episode_000000.npz` (same columns, float64).

Columns:
- `observation.state`: measured semantic arm angles (10, G1 names).
- `action`: commanded semantic arm angles (10).
- `timestamp` (= `frame_index / fps` since 2026-09-24; LeRobot v2.1 checks it to 1e-4 s) plus the LeRobot index
  columns. The real clock of each frame is `time.sim_s` / `time.wall_s`; `time.skipped_steps` counts control periods the
  loop missed before the frame. `meta/timing.json` (from `recorder.check_timing`) reports the LeRobot check, the
  sim-clock drift and the skipped steps; `meta/episodes_stats.jsonl` holds the v2.1 per-episode statistics.
- Extras:
  - `observation.motor_{q,dq,tau}` (22) and `action.motor_q` / `action.tau_ff` (10 arm motors);
  - `action.ik_q` (IK solution before the velocity limit);
  - `target.*_wrist`, `fk.*_wrist`, `fk_cmd.*_wrist`, `sim.*_wrist` (xyz + quat wxyz, torso frame);
  - `ik.pos_err_m` / `ik.rot_err_rad`;
  - `time.*`, `teleop.tracking`;
  - `latency.{compute,state_age,device_age}_ms`.

Missing for a real GR00T N1.7 fine-tune:
- a **camera stream**: GR00T is a VLA; the Newton bridge renders nothing headless;
- **many episodes of real tasks**: these are scripted motions without objects or task success;
- **objects in the scene** and a hand or gripper.

## 8. What is missing for real XR use

- Testing with an actual Quest / Pico / Vision Pro:
  - HTTPS certificate generation and trust;
  - LAN binding and the firewall rule;
  - Wi-Fi latency and jitter;
  - the Windows timer resolution of the XR data path (the simulated client only reached ~32 Hz of its 60 Hz
    `asyncio.sleep` rate).
- A robot camera stream to the headset (teleimager-style ZMQ / WebRTC). Without it only pass-through / third-person
  operation is possible.
- Operator UX:
  - scale calibration from measured arm length;
  - optional orientation calibration (`OperatorMapping.calibrate(orientation=True)` exists but is not wired to a
    button);
  - a dead-man switch (for example "hold squeeze to move");
  - feedback when targets leave Dropbear's thin 0.27-0.41 m shell.
- Safety:
  - no self-collision or arm-torso collision checks (the IK only has joint limits);
  - velocity limit only (4 rad/s semantic);
  - the watchdog and damping on exit come from the SDK.
- Hands: none on Dropbear, so pinch and trigger are unused (xr_teleoperate's hand retargeting has no target).

## 9. What is missing for whole-body teleop

This stack is fixed-base. Standing or walking while teleoperating needs a controller that keeps balance and follows
upper-body targets, as in GR00T-WholeBodyControl / SONIC or unitree_rl_lab's mimic policies:

1. **A trained tracking or WBC policy.** None exists yet: tracking of real motions is unverified
   (logs/robot_task, logs/gpu_pipeline).
2. **An interface feeding the IK's arm solution into that policy's reference.** The semantic arm angles map to motor
   columns through the same `SemanticMap`; the motion-NPZ / policy reference uses exactly those columns. It would come
   with a lower-body command (stand / velocity).
3. **Plant parity.** The passive-damping issue (5.3) must be resolved before any Isaac-trained policy is expected to
   transfer to the Newton bridge.

## 10. Files

| Path | Role |
|---|---|
| `source/dropbear_wbc/teleop/arm_ik.py` | Chain from the calibration, priority / weighted IK, seeds, reload, motor mapping |
| `source/dropbear_wbc/teleop/gravity.py` | Gravity feed-forward |
| `source/dropbear_wbc/teleop/frames.py` | XR conventions (televuer), `OperatorMapping` |
| `source/dropbear_wbc/teleop/devices.py` | `ScriptedSource`, `KeyboardSource`, `VuerXRSource` (Vuer server process) |
| `source/dropbear_wbc/teleop/recorder.py` | LeRobot-v2.1-like session writer |
| `tools/teleop_arm.py` | The teleop client (loop, phases, recording, analysis) |
| `tools/extract_arm_inertia.py` -> `data/teleop/arm_inertia.json` | Arm inertials from the Newton model (CPU) |
| `tools/sim_xr_client.py` | Simulated WebXR headset (Vuer websocket protocol) |
| `tools/analyze_teleop_probe.py`, `tools/bench_teleop_ik.py`, `tools/check_teleop_fk_vs_calibration.py` | Probe analysis, IK benchmark, IK model vs the calibration's measured FK |
| `scripts/verify_teleop.py`, `scripts/verify_teleop_gpu_set.py` | End-to-end harness, GPU set |
| `tests/test_teleop_{arm_ik,devices,gravity,loopback}.py` | Tests |
| `data/teleop/<date>_<tag>/` | Recorded sessions; `data/teleop/gr00t/` GR00T config draft |
| `third_party/televuer_NOTICE.md` | License notice (televuer MIT; xr_teleoperate Apache-2.0 as design reference) |
