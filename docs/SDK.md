# Dropbear low-level SDK (`dropbear_hg-v1`), Newton bridge and policy runner

This is the Dropbear equivalent of Unitree's `unitree_sdk2` + `unitree_mujoco` bridge + `unitree_rl_lab/deploy`
stack, in **simulation only**. A controller written against Unitree's low-level API (`rt/lowcmd` / `rt/lowstate`,
`MotorCmd{mode,q,dq,tau,kp,kd}`) runs against Dropbear in Newton with an import change and a motor table change.

Binding definitions are in `docs/CONTRACTS.md`: section 1 (motor contract), section 6 (SDK), section 6.1 (wire
details and policy sidecar). This document explains them, lists the tool options and records what has been measured.
Every number below comes from a run whose log is cited. Anything without a log is marked **UNVERIFIED**.

```
                    rt/lowstate  (PUB bind, tcp://127.0.0.1:5556, 500 Hz)
  +-------------------------+  ------------------------------------------>  +-----------------------------+
  | tools/newton_bridge.py  |                                              | tools/policy_runner.py      |
  | ("robot side")          |  <------------------------------------------ | (client; Unitree-deploy FSM)|
  | Newton 1.6 + MuJoCo     |    rt/lowcmd   (SUB bind, tcp://127.0.0.1:5555,| or any dropbear_hg client,  |
  | Warp 3.12 (GPU) or      |                command rate = client's)       | e.g. scripts/examples/      |
  | MuJoCo C (CPU)          |                                              |   dropbear_low_level_example|
  +-------------------------+                                              +-----------------------------+
   tau = tau_ff + kp(q*-q) + kd(dq*-dq), clipped to effort limit, per physics step, 22 motors
```

## 1. Status at a glance

| Item | Status | Evidence |
|---|---|---|
| Message types, CRC (Unitree `crc32_core`), msgpack wire, ZMQ transport | VERIFIED (unit tests) | `logs/sdk_bridge/pytest_cont_venv.log`, `pytest_cont_systempython.log` |
| Newton bridge, fixed base (hanging), GPU, real time 500 Hz | VERIFIED | `verify_fixed_rt_summary.json` |
| Newton bridge, free base (on the ground), GPU and CPU | VERIFIED to run; **the robot falls** (no balance policy exists) | `verify_free_rt_summary.json`, `verify_free_cpu_summary.json` |
| Unthrottled GPU throughput, lockstep (deterministic) mode, GPU and CPU | VERIFIED | `verify_fixed_gpu_unthrottled_summary.json`, `verify_fixed_gpu_lockstep_summary.json`, `verify_fixed_cpu_lockstep_summary.json` |
| Runner policy path (ONNX + embedded reference + privileged terms) | VERIFIED with a synthetic policy and with an emulated tracking export (**untrained**) | `verify_fixed_rt_synthpolicy_summary.json`, `verify_fixed_cpu_exportemu_lockstep_summary.json` |
| Runner consumes `scripts/play.py --export` output | VERIFIED against the real `export.py` code run on CPU with Isaac stubs (section 7.4). Trained exports have existed since 2026-09-24 (docs/DEMOS.md section 3). | `export_emulated.log`, `export_emulated_check_parity.log`, `tests/test_export_sidecar_runner.py`, `export_emulated_negative_controls.log` |
| Unitree G1 low-level example ported | VERIFIED (CPU) | `bridge_unitree_example_cpu.log`, `unitree_example_cpu_summary.json` |
| A trained Dropbear policy tracking a motion in Newton | VERIFIED for the wave_right policy on CPU MuJoCo C with the sim2sim fidelity settings (0.5 ms physics step + stiff closure equalities), section 8.6. With the bridge defaults (2 ms, soft closures) the policy **falls after 1.9 s**. | `logs/gpu_pipeline/sim2sim/wave_cpu_*_report.json` |
| `--clock wall` runner mode, `--viewer gl`, `deploy.yaml` end-to-end with an ONNX | **UNVERIFIED** (the `deploy.yaml` parser is unit-tested) | `tests/test_deploy.py` |
| CycloneDDS / `unitree_hg` IDL wire compatibility, ESP32 gateway, hardware | Not implemented (out of scope) | none |

All log paths below are relative to `logs/sdk_bridge/` unless they start with another directory.

## 2. Quick start

Interpreters:
- `.venv-newton/Scripts/python.exe` has Newton 1.6.0, warp 1.17.0, mujoco / mujoco_warp 3.12.0, pyzmq 27.2, msgpack 1.2 and onnxruntime 1.30. It has no torch.
- The system Python 3.12 (`python`) has torch 2.5.1, onnx and onnxruntime. It has no warp.

```bash
# 1) Bridge (robot side), hanging bench mode, real time. It takes .locks/gpu.lock itself.
.venv-newton/Scripts/python.exe tools/newton_bridge.py --fixed-base --realtime --duration 30 \
    --report logs/sdk_bridge/my_bridge.json --trace logs/sdk_bridge/my_bridge_trace.npz
#    Under the team wrapper, pass --gpu-lock-held so the bridge does not wait on its own caller:
python tools/gpu_lock_run.py --owner me --log logs/sdk_bridge/my.log --timeout 600 -- \
    .venv-newton/Scripts/python.exe -u tools/newton_bridge.py \
    --gpu-lock-held --fixed-base --realtime --duration 30
#    (use absolute interpreter paths under gpu_lock_run.py: Windows CreateProcess does not resolve
#     relative paths against its cwd; see rejected_relpath_*.log)
#    CPU only, no GPU lock needed:
CUDA_VISIBLE_DEVICES=-1 .venv-newton/Scripts/python.exe tools/newton_bridge.py --device cpu --mujoco-cpu --no-gpu-lock ...

# 2a) Hold the default pose (no policy)
.venv-newton/Scripts/python.exe tools/policy_runner.py --mode hold --duration 10
# 2b) A tracking policy exported by scripts/play.py --export
.venv-newton/Scripts/python.exe tools/policy_runner.py --mode policy \
    --sidecar logs/rsl_rl/dropbear_tracking/<run>/exported/policy.json --allow-privileged --hold-s 1 --duration 20
# 2c) The Unitree G1 low-level example, ported
.venv-newton/Scripts/python.exe scripts/examples/dropbear_low_level_example.py --duration 8

# End-to-end verification harness (bridge + runner as two processes, merged summary)
.venv-newton/Scripts/python.exe scripts/verify_sdk_bridge.py --case fixed --hold-s 5 --realtime
```

## 3. `dropbear_hg-v1` vs Unitree `unitree_hg`, field by field

Unitree reference: `$DROPBEAR_UPSTREAM/unitree_sdk2_python/unitree_sdk2py/idl/unitree_hg/msg/dds_/*.py`
(CycloneDDS IDL for G1/H1-2). Dropbear implementation: `source/dropbear_wbc/sdk/types.py`, where the
docstrings are authoritative.

### 3.1 Envelope and transport

| Aspect | `unitree_hg` (G1) | `dropbear_hg-v1` |
|---|---|---|
| Transport | CycloneDDS topics `rt/lowcmd`, `rt/lowstate` on a NIC (`ChannelFactoryInitialize(0, "eth0")`) | ZMQ PUB/SUB over TCP. `rt/lowstate` on `tcp://127.0.0.1:5556` (robot PUB binds). `rt/lowcmd` on `tcp://127.0.0.1:5555` (robot SUB binds). Clients connect. |
| Encoding | IDL struct (CDR) | One ZMQ frame = `b"rt/lowstate\0"` or `b"rt/lowcmd\0"` + msgpack map. Arrays are raw little-endian bytes (float32 values, uint8 `mode`). |
| Envelope keys | none (fixed struct) | `schema` (`"dropbear_hg-v1"`), `type` (`"LowCmd"` / `"LowState"`), `tick`, `stamp_ns`, blocks, `crc` |
| CRC | `crc32_core` over the packed struct read as uint32 words (poly 0x04C11DB7, init 0xFFFFFFFF, MSB first, no reflection, no final XOR) | Same algorithm (`sdk/crc.py`; the table version is tested equal to the bitwise loop). It runs over canonical bytes: `<I tick`, `<q stamp_ns`, then every array's LE bytes in field order. `Write()` fills it. Mismatches are dropped and counted (`crc_errors`), never raised. |
| Delivery | CycloneDDS with the channel's QoS (`unitree_sdk2py` passes `qos=None`, i.e. CycloneDDS defaults); the subscriber handler gets samples through an optional queue (`queueLen`) | Non-blocking send: a full queue drops and counts (`SNDHWM` 16, `RCVHWM` 64). Reads are **latest-message**: queued frames are drained, only the newest is decoded, and the rest are counted as `skipped`. |
| State rate | 500 Hz (G1 lowstate) | 500 Hz: one LowState per bridge tick, with default `--sim-dt 0.002 --substeps 1`. It is 50 Hz of tick-0 state while paused before the first command. |
| Python API | `unitree_sdk2py.core.channel.{ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber}` | `dropbear_wbc.sdk.transport.*` with the **same names and call pattern**. `ChannelFactoryInitialize(domain_id, network_interface, role=, host=, cmd_port=, state_port=)`; the first two arguments are accepted and ignored. `ChannelSubscriber.Init(handler, queue_len)` starts a handler thread (`queue_len` ignored), or call `Read(timeout)` to poll. |

### 3.2 `LowCmd`

| `unitree_hg` `LowCmd_` | type | `dropbear_hg-v1` `LowCmd` | type | Notes |
|---|---|---|---|---|
| `mode_pr` | uint8 | none | | Selects the G1 ankle/waist parallel-mechanism mode (PR = pitch/roll, AB = raw motors A/B). Dropbear always commands **motors**: the parallel ankle is two calf motors (slots 4/5 and 10/11). The pitch/roll view is software, via `dropbear_wbc.kinematics.semantic.SemanticMap` (`lut2d`). |
| `mode_machine` | uint8 | none | | Robot-type echo from the G1 LowState. There is no Dropbear equivalent. |
| (none) | | `tick` | uint32 | Echo of the LowState tick the command was computed from (0 if unknown). The bridge uses it to measure command latency in ticks, and lockstep mode (section 5.2) uses it. |
| (none) | | `stamp_ns` | int64 | Sender `perf_counter_ns()` (same-host clock). Measures transport time. |
| `motor_cmd[35]` of `MotorCmd_` | struct[35] | `motor` block, **22 slots** | 6 arrays of 22 | Motor-contract order (CONTRACTS section 1). Unitree-style access works: `cmd.motor_cmd[i].q = x`. |
| `MotorCmd_.mode` | uint8 | `motor.mode` | uint8 | 1 = enable (PD law), 0 = disable (zero torque, joint free). G1 uses 1/0 the same way. |
| `MotorCmd_.q, dq, tau, kp, kd` | float32 | `motor.q, dq, tau, kp, kd` | float32 | Same meaning. The law is `tau_out = tau + kp(q - q_meas) + kd(dq - dq_meas)`, clipped to the motor effort limit and applied every physics step (zero-order hold between commands). |
| `MotorCmd_.reserve` | uint32 | none | | |
| (none) | | `neck` block (6 slots) or `null` | 6 arrays of 6 | SDK slots 22–27: `head_LeadScrew1..6` (prismatic, m / N). `null` keeps the current neck targets. The neck is never part of a policy action. |
| `reserve[4]` | uint32[4] | none | | |
| `crc` | uint32 | `crc` | uint32 | See 3.1 |

### 3.3 `LowState`

| `unitree_hg` `LowState_` | `dropbear_hg-v1` `LowState` | Notes |
|---|---|---|
| `version[2]` | none | Replaced by `schema`. |
| `mode_pr`, `mode_machine` | none | See 3.2 |
| `tick` (uint32) | `tick` (uint32) | Bridge control tick at 500 Hz. It starts at 0 when the first command unpauses the bridge. |
| (none) | `stamp_ns` (int64) | Publish time (same-host clock). The runner reports `state_age_ms` from it. |
| `imu_state.quaternion[4]` (wxyz) | `imu.quat_wxyz` (wire key `quat_wxyz`) | Body to world, z up. The IMU body is the articulation root `world`. Alias `imu_state.quaternion`. |
| `imu_state.gyroscope[3]` | `imu.gyro` (`gyro`) | Body-frame angular velocity [rad/s]. Alias `gyroscope`. |
| `imu_state.accelerometer[3]` | `imu.accel` (wire key `acc`) | Body-frame specific force [m/s²]: `R^T (a_com − g)`, reading +9.81 on z when upright at rest. `a_com` is the finite difference of the root CoM velocity over one tick. No noise or bias. Alias `accelerometer`. |
| `imu_state.rpy[3]` | `imu.rpy` (`rpy`) | Intrinsic ZYX roll/pitch/yaw [rad]. |
| `imu_state.temperature` (int16) | none | |
| `motor_state[35]` of `MotorState_` | `motor` block, 22 slots | Motor-contract order. Alias `state.motor_state[i].q`. |
| `MotorState_.mode` | `motor.mode` (uint8) | Echo of the applied command mode. |
| `MotorState_.q, dq` | `motor.q, dq` (float32) | Measured joint position and velocity (rad, rad/s). |
| `MotorState_.ddq` | `motor.ddq` (float32) | Finite difference over one tick. |
| `MotorState_.tau_est` | `motor.tau_est` (float32) | Simulation: the torque actually applied after clipping. |
| `MotorState_.temperature` (int16[2]) | `motor.temperature` (**float32[1]** per slot) | Simulation: constant 25 °C. |
| `MotorState_.vol`, `sensor[2]`, `motorstate`, `reserve[4]` | none | |
| `wireless_remote[40]` | none | There is no joystick. The runner FSM is driven by a time schedule (section 6). |
| `reserve[4]`, `crc` | `crc` | |
| (none) | `neck` block (6) or `null` | Neck lead-screw states (m, m/s). |
| (none) | `sim` block or `null` | **Privileged, simulation only.** It holds `time_s`, `root_pos_w` (root link-frame origin), `root_quat_w`, `root_lin_vel_w` (root **CoM** velocity, Newton `body_qd` convention = Isaac Lab `root_lin_vel_w`), `root_ang_vel_w`, and optional `body_names` / `body_pos_w` / `body_quat_w` for bodies requested with `--sim-bodies`. Hardware never has it. `--no-sim-block` turns it off. |

### 3.4 Motor slots (CONTRACTS section 1) and default gains

The slot order is "legs first, then arms" like G1. The **slots are motors, not G1 joints**:
- The Dropbear hip motors are ordered roll-like, yaw-like, pitch.
- The knee motor is a four-bar crank.
- The ankle is a parallel pair.
- G1's waist (12–14) and third wrist DOFs do not exist.

Use `SemanticMap` for the G1-named semantic space (CONTRACTS section 2). The defaults are `motors.DEFAULT_KP/KD/POS`, which are the legacy `DROPBEAR_CFG` values. `tests/test_sdk_contract_consistency.py` asserts they are identical to the robot config.

| slot | motor | kp | kd | effort limit | default q |
|---:|---|---:|---:|---:|---:|
| 0 / 6 | `PG_left/right_leg_pitch` (hip roll-like) | 150 | 5 | 200 N·m | −0.1 |
| 1 / 7 | `PG_left/right_leg_roll` (hip yaw-like) | 150 | 5 | 200 | 0 |
| 2 / 8 | `LL/RL_hip_joint` (hip pitch) | 150 | 5 | 200 | 0 |
| 3 / 9 | `LL/RL_knee_actuator_joint` (knee crank) | 200 | 12 | 300 | 0.3 |
| 4 / 10 | `LL/RL_Revolute67` (calf motor A) | 80 | 4 | 80 | −0.2 |
| 5 / 11 | `LL/RL_Revolute81` (calf motor B) | 80 | 4 | 80 | 0 |
| 12–16 | `LH_yaw, LH_pitch, LH_roll, LH_elbow_joint, LH_wrist_roll` | 50 | 2 | 40 | 0, 0, 0, 0.3, 0 |
| 17–21 | `RH_yaw, RH_pitch, RH_roll, RH_elbow_joint, RH_wrist_roll` | 50 | 2 | 40 | 0, 0, 0, 0.3, 0 |
| 22–27 | `head_LeadScrew1..6` (neck block) | 5000 | 50 | 100 N | held at init |

## 4. Safety and timing semantics (robot side)

These mirror a Unitree robot and are the same for the future ESP32 gateway (CONTRACTS 6.1).

- **Paused until the first command.** The bridge publishes tick-0 state at 50 Hz and does not step physics until a LowCmd arrives. The limit is `--wait-timeout-s`, default 300 s. `--no-wait-for-cmd` starts immediately in damping.
- **Zero-order hold.** The last command is held and applied every physics step.
- **Watchdog.** If no LowCmd arrives for more than `--watchdog-ms` (100 ms, wall clock), all motors switch to damping: `kp = 0`, `kd = --damping-kd` or the legacy group kd, `tau = 0`, `dq* = 0`. The next command recovers. Events are logged in the report. Every verify run shows exactly one event, when the runner exits (e.g. `verify_fixed_rt_summary.json`: t = 7.202 s, 102 ms silence).
- **`mode = 0`** on a slot gives zero torque for that motor.
- **Effort clipping** per motor (table 3.4). The neck uses its own PD at the commanded or initial targets.
- **Runner exit.** The runner sends a damping LowCmd (`kp = 0`, `kd = passive_kd`) on exit, like Unitree's deploy `Passive`.

## 5. Newton bridge (`tools/newton_bridge.py`)

### 5.1 Plant

`source/dropbear_wbc/newton_sim/plant.py`, as reported in `bridge_fixed_rt.json` → `plant`:

- **Source.** Loaded from the contract USD (SHA-256 verified `45586414…`).
- **Fixes.** The CONTRACTS 0.1 fixes are applied in memory: 3 orphan bearings deactivated, joint friction 0.5/0.1 → 0, `LL_Revolute121` axis X → Z, zero inertia raised. `--raw-usd` skips them, for diagnosis only.
  - *Added 2026-09-24 (gpu_pipeline):* the CONTRACTS 0.2 default also retypes the ankle tie-rod closures `*_Revolute111/112` to spherical. SolverMuJoCo then uses 1 CONNECT per closure instead of 2 (47 → 43 equality constraints, `logs/gpu_pipeline/newton_ankle_probe.json`). `--authored-ankle` or `DROPBEAR_AUTHORED_ANKLE=1` keeps the authored revolute joints.
- **Model.** 90 bodies, 117 joints, 27 loop closures as equality constraints, 22 motor + 6 neck + 63 passive DOFs, mass 56.17 kg.
- **Solver.** Newton `SolverMuJoCo`: MuJoCo Warp on `cuda:0` (CUDA graph captured), or the MuJoCo C backend on CPU with `--mujoco-cpu`. 100 solver / 50 line-search iterations. Convex-hull mesh colliders (656 collision meshes).
- **Passive joints.** Damping 50 (legacy `parasitic_*` groups).
- **Timing.** `sim_dt` 0.002 s, 1 substep, so the control tick is 2 ms. Motor PD runs in a warp kernel every physics step.
- **Fixed base (`--fixed-base`).** The torso is welded to the world with the feet 0.30 m clear (`--hang-clearance`).
- **Free base.** The feet start 5 mm above the ground. The root `world` frame origin sits about 12.6 cm below the soles (root z = −0.126 m at start, `verify_free_rt_summary.json`), consistent with CONTRACTS 5.
- **Build time.** 18–47 s: USD import about 12 s, solver build about 3 s, graph capture about 1 s, and the rest is first-run kernel/USD cache.

### 5.2 Options

| Option | Default | Meaning |
|---|---|---|
| `--fixed-base`, `--hang-clearance M` | off, 0.30 | Hanging bench mode |
| `--realtime` | off | Throttle to the wall clock. If the sim is slower, it runs flat out and reports RTF. |
| `--lockstep-ticks N` | 0 | After every N-th tick, block until a LowCmd with `tick ≥` that tick arrives, up to `--lockstep-timeout-ms` (1000). This is deterministic sim2sim: use N = 10 for a 50 Hz policy at a 2 ms tick. The block is skipped once the watchdog has declared the client gone. |
| `--duration S`, `--max-wall-s S` | 0 = forever | Simulated / wall-clock limit |
| `--sim-dt`, `--substeps` | 0.002, 1 | Physics step; tick = `sim_dt × substeps` |
| `--device`, `--mujoco-cpu`, `--no-graph` | `cuda:0`, off, off | Backend selection |
| `--iterations`, `--ls-iterations` | 100, 50 | MuJoCo solver iterations |
| `--collisions convex-hull\|box` | convex-hull | Collision geometry |
| `--host`, `--cmd-port`, `--state-port` | 127.0.0.1, 5555, 5556 | Endpoints |
| `--watchdog-ms`, `--damping-kd` | 100, legacy kd | Safety (section 4) |
| `--no-wait-for-cmd`, `--wait-timeout-s` | off, 300 | Start-up behaviour |
| `--no-sim-block`, `--sim-bodies a,b` | off, none | Privileged `sim` block contents |
| `--trace PATH`, `--trace-every N`, `--report PATH`, `--status-every-s` | none, 5, none, 1.0 | Evidence: npz trace (q, dq, tau, q*, kp, kd, root pose, damping flag, cmd tick, and since 2026-09-24 the `--sim-bodies` poses as `body_names`/`body_pos_w`/`body_quat_wxyz`), JSON report |
| `--passive-damping X` | 0 (was 50 until 2026-09-24 13:00) | Passive-joint DOF damping. 0 = what Isaac actually simulates (CONTRACTS 0.3) |
| `--raw-usd` | off | Skip the CONTRACTS 0.1 in-memory fixes (diagnosis only) |
| `--authored-ankle` | off | Keep the authored revolute ankle tie rods (CONTRACTS 0.2 opt-out) |
| `--start-npz PATH`, `--start-frame N`, `--presettle-s S` | none, 0, 1.5 | Pre-settled start for sim2sim (added 2026-09-24). The 22 motors are ramped from the USD rest pose to that NPZ frame's motor pose over S/2 and held for S/2, at 4× legacy gains, with the root **pinned**: after every tick the root coordinates are rewritten and the root velocity is zeroed. Gravity and contacts are on. The simulator solves the closures itself. The result becomes the initial state: velocities zeroed, root re-placed 5 mm above the ground, tick 0. Reported under `plant.presettle`. |
| `--viewer gl`, `--render-hz` | null, 30 | Interactive viewer (**UNVERIFIED**) |
| `--no-gpu-lock` | off | Only with `--device cpu --mujoco-cpu` and `CUDA_VISIBLE_DEVICES=-1` (enforced) |
| `--gpu-lock-held` | off | The caller (`tools/gpu_lock_run.py`) holds `.locks/gpu.lock`. The bridge checks that it exists and never deletes it (tested in `tests/test_newton_bridge_units.py`). |

## 6. Policy runner (`tools/policy_runner.py`, `source/dropbear_wbc/deploy/`)

### 6.1 FSM

This follows unitree_rl_lab `deploy/include/FSM` (`State_Passive`, `State_FixStand`, `State_RLBase` / `State_Mimic`) and
unitree_rl_gym `deploy_real`. It runs once per control step (`step_dt`, 0.02 s). All 22 motors are commanded every step.

| State | q* | kp / kd | Enter / leave |
|---|---|---|---|
| `PASSIVE` | measured q | 0 / `passive_kd` (legacy kd) | Start. Also entered from POLICY on bad orientation. |
| `MOVE_TO_DEFAULT` | linear from the measured pose to `default_pose_sdk` over `move_to_default_s` (2 s) | hold gains (legacy kp/kd) | After `--passive-s`. Goes to HOLD when the ramp finishes. |
| `HOLD` | `default_pose_sdk` | hold gains | Unitree `FixStand` |
| `POLICY` | policy joints: `action_offset + action_scale × action`; other slots hold the default | policy joints: config kp/kd; others: hold gains | After `--hold-s` in policy mode. Goes to PASSIVE if torso tilt > `bad_orientation_rad` (1.0, unitree_rl_lab `bad_orientation`). Goes to HOLD when a reference motion ends. |

Entering POLICY does three things:
- resets the last action and the observation history;
- sets the reference time step to 0;
- **aligns** the reference to the robot (`align: yaw_xy` rotates the reference by the anchor yaw difference and translates its anchor xy onto the robot's; this is unitree_rl_lab `State_Mimic`'s `init_quat`).

Unitree switches states with the remote (`wireless_remote`). The runner uses a schedule: Passive for `--passive-s`, then
MoveToDefault, then Hold, then (policy mode) Policy after `--hold-s`.

### 6.2 Clocks, rates and logs

- `--clock sim` (default): one control step whenever the LowState tick has advanced by `step_dt / tick_dt` (10) ticks.
  The policy therefore runs at 50 Hz of **simulated** time whether the bridge is slower or faster than real time.
  While the tick does not advance, the runner re-sends the last command every 50 ms (keepalive). This covers the ZMQ
  slow-joiner start-up and slow sims.
- `--clock wall`: a fixed wall-clock period like the Unitree C++ deploy (**UNVERIFIED**, not exercised by any run).
- `--state-timeout-s` (5): abort if LowState stops.
- Outputs:
  - `--log` JSONL, one record per step: t, tick, fsm, tilt, root_z, state age, q, q*, tau_est, action.
  - `--summary` JSON:
    - loop rates and ticks per step;
    - state age and compute latency;
    - FSM transitions;
    - fall time (tilt > `--fall-tilt-rad` 1.0, or root z < `--fall-height-m`);
    - hold tracking error over the last `--settle-window-s`;
    - the privileged terms used.

### 6.3 Observation terms

These live in `deploy/observations.py` and are computed from LowState with numpy. Terms are concatenated in config
order with clip, then scale, then `history_length` stacking (oldest first), as in Isaac Lab.

| `func` | Definition | Hardware-observable |
|---|---|---|
| `base_ang_vel` | IMU gyro (root body frame) | yes |
| `projected_gravity` | `R^T (0,0,−1)` from the IMU quaternion | yes |
| `joint_pos_rel`, `joint_pos` | q − default (policy order), q | yes |
| `joint_vel_rel` (= `joint_vel`) | dq | yes |
| `last_action` (= `actions`) | previous raw action | yes |
| `velocity_commands`, `generated_commands` | `--velocity-cmd`, or the motion command | yes |
| `motion_command` | reference motor `joint_pos` (22) then `joint_vel` (22) | yes (reference) |
| `motion_joint_pos`, `motion_joint_vel` | halves of the above | yes |
| `motion_anchor_ori_b` | reference anchor orientation in the robot anchor frame, first two rotation-matrix columns row-major (6). The robot anchor is IMU quat ∘ `anchor_offset_quat`. | yes |
| `motion_anchor_pos_b` | reference anchor position in the robot anchor frame. Needs the root position from `LowState.sim`. | **no (sim only)** |
| `base_lin_vel` | root CoM linear velocity in the root link frame (Isaac Lab `root_lin_vel_b`), from `LowState.sim` | **no (sim only)** |

Privileged terms are refused unless `--allow-privileged` is given, and the summary lists them (`privileged_terms`).
The BeyondMimic tracking policy observes both sim-only terms (CONTRACTS 5.1), so it is **sim2sim only**. Hardware
needs a policy trained without them (BeyondMimic's "without state estimation" variant). The runner already handles
that layout (`tests/deploy_fixtures.OBS_TERMS`).

## 7. Deploy configuration formats

Both formats map onto one `DeployConfig` (`deploy/config.py`). Per-joint arrays are in **policy order** unless noted.

### 7.1 unitree_rl_lab `deploy.yaml` (`--deploy-yaml`)

Written upstream by `unitree_rl_lab/utils/export_deploy_cfg.py` and read by `unitree_rl_lab/deploy`.

| Key | Meaning |
|---|---|
| `joint_ids_map` | `joint_ids_map[i]` = SDK slot of policy joint i (Dropbear: 0–21). As in unitree_rl_lab. |
| `joint_names` (Dropbear extension) | Policy joint names (motor-contract names). Preferred over `joint_ids_map`. |
| `step_dt` | Policy period [s] |
| `stiffness`, `damping` | **SDK slot order** (22 values, 0 for slots outside the policy, or a scalar), exactly as unitree_rl_lab writes them (`stiffness[joint_ids_map] = …`; deploy C++ `joint_stiffness // sdk order`). *Fixed in this session*: the parser previously read them in policy order. A permuted `joint_ids_map` would then have put gains on the wrong motors. The test now uses slot-distinct gains (`tests/test_deploy.py::test_unitree_deploy_yaml_parse`). Negative control: `deploy_yaml_gain_order_negative_control.log`. |
| `default_joint_pos` | Policy order. It may also be `"legacy"` (DROPBEAR_CFG pose) or `{calibration: <semantic calibration json>}` (uses `standing_motor_pos`). |
| `actions.JointPositionAction.{scale, offset, clip}` | Policy order. `offset` defaults to `default_joint_pos`. |
| `observations: {name: {params, clip, scale, history_length[, func, dim]}}` | Ordered terms. `func` defaults to the key (unitree_rl_lab keys are function names). |
| `policy` (Dropbear extension) | ONNX path relative to the YAML |
| `motion` (Dropbear extension) | `{file\|npz\|motion_file, source: npz\|csv\|onnx, anchor_body(_name), fps, time_start, time_end, align, anchor_offset_pos, anchor_offset_quat}` |
| `fsm` (Dropbear extension) | `{move_to_default_s, passive_kd[22], hold_kp[22], hold_kd[22], bad_orientation_rad}` (SDK order) |

The Unitree exporter itself cannot be pointed at the Dropbear Isaac env. It resolves *all* 91 articulation joints
against the SDK names, and the 63 passive DOFs have no SDK slot. A Dropbear `deploy.yaml` is therefore hand-written or
generated from the 22 motors.

### 7.2 Policy sidecar `dropbear-policy-sidecar-v1` (`--sidecar`)

This is the format `scripts/play.py --export` writes (`tasks/tracking/export.py`) as `<run>/exported/policy.json`,
next to `policy_motion.onnx`, `policy.onnx`, `policy.pt` and `parity_samples.pt`. The parser is
`deploy/config.py:load_sidecar`. BeyondMimic ONNX metadata (`joint_names`, `default_joint_pos`, `action_scale`,
`observation_names`, `anchor_body_name`, …) fills any key the sidecar omits.

| Key | Written by `export.py` as | Runner use |
|---|---|---|
| `schema` | `"dropbear-policy-sidecar-v1"` | Checked (fails closed on mismatch) |
| `policy_onnx` | `"policy_motion.onnx"` | Loaded with onnxruntime (CPU EP, 1 thread) |
| `onnx_inputs` | `{"obs": "obs", "time_step": "time_step"}` | `obs` [1,125] float32; `time_step` [1,1] float32 = reference frame index |
| `step_dt` | `sim.dt × decimation` = 0.02 | Control period (10 ticks) |
| `joint_names` | 22 motor-contract names (the action term's joints) | Policy order → SDK slots |
| `default_joint_pos` | env nominal default of the 22 motors | Default pose (MoveToDefault/Hold) and `joint_pos_rel` |
| `joint_stiffness`, `joint_damping` | env motor kp/kd (**policy order**, unlike `deploy.yaml`) | LowCmd kp/kd in POLICY |
| `action_scale` | 22 values (`0.25 × effort / kp`) | `q* = offset + scale × a` |
| `action_offset` | = `default_joint_pos` | as above |
| `action_clip` | `null` | none |
| `observations` | `[{name, func, dim, scale: 1, clip: null, history_length: 1, params: {}}]`, with `func` from `OBS_FUNC` (`command → motion_command`, `joint_pos → joint_pos_rel`, `actions → last_action`, …) | Observation builder; dims checked at run time |
| `obs_dim` | 125 | Checked against the ONNX input |
| `motion.source` | `"onnx"` (reference embedded in `policy_motion.onnx`) | `OnnxMotion` queries the `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w` outputs per frame |
| `motion.anchor_body_name` | `head_5mm_ujoint_base__5__1` | Anchor index in `motion.body_names` |
| `motion.anchor_offset_pos`, `anchor_offset_quat` | Anchor link pose in the root `world` link frame: (0.101, −0.045, 1.639) m and (0.5, 0.5, 0.5, 0.5) | Robot anchor = root pose ∘ offset. **Required** when `body_names` lacks `world`, as for the export's 14 tracked bodies. The runner fails closed otherwise. The contract was updated in CONTRACTS 6.1. |
| `motion.align` | `"yaw_xy"` | Alignment at POLICY entry |
| `motion.reference_joint_names` | the 22 motors (the embedded reference holds motor columns only) | Column selection |
| `motion.body_names`, `num_frames` | 14 tracked bodies, T | Anchor lookup; POLICY → HOLD when the clip ends |
| `task`, `usd_sha256`, `run_path` | provenance | Copied to the runner summary (`config.meta`) |
| `dropbear_tracking` | layout, normalizer statistics, timing, provenance hashes, parity | Ignored by the parser |

### 7.3 How a trained tracking export is run

```bash
# Isaac side (robot_task component)
C:/isaac-sim/python.bat -u scripts/play.py --task Dropbear-Tracking-Flat-Play-v0 --motion_file <clip>.npz \
    --load_run <run> --export --headless          # -> logs/rsl_rl/dropbear_tracking/<run>/exported/
# Newton side: bridge (free base, real time), then runner (sim2sim; the policy uses sim-only terms)
.venv-newton/Scripts/python.exe tools/newton_bridge.py --realtime --duration 30
.venv-newton/Scripts/python.exe tools/policy_runner.py --mode policy --allow-privileged --hold-s 1 --duration 25 \
    --sidecar logs/rsl_rl/dropbear_tracking/<run>/exported/policy.json --log ... --summary ...
```

On a free base the robot will not survive `MOVE_TO_DEFAULT` → `HOLD` under the legacy PD gains (section 8.3).
A tracking policy must therefore take over before the robot falls, or be started from the fixed base. Starting a
free-base policy from a settled reference pose (RSI-style) is **not implemented**. The bridge always starts from the
USD pose.

### 7.4 Export ↔ runner alignment check (done in this session)

(Written before the first trained export; trained exports exist since 2026-09-24, docs/DEMOS.md section 3.) At the time `robot_task` Goal 3 was still pending, and `tests/test_export_parity.py` skips with
"no tracking-policy export found". To check the real format anyway:

- **Emulated export.** `scripts/emulate_tracking_export.py` runs the **real** `tasks/tracking/export.py::export_tracking_policy` on CPU.
  - Stubbed: Isaac-only helpers (`isaaclab_rl.rsl_rl.export_policy_as_jit/onnx`, re-implemented with Isaac Lab 2.2's exporter semantics; `isaaclab.utils.math` quaternion functions).
  - Mocked env handles: built from the real settled NPZ `data/motions/smoke/dropbear_static_stand.npz`.
  - Real: the vendored rsl-rl 2.3.3 `ActorCritic` and `EmpiricalNormalization`, with random weights and synthetic statistics.
  - Output: `logs/sdk_bridge/export_emulated/` (with `EMULATED.json` saying what was mocked), log `export_emulated.log`.
- **robot_task's checker passes on it.** `tools/check_export_parity.py` reports `ok: true` (`export_emulated_check_parity.log`):
  - layout contiguous, 125 dims;
  - deploy `load_sidecar` ok;
  - ONNX vs TorchScript max abs diff 2.4e-7;
  - `policy_motion.onnx` outputs `joint_pos` [1,22] and `body_*` [1,14,·].
- **`tests/test_export_sidecar_runner.py`** (system Python; skips without torch) builds the runner from that
  `policy.json` via `policy_runner.build` and steps the controller with the robot placed exactly on NPZ frames. It asserts:
  - the embedded reference equals the NPZ motor columns and anchor pose;
  - the runner's rebuilt robot anchor (root pose ∘ sidecar offset) equals the NPZ anchor link pose (< 20 µm);
  - every observation slice matches the NPZ ground truth: command, anchor pos ≈ 0, anchor ori = identity columns, base velocities, joint terms, last action;
  - actions equal the plain `policy.onnx`;
  - the LowCmd equals `default + scale × action` with the sidecar gains;
  - default pose and action scale equal the values the env handed to the export;
  - the runner refuses the policy without `--allow-privileged`;
  - a sidecar without `anchor_offset_*` fails closed.
- **Negative controls** (`export_emulated_negative_controls.log`). Each of these mutations of `policy.json` is detected:
  - anchor offset quat set to identity;
  - anchor offset +1 cm x;
  - anchor offset +1 mm z;
  - two joint names swapped;
  - action_scale × 1.01;
  - default +0.01;
  - action_offset +0.01;
  - joint_pos/joint_vel terms swapped.
- **Full process path** (`verify_fixed_cpu_exportemu_lockstep_summary.json`): bridge (CPU, fixed base, lockstep 10) + runner with the emulated export.
  - Transitions: `hold->policy` at 3.1 s, then exactly **200 policy steps** (the 200-frame clip), then `policy->hold(motion_end)` at 7.08 s.
  - Actions finite, |a| ≤ 0.29 (random weights); policy step p50 0.73 ms.
  - Command latency 0 ticks; closure residual 1.2 mm.

**Result.** No format mismatch was found between `export.py` and the runner. Two issues were fixed (the export format
did not change):
- a latent silent fallback to an identity anchor offset (now fail-closed);
- the `deploy.yaml` gain order (7.1).

## 8. Measured results

### 8.1 Loop rates and latency (fixed base, hold mode, 22 motors, runner at 50 Hz)

Definitions:
- **Bridge Hz**: ticks per wall second over the whole run. The **active** column excludes start-up and the terminal
  lockstep timeout or watchdog after the runner exits (`throughput_active_window.log`).
- **Physics**: physics step plus readout.
- **Cmd latency**: `bridge tick at apply − LowCmd.tick`.
- **Transport**: runner send → bridge apply.
- **State age**: bridge publish → runner receive.
- **Compute**: runner step time.

| Run (summary JSON) | Backend | Mode | Bridge Hz (whole / active) | RTF | Physics ms p50 / p99 | Runner ticks/step | Cmd latency ticks p50 / p99 / max | Transport ms p50 / p99 | State age ms p50 / p99 | Compute ms p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| `verify_fixed_rt` | GPU MuJoCo Warp | real time | 500.0 / 501.5 | 1.00 | 0.70 / 1.84 | 10 / 10 | 0 / 1 / 1 | 0.82 / 2.21 | 0.39 / 0.61 | 0.16 |
| `verify_fixed_gpu_unthrottled` | GPU | as fast as possible | 995.9 / **1097** | 1.99 / **2.19** | 0.45 / 3.63 | 10 / 10 (0 frames skipped) | 1 / 1 / 10 | 0.74 / 3.89 | 0.46 / 0.74 | 0.14 |
| `verify_fixed_gpu_lockstep` | GPU | lockstep 10 | 657.9 / **944** | 1.32 / **1.89** | 0.49 / 3.70 | 10 / 10 | **0 / 0** / 10 | 0.48 / 0.97 | 0.48 / 0.88 | 0.14 |
| `verify_fixed_cpu` | CPU MuJoCo C | real time | 500.0 / 500.0 | 1.00 | 0.83 / 1.18 | 10 / 10 | 0 / 1 / 1 | 0.54 / 2.47 | 0.57 / 0.88 | 0.21 |
| `verify_fixed_cpu_lockstep` | CPU | lockstep 10 | 602.7 / **823** | 1.21 / **1.65** | 0.75 / 1.10 | 10 / 10 | **0 / 0 / 0** | 0.38 / 0.65 | 0.47 / 0.68 | 0.19 |
| `verify_free_rt` | GPU | real time, free base, contacts | 500.0 | 1.00 | 1.07 / 1.51 | 10 / 10 | 0 / 1 / 1 | 0.59 / 2.34 | 0.42 / 0.70 | 0.17 |

Reading this table:
- **Real time holds** on both backends. The single-robot tick costs 0.7–1.1 ms of the 2 ms budget, and the runner
  never missed a control step (10 ticks/step in every run).
- **End-to-end reaction.** State publish → command applied is about state age + compute + transport ≈ 0.4 + 0.2 +
  0.8 ≈ 1.4 ms p50 in real time. So a command computed from the state of tick N drives the very next physics step
  (latency 0, p50) or the one after it (latency 1, p99).
- **Throughput.** Unthrottled GPU reaches about 1.1 kHz (RTF 2.2) with the runner in the loop. Lockstep costs
  about 15% but makes latency exactly 0 ticks.
- **Latency outliers.** The `max 10` (and `max 5` in `verify_free_cpu`) outliers are the runner's single keepalive
  re-send in those runs (`runner.keepalives = 1`; `bridge.commands.applied` = runner steps + exit command + 1). A
  keepalive repeats the last command with its original tick, so the bridge counts it as late. Runs without a
  keepalive peak at 0–1 ticks.
- **Wait out the lock.** Lockstep and unthrottled GPU runs took 35 s and 54 s of lock time including the plant build
  (`run_gpu_throughput.out`).

Runs with the policy path (ONNX inference inside the step):

| Run | Policy | Policy steps | Runner step (compute) ms p50 / p99 | Cmd latency ticks p50 / p99 | Result |
|---|---|---|---|---|---|
| `verify_fixed_rt_synthpolicy` (GPU, real time) | synthetic linear 125 → 22 with embedded reference (`synthetic_policy/`) | 250 | 0.61 / 1.04 (ONNX 0.46) | 0 / 1 | `passive→move_to_default→hold→policy`, finite, \|a\| ≤ 0.007, privileged terms reported |
| `verify_fixed_cpu_synthpolicy` (CPU, real time) | same | 250 | 0.82 / 1.43 | 1 / 1.95 | same |
| `verify_fixed_cpu_exportemu_lockstep` (CPU, lockstep) | emulated tracking export (real `export.py`, random 125-64-64-22 MLP + normalizer) | 200 (whole clip) | 0.73 / 1.17 | 0 / 0 | ends with `policy->hold(motion_end)` |

A real BeyondMimic-size actor (512-256-128) is **UNVERIFIED** for timing; it is expected to stay well under the 20 ms period.

### 8.2 Fixed-base hold tracking error per motor

Source: `verify_fixed_rt_summary.json` → `trace_analysis`. Newton GPU, hanging robot, legacy gains, hold window
t = 6.2–7.2 s, trace every 5 ticks. Error = q − q*.

| Motor | Mean err [rad] | RMS [rad] | Max \|err\| [rad] | Mean \|tau\| [N·m] | kp |
|---|---:|---:|---:|---:|---:|
| PG_left_leg_pitch | +0.0045 | 0.0047 | 0.0051 | 0.67 | 150 |
| PG_left_leg_roll | −0.0012 | 0.0014 | 0.0052 | 0.17 | 150 |
| LL_hip_joint | −0.0057 | 0.0059 | 0.0063 | 0.85 | 150 |
| LL_knee_actuator_joint | −0.0577 | 0.0606 | 0.0680 | 11.44 | 200 |
| LL_Revolute67 | +0.0009 | 0.0009 | 0.0013 | 0.07 | 80 |
| LL_Revolute81 | +0.0004 | 0.0004 | 0.0005 | 0.03 | 80 |
| PG_right_leg_pitch | +0.0373 | 0.0390 | 0.0413 | 5.59 | 150 |
| PG_right_leg_roll | +0.0006 | 0.0010 | 0.0048 | 0.08 | 150 |
| RL_hip_joint | −0.0061 | 0.0062 | 0.0066 | 0.90 | 150 |
| RL_knee_actuator_joint | −0.0590 | 0.0619 | 0.0693 | 11.70 | 200 |
| RL_Revolute67 | +0.0005 | 0.0006 | 0.0008 | 0.04 | 80 |
| RL_Revolute81 | −0.0000 | 0.0000 | 0.0001 | 0.00 | 80 |
| LH_yaw | +0.0004 | 0.0005 | 0.0008 | 0.02 | 50 |
| LH_pitch | +0.0098 | 0.0101 | 0.0106 | 0.49 | 50 |
| LH_roll | −0.0001 | 0.0002 | 0.0010 | 0.01 | 50 |
| LH_elbow_joint | −0.2213 | 0.2322 | 0.2465 | 11.04 | 50 |
| LH_wrist_roll | −0.0000 | 0.0000 | 0.0000 | 0.00 | 50 |
| RH_yaw | −0.0006 | 0.0007 | 0.0010 | 0.03 | 50 |
| RH_pitch | +0.0098 | 0.0101 | 0.0106 | 0.49 | 50 |
| RH_roll | −0.0001 | 0.0002 | 0.0010 | 0.01 | 50 |
| RH_elbow_joint | −0.2228 | 0.2337 | 0.2480 | 11.11 | 50 |
| RH_wrist_roll | +0.0000 | 0.0000 | 0.0000 | 0.00 | 50 |
| **overall** | | **0.0732** | **0.248** | | |

Findings:
- **Droop matches the motor law.** The errors are steady-state proportional droop under gravity: error ≈ −τ/kp.
  - Elbow: 11.0/50 = 0.22 rad.
  - Knee crank: 11.4/200 = 0.057 rad.
  - Right hip pitch-like motor: 5.6/150 = 0.037 rad.
  - The pure PD law has no integral term, so this is the expected behaviour. A policy compensates through its action,
    and Isaac's implicit PD with the same kp has the same droop.
- **Backends agree.** The same hold on CPU MuJoCo C gives overall RMS 0.07319 rad against 0.07319 on GPU
  (`verify_fixed_cpu_summary.json`). Closure residual after the run: 1.2 mm (GPU and CPU).
- **UNVERIFIED.** The elbow gravity torque (~11 N·m through the four-bar input crank) and the left/right asymmetry of
  the hip pitch-like motors (+0.0045 vs +0.037 rad) have not been cross-checked against Isaac Lab on the same pose.

### 8.3 Free-base time to fall

Source: `verify_free_rt_summary.json` and `verify_free_cpu_summary.json`. The robot is on the ground, and the FSM runs
Passive 0.1 s → MoveToDefault 2 s → Hold with legacy PD. **No balance controller.**

| Backend | Tilt > 0.5 rad | Tilt > 1.0 rad (fall) | FSM state at fall | Closure residual at end |
|---|---:|---:|---|---:|
| GPU MuJoCo Warp | 1.21 s | **1.54 s** (runner) / 1.53 s (trace) | `move_to_default` | 3.1 mm |
| CPU MuJoCo C | 1.21 s | **1.54 s** / 1.53 s | `move_to_default` | 3.2 mm |

The robot falls during MoveToDefault, about 1.5 s after the first command, and both backends agree to the tick.
This is the expected outcome of joint PD without balance. It does not mean the plant is wrong, and it is not a
success.

### 8.4 Unitree example port

Sources:
- `bridge_unitree_example_cpu.log`, `unitree_example_cpu_summary.json`, `unitree_example_cpu_sine_tracking.log`;
- CPU, fixed base, real time.

The run:
- Stage 1 moves all motors to default over 3 s.
- Stage 2 runs 0.5 Hz sines:
  - calf motors A/B at ±15°/±10° in anti-phase;
  - wrist rolls at ±30°.
- Messages: 4001 LowCmd sent, 0 dropped; 4005 LowState received, 1 skipped; 0 decode or CRC errors.
- Stage-2 tracking RMS:
  - calf A 0.094 rad, calf B 0.071 rad;
  - wrist roll 0.043 rad.

  This is PD lag at 0.5 Hz with legacy gains. The example's "max over all motors" metric (0.25 rad) is dominated by the
  elbow droop from 8.2.

### 8.6 Sim2sim of a trained tracking policy (gpu_pipeline, 2026-09-24)

**Setup.**
- Policy: `logs/rsl_rl/dropbear_tracking/2026-09-24_10-33-16_wave_right_s8/exported/policy.json`. It was trained in Isaac Lab at 8/4 iterations for 751 iterations, on `data/motions/synthetic/wave_right.npz`, with the CONTRACTS 0.2 plant.
- Script: `logs/gpu_pipeline/sim2sim/run_sim2sim_wave_cpu.sh <tag> [bridge args]`.
- Bridge: CPU MuJoCo C, free base, lockstep 10. Pre-settled start at NPZ frame 0 (`--start-npz`, 3 s).
- Runner: `--allow-privileged --passive-s 0 --move-s 0.02 --hold-s 0`, so the policy takes over at t = 0.04 s.
- Metrics: `tools/sim2sim_report.py`. `error_joint_pos` uses the Isaac definition (||q - q_ref|| over the 22 motors).

**Isaac reference**, from `play_2026-09-24_11-47-34.json`: 32/4 iterations, 16 envs, 2 loops. No falls, joint error 0.31 rad, anchor 4.6 cm.

| Newton setting (tag) | physics dt | closure `eq_solref` | result | policy time | joint err mean (first 1 s) | right-hand rise (ref 0.356 m) |
|---|---|---|---|---:|---:|---:|
| `cpu` (bridge defaults) | 2 ms | MuJoCo default (0.02, 1) | **falls** (tilt > 1 rad) | 1.88 s | 0.86 (0.36) | n/a |
| `cpu_eqsolref004` | 2 ms | (0.004, 1) | **falls** | 1.94 s | 0.79 (0.32) | n/a |
| `cpu_substep4` | 0.5 ms | default | **falls** | 2.52 s | 0.73 (0.28) | n/a |
| `cpu_substep10` | 0.2 ms | default | **falls** | 2.52 s | 0.73 (0.28) | n/a |
| `cpu_substep4_eq001` | 0.5 ms | (0.001, 1) | **completes the 10 s clip** (501 steps), then `policy->hold(motion_end)` | 10.0 s | 0.356 (0.29) | 0.299 m |
| `cpu_substep10_eq001` | 0.2 ms | (0.001, 1) | **completes the clip** (identical to the 0.5 ms run) | 10.0 s | 0.356 (0.29) | 0.299 m |

**Findings.**
- Neither a finer physics step nor stiffer loop closures is enough alone. Both together make the Isaac-trained policy work in Newton.
  - Joint error 0.356 rad, against 0.31 in Isaac.
  - Anchor height error 1.6 cm mean.
  - Closure residual 0.31 mm, against 3.2 mm with the defaults.
- After the clip, the runner holds with plain PD and the robot falls about 2 s later. This is the known no-balance behaviour (8.3), not the policy.
- Why the defaults fail:
  - At 2 ms the explicit per-step motor PD and the integration leave the calf motor A 0.05-0.1 rad off its target even when hanging (`probe_presettle_convergence.log`).
  - MuJoCo's default soft equalities let the loop closures open by 2-5 mm under load, where PhysX holds 0.3 mm.
- Cost: real-time factor 0.41 on CPU with 4 substeps. Lockstep sim2sim does not need real time.
- `--diag-eq-solref` is implemented for the MuJoCo C backend only. The GPU (MuJoCo Warp) path is **UNVERIFIED**.

**Review addendum (2026-09-24): the passive-damping ablation this analysis lacked.** All six runs above used passive
damping 50 (the bridge default then), although Isaac applies none (CONTRACTS 0.3, measured by
`tools/probe_passive_damping.py`). Same policy, script `logs/review_fixes/sim2sim/run_wave_ablation.sh`, reports
`logs/review_fixes/sim2sim/wave_<tag>_report.json`; one deterministic run each:

| tag | dt / substeps | passive damping | closure `eq_solref` | result | policy time | joint err mean |
|---|---|---:|---|---|---:|---:|
| `pd50_dt2` (old default) | 2 ms / 1 | 50 | default | falls | 1.88 s | 0.86 |
| `pd5_dt2` | 2 ms / 1 | 5 | default | falls | 2.42 s | 0.72 |
| `pd0.5_dt2` | 2 ms / 1 | 0.5 | default | **completes** the clip | 10.0 s | 0.58 |
| `pd0_dt2` (new default) | 2 ms / 1 | 0 | default | falls | 7.74 s | 0.61 |
| **`pd0_dt2_eq004`** | **2 ms / 1** | **0** | **(0.004, 1)** | **completes** the clip | 10.0 s | **0.285** |
| `pd0_sub4` | 0.5 ms / 4 | 0 | default | falls | 6.02 s | 0.58 |
| `pd0.5_sub4` | 0.5 ms / 4 | 0.5 | default | falls | 9.16 s | 0.61 |
| `pd0_sub4_eq001` | 0.5 ms / 4 | 0 | (0.001, 1) | completes | 10.0 s | 0.328 |
| `pd50_sub4_eq001` (old best) | 0.5 ms / 4 | 50 | (0.001, 1) | completes | 10.0 s | 0.356 |

Corrected findings:
- The passive-damping mismatch was the dominant cause of the early falls: removing it takes the default-step run from
  1.88 s to 7.74 s of policy.
- With Isaac-matched damping (0) and stiffer closures, the **default 2 ms step is enough**: the whole clip, joint error
  0.285 rad (Isaac 0.31). The statement above that a finer step is required is withdrawn; stiff closures still matter.
- `--diag-eq-solref` stays a MuJoCo-C-only diagnostic (no Warp path), so the bridge default closure stiffness is
  unchanged. Recommended sim2sim setting until that is ported: `--mujoco-cpu --passive-damping 0 --diag-eq-solref 0.004 1`.
- Each row is a single deterministic rollout of one policy on one clip: not a robustness result.
- Still open (UNVERIFIED): MuJoCo implicit position actuators for the 22 motors (mirroring PhysX's implicit drive).

### 8.5 Tests

| Suite | Interpreter | Result | Log |
|---|---|---|---|
| sdk types / transport, deploy, bridge units, runner loopback, contract consistency, export-sidecar runner | `.venv-newton` | 38 passed, 4 skipped (export tests need torch) | `pytest_cont_venv.log` |
| same | system Python 3.12 | 41 passed, 1 skipped (PD-kernel test needs warp) | `pytest_cont_systempython.log` |
| whole `tests/` | system Python 3.12 | 141 passed, 2 skipped, **1 failed**: `test_semantic_map.py::test_real_roundtrip` (calibration component). It began running when `data/calibration/dropbear_semantic_calibration.json` appeared at 02:33, and was a skip in the earlier run. It does not involve SDK code. | `pytest_cont_all_tests_systempython.log` |

The two interpreters together cover all 42 tests of this component.

## 9. Porting a Unitree `unitree_sdk2_python` low-level example to Dropbear

Reference: `$DROPBEAR_UPSTREAM/unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py`.
Port: `scripts/examples/dropbear_low_level_example.py`, run evidence in 8.4. Line by line:

| Unitree G1 example | Dropbear | Why |
|---|---|---|
| `from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize` | `from dropbear_wbc.sdk.transport import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize` | Same API over ZMQ |
| `from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_` and `unitree_hg_msg_dds__LowCmd_()` | `from dropbear_wbc.sdk.types import LowCmd, LowState`, then `LowCmd()` | Dropbear message classes (22 motor slots, optional neck) |
| `from unitree_sdk2py.utils.crc import CRC`, `self.low_cmd.crc = self.crc.Crc(self.low_cmd)` | delete: `ChannelPublisher.Write()` fills the CRC (`LowCmd.compute_crc()` exists) | Same CRC algorithm, computed automatically |
| `MotionSwitcherClient()`, `CheckMode()`, `ReleaseMode()` | delete | There is no on-board sport/high-level controller to release. The bridge simply waits (paused) for the first LowCmd. |
| `ChannelFactoryInitialize(0, sys.argv[1])` (NIC name) | `ChannelFactoryInitialize(0)` or `ChannelFactoryInitialize(0, cmd_port=…, state_port=…)` | Domain and NIC arguments are accepted and ignored; the endpoints are TCP |
| `G1_NUM_MOTOR = 29`, `class G1JointIndex` | `NUM_MOTOR = motors.NUM_MOTORS` (22); indices via `motors.motor_index("LL_Revolute67")` | Motor contract (table 3.4). Slots are motors, not G1 joints. |
| `Kp`, `Kd` lists (29) | `motors.DEFAULT_KP`, `motors.DEFAULT_KD` | Legacy DROPBEAR_CFG gains |
| `self.low_cmd.mode_pr = Mode.PR`, `self.low_cmd.mode_machine = self.mode_machine_`, waiting for `mode_machine` | delete | No PR/AB modes and no machine-type handshake. The parallel ankle is commanded in motor space (calf A/B). Pitch/roll goes through `SemanticMap`. |
| `self.low_cmd.motor_cmd[i].mode / q / dq / tau / kp / kd = …` | identical | Unitree-style slot views |
| `msg.motor_state[i].q`, `msg.imu_state.rpy` | identical (`motor_state` and `imu_state` aliases; arrays also at `msg.motor.q`, `msg.imu.rpy`) | |
| `RecurrentThread(interval=0.002, target=self.LowCmdWrite)` | a 2 ms `perf_counter` loop (or any timer) | Pacing only |
| `lowstate_subscriber.Init(self.LowStateHandler, 10)` | identical (`queue_len` ignored) | Handler thread with latest-message semantics |
| Stage 1: ramp to the all-zero posture over 3 s | Stage 1: ramp to `motors.DEFAULT_POS` (legacy DROPBEAR_CFG init pose) over 3 s | Dropbear's default is not all-zero (knee cranks and elbows 0.3, calf A −0.2, hip −0.1) |
| Stage 2: ankle pitch/roll sines in PR mode. Stage 3: ankle A/B sines in AB mode, plus wrist yaw. | Stage 2: calf motor A/B sines (anti-phase) and wrist rolls, in motor space | Dropbear's parallel ankle is always "AB" at the SDK level. It has no wrist yaw. |
| (none) | optional: `self.low_cmd.tick = self.low_state.tick` | Lets the bridge measure command latency; needed for `--lockstep-ticks` |

## 10. Files

| Path | Role |
|---|---|
| `source/dropbear_wbc/sdk/types.py`, `crc.py`, `transport.py`, `motors.py` | `dropbear_hg-v1` messages, CRC, ZMQ channels, motor table |
| `source/dropbear_wbc/newton_sim/plant.py`, `imu.py`, `usd_fixes.py`, `gpu_lock.py` | Newton plant, IMU / LowState assembly, CONTRACTS 0.1 fixes, GPU lock |
| `source/dropbear_wbc/deploy/config.py`, `fsm.py`, `observations.py`, `motion.py`, `policy.py`, `quat.py` | Runner configuration, FSM, observation terms, reference motion (NPZ / CSV / ONNX), ONNX wrapper |
| `tools/newton_bridge.py`, `tools/policy_runner.py`, `tools/probe_newton_plant.py` | Bridge, runner, plant probe |
| `scripts/verify_sdk_bridge.py` | Two-process end-to-end harness |
| `scripts/examples/dropbear_low_level_example.py` | Unitree G1 low-level example port |
| `scripts/make_synthetic_tracking_policy.py` | Synthetic BeyondMimic-layout ONNX + sidecar (plumbing only) |
| `scripts/emulate_tracking_export.py` | Runs the real `tasks/tracking/export.py` on CPU with Isaac stubs (format check only) |
| `tests/test_sdk_*.py`, `test_deploy.py`, `test_newton_bridge_units.py`, `test_runner_loopback.py`, `test_export_sidecar_runner.py` | Tests |

## 11. Known gaps

- **Trained policy.** *Updated 2026-09-24:* the wave_right tracking policy runs through the runner (8.6). It completes
  its clip only with the sim2sim fidelity settings (0.5 ms step and stiff closures, CPU backend). With the bridge
  defaults it falls after 1.9 s. Longer or whole-body clips, the GPU backend with stiff closures, and hardware-style
  (non-privileged) policies are **UNVERIFIED**.
- **Start pose.** *Addressed 2026-09-24:* `--start-npz` gives a pre-settled start (section 5.2). It is not an
  exact RSI: Newton's closures settle to their own equilibrium, which is not Isaac's. The calf motor A sits
  0.035-0.07 rad off the Isaac-settled standing pose even when hanging
  (`logs/gpu_pipeline/sim2sim/probe_presettle_convergence.log`), and the ankle joints are not at limits
  (`probe_presettle_limits.log`). This is **UNRESOLVED**.
- **Isaac vs Newton plant parity** (hold droop, contacts, closure behaviour) has not been measured side by side.
- **Not implemented.** DDS / `unitree_hg` wire compatibility, the IMU noise model, motor thermal/voltage model,
  joystick (`wireless_remote`) and hardware.
