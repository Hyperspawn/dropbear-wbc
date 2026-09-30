# H1-style velocity walking for Dropbear (`Dropbear-Velocity-Flat-v0`)

The classic Unitree H1 capability (Isaac Lab `Isaac-Velocity-Flat-H1-v0`, unitree_rl_lab `Unitree-H1-Velocity`):
a policy that follows a joystick-style command `(vx, vy, wz)` on flat ground. Contract: `docs/CONTRACTS.md` section 7.
Progress, runs and evidence: `logs/locomotion/PROGRESS.md`.

## 1. What is different from every earlier Dropbear walking attempt

The user's `dropbear_walk` velocity env, the research repo's legacy v0.1.0 (0.2 m/s on the unfixed plant) and
Codex's control-v1 (never passed weight transfer) all trained on a plant with

- the **left knee locked** (`LL_Revolute121` authored with axis X),
- the **hip motors held by joint friction** (`physxJoint:jointFriction` 0.5, scaled by the constraint force),
- an **over-constrained ankle** (revolute tie-rod closures, Grübler mobility 0, roll limited to about ±8 deg),

and reset from the **airborne CAD pose**. Here all three plant defects are fixed at spawn (CONTRACTS 0.1-0.3) and
every episode starts from a **settled, closure-consistent standing state** with both feet loaded.

## 2. Task design (and why)

| aspect | choice | reason |
|---|---|---|
| plant | `make_dropbear_cfg` (contract plant, calibration v3 standing pose as action offset) | the only plant whose kinematics match the robot |
| action | `JointPositionAction`, 22 motors, contract order; legs `0.25*effort/kp` (hips 0.333, knees 0.375, ankles 0.25 rad), arms 0.1 rad | same action interface as tracking and the SDK sidecar; arms may swing/counter-balance a little (H1 also keeps its arms in the action with a deviation penalty) but a small scale plus `joint_deviation_arms` keeps them from becoming the balance actuator (research REWARD_DESIGN.md "Arms"). A 12-motor action would need the deploy runner to hold the arms itself |
| reset | full settled joint row (91 DOFs) + root from `data/motions/smoke/dropbear_static_stand.npz`; rigid yaw ±pi and x/y ±0.5 m; noise on the 6 serial leg motors (±0.02 rad) and 8 serial arm motors (±0.1 rad) only | never the airborne CAD reset (research rule 32); never perturb passive or closure-coupled joints (CONTRACTS 1; closure-motor noise tore closures by 2.4-3.9 cm) |
| torso signals | anchor (chest) body `head_5mm_ujoint_base__5__1`: height target and fall threshold from the NPZ (1.498 m, fall below 1.198 m), orientation from the anchor frame × `ANCHOR_FRAME_OFFSET_WXYZ` | the root `world` frame origin is ~12.5 cm BELOW the soles; the loaded, reset-calibrated torso height is the anti-collapse signal (research rules 15/25/29/33) |
| policy observations | `base_lin_vel` (sim-only), `base_ang_vel`, `projected_gravity`, `velocity_commands`, 22 motor `joint_pos_rel`/`joint_vel_rel`, `last_action` (78) | H1 flat layout; the root is the rigid torso+pelvis, world-aligned at rest, so root IMU quantities are the torso's and match `LowState.imu` |
| feet | contact bodies `LL/RL_skateboard_bearing_left_2` (sole) + `LL/RL_basis_left_1` (ankle cross), grouped per foot | explicit bodies, never a `.*skateboard.*` regex that also selects knee bearings (research rule 10) |
| rewards | H1: lin/ang velocity tracking (exp, std 0.5), termination −200, `lin_vel_z` −0.5, `ang_vel_xy` −0.05, anchor flat orientation −1, anchor height −5, torques −2e-6, joint acc −1.25e-7, action rate −0.01, leg joint limits −1, hip roll/yaw deviation −0.2, arm deviation −0.2, biped air time +1 (0.5 s cap), foot slide −0.25 | Isaac Lab H1 flat weights where they exist; motor-only joint terms (passive DOFs are never penalized) |
| terminations | time out 20 s; anchor below stand −0.30 m; anchor tilt > 0.8 rad; any non-foot body in ground contact | falls detected on the chest, not the root frame |
| commands | yaw rate (no heading), 10 % standing envs, resample every 10 s; start vx 0..0.5, vy ±0.2, wz ±0.5; curriculum +0.1 per success (per-second tracking reward > 0.8 × weight at an episode boundary) up to vx −0.3..1.0, vy ±0.4, wz ±1.0 | unitree_rl_lab curriculum; the ranges are checkpointed (`LocomotionOnPolicyRunner`) because each training chunk restarts Isaac |
| randomization | friction 0.4-1.0, root mass −1..+2 kg; pushes OFF | first get flat walking, then robustness (research rule 12) |
| sim | dt 5 ms, decimation 4 (50 Hz), 2048 envs, PhysX 8/4 for training, 32/4 for play/eval | CONTRACTS 5 throughput/closure measurement |
| PPO | rsl-rl 2.3.3, [512, 256, 128] ELU, lr 1e-3 adaptive, entropy 0.01, 24 steps/env, empirical normalization | H1 rough/flat hyper-parameters |

## 3. Commands

All GPU work goes through `tools/gpu_lock_run.py`.

```bash
# smoke: env build, reset on the ground, zero-action episodes
python tools/gpu_lock_run.py --owner locomotion --log logs/locomotion/probe_smoke.log --timeout 900 -- \
  C:/isaac-sim/python.bat -u tools/locomotion_env_probe.py --num_envs 64 --steps 250 --headless \
  --out logs/locomotion/probe_smoke.json

# short PPO
python tools/gpu_lock_run.py --owner locomotion --log logs/locomotion/ppo_smoke.log --timeout 1200 -- \
  C:/isaac-sim/python.bat -u scripts/train_locomotion.py --num_envs 2048 --max_iterations 20 --run_name smoke --headless

# long chunked run (detached via WMI, see logs/locomotion/PROGRESS.md for the exact launch). --until_iteration makes
# the total resume-safe. The driver clears a GPU lock whose file predates the current boot.
python tools/run_chunked_locomotion.py --run_name vel_flat_kneehw --chunks 30 --iters_per_chunk 150 --until_iteration 3000 \
  --num_envs 2048 --solver_iters 8 4 --save_interval 50 --timeout 1800 --first_wait_minutes 240 --wait_minutes 240 \
  --actuator_profile stiff_knee_hw
# resume after a reboot or kill: the same line + --resume_run <run dir name>. The latest checkpoint, lr and curriculum
# are restored, and the profile comes from run_info.json.

# actuator-profile probe (zero action, per-env gain groups) and A/B
python tools/gpu_lock_run.py --owner locomotion --log logs/locomotion/probe_knee_authority_v1.log --timeout 900 -- \
  C:/isaac-sim/python.bat -u tools/probe_knee_authority.py --num_envs 64 --steps 250 --solver_iters 8 4 --headless \
  --out logs/locomotion/probe_knee_authority_v1.json

# demo-quality video (demo_eval's clean renderer, tracking camera, env 0 follows --video_script)
python tools/gpu_lock_run.py --owner locomotion --log logs/locomotion/play_clean_video.log --timeout 1800 -- \
  C:/isaac-sim/python.bat -u scripts/play_locomotion.py --load_run <run dir> --clean_video --steps 1000 --headless

# evaluate at 32/4 (fixed commands per env), export ONNX + sidecar, video
python tools/gpu_lock_run.py --owner locomotion --log logs/locomotion/play_eval.log --timeout 1200 -- \
  C:/isaac-sim/python.bat -u scripts/play_locomotion.py --load_run <run dir> --steps 750 --export --headless
python tools/gpu_lock_run.py --owner locomotion --log logs/locomotion/play_video.log --timeout 1200 -- \
  C:/isaac-sim/python.bat -u scripts/play_locomotion.py --load_run <run dir> --video --steps 1000 --headless

# sim2sim / deploy (Newton bridge + runner, sim-only because of base_lin_vel)
python tools/policy_runner.py --mode policy --sidecar <run>/exported/policy.json --allow-privileged --velocity-cmd 0.3 0 0
```

## 4. Status

See `logs/locomotion/PROGRESS.md` (newest entry on top) for what is VERIFIED and what is not.

## 5. Actuator profile (knee authority), added 2026-09-24

The velocity task has its own leg actuator settings, like Unitree's per-joint `deploy.yaml` stiffness:
`config/dropbear/flat_env_cfg.ACTUATOR_PROFILES`, applied by `set_actuator_profile` and selected with
`--actuator_profile` on train and play (play defaults to the run's `run_info.json`). The export carries the live
values. The tracking task keeps the legacy gains. Decision and reasons: `docs/DECISIONS.md` (2026-09-24 "Velocity task
knee gains").

| profile | knee crank kp / kd / effort | reflected at the knee (x 1/1.8^2) | status |
|---|---|---|---|
| `legacy` | 200 / 12 / 300 | 62 / 3.7 N*m/rad, 167 N*m | tracking-task values; effort 3x the motor peak |
| `stiff_knee` | 600 / 20 / 300 | 186 / 6.2, 167 N*m | fastest learner in the A/B; effort NOT plausible |
| **`stiff_knee_hw`** (default) | 600 / 20 / **100** | 186 / 6.2, **56 N*m** | RMD-X10-S2 V3 datasheet peak (rated 50) |

Evidence:
- **LUT slope.** `data/calibration/dropbear_semantic_calibration.json` `dofs.*_knee`: 1.79 / 1.82 at the stand.
- **Static torque.** Crank torque needed per leg at the 20 deg standing knee: ~75 N*m in double support, ~150 N*m on one
  leg (at 10 deg: 36 / 72). This is a virtual-work estimate with half the weight per leg, so an upper bound.
- **Gain probe.** `tools/probe_knee_authority.py` -> `logs/locomotion/probe_knee_authority_v1.json`. Stiffer knees delay the
  zero-action collapse (0.72 -> 1.02 s) but do not prevent it.
- **PPO A/B.** `logs/locomotion/ab_summary_v1.json`.
- **Consequence of the 100 N*m cap.** At the 20 deg standing knee, double support already takes ~75 % of the peak, and
  single support at 20 deg exceeds it. The policy has to walk with a straighter stance knee (<= ~10 deg). That is
  feasible, but it is a real hardware constraint worth confirming on the robot.
- **Hip pitch.** Serial, direct drive, kp 150 as in H1. It is unchanged, because kp 300 made no difference in the probe.

Other leg efforts exceed the datasheet peaks too (hips 200 vs 40-100; ankle motors 80 vs 25, i.e. 147 vs 46 N*m ankle
pitch). That is flagged in DECISIONS and is the next A/B (`hw_all`).
