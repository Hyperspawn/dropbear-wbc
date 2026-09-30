---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://github.com/Hyperspawn/dropbear-wbc/blob/main/LICENSE
tags:
  - robotics
  - humanoid
  - reinforcement-learning
  - isaac-lab
  - motion-tracking
  - sim2real
library_name: rsl-rl
pipeline_tag: reinforcement-learning
---

# Dropbear whole-body control: plant, trained policies, media and motions

Assets for [**Hyperspawn/dropbear-wbc**](https://github.com/Hyperspawn/dropbear-wbc), the whole-body-control and
sim-to-real stack of **Dropbear**, an open 3D-printed humanoid robot. The code, documentation and install guide live in
the GitHub repository; this repository holds the files too large for git. Fetch them with the repository's
`python tools/fetch_assets.py` (SHA-256 verified against `assets_manifest.json`).

**Try it without installing anything:** [browser replays of the motor twin](https://hyperspawn.github.io/dropbear-wbc/).

| folder | contents |
|---|---|
| `robot/dropbear.usd` | the plant (421 MB USDC, SHA-256 `45586414...`): 90 bodies, 22 motors, 27 loop-closure joints |
| `policies/<run>/` | trained rsl-rl checkpoints + `run_info.json` (actuator profile, calibration, target clamp) + `params/*.yaml` |
| `media/` | dashboard videos: physics rollout + per-motor torque against the motor datasheets |
| `motions/` | redistributable retargeted clips (Kimodo-generated, Kimodo G1 examples, ASAP), NPZ + validation verdicts |
| `datasets/groot/` | GR00T tabletop push dataset (LeRobot format, simulated) |

All results are **simulation** results on a motor twin: Isaac Lab 2.2 / Isaac Sim 5.0, datasheet torque-speed
envelopes, friction, latency, the 20 ms target ramp the firmware is meant to implement, and the robot's 27 closed
kinematic loops. They are not measurements on the real robot.

## Policies

| run / checkpoint | task | trained on | actuator profile | verified result (sim) |
|---|---|---|---|---|
| `lib_v7gen_v1ie_smooth/model_13600` | tracking library, no-state obs | `accepted_v7gen`: 138 clips (60 Kimodo-generated, 51 ASAP, 10 BONES-SEED, 6 synthetic, 5 Kimodo G1, 4 LAFAN1, 2 unitree_rl_lab) | `hw_v1ie` | 94/138 clips with 0 falls (51/60 generated, 43/78 original); default tracker of the live text-to-motion session |
| `lib_v6ts_v1i_fix2/model_13700` | tracking library, no-state obs | `accepted_v6ts`: 78 clips (51 ASAP, 10 BONES-SEED, 6 synthetic, 5 Kimodo G1, 4 LAFAN1, 2 unitree_rl_lab) | `hw_v1i` | 46/78 clips with 0 falls |
| `lib_v6ts_nostate_v1i_fix/model_10492` | tracking library, no-state obs | `accepted_v6ts` | `hw_v1i` | 41/78; recorded live session: 0 falls, HW gate PASS |
| `kimodo_ts119_allfix_smooth/model_5800` | single-clip tracking, no-state | Kimodo G1 walk (Froude-timed cyclic); warm-started from earlier Kimodo-walk trackers | `hw_v1ie` | walk: HW gate PASS, 0 falls, body error 6.2 cm |
| `lafan_nostate_smooth_elbow/model_8000` | single-clip tracking, no-state | LAFAN1 walk1 (cyclic) | `hw_v1ie` | HW gate PASS, 0 falls; right elbow 1.25x rated |
| `lafan_cyclic_nostate_v1i_kstop_feet/model_3100` | single-clip tracking, no-state | LAFAN1 walk1 (cyclic) | `hw_v1i` (CEM-60 knee) | HW gate PASS: knee stop contact 0.2 %, knees 0.94x / 0.82x rated |
| `vel_rough_v1i_b/model_5500` | velocity, rough terrain (blind) | no motion data (reinforcement learning only) | `hw_v1i` | HW gate PASS, 0 falls; limps, knees near the extension stop |
| `vel_hw_v1_gait3b_ft/model_1300` | velocity, flat | no motion data | `hw_v1` | flat-ground gait-shaped walker |

`hw_v1` = datasheet motor map; `hw_v1i` = + 4-step (20 ms) target ramp; `hw_v1ie` = + stiffer elbow gains. "HW gate" =
`tools/telemetry_issue_scan.py` (fails on falls, thermal overload, clipping, over-speed, hard-stop loading, skids,
slips, scuffs).

## Licenses and provenance

- **This repository** (plant USD, policies, videos, datasets) and the code that produced it: **noncommercial use only**,
  PolyForm Noncommercial License 1.0.0 (`LICENSE`). **For commercial use, contact Hyperspawn**: [hyperspawn.co](https://hyperspawn.co), priyanshu@hyperspawn.co.
- **Motions** here: retargeted from Kimodo-generated / Kimodo G1 motions (outputs of NVIDIA Kimodo-G1 under the NVIDIA
  Open Model License) and ASAP motions (MIT); the retargeted clips are this project's (noncommercial). Clips from LAFAN1 (CC BY-NC-ND 4.0), NVIDIA BONES-SEED sample data (evaluation license) and
  unitree_rl_lab dance mocap are **not** included; the repository documents how to rebuild them from your own download.
- **Policies** were trained on the libraries listed above, which include LAFAN1 and BONES-SEED sample clips, and two
  are LAFAN1-walk trackers. They are released for **noncommercial research use** only. For a clean-lineage library
  policy, train on `accepted_v7gen_public.json` (redistributable clips only).
- **Videos** show the simulated robot performing those motions; the same provenance applies.

## Use

```bash
git clone https://github.com/Hyperspawn/dropbear-wbc && cd dropbear-wbc
pip install huggingface_hub && python tools/fetch_assets.py            # USD + policies
python tools/fetch_assets.py --groups media,motions,datasets           # everything else
```

Then follow `docs/INSTALL.md` and `docs/QUICKSTART.md` in the repository.
