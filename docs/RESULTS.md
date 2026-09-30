# Results

Every number is **simulation** (Isaac Sim 5.0 + Isaac Lab 2.2, Dropbear USD with 27 loop closures). "Idealized motors"
= the original Isaac actuator settings; "motor twin" = `DatasheetMotor` with the real motor map (ACTUATORS §11). The
source of each result is given (document and section, or the time in [OVERNIGHT_BREV](OVERNIGHT_BREV_2026-09-25.md)).

**HW gate** = `tools/telemetry_issue_scan.py`: FAIL on any HIGH finding for falls, thermal overload (RMS over rated),
torque clipping, over-speed, loading a hard stop, touchdown skid, mid-stance slip or scuffing. **Body error** = mean
distance between the simulated and reference link positions. **No-state** = the policy only sees what the real robot
can measure.

## Walking on the motor twin

| policy | motors | result | source |
|---|---|---|---|
| idealized-motor walker `stiff_knee_hw`, zero-shot | `hw_v1` | falls 18/18 (median 1.2 s) | ACTUATORS §11 |
| `kimodo_cyclic_nostate_v1i_feet/3184` | `hw_v1i` | first **PASS**: skid none (was 17.7 cm), ankle rod 1.01× rated, body 7.02 cm | ON 02:30 |
| `kimodo_ts119_nostate_allfix/4400` | `hw_v1ie` | **PASS**, 0 falls, body 7.1 cm; anchor drift at 15 s **0.52 m** (2.04 m untimed) | ISSUES #27 |
| `kimodo_ts119_allfix_smooth/5800` (published) | `hw_v1ie` | **PASS**, 0 falls, body 6.2 cm, target jitter 1.01°, worst rod 1.9°, 200 Hz jumps 10 % | ON 06:55 |
| `kimodo_cyclic_nostate_v1i_feet/4577`, `kimodo_ts119_nostate_allfix/5900` | | **PASS**, 0 falls | ON 07:13 |
| `lafan_cyclic_nostate_v1i_kstop_feet/3100` (published) | `hw_v1i`, CEM-60 knee | **PASS**: knee on its stop 0.2 % / 0 %, knees 0.94× / 0.82× rated | ON 06:45; ISSUES #19 |
| `lafan_nostate_smooth_elbow/8000` (published) | `hw_v1ie` | **PASS**, 0 falls, 200 Hz jumps 7 %; right elbow 1.25× rated | ON wrap-up |
| `asap_walk_hw_reg/4700` | `hw_v1` | 0 falls, body 4.3 cm; right elbow 1.24× rated | ACTUATORS §13 |

**Knee A/B** (LAFAN walk; ACTUATORS §13): body error 10.0 cm with the CEM-60 knee vs 5.1 cm with an X10-S2; knee
clipped 82–85 % vs 14 %; knee RMS 1.65–1.68× vs 1.03–1.15× rated. **Conclusion:** an upright gait fits the CEM-60, a
crouched gait needs X10-S2-class torque unless the knee-stop penalty keeps it off the stop.

**Target ramp A/B** (same policy; ISSUES #11): 200 Hz torque jumps above 25 % of peak 27.0 % → **18.3 %** of steps,
p99 jump 34.6 → 25.6 %, tracking unchanged (0.092 vs 0.091 m).

**Knee-stop penalty** (LAFAN on the CEM-60; ON 04:05): knee on its flexion stop 32 % / 45 % → **0.9 % / 0 %**, knee RMS
1.18× / 1.46× → **0.88× / 0.81×** rated.

## Motion-library trackers

Continuous loops, one environment per clip, clips counted if they never fall.

| policy | motors | clips with 0 falls | source |
|---|---|---|---|
| `lib_v2_main` | idealized | 51/78 | ON 15:31 |
| `lib_v2_nostate` | idealized | 57/78, body 6.9 cm | ON 15:40 |
| `lib_nostate_hw_v5/8900` | `hw_v1` | 38/78 | ON 07:22 |
| `lib_nostate_kneeX10S2_v5/8200` | X10-S2 knee | 41/78; zero-shot on the Froude library 43/78 (falls -8 %) | ON 02:10, 02:58 |
| `lib_v6ts_nostate_v1i_fix/10492` (published) | `hw_v1i` | 41/78, 727 falls | ON 07:22 |
| `lib_v6ts_v1i_fix2/13700` (published) | `hw_v1i` | **46/78**, 665 falls, body 8.3 cm | ON wrap-up |
| `lib_v7gen_v1ie_smooth/13600` (published) | `hw_v1ie` | **94/138** = 43/78 original + **51/60 Kimodo-generated**, body 7.7 cm | ON wrap-up |

Repeatability: two independent evaluations gave 94/138 (851 vs 852 falls) and 46/78 (665 vs 670 falls). The generated
motions `lib_v7gen` still drops: arms up, bounce, boxing, clap, knee raise, reach down, turn around, walk in a circle,
walk-turn-right.

**Froude library v6 vs v5** (ON 02:46): clips within every motor's no-load speed 70 → 71; motor-clip pairs over the
no-load speed 18 → 16, over the peak-torque speed 39 → 34.

## Velocity and terrain

| walker | result | source |
|---|---|---|
| `vel_flat_kneehw` (idealized) | commanded → achieved: fwd 0.3 → 0.34, 0.5 → 0.53, turn ±0.5 → ±0.47, left 0.2 → 0.20 m/s; weak right / backwards | ON 08:20 |
| `vel_hw_v1_ft/2600` | 0/18 falls, but knee 1.75× rated, pinned | ACTUATORS §11 |
| `vel_rough_v1i/3200` | 0 falls; legs crossed 39 %, target jitter 6–8° | ISSUES #29 |
| `vel_rough_v1i_b/5500` (published) | **PASS**, 0 falls; crossed 2.2 %, jitter 2.9–3.9°; limps (stance 0.73 / 0.42), knees near the stop 89 % | ON wrap-up |

## Live text to motion

| metric | value | source |
|---|---|---|
| prompt → spliceable clip | 8.5–10.6 s (text 4.1–4.7 s, diffusion 6.1–6.5 s, retarget 2.0–3.9 s) | TEXT_TO_MOTION |
| physics speed | 1.00× real time between prompts, 0.92× averaged over a busy 2 min; 18.9 ms per 20 ms step on the CPU pipeline (GPU pipeline: 0.15–0.17×) | ISSUES #30 |
| session (5 prompts) | 0 falls; recorded replay HW gate **PASS**, 200 Hz jumps 1 % | TEXT_TO_MOTION |
| text encoder memory | int8 memory-mapped: 1.2 GB committed (bf16: 16.4 GB), loads in 1 s | ISSUES #32 |
| text encoder fidelity | cosine to bf16 0.9995 min / 0.9996 median on 60 prompts (dynamic int8: 0.82, rejected) | ISSUES #32 |
| speed gate | a 2.5× sped-up walk is slowed 1.53×; 3× is refused; all 14 generated clips passed unchanged | ISSUES #35 |
| retrieval (no generation) | 10–18 ms per prompt | TEXT_TO_MOTION |

## Sim-to-sim, SDK, teleop, GR00T

| item | result | source |
|---|---|---|
| wave policy, Isaac → Newton / MuJoCo (2 ms, stiff closures) | completes; joint error 0.285 rad (Isaac 0.31) | SDK §8.6 |
| dance policy sim-to-sim | upright 29.1 s; joint error 0.80 rad (Isaac 0.81) | DEMOS §1 |
| Newton bridge | 500 Hz at real-time factor 1.00, physics 0.70 ms p50 | SDK §8.1 |
| teleop wrist tracking | FK 1.8 mm mean; simulated hand 5.3–5.9 mm mean | TELEOP §5.2 |
| GR00T tabletop push, scripted | 42/50 = 84 % (Wilson 71.5–91.7 %); 34-episode dataset | GROOT §0, §2 |

## Idealized-motor trackers (early; not hardware evidence)

| policy | result | source |
|---|---|---|
| wave_right v1 `model_743` | 0 falls, body 2.8 cm | DEMOS §1 |
| `lafan_walk1_v4` | 0 falls, body 2.8 cm, anchor 5.2 cm | ON 04:25 |
| `kimodo_walk_v4` / `take102_v3_s16` / `wave_v2_s16` | 0 falls; body 3.7 / 4.6 / 5.6 cm | ON |
| `lafan_dance2_v4`, `take102_v4`, `wave_right_v2` | not learned / collapsed / terminates at 3.64 s | ON; DEMOS §2.3 |
