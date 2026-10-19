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

## Cloud session 3 (2026-10-14/15): the vendor CEM-60 knee

4x L40S (crusoe, $6.96/h), 8.5 h, about $59. Every run on the vendor knee (`hw_v2*`: 48 V, 330 W, 60 N*m rated, 75 N*m
peak) with mass randomization (0..+6 kg on the base, limbs x0.9-1.1). Evaluated locally at nominal mass, 32/4
(`tools/brev/eval_session3.sh`, `logs/hw_twin/session3/`; videos `logs/dashboards/*_v2i*_dashboard.mp4`).

| run | result |
|---|---|
| `lib_v8_v2ie/17381` (library, all hw terms) | 94/138 clips with 0 falls (50/60 generated), 802 falls (v7gen: 94/138, 852), body 7.7 cm |
| `lib_v8_v2ie_nokstop/17381` (no knee-stop term) | **97/138** (51/60), 806 falls; gains half squat, reach down, a kick, a shot, walk-turn-right, walk-wave; loses disco, march, walk-circle |
| `lowbody_v2ie/17800` on the Kimodo squat | knee **48 deg (the mechanical limit)**, hip drop **0.16 m** of the reference's 0.22 m (published policy: 12 deg, 0.03 m); gate FAIL: right knee on its flexion stop 11-13 % |
| `lowbody_v2ie` on the half squat / lunge / ASAP squat 1 | half squat tracked (drop 5.6 vs 5.9 cm) but the right elbow is pinned (99 % clipped); lunge PASS; ASAP squat PASS |
| `kimodo_walk_v2ie/7792` | **PASS**, 0 falls, body 6.3 cm, knees 0.64/0.68x rated |
| `vel_rough_v2i/11460` | limp gone (stance 0.62/0.64, was 0.73/0.42), 0 falls; gate FAIL on the right elbow resting on a stop (12 %) |

**Gap in the hardware gate found here:** both terrain walkers (old and new) walk with the knees at about 47 deg, i.e.
on the knee's flexion stop 78-89 % of the time, with the knee motors lightly loaded (0.28-0.48x rated): the printed stop
carries part of the weight. `stop_load` only fires when the motor pushes away from the stop, so the gate passes it.
ISSUES #29 called this the "extension stop"; it is the flexion stop. Next: the knee-stop term for the velocity task, and
a gate check for a joint resting on its stop in stance.

## Cloud session 4 (2026-10-15): knee-stop fine-tunes

4x L40S, fine-tunes with a knee-stop term on the vendor knee (`tools/brev/run_session4.sh`). The local watchdog died at
about 17:33 and the in-VM shutdown backstop left Brev billing the instance as "Running" until 02:47 (docs/BREV.md
lesson); the final checkpoints were recovered on 2026-10-16 after Brev reset the instance. Evaluated locally with the
`stop_rest` gate (`logs/hw_twin/session4/`).

| run | result |
|---|---|
| `lib_v8_nokstop_kstop3/19800` (library, light knee-stop -3, from the 97/138 tracker) | **99/138** (53/60 generated, 46/78 original), 810 falls, body 7.6 cm; gains disco and march, loses none. Half squat no longer rests on the knee stop (was 95 % of stance); remaining gate failures are the known elbow and calf-B overloads |
| `lowbody_v2ie_kstop3/21800` (squats, light knee-stop -3) | off the knee stop, but shallow: Kimodo squat 20.6 deg / 4.6 cm (k44 clip 23.9 deg / 5.0 cm) against 48 deg / 15.7 cm without the term; knee motors still at their 75 N*m peak; **calf B (`*_Revolute81`, X8 Pro) 1.3-1.4x rated** in squats and lunges |
| `vel_rough_v2i_kstop10/15632`, `..._kstop3/15632` (terrain, knee-stop from the crouched walker) | no change: knee on its flexion stop in 95-96 % of stance. The knee-stop reward stayed at -0.74 (as large as the velocity reward) for all 29 chunks: fine-tuning cannot leave the crouched optimum. Needs a run with the term from iteration 0 |

**Squat trade-off on this hardware:** without the knee-stop term the policy squats to the knee's 48 deg limit and rests
on the printed stop; with it, it stays off the stop by squatting shallowly. Re-retargeting the squats at a 44 deg knee
cap (`tools/retarget_g1.py --knee-max-deg 44`, library `lowbody_k44_v1`) keeps the reference's hip drop (21.5 vs 21.7 cm)
but the fine-tune did not exploit it. Either way the knee motor runs at its peak and calf B overheats: deeper squats
need more knee torque (a lower-ratio linkage or stronger motor) and a stronger calf-B motor.

## 2026-10-16: the real motors, and "did it actually do the motion?"

**The actuator map was wrong at the hips.**
- The robot has CEM-60s at hip pitch and knee, and no hip roll motor yet.
- Every sim profile before `hw_cem6` / `hw_v4` modeled the CAD's X10-S2 at hip pitch (100 N·m peak, 0.70 kg·m² reflected inertia vs ~0.06) and an X10 1:7 at hip roll (40 N·m peak).
- In the motion library that hip roll sat at its 40 N·m limit, while one-leg stance needs about 56 N·m.
- The two candidate builds are now profiles with motor masses ([ACTUATORS.md](ACTUATORS.md) section 2):
  - `hw_cem6`: 6 × CEM-60;
  - `hw_v4`: X8-120 at the hips, X10-200 at the knee.

**"No fall" overstated the library tracker.** On 14 clips with visible leg features, `lib_v8_nokstop_kstop3/19800` (99/138 without a fall) keeps both feet planted (`tools/motion_fidelity.py`, `tools/eval_fidelity_set.sh`):
- jumps: 0 % flight vs 17–25 % in the reference;
- kick and one-leg balance: about 1 cm foot lift vs 21–23 cm;
- squat: 1 cm torso drop vs 18 cm.

Only march and bow are fully performed (2 of 14, mean score 0.29). This is the baseline the builds are scored against.

**Rough-terrain walker from scratch with the knee-stop term** (`vel_rough_v2i_ks0`, session 5, old hip motors):
- By iteration 450 the knees were off their flexion stop in stance; every earlier walker rested there 95–99 % of the time.
- The knee motor then carries the load: about 62 N·m RMS at the CEM-60's 60 N·m rating.
- At iteration 1,639 it tracks 0.5 m/s and turns well, with no falls, but limps: stance L/R 0.78 / 0.27, left hip roll at 1.6× its rating.

**Sessions 5b and 6: the two motor builds.** Library tracker and rough-terrain walker, each fine-tuned on `hw_cem6`
(Build A) and `hw_v4` (Build B) with identical settings, scored on the 14-clip fidelity set and the hardware gate.

- **Terrain walkers (vel_rough_*, ~2,700 iterations on the new motors).** Both pass the hardware gate with no falls.
  Stance is even (51 / 50 %, was 78 / 27 on the old hip roll), and both track 0.5 m/s and turn at 0.46–0.47 rad/s.
  The first rough-terrain walkers to pass the gate.
- **Library tracker scores.**

  | Library tracker, mean score (old motors 0.29) | Build A | Build B |
  |---|---:|---:|
  | Motors only | 0.28 | 0.32 |
  | + head and foot height rewards, 1,000 iterations | 0.38 | 0.39 |
  | + heights, 2,800 iterations | 0.39 | 0.31 |

  Per-clip scores swing by up to 0.5 between checkpoints. This library score does not separate the builds.
- **What separates them is torque headroom on the hardest motion:**
  - Build A's CEM-60 knee spends 13 → 16 % of the time at its 75 N·m peak, and its hip pitch 4 → 9 %, as training asks
    for more motion.
  - Build B's X10-200 knee spends 1 %, and its X8-120 hip pitch at most 2 %.
- **Hardware gate pass counts** at 2,800 iterations: A 10 / 14, B 6 / 14, mostly the right elbow (X8 Pro behind its
  4.8× linkage).
- **From scratch on Build B** (single motion, height rewards, no knee-stop term, 2,800 iterations):
  - **One-foot balance is performed:** foot lift 21.4 cm vs 21.1 cm, score 1.00, no falls. The library trackers never
    got past 2 cm: an inherited habit, not the hardware.
  - **The squat reaches 6 of 18 cm** with the knee on its 47° limit: a knee-range limit.
  - Both rest the knee on a hard stop (one-foot on the straight stop, squat on the bent stop) with the motor near
    0.5 N·m, so the final training needs the knee-stop term.

**Takeaways:**
- The training objective and inherited habits were the main reason the robot skipped the hard motions.
- The knee's range is the main hardware limit.
- Build B has the torque headroom that Build A runs out of.


## Cloud session 7 (2026-10-17): protecting the printed parts

Build B (`hw_v4`) with the explicit end-stop model (`--stop_model`): each end stop is a stiff spring past the
mechanical limit, so the load on the printed knee-lock blocks is measurable (`tools/joint_loads.py`). Every policy
below was scored on the 14-clip set with the stop model on. Results in `logs/fidelity/s7_*` and `logs/loads/s7_*`.

**Library tracker, fine-tuned from `lib_v4_h/23800`:**

| | knee-lock block, worst | ankle pins, worst | mean score | HW gate | ankle motor B, avg ÷ rated (L / R) |
|---|---:|---:|---:|---:|---:|
| Before (`lib_v4_h/23800`) | 3,851 N | 4,327 N | 0.31 | 6 / 14 | 0.93 / 0.89 |
| + stop-load and impact penalties (`lib_v4_safe/27200`) | 81 N | 2,671 N | 0.35 | 1 / 14 | 1.10 / 1.46 |
| + motor-heat penalty ×10 (`lib_v4_cool/27000`, 800 iterations) | 0 N | 4,187 N | 0.29 | 8 / 14 | 0.89 / 1.07 |

- **Load moves; it doesn't disappear.** Penalising the stops alone pushed the work into the 10 N·m ankle motors.
  Scoring each motor against its own rating (`--thermal_penalty=-0.002`) pulled it back and lowered the average load
  on almost every motor (knees about 0.35× rated).
- **Still open:**
  - the ankle X8 Pros and the right elbow X8 Pro reach about 1.8–1.9× rated on the hardest motion;
  - rare hard landings remain: one 4.2 kN spike in walk-and-turn, while typical ankle-pin loads are unchanged at about 0.6 kN (p99);
  - the motion score dipped slightly.
- **From scratch** with the stricter tracking objective, the library tracker still fell within about 1 s after
  2,000 iterations, so it was stopped.

**Knee elastic** (`--knee_band 150,15`: 150 N·m/rad on the knee crank past 15°), squat and terrain walker A/B:
- **Squat:**
  - at 2,000 iterations the elastic pushed with the knee motor 80 % of the time and allowed a deeper bend, but the policy hopped (both feet up to 15 cm);
  - at 3,400 iterations it no longer hopped, but the knee motor worked more with the elastic (35 vs 30 N·m RMS);
  - neither policy squats (torso drop 8 and 7 cm vs 18 cm).
- **Walking:** the swing knee bends as far as a squat (25–27° of crank), so the elastic fights it and the knee motor
  works 32 % harder (39 vs 30 N·m RMS).
- **Conclusion:** a fixed-angle band is ruled out. A load-triggered elastic (engaged only in stance) is untested.

**Ankle** (calibrated linkage map; the tie-rod closures hold to under 0.25 mm):
- **Capacity:** at their 10 N·m rating, the two X8 Pros give the ankle about 23 N·m front/back (about 57 N·m at peak).
- **Demand:** the motions ask about 18 N·m typical and 52 N·m on the hardest 1 %.
- **Why motor B works harder:** motors A and B have the same leverage within 3 %. The robot presses steadily on the inner edges of its feet, a 7–8 N·m sideways torque per ankle, which adds to B and subtracts from A.
- **Estimated effect of fixes:**
  - removing that pressing would bring right B from 11.2 to 9.2 N·m RMS;
  - a crank 25 % shorter (40 → 30 mm) would lower every ankle motor by about 25 %.
- Chart: `logs/reports/s7_ankle.png`.

## 2026-10-19: push and terrain baseline (gate 4)

`scripts/robustness_bench.py` + `tools/plot_robustness.py` on the session-7 Build B walker
(`vel_rough_v4_safe/10950`, blind, never trained with pushes, stop model on). Chart:
`logs/reports/bench_baseline_v4safe.png`; data: `logs/bench/{push,terrain}_v4safe_10950.json`.

- **Pushes:**
  - Setup: a horizontal shove on the pelvis for 0.1 s, sized as the speed change it would give the whole robot. Eight directions, 2 gait timings each, survival = no fall within 5 s.
  - Standing: every direction survives 0.2 m/s, and most survive 0.6. Weakest from behind and behind-left (0.2 m/s); strongest from the front (1.2).
  - Walking at 0.5 m/s: every direction survives 0.4 m/s. Weakest from the right (0.4); 1.2 from the front and from behind.
- **Terrain:**
  - Setup: walk 0.5 m/s for 8 s from each tile's centre, one attempt per cell.
  - Fine on flat ground, and downhill to 27°.
  - Uphill to 18°, though slow above 9°.
  - Stairs up and down to 8–11 cm steps; falls at 11–14 cm.
  - Rough blocks: mixed above 5 cm.
- Every result is a single deterministic run per cell, so neighbouring cells can disagree (rough 5–6.5 cm fell, 6.5–8 cm walked).
