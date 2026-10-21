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
- In the motion library that hip roll sat at its 40 N·m limit. One-leg stance measured later in sim (session-6 Build B specialist) needs a median of 31 N·m, p95 43 and max 53 N·m at the hip roll. The 56 N·m quoted here before was a static estimate with the weight centred between the feet.
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

## Cloud session 8 (2026-10-20): pushes and harder terrain

4× L40 (hyperstack), about 8.5 h, $42. Build B, stop model and safety penalties on every job. The walkers trained with
`--push_velocity` (each push a random velocity change of up to ± that per horizontal axis, every 4–8 s) and harder
terrain (`--stair_max_m=0.15 --slope_max=0.35`). Scored with `scripts/robustness_bench.py`. "Weakest side" = the
strongest push survived from every direction. Terrain counts a cell only if the robot did not fall AND covered at
least half of the commanded 4 m. Charts: `logs/reports/s8_push_final.png`, `logs/reports/bench_vel_*_16050.png`.

| Walker | Standing: weakest side / 1.0 m/s shoves survived | Walking: weakest side / 1.0 m/s / 1.2 m/s | Flat, 4 m asked | Terrain cells | Stairs up / down |
|---|---|---|---:|---:|---|
| Before (`vel_rough_v4_safe/10950`) | 0.2 m/s / 4 of 16 | 0.4 m/s / 9 / 4 of 16 | 3.7 m | 23 / 36 | 8–11 / 8–11 cm |
| Pushes ≤0.6 (`vel_push_v4/16050`) | **0.8 m/s / 13 of 16** | **0.8 m/s / 12 / 7 of 16** | 3.7 m | 23 / 36 | 5–8 / 5–8 cm |
| Pushes ≤1.0 + motor-heat ×10 (`vel_push_hard_v4/16050`) | 0.8 m/s / 15 of 16 | 0.8 m/s / 14 / 7 of 16 | 2.4–2.8 m | 21 / 36 | none / 14–17 cm |
| From scratch (`vel_scratch_v4/4800`) | 0.6 m/s / 5 of 16 | 0.6 m/s / 0 / 0 of 16 | 2.7 m | 14 / 36 | none / 5–8 cm |

- **Pushes ≤0.6 is the new default walker.** Its weakest side is 4× stronger standing and 2× walking. It keeps its
  walking speed and overall terrain score; stairs up dropped one band (5–8 cm instead of 8–11).
- **Pushes ≤1.0 is the toughest against shoves but became cautious.** It walks at 60–70 % of the commanded speed and
  stands at the foot of stairs instead of climbing them, though it walks down 14–17 cm steps.
- **From scratch needs far more than 4,800 iterations.**
- **Stairs remain the gap.** A blind walker has to feel each step. A teacher that sees the ground (height scan),
  distilled into a student, plus more knee travel, is the path past about 10 cm.

**Tracker `lib_v4_cool2/30600`** (motor-heat recipe, about 3,600 more iterations): the hardware gate passes on 11 of 14
motions, the best yet (8 at 27,000; 6 before the safety work). But the motion score fell to 0.25 (from 0.29 and 0.31).
At ×10 the motor-heat term makes it track lazily, so it needs a lighter weight or fidelity-based motion sampling.

## Cloud session 9 (2026-10-20/21): arms for balance

4× RTX 6000 Ada, about 12 h including about 3 h of overrun while this PC slept (estimated $56). All three walkers
continued `vel_push_v4/16050` with the same pushes, terrain and safety terms to iteration 22,200:
- control: arms held as before;
- `--arm_action_scale=0.3 --arm_deviation_weight=-0.02` (arms free);
- the same on `hw_v4_arms` (X8-120 at shoulder pitch and roll, +1.9 kg).

Push survival is the mean of 3 runs of `scripts/robustness_bench.py`. One run varies by ±3–5 points
(`logs/bench/s9_push_repeats.json`, chart `logs/reports/s9_arms_final.png`, video `logs/reels/dropbear_s9_arms.mp4`).

| Walker | Standing: all / 1.0 / 1.2 m/s | Walking: all / 1.0 / 1.2 m/s | Terrain cells | Busiest arm motor while walking (× rated) |
|---|---|---|---:|---|
| Arms held (`vel_ctrl_v4`) | 85 / 75 / 38 % | 91 / 90 / 58 % | 23 / 36 | ≤ 0.13 |
| Arms free, stock motors (`vel_arms_v4`) | 78 / 65 / 21 % | 91 / 90 / 56 % | 24 / 36 | right shoulder roll (X8 Pro) 1.02, left elbow 0.79 |
| **Arms free, X8-120 shoulders (`vel_arms_x8_v4`)** | **90 / 88 / 60 %** | **97 / 94 / 85 %** | **26 / 36** | left elbow (X8 Pro) 0.65; shoulders ≤ 0.24 |

- **Stronger shoulders are what make the arms useful.** With X8-120s, walking survives 85 % of the hardest shoves
  (control 58 %), and standing and terrain improve too.
- **On the stock shoulder motors, free arms don't help.** Standing gets worse, and the X8 Pro shoulder roll runs at its
  rated torque.
- **How the arms are used:** the policy holds one arm out sideways like a tightrope walker. It does not use a human
  counter-swing; natural arm swing would need its own reward or a reference gait.
- **Tracker** `lib_v4_cool3/35000` (motor-heat -5e-4): motion score 0.27, hardware gate 3 / 14. The lighter heat
  penalty lost the gate without recovering motion quality.

## Cloud session 10 (2026-10-21): natural arm swing

4× L40S. Brev removed the instance about 20:00, so training ran about 5.5 h (iteration ~25,300 per job, last pull
19:56). New walker terms (`enable_arm_swing`):
- `arm_swing_counter_phase`: each shoulder pitch follows 0.45× the opposite hip pitch while walking. The ratio and signs
  come from the retargeted human clips: correlation +0.78..+0.93, k 0.40-0.55.
- An L1 pull of the shoulder rolls back to the stand pose.

Push survival is over 3 repeats per cell (`--repeats 3`); session 9 values are the means of 3 runs. Video:
`logs/reels/dropbear_s10_arm_swing.mp4`.

| Walker | Standing / walking | Hardest shoves (1.2 m/s), walking / standing | Terrain | Arm swing vs opposite hip (corr, swing range R/L) | Shoulder out to the side |
|---|---|---|---:|---|---|
| Session 9, arms held | 85 / 91 % | 58 / 38 % | 23 | none | 1-2° |
| Session 9, X8-120, arms free | 90 / 97 % | 85 / 60 % | 26 | +0.52 / +0.39, 7 / 28° | **89°** (arm held out) |
| `vel_swing_x8` (abduction -0.3) | 83 / 90 % | 60 / 35 % | 26 | **+0.98 / +0.99, 13 / 14°** | 5-8° |
| `vel_swing_x8_lite` (abduction -0.1) | 82 / 91 % | 52 / 19 % | 24 | +0.99 / +0.99, 13 / 17° | 1-8° |
| `vel_swing_x8_heavy` (65.2 kg) | 84 / 88 % | 38 / 35 % | 23 | +0.99 / +0.99, 15 / 16° | 4-8° |
| `vel_swing_v4` (stock shoulders) | 87 / 94 % | 62 / 48 % | 26 | +0.57 / +0.38, 14 / 32° | 9° (one arm lifted forward) |

- **The swing term works on the X8-120 shoulders.** The arms now move in counter-phase with the legs (correlation
  0.98-0.99) at about half the hip range, and stay at the sides. The arm motors are cool: elbow 0.14× rated.
- **The cost:** the extra push survival came from the held-out arm. With natural arms, survival is back to the
  arms-held level. Next: gate the arm-style terms off for a moment after a disturbance (natural when calm, free when
  shoved).
- **The 65 kg robot** is clearly weaker against the hardest walking shoves (38 % vs 60 %). The real mass matters; weigh
  the robot.
- **Stock shoulders + swing:** the swing is only half learned. One arm is lifted forward and the left elbow runs at
  0.84× rated.

## 2026-10-22: adversarial motion prior, first local test

The session-10 swing looked forced. Measured against the retargeted human walking clips, it is:

| | Arm vs opposite hip (corr) | Swing range | Elbow range | Shoulder out to the side |
|---|---|---|---|---|
| Human clips (walk_normal, walk_brisk, LAFAN1 walk) | 0.43-0.89 | 19-81° | 5-24° | 13-19° |
| `vel_swing_x8` (session 10) | 0.99 / 1.00 | 14 / 15° | **< 1°** | 5-9° |

The hand-written rule locks the shoulders to the legs and leaves the elbows straight. The new approach is an
adversarial motion prior (`--amp_weight`, `tasks/locomotion/amp.py`). A discriminator learns to tell the robot's
50 Hz state transitions from those of 30 retargeted human clips (walk, turn, stop, back, step, single-foot balance;
210 s). The policy is rewarded for motion it takes for human. No arm rule is set by hand.

Local runs on the RTX 4080 laptop (512 envs, 50-iteration chunks with cooldowns; Brev balance was $0):
- **`amp_local1`, from session 9:** failed. That walker holds one arm out at about 90°. The discriminator separated it
  on arm posture alone (a linear probe on arm angles: 100 %), its score went to -0.95 and the style reward to 0.07.
  The policy got almost no signal. Wrist-roll speed also jittered at about 20× the clips', so the wrist rolls were taken
  out of the features (d91224f).
- **`amp_local2`, from session 10** (arms already at the sides): 490 iterations. The style reward held at 0.17-0.33 and
  velocity tracking at 0.75-0.83 with almost no falls. From chunk 6 the discriminator's loss rose (0.34 → 0.38): the
  policy was starting to fool it.

| Walking 0.5 m/s | Arm vs opposite hip (corr R / L) | Swing range R / L | Elbow range | Shoulder out L / R |
|---|---|---|---|---|
| `vel_swing_x8` (start) | 0.99 / 1.00 | 14 / 15° | < 1° | 9 / 5° |
| `amp_local2` (+490 it) | 0.98 / 0.97 | 14 / 19° | < 1° | 9 / **29°** |

- **What this shows:** AMP trains and moves the arms toward the clips. The right shoulder rolled out into (past) the
  human range, and the left swing ratio rose 0.42 → 0.56.
- **What it doesn't show:** a more natural walk. 490 iterations at 512 envs is far too short. The elbows are still
  straight, the coupling is still locked, and the right arm overshoots. Neither walker survives a 1.2 m/s standing
  side shove in the filmed case.
- **Left/right:** only one arm has moved so far. The discriminator still finds the left arm frozen (shoulder roll, elbow
  and hand at 10-20 % of the human variation). The mirror-symmetry augmentation (Brief's 0022) should carry what the
  policy learns on one side to the other.
- **Next:** cloud session 11 (`tools/brev/run_session11.sh`). Four style weights / builds from the session-10 walkers,
  with symmetry, at 4096 envs for several thousand iterations.

Video: `logs/reels/dropbear_amp_local.mp4`.

**Continued (chunks 11-22, with `--symmetry`), stopped at iteration 26,407:**
- **Symmetry made it worse at first.** The rsl-rl symmetry loss rose 2.3 → 6.4, and the style reward fell 0.17 → 0.10.
  - Cause: the discriminator was one-sided, so the style reward wasn't mirror symmetric, which the augmentation
    assumes.
  - First, the 22-motor mirror map itself was checked against data (session-10 walking telemetry and three human
    walking clips at a half-gait shift). It is right for every joint in the features.
  - Fixed in c06416a: under `--symmetry`, the discriminator learns from mirrored clips and robot transitions, and the
    reward averages a transition and its mirror. The robot's midline is y = −0.0698 m in the root frame.
- **Arms at 26,407:** right shoulder 17° from the side (the human clips: 13-19°; at 490 iterations it overshot to
  29°). Arm coupling corr 0.94-0.96 (was 0.99). The elbows are still straight.
- **Push and terrain benchmark** (3 repeats; terrain cells need no fall and half the distance):

| Walker | Standing / walking | 1.2 m/s, walking / standing | Terrain |
|---|---|---|---:|
| `vel_swing_x8` (start) | 83 / 90 % | 60 / 35 % | 26 |
| `amp_local2/26407` | 77 / 91 % | 60 / **17 %** | 26 (better downhill and on stairs down, worse up stairs) |

  - The filmed walking shove that the AMP walker survived (`logs/reels/dropbear_amp_local2.mp4`) was a lucky case:
    walking survival is unchanged.
  - Standing push recovery is worse. The 30 reference clips show no reaction to a shove, so the style reward penalizes
    the recovery moves the old walker used. Push-recovery reference clips (`data/motions_gen/prompts/push_recovery_v1.json`,
    text-to-motion) are being built for that.
