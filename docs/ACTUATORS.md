# Dropbear actuator model from the datasheets (Phase 1a, PROPOSAL)

**Status (2026-09-25 midday):**
- Sections 0-10 are the original proposal.
- **Section 11 is now implemented.** The velocity task has two opt-in real-actuator profiles, `hw_v1` (the user's motor
  map) and `hw_v1_cad` (the CAD map). They are first results, and training defaults are unchanged.
- **The user's answers override section 1:**
  - hip roll is an **X10 1:7**, not an X10-S2;
  - the knee is an **EPS-CEM-60**, not an X10-S2;
  - "Waist Pivot" is the hip-pitch X10-S2;
  - the robot runs at 48 V.

- **Machine-readable model:** `data/robot/actuators_datasheet_v1.json` (schema `dropbear-actuators-datasheet-v1`). It
  lists every number with its source file, page and SHA-256, the joint -> motor evidence, and the derived Isaac Lab
  settings.
- **Generator:** `tools/build_actuator_datasheet.py`. It needs numpy only and does not open the USD.
  - Log: `logs/actuators/build_actuator_datasheet_v1.log`.
  - Run: `python tools/build_actuator_datasheet.py --motions > logs/actuators/build_actuator_datasheet_v1.log`.
- **CAD evidence:** `tools/probe_usd_motor_prims.py`, which opens the USD read-only and checks SHA `45586414`.
  - Output: `logs/actuators/usd_motor_prims.log` and `.json`.
- **Sources:** all are read-only. They are the `github.com/Hyperspawn/myactuator-can` PDFs and CSVs, the vendor PDFs under
  `dropbear_control/assets/vendor/myactuator/docs/RMD-X`, the embodiment JSON, the BOM and `dropbear_docs`.
  - Page numbers below are PDF page indices.
  - The short names (X10_100, PROTO_V39, ...) are the `sources` keys of the JSON.

## 0. Summary

All values are at the motor output and 48 V. The same numbers apply to the left and the right joint.

| motor joint | model | ratio | peak / rated [N*m] | no-load (gate) [rad/s] | armature [kg*m^2] | sim effort now [N*m] |
|---|---|---:|---|---|---:|---|
| `PG_*_leg_pitch` (hip roll) | RMD-X10-S2 V3 | 35 | 100 / 50 | 5.61 (5.25) | 0.695 | 200 |
| `PG_*_leg_roll` (hip yaw) | RMD-X10 V3 1:7 | 7 | 40 / 15 | 19.5 (17.3) | 0.0278 | 200 |
| `*L_hip_joint` (hip pitch) | RMD-X10-S2 V3 | 35 | 100 / 50 | 5.61 (5.25) | 0.695 | 200 |
| `*L_knee_actuator_joint` (knee crank) | RMD-X10-S2 V3 | 35 | 100 / 50 | 5.61 (5.25) | 0.695 | 300 legacy / 100 `stiff_knee_hw` |
| `*L_Revolute67/81` (calf) | RMD-X8 Pro 1:9 V2 | 9 | 25 / 10 | 16.8 (13.7), or 26.6 if X8-25 | 0.0275 | 80 |
| `*H_yaw` (shoulder pitch) | RMD-X10 V3 1:7 | 7 | 40 / 15 | 19.5 (17.3) | 0.0278 | 40 |
| `*H_pitch`, `_roll`, `_elbow_joint`, `_wrist_roll` | RMD-X8 Pro 1:9 V2 | 9 | 25 / 10 | 16.8 (13.7) | 0.0275 | 40 |

The sim's armature is 0.01 everywhere, and its velocity cap is 10 rad/s everywhere (from the USD).

The biggest gaps:

- **Armature.** The six X10-S2 joints have 70x the sim's armature.
- **Torque.** The sim allows 2-5x the peak torque at the hips, 3.2x at the calves and 3x at the legacy knee.
- **Speed.** The X10-S2 cannot reach the sim's 10 rad/s: it stops at 5.6 rad/s. 12 of the 80 accepted clips and two
  trained policies exceed that speed, and 15 clips exceed the 5.25 rad/s gate.
- **Friction.** The sim has none; the X10-S2 needs 2.9 N*m to back-drive.
- **Motion-mode limits.** The motors' impedance mode caps kp at 500, kd at 5 and feed-forward torque at 24 N*m.

## 1. Which motor drives which joint

The CAD export keeps the MyActuator component names inside the rigid bodies they were merged into. A motor joint is
driven by the motor whose stator is in one of its bodies and whose rotor is in the other. For example, the pelvis
`world` holds `PG_RMD_X10_S2_MIR4__2_Stator_1`, and the child of `PG_left_leg_pitch` *is*
`PG_RMD_X10_S2_MIR4__2_Rotor_1` (`logs/actuators/usd_motor_prims.log`). This is the strongest evidence, and it wins the
conflicts.

| motor joint (L / R) | role | model | ratio | evidence (agrees / disagrees) |
|---|---|---|---:|---|
| `PG_left/right_leg_pitch` | hip roll | **RMD-X10-S2 V3** (X10-100) | 35 | USD X10-S2 stator/rotor pair; embodiment JSON. Disagrees: CSV and render "Hip Spreader = X10 1:7" (Q5) |
| `PG_left/right_leg_roll` | hip yaw | **RMD-X10 V3 1:7** (X10-40) | 7 | USD `PG_RMD_X10_V3Stator/Rotor`; CSV "Leg Rotator"; README count. Disagrees: embodiment (X10-S2) |
| `LL/RL_hip_joint` | hip pitch | **RMD-X10-S2 V3** | 35 | USD: a second X10-S2 (`*_RMD_X10_S2_MIR4__3_Rotor` on the hip-yaw output, `__3_Stator` = thigh); CSV/render "Waist Pivot" (points at the upper-thigh discs). Disagrees: embodiment "RMD-X10", old docs "CEM-60" |
| `LL/RL_knee_actuator_joint` | knee crank | **RMD-X10-S2 V3** | 35 | USD (thigh holds the knee stator, child = rotor); CSV "Knee Bender"; embodiment |
| `LL/RL_Revolute67`, `_Revolute81` | calf motors (parallel ankle) | **RMD-X8 Pro 1:9 V2** (X8-25) | 9 | USD: 2 X8 Pro in the shank; CSV "Calf-Foot Flexor"; embodiment; BOM |
| `LH/RH_yaw` | shoulder pitch | **RMD-X10 V3 1:7** | 7 | USD `torso_RMD_X10Stator/Rotor` (no S2); CSV "Shoulder Rotator"; BOM; README. Disagrees: old docs "X10 S2" |
| `LH/RH_pitch`, `_roll`, `_elbow_joint`, `_wrist_roll` | shoulder abduction, arm roll, elbow, wrist | **RMD-X8 Pro 1:9 V2** | 9 | USD: 4 X8 Pro per arm (`__1__1`, `__3__1`, `__2__1` in the upper arm, `__4__1` in the forearm); CSV rows 9-11, 13 |

**Count check.** The robot has 6 X10-S2 (hip roll, hip pitch and knee on each side), 4 X10 1:7 (hip yaw and shoulder
pitch) and 12 X8 Pro, which makes 22. The myactuator-can README lists "RMD-X8-PRO-1:9 x 12, RMD-X10-1:7-V3 x 4" plus 6
CEM-60 for the six heavy joints of an older plan. The CAD has X10-S2 there.

**Where the CSV table comes from.** The joint table in Actuators1.csv (rows 8-18) was typed from
`myactuator-can/Actuator Config.png`, a "HyperBot (Dropbear)" callout render with the same names and descriptions. It
appeared in the folder during this session and has a file date of 2024-11. Its pointer lines show which motor each
name refers to:

- "Waist Pivot" -> the large discs at the top of both thighs;
- "Hip Spreader" -> the motor behind the pelvis;
- "Leg Rotator" -> the horizontal discs under the pelvis;
- "Knee Bender" -> the lower-thigh discs.

**Conflicts resolved (flagged in section 9):**

- **"Hip Spreader = X10 1:7" (CSV row 14 and the render).** The CAD (the USD of 2025-10) has an X10-S2 stator/rotor
  pair there. Physics agrees with the CAD: the single-support hip-roll moment (~557 N x ~0.1 m ~ 56 N*m) exceeds the
  X10 1:7 peak of 40. The 2024 render/CSV row is treated as stale. This is the one mapping to confirm (Q5). If it
  really is an X10 1:7, the two hip-roll joints get peak 40 N*m, armature 0.0278 and no-load 19.5 rad/s.
- **"Waist Pivot, RMD-X10-S2 1:35, torso forward/back bending" (CSV row 17).** The USD has no waist joint. The render's
  pointer lines end on the upper-thigh discs, which are the hip-pitch motors, and the CAD has an X10-S2 there. So
  "Waist Pivot" = the two hip-pitch motors (Q1 only confirms it).
- **The embodiment JSON (Codex overlay) looks swapped at the hip.** It lists hip yaw = X10-S2 and hip pitch =
  "RMD-X10". It also puts the elbow on `LH_Revolute41`, which belongs to Codex's other plant cut (DECISIONS
  2026-09-23). Its `runtimeParameters` are all "unverified".
- **X8 ratio.** The spec row of Actuators.csv / Actuators1.csv (row 4) is `RMD-X8-P6-20` = X8-20, "RMD-X8-Pro-H 1:6
  V3" (X Series manual p10). But the joint table, the model folder `X8-25`, DECISIONS and the BOM all say X8 Pro 1:9.
  1:9 is adopted, and the 1:6 data is kept in the JSON as `siblings_not_used`. (Q2)
- **CEM-60** (BOM, `dropbear_docs` legs, README) is an older plan. The corpus has no CEM-60 datasheet: the CEM manual
  in the new `myactuator-can/(CEM) Protocol and manual - 260520` folder covers CEM-15/25/45 only. (Q6)

## 2. Datasheet values per model

All values are at 48 V and at the actuator output shaft. "Rated" is the thermal continuous point: 24 C ambient,
60 K rise, no extra cooling, at rated speed (X10_100 p1 footnote).

| quantity | RMD-X10-S2 V3 1:35 (X10-100) | RMD-X10 V3 1:7 (X10-40) | RMD-X8 Pro 1:9 V2 (X8-25) |
|---|---|---|---|
| rated / peak torque [N*m] | 50 / 100 (X10_100 p1; profile p1 overload x2) | 15 / 40 (X10_40 p1; old profile: 12, x3 = 36) | 10 / 25 (X8_25 p1; SIMPLE p1: rated 13) |
| rated speed [rpm] | 50 | 165 (old profile: 170) | 110 (SIMPLE p1: 122) |
| no-load speed [rpm] | 53.6 (curve at 47.99 V; profile: 55) | 186.4 (curve; profile: 190) | **160 (SIMPLE p1, conservative)** vs 254.3 (X8_25 p1 curve), Q3 |
| rated / peak current [A] | 6.7 / 13.5 (6.7 A = DC input current at 50 N*m) | 6.5 / 15 | 3.2 / 8 (not the DC current: the curve draws ~6.5 A at 10 N*m) |
| torque constant | 0.32 N*m/A motor side (profile p1, suspect) -> output ~7.5 N*m per DC amp | 0.32 N*m/A motor side; ~2.3 N*m per DC amp | 2.6 N*m/A output (SIMPLE p1); X8 V2 non-Pro: 2.09 |
| rotor inertia [g*cm^2] | 5675 (profile p1) | 5675 (profile p1) | 3400 (SIMPLE p1) |
| **armature = J_rotor x N^2 [kg*m^2]** | **0.695** | **0.0278** | **0.0275** |
| back-drive torque [N*m] | 2.88 | 0.62 | 0.61 |
| backlash [arcmin] | 15 | 10 (profile 7, SIMPLE 8) | 10 |
| weight [g] | 1700 | 1150 | 710 |
| encoder | dual, input 16 bit / output 14 bit (241227 manual: 14/14) | dual 16/14 (14/14) | **single, motor side only** 16 bit (241227: 18 bit); no output encoder |
| resistance / inductance | 0.3 ohm / 0.13 mH (copied from the X10?) | 0.3 ohm / 0.13 mH | 0.54 ohm / 0.28 mH (Y) |
| pole pairs | 21 | 21 | 21 (SIMPLE: 20) |
| comms | CAN 1 Mbit/s (500 k selectable), RS485 up to 2.5 Mbit/s | same | same |

Page references:

- X10_100 = X10-100 product-parameters PDF p1, which is the same content as the X Series Product Manual 240403 p13.
- X10_40 = X10-40 PDF p1, same as the manual's p12.
- X8_25 = X8-25 PDF p1, same as the manual's p8.
- "profile" = `RMD-X10-S2 V3` / `RMD-X10 V3 - Actuator Profile.pdf` p1.
- SIMPLE = `Actuator Profile (Simple).pdf`: p1 X8 PRO, p2 X10-S2, p3 X10 1:7. Its "Reducer Ratio" cells are Excel time
  values: 0.3757 d = 9:01, i.e. 9:1.

Three findings from the datasheets:

- **The catalog "Inertia" field is the rotor inertia x ratio (not x ratio^2, and not an output inertia).**
  - The field reads 198.6 kg*cm^2 for the X10-S2, 39.7 for the X10 1:7, 30.6 for the X8-25 and 20 for the X8-20.
  - These equal 5675 x 35, 5675 x 7, 3400 x 9 and ~3400 x 6 g*cm^2 divided by 1000. Each motor family gets one
    consistent rotor.
  - Read as an output inertia instead, the same X10 motor would have a 162 g*cm^2 rotor in one product and an
    810 g*cm^2 rotor in the other.
  - The physical armature is therefore J_rotor x N^2 (gear-stage inertias are not published; add 10-30 %).
  - Do not use the CSV "Inertia (kg.cm2)" column as armature.
- **The X10-S2's motor constants are probably copy-pasted.** R, L, Kt, Kv and the rotor inertia are identical to the
  X10 1:7 sheet. But 53.6 rpm x 35 = 1876 motor rpm at 48 V (~39 rpm/V), which contradicts Kv = 30 rpm/V. The rotor
  inertia is still plausible, because a winding change does not change the rotor.
- **X8 generation.** The X8-25 catalog (254 rpm no-load) and the user's own "RMD-X8 PRO" summary (160 rpm no-load,
  Kv 30) share the weight (710 g) and the rotor (3400 g*cm^2), but the speeds differ by 60 %. The conservative 160 rpm
  is used until the label or a no-load test says otherwise (Q3, B2).

## 3. Derived actuator model

- **Torque-speed line at 48 V.** This is a straight line through the curve's No_Load and Max_Torque rows, which is
  exactly what an Isaac Lab `DCMotorCfg` models: `tau_max(w) = min(peak, sat * (1 - |w|/w_nl))`.

  | model | w_nl [rad/s] | saturation_effort [N*m] | w at peak [rad/s] | w at rated [rad/s] |
  |---|---:|---:|---:|---:|
  | X10-S2 1:35 | 5.61 | 768 | 4.88 | 5.25 |
  | X10 1:7 | 19.54 | 133 | 13.66 | 17.33 |
  | X8 Pro 1:9 (conservative) | 16.76 | 54.7 | 9.10 | 13.69 |
  | X8 Pro 1:9 (X8-25 curve) | 26.65 | 59.6 | 15.47 | 22.18 |

  The X8-25 curve is concave (1.7 rpm per N*m near 0, 6 rpm per N*m near 18 N*m), so the chord is optimistic above
  ~15 N*m.
- **Friction.** The sim has **0** on the motors: joint friction is zeroed because PhysX scales it by the constraint
  force (CONTRACTS 0.1). The datasheet back-drive torque is a Coulomb friction at the output: 2.88 N*m for the X10-S2,
  0.62 for the X10 1:7, 0.61 for the X8. It belongs in the actuator model (explicit term), not in PhysX `jointFriction`.
- **Backlash.** 15' on the X10-S2 is 4.4 mrad at the hip. That is ~4 mm at the foot, or 8 mrad at the knee through the
  four-bar. Model it as joint-position noise or a dead-band in randomization.
- **Thermal.** The standing pose's knee (~20 deg) needs ~75 N*m on each crank in double support (DECISIONS
  2026-09-24). That is 1.5 x the X10-S2's *rated* 50 N*m, so it is not sustainable continuously. At ~10 deg of knee
  it drops to ~36 N*m (0.7 x rated). A torque-RMS or thermal term in the reward or the gate is worth adding.
- **Voltage.** Every speed scales with the bus voltage (Q7).

## 4. What the linkages do to joint-side torque and speed

The ratios below come from the calibration LUT slopes and Jacobian at `standing_motor_pos`
(`dropbear_semantic_calibration.json`). Virtual work gives `tau_joint = tau_motor / G`, `w_joint = G * w_motor` and
`J_joint = J_motor / G^2`.

| linkage | G = d joint / d motor | joint-side peak (rated) torque | joint-side speed (no-load / gate) | armature at the joint |
|---|---|---|---|---|
| knee four-bar (X10-S2 on the crank) | 1.79 L / 1.82 R at the stand (1.36-2.21 over the range) | **55.7 (27.9) N*m** at the stand, 46-74 over the range | 10.1 / 9.4 rad/s | 0.216 kg*m^2 |
| parallel ankle (2 x X8 Pro) | J = d(pitch,roll)/d(A,B) = [[0.38, 0.54], [-0.29, 0.41]] (L) | pure pitch **46 (19)**, pure roll **61 (24)** N*m | pitch 12.9 / 10.5, roll 9.7 rad/s | pitch 0.071, roll 0.123 kg*m^2 (the foot's own inertia: 0.015 / 0.008) |
| elbow linkage (X8 Pro) | -4.8 L / -5.0 R at the stand (3.7-5.5) | **5.2 (2.1) N*m** at the elbow (4.5-6.8) | ~81 rad/s (never limiting) | 0.0012 kg*m^2 |

Consequences:

- The sim's legacy knee effort of 300 N*m is 167 N*m at the knee; the hardware has 56.
- The legacy ankle effort of 80 gives 148 / 196 N*m of pitch / roll against 46 / 61. With 46 N*m of ankle pitch,
  single-support CoP authority is about 46 / 557 N ~ 8 cm in front of or behind the ankle axis.
- The elbow is weak: 5 N*m holds the forearm and hand (~1.5 N*m of gravity) plus a small load. It is also fast.
- At the ankle, the two rotors reflected through the tie rods (0.07 / 0.12 kg*m^2 in pitch / roll) are 5-15x the
  foot's own inertia. The sim's 0.01 armature on the calf motors gives about a third of that.
- The link inertia each motor sees (rest pose; JSON `inertia.link_inertia_at_rest_kgm2`) shows **how much the rotor
  matters**:

  | joint | armature | link inertia | armature share of the total |
  |---|---:|---:|---:|
  | hip roll | 0.695 | 3.29 | 17 % |
  | hip pitch | 0.695 | 2.50 | 22 % |
  | knee crank | 0.695 | 0.73 (the four-bar reflects the shank x G^2) | 49 % |
  | calf motors | 0.0275 | 0.003-0.006 | 83-90 % |
  | wrist | 0.0275 | 0.0003 | 99 % |
  | elbow motor | 0.0275 | 0.74 (x G^2 = 23) | 4 % |

## 5. Sim vs datasheet (current sim = legacy `ACTUATOR_PARAMS`; velocity task = `stiff_knee_hw`)

| joint | sim effort [N*m] | datasheet peak | sim / peak | sim armature vs datasheet | sim velocity cap vs no-load |
|---|---:|---:|---:|---|---|
| hip roll `PG_*_leg_pitch` (X10-S2) | 200 | 100 | **2.0** | 0.01 vs 0.695 (**1/70**) | 10 vs 5.61 (**1.78x too fast**) |
| hip yaw `PG_*_leg_roll` (X10 1:7) | 200 | 40 | **5.0** | 0.01 vs 0.0278 (1/2.8) | 10 vs 19.5 (0.51, conservative) |
| hip pitch `*_hip_joint` (X10-S2) | 200 | 100 | **2.0** | **1/70** | **1.78x** |
| knee crank (X10-S2) | 300 legacy / 100 `stiff_knee_hw` | 100 | **3.0** / 1.0 | **1/70** | **1.78x** |
| calf A/B (X8 Pro) | 80 | 25 | **3.2** | 1/2.75 | 10 vs 16.8 (0.60) |
| shoulder pitch `*H_yaw` (X10 1:7) | 40 | 40 | 1.0 | 1/2.8 | 0.51 |
| shoulder abd / arm roll / elbow / wrist (X8 Pro) | 40 | 25 | **1.6** | 1/2.75 | 0.60 |

**Where the current policies are unrealistic.**

1. **Torque authority.** Every tracking policy trains on the legacy groups.
   - Hip yaw has 5x the hardware peak, the calf motors 3.2x, the knee 3x, hip roll and hip pitch 2x, and the arm X8s
     1.6x.
   - The velocity task (`stiff_knee_hw`) fixed only the knee effort. Its knee kp of 600 is above what the motors'
     motion mode accepts (kp <= 500, section 8), and its kd of 20 is too, unless the protocol kd acts on motor-side
     speed (B1).
   - The tabletop/teleop elbow kp of 600 is also above 500.
2. **Armature.** The six X10-S2 joints carry 0.01 instead of ~0.70 kg*m^2. The sim legs are much "lighter" to swing
   than the hardware: at the knee crank the rotor is half of the total inertia. This is the biggest dynamic gap after
   torque.
3. **Speed.** The USD caps every motor at 10 rad/s, but the X10-S2 cannot exceed 5.6 rad/s at 48 V (4.9 at peak
   torque). Recorded policy rollouts (`logs/brev/remote/*/logs/brev/eval/rollout_*.npz`, analysed in the JSON
   `motion_speed_analysis.policy_rollouts`) show:
   - `wave_v2`: knee crank up to **8.3 rad/s** (0.2-0.3 % of frames above the 5.25 gate);
   - `kimodo_walk_v4`: hip pitch 5.5 and knee 5.8 rad/s (0.8-1.9 % of frames above the gate);
   - `take102_v3`: hip pitch 4.97, just under the gate;
   - arm motors: they sit at the 10 rad/s USD cap (dance2, take102). That is below the X8/X10 capability, so it is
     conservative, not unrealistic.

   Applied torques are not in the rollout NPZs; recording `applied_torque` in play is the follow-up that quantifies
   item 1.
4. **Friction.** 0 in the sim. The X10-S2 needs 2.88 N*m to back-drive.

## 6. Proposed Isaac Lab settings (`hw_datasheet_v1`, NOT applied)

Per motor, as in the JSON `joints.<name>.isaac_lab_proposal`:

- **Actuator class.** An explicit `DCMotorCfg` with:
  - `effort_limit` = the peak torque;
  - `saturation_effort` and `velocity_limit` from the table in section 3;
  - `armature` = J_rotor x N^2.

  A small subclass should add a Coulomb friction `-f_c tanh(w / 0.05)` with `f_c` = the back-drive torque. Isaac Lab
  2.2 has no such term, and its `friction` field is a no-op on Isaac Sim 5.x (CONTRACTS 0.1).
- **`velocity_limit_sim`.** Set it to 2 x no-load, as a numerical safety net only. The torque-speed line does the
  physics, and external loads can back-drive a joint past its no-load speed.
- **Where the armature goes.** Put it on the motor joint itself: the knee crank, the calf driver and the elbow motor.
  PhysX then reflects it through the linkages correctly.
- **Gains: TODO.** An "8-10 Hz from armature + link inertia" rule is not reachable on the heavy joints. At
  8 Hz it asks for:
  - kp 10,000 at hip roll;
  - kp 8,000 at hip pitch;
  - kp 3,600 at the knee;
  - kp 1,000-2,000 at the shoulders and elbows.

  The motors' motion mode caps kp at 500, and a 25 N*m X8 at kp 1,000 saturates at 1.4 deg of error. The BeyondMimic
  rule (armature only, 10 Hz, zeta 2) gives kp 2,744 / kd 175 for the X10-S2, so it is not transferable. The JSON
  gives a starting point, `suggested_start`: `kp = min(J_eff (2 pi 8)^2, 500, peak / 0.25 rad)`, `kd = 2 sqrt(kp J_eff)`.

  | joint | J_eff [kg*m^2] | kp start | kd (zeta 1) | f_n [Hz] | sim kp / kd now |
  |---|---:|---:|---:|---:|---|
  | hip roll | 3.99 | 400 | 80 | 1.6 | 150 / 5 |
  | hip yaw | 0.20 | 160 | 11 | 4.5 | 150 / 5 |
  | hip pitch | 3.20 | 400 | 72 | 1.8 | 150 / 5 |
  | knee crank | 1.43 | 400 | 48 | 2.7 | 200 / 12 (600 / 20) |
  | calf A / B | 0.031 / 0.033 | 77 / 84 | 3.1 / 3.4 | 8.0 | 80 / 4 |
  | shoulder pitch (X10) | 0.40 | 160 | 16 | 3.2 | 50 / 2 |
  | shoulder abd (X8) | 0.40 | 100 | 13 | 2.5 | 50 / 2 |
  | arm roll | 0.032 | 82 | 3.3 | 8.0 | 50 / 2 |
  | elbow motor | 0.77 | 100 | 18 | 1.8 | 50 / 2 |
  | wrist | 0.028 | 70 | 2.8 | 8.0 | 50 / 2 |

  The kd column is the critical issue. If the protocol's kd multiplies *output* speed, the motion mode tops out at kd 5,
  so the heavy joints would run at zeta ~0.06-0.1: badly underdamped, with only gearbox friction to help. If it
  multiplies *motor-side* speed (as the setup-software manual says), 5 x ratio = 175 (X10-S2) / 45 (X8) is available.
  Bench test B1 decides this before any gain is chosen. The action scale (`0.25 * effort / kp`) then follows. For the
  X10-S2 joints it would be 0.0625 rad instead of today's 0.333 / 0.375.
- **Also to change later.** `sdk.motors.EFFORT_LIMIT`, which the Newton bridge still clips at 300 N*m for the knee.

## 7. Motion-quality gate: per-motor speed limits (currently a blanket 10 rad/s + 0.3 rad per frame)

Proposed gate = the speed at which the 48 V drive still delivers its **rated** torque, taken from the torque-speed
line. Clips are rejected above the no-load speed. The values are in motor space, which is what
`settle.quality.motor_dynamics` checks.

| motors | gate max abs(q_dot) [rad/s] | max step per 50 Hz frame [rad] | hard reject (no-load) [rad/s] |
|---|---:|---:|---:|
| X10-S2: hip roll, hip pitch, knee crank (6) | **5.25** | 0.105 | 5.61 |
| X10 1:7: hip yaw, shoulder pitch (4) | **17.3** | 0.35 | 19.5 |
| X8 Pro: calf A/B, shoulder abd, arm roll, elbow motor, wrist (12) | **13.7** (22.2 if X8-25 is confirmed) | 0.27 | 16.8 (26.6) |

**Impact on the library.** 65 of the 80 accepted v4 clips pass. All 15 failures are on X10-S2 joints: hip pitch in 9
clips per side, knee in 8 (L) / 6 (R), hip roll in 1. 12 of them exceed even the no-load speed, for example G1
gangnam, kimodo walk, the LAFAN dance2 / run1 / jumps1 and `lafan1_walk1` (right knee in 0.3 % of frames). Per-clip
detail is in the JSON under `motion_speed_analysis.reference_clips`.

The retarget's 0.18 rad/frame motor-step projection (9 rad/s) should become per-motor too: 0.105 rad/frame for the
X10-S2 joints.

## 8. Control interface and CAN timing

**Protocol facts** (the RMD-X V3.9 protocol applies to both the V2 X8 and the V3 X10s; V4.2 is identical in the parts
below):

- **Bus.** CAN 2.0A, 8-byte frames, 1 Mbit/s (PROTO_V39 p7; 500 k via 0xB4, p77). Commands go on `0x140+ID`, replies on
  `0x240+ID`, with ID 1-32. `0x280` broadcasts the **same** payload to every motor (p85), so it cannot carry
  per-motor setpoints.
- **Servo commands.** 0xA1 sets torque in iq units (0.01 A, p45-46), 0xA2 speed (0.01 dps, p49), 0xA4 absolute position
  (0.01 deg + max speed, p52-53). Their reply carries position at **1 deg/LSB** (p46), which is useless as a policy
  observation. A 0x92 read would add another frame per motor.
- **MIT-style impedance mode exists: "Motion Mode", `0x400+ID`, reply `0x500+ID`** (PROTO_V39 p89-92; PROTO_V42
  p92-93).
  - The law is `IqRef = [kp (p_des - p) + kd (v_des - v) + t_ff] * KT`.
  - Ranges: p_des +/-12.5 rad (16 bit), v_des +/-45 rad/s (12 bit), **kp 0-500**, **kd 0-5**, **t_ff +/-24 N*m**
    (12 bit each).
  - The reply returns p (16 bit), v (12 bit, 0.022 rad/s) and t (12 bit, **saturating at 24 N*m**, a quarter of the
    X10-S2 peak).
  - In V4.4 (X-V4 hardware only, PROTO_V44 p161-162), t_ff scales to each motor's max torque, and kp is 0-500 in
    Chinese vs 0-1000 in English.
  - The setup-software manual (DBG_SW p5-6) says v_des and the speed feedback are the **motor-end** speed. With
    +/-45 rad/s that would mean only 1.3 rad/s of output range on the X10-S2, which argues for output-side units on
    V3.9. This is unresolved (B1).
- **Drive loops.** Current 15 kHz, speed 5 kHz, position 1 kHz (MC_DRIVER p7). So there is no point commanding faster
  than 1 kHz. The vendor states no maximum command rate; the bus sets it.
- **Communication-loss protection.** 0xB3 takes a timeout in ms in DATA[4..7]; 0 disables it (PROTO_V39 p74-75). The
  README sketch's `setWatchdog()` writes the value into DATA[2..3], so **protection is disabled** on the current
  firmware sketch. Fix it before any powered test.
- **Periodic replies.** 0xB6 active reply is at most 100 Hz (10 ms units, p80-81).

**Timing.** Each motor needs one command and one reply frame per cycle. An 8-byte standard frame at 1 Mbit/s takes
111-135 us, depending on bit stuffing.

| layout (source) | motors per bus | worst-case cycle | max loop rate at 70 % bus load | bus load at 200 Hz / 500 Hz |
|---|---|---:|---:|---|
| L1: per limb, 5 buses (README architecture) | 5 / 5 / 4 / 4 / 4 | 1.35 ms | **518 Hz** | 27 % / 68 % |
| L2: 4 buses (BOM: 4 x MCP2515; legs share one) | 5 / 5 / 4 / 8 | 2.16 ms | **324 Hz** | 43 % / 108 % |
| L3: legs 12 + arms 10 (embodiment ids 0x141-0x14C, README sketch) | 12 / 10 | 3.24 ms | **216 Hz** | 65 % / 162 % |
| L4: all 22 on one bus | 22 | 5.94 ms | **118 Hz** | 119 % / 297 % |

- The 50 Hz policy fits every layout (6-30 % load).
- The SDK contract's 500 Hz state rate fits only L1, and even there at 68 % load.
- The per-cycle skew between the first and the last motor on a bus is up to one cycle (1.3-5.9 ms). Add 0-10 ms of
  action/observation latency to randomization.
- Not included in these numbers: the motors' reply turnaround (undocumented; B4) and the ESP32 + MCP2515 overhead. The
  MCP2515 is an SPI controller with 2 RX buffers.
- **Risk to verify (not from the corpus).** The README's MCP2515 modules use an **8 MHz crystal at 1 Mbit/s**. By the
  MCP2515's bit-timing rules (at least 5 time quanta per bit), that needs at least a 10 MHz clock, so the setting is
  outside the chip's timing rules. Check the error counters on the real harness, or use 16 MHz modules, the ESP32's
  native TWAI, or a USB/SocketCAN adapter.

## 9. Open questions for the user (yes/no or pick one)

- **Q1.** Is the CSV / render "Waist Pivot (RMD-X10-S2 1:35)" the two **hip-pitch** motors at the top of the thighs,
  with no separate waist joint on the robot? yes / no
- **Q2.** Are all 12 X8 motors **1:9** (X8 Pro, "X8-25"), and none of them the 1:6 "X8-P6-20" from the CSV spec row?
  yes / no
- **Q3.** What does the X8 label say?
  - (a) "X8-25" or "RMD-X8-P9-25";
  - (b) "RMD-X8 PRO V2" (older, 160 rpm no-load);
  - (c) it has no "Pro" (568 g X8 V2);
  - (d) don't know.
- **Q4.** Is the shoulder-pitch motor (`LH_yaw`/`RH_yaw`) an **X10 1:7**, and not an X10-S2? yes / no
- **Q5 (most important).** Is the motor behind the pelvis that spreads the leg (hip roll, `PG_*_leg_pitch`) an
  **X10-S2 (74 mm thick, 1.7 kg)**, as in the CAD, or an **X10 1:7 (53 mm, 1.15 kg)**, as in the render/CSV
  "Hip Spreader"? X10-S2 / X10 1:7
- **Q6.** Is any **CEM-60** actuator on the robot today? yes / no
- **Q7.** Motor supply voltage?
  - (a) 48 V bench supply;
  - (b) 12S LiPo (42-50 V);
  - (c) 24 V;
  - (d) other.
- **Q8.** CAN layout and adapter?
  - (a) 5 buses as in the README;
  - (b) 4 buses (BOM);
  - (c) legs 12 + arms 10;
  - (d) one bus.

  Also: ESP32 + MCP2515 (8 MHz), or another adapter?
- **Q9.** Neck: NEMA 17 + lead screw, and what is the lead in mm/rev? This only matters if the neck will ever move.
- **Q10.** Is deploying with **Motion Mode (0x400)** acceptable, instead of the 0xA4 position mode the README uses?
  yes / no
- **Q11.** Motor firmware / protocol version? The setup software shows it, or command 0xB2.

**Bench tests that settle the rest** (one motor at a time, free output, 0xB3 watchdog set):

- **B1.** Motion mode with kp = 0, kd = 1, v_des = 1 rad/s. If the output settles at 1 rad/s, kd/v are output-side; if
  at 1/N, they are motor-side. Also command t_ff = 10 N*m against a lever and read iq to confirm the 24 N*m scaling.
- **B2.** No-load speed of each model at the robot's voltage (0xA2 to a high target, read the speed).
- **B3.** Back-drive torque of each model (motor off, luggage scale on a lever). Datasheet: 2.88 / 0.62 / 0.61 N*m.
- **B4.** 0x400 -> 0x500 round-trip time per motor, and the CAN error counters on the real harness at 1 Mbit/s.
- **B5 (optional).** A 0xA1 current step on a free motor gives the acceleration, hence J_rotor, to check
  5675 / 3400 g*cm^2.

## 10. Next step (after the answers)

Add the `hw_datasheet_v1` actuator profile: DCMotor, armature, friction, per-motor velocity and effort limits, and gains
from B1. Then:

- make the motion gate and the retarget step projection per-motor (section 7);
- log applied torques in play;
- A/B the profile against `stiff_knee_hw` on the velocity task.

None of this is done here. (Section 11 does the profile, the torque logging and the A/B.)

## 11. Twin v2 applied: the `hw_v1` / `hw_v1_cad` profiles (2026-09-25)

### Code

- **`robots/hw_motor_specs.py`** (pure Python):
  - `MOTOR_MODELS` (the numbers below);
  - `JOINT_MOTOR_MAPS`, which holds `user_2026_09_25` (from `myactuator-can/AttemptToExplainActuators.md`) and
    `cad_usd`;
  - the role gains `HW_GAINS`, checked against the motion-mode range kp ≤ 500, kd ≤ 5;
  - latency 0-2 physics steps (0-10 ms) and a friction scale of 0.5-1.5 per episode.
- **`robots/hw_actuators.py`:** `DatasheetMotor`, an explicit PD at the 200 Hz physics rate with:
  - a per-joint DC torque-speed envelope;
  - Coulomb plus viscous friction, applied after the envelope;
  - per-episode delay of the position target.

  `applied_effort` is the motor torque.
- **`tasks/locomotion/config/dropbear/flat_env_cfg.py`:** `HW_ACTUATOR_PROFILES`. `set_actuator_profile("hw_v1")`
  replaces the four legacy motor groups with one `hw_<model>` group per motor model. Neck and passive groups are
  unchanged.
- **`scripts/play_locomotion.py`:** now writes `motor_torque.per_motor`: max, p95, RMS, the share of time clipped, and
  joint speed.
- **Export:** it now reads the explicit models' own kp, kd and peak torque, because the solver gains are 0 for them.

### Models

| model | ratio | peak / rated [N*m] | no-load [rad/s] | saturation [N*m] | armature [kg*m^2] | Coulomb [N*m] | status |
|---|---:|---|---:|---:|---:|---:|---|
| RMD-X10-S2 V3 | 35 | 100 / 50 | 5.61 | 768 | 0.695 | 2.88 | datasheet |
| RMD-X10 V3 1:7 | 7 | 40 / 15 | 19.5 | 133 | 0.0278 | 0.62 | datasheet |
| RMD-X8 Pro 1:9 | 9 | 25 / 10 | 16.8 | 54.7 | 0.0275 | 0.61 | datasheet |
| **EPS-CEM-60** | 30 | 60 / 34 | 6.28 | 170 | 0.063 | 2.0 | **provisional** |

**CEM-60 is provisional.** No CEM-60 datasheet exists in the corpus or on myactuator.com, which list only the CEM-15/25/45.
The values are extrapolated from the CEM series:
- every model is 30:1 at 48 V;
- rated torque is about 0.57 × peak;
- rated speed is about 0.8 × no-load;
- rotor inertia is 0.7 kg*cm^2;
- the no-load speed of 60 rpm comes from `dropbear_docs/.../legs/actuators.md`.

The back-drive torque is a guess.

### Maps

| joint | `hw_v1` (user) | `hw_v1_cad` |
|---|---|---|
| hip roll `PG_*_leg_pitch` | X10 1:7 | X10-S2 |
| knee crank | CEM-60 | X10-S2 |
| hip yaw, hip pitch, calves, arms | same in both: X10 1:7, X10-S2, X8 Pro, X10 1:7 / X8 Pro | same |

### Gains (output side)

| joints | kp / kd |
|---|---|
| hip roll, hip yaw | 150 / 5 |
| hip pitch | 200 / 5 |
| knee crank | 500 / 5 (about 155 / 1.6 at the knee) |
| calves | 80 / 4 |
| shoulder pitch | 80 / 3 |
| other arm joints | 50 / 2 |

### The linkages make the knee and elbow weak

Both four-bars are speed-ups, so joint torque = motor torque / G (section 4):

| joint | G | `hw_v1` peak at the joint | `hw_v1_cad` peak at the joint |
|---|---:|---:|---:|
| knee | 1.79 | **33 N*m** (CEM-60) | 56 N*m (X10-S2) |
| elbow | 4.8 | **5.2 N*m** (X8 Pro) | 5.2 N*m |

For scale, the knee peak is about 300 N*m on H1 and about 139 N*m on G1.

### Results

**Zero-shot test** (`logs/hw_twin/zeroshot_*.json`). The `stiff_knee_hw` walker (`model_2998`) was run at 32/4 on 9
scenarios × 2 envs:

| profile | falls | typical fall | how its torques compare with the motor limits |
|---|---|---|---|
| `stiff_knee_hw` (trained) | 0/18 | - | ankles p95 41-51 N*m (real peak 25); hip roll max 130 (X10 1:7 peak 40); knee crank pinned at its 100 N*m limit 99.7 % of the time |
| `hw_v1` | 18/18 | 1.2 s, `anchor_height` | knee crank clipped at 60 N*m 99 % of the time; hip pitch clipped 54 %; calves 14 % |
| `hw_v1_cad` | 16/18 | 2.5 s | knee crank clipped 96 % (100 N*m) |

The idealized-sim walker therefore depends on torque the robot does not have.

**Zero-shot tracking** (`logs/hw_twin/track_wave_*.log`). The Brev `wave_v2_s16` tracker (a double-support stand
with a right-arm wave) was run at 32/4 with 4 envs:

| profile | falls | body error | anchor error | joint error |
|---|---:|---:|---:|---:|
| legacy | 0 | 5.5 cm | 2.5 cm | 0.79 rad |
| `hw_v1` | **0** | 5.4 cm | 4.2 cm | 0.81 rad |

So standing on both feet with arm motion transfers to the datasheet motors unchanged. The single-support walker does
not.

**Reference speed and range screen** (`tools/hw_motion_feasibility.py`, `logs/hw_twin/motion_feasibility_*.json`).
This checks the 78 clips of `accepted_v2`:
- **Speed:** with `hw_v1`, **70/78** clips never exceed a motor's no-load speed. The 8 that do are the fast ones:
  high jump, gangnam, `output_walk`, dance2, run, the A057 walk and the fast hiphop. They exceed it on the hip-pitch
  X10-S2 (5.6 rad/s) or the CEM-60 knee (6.3 rad/s), for 0.5-4 % of frames.
- **Range:** in 32-36 clips the knee crank spends more than 5 % of frames within 2 deg of a hard limit. The knee
  reaches only about 48 deg, so range is a bigger constraint than speed.

**Policy-free stand test** (`tools/hw_stand_test.py`, `logs/hw_twin/stand_*.json`). Zero actions hold the PD at the
calibrated stand. **Every** profile falls at 0.7-0.8 s, including legacy and `hw_v1 --unlimited`. PD alone cannot
balance this pose, since the ankle stiffness is close to m*g*l. The test therefore measures balance, not torque
sufficiency, and does not rule on the motors.

**Fine-tuning on Brev**, warm-started from the cloud walker `model_2189` (16/4, 4096 envs):
- GPU2 runs `vel_hw_v1_ft`, the user's map.
- GPU3 runs `vel_hw_v1_cad_ft`, the CAD map.
- Launch script: `tools/brev/swap_to_hw_twin.sh`.
- Results are in `OVERNIGHT_BREV_2026-09-25.md`.

**Fine-tuned walkers, evaluated locally at 32/4** (9 scenarios × 2 envs; `logs/hw_twin/eval_*.json`):

| run (checkpoint) | falls | fwd 0.3 / 0.5 | turn ±0.5 | left / right 0.2 | back 0.2 |
|---|---:|---|---|---|---|
| `vel_hw_v1_ft` (2600) | **0/18** | 0.24 / 0.45 | +0.41 / -0.46 | 0.00 / -0.06 | -0.11 |
| `vel_hw_v1_cad_ft` (2700) | 0/18 | 0.19 / 0.42 | +0.48 / -0.45 | 0.00 / -0.13 | -0.09 |

**The walkers stay inside PEAK torque but break the THERMAL limits.** Values are RMS torque / rated torque (clip % = share
of time at the envelope), for `hw_v1`:

| motor | RMS / rated | clip % | note |
|---|---:|---:|---|
| knee cranks (CEM-60) | 1.75 | 99-100 | pinned at 60 N*m |
| left hip roll (X10 1:7) | 2.0 | | |
| right calf B (X8 Pro) | 2.2 | 74 | |
| left elbow (X8 Pro) | 2.45 | 90 | speed 0.09 rad/s |
| every other motor | <= 1 | | |

`hw_v1_cad` shows the same pattern: its knee is pinned at 100 N*m. The joints that sit at full torque barely move
(elbow 0.09 rad/s), so the policy is pushing them into their end stops. That costs nothing in sim and cooks a real motor.

**Fix under test:** from 12:51 IST GPU3 runs `vel_hw_v1_thermal_ft`, which is `hw_v1` with
`--thermal_penalty=-2e-4` (the `motor_torque_over_rated_l2` reward). It is warm-started from `vel_hw_v1_ft`
`model_2786`, and the CAD A/B was stopped for it. Note that `--thermal_penalty` needs the `=` form: argparse reads
`-2e-4` as a flag.

**What is still missing from the twin:**
- measured values: bench tests B1-B5, CEM-60 specs, the real weight (the CAD says 56.2 kg, the user estimates about
  60);
- motor thermal limits: rated vs peak duty, which nothing enforces yet;
- observation (sensor) latency;
- per-motor speed gates in `validate_motion_npz` (section 7).

## 12. The locked right knee, and the fix (2026-09-25 afternoon)

### What the user saw

In the walking videos the right knee never bends. This happens in the old `vel_flat_kneehw` walker and in the
real-motor walkers alike. The dance tracker bends both knees.

### Diagnosis

The cause is the policy, not the plant.
- **The mechanism is not stuck.** In the calibration sweep the right knee covers -4.9..48.0 deg on the same LUT as the
  left.
- **Telemetry of `vel_hw_v1_ft` `model_3000`** (`logs/hw_twin/telemetry_hw_v1_ft_3000.npz`):

| | right knee crank | left knee crank |
|---|---|---|
| actual angle | 0 deg (the straight stop) the entire walk, max 0.5 deg while walking | reaches 30 deg |
| commanded target | **-160..-306 deg** | mean about -55 deg |
| foot on the ground | 8-21 % of frames | 86-94 % |

The robot hops on its left leg and taps the stiff right leg. The motor pushes into the stop at -60 N*m, 100 % of the
time.

Two holes in the training setup allowed this:
1. **Position targets are not clamped.** An action far beyond a hard limit turns into a constant peak torque into the
   stop.
2. **The H1 biped air-time reward is capped, not penalized.** A permanent single stance scores the maximum.

The dance tracker is fine because its reference dictates every knee angle.

### Fix

`DropbearVelocityFlatEnvCfg.enable_gait_shaping()`, CLI `--gait_shaping`. Observations and actions are unchanged, so it
can be warm-started. It adds:
- `clamp_targets_to_limits`: targets clipped to the hard limits ±3 deg. Pressing a stop then costs at most kp × 0.052 rad,
  which is 26 N*m on the knee crank. The export writes `target_clip`, and `play_locomotion` applies it automatically
  for runs trained with it.
- `knee_flexion_in_swing` (+0.5): each swinging knee flexed 0.15 rad above the stand.
- `feet_swing_height_l2` (-20): swing clearance of 8 cm (Unitree G1).
- `feet_mode_time_exceeded` (-2): a foot in contact for more than 1.0 s, or in the air for more than 0.6 s, while moving.
- `feet_air_when_standing` (-0.5): feet off the ground under a zero command.
- `motor_torque_rate_l2` (-0.02): `sum((dtau / peak)^2)` between policy steps, which penalizes torque spikes.

### Evidence that the thermal penalty alone is not enough

`eval_vel_hw_v1_thermal_ft_3400` vs `eval_vel_hw_v1_ft_3200`, as RMS / rated:

| motor | plain (3200) | thermal -2e-4 (3400) |
|---|---:|---:|
| right calf B | 2.32 | **0.53** |
| left elbow | 2.5 | **0.24** |
| hip pitch | about 0.8 | 0.3-0.6 |
| knees | about 1.75 | **still 1.6-1.76** |

Only the target clamp addresses the knee pinning.

### Brev runs from 13:55 IST

Both use hw_v1 + thermal -5e-4 + `--gait_shaping` and are launched with `tools/brev/swap_gait.sh`:
- GPU2 `vel_hw_v1_gait_ft`, warm-started from `vel_hw_v1_thermal_ft/model_3500`;
- GPU3 `vel_hw_v1_gait_scratch`, from scratch.

### Actuator dashboard

- `play_locomotion --telemetry x.npz` records all 22 motors at 50 Hz: position, target, speed, motor torque, PD torque,
  every 200 Hz substep torque, contacts and command. The recorder is `isaac/telemetry.py`.
- `tools/render_actuator_dashboard.py` overlays that data on the clean video:
  - one row per motor with its CAN ID (`0x140 + ID`: arms from the README, legs from the planned map, unverified),
    model, position (the knee through its LUT), speed, torque, a |torque|/peak bar (green ≤ rated, amber ≤ peak,
    red = clipped) and a speed bar;
  - curves of knee torque, knee angle and the worst 200 Hz torque step.
- Example: `logs/brev/media/hw_v1_ft_3000_dashboard.mp4`.

### Results so far (local evaluation with the dashboard, `logs/hw_twin/telemetry_*.npz`)

| run (checkpoint) | foot contact L/R | knee angles while walking | RMS / rated | torque jumps > 25 % of peak |
|---|---|---|---|---|
| before: `hw_v1_ft` (3000) | 0.91 / **0.12** (hopping) | L -5..47, **R -5..13** | knees 1.75, pinned at 60 N*m | 15.4 % |
| `gait_ft` (3700, warm) | 0.40 / 0.59 | L -5..47, **R -5..13** (still straight) | knees about 0.8 | 14.2 % |
| `gait_scratch` (600) | **0.68 / 0.67**, 37 % double support | **both 45-48 deg all the time** (crouched on the max-flexion stops) | **all ≤ 0.91** | 12.0 % |

`gait_scratch` meets the "bent in swing" reward by never extending. Gait v2 (`--gait_v2`) adds `knee_flexion_in_stance`
(-1.0), which penalizes a stance knee flexed more than 0.05 rad above the stand. From 14:59 IST, GPU2 runs
`vel_hw_v1_gait2_ft`, warm-started from `gait_scratch/model_700`. GPU3 keeps `gait_scratch` running as the fallback.

### Crossed legs (user report, 15:00) and gait v3

**What the user saw:** `gait_scratch_600` walks cross-legged, with each foot stepping past the other.

**Telemetry:** both hip-roll motors sit on their ADDUCTION stops for the whole walk: `PG_left_leg_pitch` at -15 deg (its
limit) and `PG_right_leg_pitch` at +15 deg. The stance width at rest is only 0.138 m, and self-collisions are off in
the sim, so the legs pass through each other at no cost.

**Physical reason:** the hip roll is the weak X10 1:7 (40 N*m peak, 15 rated). Placing the stance foot under the
centre of mass is the cheapest way to balance on one leg with it.

**Gait v3** (`--gait_v3`) is v2 plus:
- `feet_lateral_distance_below` (-20): soles closer than 0.10 m sideways, measured in the root yaw frame;
- `joint_near_limit_l1` (-5): any motor within 5 deg of a hard limit;
- `joint_deviation_hip` raised from -0.2 to -1.0.

The telemetry and dashboard now show the sole separation ("width", red below 5 cm).

**From 15:08 IST:**
- GPU2 runs `vel_hw_v1_gait3_ft`, warm-started from `gait_scratch/model_800`.
- GPU3 runs `vel_hw_v1_gait3_scratch`, from scratch.

**Hardware note:** if v3 walks with the feet apart but overloads the hip roll (RMS above rated), that is evidence the
X10 1:7 hip roll is undersized for walking. The CAD's X10-S2 (100 N*m peak, 50 rated) would be the fix.

**15:14: v3 at the first weights (-20 / -5 / -1.0) collapsed.** The warm-started run's episode length went
990 -> 49 in 70 iterations, with 89 % anchor-height terminations.
- **Cause:** per episode the new penalties (width -4.6, near-limit -2.1) outweighed everything walking earns (about
  +1.5), while a fall costs about 0.2. Dying early became the optimum.
- **Rule:** the sum of the new penalties must stay well below the tracking reward per episode.
- **v3 weights now:** width -2, near-limit -0.5, hip deviation -0.5.
- **Relaunched at 15:17** as `vel_hw_v1_gait3b_ft` (warm-started from `gait_scratch/model_800`) and
  `vel_hw_v1_gait3b_scratch`.

## 13. Bent-knee walking on the real motors, and the knee-motor A/B (2026-09-25 evening)

**Why tracking:** reference-free RL keeps choosing straight-leg gaits on this knee. Motion tracking copies a human knee
motion, so tracking walks were fine-tuned on `hw_v1`, warm-started from the idealized-motor trackers
(`tools/brev/swap_track.sh`).

**`lafan_walk_hw_3300`** (LAFAN walk1, about 800 fine-tune iterations; dashboard
`logs/brev/media/lafan_walk_hw_3300_dashboard.mp4`):
- 0 falls in 10 s;
- the knees bend through the stride: L 8-47 deg, R 2-48 deg;
- the feet never cross (mean width 36 cm);
- **but the knee cranks (CEM-60) are clipped at 60 N*m 82-85 % of the time**, at 1.65x rated RMS. Calves run at
  1.0-1.5x rated and hip roll at 1.36x;
- torque jumps of more than 25 % occur in 21 % of 5 ms steps.

By contrast, the reference-free `gait3b_scratch` compass gait stays within every rating.

**Hardware question:** does a human-style bent-knee walk need more knee torque than CEM-60 ÷ 1.79 (about 33 N*m at the
joint)?

**A/B from 17:44 IST:** GPU3 runs `lafan_walk1_kneeX10S2`, which is the same init, clip and seed as GPU1's
`lafan_walk1_hw`. The only difference is the profile `hw_v1_knee_x10s2`: the user's map with the knee motor swapped for
an X10-S2 (100 N*m peak, 50 rated, about 56 N*m at the knee joint). Compare knee clip % and RMS/rated at equal
iterations.

**Knee A/B result (18:55 IST)**, compared at a similar fine-tune age (`lafan_walk_hw_3300` vs
`lafan_walk_kneeX10S2_3100`; the logs and telemetry are in `logs/hw_twin/`):

| LAFAN walk, `hw_v1` except the knee | CEM-60 knee (the user's map) | X10-S2 knee |
|---|---|---|
| falls in 10 s | 0 | 0 |
| body error | 10.0 cm | **5.1 cm** |
| knee angle | 2..48 deg (mostly bent) | **-5..35 deg (extends every step)** |
| knee crank clipped | **82-85 %** | **14 %** |
| knee RMS / rated | **1.65-1.68** | **1.03-1.15** |
| knee-JOINT torque RMS (crank / 1.79) | 31-32 N*m | 29-32 N*m |
| worst other motor, RMS / rated | 1.49 (calf) | 1.28 (calf) |
| torque jumps of more than 25 % per 5 ms | 21 % | 16 % |

**Conclusion (sim, provisional CEM-60 values):**
- A human-style walk at this pace needs about 30 N*m RMS at the knee joint, which is about 55 N*m RMS at the crank.
- The CEM-60's rated torque (34, extrapolated) is about 1.7x short.
- The X10-S2's rated 50 N*m is about at the limit, and is workable with the thermal regularizer.
- **Hardware options:**
  - an X10-S2-class knee motor, as in the CAD;
  - a knee four-bar with less speed-up (G < 1.79), which trades knee speed for torque;
  - accepting a straighter-legged gait.
- **Verify first:** the real CEM-60 rated torque (datasheet or bench B1-B3). If it is much higher than 34 N*m, the gap
  shrinks.

**Refinement (19:45 IST): an upright walk fits the CEM-60 once regularized.**

`asap_walk_hw_reg_4700` is the ASAP walk on `hw_v1` (CEM-60 knee) with `--hw_regularizers`, compared with the
unregularized `asap_walk_hw_3300`:

| | `asap_walk_hw_3300` | `asap_walk_hw_reg_4700` |
|---|---|---|
| falls | 0 | 0 |
| body error | 5.4 cm | **4.3 cm** |
| knees | L -5..47, R -5..30 deg | L -5..47, R -5..27 deg |
| motors above rated RMS | 9, up to 1.7x | **1** (right elbow, 1.24x) |
| max clipped | 52 % | **15 %** (knee) |
| torque jumps of more than 25 % per 5 ms | 13 % | **8.6 %** |

The reference walks differ in posture, not speed:
- ASAP walk level 4: mean speed 0.63 m/s, **mean knee flexion 14 deg**, chest at 1.47 m.
- LAFAN walk1: 0.69 m/s, **mean knee flexion 23 deg**, chest at 1.43 m.

Knee torque scales with the flexion of the loaded knee, so:
- **With the CEM-60 (provisional values), Dropbear can do an upright human-style walk within the motor ratings.** A
  crouched gait needs X10-S2-class knee torque.
- **Answered at 20:50 IST** (`lafan_walk_hw_reg_4700`): the LAFAN walk with regularizers on the CEM-60 still has
  knees at 1.30 / 1.57x rated, clipped 41 / 72 % (was 1.65 / 1.68x and 82 / 85 %).
  - Five motors are above rated (was 8), with 10.2 cm tracking. The tracked knees average 43-44 deg, near the
    48 deg stop.
  - **So the crouched-gait overload is fundamental at this knee torque, while the upright ASAP gait fits.**
