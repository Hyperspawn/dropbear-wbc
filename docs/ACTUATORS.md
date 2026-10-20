# Dropbear actuators

The motors on the robot, the two builds we are choosing between, and how the simulator models them. Current as of
2026-10-16. How we got here (including conclusions that turned out wrong) is in
[archive/ACTUATORS_HISTORY.md](archive/ACTUATORS_HISTORY.md).

## 1. The robot today

| Joint (count) | On the robot | Notes |
|---|---|---|
| Hip roll, in the butt (2) | **not installed** | The butt is designed for a CEM-60. Two motors still to buy (section 2). |
| Hip yaw, facing down in the pelvis (2) | RMD-X10 1:7 (X10-40) | |
| Hip pitch (2) | CEM-60 | |
| Knee (2) | CEM-60 | Drives the knee through a four-bar (section 5). |
| Ankle, in the shank (4) | RMD-X8 Pro 1:9 (X8-25) | Two per foot, through tie rods. |
| Shoulder pitch (2) | RMD-X10 1:7 (X10-40) | |
| Shoulder roll, arm twist, elbow, wrist (8) | RMD-X8 Pro 1:9 (X8-25) | The elbow goes through a linkage (section 5). |
| Neck (6) | NEMA 17 steppers + lead screws | Not MyActuator. Held still in the sim. |

Bought in 2024: 4 CEM-60, 4 X10 1:7, 12 X8 Pro. The original CAD had RMD-X10-S2 1:35 (X10-100) at hip roll, hip pitch
and knee; MyActuator no longer offers it, which is why the legs have CEM-60s. The plant USD (`45586414`) still carries
the CAD's X10-S2 parts, so the sim corrects their mass (section 4).

## 2. The two candidate builds

Only the six heavy joints differ. Everything else stays as in section 1.

| | **Build A: `hw_cem6`** | **Build B: `hw_v4`** |
|---|---|---|
| Hip roll (2) | CEM-60 | X8-120 (V4) |
| Hip pitch (2) | CEM-60 (have) | X8-120 (V4) |
| Knee (2) | CEM-60 (have) | X10-200 (V4.1) |
| To buy (market price) | 2 × CEM-60 = $2,580 | 4 × X8-120 = $2,580, plus 2 × X10-200 (about $1,800 retail) |
| Mass of the six motors | 5.6 kg | 8.9 kg |
| Sim plant mass (CAD 56.2 kg) | 51.6 kg | 54.9 kg |
| Fits the current CAD | yes (designed for it) | hip roll: a 76 mm motor fits where the X10-S2 was 74 mm; knee: the X10-200 is about the X10-S2's envelope |

### Motor data

At the output shaft, 48 V. Prices are market prices (MyActuator sales and US stores, October 2026); Hyperspawn's
quotes are lower and kept private.

| Motor | Gear | Rated torque | Peak torque | Rated speed | No-load speed | Weight | Size Ø × L | Market price |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CEM-60 (cycloid) | 30:1 ? | 60 N·m | 75 N·m (≤10 s) | 60 rpm | 80 rpm | 0.93 kg | 100 × 56 mm | $1,290 |
| X8-120, V4 (planetary) | 19.6:1 | 43 N·m | 120 N·m | 127 rpm | 158 rpm | 1.40 kg | 96 × 76 mm | $645 |
| X10-200, V4.1 (planetary) | 20:1 | 60 N·m | 170 N·m | 74 rpm | 80 rpm | 1.64 kg | 124 × 63.5 mm | ~$899 retail |
| X10 1:7, V3 (X10-40) | 7:1 | 15 N·m | 40 N·m | 165 rpm | 186 rpm | 1.15 kg | 122 × 53 mm | $645 |
| X8 Pro 1:9, V2 (X8-25) | 9:1 | 10 N·m | 25 N·m | 110 rpm | 160 rpm | 0.71 kg | 98 × 51 mm | $470 |
| *X10-S2 1:35 (CAD only)* | *35:1* | *50 N·m* | *100 N·m* | *50 rpm* | *54 rpm* | *1.70 kg* | *122 × 74 mm* | *not offered* |

The CEM-60 line uses the user's rating (48 V, 330 W, 60 N·m and 60 rpm rated) and the vendor table's 75 N·m peak and
80 rpm. The datasheets themselves live in the Hyperspawn web repo (`data/actuators/datasheets/`).

## 3. What each joint needs

The torque and speed the heavy motors used across 27 simulated rollouts (squats, lunges, the motion library, terrain
walking), at the motor output. These came from the old sim motors, so where a motor hit its limit the true demand is
higher.

| Joint | Average torque (RMS) | Peaks | Speed (99th pct.) | Notes |
|---|---|---|---|---|
| Hip roll | 20–32 N·m in hard motions | stuck at the old 40 N·m limit | ≤ 19 rpm | One-leg stance in sim (Build B one-foot specialist): median 31, p95 43, max 53 N·m (RMS 0.71× of the X8-120's 43 rated). The earlier static estimate of 56 N·m assumed the body's weight stayed centred between the feet. |
| Hip pitch | 15–35 N·m | up to 100 N·m (squat, lunge); 45–75 walking | ≤ 36 rpm | |
| Knee motor | 20–62 N·m | 75 N·m in almost every rollout | ≤ 55 rpm | The tightest joint. A static squat at 40–47° knee needs 66–79 N·m. |
| Hip yaw | 4–11 N·m (one 23) | ≤ 40 N·m | ≤ 25 rpm | The X10 1:7 is enough. |

## 4. How the simulator models a motor

Code: `source/dropbear_wbc/robots/hw_motor_specs.py` (numbers) and `robots/hw_actuators.py` (`DatasheetMotor`).

- **Torque limit:** the peak torque, and above it a straight line that falls to zero at the no-load speed and passes
  through the rated point.
- **Rotor inertia:** rotor inertia × gear ratio², set on the motor joint.
- **Friction:** a constant term equal to the back-drive torque plus a small speed term, both scaled by 0.5–1.5 at
  random each episode.
- **Control:** a PD loop at 200 Hz inside the motor's own command range (kp ≤ 500, kd ≤ 5), with a 0–10 ms command
  delay. Each new 50 Hz target is ramped over four control ticks (the `i` profiles).
- **Mass:** in `hw_cem6` and `hw_v4`, each new motor's weight minus the CAD's X10-S2 (1.70 kg) is added to the body
  that holds it (pelvis for hip roll, thigh for hip pitch and knee), so the robot's mass is right for each build.

### Profiles

| Profile | What it is |
|---|---|
| `hw_cem6` | Build A (section 2). Current. |
| `hw_v4` | Build B (section 2). Current. |
| `hw_v2ie` | Before 2026-10-16: wrong at the hips (X10-S2 hip pitch, X10 1:7 hip roll). The published policies and the live demo use it until new policies exist. |
| `hw_v1*`, `hw_v2`, `hw_v2i`, `hw_v2ie_r50`, `hw_v1_cad`, `hw_v1_knee_x10s2` | Older versions, kept only so earlier policies replay. |

Pass the profile with `--actuator_profile` to the train and play scripts.

## 5. Linkages

- **Knee:** the motor sits at the top of the thigh and turns the knee through a four-bar that speeds it up about 1.79×
  at standing (2.1–2.2× when deeply bent). So the knee joint gets the motor torque divided by that, and its range is
  0–47° (crank 0–30°).
- **Elbow:** a linkage that speeds the elbow up about 4.8×. With an X8 Pro the elbow joint gets only about 5 N·m peak,
  which matters once hands add weight.
- **Ankle:** two shank motors push the foot through tie rods with ball ends. Both turning the same way pitches the
  foot; opposite ways rolls it.

## 6. Open questions

For MyActuator:
1. **CEM-60 at 48 V:** its torque-speed curve and rated power. The sheet says 24 V, 4 A, 96 W, which cannot give
   60 N·m at 60 rpm.
2. **CEM-60 gear ratio** (30:1 or the drawing's 50:1), rotor inertia and back-drive torque.
3. **X10-200 (V4.1):** price, and whether a CAN (`-C`) version exists. Our bus is CAN.
4. **V4 "Moment of Inertia":** is it the rotor's own inertia? The sim currently reads the X8-120's value as the rotor
   and the X10-200's as rotor × ratio.

For the robot:
- Where the shoulders and wrists really stop (the sim has no end stops there).
- Weigh the robot (the CAD says 56.2 kg; the estimate is about 60 kg).

## 7. Where things are

- **Interactive map** of every motor on the 3D robot: `tools/build_actuator_viewer.py`.
- **Numbers per joint:** `source/dropbear_wbc/robots/hw_motor_specs.py`.
- **Hardware check of a rollout:** `tools/telemetry_issue_scan.py`. It flags thermal overload, torque clipping, and a
  joint resting on its end stop.
- **History and datasheet notes:** `archive/ACTUATORS_HISTORY.md` and `data/robot/actuators_datasheet_v1.json`.
