# Roadmap: a general-purpose Dropbear

The goal: one robot that follows any whole-body motion (the motion library, locomotion on any terrain, push recovery,
and later manipulation with hands), smoothly, and on the real hardware. Updated 2026-10-18; see
[RESULTS.md](RESULTS.md) for the evidence behind each line.

## How we get there without testing every combination

- **Software choices are cheap to undo.** Rewards, penalty weights, motion sampling and training length cost hours
  and a few dollars to change. Retrain from a checkpoint and move on.
- **Hardware choices are expensive to undo.** Motors, linkages, printed stops and elastics cost money and weeks.
  Each open one gets a targeted test before anything is bought or printed (table below).
- **Physics prunes the space.** Measure the robot and some choices are forced. For example, a planted foot's balance
  has to go through the ankle, and the knee range caps squat depth.
- **Training does the search.** We define what "good" means and the body it has; reinforcement learning finds how to
  move.
- **Gates, not a glacier.** Each stage has one pass/fail test. Pass it and move on.

## Gates

| # | Gate | Pass test | Status |
|---|---|---|---|
| 1 | Body truth | The sim matches the CAD and the motors; every motor's load is reported against its rating | Done in sim (actuator viewer, closed loops, stop model). Missing: the robot's weight. |
| 2 | Stand and walk on the real motors | Rough-terrain walker passes the hardware gate, no falls | Passed (Build B, sessions 5b/6). |
| 3 | Track the motion library | Fidelity score ≥ 0.7 and the hardware gate passes on the 14-clip set (proposed bar) | Score 0.29–0.35, gate 8 / 14. |
| 4 | Pushes and terrain | `scripts/robustness_bench.py`: survive pushes from every direction (standing, walking, mid-motion); stairs, slopes, rough ground | Walker baseline (2026-10-19): pushes 0.2 m/s standing / 0.4 m/s walking on its weakest side; stairs to 11 cm, uphill to 18°. Never trained with pushes; blind. |
| 5 | One real leg on a bench | Torque, friction, delay and position tracking match the sim | Not built. |
| 6 | Full robot | A policy that uses only the real sensors runs on the hardware | After 5. |

## Settled (revisit only with new evidence)

- **Robot model:** the closed-loop USD is the robot (27 loop closures). URDF can't represent it.
- **Motors:** Build B (X8-120 at hip roll and pitch, X10-200 at the knee) has the torque headroom. Build A's CEM-60
  knee hits its peak 13–16 % of the time.
- **Success metric:** visible motion features (flight, foot lift, torso drop), not "didn't fall".
- **Training recipe:**
  - match the reference;
  - score each motor's load against its own rating;
  - keep end-stop load and landing impact down;
  - run the stop model on.

  Fewer terms over time, not more.
- **Knee elastic:** a plain band that engages at a fixed angle is ruled out, because it fights the walking swing.

## Open hardware decisions

| Decision | What we know (simulation) | Test that settles it |
|---|---|---|
| **Knee travel** (47° now) | Squats reach 6–8 of 18 cm of torso drop, with the knee on its limit | A 90–120° knee linkage in CAD, then the squat test on the new range |
| **Ankle capacity** (two X8 Pros, 40 mm cranks) | About 23 N·m at the ankle at rated, 57 at peak, against about 18 typical and 52 on the hardest motions. Motor B is overloaded by the robot pressing on the inner edges of its feet. | 1. Train without the inner-edge pressing. 2. If still over, sim a shorter crank (40 → 30 mm, −25 % motor load) before a stronger motor. |
| **Tie-rod top ends** (pins on the robot, confirmed 2026-10-19) | Pins hold each rod in a plane, so tilting the foot sideways forces the heel ball joints about 15 mm × tilt (rad) sideways: the ankle binds at about ±8° unless parts flex. The motions use more than 8° 6 % of the time (march 42 %, up to 19°). All training so far assumed ball joints. | Fit rod-end ball joints at the cranks (sim already models this), or lower the heel ball joints to the roll-axis height |
| **Right elbow** (X8 Pro behind its 4.8× linkage) | Up to 1.9× rated on the hardest motions | Check which motions and the linkage ratio, then decide on ratio or motor |
| **Knee elastic, load-triggered** | Helps hold a squat, fights the swing | Sim an elastic engaged only when that foot carries weight |
| **Printed knee-lock blocks** | The current recipe keeps them unloaded in sim; earlier policies put up to 3.9 kN on them | Design for about 1 kN impacts regardless; the bench leg confirms |
| **Hip roll motors** (not on the robot yet) | Build B uses X8-120 there | Buy after the knee and ankle decisions settle the leg |
| **Hands** | Not in the CAD | After gate 3 |

## The stack

| Layer | What it is | Status |
|---|---|---|
| 1. Body | Motors, linkages, ranges, hands, sensors, CAN, onboard compute | See the open decisions above |
| 2. Digital twin | Sim that behaves like the robot: closed loops, motor model, stops, masses, delays | Done in sim; needs the weight and a bench leg |
| 3. Motion data | Mocap, video, text-to-motion (Kimodo), teleop | 138 library clips; needs a physics check per clip |
| 4. Universal tracker | One policy that follows any reference with the whole body | Training with the motor-heat recipe |
| 5. Command sources | Walking planner, text-to-motion, VR teleop, hand targets | Text-to-motion demo; velocity walker |
| 6. Task brain | Vision-language-action model (openpi / GR00T) trained on teleop | After hands and teleop |
| 7. Evaluation | Motion fidelity, hardware gate, part loads, all 22 motors against their ratings | In place |

## Next steps

1. Train the tracker longer with the motor-heat recipe, and stop the inner-edge pressing at the ankles.
2. Simulate a load-triggered knee elastic and a shorter ankle crank.
3. Weigh the robot; put one leg on a bench.
4. Design a knee linkage with more travel.
5. Retrain the universal tracker from scratch on the final body: a teacher with privileged information, then a
   student that uses only the real sensors.
