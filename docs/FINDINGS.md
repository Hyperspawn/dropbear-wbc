# Findings

What this project learned while bringing whole-body control to a closed-loop, 3D-printed humanoid, and what is still
open. Each point links to where the evidence lives.

## The plant

1. **Use the robot's own USD with its loop closures, not a URDF.** Dropbear's knees and elbows are four-bars, the ankle is
   a parallel mechanism and the neck is a Stewart platform. A serial model cannot represent them; the USD keeps
   **27 loop-closure joints** that PhysX solves every step (CONTRACTS §0). A derived serial MJCF exists for retargeting
   tools only (SERIAL_MODEL_AND_GMR).
2. **CAD exports need auditing, and the fixes belong at spawn time.** One joint axis in the export locked the left knee,
   hip joints carried friction, some bodies were orphaned and one inertia was wrong. They are fixed in memory when the
   robot spawns; the USD is never edited (CONTRACTS §0.1).
3. **Get the joint types right before tuning anything.** Retyping the ankle tie rods to spherical (ball-and-socket, as
   built) cut the worst closure gap from 21.3 mm to 1.19 mm (CONTRACTS §0.2).
4. **Measure what the simulator applies.** The legacy passive damping of 50 turned out never to be applied in Isaac; all
   simulators now use 0 (CONTRACTS §0.3). Evaluate at 32/4 solver iterations; train at 8/4 or 16/4.

## The motors decide

5. **A policy trained on idealized motors is not evidence.** The first walker fell **18/18** times once the motors had
   datasheet torque-speed envelopes, friction and latency (ACTUATORS §11). Everything since is trained on the motor twin.
6. **The knee is the bottleneck.** Through its four-bar the CEM-60 gives about 33 N·m at the knee over about 48° of
   range. An upright gait fits it; a crouched one sits on the flexion hard stop and saturates (knee clipped 82–85 %,
   RMS 1.65× rated), where an X10-S2 would not (ACTUATORS §13). A knee-stop penalty moves the crouched walk off the stop
   (32–45 % → 0–0.9 %) and under the rated torque (ISSUES #19).
7. **The elbow linkage multiplies gravity.** The 4.8× speed-up linkage means the X8 Pro sees 4.8× the forearm's gravity
   torque. Arm-raised motions (a wave) run the elbow at 1.4× rated: gains cannot fix gravity; it needs a stronger motor
   or a different ratio (ISSUES #24, TEXT_TO_MOTION).
8. **Step targets at 50 Hz make torque spikes at 200 Hz.** Ramping each new target over 20 ms cuts 200 Hz torque jumps
   from 27 % to 18 % of steps without hurting tracking. The motor firmware should do the same; `sdk/target_ramp.py` is
   the reference implementation (ISSUES #11, #26).

## Finding what videos hide

9. **Scan telemetry, not just videos.** The hardware gate found the knees resting on their hard stops under load in
   30–50 % of steps, feet skidding 18 cm at touchdown, and targets commanded up to 79° past a stop, none of which were
   visible in the rendered videos (ISSUES #19, #20, #22, #25).
10. **Each finding became a training term.** Knee-stop penalty, feet-slide penalty, target clamp to the joint limits,
    action-rate smoothing (target jitter 1.35 → 1.01°), thermal and torque-rate regularizers. The first gate-passing
    walk came from exactly that loop (ISSUES #19–#28).
11. **Evaluate per clip, and check repeatability.** Library trackers are scored clip by clip in continuous loops; two
    independent evaluations agree (94/138 both times) (RESULTS).

## Motion data

12. **Scale time as well as space.** Dropbear is 1.42× the size of a G1. Scaling positions without time makes every
    retargeted motion dynamically too fast; Froude similarity says stretch time by sqrt(1.42) = 1.19. With it, a walk
    keeps pace with its reference (anchor drift after 15 s 0.52 m vs 2.04 m) and fewer clips exceed motor speeds
    (ISSUES #27).
13. **Settle and validate every clip in physics** before training on it, and pin libraries by SHA-256
    (SERIAL_MODEL_AND_GMR, CONTRACTS §5).
14. **Deployable means no-state.** Tracking policies that observe the simulator's anchor position or base velocity do
    not transfer; the no-state variants observe only what the robot can measure and still track the library (ON 20:53).
15. **Reference-free walkers pick the cheapest gait**, straight-legged compass walking, whatever the shaping terms;
    tracking a reference motion gives natural gaits (ISSUES #13, #29).

## Real time on a laptop

16. **For one robot, simulate on the CPU.** Isaac's GPU pipeline runs one robot at 0.15–0.17× real time; the CPU
    pipeline runs the same physics 5× faster. With lean play settings and one indexing fix (the motion command copied
    the whole reference buffer every step) it reaches 1.0× real time (ISSUES #30, #31).
17. **Render in another process.** A viewport frame costs about 28 ms inside the physics loop; the physics publishes
    link poses to shared memory and a browser page draws them (TEXT_TO_MOTION).
18. **Committed memory, not RAM, is the limit on Windows.** The 8B text encoder in bf16 plus two Isaac processes
    exhausted the commit limit. Weight-only int8 with per-row scales keeps cosine 0.9996 to bf16 (quantizing activations
    too dropped it to 0.82), and memory-mapping the weights read-only makes them file-backed: 1.2 GB committed instead of
    16.4 GB (ISSUES #32).
19. **Windows parks background processes on efficiency cores.** A real-time physics process has to opt out of EcoQoS
    (ISSUES #34).

## Open problems

- **No hardware result yet.** Everything above is simulation.
- **Unknown actuator data.** The CEM-60 knee has no datasheet (values extrapolated from the CEM-15/25/45); Motion-Mode
  kp scaling and whether kd acts on motor- or output-side speed are unverified; friction, backlash and sensor latency
  are not measured (ACTUATORS §8–11, ISSUES #14, #28).
- **Firmware.** The ESP32 sketch writes the comms-timeout value to the wrong bytes, so the motors' timeout protection is
  disabled; it must be fixed, and the 20 ms target ramp added, before powering the legs (ACTUATORS §8, HANDOFF).
- **Mass and weight.** CAD masses (56.2 kg) vs an estimated 60 kg; printed-part densities likely too high.
- **Elbow.** 1.4× rated on arm-up motions, about 12° of lag, often near its stops (#24, #28).
- **Library coverage.** 46/78 and 94/138 clips; dynamic clips (jumps, kicks, CR7) still fall.
- **Terrain.** Blind, 10 cm steps, and the walker limps with straight knees (#29).
- **Live text to motion.** 8.5–10.6 s latency (target 1–3 s); generated clips are kinematic and only speed-screened.
- **Sim-to-sim.** Faithful Newton transfer needs stiff closures (MuJoCo C only); MuJoCo Warp at those settings is untested.
- **GR00T.** No fine-tune or closed-loop evaluation yet; no gripper. **Teleop:** no real headset tested; fixed base only.
- **Training scale.** One GPU per run; multi-GPU training is not wired (BREV §2).
