# Documentation

## Start here

| document | what it is |
|---|---|
| [INSTALL](INSTALL.md) | install tiers: browser, CPU, Isaac Sim 5.0 + Isaac Lab 2.2, text to motion, cloud |
| [QUICKSTART](QUICKSTART.md) | commands for every demo: dashboards, library evaluation, real-time preview, live text to motion, training, export, sim-to-sim |
| [RESULTS](RESULTS.md) | every result, with its source |
| [FINDINGS](FINDINGS.md) | what was learned, and the open problems |
| [WHATSNEW](WHATSNEW.md) | dated log of what was built |
| [DATA](DATA.md) | which motion data is published, the licenses, and how to rebuild the rest |

## Design and interfaces

| document | what it is |
|---|---|
| [CONTRACTS](CONTRACTS.md) | the binding interfaces: plant and fixes, motor contract, semantic joint space, motion CSV/NPZ schemas, tasks, SDK, policy sidecar |
| [DECISIONS](DECISIONS.md) | dated architecture decisions and why |
| [ACTUATORS](ACTUATORS.md) | the real motors: datasheets, the CAN protocol, the motor twin, gait shaping on real motors, the knee A/B |

## Topics

| document | what it is |
|---|---|
| [DEMOS](DEMOS.md) | the first tracking demos (wave, dance), rendering, sim-to-sim demos |
| [LOCOMOTION](LOCOMOTION.md) | the velocity task, actuator profiles, commands |
| [TERRAIN](TERRAIN.md) | rough-terrain training |
| [TEXT_TO_MOTION](TEXT_TO_MOTION.md) | retrieval, Kimodo generation, the live service and the real-time session |
| [SDK](SDK.md) | `dropbear_hg-v1` SDK, Newton bridge, policy runner, sim-to-sim |
| [TELEOP](TELEOP.md) | XR / keyboard arm teleoperation |
| [GROOT](GROOT.md) | GR00T tabletop push task and dataset |
| [SERIAL_MODEL_AND_GMR](SERIAL_MODEL_AND_GMR.md) | the derived serial MJCF, GMR retargeting, SONIC compatibility |
| [BREV](BREV.md) | cloud training: instance, setup, data sync, run queues, the budget watchdog |
| [HF_MODEL_CARD](HF_MODEL_CARD.md) | the Hugging Face card: published policies, their training data and licenses |

## Engineering journal

Dense, chronological records written during development. They cite `logs/` files that are not in the repository.

| document | what it is |
|---|---|
| [HANDOFF](HANDOFF.md) | working handoff between development sessions |
| [OVERNIGHT_BREV_2026-09-25](OVERNIGHT_BREV_2026-09-25.md) | hour-by-hour log of both cloud sessions |
| [ISSUES](ISSUES.md) | every issue found (#1–#35): symptom, cause, fix, status |

## Glossary

| term | meaning |
|---|---|
| plant / contract plant | the Dropbear USD (SHA-256 `45586414…`) plus the in-memory spawn fixes (CONTRACTS §0.1–0.3) |
| closures | the 27 loop-closure joints excluded from the articulation and solved by PhysX |
| anchor | `head_5mm_ujoint_base__5__1`, a chest-level torso body used for tracking and fall detection |
| semantic joint space | 22 G1-named joint angles; calibration maps (`linear`, `lut1d`, `lut2d`, `serial3`) convert them to motor angles |
| knee crank, four-bar | the knee motor drives a four-bar linkage, about 1.79× speed-up at the knee |
| calf motors A / B | `*_Revolute67` / `*_Revolute81`, the two motors of the parallel ankle |
| 8/4, 16/4, 32/4 | PhysX position / velocity solver iterations |
| NoState | a tracking policy without the simulator-only anchor position and base velocity: deployable |
| Future | a tracking command that also sees reference frames +0.1 / +0.2 s ahead |
| settle, verdict | a clip's physics settling pass, and the validator's accepted / rejected verdict |
| accepted_vN, v6ts, v7gen | pinned motion-library manifests (v6ts: Froude-timed; v7gen: + 60 generated clips) |
| Froude time scale | stretch time by sqrt(spatial scale): sqrt(1.42) = 1.19 from G1 to Dropbear |
| cyclic clip | a looped, seam-blended walk (`tools/make_cyclic_clip.py`) |
| DatasheetMotor | the motor-twin actuator model (torque-speed envelope, friction, latency, target ramp) |
| hw_v1 / hw_v1i / hw_v1ie | motor profiles: the real motor map; + 20 ms target ramp; + stiffer elbow gains. `hw_v1_cad` and `hw_v1_knee_x10s2` swap in X10-S2 motors |
| HW gate | pass / fail of `tools/telemetry_issue_scan.py` on a rollout's telemetry |
| kstop, feet slide, target clamp | training terms: stay off the knee hard stops; no sliding feet; motor targets clipped to the joint limits |
| CEM-60, X10-S2, X10 1:7, X8 Pro | the motors: EPS-CEM-60 knee (provisional specs), MyActuator RMD X10-S2 (hip pitch), X10 1:7 (hip roll, shoulder), X8 Pro 1:9 (the rest) |
| Motion Mode | MyActuator's impedance command (kp ≤ 500, kd ≤ 5, feed-forward ±24 N·m) |
| sidecar | `policy.json` next to an exported policy: observation layout, joint order, scales, target clip and ramp |
| dropbear_hg-v1 | the Unitree-style LowCmd / LowState SDK |
| Kimodo, LLM2Vec | NVIDIA's text-to-motion model, and its Llama-3-based text encoder |
