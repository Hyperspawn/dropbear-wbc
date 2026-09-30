# What's new

A dated log of what was built. Times are IST. The detailed engineering record behind each line is in
[HANDOFF](HANDOFF.md), [DECISIONS](DECISIONS.md), [OVERNIGHT_BREV](OVERNIGHT_BREV_2026-09-25.md) and [ISSUES](ISSUES.md).

## 2026-09-30: public release

- **Repository published** with portable configuration (`source/dropbear_wbc/paths.py` + `.dropbear.env`; no
  machine-specific paths), a noncommercial license (PolyForm Noncommercial 1.0.0; commercial licenses from Hyperspawn)
  and third-party notices.
- **Assets on Hugging Face** ([Hyperspawn/dropbear-wbc](https://huggingface.co/Hyperspawn/dropbear-wbc)): the plant USD,
  8 trained policies, 13 dashboard videos, redistributable motion clips and the GR00T tabletop dataset, fetched and
  SHA-256-verified by `tools/fetch_assets.py`.
- **Browser demo** ([hyperspawn.github.io/dropbear-wbc](https://hyperspawn.github.io/dropbear-wbc/)): recorded motor-twin
  rollouts replayed in 3D with per-motor load (`site/`, `tools/export_web_rollout.py`).
- **Public libraries** `accepted_v6ts_public` (62 clips) and `accepted_v7gen_public` (122 clips): the redistributable
  subsets, so anyone can train and evaluate without rebuilding licensed data ([DATA](DATA.md)).
- Kimodo is set up from upstream at a pinned commit plus a 3-file overlay (`tools/setup_kimodo.sh`); workflow scripts
  (`tools/make_dashboard*.sh`, `tools/run_sim2sim_cpu.sh`, `tools/setup_venv_teleop.sh`) moved into the repository;
  CI runs the CPU unit tests.

## 2026-09-26: text to motion in real time, terrain, hardware fixes

- **01:27–02:36** Cyclic walking clips; the 20 ms target ramp (`hw_v1i`) halves 200 Hz torque jumps (27.0 → 18.3 %);
  the telemetry issue scanner and the **hardware gate** (17 checks).
- **02:30** First deployable walk to pass the gate (`kimodo_cyclic_nostate_v1i_feet/3184`), thanks to the feet-slide
  penalty (right-foot skid 17.7 cm → none).
- **02:46** **Froude library v6**: every G1 clip time-stretched ×1.19 (sqrt of the 1.42 size ratio).
- **04:05** The **knee-stop penalty** takes the crouched LAFAN walk off the CEM-60 knee's hard stop (32–45 % → 0–0.9 %).
- **04:15** The all-fixes walk passes; Froude timing cuts anchor drift after 15 s from 2.04 m to 0.52 m.
- **06:45–06:55** The LAFAN walk passes on the CEM-60 knee; action smoothing lowers target jitter 1.35 → 1.01°.
- **07:22** End-of-night library evaluations: `lib_v6ts_nostate_v1i_fix/10492` 41/78.
- **12:55–13:40** **Kimodo text-to-motion end to end**: the Llama-3 text encoder on the Windows CPU, Kimodo-G1 on the
  WSL GPU; the first generated wave on the twin (tracked, but the elbow runs 1.4× rated); 60 generated everyday clips
  → library `accepted_v7gen` (138 clips); a warm live service (prompt → spliceable clip in ~8 s).
- **14:25–14:40** Rough-terrain walker checked; stronger feet-width / action-rate weights (`vel_rough_v1i_b`).
- **17:01** **Real-time live session** on the laptop: physics on Isaac's CPU pipeline at 1.0× real time, a browser
  page with a prompt box, the int8 memory-mapped text encoder (1.2 GB instead of 16.4 GB), fall recovery.
- **17:20** Hardware speed gate on every generated clip (slow down up to 1.6×, else refuse).
- **18:13–18:45** Cloud session 2 closed ($34.92): `lib_v7gen` 94/138 (51/60 generated), `lib_v6ts_v1i_fix2` 46/78,
  terrain walker PASS (legs crossed 39 → 2.2 %), LAFAN smooth-elbow walk PASS.

## 2026-09-25: real motors, cloud training

- **00:15** First Brev instance (4× RTX 6000 Ada). Idealized-motor trackers hold: LAFAN walk (2.8 cm), Kimodo walk,
  Take_102, wave.
- **Midday** **Datasheet motor twin** (`hw_v1`, `hw_v1_cad`): the idealized walker falls 18/18 on it. Thermal
  penalty; gait shaping v1–v3b for the reference-free walkers.
- **15:31–15:40** Library evaluations on idealized motors: 51/78 and 57/78 (no-state).
- **16:20–20:50** Switch to reference-motion walking on real motors; 10 cm minimum stance width (library v5); knee
  A/B: an upright walk fits the CEM-60 knee, a crouched one does not (body error 10.0 vs 5.1 cm with an X10-S2).
- **20:53** The **NoState** (deployable) tracking task: no sim-only anchor position or base velocity in the observation.

## 2026-09-24: plant, tracking, SDK, teleop, GR00T

- Spawn-time plant fixes (the USD itself is never modified): orphan bodies, hip friction, a knee axis that locked the
  left knee, bicep inertia; **spherical ankle tie rods** (closure gap 21.3 → 1.19 mm); passive damping 0.
- **BeyondMimic tracking** on the closed-loop plant (wave, dance).
- `dropbear_hg-v1` **SDK**, Newton bridge, policy runner: the wave runs sim-to-sim.
- **Arm teleop** (WebXR / keyboard, priority IK); **serial MJCF + GMR** + SONIC `Humanoid_Batch` compatibility.
- **Velocity task**, foot-contact retarget stage (library v4), **motion-library tracker** with runtime references.
- **GR00T tabletop push**: scripted baseline 42/50, a 34-episode LeRobot dataset.
- Clean rollout-based rendering; the cloud plan.

## 2026-09-23: decisions

- The plant is Dropbear's own USD (SHA-256 `45586414…`) with in-memory fixes, not a URDF. The stack is Isaac Lab 2.2
  + rsl-rl 2.3.3 (vendored). Binding interfaces go in [CONTRACTS](CONTRACTS.md).
