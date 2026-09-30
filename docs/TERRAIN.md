# Terrain (blind) for the Dropbear velocity walker

Added 2026-09-26. `scripts/train_locomotion.py --terrain rough` (also `tools/run_chunked_locomotion.py --terrain rough`)
calls `DropbearVelocityFlatEnvCfg.enable_rough_terrain()` after gait shaping.

## Terrain grid

A curriculum grid of 10 difficulty rows x 20 columns of 8 m tiles. Mix:

| sub-terrain | share | range |
|---|---|---|
| flat | 15 % | |
| random rough | 25 % | 1-4 cm |
| pyramid slopes, up / down | 15 % each | up to 0.2 (11 deg) |
| pyramid stairs, up / down | 15 % each | steps 2 -> 10 cm, 35 cm deep |

- 10 cm is the honest ceiling for Dropbear. With the CEM-60 knee (about 33 N*m at the knee) and the 48 deg knee range,
  the crouched LAFAN walk already rests the knee on its stop (docs/ISSUES.md #19). Taller steps need the X10-S2 knee.
- Robots start in rows 0-3. Isaac Lab's `terrain_levels_vel` moves them up a row after they walk far enough and down
  after they fail.

## Height references follow the ground, and the policy stays blind

- A down-facing ray caster, `ground_scan` (0.4 m grid, 5 cm spacing, 2 m above the root, yaw-aligned), measures the
  ground under the base.
- `rewards.ground_z` feeds it to:
  - the anchor-height reward;
  - the fall termination (`anchor_height_below`);
  - the swing-height term.
- Without the sensor these fall back to the env-origin height, which is every earlier (flat) run.
- `ground_scan` is **not** in the observations: the policy is blind, deployable without a height sensor. Perceptive
  walking (stairs above about 5 cm on purpose) would add a height map from a depth camera; that is a later step.

## Status

- Smoke-tested locally (64 envs, 2 iterations, hw_v1i + gait v3 + clamp margin 0): terrain curriculum active
  (mean level 2.4), height reward relative to the ground, fall terminations working.
- **Trained on 2026-09-26** (Brev): `vel_rough_v1i/3200`, then `vel_rough_v1i_b/5500` with stronger feet-width and
  action-rate weights: HW gate PASS, 0 falls, legs crossed 2.2 % (was 39 %), but it limps and keeps its knees near
  the extension stop (ISSUES #29). Published: `assets/policies/vel_rough_v1i_b` (`tools/fetch_assets.py`).
- Suggested command (Brev): `run_chunked_locomotion.py ... --actuator_profile hw_v1i --gait_v3
  --target_clamp_margin_deg=0 --terrain rough --thermal_penalty=-2e-4`, warm-started from the flat gait-v3 walker.
- Tracking on terrain (the natural-looking reference walks) needs the reference height to follow the ground. A flat
  reference over a slope drifts up to 1 m in height and trips the tracker's fall check. Only mild roughness (a few cm)
  works without that change.
