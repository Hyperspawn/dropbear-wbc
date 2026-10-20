# Reconcile 2026-10-20: local actuator/training work (3001e16) vs the sim2real hardening patch set

Branch `dropbear-brief/sim2real-hardening`, based on main 3001e16. CPU suite (torch CPU, mujoco, no Isaac):
398 passed, 36 skipped, 0 failed. Skips are Isaac/pxr/vuer/mink/G1-asset tests.
`tests/test_sdk_transport.py::test_latest_message_semantics_and_nonblocking` is a ZMQ timing test that can flake
under parallel load; it passes on rerun.

## Findings (ranked by consequence; specs left unchanged, decisions open)
1. **CEM-60 rated point exceeds its own power rating.** `hw_motor_specs.py` rates CEM-60-V2 at 60 N·m @ 60 rpm =
   377 W shaft, above the 330 W rating before efficiency losses. The drawing's 60 N·m @ 40 rpm (251 W) is
   consistent. At the knee p99 speed (55 rpm) the model gives 75 N·m; a power-consistent envelope gives ~37.5 N·m.
   Build A and all hw_v2* policies were simulated with up to 2x the CEM-60's high-speed torque, and CEM-60
   "x rated" thermal ratios are understated (~1.4x at 80 % efficiency). This strengthens Build B and makes any
   policy deployed on today's CEM-60 legs optimistic.
2. **Mass DR no longer covers the robot after the motor-mass correction.** Monte Carlo (200k) with
   `motor_mass_offsets` + `mass_dr.py` ranges:

   | profile | nominal kg | tracking DR range kg | P(>= 60 kg) |
   |---|---|---|---|
   | hw_v2ie | 56.17 | 53.2-65.2 | 0.36 |
   | hw_cem6 | 51.55 | 48.8-60.3 | 0.000 |
   | hw_v4 | 54.85 | 51.9-63.8 | 0.16 |

   The ~60 kg reference is the robot with CEM-60s installed; subtracting the motor delta from the CAD moves Build A
   from 3.8 to 8.4 kg light. Re-centre the range or weigh the robot.
3. **Build B headroom is peak-only.** Held one-leg stance needs 56 N·m at hip roll vs 43 rated on X8-120 V4
   (1.30x); static squat needs 66-79 N·m at the knee vs 60 rated on X10-200 V4.1 (1.1-1.3x). The time-at-peak
   metric does not see held poses; a thermal run on held poses is needed before purchase.
4. **New training flags were not inherited on `--continue_run`.** thermal_penalty, stop_model, stop_load_penalty,
   impact_penalty, knee_band, body_pos_std, joint_pos_weight, end_threshold_m, anchor/feet height weights were
   recorded in run_info but a resumed chunk without them silently trained a different objective/plant.
   Fixed on this branch (run settings commit).
5. **Isaac Lab 3.0 break grew.** Stop/spring code added 6 `_parse_joint_parameter` calls (3 -> 9); the API is gone in
   3.0. Fixed on this branch (DatasheetMotor compat commit), covered by a 2.2/3.0 test.
6. **Ankle crank 40 -> 30 mm** equals lever ratio s ~ 1.33: pitch ROM 84.5 -> 65 deg, walking joint speed short of the
   gate (13.0 vs 13.7 rad/s), reflected inertia x1.69. The "-25 % motor load" claim omits these. An X10 1:7 on
   ankle B remains the cleaner fix if removing inner-edge pressing is not enough.
7. **V4 "Moment of Inertia" read two ways.** X8-120 value is taken as rotor, X10-200 as rotor x N. The other X8-120
   reading gives armature 0.0029 vs 0.0577 (20x). Armature DR between readings is on this branch for the legacy
   profiles; hw_cem6/hw_v4 `_armdr` variants are not yet defined.
8. **Knee stop spring stability is marginal (analysis, untested).** k = 8000, d = 20, dt = 5 ms on armature only
   (0.063 kg·m²): k·dt²/I = 3.17 vs semi-implicit Euler bound 4 - 2c·dt/I = 0.83. Stable only with >= ~0.15 kg·m²
   reflected shank inertia. Verify with a swing-phase knee-into-stop test (oscillation in stop_effort).
9. **Checked, OK:** all kd <= 5 in motion mode; stop torque bypassing the motor clip is correct (structure, not
   motor); `apply_body_mass_offsets` is idempotent and ordered after the DR terms (~1 % inertia-ratio mismatch).

## Patch status on this branch
| # | change | status |
|---|---|---|
| 0001 | sensor/latency DR for tracking | applied |
| 0002 | sensor age ranges | applied |
| 0003 | newton_bridge --sensor-profile | applied |
| 0004 | IMU valid/age, FSM stale-IMU refusal | applied |
| 0005 | arm duty screen, sustained-window telemetry check | rebased (test file merged with stop_rest tests) |
| 0006 | G0 scoreboard, per-motor gate | applied |
| 0007 | play.py tracking motor_torque summary | applied |
| 0008 | --continue_run run-settings inheritance | rebased, extended to the 10 new flags |
| 0009 | armature DR (vendor-inertia readings) | rebased onto stop model / knee band; no hw_cem6/hw_v4 variants |
| 0010 | mass DR | dropped: superseded by `mass_dr.py` + `apply_body_mass_offsets` on main |
| 0011 | kd_side (output vs motor side) | rebased |
| 0012 | deploy kd clamp <= 5, runner enable check | applied |
| 0013 | FSM motor-mode gate | applied (merged with 0004 FSM changes) |
| 0014 | serial hardware-surrogate MJCF/URDF | regenerated for hw_v1, cad_usd, hw_cem6, hw_v4; motor-mass offsets not included |
| 0015 | DatasheetMotor Lab 2.2/3.0 compat | rebased, covers stop model and knee band |

Open: `tools/hw_stand_test.py` and `tools/probe_knee_authority.py` still write `act.effort_limit` directly
(Lab 3.0 name change).
