# Text -> Dropbear motion, live, with the real-actuator physics

Status 2026-09-26 17:10 IST. The user asked "text to dropbear motion (with hw physics) in real time: will that be
possible?". **It runs on the laptop now** (see "Real-time live session" below): type a motion in a browser page, and
8.5-10.6 s later the motor-twin robot performs it, with the physics paced to real time (1.0x between prompts, 0.92x
averaged over a busy 2-minute session). The remaining gap is latency (seconds, not the 1-3 s target) and tracker
quality on some generated motions.

## Pipeline

```
 text ──► [1] text -> motion ──► [2] G1 -> Dropbear retarget ──► [3] live reference stream ──► [4] tracking policy ──► [5] physics / robot
          retrieval (today)          (only for generated G1 motion)     MotionStream: splice,        library tracker,          Isaac (training/eval)
          Kimodo-G1 / ARDY (next)                                        align, blend, idle           no state obs, 50 Hz       Newton bridge (real time)
                                                                                                                                 robot (later)
```

| stage | code | measured | status |
|---|---|---|---|
| 1 text -> clip (retrieval) | `tools/text_to_motion.py` (MiniLM from the HF cache; hardware-safe subset only) | 10-18 ms per prompt (model load 7 s once) | DONE |
| 1 text -> motion (generation) | Kimodo-G1-RP-v1, a copy in `third_party/kimodo`, WSL venv `.venv-kimodo-wsl` (GPU). The Llama-3/LLM2Vec text encoder runs on the Windows CPU via `tools/kimodo_text_embed_server.py` and a shared folder (`TEXT_ENCODER_MODE=file`) | text 2.7 s (CPU) + 100 diffusion steps about 6 s (4080), plus a one-time model load of about 130 s | **WORKS (2026-09-26 12:55).** The 12 GB GPU cannot hold the 16 GB encoder (the Windows driver spills into shared memory and stalls), and bitsandbytes 4-bit + PEFT fails in transformers 5.1, hence the Windows CPU bridge. For live use, keep one Kimodo process warm |
| 2 G1 -> Dropbear retarget | `dropbear_wbc.motion.g1_to_dropbear` (CPU) | full foot-contact solver 1.07 s per 1 s of motion; per-frame map 0.23 s/s | works offline; needs chunked / streaming use |
| 3 live reference | `dropbear_wbc.motion.stream.MotionStream`; Isaac: `isaac/live_reference.py`, `play.py --live_script / --live_dir`; deploy: `deploy.motion.LiveMotion`, `policy_runner --live_dir` (each clip checked like `--motion`, fail closed) | splice + device copy 47-94 ms | DONE in both. Tests: `test_motion_stream.py`, `test_live_motion_splices_dropped_clips_at_the_playhead` |
| 4 tracker | library NoState policy (`lib_nostate_kneeX10S2_v5/model_8200`) | 0.1 ms per step | 43/78 library clips hold on real motors; not yet trained with the hw penalties |
| 5 physics | Isaac Lab + DatasheetMotor, **CPU pipeline** (`play.py --device cpu --realtime`) | **1.0x real time** for one robot (18.9 ms per 20 ms policy step; PhysX 6.7 ms of it). The GPU pipeline is 0.15-0.17x for one robot (130 ms per step: kernel launches and syncs) | DONE 2026-09-26: same physics (tracking errors, closure gaps and falls match GPU within run-to-run noise) |
| 5 physics (deploy sim) | `tools/newton_bridge.py` (MuJoCo) + `tools/policy_runner.py` | 2 ms step: real time with headroom (1.06-1.21 ms per step) | **but** at 2 ms the loop closures go soft and Isaac-trained policies fall in about 2 s (docs/SDK.md 8.6) |
| 5 physics (deploy sim, faithful) | same, 0.5 ms step + stiff closures (`eq_solref` 0.001) | real-time factor 0.41 on CPU | tracking works (wave clip). GPU MuJoCo Warp at these settings is UNTESTED: the next measurement |
| 5 motor law in the deploy sim | `newton_bridge.py --motor-profile hw_v1i --target-ramp-ms 20` | the warp kernel matches `hw_motor_specs.hw_motor_torque` (test) | DONE 2026-09-26: the same torque-speed envelope, friction and armature as training |

## Demos (sim, motor twin hw_v1 + X10-S2 knee, library policy)

- `logs/brev/media/live_demo1_dashboard.mp4`: stand > wave @2 s > walk @6 s > dance @14 s, spliced into the running
  reference.
  - The dance is the hardest clip. The run ended on a tracking-error termination at 20.9 s; live mode now keeps only
    the fall check, because a termination rewinds the live timeline.
- `logs/hw_twin/live_demo2`: the real text path. A typist process sends prompts; `text_to_motion.py` retrieves a clip
  and drops it into the watched folder; the sim splices it.
  - Prompts: "wave hello", "walk forward", "kick the ball", "stop".
  - 0 falls, body error 4.8 cm.
  - The prompts landed about 1 s apart in sim time because Isaac runs at 1/6 real time, which is why the real-time
    loop belongs in the Newton bridge.
- `logs/brev/media/live_text_demo3_dashboard.mp4`: the same prompts on a sim-time schedule, so each motion plays out:
  wave > walk > kick > squat > stop.

## Behaviour of the live stream

- A new command interrupts the current motion 0.1 s ahead of the playhead, cross-fading over 0.4 s.
- The clip is moved (x, y, yaw) to where the reference is, so the robot never teleports.
- After a clip ends, the reference blends over 0.8 s into the idle clip (a stand) placed where the clip ended,
  instead of freezing mid-stride.
- Unsafe requests: retrieval only offers clips that the real-motor tracker held without a fall. "Celebrate like
  Ronaldo" is refused, because every CR7 clip falls on the motor twin.

## Next, in order

1. **Real-time loop on the deploy side:**
   - `policy_runner --live_dir` with `MotionStream` as its motion source: DONE.
   - A DatasheetMotor motor law in `newton_bridge.py`: DONE (`--motor-profile`; the target ramp is `--target-ramp-ms`).
   - A physics preview that is both faithful and real-time: the closed loops need 0.5 ms steps and stiff closures
     (real-time factor 0.41 on CPU). Options: MuJoCo Warp on the GPU at those settings (to measure); a serial-equivalent
     twin (`data/robot/dropbear_serial.xml` plus the motor LUTs) for previews; or run the preview on a cloud GPU.
     On the real robot none of this matters: the loop is the motors.
   - The library policy exported with `export_library.py`.
   - Then the prompt loop runs with the bridge's viewer in real time.
   - 2026-09-26: the real-time preview runs in Isaac instead (CPU pipeline, see "Real-time live session").
2. **Generation:** Kimodo-G1 with the text encoder on the CPU (12 GB of VRAM).
   - Measure generation latency and generate short chunks (2-4 s) conditioned on the current pose, so transitions
     come from the model rather than a cross-fade.
   - Check NVIDIA ARDY, the real-time Kimodo (the gstack browse daemon would not start on 2026-09-26, so it is
     unverified).
3. **A tracker for any text:** generate thousands of motions from text prompts with Kimodo, retarget them with
   Froude timing, filter them for motor feasibility (`tools/hw_motion_feasibility.py`), and train the library tracker
   on them with the hardware penalties. This is the GR00T / SONIC recipe.
   - It needs cloud GPUs: the local 4080 laptop is about 4x slower than one RTX 6000 Ada.
4. **Safety for the robot:** keep the feasibility / safe-clip gate in front of the generator. A generated motion
   beyond the motors must be refused or slowed down, not attempted.

## First generated motion on the twin (2026-09-26 13:00)

- **Prompt:** "a person waves hello with the right hand".
- **Pipeline:** Kimodo-G1 (4 s) -> `retarget_g1.py --time-scale 1.19` (3 s) -> settle + validate (PASS, 0 % saturation,
  0 frames over the no-load speed) -> `lib_v6ts_nostate_v1i_fix/10492` (CEM-60 knee, all hardware fixes) on hw_v1i.
- **Result:** 0 falls, body error 3.6 cm, 200 Hz jumps 2 %.
- **HW gate FAIL: right elbow at 1.4x rated.** Waving holds the forearm up, and through the 4.8x speed-up linkage the
  X8 Pro sees 4.8x the gravity torque (ISSUES #24). Arm-raised motions need a stronger elbow motor or a different
  linkage ratio; gains cannot fix gravity.
- Video: `logs/brev/media/kimodo_gen_wave_cem60_dashboard.mp4`.

## Generated training data (2026-09-26 13:00-13:22)

- **60 everyday prompts** (`data/motions_gen/prompts/everyday_v1.json`: walk variants, turns, gestures, squats,
  steps, light dance, exercise) -> `tools/kimodo_batch_generate.py`, model loaded once (4 s warm): **6.8 s per clip**
  on average (text 2.7 s on the Windows CPU + diffusion).
- **Retarget (Froude x1.19) + settle + validate:** 60/60 accepted (`logs/hw_twin/build_v7gen.sh`). 51/60 never exceed
  a motor's no-load speed; 9 do briefly (at most 3 % of frames: brisk walks, march, knee raises, boxing).
- **Library `accepted_v7gen`:** 138 clips (v6 + 60 generated).
- **13:22:** Brev GPU3 trains `lib_v7gen_v1ie_smooth` (warm start from `lib_v6ts_v1ie_smooth`, hw_v1ie, all hardware
  terms). This is the first step of the "tracker for any text" recipe.

## Live generation service (2026-09-26 13:35)

- **`tools/kimodo_live_service.py`** (WSL, warm):
  1. A prompt file in `logs/live/prompts/` (`prompt` or `duration|prompt`).
  2. Kimodo-G1 generates (50 steps).
  3. The Froude retarget runs (foot-contact solver).
  4. A **kinematic** contract clip is built: motors and root from the retarget, the anchor rigid with the root, the
     rest from a settled template.
  5. The clip is written atomically to `logs/live/inbox/`, which `play.py --live_dir` / `policy_runner --live_dir`
     splice.
- **Measured:** ready in 14 s. "walk forward slowly" (5 s clip): **8.2 s** (generate 4.1 + retarget 4.1).
  "wave hello" (first call): 12.3 s.
- **Tracking check** (`logs/hw_twin/live_kimodo1`, `lib_v6ts_nostate_v1i_fix/10492` on hw_v1i):
  - both generated clips spliced live (wave at 1.1 s, walk at 7.1 s);
  - 0 falls, anchor rotation error 0.09, **HW gate PASS**, 200 Hz jumps 2 %.
  - Video: `logs/brev/media/live_kimodo1_dashboard.mp4`.
- **Run it:**
  - Windows: `.venv-embed/Scripts/python.exe tools/kimodo_text_embed_server.py` (started by Claude on 2026-09-26).
  - WSL: `G1_MJCF=$DROPBEAR_UPSTREAM/unitree_mujoco/unitree_robots/g1/g1_29dof.xml HF_HOME=$DROPBEAR_HF_CACHE
    HF_HUB_OFFLINE=1 TEXT_ENCODER_MODE=file PYTHONPATH=source .venv-kimodo-wsl/bin/python tools/kimodo_live_service.py`.
  - Viewer: superseded by `tools/live_session.sh` (below).

## Real-time live session (2026-09-26 17:01)

One command, on the laptop: `bash tools/live_session.sh [minutes]`, then open **http://127.0.0.1:8765** and type a
motion (or `python tools/live_prompt.py` in a terminal).

```
 browser page (tools/live_web.py) ──POST /prompt──► logs/live/prompts/*.txt
   ▲                                                      │
   │ /state (30 Hz: every link pose)          text encoder (Windows CPU, int8, memory-mapped)
   │                                                      │  embedding over the embed_bridge folder
 logs/live/state.bin ◄── scripts/play.py --device cpu     ▼
   (state_share seqlock)   --realtime --live_dir    Kimodo-G1 (WSL GPU) -> Froude retarget -> kinematic clip
                           Isaac PhysX, DatasheetMotor,       │
                           27 closures, 200 Hz        ◄───────┘ logs/live/inbox/*.npz (spliced at the playhead)
```

- **Measured (final run, 5 prompts in 2 min, one typed in the page):** prompt -> clip 8.5-10.6 s (text 4.1-4.7 s on the
  CPU, diffusion 6.1-6.5 s, retarget 2.0-3.9 s; the encoder runs inside the generation call). All 5 motions (wave,
  walk forward slowly, shallow squat, raise both arms, turn left) spliced live, **0 falls**. Physics 1.00x on the
  page between prompts; 0.92x averaged over the 2 minutes (a prompt's encode + retarget share the CPU for a few
  seconds). Memory headroom stayed >= 5.2 GB.
- **What made it real time (all 2026-09-26):**
  1. Isaac's CPU pipeline for one robot: 5x faster than the GPU pipeline (ISSUES #30).
  2. `--lean` (implied by `--realtime`): no reward terms, actor observations only, tracking metrics every 5th step;
     physics and policy inputs unchanged.
  3. A zero-lag `DelayBuffer` is skipped in `DatasheetMotor` (play has no latency randomization; exact).
  4. The motion command indexed the whole reference before picking the frame; with the 15000-frame live buffer that
     was 5.5 ms per step (ISSUES #31). Fixed; also 3 ms faster for ordinary clips.
  5. The physics process opts out of Windows EcoQoS and runs above normal priority; without that, an unfocused
     console process ran 2.3x slower while a GUI window had focus (efficiency cores).
- **Viewer:** a separate process, because a viewport frame costs ~28 ms inside the physics loop (0.43x). Options:
  the browser page (default, ~0.1 GB), `tools/live_viewer_gl.py` (Newton OpenGL window, ~5 GB committed, `VIEWER=gl`),
  or `scripts/live_viewer.py` (a second Isaac process, ~13 GB: does not fit with the rest on this laptop).
- **Memory was the hard limit** (31.6 GB RAM, pagefile on a nearly full C:): Llama-3 encoder bf16 16.4 GB + two Isaac
  processes + Kimodo ran out of commit (Win32 error 1455, a crash, 15:33). Fixes: the text encoder as weight-only int8
  with per-row scales (`tools/build_llm2vec_w8.py`, built by streaming the safetensors, both LoRAs merged in fp32),
  **memory-mapped read-only** from `$DROPBEAR_HF_CACHE/llm2vec_llama3_8b_sup_w8/` (1.2 GB committed instead of 16.4, loads
  in 1 s). Fidelity: cosine 0.9995 min / 0.9996 median vs bf16 on the 60 library prompts (the closest pair of
  distinct prompts is 0.10 apart). Dynamic int8 (activations quantized too) was rejected: cosine 0.82.
- **After a fall** the reference restarts at the stand clip (`LiveReference.restart`), so the robot gets up and waits
  for the next prompt instead of replaying the timeline (in the first run the same squat was replayed into 3 falls).
- **What it proves / does not:** the simulated motor twin (datasheet torque-speed envelopes, friction, the 20 ms
  target ramp, 27 closed loops) tracks freshly generated motions in real time. It does not prove the real robot:
  the tracker is `lib_v6ts_nostate_v1i_fix/10492`, which holds 41/78 library clips on the twin; generated clips are
  kinematic (not physics-settled) and are not checked by the hardware feasibility gate before they play.
- **Hardware speed gate (17:20):** every generated clip is screened before it reaches the robot, with the law of
  `tools/hw_motion_feasibility.py`: the 99th-percentile reference speed of each motor vs its no-load speed
  (`--profile hw_v1i`). A clip that is too fast is slowed down just enough (5 % margin), up to `--max_stretch 1.6`;
  beyond that it is refused and logged. The 14 clips generated today all passed unchanged (worst: a hip at 0.61x
  no-load); the same walk sped up 2.5x is slowed 1.53x (hip 1.46x -> 0.91x no-load), 3x is refused. It screens
  speed only: torque and the hard stops are handled by the tracker (target clamp) and shown by the telemetry gate.
- **HW gate on a recorded live session** (`logs/hw_twin/live_final`: the five clips of the 17:01 session replayed at
  the same sim times, 70 s): **PASS**, 0 falls, 200 Hz torque jumps 1 %. Warnings are the elbows (right elbow lags
  10.5 deg, both within 3 deg of a stop 60 % of the time, right elbow 1.01x rated: ISSUES #24) and the right ankle
  rod at 1.08x rated. Video: `logs/brev/media/live_final_dashboard.mp4`.
- **Tracker choice (17:30, `logs/hw_twin/live_checks`):** `lib_v7gen_v1ie_smooth/13000` (library + 60 Kimodo
  everyday motions, hw_v1ie) on the same five live clips: 0 falls, HW gate PASS, and cleaner than
  `lib_v6ts_v1i_fix/10492`: no elbow lag (the hw_v1ie elbow), 200 Hz jumps 0 % vs 1 %, left hip RMS 15.7 vs 23.3 N*m,
  no ankle-rod thermal warning; new warnings: right elbow clipped 9.5 %, torso tilt 7.3 deg. `live_session.sh` now
  defaults to the newest pulled v7gen checkpoint (fallback v6).
- **Fall recovery verified:** a clip known to fall on the twin (CR7) at 2 s, then a generated wave at 20 s: fall at
  5.0 s, "reference restarted at the idle clip", the wave spliced and played, 1 fall total.
- **Library eval of the final `lib_v7gen_v1ie_smooth/13600`** (18:39, 138 clips, hw_v1ie): 51/60 Kimodo-generated everyday clips and 43/78 original library clips with 0 falls (`lib_v6ts_v1i_fix2/13700` holds 46/78 of the original). Generated motions it still drops: arms up, bounce, boxing, clap, knee raise, reach down, turn around, walk circle, walk turn right. That is the live tracker's known weak set.
- **Latency:** repeated prompts now skip the text encoder (cache in the server).
- **Remaining:** latency (a 4 s encode on the CPU is the biggest single piece of a new prompt; ARDY / a smaller text
  encoder), and the final `lib_v7gen` library eval (end-of-session job after 18:10).
