# Quick start

Commands run from the repository root. `<isaac-python>` is Isaac Sim's python (`C:/isaac-sim/python.bat` on Windows,
`python` in the Linux venv; [INSTALL](INSTALL.md)). Isaac GPU jobs can go through `python tools/gpu_lock_run.py --owner
<name> --log <file> -- <command>`, which serializes jobs on a shared GPU.

`python tools/fetch_assets.py` first: it downloads the plant USD and the trained policies (`assets/policies/<run>/`).

## 1. Without a GPU

```bash
python -m pytest -q tests -k "not isaac"                            # CPU unit tests (~240 pass)
python tools/telemetry_issue_scan.py path/to/telemetry.npz --md scan.md   # the hardware gate on any recorded rollout
python tools/retarget_g1.py --list-sources                          # motion sources the retargeter can see
python tools/fetch_assets.py --groups media --list                  # the published dashboard videos
```

The browser demo needs nothing: [hyperspawn.github.io/dropbear-wbc](https://hyperspawn.github.io/dropbear-wbc/). To run it
locally: `python -m http.server 8000 --directory site`.

## 2. Watch a trained policy on the motor twin

A deployable walk, with a per-motor dashboard video and the hardware-gate verdict:

```bash
bash tools/make_dashboard.sh assets/policies/kimodo_ts119_allfix_smooth/model_5800.pt \
     data/motions_cyclic/kimodo_walk_ts119_cyclic.npz hw_v1ie kimodo_walk 1000
# -> logs/dashboards/kimodo_walk_dashboard.mp4, logs/dashboards/kimodo_walk/issue_scan.md
```

(`data/motions_cyclic/` comes with `python tools/fetch_assets.py --groups motions`.)

The same thing by hand, headless, with telemetry:

```bash
export DROPBEAR_CALIBRATION_JSON=data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json
<isaac-python> -u scripts/play.py --task Dropbear-Tracking-Flat-NoState-Play-v0 \
    --motion_file data/motions_cyclic/kimodo_walk_ts119_cyclic.npz \
    --checkpoint assets/policies/kimodo_ts119_allfix_smooth/model_5800.pt \
    --actuator_profile hw_v1ie --solver_iters 32 4 --num_envs 1 --steps 1000 \
    --telemetry logs/kimodo_walk_telemetry.npz --headless
python tools/telemetry_issue_scan.py logs/kimodo_walk_telemetry.npz --md logs/kimodo_walk_scan.md
```

`play.py` reads the run's `run_info.json` next to the checkpoint (calibration, actuator profile, target clamp), so the
policy is replayed with the settings it was trained with. Add `--clean_video` for an mp4, `--perturbed --seed 7` for
training-level randomization, `--export` for ONNX + sidecar.

**Walkers** (velocity command; the terrain a run was trained on is replayed):

```bash
bash tools/make_dashboard_locomotion.sh assets/policies/vel_rough_v1i_b/model_5500.pt hw_v1i rough_terrain
<isaac-python> -u scripts/play_locomotion.py --checkpoint assets/policies/vel_hw_v1_gait3b_ft/model_1300.pt --steps 750 --headless
```

## 3. Evaluate a library tracker clip by clip

One environment per clip, continuous loops, per-clip falls in the summary JSON (next to the checkpoint):

```bash
<isaac-python> -u scripts/play.py --task Dropbear-Tracking-Library-NoState-Play-v0 \
    --motion_library data/motions_v6ts/libraries/accepted_v7gen_public.json \
    --checkpoint assets/policies/lib_v7gen_v1ie_smooth/model_13600.pt --actuator_profile hw_v1ie \
    --num_envs 122 --steps 1700 --solver_iters 32 4 --continuous_loop --headless
```

The published result (94/138 clips with 0 falls) is on the full 138-clip library; the public manifest has the 122
redistributable clips ([DATA](DATA.md)).

## 4. Real-time preview in the browser

Any tracking policy, simulated in real time on the CPU pipeline and drawn in a browser tab:

```bash
python tools/live_web.py &                                           # http://127.0.0.1:8765
<isaac-python> -u scripts/play.py --task Dropbear-Tracking-Flat-NoState-Play-v0 \
    --motion_file data/motions_cyclic/kimodo_walk_ts119_cyclic.npz \
    --checkpoint assets/policies/kimodo_ts119_allfix_smooth/model_5800.pt --actuator_profile hw_v1ie \
    --solver_iters 32 4 --num_envs 1 --steps 3000 --headless --device cpu --realtime --state_out logs/live/state.bin
```

`--realtime` implies `--lean` (no reward terms, actor observations only) and paces the loop to wall-clock time; the
summary's `timing` block reports the real-time factor. One robot runs about 5× faster on `--device cpu` than on the
GPU pipeline (ISSUES #30).

## 5. Live text to motion

Needs [INSTALL](INSTALL.md) tier 3 (Kimodo + the text encoder).

**Windows + WSL2** (how it was built): one command starts the text encoder (Windows CPU), the Kimodo service (WSL GPU),
the browser page and the real-time physics:

```bash
bash tools/live_session.sh 10          # minutes; then open http://127.0.0.1:8765 and type a motion
python tools/live_prompt.py            # or type prompts in a terminal
```

Environment overrides: `CKPT`, `PROFILE` (default: the newest `lib_v7gen` checkpoint on `hw_v1ie`), `FIRST` (start
clip), `VIEWER=gl` (Newton OpenGL window instead of the browser), `WSL_DISTRO`.

**Linux** (one machine): run the four pieces yourself.

```bash
.venv-embed/bin/python tools/kimodo_text_embed_server.py --w8 <hf_cache>/llm2vec_llama3_8b_sup_w8 --low_priority &
bash tools/kimodo_live_service_wsl.sh &                                  # works on plain Linux too
python tools/live_web.py &
python -u scripts/play.py --task Dropbear-Tracking-Flat-NoState-Play-v0 --motion_file data/motions_v6ts/synthetic/stand.npz \
    --checkpoint assets/policies/lib_v7gen_v1ie_smooth/model_13600.pt --actuator_profile hw_v1ie \
    --solver_iters 32 4 --num_envs 1 --steps 30000 --headless --device cpu --realtime \
    --live_dir logs/live/inbox --state_out logs/live/state.bin
```

Each prompt is generated by Kimodo-G1, retargeted to Dropbear with Froude timing, screened against every motor's
no-load speed (slowed down if needed, refused beyond 1.6×), and spliced into the running reference. After a fall the
robot restarts standing and waits for the next prompt.

## 6. Train

Chunked training (restartable chunks, warm starts, the hardware terms), single clip:

```bash
python tools/run_chunked_training.py --run_name my_walk --task Dropbear-Tracking-Flat-NoState-v0 \
    --motion_file data/motions_cyclic/kimodo_walk_ts119_cyclic.npz --actuator_profile hw_v1ie \
    --hw_regularizers --knee_stop_penalty=-10 --feet_slide_penalty=-1 --target_clamp_margin_deg=0 --action_rate_weight=-0.3 \
    --chunks 20 --iters_per_chunk 300 --num_envs 4096 --solver_iters 16 4
```

A library tracker: `--task Dropbear-Tracking-Library-NoState-v0 --motion_library data/motions_v6ts/libraries/accepted_v7gen_public.json`.
Warm start from a run: `--init_run <run dir name>`. Negative weights need the `=` form. On Linux pass
`--isaac_python python` if `.dropbear.env` does not set `ISAACSIM_PYTHON`.

Velocity / terrain:

```bash
python tools/run_chunked_locomotion.py --run_name my_terrain --actuator_profile hw_v1i --gait_v3 --terrain rough \
    --feet_width_weight=-5 --action_rate_weight=-0.05 --target_clamp_margin_deg=0 --thermal_penalty=-2e-4 \
    --chunks 30 --iters_per_chunk 150 --num_envs 4096 --solver_iters 8 4
```

Throughput: about 25.7k env-steps/s for a single clip at 4096 envs on one RTX 6000 Ada (docs/BREV.md). The published
library policies took roughly 13k iterations. Cloud recipe: [BREV](BREV.md).

## 7. Export and sim-to-sim

```bash
<isaac-python> -u scripts/play.py --task Dropbear-Tracking-Flat-NoState-Play-v0 --motion_file <clip.npz> \
    --checkpoint <run>/model_<N>.pt --num_envs 1 --steps 200 --export --headless      # <run>/exported/policy.{onnx,json}
python tools/check_export_parity.py <run>/exported
bash tools/run_sim2sim_cpu.sh mytag <run> <clip.npz> 10 5555 5556                 # Newton / MuJoCo C on the CPU, needs .venv-newton
```

The runner (`tools/policy_runner.py`) is the deploy-side FSM (passive → move to default → hold → policy); the bridge
(`tools/newton_bridge.py --motor-profile hw_v1i --target-ramp-ms 20`) applies the same motor law as training.
Details: [SDK](SDK.md).

## 8. Motion data

Build or rebuild clips and libraries: [DATA](DATA.md). Retarget one G1 motion:

```bash
python tools/retarget_g1.py <g1_motion.csv> --source kimodo_g1 --out-root data/motions_v6ts --time-scale 1.19
<isaac-python> -u tools/settle_motion.py --headless --keep-going --out-suffix _v6 --csv data/motions_v6ts/kimodo_g1/<name>_ts1.19.csv
python tools/validate_motion_npz.py data/motions_v6ts/kimodo_g1/<name>_ts1.19_v6.npz --write-verdicts
```
