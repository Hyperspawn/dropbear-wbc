# Cloud training on Brev

> **Update (2026-09-26):** this plan has since been run: two 4x RTX 6000 Ada sessions (2026-09-25/26, about $35 each
> capped by the watchdog) trained every published policy. What actually ran, hour by hour, is in
> [OVERNIGHT_BREV_2026-09-25.md](OVERNIGHT_BREV_2026-09-25.md); the scripts are `tools/brev/`. The CLI now lives at
> `~/.local/bin/brev` (v0.6.335, `--no-check-latest </dev/null`). The original plan follows unchanged.

**Status (2026-09-24 21:55): PLAN, UNVERIFIED on Brev.** Every script used below has run on the local Windows laptop
(the Linux variants differ only in the Python launcher). No brev command has been run by an agent.
- **Decision still in force** (DECISIONS 2026-09-24): local proof first, cloud A100s afterwards.
- **CLI state.** The Brev CLI v0.6.323 is installed in WSL Ubuntu at `/usr/local/bin/brev`, and its session is logged
  out.
- **Who does what.** **The user runs `brev login`, creates the instance and runs the `brev` commands.** Agents never run
  brev commands and never handle API keys or tokens.

Written by the multiclip track (progress and evidence: `logs/multiclip/PROGRESS.md`). The first cloud workload is the
motion-library tracker (CONTRACTS 5.3). Locomotion (CONTRACTS 7, `docs/LOCOMOTION.md`) and a SONIC-native tracker come
later (section 8).

## 0. What is verified locally (the starting point)

| item | evidence | status |
|---|---|---|
| library env: build, RSI over (clip, bin), command and future slices == NPZ, play assignment | `logs/multiclip/library_env_probe_r2.json` (ok=true) | VERIFIED (GPU) |
| 20-iteration PPO smoke, `Dropbear-Tracking-Library-v0` and `-Future-v0`, 2048 envs, 8/4 | `train_lib_smoke20_r2.log`, `train_lib_future_smoke20_r2.log`, `smoke20*_r2_per_clip.json` | VERIFIED: finite, per-clip metrics logged |
| play + runtime-reference export + ONNX parity + deploy runner fed an NPZ | `play_lib_smoke20_r2_export.log`, `check_library_export_smoke20_r2_v2.json` (ok=true) | VERIFIED |
| chunked training with `--motion_library` (2 chunks x 10 it; chunk 2 restored learning rate + bin EMAs + clip EMAs) | `chunk_smoke_r2_driver.log`, `train_lib_chunk_smoke_r2_chunk2.log` | VERIFIED |
| a *trained* library policy (tracking quality) | none yet | NOT DONE: the first real run is section 5 |

## 1. What to run first (in this order)

1. **Environment check** (about 15 min of instance time). Run `tools/library_env_probe.py`, then a 20-iteration smoke
   (section 5, steps 1-2). Together they prove Isaac Sim, Isaac Lab, the vendored rsl-rl, the USD, the calibration and the
   manifest on the box. Compare the probe JSON with the local `library_env_probe_r2.json`: all `*_err` values must be
   0.0, with the same obs/action dims and bin count.
2. **Throughput sweep** (about 20 min). Run three 12-iteration smokes at 4096, 8192 and 16384 envs (section 5, step 3).
   Pick the env count with the best env-steps/s whose GPU memory stays below about 80 %. This number replaces every
   ESTIMATE in section 6.
3. **First real run.** Train `Dropbear-Tracking-Library-v0` on `data/motions/libraries/accepted_v1.json` (6 clips, 73 s)
   as a chunked run (section 5, step 4). In parallel, on a second GPU, run the same command with
   `Dropbear-Tracking-Library-Future-v0`. That is the A/B on the future-window command, per the DECISIONS entry.
4. **Evaluate and export.** Evaluate per clip (32/4, one env per clip) and export with a runtime reference (section 5,
   step 5). Copy the run home and check the export locally with `tools/check_library_export.py`.
5. **Later.** Train on bigger manifests as the foot-contact track gets dances and walks accepted. Regenerate the manifest
   with `tools/build_motion_library_manifest.py` (section 4). Bigger libraries are where cloud GPUs pay off: SONIC-scale
   libraries hold hours of motion, and ours holds 73 s.

## 2. Instances: which A100, and how many

**Recommendation.** Start with **2 x A100 40 GB**: one instance with 2 GPUs, or two 1-GPU instances.
- **Split.** GPU 0 runs `Library-v0` and GPU 1 runs `Library-Future-v0`: two independent single-GPU processes.
- **When to use 80 GB.** Only for more than about 16k envs per GPU, or for rendering video on the same GPU as training.

| option | memory need | when |
|---|---|---|
| **A100 40 GB** (first choice) | about 2 GB per 1k envs at 8/4 (local: 4.2 GB at 2048 envs, env + solver, CONTRACTS 5). 8192 envs is about 17 GB, 16384 is about 33 GB | every library run up to 16k envs |
| A100 80 GB | 16384-32768 envs; or 2 processes per GPU; or training plus camera rendering | only if the sweep shows throughput still rising above 16k envs |
| L40S 48 GB (measure it too) | like A100 40 GB | Ada GPU with RT cores: often cheaper, and PhysX is not tensor-core bound, so it may beat an A100 per dollar. Also renders demo video |

- **Why not one 8-GPU box?** Each run is **single-GPU**: `scripts/train.py` has no `torch.distributed` launch (rsl-rl
  2.3.3 supports multi-GPU, but our entry point does not wire `--distributed`, and it is untested). More GPUs therefore
  mean more *parallel runs*: the A/B, seeds, locomotion. A single run does not get faster. Multi-GPU per run is an open
  item for SONIC-scale libraries (section 8).
- **Host needs.** Ubuntu 22.04, the Linux platform Isaac Sim 5.0 officially supports. NVIDIA data-centre driver in the 535+ series (check
  the Isaac Sim 5.0 requirements page for the exact minimum). >= 64 GB RAM, >= 100 GB disk, >= 16 vCPUs.
- **Pinning two processes.** On a 2-GPU box, pin each process with `CUDA_VISIBLE_DEVICES=0` / `=1`; each process then
  sees its GPU as `cuda:0`. Use a separate `--log_dir` and `--run_name` for each.

## 3. Environment setup on the instance (Linux)

Pick one route. Both must end with **Isaac Sim 5.0.0 + Isaac Lab 2.2.0**, the versions every contract result was
produced with (CONTRACTS 0).

**A. pip in a Python 3.11 venv (lightest; recommended):**

```bash
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv git
python3.11 -m venv ~/isaac && source ~/isaac/bin/activate
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab && cd ~/IsaacLab && git checkout v2.2.0
./isaaclab.sh --install none        # core + isaaclab_rl wrappers, NO learning frameworks: never install another rsl-rl
export OMNI_KIT_ACCEPT_EULA=YES
python -c "import isaacsim, isaaclab; print('isaac ok')"
```

**B. Container.** Use NVIDIA's `nvcr.io/nvidia/isaac-sim:5.0.0` image (NGC login needed) plus Isaac Lab v2.2.0 mounted
or installed inside it (`./isaaclab.sh --install none`, then `./isaaclab.sh -p <script>`). Choose this if the Brev
image already has Docker and the NVIDIA container toolkit. Scripts then run with `/isaac-sim/python.sh -u` or
`~/IsaacLab/isaaclab.sh -p`.

In both routes:

- **Vendored rsl-rl only.** Our scripts put `third_party/pydeps` (rsl-rl-lib **2.3.3**) first on `sys.path` and stop
  otherwise (`isaac.launch.assert_vendored_rsl_rl`, which also checks the dist-info version). The train log prints the
  `rsl_rl_path`, and it must be under `third_party/pydeps`.
- **`ISAAC_SIM_ROOT`.** It defaults to `C:/isaac-sim` and only matters for picking the bundled Warp. Set it to the Isaac
  Sim root on route B (`/isaac-sim`); on route A it is harmless to leave unset.
- **Headless.** Pass `--headless` to every script. Do not pass `--enable_cameras` or `--clean_video` on training boxes
  (`demo_render.py` hard-codes the Windows ffmpeg and font paths; render videos locally).

## 4. Data sync (from WSL; the user runs these)

The brev CLI lives in WSL. Windows drives appear as `/mnt/h`, `/mnt/p` and `/mnt/c`. **The host alias is TBD.** Below it
is written `<instance>`; the last instance was called `dropbear-training`, so the prefix is `dropbear-training:`.

```bash
# (Windows or WSL, any Python) 1. regenerate the manifest from the verdicts if clips were accepted since, then pack.
#    The bundle is ~13 MB: code, vendored rsl-rl, calibration, each manifest + its NPZs + their verdicts, BUNDLE.json with SHA-256s.
cd <repo>
python3 tools/build_motion_library_manifest.py --dry-run            # "unchanged" or the new accepted_v<N> it would write
python3 tools/pack_brev_bundle.py --manifest data/motions/libraries/accepted_v1.json \
    --extra data/motions/smoke/dropbear_static_stand.npz --out logs/brev/dropbear-wbc-brev.tgz
sha256sum logs/brev/dropbear-wbc-brev.tgz

# (WSL) 2. log in (user), then copy the bundle and the plant USD (421 MB USDC, self-contained; read-only original on P:)
brev login
brev ls
brev copy logs/brev/dropbear-wbc-brev.tgz <instance>:~/dropbear-wbc-brev.tgz
brev copy $DROPBEAR_USD <instance>:~/assets/dropbear.usd
```

(`tools/build_motion_library_manifest.py` needs numpy; `tools/pack_brev_bundle.py` is stdlib-only. Both run from
Windows too.) If `brev copy` is unavailable in this CLI version, `brev refresh` writes an SSH alias. Then use
`rsync -avP <file> <instance>:~/`.

On the instance:

```bash
mkdir -p ~/dropbear-wbc && tar xzf ~/dropbear-wbc-brev.tgz -C ~/dropbear-wbc && cd ~/dropbear-wbc
sha256sum ~/assets/dropbear.usd      # MUST be 45586414b065cd982d487cbd868fe982108b3b8ccec64d3dfcf629652ed8db0f
python3 - <<'EOF'                    # every file of the bundle arrived intact
import hashlib, json; b = json.load(open("BUNDLE.json"))
bad = [f for f, i in b["files"].items() if hashlib.sha256(open(f, "rb").read()).hexdigest() != i["sha256"]]
print("bundle ok" if not bad else f"CORRUPT: {bad}")
EOF
export DROPBEAR_USD=~/assets/dropbear.usd OMNI_KIT_ACCEPT_EULA=YES
export DROPBEAR_CALIBRATION_JSON=$PWD/data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json
```

- **The USD check is manual.** `$DROPBEAR_USD` switches off the code's USD-SHA checks (CONTRACTS 0, 5.1), so the manual
  `sha256sum` above is the check. Do not skip it. The NPZs still have to match the plant by names and ankle variant.
- **Calibration.** The calibration is the pinned snapshot `e51033d4`, the same default pose as every local library run.

Results back to Windows (WSL):

```bash
brev copy <instance>:~/dropbear-wbc/logs/rsl_rl/dropbear_tracking_library/<run> \
    logs/rsl_rl/dropbear_tracking_library/<run>
brev copy <instance>:~/dropbear-wbc/logs/brev logs/brev
```

## 5. Commands (instance, from `~/dropbear-wbc`, venv active)

`PY="python -u"` (route A). Route B uses `PY="/isaac-sim/python.sh -u"` or `PY="$HOME/IsaacLab/isaaclab.sh -p"`.
Everything writes under `logs/` in the repo.

Step 1: env probe (256 envs, 150 zero-action steps; about 3 min):

```bash
mkdir -p logs/brev
$PY tools/library_env_probe.py --task Dropbear-Tracking-Library-Future-v0 \
    --motion_library data/motions/libraries/accepted_v1.json --num_envs 256 --steps 150 \
    --out logs/brev/library_env_probe.json --headless 2>&1 | tee logs/brev/library_env_probe.log
```

Step 2: 20-iteration PPO smoke (finite metrics and per-clip keys in `metrics.jsonl`):

```bash
$PY scripts/train.py --task Dropbear-Tracking-Library-v0 --motion_library data/motions/libraries/accepted_v1.json \
    --num_envs 4096 --max_iterations 20 --solver_iters 8 4 --run_name brev_smoke20 --headless 2>&1 | tee logs/brev/smoke20.log
```

Step 3: throughput sweep. Read `steps_per_s` in `metrics.jsonl`, ignoring iteration 0, and watch `nvidia-smi` in a
second shell:

```bash
for N in 4096 8192 16384; do
  $PY scripts/train.py --task Dropbear-Tracking-Library-v0 --motion_library data/motions/libraries/accepted_v1.json \
      --num_envs $N --max_iterations 12 --save_interval 1000 --solver_iters 8 4 --run_name brev_tp$N --headless \
      2>&1 | tee logs/brev/tp$N.log
done
python3 tools/summarize_training.py logs/rsl_rl/dropbear_tracking_library/*_brev_tp8192   # or read metrics.jsonl
```

Step 4: the first library run, chunked. Chunks survive preemption and crashes: each chunk resumes from the last
checkpoint, restores the learning rate and the sampler state (per-bin failure EMAs plus clip hazards), and is capped by
`--timeout`. `N` is the env count from step 3 (8192 assumed). 300 iterations x 10 chunks = 3000 iterations.

**One checkout per GPU.** `tools/gpu_lock_run.py` (used per chunk) serialises on ONE lock file,
`<repo>/.locks/gpu.lock`, so two runs in the same checkout would take turns. For the parallel A/B, unpack the bundle
twice, as `~/dropbear-wbc-a` and `~/dropbear-wbc-b`, and start one run in each (or use two instances):

```bash
mkdir -p ~/dropbear-wbc-b && tar xzf ~/dropbear-wbc-brev.tgz -C ~/dropbear-wbc-b && mv ~/dropbear-wbc ~/dropbear-wbc-a
N=8192
CAL=data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json
cd ~/dropbear-wbc-a && mkdir -p logs/brev && CUDA_VISIBLE_DEVICES=0 nohup python3 tools/run_chunked_training.py \
    --task Dropbear-Tracking-Library-v0 --motion_library data/motions/libraries/accepted_v1.json \
    --run_name lib_v1_brev_a --chunks 10 --iters_per_chunk 300 --num_envs $N --solver_iters 8 4 --save_interval 100 \
    --seed 101 --timeout 7200 --gap_s 5 --log_dir logs/brev --owner brev_gpu0 --isaac_python "python" \
    --calibration $CAL > logs/brev/lib_v1_brev_a_driver.log 2>&1 &
# A/B on GPU 1: the future-window command (+0.1/+0.2 s), same seed, everything else identical
cd ~/dropbear-wbc-b && mkdir -p logs/brev && CUDA_VISIBLE_DEVICES=1 nohup python3 tools/run_chunked_training.py \
    --task Dropbear-Tracking-Library-Future-v0 --motion_library data/motions/libraries/accepted_v1.json \
    --run_name lib_v1_brev_fut --chunks 10 --iters_per_chunk 300 --num_envs $N --solver_iters 8 4 --save_interval 100 \
    --seed 101 --timeout 7200 --gap_s 5 --log_dir logs/brev --owner brev_gpu1 --isaac_python "python" \
    --calibration $CAL > logs/brev/lib_v1_brev_fut_driver.log 2>&1 &
tail -f ~/dropbear-wbc-a/logs/brev/lib_v1_brev_a_driver.log   # one JSON line per chunk: rc, last_it, ep length, finite, steps/s
```

- **Export the environment variables in the shell that starts the drivers.** `DROPBEAR_USD` and `OMNI_KIT_ACCEPT_EULA`
  (section 4) are inherited by every chunk.
- **Resume after a stop or crash.** Rerun the same command with `--resume_run <run dir name>` and set `--chunks` to the
  number of chunks remaining.
- **Monitoring.** `python3 tools/summarize_training.py <run dir>`. The per-clip curves are in `metrics.jsonl` under
  `episode["Metrics/motion/clip_prob/<clip>"]`, `clip_fail_per_s/<clip>` and `clip_err_joint/<clip>`.

Step 5: per-clip evaluation and a runtime-reference export (one env per clip, 32/4 solver check, 2 loops). This runs on
the box or locally after copying the run home. It writes `play_<stamp>.json` with a `per_clip` block, plus
`<run>/exported/` (`policy.onnx`, `policy.json` with `motion.source: runtime`, no `policy_motion.onnx`):

```bash
$PY scripts/play.py --task Dropbear-Tracking-Library-Play-v0 --motion_library data/motions/libraries/accepted_v1.json \
    --load_run <run dir name> --num_envs 6 --steps 2000 --solver_iters 32 4 --continuous_loop --export --headless
python3 tools/check_library_export.py logs/rsl_rl/dropbear_tracking_library/<run>/exported --out logs/brev/check_export.json
# any accepted clip can then drive the exported policy (sim2sim / hardware runner):
python3 tools/policy_runner.py --mode policy --sidecar logs/rsl_rl/dropbear_tracking_library/<run>/exported/policy.json \
    --motion data/motions/synthetic/wave_right_v2.npz --allow-privileged ...
```

## 6. Throughput (measured locally; cloud = ESTIMATE until step 3 has run)

A PPO iteration is 24 env steps x N envs. At 2048 envs, 96 % of it is simulation: collection 3.95 s, learning 0.17 s.

| where | envs | env-steps/s | s / iteration | source |
|---|---:|---:|---:|---|
| RTX 4080 Laptop 12 GB, library task, 8/4 (PPO) | 2048 | **11.9k** (Library-v0), **12.0k** (-Future); 9.7-14.5k in the chunk smoke (shared machine) | 4.1 | VERIFIED `smoke20*_r2_per_clip.json`, chunk smoke |
| RTX 4080 Laptop, single-clip, 8/4 (PPO) | 2048 | 11.7-14.0k | 3.7-4.2 | VERIFIED take102 / wave runs |
| RTX 4080 Laptop, library task, 8/4 (PPO) | 4096 | **15.0k** (saturated: +5-25 % for 2x envs) | 6.4 | VERIFIED `train_lib_tp4096_r2.log` |
| A100 40/80 GB (ESTIMATE) | 8192 | ~20-35k | ~6-10 | 1.3-2.3x the laptop's saturated 15k |
| L40S / H100 (ESTIMATE) | 8192-16384 | ~25-50k | | measure in step 3 |

- **The laptop GPU is saturated at 2048-4096 envs** (about 15k env-steps/s at most), so the local ceiling is about
  54 M env-steps per hour when the GPU is not shared.
- **The library costs no throughput.** Library-v0 and -Future run at single-clip speed: the reference lookup is an
  index into concatenated tensors (6 clips, 3656 frames, 5.5 MB on the GPU).
- **Scaling model.** PhysX GPU throughput grows sub-linearly with env count until the GPU saturates. On the laptop it
  went 2.1k -> 7.5k -> 13.5k env-steps/s (zero-action) for 256 -> 1024 -> 2048 envs (CONTRACTS 5).
- **Main uncertainty.** This robot is expensive per env: 90 links with convex-hull colliders and 27 loop closures
  solved as maximal-coordinate joints. PhysX is bound by memory and latency, not by tensor cores. So an A100 is
  **not** expected to be many times faster than the laptop GPU per run. The cloud wins by having dedicated,
  uninterrupted GPUs (locally, 4 tracks share one), several of them in parallel, and more envs per iteration (better
  gradients).

## 7. Iterations and GPU-hours

What single clips needed locally, at 2048 envs and 8/4 (episode length hits the 500-step cap = 10 s at 50 Hz):
- `wave_right`: 270 iterations to a mean episode length >= 450 (13 M env-steps).
- `G1_Take_102` (dance, reference rejected): 974 iterations (48 M).
- A polished BeyondMimic policy trains much longer (upstream: 4096 envs, >10k iterations).

| run | budget | local laptop (11.9k/s) | 1 x A100 (ESTIMATE 20-35k/s) |
|---|---|---|---|
| env check + smoke + sweep | about 10 M env-steps + startup | 0.3 h | **~0.5 GPU-h** (mostly Isaac start-up) |
| **accepted_v1 library, first real run** | 3000 iterations at 8192 envs = **590 M env-steps** (the same samples as about 12k iterations at 2048 envs) | 13.8 h (impractical while shared) | **~4.7-8.2 GPU-h** |
| its Future A/B (parallel GPU) | same | | ~4.7-8.2 GPU-h |
| first campaign total | | | **~10-17 GPU-h** (2 GPUs for ~5-8 h wall) |
| early stop | stop when every clip's `clip_fail_per_s` is < 0.05 and `clip_err_joint` has stopped falling for 500 iterations (`metrics.jsonl`, `Metrics/motion/clip_*`) | | usually cheaper |

The cost is those GPU-hours times the hourly rate in the Brev console; this plan quotes no prices. Chunk start-up
(Isaac launch, env build) adds about 1.5 min per chunk: 10 chunks add about 15 min per run.

## 8. Later workloads on the same setup

- **Bigger libraries** (foot-contact dances and walks, minutes of motion). The same commands apply with a new
  manifest. Budget grows with the motion: roughly 1-3 B env-steps (about 8-42 A100 GPU-h per run at the rates above).
  Beyond that, wire `torch.distributed` into `scripts/train.py`: rsl-rl 2.3.3 `multi_gpu_cfg`, one env set per rank,
  `--device cuda:<local_rank>`. Verify it locally-equivalent on a 2-GPU box before scaling.
- **Locomotion** (`Dropbear-Velocity-Flat-v0`). Use `tools/run_chunked_locomotion.py` (same chunk/resume design, owner
  and log dir flags; `--stand_npz data/motions/smoke/dropbear_static_stand.npz`, which the bundle `--extra` above
  ships). H1-flat-style runs are typically 3-5k iterations at 4096 envs (0.3-0.5 B env-steps, about 3-7 A100 GPU-h).
  Run it only after the local run shows the knee-authority problem solved (`logs/locomotion/PROGRESS.md`).
- **SONIC-native tracker.** Its body-keypoint commands would be a new task id (DECISIONS: a new task, not a change of
  this one). It needs multi-GPU training and a library of hundreds of accepted clips first. Plan it as 100s of
  GPU-h, after the accepted_v1 run shows that one policy tracks all clips.

## 9. Pitfalls to check on the first cloud run

- **rsl-rl.** `run_info.json` `rsl_rl_path` must end in `third_party/pydeps/rsl_rl`.
- **USD identity.** Check it with `sha256sum`, as in section 4. With `$DROPBEAR_USD` set, the code skips the SHA check.
- **Clips and verdicts.** The library refuses clips without an accepted, non-stale `<clip>.validation.json`. It also
  refuses clips whose bytes differ from the manifest's `sha256` pin. Ship clips with `tools/pack_brev_bundle.py`, never
  by hand.
- **Windows-only defaults.** `tools/run_chunked_training.py` defaults to `C:/isaac-sim/python.bat`, so always pass
  `--isaac_python`. `ISAAC_SIM_ROOT` defaults to `C:/isaac-sim`. `demo_render.py` has Windows ffmpeg and font paths.
- **Calibration pinning.** Passing a snapshot to `--calibration` no longer creates `snapshots/snapshots/...` (fixed
  2026-09-24 21:35). Older drivers did create it, which was harmless but untidy.
- **Disk.** Checkpoints are about 7 MB each. `--save_interval 100` over 3000 iterations writes 30 of them per run.
