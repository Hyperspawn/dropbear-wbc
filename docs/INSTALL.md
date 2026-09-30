# Install

Pick the tier you need. Each builds on the previous one.

| tier | you get | needs |
|---|---|---|
| 0. browser | replays of the motor twin, in 3D, per-motor load | nothing: [hyperspawn.github.io/dropbear-wbc](https://hyperspawn.github.io/dropbear-wbc/) |
| 1. CPU | unit tests, motion pipeline, telemetry scanner, deploy runtime, data tools | Python 3.11+, ~2 GB |
| 2. Isaac | train, play and evaluate policies on the motor twin; dashboards; the real-time preview | NVIDIA RTX GPU (12 GB+), Isaac Sim 5.0.0 + Isaac Lab 2.2.0 |
| 3. text to motion | live "type a motion, watch the robot do it" | tier 2 + Kimodo (Linux/WSL2, CUDA) + the Llama-3 text encoder (gated) |
| 4. cloud training | the 4-GPU runs behind the published policies | an Ubuntu 22.04 GPU box (docs/BREV.md) |

Every command runs from the repository root. Machine-specific paths go in `.dropbear.env` (copy
`dropbear.env.example`); `PYTHONPATH=source python -m dropbear_wbc.paths` prints what is resolved.

## Tier 1: CPU

```bash
git clone https://github.com/Hyperspawn/dropbear-wbc && cd dropbear-wbc
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/tests-cpu.txt huggingface_hub
python -m pytest -q tests -k "not isaac"                   # ~240 pass; Isaac / GPU / optional-stack tests skip
python tools/fetch_assets.py                               # plant USD + trained policies (~450 MB)
```

## Tier 2: Isaac Sim 5.0 + Isaac Lab 2.2

Every contract result was produced with **Isaac Sim 5.0.0 and Isaac Lab 2.2.0** (docs/CONTRACTS.md section 0). The
scripts put the vendored **rsl-rl 2.3.3** (`third_party/pydeps`) first on `sys.path` and refuse to run with any other
version (`isaac.launch.assert_vendored_rsl_rl`), so do not install rsl-rl or another learning framework into Isaac.

**Windows** (how this project was developed; RTX 4080 laptop):

1. Install Isaac Sim 5.0.0 (standalone) to `C:/isaac-sim` (or set `ISAAC_SIM_ROOT` in `.dropbear.env`).
2. Clone Isaac Lab and check out `v2.2.0`; install it into Isaac Sim's python without learning frameworks:
   `isaaclab.bat --install none` (Isaac Lab docs, "binaries installation").
3. Run scripts with Isaac Sim's python: `C:/isaac-sim/python.bat -u scripts/<script>.py ...`

**Linux** (Ubuntu 22.04, pip route; what the cloud runs used):

```bash
sudo apt-get install -y python3.11 python3.11-venv git
python3.11 -m venv ~/isaac && source ~/isaac/bin/activate
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab && cd ~/IsaacLab && git checkout v2.2.0
./isaaclab.sh --install none
export OMNI_KIT_ACCEPT_EULA=YES
echo "ISAACSIM_PYTHON=python" >> /path/to/dropbear-wbc/.dropbear.env   # scripts then use the venv's python
```

**Then, for both:**

```bash
python tools/fetch_assets.py            # assets/dropbear.usd (SHA-256 checked) + assets/policies/
<isaac-python> -u scripts/play.py --task Dropbear-Tracking-Flat-NoState-Play-v0 \
    --motion_file data/motions_v6ts/synthetic/stand.npz \
    --checkpoint assets/policies/lib_v7gen_v1ie_smooth/model_13600.pt --num_envs 1 --steps 300 --headless
```

`<isaac-python>` is `C:/isaac-sim/python.bat` on Windows, `python` in the Linux venv.
[QUICKSTART.md](QUICKSTART.md) has the demos.

Optional environments (all tested on the development machine):

| venv | for | install |
|---|---|---|
| `.venv-newton` (Python 3.12) | deploy-side Newton/MuJoCo plant (`tools/newton_bridge.py`), USD tools, the OpenGL live viewer, GLB export | `pip install -r requirements/newton-windows-lock.txt -r requirements/sdk-extra.txt`, and clone [robit-man/dropbear_control](https://github.com/robit-man/dropbear_control) next to this repository (or set `DROPBEAR_CONTROL_DIR`) |
| `.venv-gmr` (Python 3.11) | LAFAN1 BVH -> Dropbear with GMR | `python tools/setup_gmr.py` (docs/SERIAL_MODEL_AND_GMR.md) |
| `.venv-teleop` (Python 3.12) | XR / keyboard teleoperation | docs/TELEOP.md |

## Tier 3: text to motion (live)

The Llama-3-8B text encoder (16 GB in bf16) and Kimodo do not both fit on a 12 GB laptop GPU, so the encoder runs on
the CPU and Kimodo on the GPU, talking over a shared folder (docs/TEXT_TO_MOTION.md). On a Linux box with a 24 GB+
GPU you can skip the split and let Kimodo load the encoder itself (`TEXT_ENCODER_MODE` unset).

1. **Model access.** Accept Meta's license for `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face and
   `huggingface-cli login`. The other models (`nvidia/Kimodo-G1-RP-v1`, `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`
   and `-mntp-supervised`) download without a gate. Set `DROPBEAR_HF_CACHE` if the default cache disk is small (~17 GB).
2. **Kimodo** (Linux or WSL2 Ubuntu, CUDA):

   ```bash
   bash tools/setup_kimodo.sh                                      # NVIDIA Kimodo @ pinned commit + our overlay
   python3.10 -m venv .venv-kimodo-wsl && source .venv-kimodo-wsl/bin/activate
   pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements/kimodo.txt && pip install -e third_party/kimodo   # needs cmake + g++
   ```
   The G1 retarget needs the Unitree G1 model: `git clone https://github.com/unitreerobotics/unitree_mujoco
   ../upstream/unitree_mujoco` (or set `G1_MJCF`).
3. **Text encoder** (Windows or Linux CPU, ~1.2 GB committed once converted):

   ```bash
   python -m venv .venv-embed && .venv-embed/Scripts/activate      # Linux: source .venv-embed/bin/activate
   pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements/text-embed.txt
   python tools/build_llm2vec_w8.py <hf_cache>/llm2vec_llama3_8b_sup_w8.pt         # int8, ~9 GB peak, once
   python tools/build_llm2vec_w8.py --to_dir <hf_cache>/llm2vec_llama3_8b_sup_w8 <hf_cache>/llm2vec_llama3_8b_sup_w8.pt
   ```
   The int8 encoder matches bf16 at cosine 0.9995 (min) on the 60 library prompts; `tools/embed_quant_check.py`
   re-checks it.
The live page's 3D robot (`site/assets/dropbear.glb`, 495k triangles) ships with the repository.

Run it: [QUICKSTART.md](QUICKSTART.md), "Live text to motion".

## Tier 4: cloud training

docs/BREV.md: instance choice, setup script (`tools/brev/setup_instance.sh`), data sync, the per-GPU run queues and the
budget watchdog that deletes the instance at a deadline.
