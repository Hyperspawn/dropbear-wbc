#!/usr/bin/env bash
# One-time setup of a Brev Linux box for dropbear-wbc training (docs/BREV.md section 3, route A).
# Installs Isaac Sim 5.0.0 + Isaac Lab 2.2.0 in a Python 3.11 venv (~/isaac), unpacks the bundle,
# verifies the USD and bundle hashes. Idempotent: re-running skips finished steps.
#
# The user approved accepting the NVIDIA Isaac Sim EULA on this instance (2026-09-25).
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
LOG=~/setup_instance.log
exec > >(tee -a "$LOG") 2>&1
echo "=== setup start $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
. /etc/os-release && echo "OS: $PRETTY_NAME"

# --- system packages -------------------------------------------------------------------------
if ! command -v python3.11 >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa || true
  sudo apt-get update -y
  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi
sudo apt-get install -y git tmux rsync build-essential cmake libglu1-mesa libxt6 >/dev/null || true

# --- venv + Isaac Sim + torch ----------------------------------------------------------------
if [ ! -x ~/isaac/bin/python ]; then python3.11 -m venv ~/isaac; fi
source ~/isaac/bin/activate
python -m pip install --upgrade pip wheel >/dev/null
if ! python -c "import isaacsim" 2>/dev/null; then
  pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
fi

# --- Isaac Lab 2.2.0 (core only; our vendored rsl-rl 2.3.3 is the only learning framework) ---
if [ ! -d ~/IsaacLab ]; then git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab; fi
cd ~/IsaacLab && git fetch --tags -q && git checkout -q v2.2.0
if ! python -c "import isaaclab" 2>/dev/null; then
  # known fixes (2026-09-25, reference_brev_workflow): isaaclab.sh fails building flatdict with new setuptools
  pip install "setuptools<70"
  pip install --no-build-isolation flatdict==4.0.1
  pip install -e source/isaaclab
fi
pip install click==8.1.7 psutil==5.9.8 GitPython onnx onnxscript onnxruntime >/dev/null
python -c "import isaacsim, isaaclab, torch; print('isaac ok, torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"

# --- bundle + USD ------------------------------------------------------------------------------
USD_SHA=45586414b065cd982d487cbd868fe982108b3b8ccec64d3dfcf629652ed8db0f
got=$(sha256sum ~/assets/dropbear.usd | cut -d' ' -f1)
[ "$got" = "$USD_SHA" ] || { echo "USD SHA MISMATCH: $got"; exit 3; }
echo "USD sha ok"
for g in 0 1 2 3; do
  d=~/dbw$g
  if [ ! -f $d/BUNDLE.json ]; then mkdir -p $d && tar xzf ~/dropbear-wbc-brev.tgz -C $d; fi
done
cd ~/dbw0 && python - <<'EOF'
import hashlib, json
b = json.load(open("BUNDLE.json"))
bad = [f for f, i in b["files"].items() if hashlib.sha256(open(f, "rb").read()).hexdigest() != i["sha256"]]
print("bundle ok" if not bad else f"CORRUPT: {bad[:10]}")
raise SystemExit(1 if bad else 0)
EOF
echo "=== setup done $(date -Is)"
