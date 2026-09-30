#!/bin/bash
# Clone NVIDIA Kimodo at the pinned commit into third_party/kimodo and apply the dropbear-wbc overlay
# (third_party/kimodo_overlay, see its NOTICE.md). Run from anywhere; needs git. Then install it into the Kimodo venv:
#   Linux / WSL:  python3.10 -m venv .venv-kimodo-wsl && .venv-kimodo-wsl/bin/pip install -e third_party/kimodo
#   (Kimodo builds a small C++ extension: needs cmake and g++; see docs/INSTALL.md, "Text to motion")
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DEST=$REPO/third_party/kimodo
COMMIT=54257dd8ff18aa764d620919427ce4dd29c111d0
if [ ! -d "$DEST/.git" ]; then
  if [ -e "$DEST" ]; then echo "$DEST exists but is not a git clone; move it away first" >&2; exit 1; fi
  git clone https://github.com/nv-tlabs/kimodo.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$COMMIT" 2>/dev/null || git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$COMMIT"
cp -rv "$REPO/third_party/kimodo_overlay/kimodo/." "$DEST/kimodo/"
echo "Kimodo $COMMIT + dropbear-wbc overlay ready in $DEST"
