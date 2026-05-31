#!/usr/bin/env bash
set -Eeuo pipefail

# Colab bootstrap for the repo-based Project Chimera.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_TOOLKIT_DIR="${AI_TOOLKIT_DIR:-/content/ai-toolkit}"
WORK_ROOT="${WORK_ROOT:-/content/drive/MyDrive/GenAI/Project Chimera}"
FALLBACK_WORK_ROOT="${FALLBACK_WORK_ROOT:-/content/Project Chimera}"
PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
INSTALL_FRONTEND="${INSTALL_FRONTEND:-1}"
INSTALL_FACE_QC="${INSTALL_FACE_QC:-1}"
INSTALL_AI_TOOLKIT="${INSTALL_AI_TOOLKIT:-1}"
INSTALL_TORCH="${INSTALL_TORCH:-0}"
FACE_MODEL="${FACE_MODEL:-auraface}"

echo "[chimera] Preparing Project Chimera..."

if [[ -d /content && ! -d /content/drive/MyDrive ]]; then
  echo "[chimera] Google Drive is not mounted. Trying drive.mount()."
  python3 - <<'PY' || true
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as exc:
    print(f"Drive mount skipped or failed: {exc}")
PY
fi

if [[ ! -d /content/drive/MyDrive ]]; then
  echo "[chimera] Drive not available. Using ${FALLBACK_WORK_ROOT}."
  WORK_ROOT="$FALLBACK_WORK_ROOT"
fi

mkdir -p "$WORK_ROOT"

echo "[chimera] Installing app dependencies..."
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r "$SCRIPT_DIR/requirements.txt"

if [[ "$INSTALL_FRONTEND" == "1" ]]; then
  echo "[chimera] Building React frontend..."
  if ! command -v npm >/dev/null 2>&1; then
    echo "[chimera] npm not found. Installing nodejs/npm with apt."
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nodejs npm
  fi
  if [[ -f "$SCRIPT_DIR/web/package-lock.json" ]]; then
    npm --prefix "$SCRIPT_DIR/web" ci
  else
    npm --prefix "$SCRIPT_DIR/web" install
  fi
  npm --prefix "$SCRIPT_DIR/web" run build
fi

if [[ "$INSTALL_FACE_QC" == "1" ]]; then
  echo "[chimera] Installing optional face QC dependencies..."
  python3 -m pip install -q insightface onnxruntime-gpu || \
    python3 -m pip install -q insightface onnxruntime || true
fi

if [[ "$INSTALL_AI_TOOLKIT" == "1" ]]; then
  if [[ ! -d "$AI_TOOLKIT_DIR/.git" ]]; then
    echo "[chimera] Cloning ai-toolkit into ${AI_TOOLKIT_DIR}..."
    git clone --depth 1 https://github.com/ostris/ai-toolkit.git "$AI_TOOLKIT_DIR"
  else
    echo "[chimera] ai-toolkit already exists at ${AI_TOOLKIT_DIR}."
  fi

  if [[ "$INSTALL_TORCH" == "1" ]]; then
    echo "[chimera] Installing ai-toolkit torch pins. This can take a while."
    python3 -m pip install --no-cache-dir \
      torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
      --index-url https://download.pytorch.org/whl/cu128
  fi

  echo "[chimera] Installing ai-toolkit requirements..."
  python3 -m pip install -q -r "$AI_TOOLKIT_DIR/requirements.txt" || true
fi

echo "[chimera] Starting app on ${HOST}:${PORT}."
echo "[chimera] Work root: ${WORK_ROOT}"
echo "[chimera] Open the Colab proxy URL printed by the Python server."

export AI_TOOLKIT_DIR
export WORK_ROOT
export HOST
export PORT
export FACE_MODEL
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

python3 -m automatic_lora_trainer.app
