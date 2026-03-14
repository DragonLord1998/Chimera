#!/bin/bash
# Chimera — RunPod startup script
# Usage: bash start.sh
# Or set as RunPod's Docker CMD / start command

set -e

cd "$(dirname "$0")"

echo "[Chimera] Patching transformers..."
python3 fix_transformers.py

echo "[Chimera] Installing/upgrading dependencies..."
pip install -q --break-system-packages flask pillow google-genai 2>/dev/null || true
pip install -q --break-system-packages -U "transformers>=5.0" "diffusers>=0.32.0" 2>/dev/null || true

# Clone AI Toolkit if not present
if [ ! -d "ai-toolkit" ]; then
    echo "[Chimera] Cloning AI Toolkit..."
    git clone https://github.com/ostris/ai-toolkit.git
    pip install -q --break-system-packages -r ai-toolkit/requirements.txt 2>/dev/null || true
fi

# Re-upgrade in case ai-toolkit downgraded
pip install -q --break-system-packages -U "transformers>=5.0" "diffusers>=0.32.0" "huggingface-hub>=1.0" 2>/dev/null || true

echo "[Chimera] Starting server on port ${PORT:-7860}..."
exec python3 server.py
