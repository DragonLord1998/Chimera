#!/bin/bash
# Chimera — RunPod startup script
# Usage: bash start.sh
# Or set as RunPod's Docker CMD / start command

set -e

cd "$(dirname "$0")"

echo "[Chimera] Patching transformers..."
python3 fix_transformers.py

echo "[Chimera] Installing/upgrading dependencies..."
pip install --break-system-packages flask pillow google-genai sentencepiece || true
pip install --break-system-packages -U "transformers>=5.0" || true
echo "[Chimera] Installing diffusers from GitHub main (required for Klein 9B KV)..."
pip install --break-system-packages -U git+https://github.com/huggingface/diffusers.git || true

# Clone AI Toolkit if not present
if [ ! -d "ai-toolkit" ]; then
    echo "[Chimera] Cloning AI Toolkit..."
    git clone https://github.com/ostris/ai-toolkit.git
    pip install -q --break-system-packages -r ai-toolkit/requirements.txt 2>/dev/null || true
fi

# Clone SeedVR2 CLI if not present (image upscaling)
if [ ! -d "SeedVR2-CLI" ]; then
    echo "[Chimera] Cloning SeedVR2 CLI..."
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git SeedVR2-CLI
    pip install -q --break-system-packages omegaconf rotary-embedding-torch mediapy einops 2>/dev/null || true
fi

# Re-upgrade in case ai-toolkit downgraded
# Install diffusers from GitHub main branch — required for Flux2KleinKVPipeline
pip install --break-system-packages -U "transformers>=5.0" "huggingface-hub>=1.0" || true
echo "[Chimera] Re-installing diffusers from GitHub main (post ai-toolkit)..."
pip install --break-system-packages -U git+https://github.com/huggingface/diffusers.git || true

# Re-install Flask and other deps that may have been clobbered by upgrades
pip install --break-system-packages flask pillow google-genai sentencepiece || true

# Verify critical import
echo "[Chimera] Verifying Flux2KleinKVPipeline import..."
python3 -c "from diffusers import Flux2KleinKVPipeline; print('[Chimera] Flux2KleinKVPipeline — OK')" || echo "[Chimera] WARNING: Flux2KleinKVPipeline not available — Klein 9B KV will not work"

echo "[Chimera] Starting server on port ${PORT:-7860}..."
exec python3 server.py
