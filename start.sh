#!/bin/bash
# Chimera — RunPod startup script
# Usage: bash start.sh
# Or set as RunPod's Docker CMD / start command

set -e

cd "$(dirname "$0")"

echo "[Chimera] Patching transformers..."
python3 fix_transformers.py

echo "[Chimera] Installing/upgrading dependencies..."
pip install --ignore-installed --break-system-packages flask pillow google-genai sentencepiece || true
pip install --ignore-installed --break-system-packages -U "transformers>=5.0" || true
echo "[Chimera] Installing diffusers from GitHub main (required for Klein 9B KV)..."
pip install --ignore-installed --break-system-packages -U git+https://github.com/huggingface/diffusers.git || true

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
pip install --ignore-installed --break-system-packages flask pillow google-genai sentencepiece || true

# Fix numpy/scipy binary incompatibility (scipy compiled against older numpy)
pip install --break-system-packages --no-cache-dir --force-reinstall scipy || true

# Patch AI Toolkit for transformers 5.x: ViTHybrid classes removed
if [ -f "ai-toolkit/toolkit/custom_adapter.py" ]; then
    python3 -c "
p = 'ai-toolkit/toolkit/custom_adapter.py'
with open(p) as f:
    lines = f.readlines()

target = 'from transformers import ViTHybridImageProcessor, ViTHybridForImageClassification'
correct = [
    'try:\n',
    '    ' + target + '\n',
    'except ImportError:\n',
    '    ViTHybridImageProcessor = None\n',
    '    ViTHybridForImageClassification = None\n',
]

content = ''.join(lines)
# Already correctly patched (and no stray try: above)
correct_block = 'try:\n    ' + target + '\nexcept ImportError:\n    ViTHybridImageProcessor = None'
if correct_block in content and 'try:\ntry:' not in content:
    print('[Chimera] AI Toolkit ViTHybrid: already patched')
elif target not in content:
    print('[Chimera] AI Toolkit ViTHybrid: import line not found')
else:
    # Find the import line (any indentation level)
    idx = next(i for i, l in enumerate(lines) if target in l)

    # Remove any broken try: lines stacked above
    start = idx
    while start > 0 and lines[start - 1].strip() in ('try:', ''):
        start -= 1

    # Remove any broken except/assignment lines below
    end = idx + 1
    while end < len(lines):
        s = lines[end].strip()
        if s.startswith('except') or s.startswith('ViTHybrid') or s == '':
            end += 1
        else:
            break

    lines[start:end] = correct
    with open(p, 'w') as f:
        f.writelines(lines)
    print('[Chimera] Patched AI Toolkit: ViTHybrid import')
" || true
fi

# Verify critical import
echo "[Chimera] Verifying Flux2KleinKVPipeline import..."
python3 -c "from diffusers import Flux2KleinKVPipeline; print('[Chimera] Flux2KleinKVPipeline — OK')" || echo "[Chimera] WARNING: Flux2KleinKVPipeline not available — Klein 9B KV will not work"

echo "[Chimera] Starting server on port ${PORT:-7860}..."
exec python3 server.py
