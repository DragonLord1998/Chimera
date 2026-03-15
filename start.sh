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
pip install --ignore-installed --break-system-packages flask pillow google-genai sentencepiece oyaml || true

# Fix numpy/scipy binary incompatibility (scipy compiled against older numpy)
pip install --break-system-packages --no-cache-dir --force-reinstall scipy || true

# Patch AI Toolkit for transformers 5.x: wrap all removed imports in try/except
if [ -f "ai-toolkit/toolkit/custom_adapter.py" ]; then
    # Reset to clean git state first to undo any prior broken patches
    (cd ai-toolkit && git checkout -- toolkit/custom_adapter.py) 2>/dev/null

    # Wrap every bare "from transformers import ..." line in try/except
    python3 -c "
import re

p = 'ai-toolkit/toolkit/custom_adapter.py'
with open(p) as f:
    lines = f.readlines()

patched = 0
i = 0
while i < len(lines):
    stripped = lines[i].strip()
    already_wrapped = (i > 0 and lines[i-1].strip() == 'try:')
    if stripped.startswith('from transformers import') and not already_wrapped:
        # Skip multiline imports (backslash continuation or parenthesized)
        if stripped.endswith('\\\\') or ('(' in stripped and ')' not in stripped):
            i += 1
            continue
        m = re.match(r'from transformers import (.+)', stripped)
        if m:
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            names = [n.strip() for n in m.group(1).split(',')]
            block = [indent + 'try:\n', indent + '    ' + stripped + '\n', indent + 'except ImportError:\n']
            for name in names:
                block.append(indent + '    ' + name + ' = None\n')
            lines[i:i+1] = block
            patched += 1
            i += len(block)
            continue
    i += 1

if patched:
    with open(p, 'w') as f:
        f.writelines(lines)
    print(f'[Chimera] Patched AI Toolkit: wrapped {patched} transformers import(s) in try/except')
else:
    print('[Chimera] AI Toolkit: all transformers imports already wrapped')
" || true

    # Verify syntax
    if ! python3 -c "import py_compile; py_compile.compile('ai-toolkit/toolkit/custom_adapter.py', doraise=True)" 2>/dev/null; then
        echo "[Chimera] WARNING: custom_adapter.py still has syntax errors after patching"
    fi
fi

# Verify critical import
echo "[Chimera] Verifying Flux2KleinKVPipeline import..."
python3 -c "from diffusers import Flux2KleinKVPipeline; print('[Chimera] Flux2KleinKVPipeline — OK')" || echo "[Chimera] WARNING: Flux2KleinKVPipeline not available — Klein 9B KV will not work"

echo "[Chimera] Starting server on port ${PORT:-7860}..."
exec python3 server.py
