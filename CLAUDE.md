# Chimera — Character LoRA Creator

## Project Overview

Chimera is a standalone Python/Flask web application that trains a character LoRA from a single input image. The full pipeline is automated: Input Image → Gemini Multi-View Generation → Flux 2 DEV Synthetic Dataset → Florence 2 Captioning → LoRA Training.

**Target hardware**: NVIDIA RTX PRO 6000 (96 GB VRAM) on RunPod.

## Architecture

```
server.py (Flask, SSE streaming)
├── stages/
│   ├── model_manager.py    — Auto-downloads models from HuggingFace
│   ├── multiview.py        — Gemini API multi-view generation (5 views)
│   ├── synthesize.py       — Flux 2 DEV dataset synthesis
│   ├── caption.py          — Florence 2 captioning
│   └── train.py            — AI Toolkit LoRA training wrapper
├── utils/
│   ├── prompt_templates.py — Prompt templates for synthesis
│   ├── identity_stripper.py— Caption post-processing
│   └── checkpoint.py       — Checkpoint utilities
├── static/
│   ├── index.html          — Single-page UI
│   ├── app.js              — Frontend logic + SSE client
│   └── style.css           — Dark theme styling
└── fix_transformers.py     — Patches transformers 5.x backend bugs
```

## Pipeline Stages

1. **Model Download** (Stage 0) — ModelManager downloads required models from HuggingFace
2. **Multi-View Generation** (Stage 1) — Gemini `gemini-3-pro-image-preview` generates 5 views: left, front, right, face close-up, back. OR user uploads a views ZIP to skip this.
3. **Dataset Synthesis** (Stage 2) — Flux 2 DEV (`black-forest-labs/FLUX.2-dev`) generates training images using the 5 views as reference images. Supports up to 10 reference images natively.
4. **Captioning** (Stage 2b) — Florence 2 Large auto-captions the dataset
5. **LoRA Training** (Stage 3) — AI Toolkit by Ostris trains the LoRA. Supports two base models:
   - **Z-Image De-Turbo** (`ostris/Z-Image-De-Turbo`) — default, arch=zimage
   - **FLUX.1-Krea-dev** (`black-forest-labs/FLUX.1-Krea-dev`) — alternative, is_flux=true

## Key Technical Details

### SSE Event System
- Server broadcasts events to all subscriber queues (not a single queue)
- Events stored in `job["history"]` for replay on reconnect
- `/api/jobs/active` endpoint returns the running job for auto-reconnect
- Frontend auto-reconnects on page load, replaying all past events

### Models & VRAM (96 GB RTX PRO 6000)
- Flux 2 DEV: ~32B params, loaded via `Flux2Pipeline.from_pretrained()` with `enable_model_cpu_offload()`
- Z-Image De-Turbo: ~12 GB transformer + ~8 GB text encoder. `quantize: false` (enough VRAM), `quantize_te: true` with qfloat8
- FLUX.1-Krea-dev: ~24 GB. `quantize: false`
- Florence 2 Large: ~2 GB
- Each stage loads/unloads its model to share VRAM

### Training Defaults
- Rank: 16, Alpha: 16
- Optimizer: adamw8bit
- Learning rate: 1e-4
- Noise scheduler: flowmatch
- Resolution: 1024
- Caption dropout: 0.05
- Inference steps (Flux 2 synthesis): 50
- Guidance scale (Flux 2 synthesis): 5.0

### RunPod-Specific Fixes
- `fix_transformers.py` — patches transformers 5.x backends (tf, tensorflow_text, keras_nlp) crash
- `os.environ["USE_TF"] = "0"` set at server startup
- Absolute imports in stages/ (not relative) — Flask context requires this
- pip install order matters: ai-toolkit first, THEN upgrade transformers/diffusers

## API Endpoints

```
GET  /                          — Serves index.html
POST /api/start                 — Start pipeline (multipart form)
GET  /api/stream/<job_id>       — SSE stream (replays history + live events)
GET  /api/jobs/active           — Returns current running job
GET  /api/images/<job_id>/...   — Serves generated images
GET  /api/download/<job_id>     — Downloads final .safetensors
GET  /api/download-views/<job_id> — Downloads views as ZIP
```

## SSE Event Types

```
stage       — Pipeline stage status update
view        — Multi-view image generated (position + url)
synthetic   — Synthetic training image generated (index + url)
progress    — Training step progress (step + total)
checkpoint  — Checkpoint sample images (step + image urls)
complete    — Pipeline done (lora_path + download_url)
error       — Pipeline error (message)
heartbeat   — Keep-alive (empty)
```

## Form Parameters (POST /api/start)

```
image           — Character image file (optional if views_zip provided)
views_zip       — Pre-generated views ZIP (optional, skips Gemini)
trigger_word    — Unique token for character (default: "chrx")
gemini_key      — Gemini API key (required unless views_zip)
hf_token        — HuggingFace token (required for gated models)
base_model      — "zimage" or "flux_krea"
num_images      — Training images to synthesize (10-50, default 25)
lora_rank       — LoRA rank (4-64, default 16)
lora_steps      — Training steps (250-5000, default 1500)
learning_rate   — Optimizer LR (default 1e-4)
inference_steps — Flux 2 denoising steps (10-100, default 50)
sample_prompts  — Custom checkpoint sample prompts (one per line)
```

## Dependencies

- Flask, Pillow, google-genai (Gemini)
- diffusers >= 0.32.0 (Flux2Pipeline)
- transformers >= 5.0
- ai-toolkit (git clone, not pip) — `toolkit.job.run_job(config)`
- JSZip (CDN, client-side ZIP handling)

## Development Notes

- AI Toolkit is NOT pip-installable; it's git-cloned into `./ai-toolkit/` and imported via sys.path
- The watcher thread in server.py polls for .safetensors checkpoint files to estimate training progress (AI Toolkit has no step-level callback)
- Views ZIP upload extracts client-side with JSZip for instant preview, then uploads to server
- All jobs stored in-memory dict `_jobs` — state is lost on server restart
