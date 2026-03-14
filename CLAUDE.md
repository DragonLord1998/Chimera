# Chimera — Character LoRA Creator

## Project Overview

Chimera is a standalone Python/Flask web application that trains a character LoRA from a single input image. The full pipeline is automated: Input Image → Gemini Multi-View Generation → Synthesizer (Flux 2 DEV or Klein 9B KV) → SeedVR2 Upscale → Florence 2 Captioning → LoRA Training.

**Target hardware**: NVIDIA RTX PRO 6000 (96 GB VRAM) on RunPod.

## Architecture

```
server.py (Flask, SSE streaming)
├── stages/
│   ├── model_manager.py    — Auto-downloads models from HuggingFace
│   ├── multiview.py        — Gemini API multi-view generation (5 views)
│   ├── synthesize.py       — Dataset synthesis (Flux 2 DEV or Klein 9B KV)
│   ├── upscale.py          — SeedVR2 7B image upscaling (1024→2048px)
│   ├── caption.py          — Florence 2 captioning
│   └── train.py            — AI Toolkit LoRA training wrapper
├── utils/
│   ├── prompt_templates.py — Prompt templates (50 for Flux 2 DEV, 50 for Klein 9B KV)
│   ├── identity_stripper.py— Caption post-processing
│   └── checkpoint.py       — Checkpoint utilities
├── static/
│   ├── index.html          — Single-page UI
│   ├── app.js              — Frontend logic + SSE client + comparison slider
│   └── style.css           — Dark theme styling
├── fix_transformers.py     — Patches transformers 5.x backend bugs
├── start.sh                — RunPod startup: patches, deps, clone repos, launch
├── ai-toolkit/             — Git-cloned AI Toolkit by Ostris
└── SeedVR2-CLI/            — Git-cloned ComfyUI SeedVR2 standalone CLI
```

## Pipeline Stages

1. **Model Download** (Stage 0) — ModelManager downloads required models from HuggingFace
2. **Multi-View Generation** (Stage 1) — Gemini `gemini-3-pro-image-preview` generates 5 views: left fullbody, front face, right fullbody, face close-up, back fullbody. OR user uploads a views ZIP to skip this.
3. **Dataset Synthesis** (Stage 2) — User-selectable synthesizer:
   - **Flux 2 DEV** (`black-forest-labs/FLUX.2-dev`) — 32B params, all 5 views as reference, 50 denoising steps, 1024px output.
   - **FLUX.2 Klein 9B KV** (`black-forest-labs/FLUX.2-klein-9b-kv`) — 9B params, 4 views (front, face, left, right), 4 steps (step-distilled), KV-cache for 2.5x speedup.
   Real-time latent preview every 2 denoising steps via `callback_on_step_end`.
4. **SeedVR2 Upscale** (Stage 2a) — SeedVR2 7B upscales all training images from 1024px → 2048px. Originals preserved in `_originals/` for before/after comparison slider in UI. Called via subprocess (ComfyUI CLI) to avoid dependency conflicts.
5. **Captioning** (Stage 2b) — Florence 2 Large auto-captions the 2048px dataset. Strips identity traits, prepends trigger word. Skipped if `.txt` caption files already exist.
6. **LoRA Training** (Stage 3) — AI Toolkit by Ostris trains the LoRA at 2048px. Supports two base models:
   - **Z-Image De-Turbo** (`ostris/Z-Image-De-Turbo`) — default, arch=zimage, Qwen3 text encoder
   - **FLUX.1-Krea-dev** (`black-forest-labs/FLUX.1-Krea-dev`) — alternative, is_flux=true

### Shortcut Paths
- **Upload views ZIP** → skips Stage 1, starts at Stage 2 (synthesis)
- **Upload dataset ZIP** → skips Stages 1 + 2, auto-captions if no `.txt` files, straight to training

## Key Technical Details

### SSE Event System
- Server broadcasts events to all subscriber queues (not a single queue)
- Events stored in `job["history"]` for replay on reconnect
- Supports `ephemeral=True` events (diffusion previews) that skip history storage
- `/api/jobs/active` endpoint returns the running job for auto-reconnect
- Frontend auto-reconnects on page load, replaying all past events

### Models & VRAM (96 GB RTX PRO 6000)
- Flux 2 DEV: ~32B params, loaded via `Flux2Pipeline.from_pretrained()` with `enable_model_cpu_offload()`
- FLUX.2 Klein 9B KV: ~29 GB, loaded via `Flux2KleinKVPipeline.from_pretrained()` with `enable_model_cpu_offload()`. Downloaded on demand.
- SeedVR2 7B: ~24 GB for 1024→2048 upscale, one-step diffusion transformer
- Z-Image De-Turbo: ~12 GB transformer + ~8 GB text encoder. `quantize: false` (enough VRAM), `quantize_te: true` with qfloat8
- FLUX.1-Krea-dev: ~24 GB. `quantize: false`
- Florence 2 Large: ~2 GB
- Each stage loads/unloads its model to share VRAM — only one in VRAM at a time

### Z-Image Model Structure
The `ostris/Z-Image-De-Turbo` repo only ships the transformer delta. The full model at `/workspace/models/z_image/` requires components from multiple repos:
- `transformer/` — from `ostris/Z-Image-De-Turbo` (the de-turbo fine-tune)
- `text_encoder/` — from `Tongyi-MAI/Z-Image-Turbo` (Qwen3-based, `Qwen3ForCausalLM`)
- `tokenizer/` — from `Tongyi-MAI/Z-Image-Turbo` (Qwen2Tokenizer, vocab_size 151936)
- `vae/` — from `Tongyi-MAI/Z-Image-Turbo` (AutoencoderKL)

All four subfolders are required by AI Toolkit's `z_image.py` (`AutoTokenizer.from_pretrained(subfolder="tokenizer")`, etc.).

### Training Defaults
- Rank: 32, Alpha: 32
- Optimizer: adamw8bit
- Learning rate: 1e-4
- Noise scheduler: flowmatch
- Resolution: 2048
- Caption dropout: 0.05
- Inference steps (Flux 2 DEV synthesis): 50
- Inference steps (Klein 9B KV synthesis): 4 (step-distilled)
- Guidance scale (synthesis): 5.0
- Checkpoint/sample interval: every 250 steps
- Sample steps: 50, Sample CFG: 4.0 (Z-Image), 4.5 (Flux Krea)

### Prompt Templates
- Two template sets: Flux 2 DEV (5 refs) and Klein 9B KV (4 refs)
- 50 templates each: 25 ORIGINAL_OUTFIT + 25 VARIED_OUTFIT, interleaved
- Flux 2 DEV templates reference images 1-5 (left, front, right, face, back)
- Klein templates reference images 1-4 (front, face close-up, left, right)
- `get_prompt_templates(num_images)` for Flux 2 DEV, `get_prompt_templates_klein(num_images)` for Klein
- Klein reference selection: `select_klein_references()` picks front, face, left, right (indices 1, 3, 0, 2 from 5 Gemini views)

### Real-Time Progress
- **Diffusion preview**: `callback_on_step_end` in Flux 2 pipeline, approximate RGB from first 3 latent channels (no VAE decode), sent as base64 JPEG via ephemeral SSE
- **Training progress**: Captures AI Toolkit's tqdm output from stderr (`113/1500`) via `_StderrCapture` wrapper. Emits progress events every 2 seconds.
- **Sample detection**: Polls `samples/` directory for `.jpg`/`.png` files, buffers by parsed step number from filename (`{timestamp}__{step:09d}_{idx}.jpg`), emits checkpoint row when all prompts' images arrive (or 60s timeout)

### Before/After Comparison Slider
- SeedVR2 upscale preserves originals in `dataset/_originals/`
- Frontend shows "2048px" badge on upscaled synthetic cells
- Click any upscaled cell → fullscreen overlay with drag slider (Original 1024px vs SeedVR2 2048px)
- Touch/swipe support for mobile

### RunPod-Specific Compatibility Patches (server.py startup)
Applied in order before any model loading:
1. Mock `torchaudio` with proper `__spec__` (ModuleSpec) and submodules — AI Toolkit imports but doesn't need it
2. Mock `huggingface_hub.Repository` (removed in >= 1.0) with stub class
3. Mock `huggingface_hub.HfFolder` with `get_token()` delegating to new API, patched into `huggingface_hub.utils`
4. Patch `find_pruneable_heads_and_indices` back into `transformers.pytorch_utils` (moved in 5.x)
5. Patch `load_backbone` back into `transformers.utils.backbone_utils` (moved in 5.x)
6. Patch `TokenizersBackend.additional_special_tokens` property (Florence 2 custom processor needs it)
7. Monkey-patch `AutoTokenizer.from_pretrained` to auto-retry with `use_fast=False` (transformers 5.x fast tokenizer failures)

### SeedVR2 Integration
- Uses ComfyUI standalone CLI (`inference_cli.py`) via subprocess to avoid dependency conflicts (SeedVR2 requires transformers 4.x / diffusers 0.29, we run 5.x / 0.32+)
- CLI auto-downloads and converts model weights from HuggingFace (`ByteDance-Seed/SeedVR2-7B`)
- CLI args: `--resolution 2048 --output <path> --dit_model seedvr2_ema_7b_fp16.safetensors --batch_size 1`
- Cloned to `./SeedVR2-CLI/` from `github.com/numz/ComfyUI-SeedVR2_VideoUpscaler`

## API Endpoints

```
GET  /                          — Serves index.html
POST /api/start                 — Start pipeline (multipart form)
GET  /api/stream/<job_id>       — SSE stream (replays history + live events)
GET  /api/jobs/active           — Returns current running job
GET  /api/images/<job_id>/...   — Serves generated images (inc. _originals/)
GET  /api/download/<job_id>     — Downloads final .safetensors
GET  /api/download-views/<job_id> — Downloads views as ZIP
```

## SSE Event Types

```
stage             — Pipeline stage status update
view              — Multi-view image generated (position + url)
synthetic         — Synthetic training image generated (index + url)
diffusion_preview — Real-time latent preview during synthesis (ephemeral, base64 JPEG)
upscaled          — Image upscaled by SeedVR2 (index + original_url + upscaled_url)
progress          — Training step progress (step + total)
checkpoint        — Checkpoint sample images (step + image urls)
complete          — Pipeline done (lora_path + download_url)
error             — Pipeline error (message)
heartbeat         — Keep-alive (empty)
```

## Form Parameters (POST /api/start)

```
image           — Character image file (optional if views_zip or dataset_zip provided)
views_zip       — Pre-generated views ZIP (optional, skips Gemini)
dataset_zip     — Pre-made training dataset ZIP (optional, skips Gemini + Flux 2)
trigger_word    — Unique token for character (default: "chrx")
gemini_key      — Gemini API key (required unless views_zip or dataset_zip)
hf_token        — HuggingFace token (required for gated models)
synthesizer     — "flux2_dev" or "klein_kv" (default: "flux2_dev")
base_model      — "zimage" or "flux_krea"
num_images      — Training images to synthesize (10-50, default 25)
lora_rank       — LoRA rank (4-64, default 32)
lora_steps      — Training steps (250-5000, default 1500)
learning_rate   — Optimizer LR (default 1e-4)
inference_steps — Flux 2 denoising steps (10-100, default 50)
sample_prompts  — Custom checkpoint sample prompts (one per line)
```

## Dependencies

- Flask, Pillow, google-genai (Gemini), sentencepiece
- diffusers from GitHub main branch (Flux2Pipeline, Flux2KleinKVPipeline)
- transformers >= 5.0
- ai-toolkit (git clone, not pip) — `toolkit.job.run_job(config)`
- SeedVR2-CLI (git clone) — `inference_cli.py` subprocess
- omegaconf, rotary-embedding-torch, mediapy, einops (SeedVR2 deps)
- JSZip (CDN, client-side ZIP handling)

## Development Notes

- AI Toolkit is NOT pip-installable; it's git-cloned into `./ai-toolkit/` and imported via sys.path
- SeedVR2 CLI is NOT imported directly; it's called as a subprocess to avoid transformers/diffusers version conflicts
- Training progress is captured by wrapping `sys.stderr` to parse tqdm output (`_StderrCapture` class). Restored in a `finally` block after training.
- AI Toolkit outputs sample images as `.jpg` (not `.png`) — watcher must check for both extensions
- Sample filenames follow pattern `{timestamp}__{step:09d}_{prompt_idx}.jpg`
- Views ZIP upload extracts client-side with JSZip for instant preview, then uploads to server
- All jobs stored in-memory dict `_jobs` — state is lost on server restart
- pip install order matters: ai-toolkit first, THEN upgrade transformers/diffusers (start.sh handles this)
- `os.environ["USE_TF"] = "0"` set at server startup to prevent TF import crashes
- Absolute imports in stages/ (not relative) — Flask context requires this

## RunPod Deployment

```bash
# SSH access
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519

# Start server
cd /workspace/Chimera && bash start.sh

# Models directory
/workspace/models/
├── florence2/        — Florence 2 Large (~2 GB)
├── z_image/          — Z-Image De-Turbo (transformer + text_encoder + tokenizer + vae)
├── flux_krea/        — FLUX.1-Krea-dev (optional)
└── flux2_klein_kv/   — FLUX.2 Klein 9B KV (optional, downloaded on demand)

# Flux 2 DEV is downloaded via HuggingFace Hub cache (gated, requires HF token)
# Klein 9B KV is downloaded on demand when user selects it as synthesizer
# SeedVR2 7B weights auto-downloaded by CLI on first run
```
