# Chimera

Train a character LoRA from a single image. Upload one character reference, and the fully automated pipeline generates multi-view sheets, synthesizes a diverse training dataset, upscales and captions every image, then trains your LoRA — all from a browser UI.

**Pipeline:** Input Image → Gemini Multi-View → Synthesizer (Flux 2 DEV or Klein 9B KV) → SeedVR2 Upscale → Florence 2 Captioning → LoRA Training

## Features

- **Two synthesizers** — Flux 2 DEV (32B, 50 steps, highest quality) or FLUX.2 Klein 9B KV (9B, 4 steps, 2.5x faster)
- **Two base models** — Z-Image De-Turbo (Alibaba/Ostris) or FLUX.1-Krea-dev (BFL aesthetic fine-tune)
- **SeedVR2 7B upscaling** — 1024px → 2048px super-resolution with before/after comparison slider
- **Florence 2 auto-captioning** — Identity trait stripping + trigger word prepending
- **Real-time progress** — Live latent previews during synthesis, step-by-step training progress, checkpoint sample galleries
- **Shortcut paths** — Upload a views ZIP to skip Gemini, or a full dataset ZIP to skip synthesis
- **Smart skipping** — Already-upscaled images (>= 2048px) skip SeedVR2; existing captions skip Florence 2
- **Auto-reconnect** — Refresh the page mid-job and pick up where you left off via SSE replay

## Requirements

- **GPU**: NVIDIA A40 (46GB) minimum, RTX PRO 6000 (96GB) recommended
- **Python**: 3.10+
- **Gemini API Key**: For multi-view generation
- **HuggingFace Token**: Required for gated models (Flux 2 DEV, Klein 9B KV, Krea)

## Quick Start (RunPod)

```bash
git clone https://github.com/DragonLord1998/Chimera.git /workspace/Chimera
cd /workspace/Chimera
bash start.sh
```

`start.sh` handles everything: installs dependencies, clones AI Toolkit and SeedVR2 CLI, patches transformers 5.x, and starts the server on port **7860**.

Open port **7860** in your RunPod pod template. Access the UI at `http://<your-pod-ip>:7860`.

All models (~50GB+) are auto-downloaded on first run. Incomplete downloads are detected and re-downloaded automatically.

## Pipeline Stages

| Stage | Model | What it does |
|-------|-------|-------------|
| 0 | — | Auto-downloads all models from HuggingFace (with shard verification) |
| 1 | Gemini (`gemini-3-pro-image-preview`) | Generates 5 views: left, front, right, face close-up, back |
| 2 | Flux 2 DEV or Klein 9B KV | Synthesizes ~25 diverse training images using views as reference |
| 2a | SeedVR2 7B | Upscales all training images from 1024px → 2048px |
| 2b | Florence 2 Large | Auto-captions images with identity trait stripping |
| 3 | Z-Image De-Turbo / FLUX.1-Krea-dev | Trains a LoRA with flowmatch scheduler (AI Toolkit) |

## Synthesizer Comparison

| | Flux 2 DEV | Klein 9B KV |
|---|---|---|
| **Parameters** | 32B | 9B |
| **Steps** | 50 (configurable) | 4 (fixed, step-distilled) |
| **Reference images** | 5 (all views) | 4 (front, face, left, right) |
| **Speed** | Baseline | ~2.5x faster |
| **Quality** | Highest | Good (fast iteration) |
| **License** | Gated (HF token required) | Gated, non-commercial |

## Configuration

All settings are configurable in the web UI:

| Setting | Default | Description |
|---------|---------|-------------|
| Trigger Word | `chrx` | Unique token for this character |
| Synthesizer | Flux 2 DEV | Flux 2 DEV or Klein 9B KV |
| Training Images | 25 | Number of synthetic images (10–50) |
| Base Model | Z-Image De-Turbo | Z-Image or FLUX.1-Krea-dev |
| LoRA Rank | 16 | Higher = more detail, larger file (4–64) |
| Training Steps | 1500 | 1000–2000 typical |
| Learning Rate | 1e-4 | Optimizer learning rate |
| Inference Steps | 50 | Synthesis denoising steps (Klein: fixed at 4) |
| Sample Prompts | Auto | Custom checkpoint preview prompts (one per line) |

## Architecture

```
server.py                    Flask + SSE streaming
├── stages/
│   ├── model_manager.py     Auto-downloads models (with shard verification)
│   ├── multiview.py         Gemini multi-view generation (5 views)
│   ├── synthesize.py        Dataset synthesis (Flux 2 DEV / Klein 9B KV)
│   ├── upscale.py           SeedVR2 7B super-resolution
│   ├── caption.py           Florence 2 captioning (community checkpoint)
│   └── train.py             AI Toolkit LoRA training wrapper
├── utils/
│   ├── prompt_templates.py  50 templates per synthesizer
│   ├── identity_stripper.py Caption post-processing
│   └── checkpoint.py        Checkpoint utilities
├── static/
│   ├── index.html           Single-page UI
│   ├── app.js               Frontend + SSE client + comparison slider
│   └── style.css            Dark theme
├── fix_transformers.py      Patches transformers 5.x backend bugs
├── start.sh                 RunPod startup script
├── ai-toolkit/              Git-cloned AI Toolkit (Ostris)
└── SeedVR2-CLI/             Git-cloned SeedVR2 standalone CLI
```

## Models & VRAM

| Model | VRAM | Purpose |
|-------|------|---------|
| Flux 2 DEV | ~32 GB | Dataset synthesis (CPU offload enabled) |
| Klein 9B KV | ~29 GB | Fast dataset synthesis (CPU offload enabled) |
| SeedVR2 7B | ~24 GB | Image upscaling |
| Z-Image De-Turbo | ~20 GB | LoRA training (transformer + text encoder) |
| FLUX.1-Krea-dev | ~24 GB | Alternative LoRA training base |
| Florence 2 Large | ~2 GB | Image captioning |

Models are loaded and unloaded sequentially — only one is in VRAM at a time.

## License

This project integrates multiple models with different licenses. Check each model's license on HuggingFace before commercial use.
