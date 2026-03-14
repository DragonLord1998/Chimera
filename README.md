# Chimera

Single-image to character LoRA pipeline. Upload one character image, get a trained Z-Image LoRA.

**Pipeline:** Input Image → Gemini Multi-View → Flux 2 DEV Synthesis → Florence 2 Captioning → Z-Image LoRA Training

## Requirements

- **GPU**: NVIDIA A40 (48GB VRAM) recommended
- **Python**: 3.10+
- **Gemini API Key**: For multi-view generation
- **HuggingFace Token**: Optional (only if gated models require it)

## Setup (RunPod)

```bash
git clone https://github.com/DragonLord1998/Chimera.git /workspace/Chimera
cd /workspace/Chimera
pip install -r requirements.txt
git clone --depth 1 https://github.com/ostris/ai-toolkit.git
pip install -r ai-toolkit/requirements.txt
python server.py
```

Open port **7860** in your RunPod pod template. Access the UI at `http://<your-pod-ip>:7860`.

All models (~46GB) are auto-downloaded on first run.

## Pipeline Stages

| Stage | Model | What it does |
|-------|-------|-------------|
| 0 | — | Auto-downloads all models from HuggingFace |
| 1 | Gemini (`gemini-3-pro-image-preview`) | Generates left/front/right multi-view images |
| 2 | Flux 2 DEV (fp8, local) | Synthesizes ~25 diverse training images using views as reference |
| 3 | Florence 2 Large | Auto-captions images with identity trait stripping |
| 4 | Z-Image De-Turbo (AI Toolkit) | Trains a LoRA with flowmatch scheduler |

## CLI Usage

```bash
python run.py \
  --image character.png \
  --trigger chrx \
  --gemini-key AIza... \
  --steps 1500 \
  --rank 16
```

## Configuration

All settings are configurable in the web UI:

- **Trigger Word** — unique token for the character (e.g. `chrx`)
- **Training Images** — number of synthetic images to generate (10–50)
- **LoRA Rank** — higher = more detail, larger file (4–64)
- **Training Steps** — 1000–2000 typical
- **Learning Rate** — default 1e-4
- **Sample Prompts** — custom checkpoint preview prompts (one per line, use `TRIGGER` as placeholder)

## Crash Recovery

Checkpoint files track progress at each stage. If your RunPod restarts mid-job, the pipeline resumes from the last completed stage automatically.
