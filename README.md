# Project Chimera

Repo-based Colab Pro+ launcher for an automatic character LoRA workflow.

The expected Colab flow is:

```text
Mount Drive -> clone this repo fresh -> build the React app -> launch FastAPI on localhost -> use the Colab proxy URL
```

## Run In Colab

After pushing this folder to GitHub, paste the contents of [colab_launcher_cell.md](./colab_launcher_cell.md) into one Colab notebook cell.

Change this line:

```python
REPO_URL = "https://github.com/DragonLord1998/Chimera.git"
BRANCH = "project-chimera-react"
```

Then run the cell. It will:

1. Ask for Google Drive permission through `drive.mount("/content/drive")`.
2. Fresh-clone the repo into `/content/project-chimera`.
3. Install backend dependencies.
4. Build the React frontend.
5. Run `colab_lora_factory.sh`.
6. Print the Colab proxy URL for the app.

The launcher defaults to a fresh clone on every run:

```python
FRESH_CLONE_EVERY_RUN = True
```

The only supported access point is the printed Colab proxy URL. The backend binds to `127.0.0.1` and the launcher refuses non-local host bindings. There is no separate public tunnel.

## Manual Terminal Run

In a Colab terminal:

```bash
git clone --depth 1 --branch project-chimera-react https://github.com/DragonLord1998/Chimera.git /content/project-chimera
bash /content/project-chimera/colab_lora_factory.sh
```

Useful overrides:

```bash
PORT=7861 bash /content/project-chimera/colab_lora_factory.sh
INSTALL_FRONTEND=0 bash /content/project-chimera/colab_lora_factory.sh
INSTALL_AI_TOOLKIT=0 bash /content/project-chimera/colab_lora_factory.sh
FACE_MODEL=buffalo_l bash /content/project-chimera/colab_lora_factory.sh
```

## What The App Does

1. Uploads 1-3 reference images.
2. Imports or generates synthetic candidates.
3. Scores candidates with face-identity QC.
4. Selects the best training images.
5. Writes captions with a trigger token.
6. Creates an ai-toolkit LoRA config.
7. Starts LoRA training through ai-toolkit.

The first screen is the pipeline view:

```text
Upload reference images -> synthetic candidate grid -> QC-selected grid with red highlights
-> training dashboard with steps, sample preview, system stats, and prompt cards
```

The detailed tabs remain available for generation commands, scoring tables, captions, and training configuration.

## Training Dashboard

Training progress is tracked from ai-toolkit's live stdout/tqdm stream and written to `training_state.json` in the case folder. The dashboard shows the last step line reported by the trainer; it does not infer progress from filenames or estimate a rough step. Runtime stats come from the tracked training process via `psutil` and the GPU via `nvidia-smi`.

The yellow squares under each sample prompt are actual sample snapshots, not decorative placeholders. A square is grey until the app finds a real sample image in the ai-toolkit output folder with an explicit step number in its path or filename. A square turns yellow only for a step where a real sample artifact exists. Checkpoints are separate `.safetensors`/`.pt`/`.ckpt` files and are counted separately in the dashboard.

By default, sample prompts run every `250` training steps and checkpoints save every `250` training steps. These are editable in the Pipeline training controls and are written into the ai-toolkit config as `sample_every` and `save_every`.

## Defaults

- Persistent root: `/content/drive/MyDrive/GenAI/Project Chimera`
- Server port: `7860`
- Server host: `127.0.0.1`
- ai-toolkit path: `/content/ai-toolkit`
- Default base model in the UI: `black-forest-labs/FLUX.2-klein-base-9B`

The included "Smoke Test Generate" button is only for testing the UI. For production generation, use the generation-command box to call your real FLUX/ComfyUI/PuLID workflow or import images from your ComfyUI output folder.

## Project Structure

This is a split React + FastAPI app, not a generated single-file script.

```text
automatic_lora_trainer/
  app.py          # application entrypoint
  api.py          # FastAPI routes and static frontend serving
  paths.py        # case folders and image discovery
  state.py        # logs, running processes, training_state.json
  generation.py   # reference upload, imports, smoke generation, command runner
  face_qc.py      # InsightFace/AuraFace identity scoring and dataset selection
  captioning.py   # training captions and caption preview
  training.py     # ai-toolkit config and tracked training process
  dashboard.py    # progress dashboard and prompt sample status
  media.py        # contact sheets and image previews
  system.py       # CPU/RAM/GPU/VRAM/power stats
  settings.py     # environment-driven app settings
```

Root files:

```text
web/                         # React/Vite frontend
colab_lora_factory.sh       # small Colab bootstrap
colab_launcher_cell.md      # notebook cell that mounts Drive and clones this repo
requirements.txt            # runtime dependencies used by the bootstrap
pyproject.toml              # package metadata
```
