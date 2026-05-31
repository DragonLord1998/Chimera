# Project Chimera

Project Chimera is a Colab Pro+ web app for building high-trust identity LoRA datasets and then training identity LoRAs for multiple image/video model families.

This branch is the Colab proxy-only implementation. The existing `main` branch in `DragonLord1998/Chimera` may contain older RunPod work; this branch does not depend on that implementation.

## Core Idea

The workflow is intentionally staged:

```text
1-3 real reference images
-> strict seed curation
-> first Z-Image Turbo identity LoRA
-> generate 5000 synthetic candidates
-> strict identity QC against the original references
-> curated character dataset
-> train model-specific LoRAs
```

The first Z-Image Turbo LoRA is not treated as the final product. It is a fast identity bootstrap model used to create a large synthetic candidate pool. The final source of truth remains the original user-provided identity references plus strict QC.

## Why This Plan

Training a final LoRA directly from 1-3 images is fragile. It can overfit pose, lighting, expression, camera angle, or clothing. Project Chimera instead uses a two-stage approach:

1. Train a small identity seed LoRA quickly.
2. Use that LoRA to generate many controlled variations.
3. Filter aggressively.
4. Train better downstream LoRAs from the filtered dataset.

The key rule is that synthetic images are never trusted just because the seed LoRA generated them. Every accepted image must pass identity checks against the original references.

## Target Pipeline

### Phase 1: Identity Seed Dataset

Input:

- 1-3 user-uploaded reference images.
- Consent/rights confirmation.
- A unique trigger token, for example `zphchar`.

Processing:

- Save references under the case folder.
- Detect faces.
- Reject unusable references.
- Normalize/caption the tiny seed set.
- Train the first identity LoRA on Z-Image Turbo.

Goal:

- A fast identity LoRA that can generate many plausible views of the same person.
- This LoRA is a generator for expansion, not the final model.

### Phase 2: Z-Image Turbo Identity LoRA

The first LoRA target is Z-Image Turbo because it is fast to iterate and practical on Colab Pro+ hardware.

The intended training recipe:

- Base model: `Tongyi-MAI/Z-Image-Turbo`
- Training helper: `ostris/zimage_turbo_training_adapter`
- Trainer: Ostris `ai-toolkit`
- Resolution target: 1024-style square crops where possible.
- Samples every 200-300 steps.
- Checkpoints every 250-500 steps.
- Fixed sample prompts and seeds for progress comparison.

The adapter is important because directly training a short-step distilled model can degrade the distilled behavior. The seed LoRA run should stay short and focused on identity.

### Phase 3: Synthetic Expansion

After the Z-Image Turbo identity LoRA is usable, Chimera generates a large raw candidate pool:

```text
Z-Image Turbo + identity LoRA -> 5000 candidate images
```

The 5000 number is a raw generation target, not a final dataset size. The app should expect most generated images to be discarded.

Generation should be prompt-diverse:

- Front face portrait.
- Side profile.
- Three-quarter view.
- Full body.
- Indoor natural light.
- Outdoor natural light.
- Low light.
- Studio lighting.
- Different expressions.
- Different hair/clothing contexts where identity should remain stable.
- Plain backgrounds for clean face evaluation.
- Scene prompts that test whether identity survives context changes.

Generation should also be controlled:

- Fixed prompt buckets.
- Fixed negative prompts.
- Fixed seeds where useful for checkpoint comparison.
- Balanced number of candidates per prompt category.
- Metadata saved for each image: prompt, seed, model, LoRA checkpoint, LoRA strength, timestamp.

### Phase 4: Strict Identity QC

Strict QC is the most important part of the system.

Accepted images must be scored against the original reference images, not only against other synthetic images. This prevents identity drift from becoming self-reinforcing.

QC gates should include:

- Face detected.
- Exactly one primary face.
- Face area above minimum threshold.
- Face embedding similarity to original references.
- Face embedding similarity to the accepted-set centroid.
- Sharpness threshold.
- Resolution threshold.
- Blur rejection.
- Duplicate/perceptual-near-duplicate removal.
- Pose balance.
- Prompt category balance.
- Manual review lane for borderline samples.

Rejected examples should be preserved with rejection reasons:

- `no_face`
- `multiple_faces`
- `identity_low`
- `tiny_face`
- `blur`
- `duplicate`
- `bad_crop`
- `artifact`
- `manual_reject`

The final curated dataset should usually be much smaller than 5000. The goal is not maximum volume; the goal is a high-trust, diverse identity dataset.

### Phase 5: Curated Character Dataset

The curated dataset becomes the reusable source for downstream training.

Each accepted image should have:

- Image file.
- Caption text.
- Trigger token.
- Prompt category.
- Original generation metadata.
- QC scores.
- Accepted/rejected status.
- Optional manual note.

Captions should describe the visible image without overwriting the identity:

```text
zphchar person, close-up portrait, natural skin texture, soft daylight, neutral background
```

The trigger token should be unique and consistent across all downstream LoRAs.

### Phase 6: Model Adapter Factory

Once the curated identity dataset is ready, Project Chimera trains separate LoRAs for each target model family.

Planned targets:

- Z-Image Turbo LoRA.
- Z-Image Base LoRA.
- FLUX LoRA.
- Wan LoRA.
- LTX LoRA.

Each target model needs its own config. The same curated dataset can be reused, but these should not blindly share training settings.

Per-model settings can differ:

- Resolution.
- Caption style.
- Rank.
- Learning rate.
- Optimizer.
- Batch size.
- Step count.
- Sample prompts.
- Checkpoint interval.
- LoRA save format.
- Inference validation workflow.

## Current App Capabilities

The current React + FastAPI app already provides the foundation:

- Upload 1-3 reference images.
- Store cases under Google Drive.
- Import or smoke-generate candidate images.
- Run face identity QC.
- Highlight QC-selected candidates.
- Copy selected images into the curated training folder.
- Caption curated images.
- Build an `ai-toolkit` training config.
- Start training through `ai-toolkit`.
- Track real training step lines from trainer stdout.
- Show CPU/RAM/GPU/VRAM/power stats.
- Show sample and checkpoint artifact counts.
- Serve through the Colab proxy link only.

Planned next implementation work:

- Add a dedicated Z-Image Turbo seed LoRA config preset.
- Add Z-Image Turbo training adapter handling.
- Add a 5000-image expansion job type.
- Add prompt-bucket generation metadata.
- Add stricter multi-pass QC.
- Add duplicate removal.
- Add manual review/accept/reject UI.
- Add per-model LoRA target presets.
- Add model comparison reports.

## Training Dashboard

The dashboard is designed to show actual runtime state, not guesses.

Training progress comes from `ai-toolkit` stdout/tqdm output and is written to `training_state.json`.

The dashboard shows:

- Current reported step.
- Target steps.
- Last exact trainer step line.
- Running/idle state.
- PID.
- CPU/RAM for the tracked process.
- GPU utilization.
- VRAM usage.
- Power draw and temperature from `nvidia-smi`.
- Sample artifact count.
- Checkpoint artifact count.

The yellow squares under each sample prompt are actual sample artifacts. A square stays grey until Chimera finds a real image file with an explicit step number in its path or filename. Checkpoints are separate `.safetensors`, `.pt`, or `.ckpt` files and are counted separately.

## Access Model

The only supported access point is the Colab proxy URL printed by startup.

The backend binds to localhost:

```text
HOST=127.0.0.1
PORT=7860
```

The launcher refuses non-local host bindings. There is no Gradio share link, no cloudflared tunnel, no ngrok tunnel, and no separate public tunnel.

## Run In Colab

Paste the contents of [colab_launcher_cell.md](./colab_launcher_cell.md) into one Colab notebook cell.

The launcher uses:

```python
REPO_URL = "https://github.com/DragonLord1998/Chimera.git"
BRANCH = "project-chimera-react"
```

Then run the cell. It will:

1. Ask for Google Drive permission through `drive.mount("/content/drive")`.
2. Fresh-clone this branch into `/content/project-chimera`.
3. Install backend dependencies.
4. Build the React frontend.
5. Run `colab_lora_factory.sh`.
6. Print the Colab proxy URL for the app.

The launcher defaults to a fresh clone on every run:

```python
FRESH_CLONE_EVERY_RUN = True
```

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

Do not set `HOST=0.0.0.0`. The launcher will refuse it.

## Storage Layout

Persistent storage root:

```text
/content/drive/MyDrive/GenAI/Project Chimera
```

Case layout:

```text
cases/
  <case_name>/
    refs/                 # uploaded original reference images
    candidates/           # generated or imported synthetic candidates
    curated/
      train/              # accepted training images and captions
    rejected/             # rejected images, future use
    logs/
      generate.log
      train.log
    output/               # ai-toolkit output, samples, checkpoints
    qc_scores.csv
    training_state.json
```

The original references should be kept immutable. QC and downstream training should always be traceable back to them.

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
  media.py        # image previews and artifact discovery
  system.py       # CPU/RAM/GPU/VRAM/power stats
  settings.py     # environment-driven app settings
```

Root files:

```text
web/                         # React/Vite frontend
colab_lora_factory.sh        # Colab bootstrap
colab_launcher_cell.md       # notebook cell that mounts Drive and clones this branch
requirements.txt             # backend runtime dependencies
pyproject.toml               # package metadata
```

## Safety And Consent

Only train identity LoRAs for people where the user has the rights and consent to use the images.

Project Chimera should preserve:

- User consent confirmation.
- Original reference images.
- Generation metadata.
- QC scores.
- Curated/rejected decisions.
- Training configs.
- Final LoRA outputs.

This is necessary for debugging and for proving how a dataset was created.

## Success Criteria

The system is working when:

- The Z-Image Turbo seed LoRA can generate recognizable but varied candidates.
- The 5000-image candidate pool contains broad pose/lighting/expression diversity.
- Strict QC reduces the pool to a smaller, high-trust curated set.
- The curated dataset remains closer to original references than to drifted synthetic clusters.
- Downstream LoRAs for different model families preserve identity without copying one pose or one image.
- Dashboard progress and stats are based on real process output and artifacts.

## Important Design Rule

Never let the synthetic generator become the identity authority.

The identity authority is always:

```text
original references + strict QC + manual review
```

The Z-Image Turbo LoRA is a powerful expansion tool, but it is not the ground truth.
