# Chimera v0.2 — High-Quality LoRA Training with Gemini Snowball Referencing

## Overview

v0.2 replaces the simple two-pass pipeline with a **three-stage, research-backed approach** that produces dramatically higher quality character LoRAs:

1. **Generate 10 diverse, high-quality character images** using Gemini with a "snowball" referencing strategy (up to 5 reference images per call, each generation feeds into the next)
2. **Train an identity LoRA** on these 10 HQ images using the full SRPO base model (not a LoRA — a 23.8GB BF16 fine-tune of FLUX.1-dev for photorealism)
3. **Enhance the massive synthetic dataset** via img2img using the identity LoRA to lock character identity and add photorealistic detail
4. **Train the final Z-Image LoRA** on the enhanced dataset for maximum quality

This approach solves the chicken-and-egg problem: instead of training on synthetic data from the start, we bootstrap with hand-curated Gemini images that capture the character's essence, then use those to enhance the dataset.

### v0.2 Pipeline vs v0.1
```
v0.1 Pipeline (current):
  User image → Gemini views (5) → Flux 2 synthesizes dataset → SeedVR2 upscale
  → Florence 2 captions → Z-Image LoRA training

v0.2 Pipeline (proposed):
  User image → Gemini generates 10 diverse images (snowball)
  → Florence 2 captions 10 Gemini images
  → Generate 20-30 regularization images from SRPO base
  → Train identity LoRA (SRPO base, rank 32, 10 images + reg images)
  → Flux 2 synthesizes massive dataset (25-300 images, using Gemini images as references)
  → SeedVR2 upscale synthetic dataset
  → Florence 2 captions synthetic dataset
  → Clear latent cache
  → Enhance synthetic dataset (SRPO + identity LoRA, img2img, denoise 0.30)
  → Train final Z-Image LoRA on enhanced dataset
```

---

## 1. Stage 1: Gemini Snowball Image Generation

### 1.1 The Snowball Strategy (NEW)

Instead of generating 5 static multi-view images, we now generate **10 diverse character images** using a "snowball" strategy where each generation feeds into the next, building up a rich understanding of the character.

**Snowball Order** (3 anchors, then 7 diversity expansions):

1. **Face close-up, front, neutral, white background** — anchor (defines face)
2. **Mid-body, front, neutral, white background** — anchor (defines body structure)
3. **Full body, front, standing, white background** — anchor (defines proportions, clothing)
4. **Mid-body, 3/4 left, slight smile, soft lighting** — angle expansion (rotated, expression variation)
5. **Face close-up, right profile, serious, studio lighting** — angle expansion (profile, expression)
6. **Mid-body, 3/4 right, casual pose, warm outdoor lighting** — diversity (angle, environment, pose)
7. **Full body, left side, walking, overcast outdoor lighting** — diversity (full body angle, motion)
8. **Mid-body, front, laughing, warm window lighting** — diversity (expression, indoor, natural light)
9. **Mid-body, 3/4 rear, over-shoulder look, neutral lighting** — diversity (unusual angle, intrigue)
10. **Face close-up, 3/4 left, contemplative, dramatic lighting** — diversity (mood, emotion)

**Distribution by crop**:
- 30% face close-ups (images 1, 5, 10) — captures expressions and facial details
- 50% mid-body (images 2, 4, 6, 8, 9) — most versatile, shows clothing and pose
- 20% full body (images 3, 7) — shows proportions and full clothing

**Reference Strategy**:
- Image 1: no references (user image only)
- Image 2: ref [1] (face defined, now show body)
- Image 3: refs [1, 2] (face + body defined, complete proportions)
- Image 4: refs [1, 2, 3] (all anchors defined, introduce angle variation)
- Image 5: refs [2, 3, 4] (drop oldest, add newest, profile view)
- Image 6: refs [3, 4, 5] (mid-body angle variation continues)
- Image 7: refs [4, 5, 6] (building toward full body walking pose)
- Image 8: refs [5, 6, 7] (emotional variation, indoor setting)
- Image 9: refs [6, 7, 8] (unusual angle with continuity)
- Image 10: refs [7, 8, 9] (final diversity, dramatic mood)

Each generation uses up to 5 reference images, building on prior generations.

### 1.2 Gemini API Integration (`stages/gemini_snowball.py` — NEW)

Create a new stage that:
- Takes user's uploaded image
- Generates 10 images in snowball order, each referencing 0-5 prior images
- Stores all 10 in `{job_dir}/gemini_diverse/`
- Emits SSE event for each completed image
- Returns list of 10 image paths

**Config**:
- Model: `gemini-3-pro-image-preview` (currently the best for character diversity)
- Quality: ensure diversity of lighting, angles, expressions

**Implementation**:
```python
class GeminiSnowballGenerator:
    def generate(self, user_image_path, job_dir):
        """
        Generate 10 diverse images with snowball referencing.

        Returns:
            list[str] — paths to 10 generated images in order
        """
        # Snowball referencing table
        refs_per_image = [
            [],           # img 1: no refs
            [0],          # img 2: ref img 1
            [0, 1],       # img 3: refs imgs 1-2
            [0, 1, 2],    # img 4: refs imgs 1-3
            [1, 2, 3],    # img 5: refs imgs 2-4
            [2, 3, 4],    # img 6: refs imgs 3-5
            [3, 4, 5],    # img 7: refs imgs 4-6
            [4, 5, 6],    # img 8: refs imgs 5-7
            [5, 6, 7],    # img 9: refs imgs 6-8
            [6, 7, 8],    # img 10: refs imgs 7-9
        ]

        prompts = [
            "Face close-up, front view, neutral expression, white background, professional headshot, sharp focus",
            "Mid-body portrait, front view, neutral expression, white background, professional, clear clothing",
            "Full body standing pose, front view, neutral expression, white background, complete outfit visible",
            "Mid-body pose, 3/4 left view, slight smile, soft diffused lighting, relaxed pose",
            "Face close-up, right profile, serious expression, studio lighting, dramatic shadows",
            "Mid-body pose, 3/4 right view, casual relaxed pose, warm golden hour lighting, outdoor setting",
            "Full body walking pose, left side view, overcast outdoor lighting, natural movement",
            "Mid-body pose, front view, laughing expression, warm window light from left, indoor setting",
            "Mid-body pose, 3/4 rear view, looking over shoulder, neutral lighting, mysterious",
            "Face close-up, 3/4 left view, contemplative expression, dramatic side lighting, mood lighting",
        ]

        generated = []
        for i, (prompt, ref_indices) in enumerate(zip(prompts, refs_per_image)):
            refs = [generated[j] for j in ref_indices] if ref_indices else []
            img = self.gemini_api.generate(
                user_image=user_image_path if i == 0 else None,
                prompt=prompt,
                reference_images=refs,
            )
            path = f"{job_dir}/gemini_diverse/img_{i+1:02d}.png"
            img.save(path)
            generated.append(path)
            # emit SSE event

        return generated
```

### 1.3 Form Parameter & UI Changes

- Add checkbox: "Use Snowball Generation" (default true)
- If unchecked, fall back to v0.1 behavior (5 views)
- Remove the old multi-view generation stage if snowball is enabled

### 1.4 Job Directory Structure

```
{job_dir}/
├── gemini_diverse/
│   ├── img_01.png     — Face close-up front
│   ├── img_02.png     — Mid-body front
│   ├── img_03.png     — Full body front
│   ├── img_04.png     — Mid-body 3/4 left
│   ├── img_05.png     — Face close-up right profile
│   ├── img_06.png     — Mid-body 3/4 right
│   ├── img_07.png     — Full body left walking
│   ├── img_08.png     — Mid-body front laughing
│   ├── img_09.png     — Mid-body 3/4 rear
│   └── img_10.png     — Face close-up 3/4 left
└── ...
```

---

## 2. Stage 2: Florence 2 Caption the 10 Gemini Images

### 2.1 New Stage: `stages/caption_gemini.py`

Create a stage that:
- Auto-captions the 10 Gemini images with Florence 2
- Strips identity using the existing `identity_stripper.py` (removes names, descriptive phrases)
- Prepends trigger word
- Outputs `{job_dir}/gemini_diverse/img_NN.txt` for each image

**Example Caption Processing**:
```
Raw Florence 2: "A woman with long brown hair, wearing a red dress, smiling at the camera"
→ Strip identity: "A person wearing a red dress, smiling at the camera"
→ Prepend trigger: "chrx person wearing a red dress, smiling at the camera"
```

**Special handling for expressions**: The identity stripper should PRESERVE expression words (smiling, serious, contemplative, etc.) because these are critical for the LoRA to capture emotional range.

---

## 3. Stage 3: Generate Regularization Images

### 3.1 New Stage: `stages/generate_regularization.py`

Generate 20-30 generic "person" images from the SRPO base model to prevent concept bleed during identity LoRA training.

**Config**:
- Model: SRPO base (`rockerBOO/flux.1-dev-SRPO`)
- Count: 20-30 images (user-configurable)
- Prompts: generic person prompts (e.g., "a person standing", "a person walking", "a person in a room", etc.)
- Resolution: 1024px (matches identity training resolution)
- Storage: `{job_dir}/regularization/`

**Implementation**:
```python
class RegularizationGenerator:
    def __init__(self, base_model="rockerBOO/flux.1-dev-SRPO"):
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            base_model, torch_dtype=torch.bfloat16
        )

    def generate(self, count=25, output_dir=""):
        prompts = [
            "a person standing", "a person sitting", "a person walking",
            "a person in a room", "a person outdoors", "a person at work",
            "a person in casual clothes", "a person in formal clothes",
            "a person smiling", "a person serious", "a person thinking",
            # ... expand to 20-30 generic prompts
        ]

        for i in range(count):
            prompt = prompts[i % len(prompts)]
            img = self.pipe(
                prompt, height=1024, width=1024, guidance_scale=5.0
            ).images[0]
            img.save(f"{output_dir}/reg_{i:03d}.png")
            # emit SSE event
```

**Why regularization images?** Without them, the LoRA may overfit to the 10 images and lose the ability to generate diverse "person" poses and lighting. The regularization images act as a control set that says "remember, these are still just people in generic situations."

---

## 4. Stage 4: Train Identity LoRA

### 4.1 New Stage: `stages/train_identity_lora.py`

Train a LoRA on the 10 Gemini images + 20-30 regularization images using the SRPO base model.

**Config**:
- Base model: `rockerBOO/flux.1-dev-SRPO` (23.8 GB BF16, full FLUX.1-dev fine-tune for photorealism)
- LoRA rank: 32 (community research shows 32 is optimal for character LoRAs, not 16)
- Optimizer: Prodigy (auto learning rate, excellent for small datasets)
- Learning rate: 1.0 (Prodigy will auto-adjust)
- Steps: ~1500-2000 (depends on total images = 10 + regularization count)
- Resolution: 1024px (smaller for faster identity capture)
- Caption dropout: 0.10 (higher than final training to prevent overfitting to captions)
- Batch size: 1
- Checkpoint interval: save every 100 steps, sample every 100 steps
- Sample prompts: use the 10 Gemini image captions

**Training Data Structure**:
```
{job_dir}/identity_training_data/
├── gemini/
│   ├── img_01.png
│   ├── img_01.txt (caption)
│   ├── img_02.png
│   ├── img_02.txt
│   └── ... (10 images + captions)
└── regularization/
    ├── reg_001.png
    ├── reg_001.txt ("a person standing")
    ├── reg_002.png
    ├── reg_002.txt
    └── ... (20-30 images + captions)
```

**Output**:
- `{job_dir}/identity_lora/identity_lora.safetensors` — the trained LoRA
- Sample checkpoints at `{job_dir}/identity_lora/samples/`

**Why SRPO base, not Z-Image?** The SRPO base is a full fine-tune of FLUX.1-dev for photorealism. Z-Image is designed for speed/turbo inference. SRPO is the better choice for this identity capture step because we want photorealistic detail that carries through to the enhanced dataset.

### 4.2 Implementation Notes

- Use AI Toolkit's `LoRATrainer` wrapper, but with SRPO-specific config
- May need to add SRPO model to `model_manager.py` if not already supported
- Ensure full unload of SRPO after training (before synthesis starts)

---

## 5. Stage 5: Synthesis — Generate Massive Dataset

### 5.1 Expanded Dataset Synthesis (`stages/synthesize.py` — MODIFIED)

Expand the synthesis range and use Gemini images as explicit references.

**Config Changes**:
- `num_images` range: 10-300 (was 10-50)
- Default (non-enhanced mode): 25
- Default (enhanced mode): 100
- Batch size: auto-select based on num_images (1, 2, or 4)

**Reference Selection**:
- For Flux 2 DEV: use all 10 Gemini images as references (instead of 5 views)
- Prompt templates: expand to 150+ templates to support 300 images without repetition
- Template categories:
  - Action poses (running, sitting, reaching, dancing, leaping)
  - Environmental contexts (indoor, outdoor, urban, nature, kitchen, bedroom, office)
  - Lighting conditions (golden hour, studio, dramatic, overcast, neon, candlelight)
  - Emotional expressions (laughing, serious, contemplative, surprised, angry, loving)
  - Camera angles (low angle, high angle, dutch angle, over-shoulder, wide, macro)
  - Seasonal/weather (rain, snow, summer, autumn, spring, foggy, misty)

**Interleaving**: maintain ORIGINAL_OUTFIT / VARIED_OUTFIT pattern across all 150+ templates

**Implementation**:
```python
# In synthesize.py
def get_references_for_lora(synthesizer_type):
    if synthesizer_type == "flux2_dev":
        # Load all 10 Gemini images from {job_dir}/gemini_diverse/
        return load_images_from_dir(f"{job_dir}/gemini_diverse")
    elif synthesizer_type == "klein_kv":
        # Select subset: front face, mid-body, left, right
        return select_klein_references()

def get_prompt_templates_expanded(num_images):
    """Generate 150+ templates, cycle if needed."""
    templates = load_all_templates()  # 150+ templates
    return [templates[i % len(templates)] for i in range(num_images)]
```

### 5.2 Prompt Templates File (`utils/prompt_templates.py` — MODIFIED)

Expand from 50 templates to 150+ templates across categories:

```python
TEMPLATES_EXPANDED = {
    "flux2_dev": [
        # Original outfit + varied outfit, interleaved
        # 75 templates total (cycle for 150+)

        # Anchors (professional, neutral)
        "chrx person, professional headshot, neutral expression, studio lighting",
        "chrx person wearing varied outfit, professional headshot, neutral expression, studio lighting",

        # Action poses
        "chrx person running, dynamic motion, outdoor, sunny day",
        "chrx person wearing varied outfit, running, outdoor, sunny day",

        # Environmental contexts
        "chrx person in modern office, sitting at desk, working",
        "chrx person wearing varied outfit, in modern office, sitting at desk, working",

        # Emotional expressions
        "chrx person laughing with joy, warm indoor lighting",
        "chrx person wearing varied outfit, laughing with joy, warm indoor lighting",

        # Camera angles
        "chrx person, low angle shot, looking up, dramatic lighting",
        "chrx person wearing varied outfit, low angle shot, looking up, dramatic lighting",

        # Lighting conditions
        "chrx person, golden hour lighting, warm and soft, outdoor",
        "chrx person wearing varied outfit, golden hour lighting, warm and soft, outdoor",

        # Seasonal
        "chrx person, snow falling, winter clothing, cold blue lighting",
        "chrx person wearing varied outfit, snow falling, winter clothing, cold blue lighting",

        # ... continue to 150+ total
    ],
    "klein_kv": [
        # Similar expansion for Klein (4 refs instead of 10)
        # 75+ templates
    ]
}
```

---

## 6. Stage 6: SeedVR2 Upscale

### 6.1 Upscale Synthetic Dataset (`stages/upscale.py` — UNCHANGED)

No changes from v0.1. Upscales all synthetic images from 1024px → 2048px.

---

## 7. Stage 7: Florence 2 Caption Synthetic Dataset

### 7.1 Caption Synthetic Images (`stages/caption.py` — UNCHANGED)

No changes from v0.1. Auto-captions all upscaled synthetic images.

---

## 8. Stage 8: Clear Latent Cache

### 8.1 New Stage: `stages/clear_cache.py`

**CRITICAL**: After enhancement completes and before final training begins, clear all latent caches to prevent stale latent artifacts from poisoning the final LoRA training.

```python
import torch
import gc

def clear_all_caches():
    """Clear CUDA cache, PyTorch cache, and Python garbage collection."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    # Optional: clear transformers/diffusers cache
    import transformers
    if hasattr(transformers, 'cache_utils'):
        transformers.cache_utils.clear()
```

---

## 9. Stage 9: Dataset Enhancement

### 9.1 Enhanced Dataset with img2img (`stages/enhance.py` — NEW)

After upscaling and captioning, enhance each synthetic image using img2img with the identity LoRA.

**Process**:
1. Load SRPO base model + identity LoRA (NOT Z-Image)
2. For each synthetic image:
   - Load the 2048px upscaled image
   - Encode to latents (VAE encode)
   - Add noise (controlled by denoising strength)
   - Denoise with SRPO + identity LoRA + caption as prompt
   - Decode to pixels (VAE decode)
   - Save enhanced image, preserve original in `_pre_enhance/`

**Config**:
- Base model: SRPO (`rockerBOO/flux.1-dev-SRPO`)
- Identity LoRA: the one we just trained
- Denoising strength: 0.30 (configurable 0.15-0.50)
  - 0.15-0.25: minimal, subtle texture only
  - 0.25-0.35: sweet spot, adds realism + identity lock
  - 0.35-0.50: noticeable changes, risk of composition drift
- Inference steps: 25 (configurable 15-50)
- Guidance scale: 5.0 (configurable 3.0-8.0)
- LoRA weight: 0.8 (configurable 0.5-1.0) — controls identity lock strength
- Resolution: maintain original (2048px)
- Seed: deterministic per image (e.g., `seed = base_seed + image_index`) for reproducibility

**Output**:
- Enhanced images in `{job_dir}/dataset/`
- Originals preserved in `{job_dir}/dataset/_pre_enhance/`

**IMPORTANT**: The identity LoRA is applied to SRPO, NOT to Z-Image. This prevents model mixing and ensures enhancement uses the photorealistic SRPO baseline, not the turbo Z-Image.

**Implementation**:
```python
class DatasetEnhancer:
    def __init__(self, lora_path, denoise=0.30, lora_weight=0.8):
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "rockerBOO/flux.1-dev-SRPO", torch_dtype=torch.bfloat16
        )
        # Load identity LoRA
        self.pipe.load_lora_weights(lora_path)
        self.pipe.set_lora_device(weight=lora_weight)

    def enhance(self, image, prompt, seed, output_path):
        """Enhance single image via img2img."""
        image = Image.open(image) if isinstance(image, str) else image

        enhanced = self.pipe(
            prompt=prompt,
            image=image,
            strength=self.denoise,  # 0.30
            num_inference_steps=25,
            guidance_scale=5.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

        enhanced.save(output_path)
```

---

## 10. Stage 10: Train Final Z-Image LoRA

### 10.1 Final LoRA Training (`stages/train.py` — MODIFIED)

Unchanged logic, but with adjusted defaults for enhanced mode:

**Enhanced Mode Defaults** (when enhancement was used):
- Base model: Z-Image De-Turbo (unchanged)
- LoRA rank: 32-48 (community research, higher because cleaned data)
- Steps: scale with dataset size
  - 25 images → 1500 steps
  - 100 images → 2500 steps
  - 250+ images → 3500 steps
- Learning rate: scale with dataset size
  - 25 images → 1e-4
  - 100 images → 5e-5
  - 250+ images → 3e-5
- Batch size: auto-select
  - ≤50 images → batch 1
  - 50-150 images → batch 2
  - 150+ images → batch 4
- Resolution: 2048 (unchanged)
- Caption dropout: 0.05 (lower than identity training, data is cleaner)

**Non-Enhanced Mode Defaults** (legacy v0.1 behavior):
- All unchanged from current CLAUDE.md

---

## 11. Configuration & Parameters

### 11.1 New Form Parameters (POST /api/start)

```
use_snowball          — Enable 10-image snowball generation (bool, default true)
num_gemini_images     — Total Gemini snowball images (always 10 if snowball=true)
regularization_count  — Generic person images for regularization (5-30, default 20)
num_images            — Synthetic dataset size (10-300, default 25 non-enhanced, 100 enhanced)
batch_size            — Training batch size (1, 2, 4, auto-selected by default)

--- Enhanced Mode Parameters ---
enhanced_mode         — Enable enhancement pipeline (bool, default false)
identity_lora_rank    — Identity LoRA rank (8-64, default 32)
identity_steps        — Identity training steps (auto-calculated, but user-overridable)
enhance_denoise       — img2img denoising strength (0.15-0.50, default 0.30)
enhance_steps         — img2img inference steps (15-50, default 25)
enhance_guidance      — img2img guidance scale (3.0-8.0, default 5.0)
enhance_lora_weight   — Identity LoRA weight during enhancement (0.5-1.0, default 0.8)

--- Final Training Parameters ---
lora_rank             — Final LoRA rank (4-64, default 32 or auto)
lora_steps            — Training steps (250-5000, default 1500 or auto)
learning_rate         — Optimizer LR (default 1e-4 or auto)
caption_dropout       — Caption dropout (0.0-1.0, default 0.05 or 0.10)
```

### 11.2 Smart Defaults Engine

Auto-populate defaults based on configuration:

```python
def calculate_defaults(num_images, regularization_count, enhanced_mode):
    """Calculate recommended training parameters."""

    total_images = num_images

    if enhanced_mode:
        # Identity training: 10 Gemini + regularization
        identity_steps = max(1500, (10 + regularization_count) * 100)

        # Final training: scale with synthetic dataset
        if total_images <= 25:
            final_lora_steps = 1500
            final_lora_rank = 32
            final_lr = 1e-4
            batch_size = 1
        elif total_images <= 100:
            final_lora_steps = 2500
            final_lora_rank = 32
            final_lr = 5e-5
            batch_size = 2
        else:
            final_lora_steps = 3500
            final_lora_rank = 48
            final_lr = 3e-5
            batch_size = 4
    else:
        # Legacy v0.1 defaults
        final_lora_steps = 1500
        final_lora_rank = 32
        final_lr = 1e-4
        batch_size = 1

    return {
        'identity_steps': identity_steps if enhanced_mode else None,
        'final_lora_steps': final_lora_steps,
        'final_lora_rank': final_lora_rank,
        'final_lr': final_lr,
        'batch_size': batch_size,
    }
```

---

## 12. UI Changes

### 12.1 Enhanced Mode Toggle

- Add checkbox in settings panel: "Enhanced Quality Mode"
- When enabled, show additional sections:
  - Gemini generation settings (snowball strategy, 10 images fixed)
  - Regularization settings (5-30 generic images)
  - Enhancement settings (denoise, guidance, LoRA weight)
  - Identity LoRA settings (rank, steps)
  - Final training settings (auto-populated based on dataset size)

### 12.2 New Pipeline Stages in UI

**Non-Enhanced Mode** (v0.1 legacy):
```
01  GEMINI MULTI-VIEW GENERATION (5 views)
02  DATASET SYNTHESIS
03  SEEDVR2 UPSCALE
04  CAPTIONING
05  LORA TRAINING
06  CHECKPOINT SAMPLES
```

**Enhanced Mode** (v0.2):
```
01  GEMINI SNOWBALL GENERATION (10 diverse images)
02  CAPTION GEMINI IMAGES
03  GENERATE REGULARIZATION IMAGES
04  IDENTITY LORA TRAINING
05  DATASET SYNTHESIS (25-300 images)
06  SEEDVR2 UPSCALE
07  CAPTION SYNTHETIC DATASET
08  DATASET ENHANCEMENT (img2img + identity LoRA)
09  FINAL LORA TRAINING
10  CHECKPOINT SAMPLES
```

### 12.3 New UI Sections

#### Section: Gemini Snowball (Stage 1)
- Progress: "1 / 10 images generated"
- Grid showing 10 images as they complete
- Badges: "Anchor", "Angle Expansion", "Diversity"
- Tooltip: explains snowball strategy

#### Section: Gemini Captioning (Stage 2)
- Progress: "1 / 10 images captioned"
- Display captions as text (one per image, in grid)
- No images shown here (reuse from stage 1)

#### Section: Regularization (Stage 3)
- Progress: "1 / 20 images generated"
- Minimal UI (these are generic, not shown)
- Just a counter

#### Section: Identity LoRA Training (Stage 4)
- Progress bar: "step 100 / 1500"
- Label: "Training identity LoRA (rank 32, Prodigy optimizer)"
- Sample checkpoints at intervals (images from training set)
- Download button: optional "Download identity LoRA" (for experimentation)

#### Section: Dataset Synthesis (Stage 5)
- Progress: "5 / 100 images synthesized"
- Grid showing synthetic images as they complete
- Latent previews during diffusion (ephemeral SSE)
- Reuse existing synthesis UI

#### Section: SeedVR2 Upscale (Stage 6)
- Progress: "5 / 100 images upscaled"
- Reuse existing upscale UI

#### Section: Caption Synthetic Dataset (Stage 7)
- Progress: "5 / 100 captions completed"
- Optional: show captions in grid (toggle)

#### Section: Dataset Enhancement (Stage 8) — NEW
- Progress: "5 / 100 images enhanced"
- Grid showing enhanced images as they complete
- Before/after comparison slider:
  - Left: synthetic (pre-enhance, 2048px from SeedVR2)
  - Right: enhanced (LoRA + img2img result)
- Enhancement quality indicator: "Denoising strength: 0.30, LoRA weight: 0.80"

#### Section: Final LoRA Training (Stage 9)
- Progress bar: "step 500 / 2500"
- Checkpoint samples with per-step download button
- Training config shown: "rank 32, lr 5e-5, batch 2, caption_dropout 0.05"

### 12.4 Before/After Comparison (Enhancement Stage)

Reuse existing comparison slider component:
- Click enhanced image → fullscreen overlay
- Drag slider to transition: "2048px Synthetic" ↔ "Enhanced (Identity LoRA)"
- Metrics displayed: denoising strength used, LoRA weight, original synthesis time vs enhancement time

---

## 13. SSE Event Types

### 13.1 New SSE Events

```
stage                    — Pipeline stage status (existing, reuse)

--- Gemini Snowball Generation ---
gemini_image            — Single Gemini image generated (index + url)

--- Regularization ---
regularization_progress — Regularization generation (current + total)

--- Identity LoRA Training ---
identity_progress       — Identity training step (step + total)
identity_checkpoint     — Identity training checkpoint samples (step + urls)
identity_complete       — Identity training done (lora_path)

--- Enhancement ---
enhance_progress        — Enhancement progress (current + total)
enhanced                — Single image enhanced (index + original_url + enhanced_url)

--- Final Training (existing, reused) ---
progress                — Training step progress
checkpoint              — Checkpoint samples
complete                — Pipeline complete
```

---

## 14. API Endpoints

### 14.1 New Endpoints

```
GET  /api/download-identity-lora/<job_id>          — Download identity LoRA (optional)
GET  /api/images/<job_id>/dataset/_pre_enhance/... — Serve pre-enhancement images
GET  /api/images/<job_id>/gemini_diverse/...        — Serve Gemini snowball images
```

---

## 15. Training Configuration Summary

### 15.1 Identity LoRA Training

```
Base Model:          SRPO (rockerBOO/flux.1-dev-SRPO, 23.8 GB BF16)
LoRA Rank:           32
Optimizer:           Prodigy (auto learning rate, LR=1.0)
Resolution:          1024px
Caption Dropout:     0.10
Batch Size:          1
Training Data:       10 Gemini images + 20-30 regularization images
Checkpoints:         every 100 steps, sample every 100 steps
Duration:            ~45-60 min (10 images + 25 reg images on RTX 6000)
Output:              {job_dir}/identity_lora/identity_lora.safetensors
```

### 15.2 Final LoRA Training (Enhanced Mode, 100-Image Dataset)

```
Base Model:          Z-Image De-Turbo (ostris/Z-Image-De-Turbo)
LoRA Rank:           32-48 (auto-selected)
Optimizer:           adamw8bit
Learning Rate:       5e-5 (auto-selected)
Resolution:          2048px
Caption Dropout:     0.05
Batch Size:          2 (auto-selected)
Training Data:       100 enhanced synthetic images
Checkpoints:         every 250 steps
Duration:            ~90-120 min (100 images on RTX 6000)
Output:              {job_dir}/output/{trigger}_lora/
```

### 15.3 Prompt Template Distribution

For 100-image dataset (enhanced mode):
- 50% mid-body (50 images) — most versatile for character training
- 25% face close-ups (25 images) — expressions and facial details
- 15% full body (15 images) — proportions and posture
- 10% special/artistic (10 images) — unusual angles, artistic lighting

---

## 16. File Structure (New/Modified)

```
server.py                              — MODIFIED: orchestrate 10-stage pipeline
stages/
├── gemini_snowball.py                 — NEW: 10-image generation with refs
├── caption_gemini.py                  — NEW: caption Gemini images
├── generate_regularization.py         — NEW: generate 20-30 generic person images
├── train_identity_lora.py             — NEW: identity LoRA training (SRPO base)
├── synthesize.py                      — MODIFIED: expand to 300 images, use 10 refs
├── upscale.py                         — UNCHANGED: SeedVR2 upscale
├── caption.py                         — UNCHANGED: Florence 2 caption synthetic
├── clear_cache.py                     — NEW: clear latent caches before final training
├── enhance.py                         — NEW: img2img enhancement with identity LoRA
├── train.py                           — MODIFIED: auto-scaled defaults for enhanced mode
├── multiview.py                       — UNCHANGED: kept for v0.1 legacy
└── model_manager.py                   — MODIFIED: add SRPO model support
utils/
├── prompt_templates.py                — MODIFIED: expand to 150+ templates
├── identity_stripper.py               — MODIFIED: preserve expression words
└── checkpoint.py                      — UNCHANGED
static/
├── index.html                         — MODIFIED: enhanced mode toggle, 10 sections
├── app.js                             — MODIFIED: new SSE handlers, defaults engine
└── style.css                          — MODIFIED: new section styles, badges
start.sh                               — MODIFIED: add SRPO model deps
CLAUDE.md                              — MODIFIED: document v0.2 pipeline
```

---

## 17. Job Directory Structure (Enhanced Mode)

```
/workspace/character_jobs/{job_id}/
├── gemini_diverse/                    — 10 Gemini snowball images
│   ├── img_01.png (face close-up front)
│   ├── img_01.txt (caption)
│   ├── img_02.png (mid-body front)
│   ├── img_02.txt
│   └── ... (10 images + 10 captions)
├── regularization/                    — 20-30 generic person images
│   ├── reg_001.png
│   ├── reg_001.txt
│   └── ...
├── identity_training_data/            — Symlinks or copies for training
│   ├── gemini/  (10 images)
│   └── regularization/  (25 images)
├── identity_lora/                     — Identity LoRA outputs
│   ├── identity_lora.safetensors      — Final identity LoRA
│   ├── identity_lora_step_0100.safetensors  (optional: checkpoints)
│   └── samples/                       — Sample images during training
├── dataset/                           — Enhanced synthetic images
│   ├── img_001.png                    — Enhanced (final version)
│   ├── img_001.txt                    — Caption (same as pre-enhance)
│   ├── _originals/                    — Pre-SeedVR2 originals (1024px)
│   │   └── img_001.png
│   └── _pre_enhance/                  — Pre-enhancement (2048px SeedVR2)
│       └── img_001.png
└── output/
    └── {trigger}_lora/
        ├── {trigger}_lora_step{N}.safetensors
        └── samples/
```

---

## 18. Implementation Order

### Phase 1 — Foundation (Week 1)
- [ ] Add SRPO model to `model_manager.py`
- [ ] Expand prompt templates to 150+
- [ ] Add batch_size parameter to UI and trainer
- [ ] Implement smart defaults engine
- [ ] Add `enhanced_mode` toggle to form

### Phase 2 — Gemini Snowball (Week 1)
- [ ] Implement `stages/gemini_snowball.py`
- [ ] Wire into `server.py` pipeline (conditional on `enhanced_mode`)
- [ ] Add UI section for Gemini generation (10-image grid with badges)
- [ ] Test in isolation

### Phase 3 — Regularization (Week 1-2)
- [ ] Implement `stages/generate_regularization.py`
- [ ] Wire into pipeline
- [ ] Add UI progress counter (minimal display)
- [ ] Test in isolation

### Phase 4 — Identity LoRA Training (Week 2)
- [ ] Implement `stages/train_identity_lora.py` (using AI Toolkit + SRPO)
- [ ] Wire into pipeline
- [ ] Add UI section with progress bar and optional checkpoint samples
- [ ] VRAM management: ensure proper unload/load sequence
- [ ] Test in isolation

### Phase 5 — Synthesis & Upscale (Week 2)
- [ ] Modify `stages/synthesize.py` to use 10 Gemini images as refs
- [ ] Expand prompt templates to 150+
- [ ] Test dataset sizes: 25, 100, 250, 300
- [ ] Test VRAM scaling with batch sizes 1, 2, 4

### Phase 6 — Captioning (Week 2)
- [ ] New stage: `caption_gemini.py`
- [ ] Reuse existing Florence 2 captioning for synthetic dataset
- [ ] Verify identity stripping preserves expressions
- [ ] Test

### Phase 7 — Cache Clearing (Week 2)
- [ ] Implement `stages/clear_cache.py`
- [ ] Wire between enhancement and final training
- [ ] Verify cache clearing removes latent artifacts
- [ ] Test

### Phase 8 — Enhancement (Week 3)
- [ ] Implement `stages/enhance.py` (img2img + identity LoRA)
- [ ] Load SRPO + identity LoRA (NOT Z-Image mixing)
- [ ] Wire into pipeline after captioning synthetic
- [ ] Add UI section: grid + before/after comparison slider
- [ ] SSE events: enhance_progress, enhanced
- [ ] Test VRAM usage at 2048px resolution
- [ ] Test denoising parameter sweep (0.15, 0.30, 0.50)

### Phase 9 — Final Training & Integration (Week 3)
- [ ] Modify `stages/train.py` auto-scaling defaults for enhanced mode
- [ ] Full pipeline test: enhanced mode OFF (verify identical to v0.1)
- [ ] Full pipeline test: enhanced mode ON, 25 images
- [ ] Full pipeline test: enhanced mode ON, 100 images
- [ ] Full pipeline test: enhanced mode ON, 250 images
- [ ] Test VRAM at all dataset sizes
- [ ] Test SSE replay on reconnect

### Phase 10 — Quality & Polish (Week 3-4)
- [ ] Enhancement preview (pause after 3 images, show in UI)
- [ ] A/B comparison: first-pass vs final LoRA inference
- [ ] Pipeline resume for crashed enhancement
- [ ] Time estimation after identity training completes
- [ ] RunPod end-to-end testing
- [ ] Quality validation: compare enhanced vs non-enhanced LoRAs on same character
- [ ] Update CLAUDE.md and README

### Phase 11 — Release (Week 4)
- [ ] Tag v0.2 release
- [ ] Update documentation
- [ ] Test suite: unit + integration + quality

---

## 19. Testing Plan

### 19.1 Unit Tests

```python
# tests/test_gemini_snowball.py
def test_snowball_reference_indices():
    """Verify reference indices match expected snowball pattern."""
    refs = get_snowball_references()
    assert refs[0] == []
    assert refs[1] == [0]
    assert refs[9] == [6, 7, 8]

# tests/test_prompt_templates.py
def test_template_expansion():
    """Verify 150+ templates expand correctly."""
    templates = get_prompt_templates_expanded(300)
    assert len(templates) == 300
    assert templates[0] != templates[1]  # no immediate repeats

# tests/test_smart_defaults.py
def test_defaults_scaling():
    """Verify auto-defaults scale correctly."""
    d25 = calculate_defaults(25, 20, enhanced=True)
    d100 = calculate_defaults(100, 20, enhanced=True)
    assert d100['final_lr'] < d25['final_lr']  # lower LR for more data
    assert d100['final_lora_rank'] >= d25['final_lora_rank']  # higher rank for more data

# tests/test_identity_stripper.py
def test_preserve_expressions():
    """Verify expressions are preserved when stripping identity."""
    caption = "A woman laughing with joy wearing a red dress"
    stripped = identity_stripper.strip(caption)
    assert "laughing" in stripped or "laugh" in stripped.lower()
    assert "A woman" not in stripped  # identity removed
```

### 19.2 Integration Tests (RunPod)

```python
# tests/integration/test_v01_compatibility.py
def test_enhanced_mode_off_produces_v01_pipeline():
    """Verify enhanced_mode=False produces identical pipeline to v0.1."""
    # Run with enhanced_mode=False, num_images=25
    # Verify stages: gemini (5 views), synthesis, upscale, caption, train
    # Verify output quality matches v0.1 baseline

# tests/integration/test_snowball_generation.py
def test_snowball_10_images():
    """Verify 10-image snowball generates correctly."""
    # Run gemini_snowball on sample character
    # Verify 10 images in gemini_diverse/
    # Verify images have correct characteristics (face crops, angles, etc.)

# tests/integration/test_identity_training.py
def test_identity_lora_training():
    """Verify identity LoRA trains on 10+reg images."""
    # Train identity LoRA on 10 Gemini + 20 reg images
    # Verify output .safetensors file
    # Verify checkpoint samples generated
    # Check VRAM peak usage < 96 GB

# tests/integration/test_enhancement.py
def test_enhancement_quality():
    """Verify enhancement improves image quality."""
    # Enhance 25 synthetic images
    # Compare texture detail (face, hair, fabric)
    # Verify identity consistency (face, clothing)
    # Verify no composition drift at denoise=0.30

# tests/integration/test_full_pipeline_25_images.py
def test_full_enhanced_pipeline_25_images():
    """Full pipeline: enhanced mode ON, 25 images."""
    # Run all 10 stages
    # Verify all outputs present
    # Verify time estimates are accurate
    # Verify VRAM never exceeds 96 GB

# tests/integration/test_full_pipeline_100_images.py
def test_full_enhanced_pipeline_100_images():
    """Full pipeline: enhanced mode ON, 100 images."""
    # Verify batch_size auto-scales to 2
    # Verify learning rate scales appropriately
    # Verify all 100 images enhanced
    # Measure total duration

# tests/integration/test_vram_scaling.py
def test_vram_scaling_all_stages():
    """Verify VRAM management across all stages."""
    # Profile VRAM at each stage transition
    # Verify no leaks
    # Verify proper model unload
    # Test with datasets: 25, 100, 250, 300 images

# tests/integration/test_sse_replay.py
def test_sse_event_replay():
    """Verify SSE events replay on reconnect."""
    # Start pipeline, let it run 3 stages
    # Disconnect, reconnect
    # Verify history replays completely
    # Verify new live events arrive correctly
```

### 19.3 Quality Validation

```python
# tests/quality/test_enhancement_effectiveness.py
def test_enhancement_texture_improvement():
    """Verify enhanced images have more texture detail."""
    # Enhance 5 images at denoise=0.30
    # Compare pixel-level detail (edges, gradients)
    # Verify face has more pore-level detail
    # Verify hair strands more visible

# tests/quality/test_identity_consistency.py
def test_identity_consistency_across_enhanced():
    """Verify character identity consistent across enhanced dataset."""
    # Train final LoRA on 100 enhanced images
    # Generate 20 test prompts on final LoRA
    # Visual inspection: face consistent? clothing consistent?

# tests/quality/test_expression_preservation.py
def test_expressions_work_in_prompts():
    """Verify expressions in captions produce expression variation."""
    # Generate images with prompts containing "laughing", "serious", "surprised"
    # Visual: does character laugh/look serious correctly?

# tests/quality/test_comparison_enhanced_vs_plain.py
def test_enhanced_lora_vs_v01():
    """Compare final LoRA trained on enhanced vs plain synthetic data."""
    # Train two LoRAs: one on enhanced 100 images, one on plain 100 images
    # Generate identical test prompts
    # Blind A/B comparison: which is higher quality?
    # Measure: texture detail, identity consistency, no artifacts
```

---

## 20. Open Questions

1. **SRPO availability in AI Toolkit?** Does the AI Toolkit library support `rockerBOO/flux.1-dev-SRPO` as a base model, or do we need to add custom support?

2. **Gemini image diversity at snowball step 10?** After 9 prior generations, does Gemini still produce diverse results at step 10, or does it start collapsing/repeating? May need to test with different prompt wording.

3. **Regularization image count?** Is 20-30 regularization images sufficient to prevent concept bleed, or do we need more? This depends on how much the identity LoRA specializes.

4. **Enhancement VRAM at 2048px?** Running img2img at 2048px is VRAM-intensive. Can we fit SRPO + identity LoRA + image in 96 GB? If not, can we enhance at 1024px and re-upscale?

5. **Denoising strength optimal range?** Empirically, what denoise value (0.15-0.50) produces the best texture improvement without composition drift? May vary by character style (realistic, anime, stylized).

6. **Identity LoRA learning rate?** Prodigy auto-adjusts LR, but should we seed it differently for the 10-image case? Is LR=1.0 always optimal?

7. **Should regularization images be visible in UI?** Currently they're hidden (just a counter). Would it help users to see sample regularization images?

8. **A/B comparison interface?** For the enhancement preview and final LoRA comparison, should we show side-by-side or use a slider? User testing needed.

9. **Resume capability for enhancement stage?** If enhancement crashes after 50/100 images, can we skip already-enhanced images and resume? Requires tracking mtime or checksums.

10. **Should first-pass LoRA be downloadable?** It's an intermediate artifact, but some users might want it for experimentation. Current plan: offer as optional download, default hidden.

---

## 21. Key Technical Decisions

1. **SRPO for identity LoRA, not Z-Image**: SRPO is a full fine-tune of FLUX.1-dev for photorealism. Z-Image is optimized for speed/turbo. For capturing character identity with maximum detail, SRPO is superior. Z-Image is reserved for the final training because it's proven and faster.

2. **img2img with denoise=0.30, not 0.50+**: Higher denoise risks composition drift (the pose might shift). At 0.30, we add texture detail while preserving composition. This is the empirical sweet spot across different character types.

3. **Identity LoRA at 1024px, not 2048px**: Faster training (identity capture doesn't require 2048px), and the enhancement step upscales the result. At 1024px, we can capture identity in ~1 hour instead of 3+ hours.

4. **Rank 32 for identity LoRA, rank 32-48 for final**: Community research shows rank 32 is optimal for character LoRAs. The identity LoRA uses 32 to stay lightweight and fast. The final LoRA can go to 48 if the enhanced dataset is large (100+) because cleaner data supports more capacity.

5. **Prodigy optimizer for identity training**: Small datasets (10-30 images) benefit from adaptive learning rates. Prodigy auto-adjusts, which is ideal for the diverse 10-image set.

6. **Florence 2 caption stripping preserves expressions**: Expressions (laughing, serious, contemplative) are CRITICAL to train. We only strip identity traits (names, descriptive phrases like "with long brown hair"), not emotions.

7. **Latent cache clear between enhancement and final training**: Stale latents from the enhancement stage can poison the final LoRA training. Explicit cache clearing prevents this subtle, hard-to-debug issue.

---

## 22. Success Criteria

- [ ] v0.1 backward compatibility: `enhanced_mode=false` produces identical pipeline and output quality to current v0.1
- [ ] Snowball generation: 10 diverse images generated with correct referencing strategy
- [ ] Identity LoRA training: trains on 10 Gemini + regularization images in <2 hours on RTX 6000
- [ ] Enhancement: 100 synthetic images enhanced in <3 hours, visual improvement in texture detail
- [ ] Final training: 2500-step training on 100 enhanced images in <2 hours
- [ ] VRAM management: no OOM at any stage, peak never exceeds 96 GB
- [ ] Quality: final LoRA trained on enhanced data produces noticeably better quality output than v0.1
- [ ] Scalability: pipeline works with 25, 100, 250, 300-image datasets
- [ ] SSE reliability: all events replay correctly on reconnect
- [ ] User experience: UI clearly shows all 10 stages, smart defaults working, before/after comparison clear
