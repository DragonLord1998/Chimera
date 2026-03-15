# Chimera v0.2 — Enhanced Pipeline with Dataset Bootstrapping

## Overview

v0.2 introduces a **two-pass training pipeline** that produces significantly higher quality
character LoRAs by using a bootstrapping workflow: train a first-pass LoRA on the synthetic
dataset, use it to enhance every image via img2img (adding realism while locking identity),
then train the final LoRA on the enhanced dataset.

```
v0.1 Pipeline (current):
  Gemini → Klein → SeedVR2 → Florence 2 → Z-Image LoRA

v0.2 Pipeline (proposed):
  Gemini → Klein → SeedVR2 → Florence 2
    → First-Pass LoRA (SRPO, identity capture)
    → Dataset Enhancer (img2img + LoRA, low denoise)
    → Enhanced Dataset (identity-locked, hyper-realistic)
    → Final Z-Image LoRA (trained on pristine data)
```

---

## 1. First-Pass LoRA Training (Identity Capture)

### 1.1 New stage: `stages/train_first_pass.py`

- [ ] Create `FirstPassTrainer` class wrapping AI Toolkit
- [ ] Purpose: fast identity capture, NOT final quality
- [ ] Default config:
  - Base model: Flux SRPO (or configurable — Z-Image, Flux Krea)
  - Rank: 16 (configurable 8-32 via UI)
  - Steps: 750 (configurable 500-2500 via UI)
  - Learning rate: 1e-4
  - Resolution: 1024 (lower than final — speed matters here)
  - Optimizer: adamw8bit
  - Noise scheduler: flowmatch
  - Batch size: 1
  - No sample generation (skip `sample` block — unnecessary for intermediate LoRA)
  - No checkpoint saves (only final output needed)
  - Caption dropout: 0.05
- [ ] Output: single `.safetensors` file in `{job_dir}/first_pass_lora/`
- [ ] Cleanup: unload model, clear VRAM after training
- [ ] The first-pass LoRA is a disposable artifact — not offered for download

### 1.2 Model selection for first pass

- [ ] Research and decide which base model to use for the first-pass LoRA:
  - **Flux SRPO** — if available and supported by AI Toolkit
  - **Z-Image De-Turbo** — known working, same as final pass
  - **FLUX.1-Krea-dev** — alternative
- [ ] The first-pass base model does NOT need to be the same as the final base model
- [ ] Key requirement: the base model must produce realistic outputs for the img2img
      enhancement step to work well
- [ ] Add `first_pass_model` config option (default to same as `base_model`)

### 1.3 VRAM management

- [ ] First-pass training must fully unload before enhancement stage begins
- [ ] Sequence: Synthesis model unloaded → First-pass trains → First-pass unloaded
      → Enhancement model loaded with LoRA → Enhancement done → Enhancement unloaded
      → Final training begins
- [ ] All stages share VRAM — only one model in memory at a time (existing pattern)

---

## 2. Dataset Enhancer (img2img + LoRA)

### 2.1 New stage: `stages/enhance.py`

- [ ] Create `DatasetEnhancer` class
- [ ] Loads the first-pass LoRA onto a base diffusion model
- [ ] Processes each image in the dataset via **img2img** (not txt2img)
- [ ] Uses the existing caption `.txt` file for each image as the prompt
- [ ] Preserves originals in `{job_dir}/dataset/_pre_enhance/` for comparison

### 2.2 Enhancement parameters

- [ ] Denoising strength: `0.3` (default, configurable 0.15-0.5 via UI)
  - 0.15-0.25: minimal change, subtle texture improvement
  - 0.25-0.35: sweet spot — adds realism while preserving composition
  - 0.35-0.50: noticeable changes, risk of pose/composition drift
- [ ] Inference steps: `25` (default, configurable 15-50)
- [ ] Guidance scale: `5.0` (default, configurable 3.0-8.0)
- [ ] Resolution: maintain original (2048px from SeedVR2 upscale)
- [ ] Seed: use a different seed per image for variety, but make it deterministic
      (e.g., `seed = base_seed + image_index`) for reproducibility
- [ ] LoRA scale/weight: `0.8` (default, configurable 0.5-1.0)
  - Higher = stronger identity lock, risk of homogenization
  - Lower = more variety, weaker identity
- [ ] Batch processing: process images one at a time (VRAM safety) or in small
      batches if VRAM allows

### 2.3 What the enhancer does to each image

```
Input:  synthetic image (clean but slightly plastic from Klein/SeedVR2)
        + caption text
        + first-pass character LoRA

Process:
  1. Encode image to latents (VAE encode)
  2. Add noise to latents (controlled by denoising strength)
  3. Denoise with model + LoRA + caption as prompt
  4. Decode latents to pixels (VAE decode)

Output: same composition/pose but with:
  ✓ Consistent character identity (LoRA-locked)
  ✓ Realistic skin texture (pores, micro-detail)
  ✓ Natural hair strands (not smooth blobs)
  ✓ Fabric texture and wrinkles
  ✓ Realistic lighting interaction with surfaces
  ✓ Subtle imperfections that make it look "real"
```

### 2.4 Real-time progress

- [ ] Emit SSE events during enhancement:
  - New event type: `enhanced` (index + original_url + enhanced_url)
  - Reuse `progress` event type for overall enhancement progress
  - Optional: latent preview during each image's denoising (ephemeral SSE)
- [ ] UI should show enhancement progress as a new pipeline section
- [ ] Before/after comparison: original synthetic vs enhanced (reuse comparison slider)

### 2.5 Re-captioning decision

- [ ] After enhancement, should we re-caption with Florence 2?
  - **No** (recommended): the original captions describe the scene, which hasn't changed.
    Re-captioning might introduce inconsistencies if Florence interprets enhanced
    details differently.
  - **Optional**: offer a "re-caption after enhancement" toggle for users who want it
- [ ] If re-captioning: strip identity again, prepend trigger word (same as original flow)

---

## 3. Final LoRA Training (on Enhanced Dataset)

### 3.1 Adjusted defaults for enhanced mode

- [ ] When enhanced mode is enabled, adjust final training defaults:
  - Rank: 32-48 (higher — cleaner data supports more capacity)
  - Steps: scale with dataset size:
    - 25 images → 1500 steps
    - 100 images → 2500 steps
    - 250 images → 3500 steps
  - Learning rate: scale with dataset size:
    - 25 images → 1e-4
    - 100 images → 5e-5
    - 250+ images → 3e-5
  - Batch size: auto-select based on dataset size:
    - ≤50 images → batch 1
    - 50-150 images → batch 2
    - 150+ images → batch 4
  - Resolution: 2048 (unchanged)
- [ ] These are defaults — user can still override everything in the UI

### 3.2 Training on enhanced dataset

- [ ] The final trainer reads from `{job_dir}/dataset/` which now contains
      enhanced images (originals in `_pre_enhance/`)
- [ ] Captions remain in `{job_dir}/dataset/*.txt` (unchanged by enhancement)
- [ ] No changes needed to `stages/train.py` — it already reads from dataset dir

---

## 4. Increased Synthesis Capacity

### 4.1 Expand num_images range

- [ ] Current: 10-50 images (default 25)
- [ ] v0.2: 10-300 images (default 25, enhanced mode default 100)
- [ ] Update UI slider/input range
- [ ] Update server-side validation

### 4.2 Prompt template expansion

- [ ] Current: 50 templates per synthesizer (25 original outfit + 25 varied outfit)
- [ ] v0.2: expand to 150+ templates to support 300 images without repetition
- [ ] New template categories:
  - [ ] Action poses (running, sitting, reaching, dancing)
  - [ ] Environmental contexts (indoor, outdoor, urban, nature)
  - [ ] Lighting conditions (golden hour, studio, dramatic, overcast)
  - [ ] Emotional expressions (laughing, serious, contemplative, surprised)
  - [ ] Camera angles (low angle, high angle, dutch angle, over-shoulder)
  - [ ] Seasonal/weather variations (rain, snow, summer, autumn)
- [ ] Templates should cycle if num_images exceeds template count
- [ ] Maintain the ORIGINAL_OUTFIT / VARIED_OUTFIT interleaving pattern

### 4.3 Batch size support in UI

- [ ] Add batch_size setting to the form (1, 2, 4)
- [ ] Auto-suggest based on num_images:
  - Show recommendation text: "Recommended: batch 2 for 100+ images"
- [ ] Pass to trainer via `batch_size` parameter

---

## 5. Server-Side Changes

### 5.1 Pipeline orchestration (`server.py`)

- [ ] Add `enhanced_mode` form parameter (boolean toggle, default false)
- [ ] When enhanced mode is enabled, the pipeline becomes:
  ```
  Stage 0:  Model download (existing)
  Stage 1:  Multi-view generation (existing)
  Stage 2:  Dataset synthesis (existing)
  Stage 2a: SeedVR2 upscale (existing)
  Stage 2b: Florence 2 captioning (existing)
  Stage 3:  First-pass LoRA training (NEW)
  Stage 3a: Dataset enhancement (NEW)
  Stage 4:  Final LoRA training (existing, renumbered)
  ```
- [ ] Emit `stage` SSE events for new stages:
  - `"First-pass LoRA training (identity capture)..."`
  - `"Enhancing dataset with character LoRA..."`
- [ ] Track both LoRA outputs: first-pass (disposable) and final (downloadable)
- [ ] When enhanced mode is disabled, pipeline is identical to v0.1 (no regressions)

### 5.2 New form parameters

```
enhanced_mode       — Enable two-pass training (boolean, default false)
first_pass_rank     — First-pass LoRA rank (8-32, default 16)
first_pass_steps    — First-pass training steps (500-2500, default 750)
enhance_denoise     — img2img denoising strength (0.15-0.50, default 0.30)
enhance_steps       — img2img inference steps (15-50, default 25)
enhance_lora_weight — LoRA weight during enhancement (0.5-1.0, default 0.8)
```

### 5.3 New SSE event types

```
first_pass_progress  — First-pass training step progress (step + total)
enhanced             — Image enhanced (index + original_url + enhanced_url)
enhance_progress     — Overall enhancement progress (current + total images)
```

### 5.4 New API endpoints

```
GET  /api/download-first-pass/<job_id>  — Download first-pass LoRA (optional)
GET  /api/images/<job_id>/dataset/_pre_enhance/...  — Serve pre-enhancement originals
```

---

## 6. Frontend Changes

### 6.1 Enhanced mode toggle

- [ ] Add "Enhanced Mode" toggle switch in the settings panel
- [ ] When toggled ON:
  - Show additional settings section: "Enhancement Settings"
  - Show first-pass rank, first-pass steps, denoise strength, etc.
  - Update default num_images to 100
  - Update default lora_steps based on dataset size
  - Show info tooltip explaining the two-pass workflow
- [ ] When toggled OFF:
  - Hide enhancement settings
  - Revert to v0.1 defaults

### 6.2 New pipeline sections in UI

- [ ] **Section: First-Pass Training** (between captioning and final training)
  - Progress bar (step / total)
  - Label: "Identity Capture" with first-pass rank/steps shown
  - No checkpoint samples (first pass skips sample generation)
- [ ] **Section: Dataset Enhancement** (between first-pass and final training)
  - Grid showing enhanced images as they complete
  - Before/after comparison slider (synthetic vs enhanced)
  - Progress counter: "12 / 100 images enhanced"
  - Optional: latent preview during each image's enhancement
- [ ] **Section: Final Training** (existing training section, relabeled)
  - Progress bar, checkpoint samples, download buttons (unchanged)

### 6.3 Before/after comparison (enhancement)

- [ ] Reuse existing comparison slider component
- [ ] Click enhanced image → overlay with:
  - Left: "Synthetic (pre-enhance)" from `_pre_enhance/`
  - Right: "Enhanced (LoRA + img2img)" from `dataset/`
- [ ] Badge on enhanced cells: "Enhanced" (similar to existing "2048px" badge)

### 6.4 Updated stage numbers in UI

```
v0.1:                              v0.2 (enhanced mode):
01  MULTI-VIEW GENERATION          01  MULTI-VIEW GENERATION
02  DATASET SYNTHESIS              02  DATASET SYNTHESIS
03  SEEDVR2 UPSCALE                03  SEEDVR2 UPSCALE
04  CAPTIONING                     04  CAPTIONING
05  LORA TRAINING                  05  FIRST-PASS TRAINING
06  CHECKPOINT SAMPLES             06  DATASET ENHANCEMENT
                                   07  FINAL LORA TRAINING
                                   08  CHECKPOINT SAMPLES
```

---

## 7. Smart Defaults Engine

### 7.1 Auto-calculate training parameters based on dataset size

- [ ] When num_images changes, auto-update recommended values:
  ```
  num_images    lr        steps     rank    batch
  ─────────────────────────────────────────────────
  10-25         1e-4      1500      32      1
  26-50         8e-5      2000      32      1
  51-100        5e-5      2500      32-48   2
  101-200       4e-5      3000      48      2
  201-300       3e-5      3500      48-64   4
  ```
- [ ] Show as "Recommended" values — user can still override
- [ ] Display in UI: "Recommended for 100 images: lr=5e-5, steps=2500, rank=48"

### 7.2 Enhancement parameter recommendations

- [ ] Based on first-pass configuration:
  ```
  first_pass_rank    first_pass_steps    denoise
  ────────────────────────────────────────────────
  8-16               500-750             0.30-0.40  (weaker LoRA, more denoise needed)
  16-24              750-1500            0.25-0.35  (balanced)
  32+                1500-2500           0.20-0.30  (strong LoRA, less denoise needed)
  ```

---

## 8. Quality-of-Life Improvements

### 8.1 Enhancement preview

- [ ] After enhancing the first 3 images, pause and show preview in UI
- [ ] User can approve ("Continue") or adjust denoise strength and retry
- [ ] This prevents wasting time enhancing 250 images with bad settings
- [ ] Optional: make this a toggle ("Auto-enhance" vs "Preview first")

### 8.2 A/B comparison in final output

- [ ] If enhanced mode was used, the complete event should include both:
  - Path to enhanced LoRA
  - Path to first-pass LoRA (for comparison)
- [ ] User can download both and compare inference results

### 8.3 Pipeline resume

- [ ] If enhancement crashes mid-way, skip already-enhanced images on retry
- [ ] Check: if `{img_name}` exists in `dataset/` and has newer mtime than
      `_pre_enhance/{img_name}`, it was already enhanced
- [ ] Resume from the first un-enhanced image

### 8.4 Estimated time display

- [ ] After first-pass training completes, estimate enhancement time:
  - `(time_per_image × remaining_images)` displayed in UI
- [ ] After first enhanced image completes, refine the estimate

---

## 9. Testing Plan

### 9.1 Unit tests

- [ ] `test_first_pass_trainer.py` — config generation, output path discovery
- [ ] `test_dataset_enhancer.py` — img2img parameter validation, file preservation
- [ ] Test enhanced mode ON/OFF produces correct pipeline stage sequence
- [ ] Test smart defaults calculation for various num_images values

### 9.2 Integration tests (on RunPod)

- [ ] Full pipeline: enhanced mode OFF — verify identical to v0.1
- [ ] Full pipeline: enhanced mode ON, 25 images — verify all stages complete
- [ ] Full pipeline: enhanced mode ON, 100 images — verify scaling works
- [ ] Verify `_pre_enhance/` originals are preserved correctly
- [ ] Verify comparison slider works for enhanced images
- [ ] Verify SSE event replay works for new event types (reconnect test)
- [ ] Verify existing dataset reuse works with enhanced mode
- [ ] Verify checkpoint downloads work during final training
- [ ] VRAM test: verify no OOM at any stage transition

### 9.3 Quality validation

- [ ] Generate same character with enhanced mode ON vs OFF
- [ ] Compare final LoRA outputs on identical inference prompts
- [ ] Verify enhanced dataset images have better texture/detail
- [ ] Verify character identity consistency across enhanced images
- [ ] Check for artifact amplification (does the LoRA introduce any?)

---

## 10. File Structure (New/Modified)

```
server.py                          — MODIFIED: new stages, form params, SSE events
stages/
├── train.py                       — UNCHANGED (final LoRA training)
├── train_first_pass.py            — NEW: first-pass LoRA trainer
├── enhance.py                     — NEW: dataset enhancer (img2img + LoRA)
├── synthesize.py                  — UNCHANGED
├── upscale.py                     — UNCHANGED
├── caption.py                     — UNCHANGED
├── multiview.py                   — UNCHANGED
└── model_manager.py               — MODIFIED: add SRPO model if needed
utils/
├── prompt_templates.py            — MODIFIED: expand to 150+ templates
├── identity_stripper.py           — UNCHANGED
└── checkpoint.py                  — UNCHANGED
static/
├── index.html                     — MODIFIED: enhanced mode toggle, new sections
├── app.js                         — MODIFIED: new event handlers, comparison, defaults
└── style.css                      — MODIFIED: new section styles, enhanced badge
start.sh                           — MODIFIED: add SRPO model deps if needed
CLAUDE.md                          — MODIFIED: document new stages and parameters
```

---

## 11. Job Directory Structure (Enhanced Mode)

```
/workspace/character_jobs/{job_id}/
├── views/                         — Gemini multi-view images
├── dataset/
│   ├── img_001.png                — Enhanced images (final training data)
│   ├── img_001.txt                — Captions (unchanged by enhancement)
│   ├── _originals/                — Pre-SeedVR2 originals (1024px)
│   └── _pre_enhance/              — Pre-enhancement images (post-SeedVR2, 2048px)
│       ├── img_001.png
│       └── ...
├── first_pass_lora/
│   └── first_pass.safetensors     — Intermediate LoRA (disposable)
└── output/
    └── {trigger}_lora/
        ├── {trigger}_lora_step{N}.safetensors
        └── samples/
```

---

## 12. Implementation Order

```
Phase 1 — Foundation (do first)
  1. Expand num_images range to 300
  2. Expand prompt templates to 150+
  3. Add batch_size to UI and trainer
  4. Smart defaults engine

Phase 2 — First-Pass Training
  5. Create stages/train_first_pass.py
  6. Wire into server.py pipeline
  7. Add UI section for first-pass progress
  8. Test first-pass training in isolation

Phase 3 — Dataset Enhancement
  9. Create stages/enhance.py
  10. Wire into server.py pipeline
  11. Add UI section with grid + comparison
  12. SSE events for enhancement progress
  13. Test enhancement in isolation

Phase 4 — Integration
  14. Enhanced mode toggle in UI
  15. Full pipeline testing (ON and OFF)
  16. Smart defaults for enhanced mode
  17. Enhancement preview (pause after 3 images)

Phase 5 — Polish
  18. Pipeline resume for crashed enhancement
  19. Time estimation display
  20. A/B comparison of first-pass vs final LoRA
  21. Update CLAUDE.md documentation
  22. RunPod end-to-end testing
  23. Tag v0.2 release
```

---

## Open Questions

1. **Which base model for first pass?** Z-Image (known working) vs Flux SRPO vs Flux Krea?
   Need to test which produces the best img2img enhancement results.

2. **Should the first-pass LoRA be downloadable?** It's meant to be disposable, but some
   users might want it for experimentation. Leaning toward offering it as an optional download.

3. **Re-caption after enhancement?** The scene hasn't changed, but enhanced details
   (sharper face, visible jewelry, etc.) might warrant updated captions. Needs testing.

4. **Enhancement at 2048px or downscale to 1024px?** Running img2img at 2048px is
   VRAM-intensive. Could enhance at 1024px and re-upscale with SeedVR2, but that
   adds another upscale pass. Test VRAM usage at 2048px first.

5. **Optimal denoising strength?** 0.3 is theoretical — needs empirical testing across
   different character types (realistic, anime, stylized) to find the sweet spot.
