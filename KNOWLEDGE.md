# Chimera — Knowledge Base & Project Journal

> This is a living document. Update it after every significant decision, mistake, discovery, or architectural change. Read it at the start of every session.

---

## Table of Contents

1. [Character LoRA Training — What We Learned](#1-character-lora-training--what-we-learned)
2. [Architectural Decisions & Rationale](#2-architectural-decisions--rationale)
3. [Mistakes Made & How We Fixed Them](#3-mistakes-made--how-we-fixed-them)
4. [The SRPO Model — Everything We Know](#4-the-srpo-model--everything-we-know)
5. [Captioning Philosophy](#5-captioning-philosophy)
6. [The Snowball Strategy](#6-the-snowball-strategy)
7. [Regularization Images](#7-regularization-images)
8. [VRAM & RAM Management](#8-vram--ram-management)
9. [AI Toolkit Quirks](#9-ai-toolkit-quirks)
10. [UI/UX Decisions](#10-uiux-decisions)
11. [Community Research Findings](#11-community-research-findings)
12. [Open Questions & Future Work](#12-open-questions--future-work)

---

## 1. Character LoRA Training — What We Learned

### The Captioning Rule
**"Caption what you do NOT want the LoRA to learn."**

This is the single most important insight for character LoRA training:
- Anything described in the caption is "explained away" to the base model's text encoder
- The LoRA only learns the **residual** — what's in the image but NOT in the caption
- For character identity: DO NOT caption hair color, eye color, skin tone, facial structure
- For character identity: DO caption poses, expressions, lighting, clothing, backgrounds

**Expressions are special**: Always tag expressions (smiling, serious, contemplative). If you don't, the LoRA averages all expressions into an unnatural neutral face. Expressions are NOT identity — they're transient states.

### Optimal Training Configuration (Community-Validated)
- **Rank 32** is the sweet spot for character LoRAs (not 16, not 64)
- **Prodigy optimizer** (LR=1.0, auto-adjusting) is best for small datasets (<20 images)
- **AdamW8bit** (LR=1e-4 to 5e-4) is best for larger datasets (20+ images)
- **Caption dropout**: 0.05-0.10 for character LoRAs
- **Resolution**: Train at model's native resolution (1024 for Flux)
- **Steps**: 2000-4000 for 20-30 images
- **Three-stage cropping**: 30% face close-ups, 50% mid-body, 20% full body

### Dataset Size
- Sweet spot: 15-30 images
- Minimum viable: 10 images (our identity LoRA approach)
- Quality always beats quantity — 15 excellent > 100 mediocre
- For 10 images: need regularization images to prevent overfitting

### Flux-Specific Training
- Use **natural language captions**, not Booru tags
- **Freeze text encoders** — don't train T5 or CLIP
- Use **FlowMatch** noise scheduler
- Guidance scale: 3.5 during training
- `cache_latents_to_disk: True` saves VRAM but creates stale cache risk

---

## 2. Architectural Decisions & Rationale

### Decision: Two-Pass Pipeline (v0.2)
**What**: Train an identity LoRA on 10 high-quality Gemini images, then use it to enhance the massive synthetic dataset before final training.

**Why**: Solves the chicken-and-egg problem. Instead of training the first-pass LoRA on mediocre synthetic data (v0.1 approach), we train on 10 Gemini-generated images that are small in number but high in quality. The identity LoRA captures the character's essence, then injects it into the synthetic dataset via img2img.

**Alternative rejected**: Training first-pass LoRA on the synthetic dataset itself (the old v0.2 plan). Problem: the synthetic images already contain the identity in the pixels — the LoRA just learns what's already there. With 10 HQ Gemini images, the LoRA learns from better source material.

### Decision: SRPO as Identity Training Base (Not Z-Image, Not Vanilla FLUX.1-dev)
**What**: Use the full SRPO base model (Tencent's photorealism fine-tune of FLUX.1-dev) for identity LoRA training.

**Why**: SRPO is specifically fine-tuned for photorealism. The identity LoRA trained on SRPO captures photorealistic detail that transfers to the enhancement stage. Z-Image is optimized for speed/turbo inference, not photorealism. Vanilla FLUX.1-dev is good but SRPO is better at realistic textures.

**Critical detail**: SRPO only ships transformer weights, not the full pipeline. We build a "hybrid directory" that symlinks FLUX.1-dev's tokenizer/VAE/scheduler/text_encoders with SRPO's transformer weights. AI Toolkit loads this as a normal Flux model.

### Decision: Z-Image for Final Training (Not SRPO)
**What**: The final LoRA trains on Z-Image De-Turbo, not SRPO.

**Why**: Z-Image is proven, faster, and uses a Qwen3 text encoder that handles complex prompts well. The enhanced dataset already has SRPO's photorealistic quality baked into the pixels — Z-Image just needs to learn from those pixels.

### Decision: Prodigy Optimizer for Identity LoRA
**What**: Use Prodigy (auto-adjusting LR) instead of AdamW8bit for the 10-image identity training.

**Why**: With only 10 images, manual LR tuning is fragile. Prodigy auto-adjusts and is the community default for small datasets. AdamW8bit is better for the final training on 100+ images where you want precise control.

### Decision: 10 Images with Snowball Referencing
**What**: Generate 10 diverse character images with Gemini, using progressive referencing (up to 5 refs per call).

**Why**:
- 10 images gives enough diversity for identity capture
- Snowball referencing means each generation sees prior generations, improving consistency
- Distribution: 30% face close-ups, 50% mid-body, 20% full body (research-backed)
- Some images (white bg, clean poses) double as synthesis references for Klein/Flux

### Decision: 0.30 Denoise Strength for Enhancement
**What**: Use 0.30 denoise (not 0.40 or higher) for the img2img enhancement pass.

**Why**:
- 0.15-0.25: too subtle, barely changes anything
- 0.25-0.35: sweet spot — adds realism without changing composition
- 0.35-0.50: too aggressive, risks pose/composition drift
- We initially had 0.40 as default — this was wrong and corrected to 0.30

---

## 3. Mistakes Made & How We Fixed Them

### Mistake 1: Changed Identity LoRA Base to FLUX.1-dev Without Permission
**What happened**: The code review agent flagged that AI Toolkit can't load a raw `.safetensors` file as a model directory. The fix agent changed the identity trainer to use vanilla `FLUX.1-dev` instead of SRPO — without asking the user.

**Why it was wrong**: The entire point of v0.2 is training on SRPO for its photorealistic quality. Silently downgrading to FLUX.1-dev defeats the purpose.

**How we fixed it**: Built a "hybrid directory" approach — symlink FLUX.1-dev's pipeline structure (tokenizer, VAE, etc.) with SRPO's transformer weights. AI Toolkit loads this as a complete Flux model. Same quality, proper directory structure.

**Lesson**: Never change a core architectural decision without explicit user approval, even if a review suggests it. Fix the technical problem, don't change the design.

### Mistake 2: System RAM OOM During Regularization (48GB > 46.57GB)
**What happened**: Loading FLUX.1-dev pipeline (~24GB) into RAM, then loading SRPO state dict (~24GB) to replace the transformer = ~48GB peak RAM. Container had 46.57GB.

**Why it happened**: The two-step load (load base model → replace transformer weights) temporarily holds both copies in RAM. This was fine for VRAM (GPU) because `enable_model_cpu_offload()` keeps things on CPU, but system RAM was the bottleneck we didn't anticipate.

**How we fixed it**: Used `from_pretrained(srpo_hybrid_dir)` to load SRPO weights directly from the hybrid directory. Single load, ~24GB RAM. Also added `low_cpu_mem_usage=True` to further reduce peak RAM.

**Lesson**: VRAM isn't the only memory constraint. On RunPod, system RAM (46.57GB) can be tighter than VRAM (96GB). Always consider both.

### Mistake 3: Stale Latent Cache — The Silent Killer
**What happened**: AI Toolkit's `cache_latents_to_disk=True` pre-encodes images to latents once. When the enhancement stage overwrites images, the cached latents become stale. If not cleared, final training trains on pre-enhancement data — the entire enhancement is silently wasted.

**How we caught it**: The code review identified this as a CRITICAL risk before it hit production.

**How we fixed it**: Added `LoRATrainer.clear_latent_cache(dataset_dir)` calls before both enhancement and final training. The method deletes `*_latent*.*`, `*_cached*.*`, and cache directories.

**Lesson**: When images are modified in-place, any cached representations must be invalidated. This applies to latent caches, thumbnail caches, etc.

### Mistake 4: Enhanced Mode Default Mismatch (Server vs UI)
**What happened**: Server defaulted `enhanced_mode` to `"true"`, but the UI checkbox was unchecked. When users submitted without touching the toggle, no `enhanced_mode` field was sent, server defaulted to true — giving them enhanced mode they didn't opt into.

**How we fixed it**:
1. Changed server default to `"false"`
2. Made JS always send explicit `"true"` or `"false"` (checkboxes don't send values when unchecked)

**Lesson**: HTML checkboxes are quirky — they don't send any value when unchecked. Always send an explicit value from JS and set conservative server defaults.

### Mistake 5: v0.2 View Names Didn't Map to UI Elements
**What happened**: The 5 UI placeholder slots had IDs like `viewLeft`, `viewFront`, etc. The v0.2 view names like `front_face_closeup` had no matching elements. All 10 Gemini images silently failed to display.

**How we fixed it**: Added 10 new placeholder slots in a separate enhanced grid (`viewsGridEnhanced`), with proper toggle between the 5-slot and 10-slot grids based on enhanced mode.

**Lesson**: When renaming internal identifiers (view names), trace all the way to the UI. Server → SSE events → JS handlers → DOM elements must all match.

### Mistake 6: Reg Images Appeared Dark/Greyed Out
**What happened**: CSS had `opacity: 0` with a `fade-in` animation. The animation wasn't triggering properly on dynamically appended images, leaving them invisible/dark.

**How we fixed it**: Removed the `opacity: 0` + animation, replaced with simple full opacity + hover effects.

**Lesson**: CSS animations on dynamically created elements can be unreliable. Use simpler approaches for injected DOM elements.

### Mistake 7: SSE Activity Timeout Killing Jobs During Regularization
**What happened**: The 5-minute inactivity timeout in the frontend was firing during regularization generation (~12.5 minutes for 25 images). The `reg_progress` event was emitted by the server but never listened to in the frontend, so it didn't reset the activity timer.

**How we fixed it**: Added `reg_progress` and `reg_image` event listeners, added both to the activity guard list.

**Lesson**: Every new SSE event type must be added to the activity guard list, or long-running stages will trigger false inactivity timeouts.

### Mistake 8: Page Reload During Pipeline Lost Enhanced Mode State
**What happened**: Auto-reconnect fetched the active job but didn't toggle the enhanced views grid or check the enhanced mode checkbox. The 10-slot grid was hidden, 5-slot grid was shown, SSE replay put images in the wrong (or no) slots.

**How we fixed it**: Reconnect logic now reads `data.params.enhanced_mode`, toggles grids, checks the checkbox before calling `connectToJob`.

**Lesson**: Auto-reconnect must fully restore UI state — not just reconnect the event stream. Every toggle, grid, and section that depends on a parameter must be set during reconnect.

---

## 4. The SRPO Model — Everything We Know

### What SRPO Is
- **Semantic Relative Preference Optimization** by Tencent
- Full-parameter fine-tune of FLUX.1-dev for photorealism
- Paper: arXiv 2509.06942
- NOT a LoRA — it's the complete 12B parameter transformer, retrained

### Available Versions
| Repo | Type | Size | Notes |
|------|------|------|-------|
| `tencent/SRPO` | Official FP32 | 47.6 GB | Too large, avoid FP32→FP8 conversion |
| `rockerBOO/flux.1-dev-SRPO` | Community BF16 | 23.8 GB | **Use this one** |
| `Alissonerdx/flux.1-dev-SRPO-LoRas` | LoRA extractions | ~1.5 GB | Distilled, not the full model |

### The Hybrid Directory Approach
SRPO only ships transformer weights. AI Toolkit needs a full model directory. Solution:
```
srpo_hybrid/
  tokenizer/       → symlink to FLUX.1-dev (HF cache)
  tokenizer_2/     → symlink to FLUX.1-dev
  text_encoder/    → symlink to FLUX.1-dev
  text_encoder_2/  → symlink to FLUX.1-dev
  vae/             → symlink to FLUX.1-dev
  scheduler/       → symlink to FLUX.1-dev
  model_index.json → symlink to FLUX.1-dev
  transformer/
    config.json                        → symlink to FLUX.1-dev
    diffusion_pytorch_model.safetensors → symlink to SRPO weights
```
Built once in Stage 0, reused across all jobs.

### RAM Considerations
- Loading SRPO: ~24GB system RAM
- MUST use `low_cpu_mem_usage=True` with `from_pretrained()`
- MUST use hybrid directory approach (not load + swap which doubles RAM)
- Container RAM on RunPod: 46.57GB — tight with SRPO + Python overhead

---

## 5. Captioning Philosophy

### For Identity LoRA (10 Gemini images)
- Use `<MORE_DETAILED_CAPTION>` from Florence 2
- Apply identity stripping (remove hair color, eye color, skin tone, facial structure)
- **Preserve expressions** (smiling, serious, contemplative) — they're NOT identity
- Prepend trigger word: `chrx, mid-body portrait, slight smile, soft lighting...`
- Caption dropout: 0.10 (higher than final, prevents overfitting to captions on small dataset)

### For Final Training (enhanced synthetic dataset)
- Same Florence 2 captioning
- Same identity stripping
- Same expression preservation
- Caption dropout: 0.05 (standard, cleaner data supports lower dropout)

### Why This Works
The trigger word `chrx` becomes the ONLY hook for the character's physical identity. All physical features get packed into the LoRA weights behind that trigger. Everything else (pose, lighting, clothing) is described in the caption and handled by the text encoder — the LoRA doesn't waste capacity learning them.

---

## 6. The Snowball Strategy

### The Problem
Gemini has no "character lock" mechanism. Each API call is independent. More diversity = more identity drift between images.

### The Solution
Progressive referencing — each generation feeds up to 5 prior images as references:

**Phase 1 (Anchors, Gen 1-3)**: White bg, neutral, front-facing. Builds identity foundation.
- Gen 1: [original] → face close-up
- Gen 2: [original, gen1] → mid-body
- Gen 3: [original, gen1, gen2] → full body

**Phase 2 (Angle Expansion, Gen 4-5)**: First non-frontal views with strong reference context.
- Gen 4: [original, gen1, gen2, gen3] → 3/4 left
- Gen 5: [original, gen1, gen2, gen3, gen4] → right profile (MAX 5 refs)

**Phase 3 (Diversity, Gen 6-10)**: Varied lighting, expressions, poses. 3 constant refs + 2 dynamic.
- Constants: [original, gen1_face, gen2_midbody]
- Dynamic: pick most relevant by angle + crop similarity

### Why Generation Order Matters
Early images bootstrap consistency for later ones. By Gen 10, the identity has been reinforced through 9 prior generations. Each image is consistent not just with the original, but with the entire set.

---

## 7. Regularization Images

### What They Are
20-25 generic "person" images generated from the SRPO base model before identity training.

### Why They're Needed
With only 10 training images, the LoRA aggressively overfits. Without regularization, it overwrites the base model's general understanding of "person" — so even prompts WITHOUT the trigger word produce the character's face (concept bleed).

Regularization images tell the model: "these are what generic people look like — don't forget this."

### How They Work
- Generated with SRPO txt2img at 1024px
- Captioned with generic prompts: "a person, full body, studio lighting"
- NO trigger word in their captions
- Added as a second dataset entry with `is_reg: True` and zero caption dropout
- Ratio: ~2.5x regularization per training image (25 reg : 10 training)

### Display
- Shown in a small thumbnail grid during generation
- Clickable for fullscreen view
- Progress counter: "5 / 25"

---

## 8. VRAM & RAM Management

### The Golden Rule
**Only one model in VRAM at a time.** Every stage loads → uses → unloads its model before the next stage starts.

### Stage Sequence (Enhanced Mode)
1. SRPO txt2img (reg images) → `del reg_pipe` + `gc.collect()` + `torch.cuda.empty_cache()`
2. SRPO + AI Toolkit (identity LoRA) → `trainer.cleanup()` + `gc.collect()` + `torch.cuda.empty_cache()`
3. Klein/Flux 2 (synthesis) → `synth.unload_model()` + `gc.collect()`
4. SeedVR2 (upscale) → subprocess, no VRAM leak
5. Florence 2 (captioning) → `cap.unload_model()` + `gc.collect()`
6. SRPO img2img + identity LoRA (enhancement) → `enhancer.unload_model()` + `gc.collect()`
7. Z-Image + AI Toolkit (final LoRA) → `trainer.cleanup()` + `gc.collect()`

### System RAM Gotchas
- Container RAM: 46.57 GB (RunPod)
- SRPO model: ~24 GB in RAM with CPU offload
- MUST use `low_cpu_mem_usage=True` on `from_pretrained()`
- MUST use hybrid directory (not load + swap which doubles RAM to ~48 GB)
- After state dict deletion, MUST call `gc.collect()` + `torch.cuda.empty_cache()`

### VRAM Budget (96 GB RTX PRO 6000)
- SRPO transformer: ~24 GB
- Z-Image transformer: ~12 GB + ~8 GB text encoder
- Klein 9B KV: ~29 GB
- SeedVR2 7B: ~24 GB
- Florence 2 Large: ~2 GB
- All comfortably fit individually; never loaded simultaneously

---

## 9. AI Toolkit Quirks

### Model Path Requirements
AI Toolkit's `run_job()` expects `name_or_path` to be either:
- A HuggingFace repo ID (downloads automatically)
- A local directory with full model structure (config.json, tokenizer/, transformer/, etc.)
- NOT a raw `.safetensors` file

This is why we build the hybrid directory for SRPO.

### Latent Caching
`cache_latents_to_disk: True` pre-encodes images once. **Must be cleared when images change** (e.g., after enhancement). Use `LoRATrainer.clear_latent_cache()`.

### Sampling Crash with Z-Image
AI Toolkit unconditionally calls `self.sample()` for baseline images, which crashes on Z-Image's Qwen3 tokenizer. Workaround: monkey-patch `BaseSDTrainProcess.sample` to a no-op when `sample_every >= steps`.

### Tokenizer None Handling
AI Toolkit's `encode_prompts_flux` passes `None` as the unconditional prompt text, which crashes transformers 5.x tokenizers. Patched by wrapping `PreTrainedTokenizerBase.__call__` to convert `None` → `""`.

### Transformers 5.x Compatibility
Extensive monkey-patching needed:
- Mock removed classes (ViTHybridImageProcessor, etc.)
- Patch `find_pruneable_heads_and_indices` back into pytorch_utils
- Patch `load_backbone` back into backbone_utils
- Auto-fallback to slow tokenizer when fast tokenizer fails

### Checkpoint Filename Format
AI Toolkit writes: `{output_name}_step{N}.safetensors` — zero-padded but width varies. Use flexible regex matching, not exact format strings.

---

## 10. UI/UX Decisions

### Enhanced Mode Toggle
- Checkbox in settings panel, unchecked by default
- Server defaults to `"false"` (conservative)
- JS always sends explicit `"true"` or `"false"` (checkbox quirk: unchecked sends nothing)
- Toggling shows/hides: 10-slot vs 5-slot view grid, enhancement settings, identity settings

### Fullscreen Modal
- Single reusable modal for all image types
- Detects image type via context:
  - Checkpoint image → shows "Download LoRA (Step N)" button
  - Synthetic image → fetches and shows caption overlay
  - View/reg image → plain fullscreen
- Close: click outside, ESC key, X button
- Body scroll locked when open

### Checkpoint Downloads
- Identity LoRA: save every 100 steps, sample every 100 steps
- Final LoRA: save every 250 steps, sample every 250 steps
- Click checkpoint sample → fullscreen → download button for that step's `.safetensors`

### Auto-Reconnect
- On page load, fetch `/api/jobs/active`
- If job running: toggle enhanced grid, check checkbox, init synthetic grid, reconnect SSE
- SSE replays full event history — UI rebuilds completely

### Stage Numbers
```
Enhanced mode (10 stages):        Legacy mode (3 stages):
01 MULTI-VIEW (10 images)         01 MULTI-VIEW (5 images)
02 CAPTION REFERENCES             02 SYNTHESIS + UPSCALE + CAPTION
03 REGULARIZATION IMAGES          03 LORA TRAINING
04 IDENTITY LORA TRAINING
05 DATASET SYNTHESIS
06 SEEDVR2 UPSCALE
07 CAPTION DATASET
08 DATASET ENHANCEMENT
09 FINAL LORA TRAINING
10 COMPLETE
```

---

## 11. Community Research Findings

### Sources Consulted (March 2026)
- CivitAI: 15+ guides from top LoRA creators
- SimpleTuner GitHub discussions (#634, #635)
- kohya_ss wiki and discussions
- Replicate blog (synthetic data for Flux fine-tunes)
- HuggingFace blog posts
- arXiv papers (LoRA, LoFT, SRPO)
- Multiple community comparison tests

### Key Findings
1. **Rank 32** is consensus optimal for character LoRAs (not 16, not 64)
2. **Prodigy optimizer** is the community default for beginners and small datasets (2025+)
3. **Natural language captions** dramatically outperform Booru tags on Flux
4. **Three-stage cropping** (face/mid/full) significantly improves fidelity
5. **Expression tagging** is critical — untagged expressions cause neutral face averaging
6. **Two-pass bootstrap** is a validated approach (documented by Replicate)
7. **Don't train text encoders** on Flux — freeze them
8. **Regularization images** are essential for datasets under 20 images
9. **Synthetic training data works** but quality << real photos; curate aggressively
10. **SRPO is "perfectly trainable with existing configs"** per community reports

---

## 12. Open Questions & Future Work

### Unvalidated Assumptions
1. Does the identity LoRA actually add value, or does the SRPO texture improvement alone account for most quality gain?
2. What's the optimal denoise strength? 0.30 is theoretical — needs A/B testing across character types
3. Is 10 Gemini images enough, or would 15-20 be significantly better?
4. Does the Gemini snowball strategy actually improve consistency vs independent generation?
5. How does enhancement at 2048px compare to enhancement at 1024px + re-upscale?

### Future Improvements
- Enhancement preview: pause after 3 images, let user approve before continuing
- A/B comparison: generate with identity LoRA vs final LoRA side-by-side
- Pipeline resume: skip already-enhanced images after crash
- Time estimation display after identity training completes
- Support for anime/stylized characters (different denoise settings?)
- Consider Qwen Image model as alternative to SRPO (reportedly outperforms at complex prompts)

### Performance Targets
- Identity LoRA training: <2 hours on RTX PRO 6000
- Enhancement (100 images): <3 hours
- Full pipeline: <6 hours for 100-image enhanced dataset

---

*Last updated: 2026-03-18 — v0.2 initial implementation complete*
