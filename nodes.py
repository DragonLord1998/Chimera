"""
Chimera — ComfyUI mega-node that takes a single character image
and produces a trained Z-Image LoRA.

Pipeline stages:
    0. Auto-download all required models from HuggingFace
    1. Generate left/front/right views via Google Nano Banana Pro API
    2. Synthesize 25 training images using Flux 2 DEV (local) + caption with Florence 2
    3. Train a Z-Image LoRA using AI Toolkit by Ostris
    4. Assemble output: LoRA path + preview grid

Each stage writes a checkpoint marker on completion.  If the pod restarts
mid-job, re-queuing the node resumes from the last completed stage.

Designed for RunPod with NVIDIA RTX PRO 6000 (96 GB VRAM).
"""

from __future__ import annotations

import gc
import os
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# ComfyUI utilities — available at runtime inside the ComfyUI process.
import comfy.utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a ComfyUI IMAGE tensor [1, H, W, 3]."""
    arr = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor [B, H, W, C] to a PIL image (first frame)."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _build_preview_grid(image_dir: str, lora_path: str, cols: int = 4) -> Image.Image:
    """Build a preview grid from the training dataset images.

    Returns a single PIL image showing a grid of up to 16 training samples.
    """
    png_files = sorted(Path(image_dir).glob("*.png"))[:16]
    if not png_files:
        # Return a small placeholder if no images found
        placeholder = Image.new("RGB", (512, 512), (40, 40, 40))
        return placeholder

    thumb_size = 256
    images = []
    for p in png_files:
        with Image.open(p) as img:
            images.append(img.convert("RGB").resize((thumb_size, thumb_size), Image.LANCZOS))

    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (20, 20, 20))

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * thumb_size, r * thumb_size))

    return grid


def _generate_job_id(trigger_word: str) -> str:
    """Create a unique job directory name from trigger word and timestamp."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_trigger = "".join(c if c.isalnum() else "_" for c in trigger_word)
    return f"{safe_trigger}_{ts}"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class Chimera:
    """
    Single-node pipeline: 1 character image -> trained Z-Image LoRA.

    Internally orchestrates:
    - Model auto-download (HuggingFace)
    - Multi-view generation (Google Nano Banana Pro API)
    - Training dataset synthesis (Flux 2 DEV local)
    - Auto-captioning (Florence 2) with identity trait stripping
    - LoRA training (AI Toolkit by Ostris -> Z-Image)

    Checkpoints every stage for RunPod crash resilience.
    """

    CATEGORY = "training/character-lora"
    FUNCTION = "execute"
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("lora_path", "preview_grid")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "trigger_word": ("STRING", {
                    "default": "chrx",
                    "tooltip": "Unique token for this character (e.g. 'chrx', 'ohwx'). Use something that isn't a real word.",
                }),
                "google_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Google Nano Banana Pro API key for multi-view generation.",
                }),
                "num_images": ("INT", {
                    "default": 25,
                    "min": 10,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Number of training images to synthesize. 15-30 is the sweet spot.",
                }),
                "lora_rank": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "step": 4,
                    "tooltip": "LoRA rank. Higher = more detail but larger file. 16 is recommended.",
                }),
                "lora_steps": ("INT", {
                    "default": 1000,
                    "min": 250,
                    "max": 5000,
                    "step": 250,
                    "tooltip": "Total training steps. ~40 steps per training image is a good rule.",
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.0001,
                    "min": 0.00001,
                    "max": 0.001,
                    "step": 0.00001,
                    "tooltip": "AdamW8Bit learning rate. 1e-4 is recommended for character LoRAs.",
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace token for gated models (Flux 2 DEV). Also reads HF_TOKEN env var.",
                }),
                "models_dir": ("STRING", {
                    "default": "/workspace/models",
                    "tooltip": "Directory for downloaded models. Use persistent volume on RunPod.",
                }),
                "jobs_dir": ("STRING", {
                    "default": "/workspace/character_jobs",
                    "tooltip": "Directory for job outputs. Each run creates a subdirectory.",
                }),
                "resume_job_id": ("STRING", {
                    "default": "",
                    "tooltip": "Job ID to resume (e.g. 'chrx_20260313_091500'). Leave empty for new job.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute — training is stateful via checkpoints, not ComfyUI cache.
        return float("NaN")

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(
        self,
        image: torch.Tensor,
        trigger_word: str,
        google_api_key: str,
        num_images: int,
        lora_rank: int,
        lora_steps: int,
        learning_rate: float,
        hf_token: str = "",
        models_dir: str = "/workspace/models",
        jobs_dir: str = "/workspace/character_jobs",
        resume_job_id: str = "",
    ) -> tuple:
        """
        Orchestrate the full Character LoRA creation pipeline.

        Returns (lora_path: str, preview_grid: IMAGE tensor).
        """
        from .stages.model_manager import ModelManager
        from .stages.multiview import MultiViewGenerator
        from .stages.synthesize import DatasetSynthesizer
        from .stages.caption import CaptionGenerator
        from .stages.train import LoRATrainer
        from .utils.checkpoint import CheckpointManager

        # ---- Job setup ----
        job_id = resume_job_id if resume_job_id else _generate_job_id(trigger_word)
        job_dir = os.path.join(jobs_dir, job_id)

        ckpt = CheckpointManager(job_dir)
        ckpt.create_job_dir()

        stage1_dir = os.path.join(job_dir, "stage1")
        dataset_dir = os.path.join(job_dir, "dataset")
        output_dir = os.path.join(job_dir, "output")

        ai_toolkit_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ai-toolkit"
        )

        print(f"[Chimera] Job ID: {job_id}")
        print(f"[Chimera] Job directory: {job_dir}")

        # ============================================================
        # STAGE 0: Auto-download models
        # ============================================================
        if not ckpt.is_stage_complete("stage0_models_ready"):
            print("[Chimera] === STAGE 0: Downloading models ===")
            model_mgr = ModelManager(
                base_path=models_dir,
                hf_token=hf_token if hf_token else None,
            )
            model_mgr.ensure_all_models()
            ckpt.mark_stage_complete("stage0_models_ready")
        else:
            print("[Chimera] Stage 0 already complete — skipping model download.")
            model_mgr = ModelManager(base_path=models_dir)

        # ============================================================
        # STAGE 1: Multi-view generation (Google API)
        # ============================================================
        if not ckpt.is_stage_complete("stage1_complete"):
            print("[Chimera] === STAGE 1: Multi-view generation ===")
            pil_input = _tensor_to_pil(image)

            mv_gen = MultiViewGenerator(api_key=google_api_key)
            view_paths = mv_gen.generate_views(pil_input, stage1_dir)

            ckpt.mark_stage_complete("stage1_complete", metadata={
                "view_paths": view_paths,
            })
            print(f"[Chimera] Stage 1 complete — {len(view_paths)} views generated.")
        else:
            print("[Chimera] Stage 1 already complete — loading views from disk.")
            view_paths = [
                os.path.join(stage1_dir, f"{name}.png")
                for name in ["left", "front", "right"]
            ]

        # ============================================================
        # STAGE 2: Synthesize training dataset + caption
        # ============================================================
        if not ckpt.is_stage_complete("stage2_complete"):
            print("[Chimera] === STAGE 2: Dataset synthesis + captioning ===")

            # Load reference images
            reference_images = [Image.open(p).convert("RGB") for p in view_paths]

            # -- 2a: Generate images with Flux 2 DEV --
            resume_from = ckpt.get_resume_point("stage2_generation")
            if resume_from > 0:
                print(f"[Chimera] Resuming generation from image {resume_from + 1}/{num_images}")

            synthesizer = DatasetSynthesizer(
                hf_token=hf_token if hf_token else None,
            )
            synthesizer.load_model()

            pbar = comfy.utils.ProgressBar(num_images)

            def _synth_progress(current: int, total: int) -> None:
                pbar.update_absolute(current, total)
                ckpt.update_progress("stage2_generation", current, total)

            synthesizer.synthesize_dataset(
                reference_images=reference_images,
                output_dir=dataset_dir,
                num_images=num_images,
                start_from=resume_from,
                progress_callback=_synth_progress,
            )

            synthesizer.unload_model()
            del synthesizer

            # Close reference images
            for ref in reference_images:
                ref.close()

            # -- 2b: Caption with Florence 2 --
            print("[Chimera] Captioning dataset with Florence 2...")
            captioner = CaptionGenerator(
                model_path=model_mgr.get_model_path("florence2"),
            )
            captioner.load_model()
            captioner.caption_dataset(dataset_dir, trigger_word)
            captioner.unload_model()
            del captioner

            # Final VRAM cleanup before training
            torch.cuda.empty_cache()
            gc.collect()

            ckpt.mark_stage_complete("stage2_complete", metadata={
                "num_images": num_images,
                "trigger_word": trigger_word,
            })
            print(f"[Chimera] Stage 2 complete — {num_images} images + captions.")
        else:
            print("[Chimera] Stage 2 already complete — dataset exists on disk.")

        # ============================================================
        # STAGE 3: Train Z-Image LoRA
        # ============================================================
        if not ckpt.is_stage_complete("stage3_complete"):
            print("[Chimera] === STAGE 3: LoRA training ===")

            trainer = LoRATrainer(
                model_path=model_mgr.get_model_path("zimage_base"),
                toolkit_path=ai_toolkit_path,
            )

            lora_path = trainer.train(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                output_name=f"{trigger_word}_lora",
                trigger_word=trigger_word,
                rank=lora_rank,
                learning_rate=learning_rate,
                steps=lora_steps,
            )

            trainer.cleanup()
            del trainer

            torch.cuda.empty_cache()
            gc.collect()

            ckpt.mark_stage_complete("stage3_complete", metadata={
                "lora_path": lora_path,
                "rank": lora_rank,
                "steps": lora_steps,
                "learning_rate": learning_rate,
            })
            print(f"[Chimera] Stage 3 complete — LoRA saved to {lora_path}")
        else:
            print("[Chimera] Stage 3 already complete — loading LoRA path.")
            meta = ckpt.get_stage_metadata("stage3_complete")
            lora_path = meta.get("lora_path", "")
            if not lora_path:
                # Fallback: find the safetensors file
                candidates = sorted(Path(output_dir).rglob("*.safetensors"))
                lora_path = str(candidates[-1]) if candidates else ""

        # ============================================================
        # STAGE 4: Build preview grid and return
        # ============================================================
        print("[Chimera] === STAGE 4: Assembling output ===")

        preview_pil = _build_preview_grid(dataset_dir, lora_path)
        preview_tensor = _pil_to_tensor(preview_pil)

        # Clean up checkpoints (job is fully complete)
        ckpt.cleanup()

        print(f"[Chimera] Pipeline complete!")
        print(f"[Chimera] LoRA: {lora_path}")
        print(f"[Chimera] Job: {job_dir}")

        return {"ui": {"text": [lora_path]}, "result": (lora_path, preview_tensor)}
