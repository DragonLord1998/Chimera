#!/usr/bin/env python3
"""
Chimera — CLI entrypoint.

Usage:
    python run.py --image character.png --trigger chrx --gemini-key YOUR_KEY

All stages run sequentially with console output. For a web UI, use app.py instead.
"""
import argparse
import os
import sys
import datetime
import gc

import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Chimera")
    parser.add_argument("--image", required=True, help="Path to character image")
    parser.add_argument("--trigger", required=True, help="Trigger word (e.g. 'chrx')")
    parser.add_argument("--gemini-key", required=True, help="Google Gemini API key")
    parser.add_argument("--hf-token", default="", help="HuggingFace token for gated models")
    parser.add_argument("--num-images", type=int, default=25, help="Number of training images (default: 25)")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps (default: 1000)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--models-dir", default="/workspace/models", help="Models directory")
    parser.add_argument("--jobs-dir", default="/workspace/character_jobs", help="Jobs directory")
    parser.add_argument("--resume", default="", help="Job ID to resume")
    args = parser.parse_args()

    from stages.model_manager import ModelManager
    from stages.multiview import MultiViewGenerator
    from stages.synthesize import DatasetSynthesizer
    from stages.caption import CaptionGenerator
    from stages.train import LoRATrainer
    from utils.checkpoint import CheckpointManager

    # Job setup
    if args.resume:
        job_id = args.resume
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() else "_" for c in args.trigger)
        job_id = f"{safe}_{ts}"

    job_dir = os.path.join(args.jobs_dir, job_id)
    ckpt = CheckpointManager(job_dir)
    ckpt.create_job_dir()

    stage1_dir = os.path.join(job_dir, "stage1")
    dataset_dir = os.path.join(job_dir, "dataset")
    output_dir = os.path.join(job_dir, "output")
    toolkit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai-toolkit")

    print(f"\n{'='*60}")
    print(f"  Chimera")
    print(f"  Job ID: {job_id}")
    print(f"  Job dir: {job_dir}")
    print(f"{'='*60}\n")

    # Stage 0
    print("=== STAGE 0: Models ===")
    model_mgr = ModelManager(base_path=args.models_dir, hf_token=args.hf_token or None)
    model_mgr.ensure_all_models()

    # Stage 1
    if not ckpt.is_stage_complete("stage1_complete"):
        print("\n=== STAGE 1: Multi-view generation (Gemini) ===")
        pil_input = Image.open(args.image).convert("RGB")
        mv = MultiViewGenerator(api_key=args.gemini_key)
        view_paths, _ = mv.generate_views(pil_input, stage1_dir)
        ckpt.mark_stage_complete("stage1_complete", {"view_paths": view_paths})
    else:
        print("\n=== STAGE 1: Already complete, skipping ===")
        view_paths = [os.path.join(stage1_dir, f"{n}.png") for n in ["left", "front", "right"]]

    # Stage 2
    if not ckpt.is_stage_complete("stage2_complete"):
        print("\n=== STAGE 2: Dataset synthesis + captioning ===")
        refs = [Image.open(p).convert("RGB") for p in view_paths]
        resume_from = ckpt.get_resume_point("stage2_generation")

        synth = DatasetSynthesizer(
            hf_token=args.hf_token or None,
        )
        synth.load_model()

        def progress(cur, total):
            print(f"  Image {cur}/{total}")
            ckpt.update_progress("stage2_generation", cur, total)

        synth.synthesize_dataset(refs, dataset_dir, args.num_images, resume_from, progress)
        synth.unload_model()
        del synth

        for r in refs:
            r.close()

        print("  Captioning with Florence 2...")
        cap = CaptionGenerator(model_path=model_mgr.get_model_path("florence2"))
        cap.load_model()
        cap.caption_dataset(dataset_dir, args.trigger)
        cap.unload_model()
        del cap

        torch.cuda.empty_cache()
        gc.collect()
        ckpt.mark_stage_complete("stage2_complete")
    else:
        print("\n=== STAGE 2: Already complete, skipping ===")

    # Stage 3
    if not ckpt.is_stage_complete("stage3_complete"):
        print("\n=== STAGE 3: LoRA training ===")
        trainer = LoRATrainer(
            model_path=model_mgr.get_model_path("zimage_base"),
            toolkit_path=toolkit_path,
        )
        lora_path = trainer.train(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            output_name=f"{args.trigger}_lora",
            trigger_word=args.trigger,
            rank=args.rank,
            learning_rate=args.lr,
            steps=args.steps,
        )
        trainer.cleanup()
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        ckpt.mark_stage_complete("stage3_complete", {"lora_path": lora_path})
    else:
        meta = ckpt.get_stage_metadata("stage3_complete")
        lora_path = meta.get("lora_path", "")

    ckpt.cleanup()
    print(f"\n{'='*60}")
    print(f"  DONE! LoRA saved to: {lora_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
