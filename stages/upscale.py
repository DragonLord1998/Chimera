"""
ImageUpscaler — uses SeedVR2 (ByteDance, 7B) to upscale training images
before LoRA training for maximum detail.

SeedVR2 is a one-step diffusion-transformer super-resolution model.
We use the ComfyUI standalone CLI (inference_cli.py) via subprocess
to avoid dependency conflicts with transformers 5.x / diffusers 0.32+.

The 7B model produces the best detail recovery for skin textures,
hair, and fabric. On a 96 GB RTX PRO 6000, it runs comfortably at
~24 GB for 1024→2048 upscale, leaving plenty of room for the LoRA
training stage that follows.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Callable, Optional


class UpscalerError(Exception):
    """Raised when upscaling fails."""


class ImageUpscaler:
    """
    Wraps SeedVR2 7B for image super-resolution via the ComfyUI standalone CLI.

    Args:
        cli_dir: Path to the cloned ComfyUI-SeedVR2 directory containing
                 ``inference_cli.py``.
        python_bin: Python binary to use. Defaults to the current interpreter.
    """

    def __init__(
        self,
        cli_dir: str,
        python_bin: str | None = None,
    ) -> None:
        self.cli_dir = os.path.abspath(cli_dir)
        self.cli_script = os.path.join(self.cli_dir, "inference_cli.py")
        self.python_bin = python_bin or sys.executable

        if not os.path.isfile(self.cli_script):
            raise UpscalerError(
                f"SeedVR2 CLI not found at {self.cli_script}. "
                "Clone it with: git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler"
            )

    def upscale_dataset(
        self,
        dataset_dir: str,
        target_resolution: int = 2048,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Upscale all images in dataset_dir in-place using SeedVR2 7B.

        Each image is upscaled to target_resolution (short side) and the
        original file is replaced with the upscaled version.

        Args:
            dataset_dir: Directory containing .png images.
            target_resolution: Target short-side resolution in pixels.
            progress_callback: Optional (current, total) callback.

        Raises:
            UpscalerError: If the CLI fails or produces no output.
        """
        if not os.path.isdir(dataset_dir):
            raise UpscalerError(f"Dataset directory not found: {dataset_dir}")

        image_files = sorted(
            f for f in os.listdir(dataset_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        )

        if not image_files:
            print("[Chimera] Upscaler: no images found to upscale.")
            return

        total = len(image_files)
        output_dir = os.path.join(dataset_dir, "_upscaled")
        os.makedirs(output_dir, exist_ok=True)

        print(f"[Chimera] Upscaler: upscaling {total} images to {target_resolution}px with SeedVR2 7B...")

        for i, filename in enumerate(image_files):
            input_path = os.path.join(dataset_dir, filename)

            cmd = [
                self.python_bin,
                self.cli_script,
                input_path,
                "--resolution", str(target_resolution),
                "--output_dir", output_dir,
                "--dit_model", "seedvr2_ema_7b_fp16.safetensors",
            ]

            print(f"[Chimera] Upscaler: [{i + 1}/{total}] {filename}...")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 min per image max
                    cwd=self.cli_dir,
                )
                if result.returncode != 0:
                    print(f"[Chimera] Upscaler: WARNING — failed on {filename}: {result.stderr[-500:]}")
                    continue
            except subprocess.TimeoutExpired:
                print(f"[Chimera] Upscaler: WARNING — timeout on {filename}, skipping.")
                continue

            # Find the upscaled output file
            upscaled = self._find_output(output_dir, filename)
            if upscaled:
                # Replace original with upscaled version
                shutil.move(upscaled, input_path)
            else:
                print(f"[Chimera] Upscaler: WARNING — no output for {filename}")

            if progress_callback:
                progress_callback(i + 1, total)

        # Clean up temp directory
        shutil.rmtree(output_dir, ignore_errors=True)

        print(f"[Chimera] Upscaler: done — {total} images upscaled to {target_resolution}px.")

    @staticmethod
    def _find_output(output_dir: str, original_name: str) -> str | None:
        """Find the upscaled output file corresponding to the original."""
        stem = os.path.splitext(original_name)[0]

        # Check for exact name match first, then any file containing the stem
        for f in os.listdir(output_dir):
            if not f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            if f.startswith(stem) or stem in f:
                return os.path.join(output_dir, f)

        # Fallback: if only one file in output dir, it's probably the result
        candidates = [
            f for f in os.listdir(output_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        if len(candidates) == 1:
            return os.path.join(output_dir, candidates[0])

        return None
