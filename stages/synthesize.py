"""
synthesize.py — Flux 2 DEV dataset synthesizer.

Uses the Flux2Pipeline from diffusers to generate training images from
reference images.  Designed for RunPod A40 (48 GB VRAM).

Memory budget at inference time:
  fp8 diffusion model  ~17 GB
  VAE                  ~0.5 GB
  text encoder         ~7 GB
  activations/buffers  ~10 GB
  ──────────────────────────
  total                ~34.5 GB  (comfortable on 48 GB A40)
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

try:
    from diffusers import Flux2Pipeline
except ImportError as _flux2_import_error:  # noqa: F841
    Flux2Pipeline = None  # type: ignore[assignment,misc]
    _FLUX2_UNAVAILABLE_MSG = (
        "[DatasetSynthesizer] Flux2Pipeline is not available in the installed "
        "version of diffusers.  Flux 2 DEV support was introduced in diffusers "
        ">=0.32.0.  Please upgrade:\n"
        "    pip install -U 'diffusers>=0.32.0'\n"
        "If you are on an older version that ships Flux 2 under a different "
        "class name, set DIFFUSERS_FLUX2_CLASS in your environment and patch "
        "this module accordingly."
    )
else:
    _FLUX2_UNAVAILABLE_MSG = ""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DatasetSynthesizer
# ---------------------------------------------------------------------------


class DatasetSynthesizer:
    """Generate a character training dataset using Flux 2 DEV locally.

    Parameters
    ----------
    model_path:
        Absolute path to ``flux2_dev_fp8mixed.safetensors``.
    vae_path:
        Absolute path to ``flux2-vae.safetensors``.
    text_enc_path:
        Absolute path to the Mistral Small 3 text encoder (safetensors).
    device:
        Torch device string.  Defaults to ``"cuda"``.
    """

    def __init__(
        self,
        model_path: str,
        vae_path: str,
        text_enc_path: str,
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.vae_path = vae_path
        self.text_enc_path = text_enc_path
        self.device = device
        self.pipe: Optional[Flux2Pipeline] = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the Flux 2 DEV pipeline in fp8 / bfloat16 mixed precision.

        Occupies ~35 GB VRAM on an A40.  Must be called before
        :meth:`generate_image` or :meth:`synthesize_dataset`.

        Raises
        ------
        RuntimeError
            If ``diffusers>=0.32.0`` is not installed.
        """
        if Flux2Pipeline is None:
            raise RuntimeError(_FLUX2_UNAVAILABLE_MSG)

        if self.pipe is not None:
            logger.debug("[DatasetSynthesizer] Pipeline already loaded — skipping.")
            return

        logger.info("[DatasetSynthesizer] Loading Flux 2 DEV pipeline from %s ...", self.model_path)

        self.pipe = Flux2Pipeline.from_single_file(
            self.model_path,
            vae=self.vae_path,
            text_encoder=self.text_enc_path,
            torch_dtype=torch.bfloat16,
        )
        self.pipe = self.pipe.to(self.device)

        # Enable attention slicing as a lightweight memory safety net.
        # On a 48 GB A40 this has negligible performance impact.
        self.pipe.enable_attention_slicing()

        logger.info("[DatasetSynthesizer] Pipeline ready on %s.", self.device)

    def unload_model(self) -> None:
        """Delete the pipeline and release all VRAM for the training stage."""
        if self.pipe is not None:
            logger.info("[DatasetSynthesizer] Unloading Flux 2 DEV pipeline...")
            del self.pipe
            self.pipe = None

        torch.cuda.empty_cache()
        gc.collect()
        logger.info("[DatasetSynthesizer] VRAM released.")

    # ------------------------------------------------------------------
    # Single-image generation
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt: str,
        reference_images: list[Image.Image],
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 2.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate a single image using Flux 2 DEV with reference images.

        Flux 2 DEV accepts up to 10 reference images via sequential token
        concatenation.  They are passed as a list to the ``image`` parameter
        of the pipeline.  Prompts should reference them as
        ``"the character from image 1"``, ``"image 2"``, etc.

        Parameters
        ----------
        prompt:
            Text prompt.  Should reference images as ``"image 1"`` etc.
        reference_images:
            List of PIL Images.  Up to 10 supported by Flux 2 DEV natively.
        width, height:
            Output resolution in pixels.  Defaults to 1024×1024.
        guidance_scale:
            CFG scale.  2.5 is recommended for image-conditioned Flux 2 DEV
            (lower than the text-only default of 4.0).
        num_inference_steps:
            Denoising steps.  50 gives a good quality/speed balance.
        seed:
            Optional integer seed for reproducibility.

        Returns
        -------
        PIL.Image.Image
            The generated image.

        Raises
        ------
        RuntimeError
            If the pipeline has not been loaded via :meth:`load_model`.
        ValueError
            If ``reference_images`` is empty or contains more than 10 images.
        """
        if self.pipe is None:
            raise RuntimeError(
                "[DatasetSynthesizer] Pipeline is not loaded. "
                "Call load_model() before generate_image()."
            )

        if not reference_images:
            raise ValueError("[DatasetSynthesizer] reference_images must not be empty.")

        if len(reference_images) > 10:
            raise ValueError(
                f"[DatasetSynthesizer] Flux 2 DEV supports at most 10 reference images; "
                f"got {len(reference_images)}."
            )

        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=reference_images,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        return result

    # ------------------------------------------------------------------
    # Full dataset synthesis
    # ------------------------------------------------------------------

    def synthesize_dataset(
        self,
        reference_images: list[Image.Image],
        output_dir: str,
        num_images: int = 25,
        start_from: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[str]:
        """Generate a full training dataset and save it to ``output_dir``.

        Images are saved as ``img_001.png``, ``img_002.png``, … and use
        deterministic seeds (``42 + i``) so a run can be resumed exactly
        from any point after a pod restart.

        Parameters
        ----------
        reference_images:
            Source images for conditioning.  Up to 10 supported.
        output_dir:
            Directory where generated images are written.  Created if it
            does not exist.
        num_images:
            Total number of images to produce.  Defaults to 25.
        start_from:
            Resume index (0-based).  Images ``0`` … ``start_from - 1`` are
            assumed to already exist and are skipped.
        progress_callback:
            Optional callable ``(current: int, total: int) -> None`` invoked
            after each image is saved.

        Returns
        -------
        list[str]
            Absolute paths of all images saved in this run (not including
            previously completed images from a resumed run).

        Raises
        ------
        RuntimeError
            If the pipeline has not been loaded.
        ValueError
            If ``start_from`` is out of range.
        """
        if self.pipe is None:
            raise RuntimeError(
                "[DatasetSynthesizer] Pipeline is not loaded. "
                "Call load_model() before synthesize_dataset()."
            )

        if start_from < 0 or start_from >= num_images:
            raise ValueError(
                f"[DatasetSynthesizer] start_from={start_from} is out of range "
                f"for num_images={num_images}."
            )

        from ..utils.prompt_templates import get_prompt_templates

        templates = get_prompt_templates(num_images)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths: list[str] = []
        remaining = num_images - start_from

        logger.info(
            "[DatasetSynthesizer] Synthesizing %d image(s) (indices %d–%d) into %s ...",
            remaining,
            start_from,
            num_images - 1,
            output_dir,
        )

        for i in range(start_from, num_images):
            prompt = templates[i]
            seed = 42 + i  # deterministic per slot for reproducibility

            logger.debug("[DatasetSynthesizer] Generating image %d/%d — seed=%d", i + 1, num_images, seed)

            image = self.generate_image(
                prompt=prompt,
                reference_images=reference_images,
                seed=seed,
            )

            filename = f"img_{i + 1:03d}.png"
            filepath = output_path / filename
            image.save(filepath)
            saved_paths.append(str(filepath))

            logger.debug("[DatasetSynthesizer] Saved %s", filepath)

            if progress_callback is not None:
                progress_callback(i + 1, num_images)

        logger.info("[DatasetSynthesizer] Done — %d image(s) saved.", len(saved_paths))
        return saved_paths
