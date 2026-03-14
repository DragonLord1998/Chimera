"""
synthesize.py — Flux 2 DEV dataset synthesizer.

Uses Flux2Pipeline from diffusers to generate training images from
reference images.  Designed for RunPod RTX PRO 6000 (96 GB VRAM).

The pipeline is loaded via ``from_pretrained`` from a local diffusers-format
directory and uses ``enable_model_cpu_offload()`` to manage VRAM.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

# Fix transformers 5.x bug: tf/tensorflow_text backends referenced but not
# in BACKENDS_MAPPING.  Patch before importing diffusers to avoid crash.
try:
    from transformers.utils import import_utils as _tiu
    for _be in ("tf", "tensorflow_text"):
        if _be not in _tiu.BACKENDS_MAPPING:
            _tiu.BACKENDS_MAPPING[_be] = (lambda: False, f"{_be} is not installed.")
except Exception:
    pass

try:
    from diffusers import Flux2Pipeline
except ImportError as _flux2_import_error:  # noqa: F841
    Flux2Pipeline = None  # type: ignore[assignment,misc]
    _FLUX2_UNAVAILABLE_MSG = (
        "[DatasetSynthesizer] Flux2Pipeline is not available in the installed "
        "version of diffusers.  Flux 2 DEV support was introduced in diffusers "
        ">=0.32.0.  Please upgrade:\n"
        "    pip install -U 'diffusers>=0.32.0'\n"
    )
else:
    _FLUX2_UNAVAILABLE_MSG = ""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Latent preview helper
# ---------------------------------------------------------------------------


def _latents_to_preview(latents: torch.Tensor, size: int = 256) -> Image.Image:
    """Approximate RGB preview from latents without full VAE decode.

    Works with both packed (B, seq_len, C) and spatial (B, C, H, W) latents.
    The result is a rough, progressively-sharpening preview — intentionally
    lo-fi but near-zero cost compared to a real VAE decode.
    """
    try:
        if latents.dim() == 3:
            # Packed format (B, seq_len, channels) — common in Flux
            b, seq_len, c = latents.shape
            h = w = int(seq_len ** 0.5)
            if h * w < seq_len:
                h += 1
            usable = min(h * w, seq_len)
            lat = latents[0, :usable, :3].float().cpu()
            lat = lat.reshape(min(h, int(usable ** 0.5) + 1), -1, 3)[:h, :w, :]
        elif latents.dim() == 4:
            # Spatial format (B, C, H, W)
            lat = latents[0, :3].float().cpu().permute(1, 2, 0)
        else:
            return Image.new("RGB", (size, size), (40, 40, 60))

        # Normalize to 0-1
        vmin, vmax = lat.min(), lat.max()
        if (vmax - vmin) > 1e-8:
            lat = (lat - vmin) / (vmax - vmin)
        else:
            lat = torch.full_like(lat, 0.5)

        rgb_np = (lat.numpy() * 255).clip(0, 255).astype(np.uint8)
        preview = Image.fromarray(rgb_np)
        return preview.resize((size, size), Image.LANCZOS)

    except Exception:
        return Image.new("RGB", (size, size), (40, 40, 60))


# ---------------------------------------------------------------------------
# DatasetSynthesizer
# ---------------------------------------------------------------------------


class DatasetSynthesizer:
    """Generate a character training dataset using Flux 2 DEV locally.

    Parameters
    ----------
    hf_token:
        HuggingFace access token.  Required because ``black-forest-labs/FLUX.2-dev``
        is a gated repository.
    device:
        Torch device string.  Defaults to ``"cuda"``.
    """

    REPO_ID: str = "black-forest-labs/FLUX.2-dev"

    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        self.hf_token = hf_token
        self.device = device
        self.pipe: Optional[Flux2Pipeline] = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the Flux 2 DEV pipeline via ``from_pretrained``.

        Uses ``enable_model_cpu_offload()`` to keep peak VRAM usage
        manageable on an RTX PRO 6000 (96 GB).

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

        logger.info("[DatasetSynthesizer] Loading Flux 2 DEV pipeline from %s ...", self.REPO_ID)

        self.pipe = Flux2Pipeline.from_pretrained(
            self.REPO_ID,
            torch_dtype=torch.bfloat16,
            token=self.hf_token,
        )
        # CPU offloading moves each module to GPU only when needed,
        # keeping peak VRAM manageable for the 32B param model.
        self.pipe.enable_model_cpu_offload()

        logger.info("[DatasetSynthesizer] Pipeline ready (cpu offload enabled).")

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
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        preview_callback: Optional[Callable[[int, int, Image.Image], None]] = None,
        preview_every: int = 2,
    ) -> Image.Image:
        """Generate a single image using Flux 2 DEV with reference images.

        Flux 2 DEV accepts up to 10 reference images via the ``image``
        parameter of the pipeline call.

        Parameters
        ----------
        prompt:
            Text prompt.  Should reference images as ``"image 1"`` etc.
        reference_images:
            List of PIL Images.  Up to 10 supported by Flux 2 DEV natively.
        width, height:
            Output resolution in pixels.  Defaults to 1024x1024.
        guidance_scale:
            CFG scale.  5.0 for stronger reference adherence.
        num_inference_steps:
            Denoising steps.  50 gives a good quality/speed balance.
        seed:
            Optional integer seed for reproducibility.
        preview_callback:
            Optional ``(step, total_steps, preview_image)`` callable.
            Called every ``preview_every`` steps with an approximate RGB
            preview decoded from the latents (near-zero overhead).
        preview_every:
            Emit a latent preview every N steps.  Defaults to 2.

        Returns
        -------
        PIL.Image.Image
            The generated image.
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
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # Build step callback for latent previews
        step_callback = None
        if preview_callback is not None:
            def step_callback(pipe, step_index, timestep, callback_kwargs):
                if (step_index + 1) % preview_every == 0:
                    try:
                        latents = callback_kwargs.get("latents")
                        if latents is not None:
                            preview = _latents_to_preview(latents)
                            preview_callback(step_index + 1, num_inference_steps, preview)
                    except Exception:
                        pass  # Preview failure must never block generation
                return callback_kwargs

        pipe_kwargs: dict = dict(
            prompt=prompt,
            image=reference_images,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        if step_callback is not None:
            pipe_kwargs["callback_on_step_end"] = step_callback
            pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        result = self.pipe(**pipe_kwargs).images[0]

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
        num_inference_steps: int = 50,
        preview_callback: Optional[Callable[[int, int, int, Image.Image], None]] = None,
    ) -> list[str]:
        """Generate a full training dataset and save it to ``output_dir``.

        Images are saved as ``img_001.png``, ``img_002.png``, ... and use
        deterministic seeds (``42 + i``) so a run can be resumed exactly
        from any point after a pod restart.

        Parameters
        ----------
        reference_images:
            Source images for conditioning.  Up to 10 supported.
        output_dir:
            Directory where generated images are written.
        num_images:
            Total number of images to produce.  Defaults to 25.
        start_from:
            Resume index (0-based).
        progress_callback:
            Optional ``(current: int, total: int)`` callable.
        preview_callback:
            Optional ``(image_index, step, total_steps, preview_image)``
            callable.  Fires every 2 denoising steps with a cheap latent
            preview so the UI can show the image forming in real time.

        Returns
        -------
        list[str]
            Absolute paths of images saved in this run.
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

        from utils.prompt_templates import get_prompt_templates

        templates = get_prompt_templates(num_images)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths: list[str] = []
        remaining = num_images - start_from

        logger.info(
            "[DatasetSynthesizer] Synthesizing %d image(s) (indices %d-%d) into %s ...",
            remaining,
            start_from,
            num_images - 1,
            output_dir,
        )

        for i in range(start_from, num_images):
            prompt = templates[i]
            seed = 42 + i  # deterministic per slot for reproducibility

            logger.debug("[DatasetSynthesizer] Generating image %d/%d — seed=%d", i + 1, num_images, seed)

            # Per-image preview callback closure
            img_preview_cb = None
            if preview_callback is not None:
                def img_preview_cb(step, total, preview, _idx=i):
                    preview_callback(_idx, step, total, preview)

            image = self.generate_image(
                prompt=prompt,
                reference_images=reference_images,
                seed=seed,
                num_inference_steps=num_inference_steps,
                preview_callback=img_preview_cb,
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
