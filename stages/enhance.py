"""
enhance.py — Dataset enhancer for Chimera.

Uses FLUX.1-dev img2img with a character LoRA (from first-pass training) and
an optional SRPO LoRA for photorealism to add realistic detail to synthetic
training images while preserving composition and identity.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Supported image extensions scanned by enhance_dataset().
_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")

# Subdirectories that are skipped during dataset scanning.
_SKIP_SUBDIRS: frozenset[str] = frozenset({"_originals", "_pre_enhance"})


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class DatasetEnhancerError(Exception):
    """Raised when dataset enhancement fails."""


# ---------------------------------------------------------------------------
# DatasetEnhancer
# ---------------------------------------------------------------------------


class DatasetEnhancer:
    """
    Enhances synthetic training images using FLUX.1-dev img2img pipeline
    with a character LoRA (from first-pass training) and optional SRPO LoRA
    for photorealism.

    The enhancer preserves the original composition/pose while adding:
    - Consistent character identity (locked by character LoRA)
    - Realistic skin texture, hair detail, fabric wrinkles
    - Natural lighting interaction

    Load the model with :meth:`load_model` before calling any enhancement
    methods, and release VRAM when done with :meth:`unload_model`.

    Parameters
    ----------
    hf_token:
        HuggingFace access token.  Required because ``black-forest-labs/FLUX.1-dev``
        is a gated repository.
    device:
        Torch device string.  Defaults to ``"cuda"``.
    """

    REPO_ID: str = "black-forest-labs/FLUX.1-dev"

    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        self.hf_token = hf_token
        self.device = device
        self.pipe = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(
        self,
        character_lora_path: str,
        srpo_lora_path: Optional[str] = None,
        lora_weight: float = 0.75,
        srpo_weight: float = 0.5,
    ) -> None:
        """Load the FLUX.1-dev img2img pipeline with LoRA adapters.

        Uses ``enable_model_cpu_offload()`` to keep peak VRAM usage
        manageable, and ``vae.enable_tiling()`` for safe 2048px inference.

        Character LoRA is always loaded.  SRPO LoRA is optional — pass
        ``srpo_lora_path`` to stack both adapters for photorealism.

        Parameters
        ----------
        character_lora_path:
            Path to the character ``.safetensors`` LoRA checkpoint from
            first-pass training.
        srpo_lora_path:
            Optional path to an SRPO photorealism LoRA file.
        lora_weight:
            Adapter weight for the character LoRA.  Defaults to 0.75.
        srpo_weight:
            Adapter weight for the SRPO LoRA.  Defaults to 0.8.

        Raises
        ------
        DatasetEnhancerError
            If ``FluxImg2ImgPipeline`` is not available or the character
            LoRA file cannot be found.
        RuntimeError
            Re-raised from diffusers if the pipeline cannot be loaded.
        """
        if self.pipe is not None:
            logger.debug("[DatasetEnhancer] Pipeline already loaded — skipping.")
            return

        if not os.path.isfile(character_lora_path):
            raise DatasetEnhancerError(
                f"Character LoRA file not found: {character_lora_path!r}"
            )

        if srpo_lora_path is not None and not os.path.isfile(srpo_lora_path):
            raise DatasetEnhancerError(
                f"SRPO LoRA file not found: {srpo_lora_path!r}"
            )

        try:
            from diffusers import FluxImg2ImgPipeline
        except ImportError as exc:
            raise DatasetEnhancerError(
                "[DatasetEnhancer] FluxImg2ImgPipeline is not available in the "
                "installed version of diffusers.  Ensure diffusers>=0.30.0:\n"
                "    pip install -U 'diffusers>=0.30.0'\n"
            ) from exc

        print(
            f"[Chimera] DatasetEnhancer: loading FLUX.1-dev img2img pipeline "
            f"from {self.REPO_ID} ..."
        )

        # Ensure HF_TOKEN env var is set — some HF hub code paths check
        # the env var rather than the token= parameter.
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token

        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            self.REPO_ID,
            torch_dtype=torch.bfloat16,
            token=self.hf_token,
        )

        # CPU offloading moves each module to GPU only when needed.
        self.pipe.enable_model_cpu_offload()

        # VAE tiling prevents OOM on 2048px images.
        self.pipe.vae.enable_tiling()

        # Load character LoRA.
        print(
            f"[Chimera] DatasetEnhancer: loading character LoRA from "
            f"{character_lora_path} ..."
        )
        self.pipe.load_lora_weights(character_lora_path, adapter_name="character")

        adapters: list[str] = ["character"]
        weights: list[float] = [lora_weight]

        # Optionally stack SRPO LoRA.
        if srpo_lora_path is not None:
            print(
                f"[Chimera] DatasetEnhancer: loading SRPO LoRA from "
                f"{srpo_lora_path} ..."
            )
            self.pipe.load_lora_weights(srpo_lora_path, adapter_name="srpo")
            adapters.append("srpo")
            weights.append(srpo_weight)

        self.pipe.set_adapters(adapters, adapter_weights=weights)

        logger.info(
            "[DatasetEnhancer] Pipeline ready — adapters=%s weights=%s",
            adapters,
            weights,
        )
        print(
            f"[Chimera] DatasetEnhancer: pipeline ready "
            f"(adapters={adapters}, weights={weights})."
        )

    def unload_model(self) -> None:
        """Delete the pipeline and release all VRAM.

        Safe to call when the model is not loaded — it is a no-op in
        that case.
        """
        if self.pipe is not None:
            logger.info("[DatasetEnhancer] Unloading FLUX.1-dev img2img pipeline...")
            del self.pipe
            self.pipe = None

        torch.cuda.empty_cache()
        gc.collect()
        print("[Chimera] DatasetEnhancer: model unloaded.")

    # ------------------------------------------------------------------
    # Dataset enhancement
    # ------------------------------------------------------------------

    def enhance_dataset(
        self,
        dataset_dir: str,
        strength: float = 0.40,
        inference_steps: int = 28,
        guidance_scale: float = 3.5,
        base_seed: int = 42,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        image_callback: Optional[Callable[[int, str, str], None]] = None,
    ) -> list[str]:
        """Enhance all images in *dataset_dir* in-place using img2img.

        For each image:

        1. Reads the matching ``.txt`` caption file as the prompt.  Falls back
           to a generic prompt if no caption exists.
        2. Copies the original to ``dataset_dir/_pre_enhance/`` (idempotent
           resume: if the copy already exists the image is considered
           already-enhanced and skipped).
        3. Runs FLUX.1-dev img2img with the character LoRA.
        4. Overwrites the original path with the enhanced result.

        Files inside ``_originals/`` and ``_pre_enhance/`` subdirectories are
        always skipped.

        Parameters
        ----------
        dataset_dir:
            Directory containing ``.png``/``.jpg``/``.jpeg``/``.webp`` images.
        strength:
            img2img noise strength (0–1).  Lower preserves more of the
            original.  Defaults to 0.40.
        inference_steps:
            Total denoising steps for the img2img pass.  Defaults to 28.
        guidance_scale:
            CFG guidance scale.  Defaults to 3.5.
        base_seed:
            Starting seed; each image uses ``base_seed + index`` for
            deterministic, resumable generation.  Defaults to 42.
        progress_callback:
            Optional ``(current: int, total: int)`` callable fired after
            each image is enhanced.
        image_callback:
            Optional ``(index: int, pre_enhance_path: str, enhanced_path: str)``
            callable fired after each image is saved.  Paths are relative
            to *dataset_dir*.

        Returns
        -------
        list[str]
            Absolute paths of the enhanced images (images that were skipped
            due to resume are not included).

        Raises
        ------
        DatasetEnhancerError
            If the pipeline is not loaded or *dataset_dir* does not exist.
        FileNotFoundError
            If *dataset_dir* does not exist on disk.
        """
        if self.pipe is None:
            raise DatasetEnhancerError(
                "[DatasetEnhancer] Pipeline is not loaded. "
                "Call load_model() before enhance_dataset()."
            )

        dataset_path = Path(dataset_dir)
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")

        pre_enhance_dir = dataset_path / "_pre_enhance"
        pre_enhance_dir.mkdir(parents=True, exist_ok=True)

        # Collect images — only top-level files, skip reserved subdirectories.
        image_files: list[Path] = sorted(
            p
            for p in dataset_path.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            and p.parent.name not in _SKIP_SUBDIRS
        )

        total_images = len(image_files)
        if total_images == 0:
            print(
                f"[Chimera] DatasetEnhancer: no images found in {dataset_dir}."
            )
            return []

        logger.info(
            "[DatasetEnhancer] Enhancing %d image(s) in %s ...",
            total_images,
            dataset_dir,
        )
        print(
            f"[Chimera] DatasetEnhancer: enhancing {total_images} image(s) "
            f"in {dataset_dir} ..."
        )

        enhanced_paths: list[str] = []

        for index, image_path in enumerate(image_files):
            pre_enhance_copy = pre_enhance_dir / image_path.name

            # Resume support: if the pre-enhance copy already exists, the
            # image was previously enhanced — skip it.
            if pre_enhance_copy.exists():
                logger.debug(
                    "[DatasetEnhancer] Skipping %s (already enhanced).",
                    image_path.name,
                )
                print(
                    f"[Chimera] DatasetEnhancer: [{index + 1}/{total_images}] "
                    f"skip {image_path.name} (already enhanced)."
                )
                continue

            # Read caption from matching .txt file.
            caption = self._read_caption(image_path)

            logger.debug(
                "[DatasetEnhancer] Enhancing %d/%d — %s",
                index + 1,
                total_images,
                image_path.name,
            )
            print(
                f"[Chimera] DatasetEnhancer: [{index + 1}/{total_images}] "
                f"enhancing {image_path.name} ..."
            )

            # Preserve original before overwriting.
            shutil.copy2(image_path, pre_enhance_copy)

            # Load image and run enhancement.
            with Image.open(image_path) as pil_image:
                pil_image = pil_image.convert("RGB")
                enhanced = self.enhance_single(
                    image_path=str(image_path),
                    caption=caption,
                    strength=strength,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    seed=base_seed + index,
                    _pil_image=pil_image,
                )

            # Overwrite the original with the enhanced result.
            enhanced.save(str(image_path))
            enhanced_paths.append(str(image_path))

            logger.debug("[DatasetEnhancer] Saved enhanced image to %s", image_path)

            # Fire callbacks with paths relative to dataset_dir.
            if image_callback is not None:
                try:
                    rel_pre = str(pre_enhance_copy.relative_to(dataset_path))
                    rel_enhanced = str(image_path.relative_to(dataset_path))
                    image_callback(index, rel_pre, rel_enhanced)
                except Exception as exc:
                    logger.warning("image_callback failed: %s", exc)

            if progress_callback is not None:
                try:
                    progress_callback(index + 1, total_images)
                except Exception as exc:
                    logger.warning("progress_callback failed: %s", exc)

        logger.info(
            "[DatasetEnhancer] Done — %d image(s) enhanced.", len(enhanced_paths)
        )
        print(
            f"[Chimera] DatasetEnhancer: done — {len(enhanced_paths)} image(s) enhanced."
        )
        return enhanced_paths

    # ------------------------------------------------------------------
    # Single-image enhancement
    # ------------------------------------------------------------------

    def enhance_single(
        self,
        image_path: str,
        caption: str,
        strength: float,
        inference_steps: int,
        guidance_scale: float,
        seed: int,
        _pil_image: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Enhance a single image and return the result as a PIL Image.

        Used for preview mode (enhance first 3, pause for approval) and
        internally by :meth:`enhance_dataset`.

        Parameters
        ----------
        image_path:
            Absolute path to the source image.  Ignored when ``_pil_image``
            is provided (internal fast-path from :meth:`enhance_dataset`).
        caption:
            Text prompt for the img2img pass.
        strength:
            img2img noise strength (0–1).
        inference_steps:
            Total denoising steps.
        guidance_scale:
            CFG guidance scale.
        seed:
            Integer seed for reproducibility.
        _pil_image:
            Pre-loaded PIL image (optional internal fast-path, avoids a
            second open/close cycle when called from :meth:`enhance_dataset`).

        Returns
        -------
        PIL.Image.Image
            The enhanced image at the original resolution.

        Raises
        ------
        DatasetEnhancerError
            If the pipeline is not loaded.
        """
        if self.pipe is None:
            raise DatasetEnhancerError(
                "[DatasetEnhancer] Pipeline is not loaded. "
                "Call load_model() before enhance_single()."
            )

        if _pil_image is not None:
            pil_image = _pil_image
        else:
            pil_image = Image.open(image_path).convert("RGB")

        # Generator must be on CPU when enable_model_cpu_offload() is active.
        generator = torch.Generator("cpu").manual_seed(seed)

        result = self.pipe(
            prompt=caption,
            image=pil_image,
            strength=strength,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=512,
        ).images[0]

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_caption(image_path: Path) -> str:
        """Return the caption from the matching .txt sidecar file.

        Falls back to a generic prompt using the image stem as the trigger
        word if no caption file exists.

        Parameters
        ----------
        image_path:
            Path to the image whose caption should be read.

        Returns
        -------
        str
            Caption string (stripped of leading/trailing whitespace).
        """
        txt_path = image_path.with_suffix(".txt")
        if txt_path.is_file():
            try:
                return txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        # Fallback: use the image stem as a crude trigger word.
        trigger = image_path.stem.split("_")[0] if "_" in image_path.stem else image_path.stem
        return (
            f"{trigger} person, photorealistic, detailed skin texture, "
            f"natural lighting, high quality photograph"
        )
