"""
CaptionGenerator — uses Florence 2 Large to auto-caption images and produce
LoRA-ready training captions by stripping identity traits and prepending the
trigger word.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from PIL import Image

from utils.identity_stripper import IdentityStripper


class CaptionGeneratorError(Exception):
    """Raised when captioning fails or the model is not loaded."""


def _patch_tokenizer_backend() -> None:
    """Fix transformers 5.x: TokenizersBackend missing additional_special_tokens.

    Florence 2's custom processor code accesses tokenizer.additional_special_tokens,
    which was available in transformers 4.x but moved/removed in 5.x's backend
    refactor.  This patches any Backend class that's missing the attribute.
    """
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if not mod or "transformers" not in mod_name:
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if not isinstance(obj, type):
                continue
            if "Backend" in attr_name and not hasattr(obj, "additional_special_tokens"):
                obj.additional_special_tokens = property(
                    lambda self: getattr(self, "_additional_special_tokens", []),
                    lambda self, v: setattr(self, "_additional_special_tokens", v),
                )


class CaptionGenerator:
    """
    Wraps Florence 2 Large for detailed image captioning.

    Load the model with :meth:`load_model` before calling any captioning
    methods, and release VRAM when done with :meth:`unload_model`.

    Args:
        model_path: Local directory containing the Florence 2 Large model weights
                    and processor config (e.g. a Hugging Face snapshot download).
        device: PyTorch device string.  Defaults to ``"cuda"``.
    """

    # Florence 2 task token for detailed captioning.
    CAPTION_TASK: str = "<MORE_DETAILED_CAPTION>"

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self.model_path = model_path
        self.device = device
        self.model: Optional[object] = None
        self.processor: Optional[object] = None
        self._stripper = IdentityStripper()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load Florence 2 Large into VRAM (or CPU if device is ``"cpu"``).

        Uses ``AutoModelForCausalLM`` and ``AutoProcessor`` from the
        ``transformers`` library.  Safe to call multiple times — subsequent
        calls are no-ops if the model is already loaded.

        Raises:
            CaptionGeneratorError: If the model cannot be loaded from
                :attr:`model_path`.
        """
        if self.model is not None:
            return  # already loaded

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore[import]

            print(f"[Chimera] CaptionGenerator: loading Florence 2 from {self.model_path}...")

            # Load processor — retry with tokenizer backend patch on transformers 5.x
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
            except AttributeError as attr_err:
                if "additional_special_tokens" in str(attr_err):
                    print("[Chimera] Patching tokenizer backend for transformers 5.x...")
                    _patch_tokenizer_backend()
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                    )
                else:
                    raise

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()  # type: ignore[union-attr]

            print("[Chimera] CaptionGenerator: Florence 2 loaded.")
        except CaptionGeneratorError:
            raise
        except Exception as exc:
            raise CaptionGeneratorError(
                f"Failed to load Florence 2 from '{self.model_path}': {exc}"
            ) from exc

    def unload_model(self) -> None:
        """
        Unload the model and processor from VRAM and free GPU memory.

        Safe to call when the model is not loaded — it is a no-op in that case.
        """
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[Chimera] CaptionGenerator: model unloaded.")

    # ------------------------------------------------------------------
    # Captioning
    # ------------------------------------------------------------------

    def caption_image(self, image: Image.Image) -> str:
        """
        Generate a detailed caption for a single image using Florence 2's
        ``<MORE_DETAILED_CAPTION>`` task.

        Args:
            image: PIL image to caption.  Will be converted to RGB if necessary.

        Returns:
            Raw caption string produced by Florence 2.

        Raises:
            CaptionGeneratorError: If the model is not loaded or inference fails.
        """
        self._require_model()

        image = image.convert("RGB")

        try:
            inputs = self.processor(  # type: ignore[call-arg]
                text=self.CAPTION_TASK,
                images=image,
                return_tensors="pt",
            )
            # Move all tensor inputs to the target device.
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(  # type: ignore[union-attr]
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )

            generated_text = self.processor.batch_decode(  # type: ignore[union-attr]
                generated_ids, skip_special_tokens=False
            )[0]

            # Florence 2 wraps output in task tokens; post-process them off.
            parsed = self.processor.post_process_generation(  # type: ignore[union-attr]
                generated_text,
                task=self.CAPTION_TASK,
                image_size=(image.width, image.height),
            )
            caption: str = parsed.get(self.CAPTION_TASK, "").strip()
            return caption

        except Exception as exc:
            raise CaptionGeneratorError(f"Florence 2 inference failed: {exc}") from exc

    def caption_and_clean(self, image: Image.Image, trigger_word: str) -> str:
        """
        Caption an image and return a LoRA-ready training caption.

        Steps:
        1. Generate a raw detailed caption via Florence 2.
        2. Strip identity traits using :class:`~utils.identity_stripper.IdentityStripper`.
        3. Prepend the trigger word.

        Args:
            image: PIL image to caption.
            trigger_word: Token to prepend (e.g. ``"ohwx person"``).

        Returns:
            Cleaned caption with trigger word prepended.

        Raises:
            CaptionGeneratorError: If the model is not loaded or inference fails.
        """
        raw_caption = self.caption_image(image)
        return self._stripper.process(raw_caption, trigger_word)

    def caption_dataset(self, image_dir: str, trigger_word: str) -> list[str]:
        """
        Caption all ``.png`` images in *image_dir* and write sidecar ``.txt`` files.

        For each ``<name>.png`` a ``<name>.txt`` file is written alongside it
        containing the cleaned, trigger-prepended caption.

        Args:
            image_dir: Directory containing ``.png`` images to caption.
            trigger_word: Trigger token prepended to every caption.

        Returns:
            List of absolute paths to the written ``.txt`` caption files, in
            the same order as the source images were discovered.

        Raises:
            CaptionGeneratorError: If the model is not loaded or any image
                cannot be captioned.
            FileNotFoundError: If *image_dir* does not exist.
        """
        self._require_model()

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

        png_files = sorted(
            f for f in os.listdir(image_dir) if f.lower().endswith(".png")
        )

        if not png_files:
            print(f"[Chimera] CaptionGenerator: no .png files found in {image_dir}.")
            return []

        caption_paths: list[str] = []

        for filename in png_files:
            img_path = os.path.join(image_dir, filename)
            txt_path = os.path.join(image_dir, os.path.splitext(filename)[0] + ".txt")

            print(f"[Chimera] CaptionGenerator: captioning {filename}...")

            with Image.open(img_path) as img:
                caption = self.caption_and_clean(img, trigger_word)

            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(caption)

            print(f"[Chimera] CaptionGenerator: wrote {txt_path}")
            caption_paths.append(txt_path)

        return caption_paths

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_model(self) -> None:
        """Raise if the model has not been loaded yet."""
        if self.model is None or self.processor is None:
            raise CaptionGeneratorError(
                "Model is not loaded. Call load_model() before captioning."
            )
