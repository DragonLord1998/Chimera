"""
MultiViewGenerator — sends a character image to the Google Gemini API
(gemini-3-pro-image-preview, "Nano Banana Pro") and generates 10 diverse
views using a snowball reference strategy.

Each generation feeds previously generated images as references (up to 5
references per call), progressively building consistency across all views.

Generation order:
  1. front_face_closeup       — Face identity anchor
  2. front_midbody            — Body proportions anchor
  3. front_fullbody           — Full silhouette anchor
  4. left_34_midbody          — First angle variation
  5. right_profile_closeup    — Side face definition
  6. right_34_midbody         — Right side + expression
  7. left_fullbody_walking    — Dynamic pose
  8. front_midbody_laughing   — Expression range
  9. rear_34_midbody          — Back of head/hair
  10. left_34_closeup_dramatic — Dramatic lighting
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from google import genai
from google.genai import types
from PIL import Image


class MultiViewGeneratorError(Exception):
    """Raised when a Gemini API call fails after all retries."""


@dataclass
class _ViewSpec:
    """Specification for a single view to generate."""

    name: str
    prompt: str
    crop: str    # "face_closeup" | "midbody" | "fullbody"
    angle: str   # "front" | "left_34" | "right_profile" | "right_34" | "left" | "rear_34"


_VIEW_SPECS: List[_ViewSpec] = [
    _ViewSpec(
        name="front_face_closeup",
        prompt=(
            "Generate a face close-up portrait photograph of the exact same person shown "
            "in the reference images. Front-facing, neutral expression, plain white studio "
            "background. Photorealistic, high detail skin texture, sharp focus on facial "
            "features, eyes open, relaxed mouth."
        ),
        crop="face_closeup",
        angle="front",
    ),
    _ViewSpec(
        name="front_midbody",
        prompt=(
            "Generate a mid-body (waist-up) photograph of the exact same person shown in "
            "the reference images. Front-facing, neutral expression, plain white studio "
            "background. Both shoulders visible, arms at sides, photorealistic high quality."
        ),
        crop="midbody",
        angle="front",
    ),
    _ViewSpec(
        name="front_fullbody",
        prompt=(
            "Generate a full body photograph of the exact same person shown in the reference "
            "images. Front-facing, standing relaxed with arms loosely at sides, plain white "
            "studio background. Head to feet fully visible, photorealistic high quality."
        ),
        crop="fullbody",
        angle="front",
    ),
    _ViewSpec(
        name="left_34_midbody",
        prompt=(
            "Generate a mid-body (waist-up) photograph of the exact same person shown in "
            "the reference images. Three-quarter view from the left side, slight smile, "
            "soft natural light, plain white or light background. Same clothing and features "
            "as in references, photorealistic high quality."
        ),
        crop="midbody",
        angle="left_34",
    ),
    _ViewSpec(
        name="right_profile_closeup",
        prompt=(
            "Generate a face close-up portrait photograph of the exact same person shown "
            "in the reference images. Pure right side profile (nose pointing left of frame), "
            "serious expression, studio side-lit with white background. Sharp focus on "
            "facial features, ear and right side of face fully visible, photorealistic."
        ),
        crop="face_closeup",
        angle="right_profile",
    ),
    _ViewSpec(
        name="right_34_midbody",
        prompt=(
            "Generate a mid-body (waist-up) photograph of the exact same person shown in "
            "the reference images. Three-quarter view from the right side, casual hand "
            "gesture (one hand slightly raised or gesturing naturally), warm outdoor light, "
            "simple background. Same person, same clothing, photorealistic high quality."
        ),
        crop="midbody",
        angle="right_34",
    ),
    _ViewSpec(
        name="left_fullbody_walking",
        prompt=(
            "Generate a full body photograph of the exact same person shown in the reference "
            "images. Left side view, mid-stride walking pose, overcast daylight outdoor "
            "lighting, neutral simple background. Full body head to feet visible, natural "
            "motion, same clothing and features, photorealistic high quality."
        ),
        crop="fullbody",
        angle="left",
    ),
    _ViewSpec(
        name="front_midbody_laughing",
        prompt=(
            "Generate a mid-body (waist-up) photograph of the exact same person shown in "
            "the reference images. Front-facing, laughing with genuine joy, warm window "
            "light from one side, light background. Same clothing and features as references, "
            "photorealistic high quality, natural expression."
        ),
        crop="midbody",
        angle="front",
    ),
    _ViewSpec(
        name="rear_34_midbody",
        prompt=(
            "Generate a mid-body (waist-up) photograph of the exact same person shown in "
            "the reference images. Three-quarter rear view, person is looking back over "
            "their shoulder toward the camera, neutral or simple background. Back of head "
            "and hair clearly visible, same clothing as references, photorealistic high quality."
        ),
        crop="midbody",
        angle="rear_34",
    ),
    _ViewSpec(
        name="left_34_closeup_dramatic",
        prompt=(
            "Generate a face close-up portrait photograph of the exact same person shown "
            "in the reference images. Three-quarter view from the left side, contemplative "
            "expression with eyes slightly downcast, dramatic side lighting with deep shadows "
            "on one side of the face, dark or moody background. Same facial features as "
            "references, photorealistic high quality, cinematic mood."
        ),
        crop="face_closeup",
        angle="left_34",
    ),
]


class MultiViewGenerator:
    """
    Generates 10 diverse character views from a single input image using a
    snowball reference strategy.

    Each API call receives up to 5 reference images: the original character
    image plus previously generated views selected for maximum relevance.

    Phases:
      Phase 1 — Anchors (gen 1-3): refs grow from [orig] to [orig, g1, g2]
      Phase 2 — Angle expansion (gen 4-5): refs grow to 5 (max)
      Phase 3 — Diversity (gen 6-10): always 5 refs; 3 constants + 2 dynamic

    Attributes:
        MODEL: Gemini model identifier for image generation.
        INPUT_SIZE: Resolution to which the input image is resized before sending.
        MAX_RETRIES: Number of attempts before giving up on a failed request.
        BACKOFF_BASE: Initial back-off interval in seconds (doubles each retry).
    """

    MODEL: str = "gemini-3-pro-image-preview"
    INPUT_SIZE: tuple[int, int] = (1024, 1024)
    MAX_RETRIES: int = 3
    BACKOFF_BASE: float = 1.0  # seconds

    def __init__(self, api_key: str) -> None:
        """
        Initialise the generator.

        Args:
            api_key: Google Gemini API key used to authenticate all requests.
        """
        if not api_key:
            raise ValueError("api_key must not be empty.")
        self._client = genai.Client(api_key=api_key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_views(
        self,
        image: Image.Image,
        output_dir: str,
        callback: Optional[Callable[[str, str], None]] = None,
    ) -> tuple[list[str], list[Image.Image]]:
        """
        Generate 10 character views from a single PIL image using the snowball
        reference strategy.

        Args:
            image: Source PIL image of the character.
            output_dir: Directory where view images will be written. Created if
                        it does not already exist.
            callback: Optional function called after each view is saved with
                      signature ``callback(view_name, image_path)``.

        Returns:
            A 2-tuple of:
            - list of 10 absolute file paths in generation order
            - list of 10 PIL images in the same order

        Raises:
            MultiViewGeneratorError: If the API returns an error after
                MAX_RETRIES attempts, or if no image is found in the response.
            OSError: If the output directory cannot be created or files cannot
                     be written.
        """
        os.makedirs(output_dir, exist_ok=True)

        original = self._resize(image)
        generated_images: list[Image.Image] = []
        paths: list[str] = []

        for idx, spec in enumerate(_VIEW_SPECS):
            refs = self._build_references(idx, original, generated_images)

            print(
                f"[Chimera] MultiViewGenerator: generating {spec.name} "
                f"(gen {idx + 1}/10, {len(refs)} ref(s))..."
            )

            gen_image = self._generate_view_with_retry(refs, spec.prompt, spec.name)
            path = os.path.join(output_dir, f"{spec.name}.png")
            gen_image.save(path, format="PNG")

            print(
                f"[Chimera] MultiViewGenerator: saved {spec.name} → {path}"
            )

            generated_images.append(gen_image)
            paths.append(path)

            if callback is not None:
                callback(spec.name, path)

        return paths, generated_images

    # ------------------------------------------------------------------
    # Private helpers — snowball reference selection
    # ------------------------------------------------------------------

    def _build_references(
        self,
        gen_idx: int,
        original: Image.Image,
        generated: list[Image.Image],
    ) -> list[Image.Image]:
        """
        Build the reference image list for a given generation step.

        Phase 1 (gen_idx 0-2): progressively grow refs from [orig] to [orig,g0,g1].
        Phase 2 (gen_idx 3-4): grow to [orig,g0,g1,g2] and [orig,g0,g1,g2,g3].
        Phase 3 (gen_idx 5-9): always 5 refs using _select_references().

        Args:
            gen_idx: Zero-based index of the generation being prepared (0-9).
            original: Resized original character image.
            generated: List of already-generated PIL images (length == gen_idx).

        Returns:
            List of 1-5 PIL images to include as references.
        """
        if gen_idx == 0:
            # Gen 1: only original
            return [original]
        if gen_idx < 5:
            # Phases 1 & 2: take original + all prior generations (up to 5 total)
            refs = [original] + list(generated)
            return refs[:5]
        # Phase 3: 5 refs — 3 constants + 2 dynamic
        return self._select_references(gen_idx, original, generated)

    def _select_references(
        self,
        gen_idx: int,
        original: Image.Image,
        generated: list[Image.Image],
    ) -> list[Image.Image]:
        """
        Select 5 references for Phase 3 generations (gen_idx 5-9).

        Constants (indices into generated list):
          g0 = front_face_closeup  (generated[0])
          g1 = front_midbody       (generated[1])

        Dynamic 2 refs are chosen per generation based on angle/crop similarity:
          gen 5 (right_34_midbody):       + g4 (right_profile) + g3 (3/4 left)
          gen 6 (left_fullbody_walking):  + g2 (front_fullbody) + g3 (3/4 left)
          gen 7 (front_midbody_laughing): + g3 (smile/left34) + g5 (casual right34)
          gen 8 (rear_34_midbody):        + g3 (3/4 left) + g6 (left walking/side)
          gen 9 (left_34_closeup_drama):  + g4 (right profile face) + g3 (3/4 left)

        Args:
            gen_idx: Zero-based generation index (5-9).
            original: Resized original character image.
            generated: All previously generated images (length == gen_idx).

        Returns:
            List of exactly 5 PIL images.
        """
        constants = [original, generated[0], generated[1]]

        dynamic_pairs: dict[int, tuple[int, int]] = {
            5: (4, 3),  # right_34: right_profile_closeup + left_34_midbody
            6: (2, 3),  # left_fullbody_walking: front_fullbody + left_34_midbody
            7: (3, 5),  # front_midbody_laughing: left_34_midbody + right_34_midbody
            8: (3, 6),  # rear_34_midbody: left_34_midbody + left_fullbody_walking
            9: (4, 3),  # left_34_closeup_dramatic: right_profile_closeup + left_34_midbody
        }

        d1_idx, d2_idx = dynamic_pairs[gen_idx]
        return constants + [generated[d1_idx], generated[d2_idx]]

    # ------------------------------------------------------------------
    # Private helpers — API
    # ------------------------------------------------------------------

    def _generate_view_with_retry(
        self,
        references: list[Image.Image],
        prompt: str,
        view_name: str,
    ) -> Image.Image:
        """
        Call the Gemini API to generate a single view, with exponential back-off.

        Args:
            references: List of 1-5 PIL reference images to send alongside the prompt.
            prompt: Natural-language description of the desired view.
            view_name: Human-readable label used in log messages and errors.

        Returns:
            A PIL image containing the generated view.

        Raises:
            MultiViewGeneratorError: After MAX_RETRIES failed attempts, or if the
                API response contains no image part.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Content list: prompt text first, then all reference images
                contents: list = [prompt] + list(references)

                response = self._client.models.generate_content(
                    model=self.MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio="3:4",
                            image_size="2K",
                        ),
                    ),
                )
                return self._extract_image(response, view_name)
            except MultiViewGeneratorError:
                raise  # no point retrying a bad response structure
            except Exception as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES:
                    wait = self.BACKOFF_BASE * (2 ** (attempt - 1))
                    print(
                        f"[Chimera] MultiViewGenerator: {view_name} attempt "
                        f"{attempt} failed ({exc}). Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)

        raise MultiViewGeneratorError(
            f"Gemini API failed to generate {view_name} view after {self.MAX_RETRIES} "
            f"attempts. Last error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Private helpers — response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_image(response: object, view_name: str) -> Image.Image:
        """
        Pull the first image part out of a Gemini GenerateContentResponse.

        Args:
            response: The raw response object returned by ``generate_content``.
            view_name: Label used in the error message if no image is found.

        Returns:
            PIL image decoded from the first ``inline_data`` part.

        Raises:
            MultiViewGeneratorError: If no image part is present in the response.
        """
        for part in response.parts:
            if part.inline_data is not None:
                raw_bytes = part.inline_data.data
                return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        raise MultiViewGeneratorError(
            f"Gemini response for '{view_name}' view contained no image part. "
            f"Parts received: {[type(p).__name__ for p in response.parts]}"
        )

    # ------------------------------------------------------------------
    # Private helpers — image utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resize(image: Image.Image) -> Image.Image:
        """Resize image to INPUT_SIZE using high-quality Lanczos resampling."""
        return image.convert("RGB").resize(MultiViewGenerator.INPUT_SIZE, Image.LANCZOS)
