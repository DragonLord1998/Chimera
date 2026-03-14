"""
MultiViewGenerator — sends a single character image to the Google Gemini API
(gemini-3-pro-image-preview, "Nano Banana Pro") and generates three canonical
views: left side fullbody, front face, right side fullbody.

Each view is generated via a separate API call for reliability, with exponential
back-off on transient failures.
"""

from __future__ import annotations

import io
import os
import time
from typing import Optional

from google import genai
from google.genai import types
from PIL import Image


class MultiViewGeneratorError(Exception):
    """Raised when a Gemini API call fails after all retries."""


class MultiViewGenerator:
    """
    Generates left / front / right views of a character from a single input image
    by calling the Google Gemini image generation API.

    Attributes:
        MODEL: Gemini model identifier for image generation.
        VIEWS: Ordered mapping of view name to generation prompt.
        INPUT_SIZE: Resolution to which the input image is resized before sending.
        MAX_RETRIES: Number of attempts before giving up on a failed request.
        BACKOFF_BASE: Initial back-off interval in seconds (doubles each retry).
    """

    MODEL: str = "gemini-3-pro-image-preview"

    VIEWS: dict[str, str] = {
        "left": (
            "Generate a full body left side profile view of this exact character. "
            "Keep all features, clothing, and appearance identical. "
            "Plain white background. High quality, detailed."
        ),
        "front": (
            "Generate a front-facing full body view of this exact character. "
            "Keep all features, clothing, and appearance identical. "
            "Plain white background. High quality, detailed."
        ),
        "right": (
            "Generate a full body right side profile view of this exact character. "
            "Keep all features, clothing, and appearance identical. "
            "Plain white background. High quality, detailed."
        ),
    }

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
    ) -> tuple[list[str], list[Image.Image]]:
        """
        Generate three character views from a single PIL image.

        The input image is resized to :attr:`INPUT_SIZE` and sent to the Gemini
        API once per view (left, front, right).  Each call uses a tailored prompt
        that instructs the model to reproduce the character from a specific angle.

        Saved files are named ``left.png``, ``front.png``, and ``right.png``
        inside *output_dir*.

        Args:
            image: Source PIL image of the character.
            output_dir: Directory where the three view images will be written.
                        Created if it does not already exist.

        Returns:
            A 2-tuple of:
            - list of three absolute file paths: ``[left.png, front.png, right.png]``
            - list of three PIL images in the same order, for immediate display

        Raises:
            MultiViewGeneratorError: If the API returns an error after
                :attr:`MAX_RETRIES` attempts, or if no image is found in the
                response for a given view.
            OSError: If the output directory cannot be created or files cannot
                be written.
        """
        os.makedirs(output_dir, exist_ok=True)

        resized = self._resize(image)

        paths: list[str] = []
        pil_images: list[Image.Image] = []

        for view_name, prompt in self.VIEWS.items():
            print(
                f"[Chimera] MultiViewGenerator: generating {view_name} view..."
            )
            generated = self._generate_view_with_retry(resized, prompt, view_name)
            path = os.path.join(output_dir, f"{view_name}.png")
            generated.save(path, format="PNG")
            print(
                f"[Chimera] MultiViewGenerator: saved {view_name} view → {path}"
            )
            paths.append(path)
            pil_images.append(generated)

        return paths, pil_images

    # ------------------------------------------------------------------
    # Private helpers — API
    # ------------------------------------------------------------------

    def _generate_view_with_retry(
        self,
        image: Image.Image,
        prompt: str,
        view_name: str,
    ) -> Image.Image:
        """
        Call the Gemini API to generate a single view, with exponential back-off.

        Args:
            image: Resized PIL image of the character (the reference input).
            prompt: Natural-language description of the desired view angle.
            view_name: Human-readable label used in log messages and errors.

        Returns:
            A PIL image containing the generated view.

        Raises:
            MultiViewGeneratorError: After :attr:`MAX_RETRIES` failed attempts,
                or if the API response contains no image part.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.MODEL,
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
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
