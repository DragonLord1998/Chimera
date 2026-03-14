"""
prompt_templates.py — Diverse prompt templates for character LoRA training.

All 25 templates are designed to maximise coverage across five axes:
  - Camera angle   (front, 3/4, profile, back-3/4, overhead)
  - Crop           (extreme close-up, close-up, head+shoulders, waist-up, full-body)
  - Expression     (neutral, slight smile, broad smile, laughing, serious, surprised, pensive)
  - Lighting       (soft studio, natural window, golden hour, dramatic side, backlit, overcast)
  - Background     (solid studio, blurred indoor, urban street, nature, abstract gradient)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Template bank — 25 entries, one per training slot
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES: list[str] = [
    # ---- 1-5: Extreme close-up / close-up ----
    "the character from image 1, extreme close-up portrait, front facing, neutral expression, "
    "soft diffused studio lighting, solid light-gray background, sharp facial detail",

    "the character from image 1, close-up portrait, slight smile, natural soft window lighting, "
    "blurred warm indoor background, shallow depth of field",

    "the character from image 1, close-up portrait, looking slightly left, serious expression, "
    "dramatic hard side lighting, deep shadow on one side, dark charcoal background",

    "the character from image 1, extreme close-up portrait, three-quarter view from left, "
    "surprised expression, high-key studio lighting, white background",

    "the character from image 1, close-up portrait, looking slightly right, pensive expression, "
    "overcast soft natural light, muted blurred outdoor background",

    # ---- 6-10: Head and shoulders ----
    "the character from image 1, head and shoulders portrait, front facing, gentle smile, "
    "golden hour warm backlight with soft fill, blurred outdoor park background",

    "the character from image 1, head and shoulders portrait, three-quarter view from right, "
    "neutral expression, clean butterfly studio lighting, solid dark-navy background",

    "the character from image 1, head and shoulders portrait, direct profile from left, "
    "laughing expression, bright natural daylight, blurred urban street background",

    "the character from image 1, head and shoulders portrait, three-quarter view from left, "
    "serious focused expression, dramatic Rembrandt lighting, dark moody background",

    "the character from image 1, head and shoulders portrait, front facing, broad smile, "
    "ring-flash studio lighting, abstract light-blue gradient background",

    # ---- 11-15: Waist-up / torso ----
    "the character from image 1, waist-up portrait, front facing, relaxed neutral expression, "
    "soft box studio lighting, solid off-white background",

    "the character from image 1, waist-up portrait, three-quarter view from right, slight smile, "
    "golden hour sunlight from the right, blurred nature forest background",

    "the character from image 1, waist-up portrait, looking down slightly, pensive expression, "
    "moody overcast light, blurred rainy city street background",

    "the character from image 1, waist-up portrait, three-quarter view from left, surprised "
    "expression, dramatic upward rim lighting, dark abstract background",

    "the character from image 1, waist-up portrait, direct profile from right, serious "
    "expression, natural window side light, blurred minimal indoor room background",

    # ---- 16-20: Full body ----
    "the character from image 1, full body portrait, front facing, neutral standing pose, "
    "soft even studio lighting, seamless white cyclorama background",

    "the character from image 1, full body portrait, three-quarter view from left, casual "
    "relaxed stance, gentle smile, golden hour outdoor lighting, park lawn background",

    "the character from image 1, full body portrait, walking toward camera, slight smile, "
    "bright natural midday light, blurred urban sidewalk background",

    "the character from image 1, full body portrait, three-quarter view from right, "
    "looking back over shoulder, serious expression, backlit golden hour, nature path background",

    "the character from image 1, full body portrait, side profile from left, pensive walking "
    "pose, overcast diffused outdoor light, blurred city background",

    # ---- 21-25: Mixed / specialty ----
    "the character from image 1, head and shoulders portrait, front facing, laughing expression, "
    "bright cheerful natural outdoor light, blurred green park background",

    "the character from image 1, close-up portrait, slight downward look, calm expression, "
    "warm golden candle-like practical lighting, dark intimate indoor background",

    "the character from image 1, waist-up portrait, front facing, confident neutral expression, "
    "clean split two-tone studio lighting, abstract dark-gray gradient background",

    "the character from image 1, extreme close-up portrait, three-quarter view from right, "
    "soft genuine smile, backlit natural window light, creamy blurred indoor bokeh",

    "the character from image 1, full body portrait, front facing, relaxed neutral expression, "
    "dramatic neon-tinted side lighting, blurred abstract urban night background",
]

assert len(PROMPT_TEMPLATES) == 25, (
    f"Expected 25 prompt templates, got {len(PROMPT_TEMPLATES)}"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_prompt_templates(num_images: int = 25) -> list[str]:
    """Return a list of ``num_images`` prompt strings.

    Parameters
    ----------
    num_images:
        Number of prompts to return.  When ``num_images`` is less than or
        equal to 25 a diverse subset is returned (evenly spaced across the
        full bank so all axes remain covered).  When ``num_images`` exceeds
        25 the bank is cycled and a minor variation suffix is appended to
        each repeated prompt so the generator does not produce exact
        duplicates.

    Returns
    -------
    list[str]
        Exactly ``num_images`` prompt strings.
    """
    if num_images <= 0:
        raise ValueError(f"num_images must be positive, got {num_images}")

    bank_size = len(PROMPT_TEMPLATES)

    if num_images <= bank_size:
        if num_images == bank_size:
            return list(PROMPT_TEMPLATES)

        # Evenly spaced indices to preserve diversity across all 25 slots.
        step = bank_size / num_images
        indices = [int(i * step) for i in range(num_images)]
        return [PROMPT_TEMPLATES[idx] for idx in indices]

    # More images requested than templates — cycle with variation suffixes.
    result: list[str] = []
    for i in range(num_images):
        base_prompt = PROMPT_TEMPLATES[i % bank_size]
        cycle = i // bank_size  # 0 on first pass, 1 on second, etc.
        if cycle == 0:
            result.append(base_prompt)
        else:
            # Append a lightweight variation marker so seeds produce
            # different outputs for the same logical prompt slot.
            result.append(f"{base_prompt}, variation {cycle}")
    return result
