"""
prompt_templates.py — Diverse prompt templates for character LoRA training.

50% of templates preserve the character's original clothing/appearance.
50% vary the clothing, poses, backgrounds, and lighting to teach the model
that the trigger word represents the character's identity, not their outfit.

All prompts reference all 5 input images for identity, with specific image
callouts matched to the shot angle:
  - Image 1: Left side fullbody
  - Image 2: Front face
  - Image 3: Right side fullbody
  - Image 4: Face close-up
  - Image 5: Back fullbody

Coverage axes:
  - Camera angle   (front, 3/4, profile, back-3/4)
  - Crop           (extreme close-up, close-up, head+shoulders, waist-up, full-body)
  - Expression     (neutral, slight smile, broad smile, laughing, serious, surprised, pensive)
  - Lighting       (soft studio, natural window, golden hour, dramatic side, backlit, overcast)
  - Background     (solid studio, blurred indoor, urban street, nature, abstract gradient)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Template bank — 50 entries (25 original outfit + 25 varied outfit)
# ---------------------------------------------------------------------------

# ---- BLOCK A: Original clothing (slots 1-25) ----
# These preserve the character's appearance exactly as in the reference images.

ORIGINAL_OUTFIT: list[str] = [
    # Close-ups (1-5)
    "the character from images 1 through 5, with face details from image 4, "
    "extreme close-up portrait, front facing, neutral expression, "
    "soft diffused studio lighting, solid light-gray background, sharp facial detail",

    "the character from images 1 through 5, with face from image 2, "
    "close-up portrait, slight smile, natural soft window lighting, "
    "blurred warm indoor background, shallow depth of field",

    "the character from images 1 through 5, with face details from image 4, "
    "close-up portrait, looking slightly left, serious expression, "
    "dramatic hard side lighting, deep shadow on one side, dark charcoal background",

    "the character from images 1 through 5, with face from image 2 and left profile from image 1, "
    "extreme close-up portrait, three-quarter view from left, "
    "surprised expression, high-key studio lighting, white background",

    "the character from images 1 through 5, with face from image 2 and right profile from image 3, "
    "close-up portrait, looking slightly right, pensive expression, "
    "overcast soft natural light, muted blurred outdoor background",

    # Head and shoulders (6-10)
    "the character from images 1 through 5, with front view from image 2, "
    "head and shoulders portrait, front facing, gentle smile, "
    "golden hour warm backlight with soft fill, blurred outdoor park background",

    "the character from images 1 through 5, with right side from image 3 and face from image 4, "
    "head and shoulders portrait, three-quarter view from right, "
    "neutral expression, clean butterfly studio lighting, solid dark-navy background",

    "the character from images 1 through 5, with left profile from image 1, "
    "head and shoulders portrait, direct profile from left, "
    "laughing expression, bright natural daylight, blurred urban street background",

    "the character from images 1 through 5, with left side from image 1 and face from image 4, "
    "head and shoulders portrait, three-quarter view from left, "
    "serious focused expression, dramatic Rembrandt lighting, dark moody background",

    "the character from images 1 through 5, with front view from image 2, "
    "head and shoulders portrait, front facing, broad smile, "
    "ring-flash studio lighting, abstract light-blue gradient background",

    # Waist-up (11-15)
    "the character from images 1 through 5, with front view from image 2, "
    "waist-up portrait, front facing, relaxed neutral expression, "
    "soft box studio lighting, solid off-white background",

    "the character from images 1 through 5, with right side from image 3, "
    "waist-up portrait, three-quarter view from right, slight smile, "
    "golden hour sunlight from the right, blurred nature forest background",

    "the character from images 1 through 5, with front view from image 2 and face from image 4, "
    "waist-up portrait, looking down slightly, pensive expression, "
    "moody overcast light, blurred rainy city street background",

    "the character from images 1 through 5, with left side from image 1, "
    "waist-up portrait, three-quarter view from left, surprised "
    "expression, dramatic upward rim lighting, dark abstract background",

    "the character from images 1 through 5, with right profile from image 3, "
    "waist-up portrait, direct profile from right, serious "
    "expression, natural window side light, blurred minimal indoor room background",

    # Full body (16-20)
    "the character from images 1 through 5, with full body from image 1 and image 3, "
    "full body portrait, front facing, neutral standing pose, "
    "soft even studio lighting, seamless white cyclorama background",

    "the character from images 1 through 5, with left fullbody from image 1, "
    "full body portrait, three-quarter view from left, casual "
    "relaxed stance, gentle smile, golden hour outdoor lighting, park lawn background",

    "the character from images 1 through 5, with front view from image 2 and fullbody from image 3, "
    "full body portrait, walking toward camera, slight smile, "
    "bright natural midday light, blurred urban sidewalk background",

    "the character from images 1 through 5, with right fullbody from image 3 and back from image 5, "
    "full body portrait, three-quarter view from right, "
    "looking back over shoulder, serious expression, backlit golden hour, nature path background",

    "the character from images 1 through 5, with left fullbody from image 1, "
    "full body portrait, side profile from left, pensive walking "
    "pose, overcast diffused outdoor light, blurred city background",

    # Mixed (21-25)
    "the character from images 1 through 5, with front view from image 2 and face from image 4, "
    "head and shoulders portrait, front facing, laughing expression, "
    "bright cheerful natural outdoor light, blurred green park background",

    "the character from images 1 through 5, with face details from image 4, "
    "close-up portrait, slight downward look, calm expression, "
    "warm golden candle-like practical lighting, dark intimate indoor background",

    "the character from images 1 through 5, with front view from image 2, "
    "waist-up portrait, front facing, confident neutral expression, "
    "clean split two-tone studio lighting, abstract dark-gray gradient background",

    "the character from images 1 through 5, with face from image 4 and right side from image 3, "
    "extreme close-up portrait, three-quarter view from right, "
    "soft genuine smile, backlit natural window light, creamy blurred indoor bokeh",

    "the character from images 1 through 5, with full body from image 1 and image 3, "
    "full body portrait, front facing, relaxed neutral expression, "
    "dramatic neon-tinted side lighting, blurred abstract urban night background",
]

# ---- BLOCK B: Different clothing, poses, backgrounds, lighting (slots 26-50) ----
# These teach the model that the trigger word = identity, not outfit.

VARIED_OUTFIT: list[str] = [
    # Close-ups with different styling (1-5)
    "the character from images 1 through 5, with face from image 2, "
    "close-up portrait, wearing a black leather jacket, "
    "front facing, confident smirk, harsh flash photography, gritty urban wall background",

    "the character from images 1 through 5, with face details from image 4, "
    "extreme close-up portrait, wearing a cozy knit sweater, "
    "soft genuine smile, warm fireplace glow lighting, blurred cabin interior background",

    "the character from images 1 through 5, with face from image 4 and right side from image 3, "
    "close-up portrait, wearing elegant formal attire with collar, "
    "three-quarter view, neutral composed expression, studio rim lighting, dark background",

    "the character from images 1 through 5, with face from image 2, "
    "close-up portrait, wearing a hoodie, looking slightly up, "
    "curious expression, cool blue neon lighting, blurred night city background",

    "the character from images 1 through 5, with right profile from image 3 and face from image 4, "
    "extreme close-up portrait, wearing a white t-shirt, "
    "direct profile from right, peaceful expression, soft morning sunlight, bright airy room",

    # Head and shoulders with different outfits (6-10)
    "the character from images 1 through 5, with front view from image 2, "
    "head and shoulders, wearing a business suit and tie, "
    "front facing, professional neutral expression, corporate studio lighting, gray backdrop",

    "the character from images 1 through 5, with left side from image 1 and face from image 4, "
    "head and shoulders, wearing a denim jacket over a graphic tee, "
    "three-quarter view from left, relaxed smile, warm afternoon sun, cafe terrace background",

    "the character from images 1 through 5, with right profile from image 3, "
    "head and shoulders, wearing a turtleneck sweater, "
    "looking right, thoughtful expression, moody overcast window light, library background",

    "the character from images 1 through 5, with front view from image 2, "
    "head and shoulders, wearing athletic sportswear, "
    "front facing, determined expression, bright outdoor daylight, sports field background",

    "the character from images 1 through 5, with right side from image 3 and face from image 4, "
    "head and shoulders, wearing a flannel shirt, "
    "three-quarter view from right, gentle laugh, golden hour backlight, countryside background",

    # Waist-up with different outfits and poses (11-15)
    "the character from images 1 through 5, with front view from image 2, "
    "waist-up, wearing a rain coat, arms crossed, "
    "front facing, serious expression, overcast stormy light, rainy street background",

    "the character from images 1 through 5, with right side from image 3, "
    "waist-up, wearing a Hawaiian shirt, leaning casually, "
    "three-quarter view, broad smile, bright tropical sunlight, beach background",

    "the character from images 1 through 5, with left side from image 1, "
    "waist-up, wearing a chef apron over a plain shirt, "
    "three-quarter view from left, focused expression, warm kitchen lighting, kitchen background",

    "the character from images 1 through 5, with front view from image 2, "
    "waist-up, wearing a vintage band t-shirt, hands in pockets, "
    "front facing, cool neutral expression, dramatic side lighting, concert venue background",

    "the character from images 1 through 5, with left profile from image 1, "
    "waist-up, wearing a cable-knit cardigan, reading a book, "
    "direct profile from left, peaceful expression, soft lamp light, cozy living room background",

    # Full body with different outfits and poses (16-20)
    "the character from images 1 through 5, with full body from image 1 and image 3, "
    "full body, wearing a tailored overcoat and scarf, "
    "walking confidently, three-quarter view, slight smile, cold winter daylight, snowy street",

    "the character from images 1 through 5, with front view from image 2 and fullbody from image 1, "
    "full body, wearing shorts and a tank top, "
    "standing relaxed, front facing, happy smile, bright summer sunlight, poolside background",

    "the character from images 1 through 5, with left fullbody from image 1, "
    "full body, wearing workout clothes and sneakers, "
    "mid-stride jogging pose, side profile, focused expression, morning light, park trail",

    "the character from images 1 through 5, with right fullbody from image 3, "
    "full body, wearing a long flowing dress or formal suit, "
    "standing elegantly, three-quarter view from right, composed expression, grand hall interior",

    "the character from images 1 through 5, with front view from image 2, "
    "full body, wearing jeans and a plain white shirt, "
    "sitting on steps, front facing, relaxed smile, golden hour light, old building exterior",

    # Mixed specialty with different contexts (21-25)
    "the character from images 1 through 5, with face from image 4 and left side from image 1, "
    "head and shoulders, wearing safety goggles on forehead "
    "and a work shirt, three-quarter view, proud smile, workshop fluorescent lighting, garage",

    "the character from images 1 through 5, with full body from image 3 and back from image 5, "
    "waist-up, wearing a backpack and hiking gear, "
    "looking into distance, adventurous expression, bright mountain sunlight, mountain peak",

    "the character from images 1 through 5, with face details from image 4, "
    "close-up, wearing round glasses and a scarf, "
    "front facing, warm smile, soft autumn light, blurred fall foliage background",

    "the character from images 1 through 5, with left fullbody from image 1 and back from image 5, "
    "full body, wearing a bathrobe, standing casually, "
    "three-quarter view from left, sleepy yawn expression, soft morning window light, bedroom",

    "the character from images 1 through 5, with front view from image 2 and face from image 4, "
    "waist-up, wearing a paint-splattered art smock, "
    "holding a paintbrush, focused expression, natural skylight, bright art studio background",
]

assert len(ORIGINAL_OUTFIT) == 25, f"Expected 25, got {len(ORIGINAL_OUTFIT)}"
assert len(VARIED_OUTFIT) == 25, f"Expected 25, got {len(VARIED_OUTFIT)}"

# Interleave: original, varied, original, varied...
# This ensures any subset preserves the 50/50 ratio.
PROMPT_TEMPLATES: list[str] = []
for orig, varied in zip(ORIGINAL_OUTFIT, VARIED_OUTFIT):
    PROMPT_TEMPLATES.append(orig)
    PROMPT_TEMPLATES.append(varied)

assert len(PROMPT_TEMPLATES) == 50, f"Expected 50, got {len(PROMPT_TEMPLATES)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_prompt_templates(num_images: int = 25) -> list[str]:
    """Return a list of ``num_images`` prompt strings.

    Always maintains a 50/50 split between original-outfit and varied-outfit
    prompts.  Each half is drawn evenly from its 25-entry bank.

    When ``num_images`` exceeds 50, prompts cycle with variation suffixes.

    Parameters
    ----------
    num_images:
        Number of prompts to return.

    Returns
    -------
    list[str]
        Exactly ``num_images`` prompt strings.
    """
    if num_images <= 0:
        raise ValueError(f"num_images must be positive, got {num_images}")

    half = num_images // 2
    other_half = num_images - half  # handles odd numbers

    def pick_from(bank: list[str], count: int) -> list[str]:
        bank_size = len(bank)
        if count <= bank_size:
            step = bank_size / count
            return [bank[int(i * step)] for i in range(count)]
        # Cycle with variation suffixes
        result: list[str] = []
        for i in range(count):
            base = bank[i % bank_size]
            cycle = i // bank_size
            result.append(base if cycle == 0 else f"{base}, variation {cycle}")
        return result

    orig = pick_from(ORIGINAL_OUTFIT, half)
    varied = pick_from(VARIED_OUTFIT, other_half)

    # Interleave: orig, varied, orig, varied...
    result: list[str] = []
    for i in range(max(len(orig), len(varied))):
        if i < len(orig):
            result.append(orig[i])
        if i < len(varied):
            result.append(varied[i])
    return result
