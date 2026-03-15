"""
prompt_templates.py — Diverse prompt templates for character LoRA training.

50% of templates preserve the character's original clothing/appearance.
50% vary the clothing, poses, backgrounds, and lighting to teach the model
that the trigger word represents the character's identity, not their outfit.

Two template sets are provided:

**Flux 2 DEV templates** (5 reference images):
  - Image 1: Left side fullbody
  - Image 2: Front face
  - Image 3: Right side fullbody
  - Image 4: Face close-up
  - Image 5: Back fullbody

**Klein 9B KV templates** (4 reference images):
  - Image 1: Front face
  - Image 2: Face close-up
  - Image 3: Left fullbody
  - Image 4: Right fullbody

Coverage axes:
  - Camera angle   (front, 3/4, profile, back-3/4, low-angle, high-angle, dutch, over-shoulder, ECU)
  - Crop           (extreme close-up, close-up, head+shoulders, waist-up, full-body)
  - Expression     (neutral, slight smile, broad smile, laughing, serious, surprised, pensive,
                    confident, contemplative, playful)
  - Lighting       (soft studio, natural window, golden hour, dramatic side, backlit, overcast,
                    neon, rim light, Rembrandt, ring-flash, foggy, snow-diffuse)
  - Background     (solid studio, blurred indoor, urban street, nature, abstract gradient,
                    park, beach, rooftop, cafe, forest, office, gym, kitchen, library, workshop)
  - Action poses   (running, sitting, reaching, dancing, crouching, leaning, walking)
  - Season/weather (golden hour, rain, snow, autumn, summer, foggy morning)
  - Professional   (office, gym, kitchen, library, workshop)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Template bank — 150 entries (75 original outfit + 75 varied outfit)
# ---------------------------------------------------------------------------

# ---- BLOCK A: Original clothing (slots 1-75) ----
# These preserve the character's appearance exactly as in the reference images.

ORIGINAL_OUTFIT: list[str] = [
    # --- Close-ups (1-10) ---

    # 1
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "extreme close-up portrait, front facing, neutral expression, "
    "soft diffused studio lighting, solid light-gray background, sharp facial detail, "
    "wearing their original outfit, 85mm lens, f/1.8, shallow depth of field",

    # 2
    "the person in reference image 1 through 5, with face from reference image 2, "
    "close-up portrait, slight smile, natural soft window lighting, "
    "blurred warm indoor background, shallow depth of field, "
    "wearing their original outfit, 50mm lens, f/2.0, bokeh",

    # 3
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "close-up portrait, looking slightly left, serious expression, "
    "dramatic hard side lighting, deep shadow on one side, dark charcoal background, "
    "wearing their original outfit, 85mm lens, high contrast",

    # 4
    "the person in reference image 1 through 5, with face from reference image 2 and left profile from reference image 1, "
    "extreme close-up portrait, three-quarter view from left, "
    "surprised expression, high-key studio lighting, white background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 5
    "the person in reference image 1 through 5, with face from reference image 2 and right profile from reference image 3, "
    "close-up portrait, looking slightly right, pensive expression, "
    "overcast soft natural light, muted blurred outdoor background, "
    "wearing their original outfit, 50mm lens, f/2.0",

    # 6
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "extreme close-up portrait, front facing, genuine laugh, "
    "golden hour warm side light, blurred outdoor bokeh background, "
    "wearing their original outfit, 85mm lens, f/1.4",

    # 7
    "the person in reference image 1 through 5, with face from reference image 2, "
    "close-up portrait, looking slightly down, contemplative expression, "
    "warm candle-like practical lighting, dark intimate indoor background, "
    "wearing their original outfit, 50mm lens, f/1.8",

    # 8
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "extreme close-up portrait, three-quarter view from right, "
    "confident smirk, dramatic neon-blue side light, blurred night-city bokeh, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 9
    "the person in reference image 1 through 5, with face from reference image 2 and left side from reference image 1, "
    "close-up portrait, three-quarter view from left, playful expression, "
    "bright overcast diffuse light, blurred park background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 10
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "extreme close-up portrait, direct front, calm neutral expression, "
    "soft Rembrandt studio lighting, dark-brown gradient background, "
    "wearing their original outfit, 85mm lens, sharp eye detail",

    # --- Head and shoulders (11-20) ---

    # 11
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "head and shoulders portrait, front facing, gentle smile, "
    "golden hour warm backlight with soft fill, blurred outdoor park background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 12
    "the person in reference image 1 through 5, with right side from reference image 3 and face from reference image 4, "
    "head and shoulders portrait, three-quarter view from right, "
    "neutral expression, clean butterfly studio lighting, solid dark-navy background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 13
    "the person in reference image 1 through 5, with left profile from reference image 1, "
    "head and shoulders portrait, direct profile from left, "
    "laughing expression, bright natural daylight, blurred urban street background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 14
    "the person in reference image 1 through 5, with left side from reference image 1 and face from reference image 4, "
    "head and shoulders portrait, three-quarter view from left, "
    "serious focused expression, dramatic Rembrandt lighting, dark moody background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 15
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "head and shoulders portrait, front facing, broad smile, "
    "ring-flash studio lighting, abstract light-blue gradient background, "
    "wearing their original outfit, 50mm lens, f/2.8",

    # 16
    "the person in reference image 1 through 5, with front view from reference image 2 and face from reference image 4, "
    "head and shoulders portrait, front facing, laughing expression, "
    "bright cheerful natural outdoor light, blurred green park background, "
    "wearing their original outfit, 85mm lens, f/1.8",

    # 17
    "the person in reference image 1 through 5, with right side from reference image 3, "
    "head and shoulders portrait, three-quarter view from right, surprised open-mouth expression, "
    "soft overcast outdoor light, blurred autumn foliage background, "
    "wearing their original outfit, 85mm lens, f/2.5",

    # 18
    "the person in reference image 1 through 5, with left side from reference image 1, "
    "head and shoulders portrait, low-angle hero shot looking up, "
    "confident neutral expression, dramatic upward rim lighting, dark abstract background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 19
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "head and shoulders portrait, high-angle overhead looking down, "
    "pensive downward gaze, soft diffuse studio light, seamless white background, "
    "wearing their original outfit, 50mm lens, f/2.0",

    # 20
    "the person in reference image 1 through 5, with right profile from reference image 3 and face from reference image 4, "
    "head and shoulders portrait, dutch-angle tilt composition, "
    "playful expression, dramatic split studio lighting, dark gradient background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # --- Waist-up (21-30) ---

    # 21
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "waist-up portrait, front facing, relaxed neutral expression, "
    "soft box studio lighting, solid off-white background, "
    "wearing their original outfit, 50mm lens, f/4.0",

    # 22
    "the person in reference image 1 through 5, with right side from reference image 3, "
    "waist-up portrait, three-quarter view from right, slight smile, "
    "golden hour sunlight from the right, blurred nature forest background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 23
    "the person in reference image 1 through 5, with front view from reference image 2 and face from reference image 4, "
    "waist-up portrait, looking down slightly, pensive expression, "
    "moody overcast light, blurred rainy city street background, "
    "wearing their original outfit, 50mm lens, f/2.8",

    # 24
    "the person in reference image 1 through 5, with left side from reference image 1, "
    "waist-up portrait, three-quarter view from left, surprised expression, "
    "dramatic upward rim lighting, dark abstract background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 25
    "the person in reference image 1 through 5, with right profile from reference image 3, "
    "waist-up portrait, direct profile from right, serious expression, "
    "natural window side light, blurred minimal indoor room background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 26
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "waist-up portrait, front facing, confident neutral expression, "
    "clean split two-tone studio lighting, abstract dark-gray gradient background, "
    "wearing their original outfit, 50mm lens, f/2.8",

    # 27
    "the person in reference image 1 through 5, with left side from reference image 1 and face from reference image 4, "
    "waist-up portrait, seated pose, leaning forward with arms on knees, "
    "warm golden hour light, blurred cafe window background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 28
    "the person in reference image 1 through 5, with right side from reference image 3, "
    "waist-up portrait, three-quarter view, reaching arm out to the side, "
    "bright overcast outdoor light, blurred garden background, "
    "wearing their original outfit, 50mm lens, f/3.5",

    # 29
    "the person in reference image 1 through 5, with front view from reference image 2 and left side from reference image 1, "
    "waist-up portrait, slight body turn, hands clasped in front, "
    "soft studio beauty lighting, seamless cream background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 30
    "the person in reference image 1 through 5, with face details from reference image 4 and right side from reference image 3, "
    "waist-up portrait, three-quarter view, laughing with head tilted back, "
    "bright natural midday sun, blurred urban rooftop background, "
    "wearing their original outfit, 85mm lens, f/2.5",

    # --- Full body (31-45) ---

    # 31
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body portrait, front facing, neutral standing pose, "
    "soft even studio lighting, seamless white cyclorama background, "
    "wearing their original outfit, 35mm lens, f/5.6",

    # 32
    "the person in reference image 1 through 5, with left fullbody from reference image 1, "
    "full body portrait, three-quarter view from left, casual relaxed stance, gentle smile, "
    "golden hour outdoor lighting, park lawn background, "
    "wearing their original outfit, 35mm lens, f/4.0",

    # 33
    "the person in reference image 1 through 5, with front view from reference image 2 and fullbody from reference image 3, "
    "full body portrait, walking toward camera, slight smile, "
    "bright natural midday light, blurred urban sidewalk background, "
    "wearing their original outfit, 35mm lens, f/4.0",

    # 34
    "the person in reference image 1 through 5, with right fullbody from reference image 3 and back from reference image 5, "
    "full body portrait, three-quarter view from right, looking back over shoulder, "
    "serious expression, backlit golden hour, nature path background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 35
    "the person in reference image 1 through 5, with left fullbody from reference image 1, "
    "full body portrait, side profile from left, pensive walking pose, "
    "overcast diffused outdoor light, blurred city background, "
    "wearing their original outfit, 50mm lens, f/3.5",

    # 36
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body portrait, front facing, relaxed neutral expression, "
    "dramatic neon-tinted side lighting, blurred abstract urban night background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 37
    "the person in reference image 1 through 5, with left fullbody from reference image 1 and front from reference image 2, "
    "full body portrait, running pose mid-stride, dynamic energy, "
    "motion-frozen bright daylight, blurred park path background, "
    "wearing their original outfit, 35mm lens, f/4.0, slight motion blur on feet",

    # 38
    "the person in reference image 1 through 5, with full body from reference image 3 and back from reference image 5, "
    "full body portrait, crouching down, looking up at camera from below, "
    "low-angle hero composition, dramatic backlit sunset, silhouette-kissed edges, "
    "wearing their original outfit, 24mm lens, f/4.0",

    # 39
    "the person in reference image 1 through 5, with right fullbody from reference image 3, "
    "full body portrait, sitting on a low concrete ledge, relaxed posture, "
    "soft golden afternoon light, blurred urban street background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 40
    "the person in reference image 1 through 5, with left fullbody from reference image 1 and front from reference image 2, "
    "full body portrait, arms crossed, confident front-facing stance, "
    "bright overcast diffuse light, seamless gradient gray background, "
    "wearing their original outfit, 35mm lens, f/4.0",

    # 41
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body portrait, low-angle shot from ground level looking up, "
    "triumphant pose with arms relaxed at sides, cloudy sky background, "
    "wearing their original outfit, 24mm lens, f/5.6",

    # 42
    "the person in reference image 1 through 5, with right fullbody from reference image 3 and face from reference image 4, "
    "full body portrait, leaning against a wall, one foot up, "
    "cool neutral expression, dramatic soft sidelight, dark studio background, "
    "wearing their original outfit, 50mm lens, f/2.8",

    # 43
    "the person in reference image 1 through 5, with left fullbody from reference image 1, "
    "full body portrait, three-quarter view, walking away then glancing back over shoulder, "
    "gentle smile, soft morning fog light, blurred urban alley background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 44
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body portrait, front facing, snow falling in background, "
    "content expression, cold diffuse winter light, snow-covered street background, "
    "wearing their original outfit, 50mm lens, f/4.0",

    # 45
    "the person in reference image 1 through 5, with left fullbody from reference image 1 and back from reference image 5, "
    "full body portrait, dynamic dancing pose, one arm raised, "
    "joyful expression, colorful stage lighting, dark concert background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # --- Action, environment, and specialty (46-60) ---

    # 46
    "the person in reference image 1 through 5, with front view from reference image 2 and left side from reference image 1, "
    "waist-up portrait, seated at a wooden desk with hands on table, "
    "focused working expression, warm desk-lamp light, blurred office bookshelf background, "
    "wearing their original outfit, 50mm lens, f/2.0",

    # 47
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "close-up portrait, genuine warm smile, late afternoon golden hour, "
    "blurred autumn leaves falling in background, "
    "wearing their original outfit, 85mm lens, f/1.4",

    # 48
    "the person in reference image 1 through 5, with right side from reference image 3 and face from reference image 4, "
    "head and shoulders portrait, three-quarter view, over-shoulder framing, "
    "looking back at camera, cool expression, moody foggy morning light, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 49
    "the person in reference image 1 through 5, with left fullbody from reference image 1, "
    "full body portrait, sitting cross-legged on the ground, relaxed, "
    "peaceful expression, soft dappled forest light, blurred green forest background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 50
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "waist-up portrait, slight lean to one side, relaxed posture, "
    "gentle smile, bright beach sunlight, blurred ocean horizon background, "
    "wearing their original outfit, 85mm lens, f/3.5",

    # 51
    "the person in reference image 1 through 5, with face from reference image 4 and right side from reference image 3, "
    "extreme close-up portrait, three-quarter view from right, soft genuine smile, "
    "backlit natural window light, creamy blurred indoor bokeh, "
    "wearing their original outfit, 85mm lens, f/1.4",

    # 52
    "the person in reference image 1 through 5, with left side from reference image 1 and front from reference image 2, "
    "head and shoulders portrait, rain visible through background window, "
    "contemplative expression staring outward, cool overcast window light, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 53
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body portrait, standing on a rooftop, city skyline at golden hour behind them, "
    "confident expression looking at camera, warm backlight with fill, "
    "wearing their original outfit, 35mm lens, f/4.0",

    # 54
    "the person in reference image 1 through 5, with front view from reference image 2 and face from reference image 4, "
    "waist-up portrait, library setting with bookshelves behind, "
    "thoughtful expression, soft warm reading-lamp light, shallow focus on subject, "
    "wearing their original outfit, 50mm lens, f/1.8",

    # 55
    "the person in reference image 1 through 5, with right fullbody from reference image 3 and left from reference image 1, "
    "full body portrait, standing in a gym, weights rack in background, "
    "determined expression, harsh fluorescent gym lighting with fill, "
    "wearing their original outfit, 35mm lens, f/4.0",

    # 56
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "close-up portrait, side-lit by neon sign, purple-pink rim light, "
    "cool neutral expression, blurred neon-lit urban night background, "
    "wearing their original outfit, 85mm lens, f/1.8",

    # 57
    "the person in reference image 1 through 5, with left side from reference image 1 and back from reference image 5, "
    "full body portrait, walking up outdoor stone steps, mid-step, "
    "slight smile glancing sideways, warm afternoon side light, blurred garden stairway, "
    "wearing their original outfit, 35mm lens, f/3.5",

    # 58
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "head and shoulders portrait, front facing, eyes closed with serene expression, "
    "soft rim backlight creating a halo, dark gradient background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 59
    "the person in reference image 1 through 5, with face from reference image 4 and right side from reference image 3, "
    "close-up portrait, standing in front of frosted glass window, snow visible outside, "
    "warm expression, cool blue exterior light contrasting warm interior, "
    "wearing their original outfit, 85mm lens, f/1.8",

    # 60
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body portrait, standing at edge of a beach, sand and shallow water at feet, "
    "squinting gently in bright summer sunlight, high-key summer light, "
    "wearing their original outfit, 35mm lens, f/5.6",

    # --- Mixed camera specialty (61-75) ---

    # 61
    "the person in reference image 1 through 5, with face details from reference image 4 and front from reference image 2, "
    "extreme close-up portrait, slightly low camera angle, strong jaw line visible, "
    "confident composed expression, dramatic single-source side light, dark studio, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 62
    "the person in reference image 1 through 5, with left side from reference image 1 and face from reference image 4, "
    "head and shoulders portrait, over-shoulder camera angle from behind-left, "
    "looking back at camera with slight smile, soft morning light, blurred park path, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 63
    "the person in reference image 1 through 5, with right fullbody from reference image 3 and face from reference image 4, "
    "full body portrait, kneeling on one knee in dynamic pose, "
    "focused intense expression, dramatic low rim lighting from side, dark gradient background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 64
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "waist-up portrait, direct camera stare with very subtle smile, "
    "split studio lighting — one half warm, one half cool, seamless dark background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 65
    "the person in reference image 1 through 5, with left fullbody from reference image 1, "
    "full body portrait, sitting on a park bench, legs crossed, looking to the side, "
    "relaxed thoughtful expression, dappled golden light through trees, blurred park, "
    "wearing their original outfit, 50mm lens, f/2.5",

    # 66
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "extreme close-up portrait, backlit by bright window, face in soft warm shadow, "
    "serene eyes looking slightly upward, clean silhouette-kissed rim light, "
    "wearing their original outfit, 85mm lens, f/1.4",

    # 67
    "the person in reference image 1 through 5, with full body from reference image 3 and back from reference image 5, "
    "full body portrait, walking through foggy morning street, "
    "slightly away from camera then looking back, moody foggy diffuse light, "
    "wearing their original outfit, 50mm lens, f/2.8",

    # 68
    "the person in reference image 1 through 5, with right side from reference image 3 and face from reference image 4, "
    "head and shoulders portrait, three-quarter view, standing in a cafe doorway, "
    "warm cheerful expression, warm golden backlight from cafe interior, blurred street outside, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 69
    "the person in reference image 1 through 5, with left fullbody from reference image 1 and front from reference image 2, "
    "full body portrait, standing in shallow water at the beach, waves at ankles, "
    "happy carefree expression, bright summer backlight, golden sand and blue water, "
    "wearing their original outfit, 35mm lens, f/5.6",

    # 70
    "the person in reference image 1 through 5, with face from reference image 2, "
    "close-up portrait, standing under warm street light at night, "
    "soft romantic expression, warm orange-yellow practical light against dark night background, "
    "wearing their original outfit, 85mm lens, f/1.4",

    # 71
    "the person in reference image 1 through 5, with right fullbody from reference image 3 and left from reference image 1, "
    "full body portrait, leaning against a rustic brick wall, arms relaxed at sides, "
    "cool confident expression, warm afternoon golden side light, urban alley background, "
    "wearing their original outfit, 35mm lens, f/3.5",

    # 72
    "the person in reference image 1 through 5, with face details from reference image 4 and left side from reference image 1, "
    "close-up portrait, three-quarter view from left, rain-soaked look, "
    "wet strands framing face, steely serious expression, cool overcast rainy light, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 73
    "the person in reference image 1 through 5, with front view from reference image 2 and face from reference image 4, "
    "waist-up portrait, standing in a sunlit kitchen, natural light from window, "
    "relaxed warm smile, bright airy soft-box window light, blurred kitchen background, "
    "wearing their original outfit, 50mm lens, f/2.8",

    # 74
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body portrait, standing on a mountain overlook at dusk, "
    "looking into the distance, epic golden-pink sky behind, silhouette edge from backlight, "
    "wearing their original outfit, 35mm lens, f/5.6",

    # 75
    "the person in reference image 1 through 5, with face from reference image 4 and right side from reference image 3, "
    "close-up portrait, directional three-quarter light from upper-left, "
    "slight upward gaze, composed serene expression, gradient steel-blue background, "
    "wearing their original outfit, 85mm lens, f/1.8",
]

# ---- BLOCK B: Different clothing, poses, backgrounds, lighting (slots 76-150) ----
# These teach the model that the trigger word = identity, not outfit.

VARIED_OUTFIT: list[str] = [
    # --- Close-ups with different styling (1-10) ---

    # 1
    "the person in reference image 1 through 5, with face from reference image 2, "
    "close-up portrait, wearing a black leather biker jacket, "
    "front facing, confident smirk, harsh single-flash photography, gritty urban wall background, "
    "85mm lens, f/2.0, high contrast",

    # 2
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "extreme close-up portrait, wearing a cozy oversized knit sweater, "
    "soft genuine smile, warm fireplace glow from the right, blurred cabin interior background, "
    "85mm lens, f/1.8, warm tones",

    # 3
    "the person in reference image 1 through 5, with face from reference image 4 and right side from reference image 3, "
    "close-up portrait, wearing elegant formal attire with stiff collar, "
    "three-quarter view, neutral composed expression, crisp studio rim lighting, dark background, "
    "85mm lens, f/2.8",

    # 4
    "the person in reference image 1 through 5, with face from reference image 2, "
    "close-up portrait, wearing a zip-up hoodie, looking slightly up, "
    "curious open expression, cool blue-purple neon lighting, blurred night city background, "
    "50mm lens, f/2.0",

    # 5
    "the person in reference image 1 through 5, with right profile from reference image 3 and face from reference image 4, "
    "extreme close-up portrait, wearing a crisp white t-shirt, "
    "direct profile from right, peaceful calm expression, soft morning sunlight, bright airy room, "
    "85mm lens, f/2.0",

    # 6
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "close-up portrait, wearing round tortoiseshell glasses and a knit beanie, "
    "front facing, playful grin, bright overcast outdoor light, blurred city background, "
    "85mm lens, f/1.8",

    # 7
    "the person in reference image 1 through 5, with face from reference image 2 and left side from reference image 1, "
    "close-up portrait, wearing a wool peacoat with collar turned up, "
    "three-quarter view from left, pensive expression, cold blue winter side light, blurred snowy street, "
    "85mm lens, f/2.0",

    # 8
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "extreme close-up portrait, wearing a dark turtleneck, "
    "front facing, serious intellectual expression, clean soft studio beauty light, dark gradient background, "
    "85mm lens, f/1.8",

    # 9
    "the person in reference image 1 through 5, with face from reference image 2, "
    "close-up portrait, wearing a rain-wet windbreaker jacket, "
    "rain visible falling in background, steely serious expression, cool overcast rainy light, "
    "85mm lens, f/2.5",

    # 10
    "the person in reference image 1 through 5, with face from reference image 4 and right side from reference image 3, "
    "close-up portrait, wearing a crisp button-down Oxford shirt, collar open, "
    "three-quarter view from right, relaxed genuine smile, warm golden hour window side light, "
    "85mm lens, f/1.8",

    # --- Head and shoulders with different outfits (11-20) ---

    # 11
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "head and shoulders, wearing a fitted business suit with pocket square, "
    "front facing, professional neutral expression, corporate studio lighting, clean gray backdrop, "
    "85mm lens, f/2.8",

    # 12
    "the person in reference image 1 through 5, with left side from reference image 1 and face from reference image 4, "
    "head and shoulders, wearing a denim jacket over a graphic band tee, "
    "three-quarter view from left, relaxed easy smile, warm afternoon sun, cafe terrace background, "
    "85mm lens, f/2.0",

    # 13
    "the person in reference image 1 through 5, with right profile from reference image 3, "
    "head and shoulders, wearing a chunky ribbed turtleneck sweater, "
    "direct profile from right, thoughtful expression, moody overcast window light, library background, "
    "85mm lens, f/2.5",

    # 14
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "head and shoulders, wearing compression athletic sportswear with a jersey collar, "
    "front facing, determined focused expression, bright outdoor daylight, sports field background, "
    "50mm lens, f/2.8",

    # 15
    "the person in reference image 1 through 5, with right side from reference image 3 and face from reference image 4, "
    "head and shoulders, wearing a soft flannel shirt, sleeves rolled up, "
    "three-quarter view from right, gentle genuine laugh, golden hour backlight, countryside background, "
    "85mm lens, f/2.0",

    # 16
    "the person in reference image 1 through 5, with front view from reference image 2 and face from reference image 4, "
    "head and shoulders, wearing a casual linen button-up shirt, top button open, "
    "front facing, easygoing happy smile, bright summer natural light, blurred garden background, "
    "85mm lens, f/2.0",

    # 17
    "the person in reference image 1 through 5, with left side from reference image 1, "
    "head and shoulders, wearing a slim-fit blazer over a crew-neck tee, "
    "three-quarter view from left, composed confident expression, clean studio split lighting, dark background, "
    "85mm lens, f/2.8",

    # 18
    "the person in reference image 1 through 5, with right side from reference image 3 and face from reference image 4, "
    "head and shoulders, wearing a printed camp-collar vacation shirt, "
    "three-quarter view, wide bright smile, tropical beach sunlight, blurred palm-tree background, "
    "50mm lens, f/2.8",

    # 19
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "head and shoulders, wearing a cozy zip-up fleece vest over a long-sleeve shirt, "
    "front facing, cheerful relaxed expression, soft morning golden light, blurred autumn forest background, "
    "85mm lens, f/2.0",

    # 20
    "the person in reference image 1 through 5, with left side from reference image 1 and face from reference image 4, "
    "head and shoulders, wearing a sleek mock-neck black top, "
    "three-quarter view from left, cool intense expression, dramatic neon-pink rim light from behind, dark background, "
    "85mm lens, f/2.0",

    # --- Waist-up with different outfits and poses (21-30) ---

    # 21
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "waist-up, wearing a belted rain coat, arms crossed, "
    "front facing, serious expression, overcast stormy light, rainy city street background, "
    "50mm lens, f/2.8",

    # 22
    "the person in reference image 1 through 5, with right side from reference image 3, "
    "waist-up, wearing a bright Hawaiian camp shirt, leaning casually against a railing, "
    "three-quarter view, broad sun-squinting smile, bright tropical sunlight, beach background, "
    "35mm lens, f/4.0",

    # 23
    "the person in reference image 1 through 5, with left side from reference image 1, "
    "waist-up, wearing a chef's apron over a plain linen shirt, arms slightly forward, "
    "three-quarter view from left, focused engaged expression, warm overhead kitchen lighting, "
    "50mm lens, f/2.8",

    # 24
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "waist-up, wearing a vintage band t-shirt, hands relaxed in pockets, "
    "front facing, cool understated expression, dramatic concert-style side lighting, "
    "50mm lens, f/2.8",

    # 25
    "the person in reference image 1 through 5, with left profile from reference image 1, "
    "waist-up, wearing a cable-knit cardigan, holding an open book, "
    "direct profile from left, peaceful reading expression, soft warm lamp light, cozy living room background, "
    "85mm lens, f/2.0",

    # 26
    "the person in reference image 1 through 5, with right side from reference image 3 and face from reference image 4, "
    "waist-up, wearing business casual — chinos and a collared shirt, "
    "three-quarter view, seated at a desk with hands folded, confident professional expression, "
    "warm office window light, blurred open-plan office background, 50mm lens, f/2.0",

    # 27
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "waist-up, wearing a slim-fit bomber jacket, "
    "front facing, casual cool stance, one hand in pocket, "
    "warm amber evening street light, blurred urban street background, 85mm lens, f/1.8",

    # 28
    "the person in reference image 1 through 5, with left side from reference image 1, "
    "waist-up, wearing gym workout gear — fitted tank top and shorts, "
    "three-quarter view, arms slightly raised showing effort, determined sweaty expression, "
    "harsh fluorescent gym lighting, blurred gym equipment background, 50mm lens, f/2.8",

    # 29
    "the person in reference image 1 through 5, with front view from reference image 2 and face from reference image 4, "
    "waist-up, wearing a cozy oversized university hoodie, "
    "front facing, sleepy relaxed expression, soft morning window light, blurred apartment living room, "
    "50mm lens, f/2.0",

    # 30
    "the person in reference image 1 through 5, with right side from reference image 3, "
    "waist-up, wearing a crisp white lab coat over a shirt, "
    "three-quarter view, focused analytical expression, bright clinical fluorescent lighting, "
    "blurred laboratory background, 50mm lens, f/4.0",

    # --- Full body with different outfits and poses (31-45) ---

    # 31
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body, wearing a tailored wool overcoat and wool scarf, "
    "walking confidently at three-quarter view, slight easy smile, cold winter daylight, snowy street background, "
    "35mm lens, f/4.0",

    # 32
    "the person in reference image 1 through 5, with front view from reference image 2 and fullbody from reference image 1, "
    "full body, wearing shorts and a casual tank top, "
    "standing relaxed front-facing, happy open smile, bright summer sunlight, poolside background, "
    "35mm lens, f/5.6",

    # 33
    "the person in reference image 1 through 5, with left fullbody from reference image 1, "
    "full body, wearing running shoes and athletic tracksuit, "
    "mid-stride jogging pose, side profile from left, focused expression, early morning light, park trail, "
    "35mm lens, f/4.0, slight motion blur on feet",

    # 34
    "the person in reference image 1 through 5, with right fullbody from reference image 3, "
    "full body, wearing a long flowing evening dress or sharp formal suit, "
    "standing elegantly at three-quarter view from right, composed poised expression, "
    "grand hall interior with chandeliers, 35mm lens, f/4.0",

    # 35
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "full body, wearing dark slim jeans and a plain white shirt, "
    "sitting on stone steps, front facing, relaxed warm smile, "
    "golden hour light, old stone building exterior, 35mm lens, f/3.5",

    # 36
    "the person in reference image 1 through 5, with left fullbody from reference image 1, "
    "full body, wearing casual streetwear — jogger pants and a graphic hoodie, "
    "leaning against a graffiti mural wall, relaxed cool pose, "
    "overcast diffuse light, urban street background, 35mm lens, f/3.5",

    # 37
    "the person in reference image 1 through 5, with right fullbody from reference image 3 and front from reference image 2, "
    "full body, wearing a smart business suit and carrying a briefcase, "
    "walking purposefully toward camera, confident neutral expression, "
    "morning city sunlight, blurred downtown office building background, 35mm lens, f/4.0",

    # 38
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body, wearing cargo pants and a utility vest over a long-sleeve shirt, "
    "crouching in an action-ready pose, alert scanning expression, "
    "moody overcast industrial background, 35mm lens, f/4.0",

    # 39
    "the person in reference image 1 through 5, with left fullbody from reference image 1 and back from reference image 5, "
    "full body, wearing a bathrobe, standing casually in a doorway, "
    "three-quarter view from left, sleepy warm expression, soft morning window light, cozy bedroom background, "
    "35mm lens, f/2.8",

    # 40
    "the person in reference image 1 through 5, with right fullbody from reference image 3, "
    "full body, wearing hiking boots, cargo trousers, and a fleece jacket, "
    "standing on a mountain trail, looking into the distance, "
    "bright mountain sunlight, rocky mountain peak background, 35mm lens, f/5.6",

    # 41
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body, wearing swim shorts or a swimsuit, "
    "standing at the edge of a swimming pool, relaxed happy pose, "
    "bright tropical midday sunlight, blurred poolside background, 35mm lens, f/5.6",

    # 42
    "the person in reference image 1 through 5, with left fullbody from reference image 1 and front from reference image 2, "
    "full body, wearing a leather jacket and dark jeans, boots, "
    "standing on a rooftop at golden hour, arms relaxed, confident expression, "
    "warm golden backlight, cityscape background, 35mm lens, f/2.8",

    # 43
    "the person in reference image 1 through 5, with right fullbody from reference image 3 and face from reference image 4, "
    "full body, wearing yoga wear — fitted leggings and a sports top, "
    "doing a standing balance yoga pose, arms extended, serene expression, "
    "soft studio yoga-room lighting, clean white background, 35mm lens, f/4.0",

    # 44
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body, wearing a cozy winter parka and knit hat, "
    "standing in a snow-covered park, happy expression, snow falling, "
    "cold diffuse winter daylight, snowy trees background, 35mm lens, f/4.0",

    # 45
    "the person in reference image 1 through 5, with left fullbody from reference image 1, "
    "full body, wearing paint-splattered overalls and a plain t-shirt, "
    "holding a large paintbrush, creative focused expression, "
    "natural skylight, bright art studio background with canvases, 35mm lens, f/3.5",

    # --- Mixed specialty with different contexts (46-60) ---

    # 46
    "the person in reference image 1 through 5, with face from reference image 4 and left side from reference image 1, "
    "head and shoulders, wearing safety goggles pushed up on forehead and a sturdy work shirt, "
    "three-quarter view from left, proud satisfied smile, "
    "bright workshop fluorescent lighting, blurred garage workshop background, 50mm lens, f/2.8",

    # 47
    "the person in reference image 1 through 5, with full body from reference image 3 and back from reference image 5, "
    "waist-up, wearing a backpack and technical hiking gear, "
    "looking into the distance dramatically, adventurous determined expression, "
    "bright mountain sunlight, mountain peak panorama background, 35mm lens, f/5.6",

    # 48
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "close-up portrait, wearing round wire-rimmed glasses and a wool scarf, "
    "front facing, warm intellectual smile, "
    "soft autumn afternoon side light, blurred fall foliage background, 85mm lens, f/1.8",

    # 49
    "the person in reference image 1 through 5, with front view from reference image 2 and face from reference image 4, "
    "waist-up, wearing paint-splattered art smock, holding a fine paintbrush, "
    "focused creative expression, "
    "natural skylight from above, bright art studio background, 50mm lens, f/2.8",

    # 50
    "the person in reference image 1 through 5, with left side from reference image 1 and face from reference image 4, "
    "head and shoulders, wearing a crisp white chef jacket, "
    "three-quarter view from left, proud accomplished expression, "
    "warm overhead kitchen lighting, blurred professional kitchen background, 50mm lens, f/2.8",

    # 51
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "waist-up, wearing a barista apron over a casual shirt, "
    "front facing, cheerful welcoming smile, "
    "warm golden cafe lighting, blurred espresso machine and cafe shelves background, 50mm lens, f/2.0",

    # 52
    "the person in reference image 1 through 5, with right fullbody from reference image 3 and left from reference image 1, "
    "full body, wearing a graduation gown and cap, "
    "standing proudly front-facing, big celebratory smile, "
    "bright outdoor graduation ceremony daylight, blurred green campus background, 35mm lens, f/4.0",

    # 53
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "close-up portrait, wearing boxing wraps on hands, athletic tank top visible at shoulders, "
    "fierce determined expression, "
    "harsh directional gym spotlight, dark gym background, 85mm lens, f/2.0",

    # 54
    "the person in reference image 1 through 5, with left fullbody from reference image 1 and front from reference image 2, "
    "full body, wearing a vintage varsity jacket over a t-shirt, jeans, and sneakers, "
    "relaxed cool stance, leaning on a classic car, "
    "warm late-afternoon sunlight, blurred retro diner background, 35mm lens, f/3.5",

    # 55
    "the person in reference image 1 through 5, with front view from reference image 2 and face from reference image 4, "
    "head and shoulders, wearing a motorcycle jacket with patches, "
    "front facing, bold direct stare at camera, "
    "hard side flash photography, dark concrete wall background, 85mm lens, f/4.0",

    # 56
    "the person in reference image 1 through 5, with right side from reference image 3, "
    "waist-up, wearing a medical scrubs top, stethoscope around neck, "
    "three-quarter view, compassionate professional expression, "
    "bright clinical hospital corridor lighting, blurred medical background, 50mm lens, f/3.5",

    # 57
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body, wearing an elegant evening outfit — cocktail dress or suit, "
    "standing at a rooftop bar, one hand resting on railing, "
    "golden dusk light from the skyline, city lights bokeh background, 35mm lens, f/2.8",

    # 58
    "the person in reference image 1 through 5, with face from reference image 4 and left side from reference image 1, "
    "close-up portrait, wearing a firefighter jacket with reflective stripes, helmet in hand, "
    "three-quarter view from left, stoic heroic expression, "
    "dramatic golden backlight, blurred smoke-tinged background, 85mm lens, f/2.0",

    # 59
    "the person in reference image 1 through 5, with right fullbody from reference image 3 and front from reference image 2, "
    "full body, wearing dark tactical cargo trousers and a fitted long-sleeve top, "
    "dynamic crouching action pose, tense alert expression, "
    "blue-gray industrial warehouse lighting, dark concrete background, 35mm lens, f/3.5",

    # 60
    "the person in reference image 1 through 5, with face details from reference image 4 and front from reference image 2, "
    "close-up portrait, wearing a cozy Christmas sweater, "
    "bright happy festive smile, "
    "warm holiday fairy-light bokeh background, cozy indoor Christmas setting, 85mm lens, f/1.8",

    # --- Final specialty mixed (61-75) ---

    # 61
    "the person in reference image 1 through 5, with left fullbody from reference image 1 and face from reference image 4, "
    "full body, wearing a sleek wetsuit, surfboard under arm, "
    "looking at the ocean with excited anticipation, "
    "bright golden morning beach light, blurred ocean waves background, 35mm lens, f/5.6",

    # 62
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "waist-up, wearing a striped Breton marinière top, "
    "front facing, relaxed intellectual expression, "
    "soft natural light, blurred coastal dock background, 50mm lens, f/2.0",

    # 63
    "the person in reference image 1 through 5, with right side from reference image 3 and face from reference image 4, "
    "head and shoulders, wearing a denim overalls outfit with a fitted long-sleeve underneath, "
    "three-quarter view from right, playful cheeky expression, "
    "bright summer park light, blurred floral garden background, 85mm lens, f/2.0",

    # 64
    "the person in reference image 1 through 5, with full body from reference image 1 and reference image 3, "
    "full body, wearing a warm knit sweater, corduroy trousers, and ankle boots, "
    "walking on a leaf-covered autumn path, coat slightly billowing, "
    "warm golden autumn light, blurred forest fall foliage, 35mm lens, f/3.5",

    # 65
    "the person in reference image 1 through 5, with face details from reference image 4, "
    "close-up portrait, wearing a military-style olive green jacket, collar up, "
    "front facing, stoic serious expression, "
    "cool overcast flat light, blurred urban concrete background, 85mm lens, f/2.0",

    # 66
    "the person in reference image 1 through 5, with left side from reference image 1 and front from reference image 2, "
    "head and shoulders, wearing a cozy cable-knit fisherman sweater, "
    "three-quarter view from left, warm contented smile, "
    "soft Nordic window light, blurred snowy landscape outside window, 85mm lens, f/2.0",

    # 67
    "the person in reference image 1 through 5, with right fullbody from reference image 3, "
    "full body, wearing a bright yellow rain slicker and rain boots, "
    "standing in rain, laughing with head up, "
    "gray rainy overcast light, wet city street background with puddle reflections, 35mm lens, f/4.0",

    # 68
    "the person in reference image 1 through 5, with front view from reference image 2 and face from reference image 4, "
    "waist-up, wearing a smart double-breasted peacoat, gloved hands visible, "
    "front facing, polished composed expression, "
    "cold winter morning sun, blurred cobblestone European city street background, 50mm lens, f/2.8",

    # 69
    "the person in reference image 1 through 5, with left fullbody from reference image 1 and back from reference image 5, "
    "full body, wearing track pants and a loose athletic hoodie, headphones around neck, "
    "relaxed urban streetwear pose leaning on a fence, "
    "late afternoon golden side light, blurred urban park background, 35mm lens, f/3.5",

    # 70
    "the person in reference image 1 through 5, with face from reference image 4 and right side from reference image 3, "
    "close-up portrait, wearing a silk or satin collared blouse or dress shirt, "
    "three-quarter view from right, sophisticated poised expression, "
    "elegant studio beauty lighting, seamless dark-green gradient background, 85mm lens, f/1.8",

    # 71
    "the person in reference image 1 through 5, with full body from reference image 3 and front from reference image 2, "
    "full body, wearing a dark tuxedo or gala gown, "
    "standing at the base of a marble staircase, "
    "elegant composed expression, grand event lighting with chandeliers, 35mm lens, f/4.0",

    # 72
    "the person in reference image 1 through 5, with left side from reference image 1 and face from reference image 4, "
    "head and shoulders, wearing a kimono-inspired wrap top, "
    "three-quarter view from left, serene contemplative expression, "
    "soft diffuse natural light, blurred Japanese garden background, 85mm lens, f/2.0",

    # 73
    "the person in reference image 1 through 5, with front view from reference image 2, "
    "waist-up, wearing a bright orange safety vest over a work shirt, hard hat under arm, "
    "front facing, confident workman expression, "
    "bright outdoor construction site daylight, blurred construction background, 50mm lens, f/4.0",

    # 74
    "the person in reference image 1 through 5, with right fullbody from reference image 3 and face from reference image 4, "
    "full body, wearing a classic trench coat, belt cinched, "
    "walking in light rain with umbrella, three-quarter view from right, "
    "moody gray rainy light, blurred wet city street, 35mm lens, f/2.8",

    # 75
    "the person in reference image 1 through 5, with face details from reference image 4 and left side from reference image 1, "
    "close-up portrait, wearing a warm terracotta-colored linen shirt, "
    "three-quarter view from left, relaxed easy confidence, "
    "golden Mediterranean afternoon light, blurred whitewashed terrace background, 85mm lens, f/1.8",
]

assert len(ORIGINAL_OUTFIT) == 75, f"Expected 75, got {len(ORIGINAL_OUTFIT)}"
assert len(VARIED_OUTFIT) == 75, f"Expected 75, got {len(VARIED_OUTFIT)}"

# Interleave: original, varied, original, varied...
# This ensures any subset preserves the 50/50 ratio.
PROMPT_TEMPLATES: list[str] = []
for orig, varied in zip(ORIGINAL_OUTFIT, VARIED_OUTFIT):
    PROMPT_TEMPLATES.append(orig)
    PROMPT_TEMPLATES.append(varied)

assert len(PROMPT_TEMPLATES) == 150, f"Expected 150, got {len(PROMPT_TEMPLATES)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _pick_from(bank: list[str], count: int) -> list[str]:
    """Select ``count`` prompts from a bank, cycling with suffixes if needed."""
    bank_size = len(bank)
    if count <= bank_size:
        step = bank_size / count
        return [bank[int(i * step)] for i in range(count)]
    result: list[str] = []
    for i in range(count):
        base = bank[i % bank_size]
        cycle = i // bank_size
        result.append(base if cycle == 0 else f"{base}, variation {cycle}")
    return result


def _interleave(orig: list[str], varied: list[str]) -> list[str]:
    """Interleave two lists: orig, varied, orig, varied..."""
    result: list[str] = []
    for i in range(max(len(orig), len(varied))):
        if i < len(orig):
            result.append(orig[i])
        if i < len(varied):
            result.append(varied[i])
    return result


def get_prompt_templates(num_images: int = 25) -> list[str]:
    """Return ``num_images`` prompt strings for Flux 2 DEV (5 reference images).

    Always maintains a 50/50 split between original-outfit and varied-outfit
    prompts.  Each half is drawn evenly from its 75-entry bank.

    When ``num_images`` exceeds 150, prompts cycle with variation suffixes.
    """
    if num_images <= 0:
        raise ValueError(f"num_images must be positive, got {num_images}")

    half = num_images // 2
    other_half = num_images - half

    orig = _pick_from(ORIGINAL_OUTFIT, half)
    varied = _pick_from(VARIED_OUTFIT, other_half)
    return _interleave(orig, varied)


# ---------------------------------------------------------------------------
# Klein 9B KV templates — 4 reference images
# ---------------------------------------------------------------------------
# Image mapping for Klein:
#   Image 1: Front face
#   Image 2: Face close-up
#   Image 3: Left fullbody
#   Image 4: Right fullbody

KLEIN_ORIGINAL_OUTFIT: list[str] = [
    # --- Close-ups (1-10) ---

    # 1
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "extreme close-up portrait, front facing, neutral expression, "
    "soft diffused studio lighting, solid light-gray background, sharp facial detail, "
    "wearing their original outfit, 85mm lens, f/1.8",

    # 2
    "the person in reference image 1 through 4, with face from reference image 1, "
    "close-up portrait, slight smile, natural soft window lighting, "
    "blurred warm indoor background, shallow depth of field, "
    "wearing their original outfit, 50mm lens, f/2.0, bokeh",

    # 3
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "close-up portrait, looking slightly left, serious expression, "
    "dramatic hard side lighting, deep shadow on one side, dark charcoal background, "
    "wearing their original outfit, 85mm lens, high contrast",

    # 4
    "the person in reference image 1 through 4, with face from reference image 1 and left profile from reference image 3, "
    "extreme close-up portrait, three-quarter view from left, "
    "surprised expression, high-key studio lighting, white background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 5
    "the person in reference image 1 through 4, with face from reference image 1 and right profile from reference image 4, "
    "close-up portrait, looking slightly right, pensive expression, "
    "overcast soft natural light, muted blurred outdoor background, "
    "wearing their original outfit, 50mm lens, f/2.0",

    # 6
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "extreme close-up portrait, front facing, genuine laugh, "
    "golden hour warm side light, blurred outdoor bokeh background, "
    "wearing their original outfit, 85mm lens, f/1.4",

    # 7
    "the person in reference image 1 through 4, with face from reference image 1, "
    "close-up portrait, looking slightly down, contemplative expression, "
    "warm candle-like practical lighting, dark intimate indoor background, "
    "wearing their original outfit, 50mm lens, f/1.8",

    # 8
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "extreme close-up portrait, three-quarter view from right, "
    "confident smirk, dramatic neon-blue side light, blurred night-city bokeh, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 9
    "the person in reference image 1 through 4, with face from reference image 1 and left side from reference image 3, "
    "close-up portrait, three-quarter view from left, playful expression, "
    "bright overcast diffuse light, blurred park background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 10
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "extreme close-up portrait, direct front, calm neutral expression, "
    "soft Rembrandt studio lighting, dark-brown gradient background, "
    "wearing their original outfit, 85mm lens, sharp eye detail",

    # --- Head and shoulders (11-20) ---

    # 11
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "head and shoulders portrait, front facing, gentle smile, "
    "golden hour warm backlight with soft fill, blurred outdoor park background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 12
    "the person in reference image 1 through 4, with right side from reference image 4 and face from reference image 2, "
    "head and shoulders portrait, three-quarter view from right, "
    "neutral expression, clean butterfly studio lighting, solid dark-navy background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 13
    "the person in reference image 1 through 4, with left profile from reference image 3, "
    "head and shoulders portrait, direct profile from left, "
    "laughing expression, bright natural daylight, blurred urban street background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 14
    "the person in reference image 1 through 4, with left side from reference image 3 and face from reference image 2, "
    "head and shoulders portrait, three-quarter view from left, "
    "serious focused expression, dramatic Rembrandt lighting, dark moody background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 15
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "head and shoulders portrait, front facing, broad smile, "
    "ring-flash studio lighting, abstract light-blue gradient background, "
    "wearing their original outfit, 50mm lens, f/2.8",

    # 16
    "the person in reference image 1 through 4, with front view from reference image 1 and face from reference image 2, "
    "head and shoulders portrait, front facing, laughing expression, "
    "bright cheerful natural outdoor light, blurred green park background, "
    "wearing their original outfit, 85mm lens, f/1.8",

    # 17
    "the person in reference image 1 through 4, with right side from reference image 4, "
    "head and shoulders portrait, three-quarter view from right, surprised open-mouth expression, "
    "soft overcast outdoor light, blurred autumn foliage background, "
    "wearing their original outfit, 85mm lens, f/2.5",

    # 18
    "the person in reference image 1 through 4, with left side from reference image 3, "
    "head and shoulders portrait, low-angle hero shot looking up, "
    "confident neutral expression, dramatic upward rim lighting, dark abstract background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 19
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "head and shoulders portrait, high-angle overhead looking down, "
    "pensive downward gaze, soft diffuse studio light, seamless white background, "
    "wearing their original outfit, 50mm lens, f/2.0",

    # 20
    "the person in reference image 1 through 4, with right profile from reference image 4 and face from reference image 2, "
    "head and shoulders portrait, dutch-angle tilt composition, "
    "playful expression, dramatic split studio lighting, dark gradient background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # --- Waist-up (21-30) ---

    # 21
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "waist-up portrait, front facing, relaxed neutral expression, "
    "soft box studio lighting, solid off-white background, "
    "wearing their original outfit, 50mm lens, f/4.0",

    # 22
    "the person in reference image 1 through 4, with right side from reference image 4, "
    "waist-up portrait, three-quarter view from right, slight smile, "
    "golden hour sunlight from the right, blurred nature forest background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 23
    "the person in reference image 1 through 4, with front view from reference image 1 and face from reference image 2, "
    "waist-up portrait, looking down slightly, pensive expression, "
    "moody overcast light, blurred rainy city street background, "
    "wearing their original outfit, 50mm lens, f/2.8",

    # 24
    "the person in reference image 1 through 4, with left side from reference image 3, "
    "waist-up portrait, three-quarter view from left, surprised expression, "
    "dramatic upward rim lighting, dark abstract background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 25
    "the person in reference image 1 through 4, with right profile from reference image 4, "
    "waist-up portrait, direct profile from right, serious expression, "
    "natural window side light, blurred minimal indoor room background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 26
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "waist-up portrait, front facing, confident neutral expression, "
    "clean split two-tone studio lighting, abstract dark-gray gradient background, "
    "wearing their original outfit, 50mm lens, f/2.8",

    # 27
    "the person in reference image 1 through 4, with left side from reference image 3 and face from reference image 2, "
    "waist-up portrait, seated pose, leaning forward with arms on knees, "
    "warm golden hour light, blurred cafe window background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 28
    "the person in reference image 1 through 4, with right side from reference image 4, "
    "waist-up portrait, three-quarter view, reaching arm out to the side, "
    "bright overcast outdoor light, blurred garden background, "
    "wearing their original outfit, 50mm lens, f/3.5",

    # 29
    "the person in reference image 1 through 4, with front view from reference image 1 and left side from reference image 3, "
    "waist-up portrait, slight body turn, hands clasped in front, "
    "soft studio beauty lighting, seamless cream background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 30
    "the person in reference image 1 through 4, with face details from reference image 2 and right side from reference image 4, "
    "waist-up portrait, three-quarter view, laughing with head tilted back, "
    "bright natural midday sun, blurred urban rooftop background, "
    "wearing their original outfit, 85mm lens, f/2.5",

    # --- Full body (31-45) ---

    # 31
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body portrait, front facing, neutral standing pose, "
    "soft even studio lighting, seamless white cyclorama background, "
    "wearing their original outfit, 35mm lens, f/5.6",

    # 32
    "the person in reference image 1 through 4, with left fullbody from reference image 3, "
    "full body portrait, three-quarter view from left, casual relaxed stance, gentle smile, "
    "golden hour outdoor lighting, park lawn background, "
    "wearing their original outfit, 35mm lens, f/4.0",

    # 33
    "the person in reference image 1 through 4, with front view from reference image 1 and fullbody from reference image 4, "
    "full body portrait, walking toward camera, slight smile, "
    "bright natural midday light, blurred urban sidewalk background, "
    "wearing their original outfit, 35mm lens, f/4.0",

    # 34
    "the person in reference image 1 through 4, with right fullbody from reference image 4, "
    "full body portrait, three-quarter view from right, looking back over shoulder, "
    "serious expression, backlit golden hour, nature path background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 35
    "the person in reference image 1 through 4, with left fullbody from reference image 3, "
    "full body portrait, side profile from left, pensive walking pose, "
    "overcast diffused outdoor light, blurred city background, "
    "wearing their original outfit, 50mm lens, f/3.5",

    # 36
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body portrait, front facing, relaxed neutral expression, "
    "dramatic neon-tinted side lighting, blurred abstract urban night background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 37
    "the person in reference image 1 through 4, with left fullbody from reference image 3 and front from reference image 1, "
    "full body portrait, running pose mid-stride, dynamic energy, "
    "motion-frozen bright daylight, blurred park path background, "
    "wearing their original outfit, 35mm lens, f/4.0, slight motion blur on feet",

    # 38
    "the person in reference image 1 through 4, with full body from reference image 4, "
    "full body portrait, crouching down, looking up at camera from below, "
    "low-angle hero composition, dramatic backlit sunset, silhouette-kissed edges, "
    "wearing their original outfit, 24mm lens, f/4.0",

    # 39
    "the person in reference image 1 through 4, with right fullbody from reference image 4, "
    "full body portrait, sitting on a low concrete ledge, relaxed posture, "
    "soft golden afternoon light, blurred urban street background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 40
    "the person in reference image 1 through 4, with left fullbody from reference image 3 and front from reference image 1, "
    "full body portrait, arms crossed, confident front-facing stance, "
    "bright overcast diffuse light, seamless gradient gray background, "
    "wearing their original outfit, 35mm lens, f/4.0",

    # 41
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body portrait, low-angle shot from ground level looking up, "
    "triumphant pose with arms relaxed at sides, cloudy sky background, "
    "wearing their original outfit, 24mm lens, f/5.6",

    # 42
    "the person in reference image 1 through 4, with right fullbody from reference image 4 and face from reference image 2, "
    "full body portrait, leaning against a wall, one foot up, "
    "cool neutral expression, dramatic soft sidelight, dark studio background, "
    "wearing their original outfit, 50mm lens, f/2.8",

    # 43
    "the person in reference image 1 through 4, with left fullbody from reference image 3, "
    "full body portrait, three-quarter view, walking away then glancing back over shoulder, "
    "gentle smile, soft morning fog light, blurred urban alley background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 44
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body portrait, front facing, snow falling in background, "
    "content expression, cold diffuse winter light, snow-covered street background, "
    "wearing their original outfit, 50mm lens, f/4.0",

    # 45
    "the person in reference image 1 through 4, with left fullbody from reference image 3, "
    "full body portrait, dynamic dancing pose, one arm raised, "
    "joyful expression, colorful stage lighting, dark concert background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # --- Action, environment, and specialty (46-60) ---

    # 46
    "the person in reference image 1 through 4, with front view from reference image 1 and left side from reference image 3, "
    "waist-up portrait, seated at a wooden desk with hands on table, "
    "focused working expression, warm desk-lamp light, blurred office bookshelf background, "
    "wearing their original outfit, 50mm lens, f/2.0",

    # 47
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "close-up portrait, genuine warm smile, late afternoon golden hour, "
    "blurred autumn leaves falling in background, "
    "wearing their original outfit, 85mm lens, f/1.4",

    # 48
    "the person in reference image 1 through 4, with right side from reference image 4 and face from reference image 2, "
    "head and shoulders portrait, three-quarter view, over-shoulder framing, "
    "looking back at camera, cool expression, moody foggy morning light, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 49
    "the person in reference image 1 through 4, with left fullbody from reference image 3, "
    "full body portrait, sitting cross-legged on the ground, relaxed, "
    "peaceful expression, soft dappled forest light, blurred green forest background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 50
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "waist-up portrait, slight lean to one side, relaxed posture, "
    "gentle smile, bright beach sunlight, blurred ocean horizon background, "
    "wearing their original outfit, 85mm lens, f/3.5",

    # 51
    "the person in reference image 1 through 4, with face from reference image 2 and right side from reference image 4, "
    "extreme close-up portrait, three-quarter view from right, soft genuine smile, "
    "backlit natural window light, creamy blurred indoor bokeh, "
    "wearing their original outfit, 85mm lens, f/1.4",

    # 52
    "the person in reference image 1 through 4, with left side from reference image 3 and front from reference image 1, "
    "head and shoulders portrait, rain visible through background window, "
    "contemplative expression staring outward, cool overcast window light, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 53
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body portrait, standing on a rooftop, city skyline at golden hour behind them, "
    "confident expression looking at camera, warm backlight with fill, "
    "wearing their original outfit, 35mm lens, f/4.0",

    # 54
    "the person in reference image 1 through 4, with front view from reference image 1 and face from reference image 2, "
    "waist-up portrait, library setting with bookshelves behind, "
    "thoughtful expression, soft warm reading-lamp light, shallow focus on subject, "
    "wearing their original outfit, 50mm lens, f/1.8",

    # 55
    "the person in reference image 1 through 4, with right fullbody from reference image 4 and left from reference image 3, "
    "full body portrait, standing in a gym, weights rack in background, "
    "determined expression, harsh fluorescent gym lighting with fill, "
    "wearing their original outfit, 35mm lens, f/4.0",

    # 56
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "close-up portrait, side-lit by neon sign, purple-pink rim light, "
    "cool neutral expression, blurred neon-lit urban night background, "
    "wearing their original outfit, 85mm lens, f/1.8",

    # 57
    "the person in reference image 1 through 4, with left side from reference image 3, "
    "full body portrait, walking up outdoor stone steps, mid-step, "
    "slight smile glancing sideways, warm afternoon side light, blurred garden stairway, "
    "wearing their original outfit, 35mm lens, f/3.5",

    # 58
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "head and shoulders portrait, front facing, eyes closed with serene expression, "
    "soft rim backlight creating a halo, dark gradient background, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 59
    "the person in reference image 1 through 4, with face from reference image 2 and right side from reference image 4, "
    "close-up portrait, standing in front of frosted glass window, snow visible outside, "
    "warm expression, cool blue exterior light contrasting warm interior, "
    "wearing their original outfit, 85mm lens, f/1.8",

    # 60
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body portrait, standing at edge of a beach, sand and shallow water at feet, "
    "squinting gently in bright summer sunlight, high-key summer light, "
    "wearing their original outfit, 35mm lens, f/5.6",

    # --- Mixed camera specialty (61-75) ---

    # 61
    "the person in reference image 1 through 4, with face details from reference image 2 and front from reference image 1, "
    "extreme close-up portrait, slightly low camera angle, strong jaw line visible, "
    "confident composed expression, dramatic single-source side light, dark studio, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 62
    "the person in reference image 1 through 4, with left side from reference image 3 and face from reference image 2, "
    "head and shoulders portrait, over-shoulder camera angle from behind-left, "
    "looking back at camera with slight smile, soft morning light, blurred park path, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 63
    "the person in reference image 1 through 4, with right fullbody from reference image 4 and face from reference image 2, "
    "full body portrait, kneeling on one knee in dynamic pose, "
    "focused intense expression, dramatic low rim lighting from side, dark gradient background, "
    "wearing their original outfit, 35mm lens, f/2.8",

    # 64
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "waist-up portrait, direct camera stare with very subtle smile, "
    "split studio lighting — one half warm, one half cool, seamless dark background, "
    "wearing their original outfit, 85mm lens, f/2.8",

    # 65
    "the person in reference image 1 through 4, with left fullbody from reference image 3, "
    "full body portrait, sitting on a park bench, legs crossed, looking to the side, "
    "relaxed thoughtful expression, dappled golden light through trees, blurred park, "
    "wearing their original outfit, 50mm lens, f/2.5",

    # 66
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "extreme close-up portrait, backlit by bright window, face in soft warm shadow, "
    "serene eyes looking slightly upward, clean silhouette-kissed rim light, "
    "wearing their original outfit, 85mm lens, f/1.4",

    # 67
    "the person in reference image 1 through 4, with full body from reference image 4, "
    "full body portrait, walking through foggy morning street, "
    "slightly away from camera then looking back, moody foggy diffuse light, "
    "wearing their original outfit, 50mm lens, f/2.8",

    # 68
    "the person in reference image 1 through 4, with right side from reference image 4 and face from reference image 2, "
    "head and shoulders portrait, three-quarter view, standing in a cafe doorway, "
    "warm cheerful expression, warm golden backlight from cafe interior, blurred street outside, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 69
    "the person in reference image 1 through 4, with left fullbody from reference image 3 and front from reference image 1, "
    "full body portrait, standing in shallow water at the beach, waves at ankles, "
    "happy carefree expression, bright summer backlight, golden sand and blue water, "
    "wearing their original outfit, 35mm lens, f/5.6",

    # 70
    "the person in reference image 1 through 4, with face from reference image 1, "
    "close-up portrait, standing under warm street light at night, "
    "soft romantic expression, warm orange-yellow practical light against dark night background, "
    "wearing their original outfit, 85mm lens, f/1.4",

    # 71
    "the person in reference image 1 through 4, with right fullbody from reference image 4 and left from reference image 3, "
    "full body portrait, leaning against a rustic brick wall, arms relaxed at sides, "
    "cool confident expression, warm afternoon golden side light, urban alley background, "
    "wearing their original outfit, 35mm lens, f/3.5",

    # 72
    "the person in reference image 1 through 4, with face details from reference image 2 and left side from reference image 3, "
    "close-up portrait, three-quarter view from left, rain-soaked look, "
    "wet strands framing face, steely serious expression, cool overcast rainy light, "
    "wearing their original outfit, 85mm lens, f/2.0",

    # 73
    "the person in reference image 1 through 4, with front view from reference image 1 and face from reference image 2, "
    "waist-up portrait, standing in a sunlit kitchen, natural light from window, "
    "relaxed warm smile, bright airy soft-box window light, blurred kitchen background, "
    "wearing their original outfit, 50mm lens, f/2.8",

    # 74
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body portrait, standing on a mountain overlook at dusk, "
    "looking into the distance, epic golden-pink sky behind, silhouette edge from backlight, "
    "wearing their original outfit, 35mm lens, f/5.6",

    # 75
    "the person in reference image 1 through 4, with face from reference image 2 and right side from reference image 4, "
    "close-up portrait, directional three-quarter light from upper-left, "
    "slight upward gaze, composed serene expression, gradient steel-blue background, "
    "wearing their original outfit, 85mm lens, f/1.8",
]

KLEIN_VARIED_OUTFIT: list[str] = [
    # --- Close-ups with different styling (1-10) ---

    # 1
    "the person in reference image 1 through 4, with face from reference image 1, "
    "close-up portrait, wearing a black leather biker jacket, "
    "front facing, confident smirk, harsh single-flash photography, gritty urban wall background, "
    "85mm lens, f/2.0, high contrast",

    # 2
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "extreme close-up portrait, wearing a cozy oversized knit sweater, "
    "soft genuine smile, warm fireplace glow from the right, blurred cabin interior background, "
    "85mm lens, f/1.8, warm tones",

    # 3
    "the person in reference image 1 through 4, with face from reference image 2 and right side from reference image 4, "
    "close-up portrait, wearing elegant formal attire with stiff collar, "
    "three-quarter view, neutral composed expression, crisp studio rim lighting, dark background, "
    "85mm lens, f/2.8",

    # 4
    "the person in reference image 1 through 4, with face from reference image 1, "
    "close-up portrait, wearing a zip-up hoodie, looking slightly up, "
    "curious open expression, cool blue-purple neon lighting, blurred night city background, "
    "50mm lens, f/2.0",

    # 5
    "the person in reference image 1 through 4, with right profile from reference image 4 and face from reference image 2, "
    "extreme close-up portrait, wearing a crisp white t-shirt, "
    "direct profile from right, peaceful calm expression, soft morning sunlight, bright airy room, "
    "85mm lens, f/2.0",

    # 6
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "close-up portrait, wearing round tortoiseshell glasses and a knit beanie, "
    "front facing, playful grin, bright overcast outdoor light, blurred city background, "
    "85mm lens, f/1.8",

    # 7
    "the person in reference image 1 through 4, with face from reference image 1 and left side from reference image 3, "
    "close-up portrait, wearing a wool peacoat with collar turned up, "
    "three-quarter view from left, pensive expression, cold blue winter side light, blurred snowy street, "
    "85mm lens, f/2.0",

    # 8
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "extreme close-up portrait, wearing a dark turtleneck, "
    "front facing, serious intellectual expression, clean soft studio beauty light, dark gradient background, "
    "85mm lens, f/1.8",

    # 9
    "the person in reference image 1 through 4, with face from reference image 1, "
    "close-up portrait, wearing a rain-wet windbreaker jacket, "
    "rain visible falling in background, steely serious expression, cool overcast rainy light, "
    "85mm lens, f/2.5",

    # 10
    "the person in reference image 1 through 4, with face from reference image 2 and right side from reference image 4, "
    "close-up portrait, wearing a crisp button-down Oxford shirt, collar open, "
    "three-quarter view from right, relaxed genuine smile, warm golden hour window side light, "
    "85mm lens, f/1.8",

    # --- Head and shoulders with different outfits (11-20) ---

    # 11
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "head and shoulders, wearing a fitted business suit with pocket square, "
    "front facing, professional neutral expression, corporate studio lighting, clean gray backdrop, "
    "85mm lens, f/2.8",

    # 12
    "the person in reference image 1 through 4, with left side from reference image 3 and face from reference image 2, "
    "head and shoulders, wearing a denim jacket over a graphic band tee, "
    "three-quarter view from left, relaxed easy smile, warm afternoon sun, cafe terrace background, "
    "85mm lens, f/2.0",

    # 13
    "the person in reference image 1 through 4, with right profile from reference image 4, "
    "head and shoulders, wearing a chunky ribbed turtleneck sweater, "
    "direct profile from right, thoughtful expression, moody overcast window light, library background, "
    "85mm lens, f/2.5",

    # 14
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "head and shoulders, wearing compression athletic sportswear with a jersey collar, "
    "front facing, determined focused expression, bright outdoor daylight, sports field background, "
    "50mm lens, f/2.8",

    # 15
    "the person in reference image 1 through 4, with right side from reference image 4 and face from reference image 2, "
    "head and shoulders, wearing a soft flannel shirt, sleeves rolled up, "
    "three-quarter view from right, gentle genuine laugh, golden hour backlight, countryside background, "
    "85mm lens, f/2.0",

    # 16
    "the person in reference image 1 through 4, with front view from reference image 1 and face from reference image 2, "
    "head and shoulders, wearing a casual linen button-up shirt, top button open, "
    "front facing, easygoing happy smile, bright summer natural light, blurred garden background, "
    "85mm lens, f/2.0",

    # 17
    "the person in reference image 1 through 4, with left side from reference image 3, "
    "head and shoulders, wearing a slim-fit blazer over a crew-neck tee, "
    "three-quarter view from left, composed confident expression, clean studio split lighting, dark background, "
    "85mm lens, f/2.8",

    # 18
    "the person in reference image 1 through 4, with right side from reference image 4 and face from reference image 2, "
    "head and shoulders, wearing a printed camp-collar vacation shirt, "
    "three-quarter view, wide bright smile, tropical beach sunlight, blurred palm-tree background, "
    "50mm lens, f/2.8",

    # 19
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "head and shoulders, wearing a cozy zip-up fleece vest over a long-sleeve shirt, "
    "front facing, cheerful relaxed expression, soft morning golden light, blurred autumn forest background, "
    "85mm lens, f/2.0",

    # 20
    "the person in reference image 1 through 4, with left side from reference image 3 and face from reference image 2, "
    "head and shoulders, wearing a sleek mock-neck black top, "
    "three-quarter view from left, cool intense expression, dramatic neon-pink rim light from behind, dark background, "
    "85mm lens, f/2.0",

    # --- Waist-up with different outfits (21-30) ---

    # 21
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "waist-up, wearing a belted rain coat, arms crossed, "
    "front facing, serious expression, overcast stormy light, rainy city street background, "
    "50mm lens, f/2.8",

    # 22
    "the person in reference image 1 through 4, with right side from reference image 4, "
    "waist-up, wearing a bright Hawaiian camp shirt, leaning casually against a railing, "
    "three-quarter view, broad sun-squinting smile, bright tropical sunlight, beach background, "
    "35mm lens, f/4.0",

    # 23
    "the person in reference image 1 through 4, with left side from reference image 3, "
    "waist-up, wearing a chef's apron over a plain linen shirt, arms slightly forward, "
    "three-quarter view from left, focused engaged expression, warm overhead kitchen lighting, "
    "50mm lens, f/2.8",

    # 24
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "waist-up, wearing a vintage band t-shirt, hands relaxed in pockets, "
    "front facing, cool understated expression, dramatic concert-style side lighting, "
    "50mm lens, f/2.8",

    # 25
    "the person in reference image 1 through 4, with left profile from reference image 3, "
    "waist-up, wearing a cable-knit cardigan, holding an open book, "
    "direct profile from left, peaceful reading expression, soft warm lamp light, cozy living room background, "
    "85mm lens, f/2.0",

    # 26
    "the person in reference image 1 through 4, with right side from reference image 4 and face from reference image 2, "
    "waist-up, wearing business casual — chinos and a collared shirt, "
    "three-quarter view, seated at a desk with hands folded, confident professional expression, "
    "warm office window light, blurred open-plan office background, 50mm lens, f/2.0",

    # 27
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "waist-up, wearing a slim-fit bomber jacket, "
    "front facing, casual cool stance, one hand in pocket, "
    "warm amber evening street light, blurred urban street background, 85mm lens, f/1.8",

    # 28
    "the person in reference image 1 through 4, with left side from reference image 3, "
    "waist-up, wearing gym workout gear — fitted tank top and shorts, "
    "three-quarter view, arms slightly raised showing effort, determined sweaty expression, "
    "harsh fluorescent gym lighting, blurred gym equipment background, 50mm lens, f/2.8",

    # 29
    "the person in reference image 1 through 4, with front view from reference image 1 and face from reference image 2, "
    "waist-up, wearing a cozy oversized university hoodie, "
    "front facing, sleepy relaxed expression, soft morning window light, blurred apartment living room, "
    "50mm lens, f/2.0",

    # 30
    "the person in reference image 1 through 4, with right side from reference image 4, "
    "waist-up, wearing a crisp white lab coat over a shirt, "
    "three-quarter view, focused analytical expression, bright clinical fluorescent lighting, "
    "blurred laboratory background, 50mm lens, f/4.0",

    # --- Full body with different outfits (31-45) ---

    # 31
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body, wearing a tailored wool overcoat and wool scarf, "
    "walking confidently at three-quarter view, slight easy smile, cold winter daylight, snowy street background, "
    "35mm lens, f/4.0",

    # 32
    "the person in reference image 1 through 4, with front view from reference image 1 and fullbody from reference image 3, "
    "full body, wearing shorts and a casual tank top, "
    "standing relaxed front-facing, happy open smile, bright summer sunlight, poolside background, "
    "35mm lens, f/5.6",

    # 33
    "the person in reference image 1 through 4, with left fullbody from reference image 3, "
    "full body, wearing running shoes and athletic tracksuit, "
    "mid-stride jogging pose, side profile from left, focused expression, early morning light, park trail, "
    "35mm lens, f/4.0, slight motion blur on feet",

    # 34
    "the person in reference image 1 through 4, with right fullbody from reference image 4, "
    "full body, wearing a long flowing evening dress or sharp formal suit, "
    "standing elegantly at three-quarter view from right, composed poised expression, "
    "grand hall interior with chandeliers, 35mm lens, f/4.0",

    # 35
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "full body, wearing dark slim jeans and a plain white shirt, "
    "sitting on stone steps, front facing, relaxed warm smile, "
    "golden hour light, old stone building exterior, 35mm lens, f/3.5",

    # 36
    "the person in reference image 1 through 4, with left fullbody from reference image 3, "
    "full body, wearing casual streetwear — jogger pants and a graphic hoodie, "
    "leaning against a graffiti mural wall, relaxed cool pose, "
    "overcast diffuse light, urban street background, 35mm lens, f/3.5",

    # 37
    "the person in reference image 1 through 4, with right fullbody from reference image 4 and front from reference image 1, "
    "full body, wearing a smart business suit and carrying a briefcase, "
    "walking purposefully toward camera, confident neutral expression, "
    "morning city sunlight, blurred downtown office building background, 35mm lens, f/4.0",

    # 38
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body, wearing cargo pants and a utility vest over a long-sleeve shirt, "
    "crouching in an action-ready pose, alert scanning expression, "
    "moody overcast industrial background, 35mm lens, f/4.0",

    # 39
    "the person in reference image 1 through 4, with left fullbody from reference image 3, "
    "full body, wearing a bathrobe, standing casually in a doorway, "
    "three-quarter view from left, sleepy warm expression, soft morning window light, cozy bedroom background, "
    "35mm lens, f/2.8",

    # 40
    "the person in reference image 1 through 4, with right fullbody from reference image 4, "
    "full body, wearing hiking boots, cargo trousers, and a fleece jacket, "
    "standing on a mountain trail, looking into the distance, "
    "bright mountain sunlight, rocky mountain peak background, 35mm lens, f/5.6",

    # 41
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body, wearing swim shorts or a swimsuit, "
    "standing at the edge of a swimming pool, relaxed happy pose, "
    "bright tropical midday sunlight, blurred poolside background, 35mm lens, f/5.6",

    # 42
    "the person in reference image 1 through 4, with left fullbody from reference image 3 and front from reference image 1, "
    "full body, wearing a leather jacket and dark jeans, boots, "
    "standing on a rooftop at golden hour, arms relaxed, confident expression, "
    "warm golden backlight, cityscape background, 35mm lens, f/2.8",

    # 43
    "the person in reference image 1 through 4, with right fullbody from reference image 4 and face from reference image 2, "
    "full body, wearing yoga wear — fitted leggings and a sports top, "
    "doing a standing balance yoga pose, arms extended, serene expression, "
    "soft studio yoga-room lighting, clean white background, 35mm lens, f/4.0",

    # 44
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body, wearing a warm knit sweater, corduroy trousers, and ankle boots, "
    "standing in a snow-covered park, happy expression, snow falling, "
    "cold diffuse winter daylight, snowy trees background, 35mm lens, f/4.0",

    # 45
    "the person in reference image 1 through 4, with left fullbody from reference image 3, "
    "full body, wearing paint-splattered overalls and a plain t-shirt, "
    "holding a large paintbrush, creative focused expression, "
    "natural skylight, bright art studio background with canvases, 35mm lens, f/3.5",

    # --- Mixed specialty with different contexts (46-60) ---

    # 46
    "the person in reference image 1 through 4, with face from reference image 2 and left side from reference image 3, "
    "head and shoulders, wearing safety goggles pushed up on forehead and a sturdy work shirt, "
    "three-quarter view from left, proud satisfied smile, "
    "bright workshop fluorescent lighting, blurred garage workshop background, 50mm lens, f/2.8",

    # 47
    "the person in reference image 1 through 4, with full body from reference image 4, "
    "waist-up, wearing a backpack and technical hiking gear, "
    "looking into the distance dramatically, adventurous determined expression, "
    "bright mountain sunlight, mountain peak panorama background, 35mm lens, f/5.6",

    # 48
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "close-up portrait, wearing round wire-rimmed glasses and a wool scarf, "
    "front facing, warm intellectual smile, "
    "soft autumn afternoon side light, blurred fall foliage background, 85mm lens, f/1.8",

    # 49
    "the person in reference image 1 through 4, with front view from reference image 1 and face from reference image 2, "
    "waist-up, wearing paint-splattered art smock, holding a fine paintbrush, "
    "focused creative expression, "
    "natural skylight from above, bright art studio background, 50mm lens, f/2.8",

    # 50
    "the person in reference image 1 through 4, with left side from reference image 3 and face from reference image 2, "
    "head and shoulders, wearing a crisp white chef jacket, "
    "three-quarter view from left, proud accomplished expression, "
    "warm overhead kitchen lighting, blurred professional kitchen background, 50mm lens, f/2.8",

    # 51
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "waist-up, wearing a barista apron over a casual shirt, "
    "front facing, cheerful welcoming smile, "
    "warm golden cafe lighting, blurred espresso machine and cafe shelves background, 50mm lens, f/2.0",

    # 52
    "the person in reference image 1 through 4, with right fullbody from reference image 4 and left from reference image 3, "
    "full body, wearing a graduation gown and cap, "
    "standing proudly front-facing, big celebratory smile, "
    "bright outdoor graduation ceremony daylight, blurred green campus background, 35mm lens, f/4.0",

    # 53
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "close-up portrait, wearing boxing wraps on hands, athletic tank top visible at shoulders, "
    "fierce determined expression, "
    "harsh directional gym spotlight, dark gym background, 85mm lens, f/2.0",

    # 54
    "the person in reference image 1 through 4, with left fullbody from reference image 3 and front from reference image 1, "
    "full body, wearing a vintage varsity jacket over a t-shirt, jeans, and sneakers, "
    "relaxed cool stance, leaning on a classic car, "
    "warm late-afternoon sunlight, blurred retro diner background, 35mm lens, f/3.5",

    # 55
    "the person in reference image 1 through 4, with front view from reference image 1 and face from reference image 2, "
    "head and shoulders, wearing a motorcycle jacket with patches, "
    "front facing, bold direct stare at camera, "
    "hard side flash photography, dark concrete wall background, 85mm lens, f/4.0",

    # 56
    "the person in reference image 1 through 4, with right side from reference image 4, "
    "waist-up, wearing medical scrubs top, stethoscope around neck, "
    "three-quarter view, compassionate professional expression, "
    "bright clinical hospital corridor lighting, blurred medical background, 50mm lens, f/3.5",

    # 57
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body, wearing an elegant evening outfit — cocktail dress or suit, "
    "standing at a rooftop bar, one hand resting on railing, "
    "golden dusk light from the skyline, city lights bokeh background, 35mm lens, f/2.8",

    # 58
    "the person in reference image 1 through 4, with face from reference image 2 and left side from reference image 3, "
    "close-up portrait, wearing a firefighter jacket with reflective stripes, helmet in hand, "
    "three-quarter view from left, stoic heroic expression, "
    "dramatic golden backlight, blurred smoke-tinged background, 85mm lens, f/2.0",

    # 59
    "the person in reference image 1 through 4, with right fullbody from reference image 4 and front from reference image 1, "
    "full body, wearing dark tactical cargo trousers and a fitted long-sleeve top, "
    "dynamic crouching action pose, tense alert expression, "
    "blue-gray industrial warehouse lighting, dark concrete background, 35mm lens, f/3.5",

    # 60
    "the person in reference image 1 through 4, with face details from reference image 2 and front from reference image 1, "
    "close-up portrait, wearing a cozy Christmas sweater, "
    "bright happy festive smile, "
    "warm holiday fairy-light bokeh background, cozy indoor Christmas setting, 85mm lens, f/1.8",

    # --- Final specialty mixed (61-75) ---

    # 61
    "the person in reference image 1 through 4, with left fullbody from reference image 3 and face from reference image 2, "
    "full body, wearing a sleek wetsuit, surfboard under arm, "
    "looking at the ocean with excited anticipation, "
    "bright golden morning beach light, blurred ocean waves background, 35mm lens, f/5.6",

    # 62
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "waist-up, wearing a striped Breton marinière top, "
    "front facing, relaxed intellectual expression, "
    "soft natural light, blurred coastal dock background, 50mm lens, f/2.0",

    # 63
    "the person in reference image 1 through 4, with right side from reference image 4 and face from reference image 2, "
    "head and shoulders, wearing denim overalls with a fitted long-sleeve underneath, "
    "three-quarter view from right, playful cheeky expression, "
    "bright summer park light, blurred floral garden background, 85mm lens, f/2.0",

    # 64
    "the person in reference image 1 through 4, with full body from reference image 3 and reference image 4, "
    "full body, wearing a warm knit sweater, corduroy trousers, and ankle boots, "
    "walking on a leaf-covered autumn path, coat slightly billowing, "
    "warm golden autumn light, blurred forest fall foliage, 35mm lens, f/3.5",

    # 65
    "the person in reference image 1 through 4, with face details from reference image 2, "
    "close-up portrait, wearing a military-style olive green jacket, collar up, "
    "front facing, stoic serious expression, "
    "cool overcast flat light, blurred urban concrete background, 85mm lens, f/2.0",

    # 66
    "the person in reference image 1 through 4, with left side from reference image 3 and front from reference image 1, "
    "head and shoulders, wearing a cozy cable-knit fisherman sweater, "
    "three-quarter view from left, warm contented smile, "
    "soft Nordic window light, blurred snowy landscape outside window, 85mm lens, f/2.0",

    # 67
    "the person in reference image 1 through 4, with right fullbody from reference image 4, "
    "full body, wearing a bright yellow rain slicker and rain boots, "
    "standing in rain, laughing with head up, "
    "gray rainy overcast light, wet city street background with puddle reflections, 35mm lens, f/4.0",

    # 68
    "the person in reference image 1 through 4, with front view from reference image 1 and face from reference image 2, "
    "waist-up, wearing a smart double-breasted peacoat, gloved hands visible, "
    "front facing, polished composed expression, "
    "cold winter morning sun, blurred cobblestone European city street background, 50mm lens, f/2.8",

    # 69
    "the person in reference image 1 through 4, with left fullbody from reference image 3, "
    "full body, wearing track pants and a loose athletic hoodie, headphones around neck, "
    "relaxed urban streetwear pose leaning on a fence, "
    "late afternoon golden side light, blurred urban park background, 35mm lens, f/3.5",

    # 70
    "the person in reference image 1 through 4, with face from reference image 2 and right side from reference image 4, "
    "close-up portrait, wearing a silk or satin collared blouse or dress shirt, "
    "three-quarter view from right, sophisticated poised expression, "
    "elegant studio beauty lighting, seamless dark-green gradient background, 85mm lens, f/1.8",

    # 71
    "the person in reference image 1 through 4, with full body from reference image 4 and front from reference image 1, "
    "full body, wearing a dark tuxedo or gala gown, "
    "standing at the base of a marble staircase, "
    "elegant composed expression, grand event lighting with chandeliers, 35mm lens, f/4.0",

    # 72
    "the person in reference image 1 through 4, with left side from reference image 3 and face from reference image 2, "
    "head and shoulders, wearing a kimono-inspired wrap top, "
    "three-quarter view from left, serene contemplative expression, "
    "soft diffuse natural light, blurred Japanese garden background, 85mm lens, f/2.0",

    # 73
    "the person in reference image 1 through 4, with front view from reference image 1, "
    "waist-up, wearing a bright orange safety vest over a work shirt, hard hat under arm, "
    "front facing, confident workman expression, "
    "bright outdoor construction site daylight, blurred construction background, 50mm lens, f/4.0",

    # 74
    "the person in reference image 1 through 4, with right fullbody from reference image 4 and face from reference image 2, "
    "full body, wearing a classic trench coat, belt cinched, "
    "walking in light rain with umbrella, three-quarter view from right, "
    "moody gray rainy light, blurred wet city street, 35mm lens, f/2.8",

    # 75
    "the person in reference image 1 through 4, with face details from reference image 2 and left side from reference image 3, "
    "close-up portrait, wearing a warm terracotta-colored linen shirt, "
    "three-quarter view from left, relaxed easy confidence, "
    "golden Mediterranean afternoon light, blurred whitewashed terrace background, 85mm lens, f/1.8",
]

assert len(KLEIN_ORIGINAL_OUTFIT) == 75, f"Expected 75, got {len(KLEIN_ORIGINAL_OUTFIT)}"
assert len(KLEIN_VARIED_OUTFIT) == 75, f"Expected 75, got {len(KLEIN_VARIED_OUTFIT)}"


def get_prompt_templates_klein(num_images: int = 25) -> list[str]:
    """Return ``num_images`` prompt strings for Klein 9B KV (4 reference images).

    Same 50/50 split between original and varied outfits as the Flux 2 DEV
    templates, but references only 4 images (front, face close-up, left, right).

    When ``num_images`` exceeds 150, prompts cycle with variation suffixes.
    """
    if num_images <= 0:
        raise ValueError(f"num_images must be positive, got {num_images}")

    half = num_images // 2
    other_half = num_images - half

    orig = _pick_from(KLEIN_ORIGINAL_OUTFIT, half)
    varied = _pick_from(KLEIN_VARIED_OUTFIT, other_half)
    return _interleave(orig, varied)
