"""
IdentityStripper — removes physical and demographic identity descriptors from
image captions so those traits bind to the trigger token during LoRA training
rather than being learned as general scene features.
"""

from __future__ import annotations

import regex as re

# ---------------------------------------------------------------------------
# Default identity patterns
# ---------------------------------------------------------------------------
# Each pattern describes WHO the character is (appearance, demographics, body).
# These are removed so the LoRA associates the trigger word with the identity.
#
# EXPRESSIONS ARE INTENTIONALLY PRESERVED — DO NOT ADD THEM HERE.
# Words like "smiling", "laughing", "grinning", "serious", "neutral",
# "contemplative", "surprised", "happy", "sad", "angry", "frowning",
# "looking away", "eyes closed", "squinting", "smirking", "pensive",
# "thoughtful", "joyful", "excited", etc. are NOT identity features.
# They are transient states that vary between images and MUST remain in
# captions so the LoRA learns to respond to them rather than averaging
# all expressions into an unnatural neutral face.  If expressions are
# stripped, the model loses the ability to generate them on request.

IDENTITY_PATTERNS: list[str] = [
    # Hair — style+colour combos first (most specific), then colour-only, then standalone style
    # e.g. "long brown hair", "curly red hair", "straight black hair"
    r'\b(?:long|short|curly|straight|wavy|braided|messy|slicked|spiky)\s+'
    r'(?:black|white|gray|grey|brown|dark\s+brown|light\s+brown|blonde|blond|golden|auburn|red|ginger|silver|platinum|dark|light)\s+hair\b',
    # colour-only hair: "brown hair", "dark brown hair", "jet black hair"
    r'\b(?:jet\s+)?(?:black|white|gray|grey|brown|dark\s+brown|light\s+brown|blonde|blond|golden|auburn|red|ginger|silver|platinum|dark|light)\s+hair\b',
    # standalone style: "curly hair", "long hair", "bald"
    r'\b(?:long|short|curly|straight|wavy|braided|messy|slicked|spiky|bald)\s+hair\b',
    r'\bbald\b',
    r'\b(?:blonde|brunette|redhead)\b',

    # Eyes — specific shape patterns before general colour pattern
    r'\b(?:almond|round|narrow|wide)[\s-]+shaped\s+eyes?\b',
    r'\b(?:almond|round|narrow|wide)[\s-]+eyes?\b',
    # colour eyes: "green eyes", "dark brown eyes" — exclude innocent "her eyes", "his eyes"
    r'\b(?:black|blue|brown|dark\s+brown|gray|grey|green|hazel|amber|violet|pale|dark|light)\s+eyes?\b',

    # Skin
    r'\b(?:dark|light|pale|fair|olive|tan|tanned|brown|black|white)\s+skin(?:\s+tone)?\b',
    r'\bfreckl(?:es|ed)\b',
    r'\bdimples?\b',

    # Face structure
    r'\b(?:sharp|round|oval|square|angular|chiseled|soft)\s+(?:face|jaw|chin|features|cheekbones?)\b',
    r'\b(?:thin|full|thick|wide|narrow)\s+(?:lips?|nose|brow|eyebrows?|forehead)\b',
    r'\bhigh\s+cheekbones?\b',

    # Age / gender descriptors
    r'\b(?:young|old|elderly|middle-aged|teenage)\s+(?:woman|man|person|lady|guy|girl|boy)\b',
    r'\ba\s+(?:woman|man|person|lady|guy|girl|boy)\b',

    # Body type — require noun when adjective is ambiguous (e.g. "short" alone is too broad)
    r'\b(?:tall|slim|slender|stocky|heavyset|athletic|petite|muscular)\s+(?:build|figure|frame|body|woman|man|person|girl|boy|guy|lady)?\b',
    # "short" only when followed by stature nouns to avoid stripping "short sleeve"
    r'\bshort\s+(?:build|figure|frame|stature)\b',

    # Racial / ethnic descriptors
    r'\b(?:asian|caucasian|african|hispanic|latino|latina|east\s+asian|south\s+asian|middle\s+eastern)\b',

    # Facial hair
    r'\b(?:beard|goatee|mustache|moustache|stubble|clean-shaven|sideburns)\b',
]


class IdentityStripper:
    """
    Remove identity-describing traits from image captions.

    Captions produced by vision models often include physical descriptors
    (hair colour, eye colour, skin tone, body build, etc.) that should be
    attributed to the trigger word, not memorised as generic scene content.
    This class strips those descriptors so only pose, clothing, and environment
    remain in the text that the diffusion model trains on.

    Args:
        extra_patterns: Additional regex patterns (as raw strings) to strip
                        beyond the built-in :data:`IDENTITY_PATTERNS` set.

    Example::

        stripper = IdentityStripper()
        clean = stripper.process(
            "A young woman with long brown hair and green eyes sits reading.",
            trigger_word="ohwx person",
        )
        # -> "ohwx person, Sits reading."
    """

    def __init__(self, extra_patterns: list[str] | None = None) -> None:
        self.patterns: list[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in IDENTITY_PATTERNS
        ]
        if extra_patterns:
            self.patterns.extend(
                re.compile(p, re.IGNORECASE) for p in extra_patterns
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def strip(self, caption: str) -> str:
        """
        Remove identity traits from *caption* and clean up artifacts.

        Each compiled pattern is applied in order.  After all substitutions,
        the text is normalised:

        * Consecutive commas (with optional whitespace) are collapsed to one.
        * Leading and trailing commas/whitespace are removed.
        * Runs of two or more spaces are collapsed to a single space.
        * Hanging articles (``a``, ``an``) left at the end of the string
          after stripping are removed.
        * The first letter of the result is capitalised.

        Args:
            caption: Raw caption string to clean.

        Returns:
            Caption with identity descriptors removed and punctuation normalised.
        """
        result = caption

        for pattern in self.patterns:
            result = pattern.sub("", result)

        result = self._normalise(result)
        return result

    def prepend_trigger(self, caption: str, trigger_word: str) -> str:
        """
        Prepend *trigger_word* to *caption* with a comma separator.

        Args:
            caption: Already-cleaned caption text.
            trigger_word: LoRA trigger token (e.g. ``"ohwx person"``).

        Returns:
            ``"<trigger_word>, <Capitalised caption>"``
        """
        capitalised = caption[0].upper() + caption[1:] if caption else ""
        return f"{trigger_word}, {capitalised}"

    def process(self, caption: str, trigger_word: str) -> str:
        """
        Strip identity traits from *caption* and prepend *trigger_word*.

        Convenience wrapper that calls :meth:`strip` then
        :meth:`prepend_trigger`.

        Args:
            caption: Raw caption string (e.g. from Florence 2).
            trigger_word: LoRA trigger token to prepend.

        Returns:
            Cleaned, trigger-prepended caption ready for LoRA training.
        """
        cleaned = self.strip(caption)
        return self.prepend_trigger(cleaned, trigger_word)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        """
        Clean up punctuation and whitespace artifacts left after pattern removal.

        Steps applied in order:

        1. Remove connective words (``with``, ``and``, ``or``, ``but``, ``of``,
           ``a``, ``an``) that are left dangling between commas, at the start,
           or at the end of the string.
        2. Collapse multiple consecutive commas into one.
        3. Collapse runs of two or more spaces into a single space.
        4. Strip leading/trailing commas and whitespace.
        5. Capitalise the first character.

        Args:
            text: Intermediate string after regex substitutions.

        Returns:
            Normalised string.
        """
        # One compiled group for connective/filler words we want to drop when
        # they appear without meaningful content around them.
        # Each alternative is a full word — always used with \b anchors below.
        _CONN = r'(?:with|and|or|but|of|for|a|an)'

        for _ in range(8):  # iterate to dissolve chains like "with and or"
            prev = text

            # connectives sandwiched between two commas: ", and with ,"  ->  ","
            text = re.sub(
                r',\s*(?:\b' + _CONN + r'\b\s+)*\b' + _CONN + r'\b\s*,',
                ',', text, flags=re.IGNORECASE,
            )

            # connectives at string start: "with and foo"  ->  "foo"
            text = re.sub(
                r'^(?:\b' + _CONN + r'\b\s+)+',
                '', text, flags=re.IGNORECASE,
            )

            # connectives at string end (before optional punctuation): "foo and ."  ->  "foo."
            text = re.sub(
                r'(?:\s+\b' + _CONN + r'\b)+(\s*[\.!\?]?\s*)$',
                r'\1', text, flags=re.IGNORECASE,
            )

            # connectives before a comma or end-of-string: ", and,"  ->  ","
            text = re.sub(
                r',\s*(?:\b' + _CONN + r'\b\s+)*\b' + _CONN + r'\b\s*(?=,|$)',
                '', text, flags=re.IGNORECASE,
            )

            # connectives immediately after a comma before terminal punctuation:
            # "has a, and ."  ->  "has ."
            # Use \b anchors and require the connective to follow a comma or
            # a run of whitespace — never match mid-word.
            text = re.sub(
                r'(?:,\s*)(?:\b' + _CONN + r'\b\s+)*\b' + _CONN + r'\b\s*(?=\s*[\.!\?])',
                '', text, flags=re.IGNORECASE,
            )

            if text == prev:
                break

        # Collapse ",,", " , , " etc. into a single ", "
        text = re.sub(r'(?:\s*,\s*){2,}', ', ', text)

        # Collapse multiple spaces
        text = re.sub(r'  +', ' ', text)

        # Strip leading/trailing punctuation and whitespace
        text = text.strip(' ,')

        # Capitalise first character
        if text:
            text = text[0].upper() + text[1:]

        return text
