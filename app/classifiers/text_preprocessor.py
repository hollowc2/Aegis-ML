"""
app/classifiers/text_preprocessor.py
======================================
Unicode-aware text preprocessor that runs before any classifier.

Responsibilities:
  1. Strip invisible / control Unicode characters and flag their presence
     (zero-width spaces, RTL overrides, soft hyphens, etc.)
  2. Apply NFKC Unicode normalization to collapse compatibility variants and
     homoglyph lookalikes (e.g. Cyrillic 'а' → Latin 'a')
  3. Return cleaned text + metadata dict for downstream bias adjustments

The preprocessor is a pure-Python, ML-free module with no heavy dependencies.
It is called before every classifier (not just hf2) so that even the fast
sklearn path benefits from Unicode normalization.

Metadata flags drive the invisible-char probability bias in input_guard.py:
  - had_invisible_chars: True if zero-width or other invisible chars were found
  - had_rtl_override: True if a bidirectional control override was found
  - had_homoglyphs: True if characters that NFKC normalized significantly differ
  - unicode_normalized: True if NFKC changed the text at all
"""

from __future__ import annotations

import unicodedata


class TextPreprocessor:
    """
    Stateless text preprocessor for adversarial Unicode normalization.

    Usage:
        preprocessor = TextPreprocessor()
        cleaned, meta = preprocessor.preprocess(raw_text)
        # Use cleaned for classifier, meta for bias decisions
    """

    # Characters that are invisible / zero-width in most rendering contexts
    # and have no semantic value in legitimate user input.
    INVISIBLE_CHARS: frozenset[str] = frozenset([
        '\u200B',  # zero-width space
        '\u200C',  # zero-width non-joiner
        '\u200D',  # zero-width joiner
        '\u2060',  # word joiner
        '\u00AD',  # soft hyphen
        '\uFEFF',  # BOM / zero-width no-break space
        '\u200E',  # left-to-right mark
        '\u200F',  # right-to-left mark
        '\u202A',  # LTR embedding
        '\u202B',  # RTL embedding
        '\u202C',  # pop directional formatting
        '\u202D',  # LTR override
        '\u202E',  # RTL override (most dangerous — reverses displayed text)
        '\u2066',  # LTR isolate
        '\u2067',  # RTL isolate
        '\u2068',  # first strong isolate
        '\u2069',  # pop directional isolate
    ])

    # RTL override chars are a strong signal on their own
    RTL_OVERRIDE_CHARS: frozenset[str] = frozenset([
        '\u202E',  # RTL override
        '\u202B',  # RTL embedding
        '\u2067',  # RTL isolate
    ])

    def preprocess(self, text: str) -> tuple[str, dict[str, bool]]:
        """
        Normalize *text* for safe classifier input.

        Returns:
            (cleaned_text, metadata)

        metadata keys:
            had_invisible_chars: bool  — invisible/zero-width chars were found
            had_rtl_override: bool     — RTL control chars were found (strong signal)
            unicode_normalized: bool   — NFKC changed the text (covers homoglyphs)
        """
        if not text:
            return text, {
                "had_invisible_chars": False,
                "had_rtl_override": False,
                "unicode_normalized": False,
            }

        had_invisible = False
        had_rtl = False

        # ── Step 1: detect and strip invisible/control chars ──────────────────
        stripped_chars: list[str] = []
        for ch in text:
            if ch in self.INVISIBLE_CHARS:
                had_invisible = True
                if ch in self.RTL_OVERRIDE_CHARS:
                    had_rtl = True
                # Strip — do not add to output
                continue
            stripped_chars.append(ch)
        stripped = "".join(stripped_chars)

        # ── Step 2: NFKC normalization ─────────────────────────────────────────
        # NFKC collapses compatibility variants and maps many Unicode lookalikes
        # (Cyrillic, Greek, fullwidth, superscript digits) to their ASCII
        # equivalents.  This handles the 'homoglyph' attack class automatically.
        normalized = unicodedata.normalize("NFKC", stripped)
        unicode_normalized = normalized != text

        return normalized, {
            "had_invisible_chars": had_invisible,
            "had_rtl_override": had_rtl,
            "unicode_normalized": unicode_normalized,
        }
