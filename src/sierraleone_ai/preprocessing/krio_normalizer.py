"""
Krio text normalisation.

Sierra Leonean Krio is a distinct English-based creole language.
It is NOT Nigerian Pidgin, Liberian English, or Cameroon Pidgin.
This normaliser handles Krio-specific orthography, common spelling
variants, and preprocessing steps for language model training.

References:
  - Fyle, C.M. & Jones, E.D. (1980). Krio-English Dictionary. Oxford University Press.
  - Sengova, J. (1987). The National Languages of Sierra Leone. Africana Journal.

NOTE: This is a scaffold. Native speaker review is required before
      using normalised output as training data.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

import ftfy

# ---------------------------------------------------------------------------
# Common Krio spelling normalisation table
# (must be reviewed and extended by a native Krio speaker)
# ---------------------------------------------------------------------------
KRIO_SPELLING_VARIANTS: dict[str, str] = {
    # These are illustrative placeholders only – do NOT treat as authoritative
    # until reviewed by native speakers.
    "aw": "aw",        # Krio pronoun 'we' – already standard
    "una": "una",      # second-person plural – already standard
    "na": "na",        # copula / locative – already standard
}

# ---------------------------------------------------------------------------
# Non-Krio creole markers (SLAI-016: categorised by diagnostic confidence)
# ---------------------------------------------------------------------------
# HARD markers: very strong indication of non-Sierra Leonean origin (Nigerian / Ghanaian / Liberian)
HARD_CREOLE_MARKERS: dict[str, str] = {
    "abeg": "Nigerian Pidgin ('please' – SL Krio uses 'duya' / 'pliz')",
    "oyibo": "Nigerian Pidgin (SL Krio uses 'potho' / 'waitman')",
    "shey": "Nigerian Pidgin question particle (SL Krio uses 'noto so' / 'ɛnti')",
    "chale": "Ghanaian Pidgin ('friend' – SL Krio uses 'padi' / 'bo')",
}

# SOFT markers: colloquialisms or spelling variants that warrant human review
SOFT_CREOLE_MARKERS: dict[str, str] = {
    "abi": "Nigerian Pidgin tag particle",
    "comot": "Nigerian Pidgin spelling (SL Krio standard orthography: 'kɔmɔt')",
    "my guy": "Liberian/Nigerian Pidgin colloquialism",
}

NON_KRIO_CREOLE_MARKERS: dict[str, str] = {**HARD_CREOLE_MARKERS, **SOFT_CREOLE_MARKERS}

# Characters to strip (control characters, BOM, etc.)
_STRIP_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff]")

# Collapse multiple whitespace into single space
_WHITESPACE_PATTERN = re.compile(r"[ \t]+")


def detect_non_krio_creole_markers(
    text: str,
    include_soft: bool = True,
) -> list[str]:
    """
    Scan text for lexical markers from Nigerian Pidgin, Liberian English,
    or Ghanaian Pidgin to prevent cross-creole contamination in Krio datasets.
    """
    markers_to_check = NON_KRIO_CREOLE_MARKERS if include_soft else HARD_CREOLE_MARKERS
    flagged: list[str] = []
    for marker, reason in markers_to_check.items():
        pattern = re.compile(r"\b" + re.escape(marker) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            flagged.append(f"{marker}: {reason}")
    return flagged


def orthography_profile(text: str) -> dict[str, int | float]:
    """
    Analyze Krio orthography characteristics.

    Counts Sierra Leone standard IPA-derived open vowels:
      - open-e: ɛ (U+025B) / Ɛ (U+0190)
      - open-o: ɔ (U+0254) / Ɔ (U+0186)
    and computes the open-vowel adoption metric.
    """
    normalized = unicodedata.normalize("NFC", text)
    open_e = len(re.findall(r"[ɛƐ]", normalized))
    open_o = len(re.findall(r"[ɔƆ]", normalized))
    total_open = open_e + open_o
    latin_vowels = len(re.findall(r"[aeiouAEIOU]", normalized))

    total_vowels = latin_vowels + total_open
    open_ratio = (total_open / total_vowels) if total_vowels > 0 else 0.0

    return {
        "open_e_count": open_e,
        "open_o_count": open_o,
        "total_open_vowels": total_open,
        "standard_orthography_ratio": round(open_ratio, 4),
    }


@dataclass
class NormalisationConfig:
    """Configuration for the Krio text normaliser."""

    fix_unicode: bool = True          # run ftfy unicode repair
    lowercase: bool = False           # do NOT lowercase by default – Krio uses case
    strip_control_chars: bool = True
    normalise_whitespace: bool = True
    apply_spelling_variants: bool = False  # only after native-speaker review
    check_creole_contamination: bool = True
    max_length: int = 4096            # characters
    on_overlong: Literal["truncate", "reject"] = "truncate"  # action when text exceeds max_length


@dataclass
class NormalisationResult:
    original: str
    normalised: str
    changes: list[str] = field(default_factory=list)
    flagged_non_krio_markers: list[str] = field(default_factory=list)
    truncated: bool = False
    rejected: bool = False
    rejection_reason: str | None = None


class KrioNormalizer:
    """
    Normalises Sierra Leonean Krio text for language model training.

    Usage::

        normalizer = KrioNormalizer()
        result = normalizer.normalise("Mi  de go  maket.")
        print(result.normalised)  # "Mi de go maket."
    """

    def __init__(self, config: NormalisationConfig | None = None) -> None:
        self.config = config or NormalisationConfig()

    def normalise(self, text: str) -> NormalisationResult:
        """Normalise a single Krio text string."""
        result = NormalisationResult(original=text, normalised=text)
        cfg = self.config

        # Check overlong rejection early if configured
        if cfg.max_length and len(text) > cfg.max_length:
            if cfg.on_overlong == "reject":
                result.rejected = True
                result.rejection_reason = (
                    f"Text length {len(text)} exceeds max_length {cfg.max_length}"
                )
                result.changes.append("rejected_overlong")
                return result

        # 1. Unicode repair (ftfy handles mojibake, ligatures, etc.)
        if cfg.fix_unicode:
            fixed = ftfy.fix_text(text)
            if fixed != text:
                result.changes.append("unicode_repaired")
            text = fixed

        # 2. Unicode normalisation to NFC (composed form)
        text_nfc = unicodedata.normalize("NFC", text)
        if text_nfc != text:
            result.changes.append("nfc_normalised")
        text = text_nfc

        # 3. Strip control characters
        if cfg.strip_control_chars:
            cleaned = _STRIP_PATTERN.sub("", text)
            if cleaned != text:
                result.changes.append("control_chars_stripped")
            text = cleaned

        # 4. Normalise whitespace
        if cfg.normalise_whitespace:
            normalised_ws = _WHITESPACE_PATTERN.sub(" ", text).strip()
            if normalised_ws != text:
                result.changes.append("whitespace_normalised")
            text = normalised_ws

        # 5. Optional lowercasing (disabled by default for Krio)
        if cfg.lowercase:
            lowered = text.lower()
            if lowered != text:
                result.changes.append("lowercased")
            text = lowered

        # 6. Optional spelling variant normalisation
        if cfg.apply_spelling_variants:
            if not KRIO_SPELLING_VARIANTS:
                raise ValueError(
                    "apply_spelling_variants=True but KRIO_SPELLING_VARIANTS is empty. "
                    "Review and populate standard spelling mappings first."
                )
            for variant, standard in KRIO_SPELLING_VARIANTS.items():
                pattern = re.compile(r"\b" + re.escape(variant) + r"\b", re.IGNORECASE)
                new_text, n = pattern.subn(standard, text)
                if n > 0:
                    result.changes.append(f"variant_normalised:{variant}→{standard}")
                text = new_text

        # 7. Length check and truncation if on_overlong == "truncate"
        if cfg.max_length and len(text) > cfg.max_length:
            text = text[: cfg.max_length]
            result.truncated = True
            result.changes.append(f"truncated_at_{cfg.max_length}")

        # 8. Cross-creole contamination check (Nigerian Pidgin / Liberian English / Ghanaian Pidgin)
        if cfg.check_creole_contamination:
            markers = detect_non_krio_creole_markers(text)
            if markers:
                result.flagged_non_krio_markers = markers
                result.changes.append("flagged_non_krio_creole_markers")

        result.normalised = text
        return result

    def normalise_batch(self, texts: list[str]) -> list[NormalisationResult]:
        """Normalise a list of texts."""
        return [self.normalise(t) for t in texts]
