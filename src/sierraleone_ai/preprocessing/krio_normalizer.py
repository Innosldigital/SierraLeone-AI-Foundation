"""
Krio text normalisation (SLAI-016 / SLAI-030).

Sierra Leonean Krio is a distinct English-based creole language.
It is NOT Nigerian Pidgin, Liberian English, or Cameroon Pidgin.
This normaliser handles Krio-specific orthography, Unicode repair,
and preprocessing steps for language model training.

Reference:
  - Fyle, C.M. & Jones, E.D. (1980). A Krio-English Dictionary. Oxford University Press / Sierra Leone University Press.

NOTE: The cross-creole overlap marker list below is a heuristic diagnostic aid
(`reviewed_by_native=False`), NOT an automatic rejection rule. Flagged records
set `needs_native_review=True` and `possible_cross_creole_overlap`, while
never setting `rejected=True` and always preserving `original` text.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

import ftfy

# ---------------------------------------------------------------------------
# Common Krio spelling normalisation table
# (must be reviewed and approved by a native Krio speaker panel before enabling)
# ---------------------------------------------------------------------------
SPELLING_VARIANTS_REVIEWED_BY_NATIVE: bool = False

KRIO_SPELLING_VARIANTS: dict[str, str] = {
    # Identity placeholders only – apply_spelling_variants=True will raise ValueError
    # until SPELLING_VARIANTS_REVIEWED_BY_NATIVE is True and reviewed mappings exist.
    "aw": "aw",
    "una": "una",
    "na": "na",
}

# ---------------------------------------------------------------------------
# Cross-creole overlap heuristic markers (SLAI-016 / SLAI-030)
# ---------------------------------------------------------------------------
# Heuristic, incomplete, and unreviewed by a native panel (`reviewed_by_native=False`).
# NEVER use these markers to auto-discard or auto-reject a record.
HARD_CREOLE_MARKERS: dict[str, str] = {
    "abeg": "Possible West African Pidgin overlap (verify with native SL Krio panel)",
    "oyibo": "Possible Nigerian Pidgin overlap (SL Krio commonly uses 'potho' / 'waitman')",
    "shey": "Possible Nigerian Pidgin question particle (SL Krio commonly uses 'noto so' / 'ɛnti')",
    "chale": "Possible Ghanaian Pidgin overlap (SL Krio commonly uses 'padi' / 'bo')",
}

SOFT_CREOLE_MARKERS: dict[str, str] = {
    "abi": "Possible cross-creole tag particle (needs native review)",
    "comot": "Possible non-standard orthography variant for 'kɔmɔt' (needs native review)",
    "my guy": "Possible colloquialism overlap (needs native review)",
}

NON_KRIO_CREOLE_MARKERS: dict[str, str] = {**HARD_CREOLE_MARKERS, **SOFT_CREOLE_MARKERS}

CREOLE_MARKER_METADATA: dict[str, dict[str, Any]] = {
    marker: {
        "tier": "hard" if marker in HARD_CREOLE_MARKERS else "soft",
        "note": note,
        "reviewed_by_native": False,
    }
    for marker, note in NON_KRIO_CREOLE_MARKERS.items()
}

# Characters to strip (control characters, BOM, etc.)
_STRIP_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff]")

# Collapse multiple whitespace into single space
_WHITESPACE_PATTERN = re.compile(r"[ \t]+")


def detect_non_krio_creole_markers(
    text: str,
    include_soft: bool = True,
) -> list[str]:
    """
    Scan text for heuristic lexical markers that may overlap with Nigerian Pidgin,
    Liberian English, or Ghanaian Pidgin.

    Returns a list of diagnostic notes for human native-speaker review.
    Never used to auto-reject records (SLAI-030).
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
    Analyze Krio orthography characteristics (SLAI-016).

    Counts Sierra Leone standard IPA-derived open vowels:
      - open-e: ɛ (U+025B) / Ɛ (U+0190)
      - open-o: ɔ (U+0254) / Ɔ (U+0186)
    versus plain Latin vowels (`e`/`o` and total `a e i o u`) without auto-converting.
    """
    normalized = unicodedata.normalize("NFC", text)
    open_e = len(re.findall(r"[ɛƐ]", normalized))
    open_o = len(re.findall(r"[ɔƆ]", normalized))
    plain_e = len(re.findall(r"[eE]", normalized))
    plain_o = len(re.findall(r"[oO]", normalized))
    total_open = open_e + open_o
    latin_vowels = len(re.findall(r"[aeiouAEIOU]", normalized))

    total_vowels = latin_vowels + total_open
    open_ratio = (total_open / total_vowels) if total_vowels > 0 else 0.0

    return {
        "open_e_count": open_e,
        "open_o_count": open_o,
        "plain_e_count": plain_e,
        "plain_o_count": plain_o,
        "total_open_vowels": total_open,
        "standard_orthography_ratio": round(open_ratio, 4),
    }


@dataclass
class NormalisationConfig:
    """Configuration for the Krio text normaliser."""

    fix_unicode: bool = True          # run ftfy unicode repair with explanation tracking
    lowercase: bool = False           # do NOT lowercase by default – Krio uses case
    strip_control_chars: bool = True
    normalise_whitespace: bool = True
    apply_spelling_variants: bool = False  # only after native-speaker review
    check_creole_contamination: bool = True
    max_length: int = 4096            # characters
    on_overlong: Literal["truncate", "reject"] = "reject"  # SLAI-016: default to flag-and-reject


@dataclass
class NormalisationResult:
    original: str
    normalised: str
    changes: list[str] = field(default_factory=list)
    possible_cross_creole_overlap: list[str] = field(default_factory=list)
    needs_native_review: bool = False
    truncated: bool = False
    rejected: bool = False
    rejection_reason: str | None = None

    @property
    def flagged_non_krio_markers(self) -> list[str]:
        """Backward-compatible alias for `possible_cross_creole_overlap` (SLAI-030)."""
        return self.possible_cross_creole_overlap


class KrioNormalizer:
    """
    Normalises Sierra Leonean Krio text for language model training.

    Usage::

        normalizer = KrioNormalizer()
        result = normalizer.normalise("TEST  FIXTURE  TEXT.")
        print(result.normalised)  # "TEST FIXTURE TEXT."
    """

    def __init__(self, config: NormalisationConfig | None = None) -> None:
        self.config = config or NormalisationConfig()

    def normalise(self, text: str) -> NormalisationResult:
        """Normalise a single Krio text string while preserving `result.original`."""
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

        # 1. Unicode repair (ftfy.fix_and_explain records exact transformations: SLAI-016)
        if cfg.fix_unicode:
            fixed, explanation = ftfy.fix_and_explain(text)
            if fixed != text:
                result.changes.append("unicode_repaired")
                if explanation:
                    for step in explanation:
                        action = step[0] if isinstance(step, tuple) and step else str(step)
                        result.changes.append(f"ftfy:{action}")
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

        # 6. Optional spelling variant normalisation (SLAI-016 guard)
        if cfg.apply_spelling_variants:
            active_variants = {
                k: v for k, v in KRIO_SPELLING_VARIANTS.items() if k != v
            }
            if not SPELLING_VARIANTS_REVIEWED_BY_NATIVE or not active_variants:
                raise ValueError(
                    "apply_spelling_variants=True is disabled because KRIO_SPELLING_VARIANTS "
                    "contains only unreviewed placeholders. A native Krio speaker panel must "
                    "review and approve spelling mappings first."
                )
            for variant, standard in active_variants.items():
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

        # 8. Heuristic cross-creole overlap check (SLAI-030: never auto-rejects!)
        if cfg.check_creole_contamination:
            markers = detect_non_krio_creole_markers(text)
            if markers:
                result.possible_cross_creole_overlap = markers
                result.needs_native_review = True
                result.changes.append("possible_cross_creole_overlap")
                result.changes.append("flagged_non_krio_creole_markers")

        result.normalised = text
        return result

    def normalise_batch(self, texts: list[str]) -> list[NormalisationResult]:
        """Normalise a list of texts."""
        return [self.normalise(t) for t in texts]
