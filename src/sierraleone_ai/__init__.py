"""
Sierra Leone AI Foundation
==========================
An open research platform for Sierra Leonean language AI.

Languages targeted:
  - Krio (primary)
  - Mende
  - Temne
  - Limba
  - Loko
  - Kono
  - Kissi
  - Mandingo
  - Other Sierra Leonean languages

Status: Foundation phase – dataset infrastructure only.
No production model is operational yet.
"""

from __future__ import annotations

from typing import Literal, TypedDict

__version__ = "0.1.0"
__author__ = "CAFP Sierra Leone AI Team"
__license__ = "Apache-2.0"

LanguageState = Literal["planned", "active", "operational"]


class LanguageEntry(TypedDict, total=False):
    code: str
    name: str
    iso_639_3: str
    status: LanguageState
    note: str
    is_target_sl_language: bool


# Single source of truth for language metadata and operational state (SLAI-027 / SLAI-028)
LANGUAGE_REGISTRY: dict[str, LanguageEntry] = {
    "krio": {
        "code": "krio",
        "name": "Krio",
        "iso_639_3": "kri",
        "status": "active",
        "note": "Sierra Leonean Krio – primary target language",
        "is_target_sl_language": True,
    },
    "mende": {
        "code": "mende",
        "name": "Mende",
        "iso_639_3": "men",
        "status": "planned",
        "is_target_sl_language": True,
    },
    "temne": {
        "code": "temne",
        "name": "Temne",
        "iso_639_3": "tem",
        "status": "planned",
        "is_target_sl_language": True,
    },
    "limba": {
        "code": "limba",
        "name": "Limba",
        "iso_639_3": "lia",
        "status": "planned",
        "is_target_sl_language": True,
    },
    "loko": {
        "code": "loko",
        "name": "Loko",
        "iso_639_3": "lok",
        "status": "planned",
        "is_target_sl_language": True,
    },
    "kono": {
        "code": "kono",
        "name": "Kono",
        "iso_639_3": "kno",
        "status": "planned",
        "is_target_sl_language": True,
    },
    "kissi": {
        "code": "kissi",
        "name": "Kissi",
        "iso_639_3": "kss",
        "status": "planned",
        "is_target_sl_language": True,
    },
    "mandingo": {
        "code": "mandingo",
        "name": "Mandingo",
        "iso_639_3": "man",
        "status": "planned",
        "is_target_sl_language": True,
    },
    "english": {
        "code": "english",
        "name": "English",
        "iso_639_3": "eng",
        "status": "active",
        "note": "Official language of Sierra Leone; translation pivot language",
        "is_target_sl_language": False,
    },
}

# Derived dictionary of Sierra Leonean target languages for API/CLI compatibility
SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    code: {k: str(v) for k, v in entry.items() if k in ("name", "iso_639_3", "status", "note")}
    for code, entry in LANGUAGE_REGISTRY.items()
    if entry.get("is_target_sl_language", False)
}


def get_language_state(code: str) -> LanguageState | None:
    """Return the state ('planned', 'active', 'operational') for a language code, or None."""
    entry = LANGUAGE_REGISTRY.get(code.lower())
    if entry is None:
        return None
    return entry["status"]
