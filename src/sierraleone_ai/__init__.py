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
  - Other Sierra Leonean languages

Status: Foundation phase – dataset infrastructure only.
No production model is operational yet.
"""

__version__ = "0.1.0"
__author__ = "CAFP Sierra Leone AI Team"
__license__ = "Apache-2.0"

SUPPORTED_LANGUAGES = {
    "krio": {
        "name": "Krio",
        "iso_639_3": "kri",
        "status": "active",
        "note": "Sierra Leonean Krio – primary target language",
    },
    "mende": {
        "name": "Mende",
        "iso_639_3": "men",
        "status": "planned",
    },
    "temne": {
        "name": "Temne",
        "iso_639_3": "tem",
        "status": "planned",
    },
    "limba": {
        "name": "Limba",
        "iso_639_3": "lia",
        "status": "planned",
    },
    "loko": {
        "name": "Loko",
        "iso_639_3": "lok",
        "status": "planned",
    },
    "kono": {
        "name": "Kono",
        "iso_639_3": "kno",
        "status": "planned",
    },
    "kissi": {
        "name": "Kissi",
        "iso_639_3": "kss",
        "status": "planned",
    },
}
