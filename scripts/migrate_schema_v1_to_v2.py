"""
One-off schema migration script from schema v1 to schema v2 (PA-011).

Migrates JSONL records:
- Maps legacy consent_scope values:
  * "training" -> "text_training"
  * "publication" -> "public_release"
  * "commercial" -> "commercial_use"
  * "evaluation" -> dropped
- Sets schema_version = 2
- Refuses to guess training_permitted: sets training_permitted = False
- Renames is_native_sl_krio_speaker -> is_native_speaker_of_record_language in reviews
- Preserves or validates split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_CONSENT_SCOPE_MAP = {
    "training": "text_training",
    "publication": "public_release",
    "commercial": "commercial_use",
}


def migrate_record_v1_to_v2(record: dict[str, Any], default_split: str = "train") -> dict[str, Any]:
    """Migrate a single dictionary representation of a dataset record."""
    migrated = dict(record)
    meta = dict(migrated.get("metadata", {}))

    # 1. Set schema_version = 2
    meta["schema_version"] = 2

    # 2. Ensure split is explicitly set
    if "split" not in meta or not meta["split"]:
        meta["split"] = default_split

    # 3. Refuse to guess training_permitted and redistribution_permitted
    meta["training_permitted"] = False
    meta["redistribution_permitted"] = False

    # 4. Migrate consent_scope aliases
    raw_scopes = meta.get("consent_scope", [])
    new_scopes: list[str] = []
    for s in raw_scopes:
        if s == "evaluation":
            continue
        mapped = _CONSENT_SCOPE_MAP.get(s, s)
        if mapped not in new_scopes:
            new_scopes.append(mapped)
    meta["consent_scope"] = new_scopes

    # 5. Migrate reviewer sign-offs
    raw_reviews = meta.get("reviews", [])
    new_reviews: list[dict[str, Any]] = []
    for r in raw_reviews:
        rev = dict(r)
        if "is_native_sl_krio_speaker" in rev and "is_native_speaker_of_record_language" not in rev:
            rev["is_native_speaker_of_record_language"] = rev.pop("is_native_sl_krio_speaker")
        new_reviews.append(rev)
    meta["reviews"] = new_reviews

    migrated["metadata"] = meta
    return migrated


def migrate_file(path: Path, default_split: str = "train") -> int:
    """Migrate all JSONL rows in a file in-place."""
    if not path.exists():
        return 0
    lines: list[str] = []
    count = 0
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                lines.append(line)
                continue
            data = json.loads(stripped)
            migrated = migrate_record_v1_to_v2(data, default_split=default_split)
            lines.append(json.dumps(migrated, ensure_ascii=False) + "\n")
            count += 1

    with path.open("w", encoding="utf-8") as f:
        f.writelines(lines)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate dataset JSONL files from v1 to v2 schema")
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Paths or globs of JSONL files to migrate"
    )
    parser.add_argument(
        "--default-split", default="train", help="Default split if missing in record"
    )
    args = parser.parse_args()

    total = 0
    for p in args.paths:
        migrated_count = migrate_file(p, default_split=args.default_split)
        print(f"Migrated {migrated_count} records in {p}")
        total += migrated_count
    print(f"Total migrated: {total}")


if __name__ == "__main__":
    main()
