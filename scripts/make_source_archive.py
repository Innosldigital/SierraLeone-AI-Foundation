"""
Cross-platform clean source archive builder for SierraLeone-AI-Foundation (SLAI-018).

Excludes:
  - .venv/, venv/, env/
  - .pytest_cache/, __pycache__/, .mypy_cache/, .ruff_cache/
  - *.egg-info/, dist/, build/, .eggs/
  - .git/
  - .env, *.pem, *.key, secrets/
  - model weights (*.safetensors, *.bin, *.pt, *.pth, *.gguf, *.ckpt)
  - private raw/audio/quarantine datasets (preserving .gitkeep files)
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path, PurePosixPath

EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "ENV",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".eggs",
        "secrets",
        "logs",
        "htmlcov",
        "quarantine",
    }
)

EXCLUDED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".pyd",
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".gguf",
        ".ckpt",
        ".parquet",
        ".arrow",
        ".pem",
        ".key",
        ".log",
        ".whl",
        ".egg",
    }
)

EXCLUDED_FILES: frozenset[str] = frozenset(
    {
        ".env",
        ".coverage",
        "coverage.xml",
        ".DS_Store",
        "Thumbs.db",
    }
)


def _data_allowed(rel: PurePosixPath) -> bool:
    """Return True if rel is an allowed template, keepfile, register, or manifest."""
    parts = rel.parts
    if not parts:
        return True
    name = parts[-1]
    if parts[0] == "datasets":
        return (
            name == ".gitkeep"
            or name.startswith("example_")
            or name == "source_register.csv"
            or "manifests" in parts
        )
    if parts[0] == "benchmarks":
        return name == ".gitkeep" or "schema" in name or "manifests" in parts
    return True


def should_include_path(rel_path: PurePosixPath) -> bool:
    """Return True if `rel_path` belongs in a clean source archive."""
    parts = rel_path.parts
    if not parts:
        return False

    for part in parts[:-1]:
        if part in EXCLUDED_DIRS or part.endswith(".egg-info"):
            return False

    name = parts[-1]
    if name in EXCLUDED_FILES or name.endswith(".zip"):
        return False

    suffix = rel_path.suffix.lower()
    if suffix in EXCLUDED_SUFFIXES:
        return False

    if not _data_allowed(rel_path):
        return False

    return True


def build_clean_archive(project_root: Path, output_zip: Path) -> list[str]:
    """Create a clean source zip at `output_zip` and return the list of archived relative paths."""
    archived: list[str] = []
    output_resolved = output_zip.resolve()
    with zipfile.ZipFile(output_resolved, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(project_root.rglob("*")):
            if not item.is_file():
                continue
            if item.resolve() == output_resolved:
                continue
            rel = PurePosixPath(item.relative_to(project_root).as_posix())
            if should_include_path(rel):
                zf.write(item, arcname=str(rel))
                archived.append(str(rel))
    return archived


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create clean source archive for SierraLeone-AI-Foundation"
    )
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "SierraLeone-AI-Foundation-clean.zip",
        help="Destination .zip path",
    )
    args = parser.parse_args()
    files = build_clean_archive(default_root, args.output)
    print(f"Created clean archive {args.output} with {len(files)} files.")


if __name__ == "__main__":
    main()
