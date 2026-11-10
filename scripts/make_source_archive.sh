#!/usr/bin/env bash
# Clean source archive builder for SierraLeone-AI-Foundation (SLAI-018).
# Excludes virtual environments, caches, build artifacts, secrets (.env),
# private raw/audio datasets, and model weights.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ZIP="${1:-${PROJECT_ROOT}/SierraLeone-AI-Foundation-clean.zip}"

python "${SCRIPT_DIR}/make_source_archive.py" --output "${OUTPUT_ZIP}"
