"""
Shared pytest fixtures.

The production `datasets/source_register.csv` must only contain real, human-reviewed
rights decisions. Tests that need an APPROVED source use `TEST_APPROVED_SOURCE`, which
is injected here at runtime and never written to the real register (PA-008).
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

TEST_APPROVED_SOURCE = "test_fixture_approved_source"


@pytest.fixture(autouse=True)
def _inject_test_approved_source(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from sierraleone_ai.datasets import loader

    original = loader.load_source_register

    def _patched(register_path: Path | None = None) -> dict[str, dict[str, str]]:
        register = dict(original(register_path))
        register.setdefault(
            TEST_APPROVED_SOURCE,
            {"source_name": TEST_APPROVED_SOURCE, "decision": "APPROVED"},
        )
        return register

    monkeypatch.setattr(loader, "load_source_register", _patched)
    yield

