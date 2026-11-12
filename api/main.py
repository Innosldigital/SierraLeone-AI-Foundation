"""
Top-level API entrypoint re-exporting the FastAPI application from `sierraleone_ai.api.app`.

Run with:
    uvicorn api.main:app --reload
or:
    uvicorn sierraleone_ai.api.app:app --reload
"""

from sierraleone_ai.api.app import app

__all__ = ["app"]
