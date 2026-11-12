"""
Model serving state management (PA-012 / PA-013).

Tracks which model artifacts are loaded in memory for serving.
The `/health/ready` endpoint uses ModelState to determine readiness rather
than hardcoded or metadata-level language status dictionaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelState:
    """In-memory state of loaded models across all tasks."""

    _loaded_models: dict[str, str] = field(default_factory=dict)
    _model_instances: dict[str, Any] = field(default_factory=dict)

    def is_ready(self, task: str | None = None) -> bool:
        """
        Return True if operational models are loaded.
        If task is specified, returns whether that specific task model is loaded.
        """
        if task is not None:
            return task in self._loaded_models
        return len(self._loaded_models) > 0

    def get_loaded_models(self) -> list[str]:
        """Return a sorted list of registered model identifiers."""
        return sorted(self._loaded_models.values())

    def get_loaded_tasks(self) -> dict[str, str]:
        """Return a copy of the mapping task -> model_id."""
        return dict(self._loaded_models)

    def register_model(self, task: str, model_id: str, instance: Any = None) -> None:
        """Register a model as loaded and operational for a given task."""
        self._loaded_models[task] = model_id
        if instance is not None:
            self._model_instances[task] = instance

    def unload_model(self, task: str) -> None:
        """Unload a model for a given task."""
        self._loaded_models.pop(task, None)
        self._model_instances.pop(task, None)

    def reset(self) -> None:
        """Reset all loaded models (unload all)."""
        self._loaded_models.clear()
        self._model_instances.clear()


# Global singleton serving state
model_state = ModelState()
