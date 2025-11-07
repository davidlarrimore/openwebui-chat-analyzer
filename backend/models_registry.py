"""Models registry management for completion-capable models.

This module manages a provider-agnostic registry of models that have been
validated to support text completion/generation. The registry is persisted
as a JSON file and serves as a cache to avoid repeated validation tests.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from . import config

LOGGER = logging.getLogger(__name__)

# Registry file path
DEFAULT_REGISTRY_DIR = Path(__file__).resolve().with_name("data")
REGISTRY_PATH = DEFAULT_REGISTRY_DIR / "provider_registry.json"
LEGACY_BACKUP_PATH = DEFAULT_REGISTRY_DIR / "provider_registry.json.v1.backup"

# Potential legacy registry locations to migrate from
LEGACY_REGISTRY_CANDIDATES = [
    DEFAULT_REGISTRY_DIR / "models_registry.json",
    config.DATA_DIR / "provider_registry.json",
    config.DATA_DIR / "models_registry.json",
]


class ModelsRegistry:
    """Manager for the completion-capable models registry.

    The registry tracks which models (across all providers) have been validated
    to support text completion/generation. This provides a persistent cache to
    avoid re-validating models on every application restart.

    Registry Format (v2):
    {
        "completion_capable_models": ["model-1", "model-2", ...],
        "meta": {
            "last_updated": "2025-11-06T12:00:00Z",
            "version": "2.0"
        }
    }
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize registry manager.

        Args:
            registry_path: Path to registry JSON file (defaults to REGISTRY_PATH)
        """
        self.registry_path = registry_path or REGISTRY_PATH
        self._models: Set[str] = set()
        self._loaded = False

    def load(self) -> None:
        """Load registry from disk, migrating from v1 format if needed.

        If the registry file doesn't exist, initializes an empty registry.
        If the file is in v1 format (legacy), migrates to v2 format automatically.
        """
        if self._loaded:
            return

        data, source_path = self._load_raw_registry()

        if data is None:
            LOGGER.info(
                "Registry file not found at %s, initializing empty registry",
                self.registry_path,
            )
            self._models = set()
            self._loaded = True
            return

        # Check format version
        if "completion_capable_models" in data:
            # v2 format (new)
            self._models = set(data["completion_capable_models"])
            LOGGER.info(
                "Loaded registry v2 with %d completion-capable models",
                len(self._models),
            )
        elif "models" in data:
            # v1 format (legacy) - migrate
            LOGGER.info("Detected legacy registry format, migrating to v2...")
            self._migrate_from_v1(data, source_path)
            LOGGER.info(
                "Migration complete: %d completion-capable models",
                len(self._models),
            )
        else:
            LOGGER.warning("Unknown registry format, initializing empty registry")
            self._models = set()

        self._loaded = True

        if source_path and source_path != self.registry_path:
            LOGGER.info(
                "Migrated registry contents from %s to %s",
                source_path,
                self.registry_path,
            )
            self.save()

    def _legacy_paths(self) -> List[Path]:
        """Return legacy registry file paths to search when migrating."""
        seen: Set[Path] = set()
        resolved_target = self.registry_path.resolve()
        candidates = []
        for path in LEGACY_REGISTRY_CANDIDATES:
            try:
                resolved = path.resolve()
            except FileNotFoundError:
                resolved = path
            if resolved == resolved_target:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(path)
        return candidates

    def _load_raw_registry(self) -> tuple[Optional[Dict], Optional[Path]]:
        """Load registry JSON from the primary or legacy paths."""
        search_paths: List[Path] = [self.registry_path]
        if self.registry_path == REGISTRY_PATH:
            search_paths.extend(self._legacy_paths())

        for path in search_paths:
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f), path
            except json.JSONDecodeError as exc:
                LOGGER.error("Failed to parse registry JSON at %s: %s", path, exc)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.error("Failed to load registry from %s: %s", path, exc)

        return None, None

    def _migrate_from_v1(self, v1_data: Dict, source_path: Optional[Path]) -> None:
        """Migrate from v1 (provider-specific) to v2 (provider-agnostic) format.

        Args:
            v1_data: Legacy registry data with per-model metadata
            source_path: Filesystem path of the legacy registry file

        Side Effects:
            - Backs up v1 file to models_registry.json.v1.backup
            - Saves migrated v2 data to registry_path
            - Updates self._models with validated models
        """
        # Extract models where supports_completions == true
        models_dict = v1_data.get("models", {})
        completion_capable = []

        for model_name, model_data in models_dict.items():
            if isinstance(model_data, dict) and model_data.get("supports_completions"):
                completion_capable.append(model_name)

        # Back up v1 file
        backup_target: Optional[Path] = None
        if source_path and source_path.exists():
            backup_target = source_path.with_name(f"{source_path.name}.v1.backup")
        else:
            backup_target = LEGACY_BACKUP_PATH

        try:
            if source_path and source_path.exists() and backup_target and not backup_target.exists():
                shutil.copy2(source_path, backup_target)
                LOGGER.info("Backed up v1 registry to %s", backup_target)
        except Exception as exc:
            LOGGER.warning("Failed to backup v1 registry: %s", exc)

        # Update in-memory state
        self._models = set(completion_capable)

        # Save v2 format
        self.save()

    def save(self) -> None:
        """Persist registry to disk in v2 format.

        Creates parent directory if it doesn't exist.
        Writes atomically to avoid corruption (write to temp file, then rename).
        """
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "completion_capable_models": sorted(self._models),
            "meta": {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "version": "2.0",
            },
        }

        # Write atomically via temp file
        temp_path = self.registry_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")  # Trailing newline

            # Atomic rename
            temp_path.replace(self.registry_path)
            LOGGER.debug("Saved registry with %d models", len(self._models))

        except Exception as exc:
            LOGGER.error("Failed to save registry: %s", exc)
            if temp_path.exists():
                temp_path.unlink()
            raise

    def add_model(self, model_name: str) -> None:
        """Add a model to the completion-capable registry.

        Args:
            model_name: Model identifier (e.g., "gpt-4o-mini", "llama3:8b")

        Note:
            Call save() to persist changes to disk.
        """
        if not self._loaded:
            self.load()

        self._models.add(model_name)
        LOGGER.info("Added model to registry: %s", model_name)

    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the registry.

        Args:
            model_name: Model identifier to remove

        Returns:
            True if model was present and removed, False if not found.

        Note:
            Call save() to persist changes to disk.
        """
        if not self._loaded:
            self.load()

        if model_name in self._models:
            self._models.remove(model_name)
            LOGGER.info("Removed model from registry: %s", model_name)
            return True

        return False

    def is_validated(self, model_name: str) -> bool:
        """Check if a model has been validated for completions.

        Args:
            model_name: Model identifier to check

        Returns:
            True if model is in the completion-capable registry, False otherwise.
        """
        if not self._loaded:
            self.load()

        return model_name in self._models

    def get_all_models(self) -> List[str]:
        """Get all completion-capable models in the registry.

        Returns:
            Sorted list of model names.
        """
        if not self._loaded:
            self.load()

        return sorted(self._models)

    def clear(self) -> None:
        """Clear all models from the registry.

        Note:
            Call save() to persist changes to disk.
        """
        if not self._loaded:
            self.load()

        count = len(self._models)
        self._models.clear()
        LOGGER.warning("Cleared all %d models from registry", count)


# Singleton instance
_registry: Optional[ModelsRegistry] = None


def get_models_registry() -> ModelsRegistry:
    """Get the global models registry instance.

    Returns:
        Singleton ModelsRegistry instance.

    Note:
        The registry is lazily loaded on first access.
    """
    global _registry
    if _registry is None:
        _registry = ModelsRegistry()
        _registry.load()
    return _registry
