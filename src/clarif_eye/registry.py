"""Config-driven model registry.

Model IDs for OpenRouter live in config/models.toml, never in code, so
swapping a model is a one-line config edit. Each role ("eyes", "brain")
has an ordered ladder of model IDs tried in sequence on failure.

All validation happens at load time (startup), so a malformed config
fails immediately and loudly rather than mid-request.
"""

import tomllib
from pathlib import Path

ROLES = ("eyes", "brain")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "models.toml"


class RegistryError(Exception):
    """Raised for any model registry configuration or usage error."""


class ModelRegistry:
    """Holds validated, ordered model ladders for each role."""

    def __init__(self, ladders):
        self._ladders = ladders

    def ladder(self, role):
        """Return the ordered, immutable ladder of model IDs for a role."""
        if role not in self._ladders:
            raise RegistryError(
                f"unknown role {role!r}; known roles are {', '.join(ROLES)}"
            )
        return self._ladders[role]


def load_registry(path=None):
    """Load and validate the model registry config.

    `path` defaults to config/models.toml resolved relative to the repo,
    never relative to the current working directory.
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    ladders = {}
    for role in ROLES:
        if role not in data:
            raise RegistryError(f"model config is missing required role section [{role}]")

        section = data[role]
        raw_ladder = section.get("ladder") if isinstance(section, dict) else None

        if not isinstance(raw_ladder, list):
            raise RegistryError(f"role {role!r} must have a 'ladder' that is a list of model IDs")

        if len(raw_ladder) == 0:
            raise RegistryError(f"role {role!r} has an empty ladder; at least one model ID is required")

        seen = set()
        cleaned = []
        for entry in raw_ladder:
            if not isinstance(entry, str) or not entry.strip():
                raise RegistryError(
                    f"role {role!r} ladder contains a blank or non-string entry: {entry!r}"
                )
            if entry in seen:
                raise RegistryError(f"role {role!r} ladder contains duplicate entry: {entry!r}")
            seen.add(entry)

            if not entry.endswith(":free"):
                raise RegistryError(
                    f"role {role!r} ladder entry {entry!r} is not a free model; "
                    "policy (decision D10) requires every model ID to end in ':free'"
                )

            cleaned.append(entry)

        ladders[role] = tuple(cleaned)

    return ModelRegistry(ladders)
