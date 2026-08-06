"""Config-driven model registry.

Model IDs for OpenRouter live in src/clarif_eye/config/models.toml, never in code, so
swapping a model is a one-line config edit. Each role ("eyes", "brain")
has an ordered ladder of model IDs tried in sequence on failure.

All validation happens at load time (startup), so a malformed config
fails immediately and loudly rather than mid-request.

Note: beyond requiring the ":free" suffix, model IDs are not validated
for shape (e.g. "../../etc/passwd:free" or ":free" load cleanly). This is
deliberate: OpenRouter is the authority on valid IDs, config is trusted
input, and a bad ID simply surfaces as a 404 that the P0.3 ladder
failover already handles. Don't assume the ":free" suffix check is the
only thing that can be wrong with an entry.
"""

import tomllib
from importlib import resources
from types import MappingProxyType

ROLES = ("eyes", "brain")


class RegistryError(Exception):
    """Raised for any model registry configuration or usage error."""


def _validate_ladders(ladders):
    """Validate a role -> tuple-of-model-IDs mapping and return a cleaned dict.

    Shared by ModelRegistry.__init__ and load_registry so the D10 free-only
    policy (and the other ladder rules) can't be bypassed by constructing a
    ModelRegistry directly. Expects each ladder to already be a tuple (TOML
    list-shape checks are the caller's responsibility).
    """
    cleaned_ladders = {}
    for role in ROLES:
        if role not in ladders:
            raise RegistryError(f"model config is missing required role section [{role}]")

        raw_ladder = ladders[role]
        if not isinstance(raw_ladder, tuple):
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

            if entry.strip() != entry:
                raise RegistryError(
                    f"role {role!r} ladder entry {entry!r} has leading or trailing whitespace; "
                    "model IDs must not have surrounding whitespace"
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

        cleaned_ladders[role] = tuple(cleaned)

    return cleaned_ladders


class ModelRegistry:
    """Holds validated, ordered model ladders for each role."""

    def __init__(self, ladders):
        self._ladders = MappingProxyType(_validate_ladders(ladders))

    def ladder(self, role):
        """Return the ordered, immutable ladder of model IDs for a role."""
        if role not in self._ladders:
            raise RegistryError(
                f"unknown role {role!r}; known roles are {', '.join(ROLES)}"
            )
        return self._ladders[role]


def load_registry(path=None):
    """Load and validate the model registry config.

    `path` defaults to the packaged config/models.toml (resolved via
    importlib.resources so it works identically editable and installed),
    never relative to the current working directory.
    """
    if path is not None:
        config_path = path
        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError as e:
            raise RegistryError(f"model config file not found: {config_path}") from e
        except tomllib.TOMLDecodeError as e:
            raise RegistryError(f"model config file is not valid TOML: {config_path}") from e
    else:
        traversable = resources.files("clarif_eye").joinpath("config", "models.toml")
        try:
            with traversable.open("rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError as e:
            raise RegistryError(f"model config file not found: {traversable}") from e
        except tomllib.TOMLDecodeError as e:
            raise RegistryError(f"model config file is not valid TOML: {traversable}") from e

    raw_ladders = {}
    for role in ROLES:
        if role not in data:
            raise RegistryError(f"model config is missing required role section [{role}]")

        section = data[role]
        raw_ladder = section.get("ladder") if isinstance(section, dict) else None

        if not isinstance(raw_ladder, list):
            raise RegistryError(f"role {role!r} must have a 'ladder' that is a list of model IDs")

        raw_ladders[role] = tuple(raw_ladder)

    return ModelRegistry(raw_ladders)
