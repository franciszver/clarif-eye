"""Tests for the config-driven model registry."""

import pytest

from clarif_eye.registry import ModelRegistry, RegistryError, load_registry


def write_config(tmp_path, text):
    path = tmp_path / "models.toml"
    path.write_text(text, encoding="utf-8")
    return path


# --- Happy path -------------------------------------------------------


def test_default_config_loads_both_ladders_in_order():
    registry = load_registry()

    assert registry.ladder("eyes") == (
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "google/gemma-4-26b-a4b-it:free",
    )
    assert registry.ladder("brain") == (
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
    )


def test_ladder_is_immutable_tuple():
    registry = load_registry()
    ladder = registry.ladder("eyes")

    assert isinstance(ladder, tuple)
    with pytest.raises(TypeError):
        ladder[0] = "something/else:free"


def test_custom_config_path_loads(tmp_path):
    path = write_config(
        tmp_path,
        """
        [eyes]
        ladder = ["a/model:free"]

        [brain]
        ladder = ["b/model:free"]
        """,
    )
    registry = load_registry(path)

    assert registry.ladder("eyes") == ("a/model:free",)
    assert registry.ladder("brain") == ("b/model:free",)


# --- Validation: missing role ------------------------------------------


def test_missing_role_section_raises(tmp_path):
    path = write_config(
        tmp_path,
        """
        [eyes]
        ladder = ["a/model:free"]
        """,
    )
    with pytest.raises(RegistryError, match="brain"):
        load_registry(path)


# --- Validation: empty ladder --------------------------------------------


def test_empty_ladder_raises(tmp_path):
    path = write_config(
        tmp_path,
        """
        [eyes]
        ladder = []

        [brain]
        ladder = ["b/model:free"]
        """,
    )
    with pytest.raises(RegistryError, match="eyes"):
        load_registry(path)


# --- Validation: ladder type/shape ---------------------------------------


def test_ladder_not_a_list_raises(tmp_path):
    path = write_config(
        tmp_path,
        """
        [eyes]
        ladder = "a/model:free"

        [brain]
        ladder = ["b/model:free"]
        """,
    )
    with pytest.raises(RegistryError, match="eyes"):
        load_registry(path)


def test_ladder_entry_not_a_string_raises(tmp_path):
    path = write_config(
        tmp_path,
        """
        [eyes]
        ladder = [123]

        [brain]
        ladder = ["b/model:free"]
        """,
    )
    with pytest.raises(RegistryError, match="eyes"):
        load_registry(path)


def test_ladder_entry_blank_raises(tmp_path):
    path = write_config(
        tmp_path,
        """
        [eyes]
        ladder = ["   "]

        [brain]
        ladder = ["b/model:free"]
        """,
    )
    with pytest.raises(RegistryError, match="eyes"):
        load_registry(path)


# --- Validation: duplicates -----------------------------------------------


def test_duplicate_entries_raise(tmp_path):
    path = write_config(
        tmp_path,
        """
        [eyes]
        ladder = ["a/model:free", "a/model:free"]

        [brain]
        ladder = ["b/model:free"]
        """,
    )
    with pytest.raises(RegistryError, match="eyes"):
        load_registry(path)


# --- Validation: free-only policy (D10) ------------------------------------


def test_non_free_model_id_raises(tmp_path):
    path = write_config(
        tmp_path,
        """
        [eyes]
        ladder = ["a/model:free"]

        [brain]
        ladder = ["b/model-not-free"]
        """,
    )
    with pytest.raises(RegistryError, match="free"):
        load_registry(path)


# --- Validation: unknown role -----------------------------------------------


def test_unknown_role_raises():
    registry = load_registry()
    with pytest.raises(RegistryError, match="unknown"):
        registry.ladder("hands")


# --- Direct construction cannot bypass the free-only policy ---------------


def test_direct_construction_with_non_free_model_raises():
    with pytest.raises(RegistryError, match="free"):
        ModelRegistry({"eyes": ("paid/model-not-free",), "brain": ("also-paid",)})


def test_direct_construction_with_empty_ladder_raises():
    with pytest.raises(RegistryError, match="eyes"):
        ModelRegistry({"eyes": (), "brain": ("b/model:free",)})


def test_direct_construction_with_blank_entry_raises():
    with pytest.raises(RegistryError, match="eyes"):
        ModelRegistry({"eyes": ("   ",), "brain": ("b/model:free",)})


def test_direct_construction_with_duplicate_entries_raises():
    with pytest.raises(RegistryError, match="eyes"):
        ModelRegistry(
            {"eyes": ("a/model:free", "a/model:free"), "brain": ("b/model:free",)}
        )


def test_direct_construction_missing_role_raises():
    with pytest.raises(RegistryError, match="brain"):
        ModelRegistry({"eyes": ("a/model:free",)})


# --- Returned ladders mapping cannot be mutated to inject a model ---------


def test_ladders_mapping_cannot_be_mutated():
    registry = load_registry()
    with pytest.raises(TypeError):
        registry._ladders["eyes"] = ("paid-model",)


def test_ladders_attribute_cannot_be_rebound():
    registry = load_registry()
    with pytest.raises(RegistryError):
        registry._ladders = {"eyes": ("paid/model-not-free",), "brain": ("paid/y:free",)}

    # The original, validated ladder must still be in effect.
    assert registry.ladder("eyes") == (
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "google/gemma-4-26b-a4b-it:free",
    )


# --- Direct construction: list ladders accepted, other shapes rejected ----


def test_direct_construction_with_list_ladders_succeeds():
    registry = ModelRegistry({"eyes": ["a/model:free"], "brain": ["b/model:free"]})

    assert registry.ladder("eyes") == ("a/model:free",)
    assert registry.ladder("brain") == ("b/model:free",)


def test_direct_construction_with_set_ladder_raises():
    with pytest.raises(RegistryError, match="eyes"):
        ModelRegistry({"eyes": {"a/model:free"}, "brain": ("b/model:free",)})


def test_direct_construction_with_generator_ladder_raises():
    with pytest.raises(RegistryError, match="eyes"):
        ModelRegistry({"eyes": (x for x in ["a/model:free"]), "brain": ("b/model:free",)})


def test_direct_construction_with_string_ladder_raises():
    with pytest.raises(RegistryError, match="eyes"):
        ModelRegistry({"eyes": "a/model:free", "brain": ("b/model:free",)})


# --- Validation: whitespace in entries -------------------------------------


def test_ladder_entry_leading_whitespace_raises(tmp_path):
    path = write_config(
        tmp_path,
        """
        [eyes]
        ladder = [" a/model:free"]

        [brain]
        ladder = ["b/model:free"]
        """,
    )
    with pytest.raises(RegistryError, match="whitespace"):
        load_registry(path)


def test_ladder_entry_trailing_whitespace_raises(tmp_path):
    path = write_config(
        tmp_path,
        """
        [eyes]
        ladder = ["a/model:free "]

        [brain]
        ladder = ["b/model:free"]
        """,
    )
    with pytest.raises(RegistryError, match="whitespace"):
        load_registry(path)


# --- Validation: single error surface --------------------------------------


def test_missing_config_file_raises_registry_error(tmp_path):
    missing_path = tmp_path / "does-not-exist.toml"
    with pytest.raises(RegistryError):
        load_registry(missing_path)


def test_malformed_toml_raises_registry_error(tmp_path):
    path = write_config(tmp_path, "this is not [valid toml")
    with pytest.raises(RegistryError):
        load_registry(path)


def test_directory_as_path_raises_registry_error(tmp_path):
    with pytest.raises(RegistryError):
        load_registry(tmp_path)
