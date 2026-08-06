"""Tests for the config-driven model registry."""

import pytest

from clarif_eye.registry import RegistryError, load_registry


def write_config(tmp_path, text):
    path = tmp_path / "models.toml"
    path.write_text(text, encoding="utf-8")
    return path


# --- Happy path -------------------------------------------------------


def test_default_config_loads_both_ladders_in_order():
    registry = load_registry()

    assert registry.ladder("eyes") == (
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "google/gemma-4-31b-it:free",
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
