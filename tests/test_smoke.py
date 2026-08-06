"""Smoke test for clarif_eye package."""

import tomllib
from pathlib import Path

import clarif_eye


def test_import_clarif_eye():
    """Test that clarif_eye can be imported."""
    assert clarif_eye is not None


def test_version_exists():
    """Test that clarif_eye has a __version__ attribute."""
    assert hasattr(clarif_eye, "__version__")


def test_version_is_string():
    """Test that __version__ is a non-empty string."""
    assert isinstance(clarif_eye.__version__, str)
    assert len(clarif_eye.__version__) > 0


def test_version_matches_pyproject():
    """Test that __version__ matches the declared version in pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    expected_version = pyproject_data["project"]["version"]
    assert clarif_eye.__version__ == expected_version
