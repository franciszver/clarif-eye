"""Failing-first checks for the Render deploy configuration (issue #22 /
P8.2, re-scoped from Hugging Face to Render).

Render assigns the network port through the PORT environment variable and
requires the process to listen on 0.0.0.0. Gradio's own default,
127.0.0.1:7860, is not reachable from Render's health check, so the first
deploy would never come up. `app.resolve_bind_address()` is the pure
function these tests check: no server is launched and no network call is
made anywhere in this file, matching the discipline the rest of this
project already uses for app.py/ui.py (see clarif_eye.ui's module
docstring, "TESTABLE without launching a server").
"""

from pathlib import Path

import yaml

import app

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Port/host resolution (pure function, no server launched) --------------


def test_resolves_render_host_and_port_when_port_is_set(monkeypatch):
    monkeypatch.setenv("PORT", "10000")

    host, port = app.resolve_bind_address()

    assert host == "0.0.0.0"
    assert port == 10000


def test_resolves_render_port_from_whatever_value_render_assigns(monkeypatch):
    """PORT is not hardcoded to any one number - it must be read, not assumed."""
    monkeypatch.setenv("PORT", "54321")

    host, port = app.resolve_bind_address()

    assert host == "0.0.0.0"
    assert port == 54321


def test_falls_back_to_gradio_defaults_when_port_is_unset(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)

    host, port = app.resolve_bind_address()

    assert host is None
    assert port is None


# --- render.yaml -------------------------------------------------------


def test_render_yaml_exists_and_is_valid_yaml():
    render_yaml_path = REPO_ROOT / "render.yaml"
    assert render_yaml_path.exists(), "render.yaml is missing"

    with open(render_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["services"], "render.yaml declares no services"


def test_render_yaml_names_the_free_plan_and_a_start_command():
    with open(REPO_ROOT / "render.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    service = data["services"][0]
    assert service["plan"] == "free"
    assert service.get("startCommand")


def test_render_yaml_declares_openrouter_key_as_not_synced():
    with open(REPO_ROOT / "render.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    service = data["services"][0]
    env_vars = {ev["key"]: ev for ev in service.get("envVars", [])}

    assert "OPENROUTER_API_KEY" in env_vars
    assert env_vars["OPENROUTER_API_KEY"].get("sync") is False
    assert "value" not in env_vars["OPENROUTER_API_KEY"]


def test_render_yaml_has_no_literal_api_key():
    text = (REPO_ROOT / "render.yaml").read_text(encoding="utf-8")

    # OpenRouter keys share this prefix; none may appear literally.
    assert "sk-or-" not in text


# --- README mentions the cold start -------------------------------------


def test_readme_mentions_the_render_cold_start():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "15 minutes" in text
    assert "Render" in text


# --- render.yaml documents the CI-triggered deploy hook -------------------


def test_render_yaml_comment_documents_the_ci_deploy_hook():
    text = (REPO_ROOT / "render.yaml").read_text(encoding="utf-8")

    assert "deploy hook" in text.lower()
    assert "Auto-Deploy" in text
