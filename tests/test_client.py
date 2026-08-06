"""Tests for the OpenRouter client with ladder failover.

No network calls: every test mocks the HTTP transport via
httpx.MockTransport. The suite must pass fully offline.
"""

import httpx
import pytest

from clarif_eye.client import (
    CompletionResult,
    LadderExhaustedError,
    OpenRouterClient,
    OpenRouterError,
)
from clarif_eye.registry import load_registry

EYES_LADDER = load_registry().ladder("eyes")
BRAIN_LADDER = load_registry().ladder("brain")

SENTINEL_KEY = "sk-sentinel-super-secret-do-not-leak-9f3a2b1c"


def make_client(handler, api_key=SENTINEL_KEY, **kwargs):
    transport = httpx.MockTransport(handler)
    return OpenRouterClient(api_key=api_key, transport=transport, **kwargs)


def json_response(status_code, message=None, code=None):
    if message is None and code is None:
        body = {"choices": [{"message": {"content": "a description"}}]}
    else:
        error = {}
        if message is not None:
            error["message"] = message
        if code is not None:
            error["code"] = code
        body = {"error": error}
    return httpx.Response(status_code, json=body)


# --- Construction-time key validation --------------------------------------


def test_missing_api_key_raises_at_construction(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenRouterError):
        OpenRouterClient()


def test_blank_api_key_raises_at_construction():
    with pytest.raises(OpenRouterError):
        OpenRouterClient(api_key="   ")


def test_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SENTINEL_KEY)

    def handler(request):
        return json_response(200)

    client = OpenRouterClient(transport=httpx.MockTransport(handler))
    result = client.complete("eyes", [{"role": "user", "content": "hi"}])
    assert result.model == EYES_LADDER[0]


# --- Success on the first rung -----------------------------------------


def test_success_on_first_rung():
    calls = []

    def handler(request):
        calls.append(request)
        return json_response(200)

    client = make_client(handler)
    result = client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert isinstance(result, CompletionResult)
    assert result.content == "a description"
    assert result.model == EYES_LADDER[0]
    assert len(calls) == 1


# --- Failover: 429 then success -----------------------------------------


def test_429_on_first_rung_then_success_on_second():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return json_response(429, message="rate limited")
        return json_response(200)

    client = make_client(handler)
    result = client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert result.model == EYES_LADDER[1]
    assert len(calls) == 2


# --- Failover: 5xx -----------------------------------------------------


def test_5xx_failover():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return json_response(503, message="upstream unavailable")
        return json_response(200)

    client = make_client(handler)
    result = client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert result.model == EYES_LADDER[1]
    assert len(calls) == 2


# --- Failover: model-not-found (404) ------------------------------------


def test_404_model_not_found_failover():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return json_response(404, message="No such model")
        return json_response(200)

    client = make_client(handler)
    result = client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert result.model == EYES_LADDER[1]
    assert len(calls) == 2


# --- Failover: model-not-found (400, mentions model) ---------------------


def test_400_model_not_found_failover():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return json_response(400, message="model 'x' is not a valid model ID", code="model_not_found")
        return json_response(200)

    client = make_client(handler)
    result = client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert result.model == EYES_LADDER[1]
    assert len(calls) == 2


# --- No failover: caller error (400, not about the model) ----------------


def test_caller_error_400_does_not_failover():
    calls = []

    def handler(request):
        calls.append(request)
        return json_response(400, message="messages field is required")

    client = make_client(handler)
    with pytest.raises(OpenRouterError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert len(calls) == 1  # must not burn the rest of the ladder
    assert not isinstance(exc_info.value, LadderExhaustedError)


# --- All rungs exhausted -------------------------------------------------


def test_all_rungs_exhausted_raises_typed_error_naming_role_and_models():
    def handler(request):
        return json_response(500, message="upstream down")

    client = make_client(handler)
    with pytest.raises(LadderExhaustedError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    err = exc_info.value
    assert err.role == "eyes"
    assert len(err.attempts) == len(EYES_LADDER)
    message = str(err)
    assert "eyes" in message
    for model in EYES_LADDER:
        assert model in message


# --- Key never leaks -----------------------------------------------------


def test_api_key_never_leaks_in_repr_or_str():
    def handler(request):
        return json_response(200)

    client = make_client(handler)

    assert SENTINEL_KEY not in repr(client)
    assert SENTINEL_KEY not in str(client)


def test_api_key_never_leaks_in_error_on_exhaustion():
    def handler(request):
        return json_response(500, message="upstream down")

    client = make_client(handler)
    with pytest.raises(LadderExhaustedError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert SENTINEL_KEY not in str(exc_info.value)
    assert SENTINEL_KEY not in repr(exc_info.value)


def test_api_key_never_leaks_in_error_on_caller_error():
    def handler(request):
        return json_response(400, message="messages field is required")

    client = make_client(handler)
    with pytest.raises(OpenRouterError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert SENTINEL_KEY not in str(exc_info.value)
    assert SENTINEL_KEY not in repr(exc_info.value)


def test_api_key_never_leaks_in_headers_exposed_via_repr():
    # Even the underlying transport/headers must not stringify the key.
    def handler(request):
        return json_response(200)

    client = make_client(handler)
    client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert SENTINEL_KEY not in repr(client)
    assert SENTINEL_KEY not in str(client)


# --- Timeout is derived from the role's latency budget --------------------


def test_timeout_is_set_for_eyes_role():
    captured = {}

    def handler(request):
        captured["timeout"] = request.extensions.get("timeout")
        return json_response(200)

    client = make_client(handler)
    client.complete("eyes", [{"role": "user", "content": "hi"}])

    timeout = captured["timeout"]
    assert timeout is not None
    assert all(v is not None for v in timeout.values())
    assert timeout["read"] <= 8.0


def test_timeout_is_set_for_brain_role():
    captured = {}

    def handler(request):
        captured["timeout"] = request.extensions.get("timeout")
        return json_response(200)

    client = make_client(handler)
    client.complete("brain", [{"role": "user", "content": "hi"}])

    timeout = captured["timeout"]
    assert timeout is not None
    assert all(v is not None for v in timeout.values())
    assert timeout["read"] <= 25.0


# --- Headers: optional app URL / name -------------------------------------


def test_optional_app_headers_are_sent():
    captured = {}

    def handler(request):
        captured["headers"] = request.headers
        return json_response(200)

    client = make_client(
        handler, app_url="https://example.com/app", app_name="Clarif-Eye"
    )
    client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert captured["headers"]["HTTP-Referer"] == "https://example.com/app"
    assert captured["headers"]["X-Title"] == "Clarif-Eye"
