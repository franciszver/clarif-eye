"""Tests for the OpenRouter client with ladder failover.

No network calls: every test mocks the HTTP transport via
httpx.MockTransport. The suite must pass fully offline.
"""

import time

import httpx
import pytest

from clarif_eye import client as client_module
from clarif_eye.client import (
    CompletionResult,
    LadderExhaustedError,
    OpenRouterClient,
    OpenRouterError,
)
from clarif_eye.registry import load_registry
from clarif_eye import registry as registry_module

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


def test_api_key_never_leaks_anywhere_it_could_plausibly_surface():
    # Consolidates the two near-duplicate leak tests this replaces, which
    # only exercised the hardcoded __repr__ and would pass unconditionally
    # regardless of where the key was actually stored. This checks every
    # place a leak could realistically occur: the client's own repr/str,
    # its __dict__ (where the key might be stashed as an attribute), the
    # underlying httpx.Client's header store, and the text of an error
    # raised from a failing request.
    def handler(request):
        return json_response(500, message="upstream down")

    client = make_client(handler)
    with pytest.raises(LadderExhaustedError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert SENTINEL_KEY not in repr(client)
    assert SENTINEL_KEY not in str(client)
    assert SENTINEL_KEY not in repr(client.__dict__)
    assert SENTINEL_KEY not in repr(client._client.headers)
    assert SENTINEL_KEY not in str(exc_info.value)
    assert SENTINEL_KEY not in repr(exc_info.value)


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


# --- Total latency budget (not per-attempt) --------------------------------


def test_total_budget_is_enforced_across_all_attempts_not_per_attempt(monkeypatch):
    # Patch the role budget small so the test stays fast, and make every
    # rung slow (but each individually well under the real 8s/25s budgets
    # to prove this isn't relying on httpx's own timeout).
    monkeypatch.setitem(client_module.ROLE_TIMEOUTS, "eyes", 0.3)
    calls = []

    def handler(request):
        calls.append(request)
        time.sleep(0.2)
        return json_response(500, message="upstream down")

    client = make_client(handler)

    start = time.monotonic()
    with pytest.raises(LadderExhaustedError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])
    elapsed = time.monotonic() - start

    # N x per-attempt budget would be 3 x 0.3s = 0.9s; the total budget is
    # 0.3s, so wall-clock must stay well under both N x budget and a small
    # margin above the budget itself.
    assert elapsed < 0.6
    assert len(calls) < len(EYES_LADDER)  # at least one rung never attempted

    attempts = exc_info.value.attempts
    assert len(attempts) == len(EYES_LADDER)
    last_attempt = attempts[-1]
    assert last_attempt.category == "budget_exhausted"
    assert "budget" in last_attempt.detail.lower()


# --- Terminal failures: no failover, exactly one request -------------------


@pytest.mark.parametrize("status", [401, 402, 403, 413])
def test_terminal_status_raises_immediately_with_one_request(status):
    calls = []

    def handler(request):
        calls.append(request)
        return json_response(status, message="terminal failure", code=status)

    client = make_client(handler)
    with pytest.raises(OpenRouterError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert len(calls) == 1
    assert not isinstance(exc_info.value, LadderExhaustedError)
    assert SENTINEL_KEY not in str(exc_info.value)


def test_401_message_points_at_the_api_key():
    def handler(request):
        return json_response(401, message="User not found.", code=401)

    client = make_client(handler)
    with pytest.raises(OpenRouterError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    message = str(exc_info.value).lower()
    assert "api" in message and "key" in message


# --- 400 classifier: known false positive / false negative -----------------


def test_400_classifier_does_not_misfire_on_parameter_error_mentioning_model():
    # Verified live: a caller-side parameter error can mention "model" in
    # its message without being a model-availability failure. Must raise
    # immediately, not burn the ladder.
    calls = []

    def handler(request):
        calls.append(request)
        return json_response(
            400,
            message="Invalid request: 'temperature' must be between 0 and 2 for this model",
        )

    client = make_client(handler)
    with pytest.raises(OpenRouterError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert len(calls) == 1
    assert not isinstance(exc_info.value, LadderExhaustedError)


def test_400_classifier_recognizes_no_endpoints_found_as_model_failure():
    # Verified live-style message that the old "model" substring test
    # missed entirely (no literal word "model" in it).
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return json_response(400, message="No endpoints found matching your data policy")
        return json_response(200)

    client = make_client(handler)
    result = client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert len(calls) == 2
    assert result.model == EYES_LADDER[1]


# --- Malformed 200 bodies advance the ladder --------------------------------


@pytest.mark.parametrize(
    "bad_response",
    [
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"id": "x"}),
        httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
        httpx.Response(200, text="not json"),
    ],
    ids=["empty-choices", "missing-choices-key", "null-content", "non-json"],
)
def test_malformed_200_body_advances_the_ladder(bad_response):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return bad_response
        return json_response(200)

    client = make_client(handler)
    result = client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert len(calls) == 2
    assert result.model == EYES_LADDER[1]
    assert result.content == "a description"


def test_malformed_200_never_returns_null_or_blank_content():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})

    client = make_client(handler)
    with pytest.raises(LadderExhaustedError):
        client.complete("eyes", [{"role": "user", "content": "hi"}])


# --- Context manager --------------------------------------------------------


def test_client_works_as_a_context_manager():
    def handler(request):
        return json_response(200)

    transport = httpx.MockTransport(handler)
    with OpenRouterClient(api_key=SENTINEL_KEY, transport=transport) as client:
        result = client.complete("eyes", [{"role": "user", "content": "hi"}])
        assert result.model == EYES_LADDER[0]

    assert client._client.is_closed


# --- Transport failures: timeout / connection error failover ---------------


def test_connect_timeout_fails_over_and_is_categorized():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("connect timed out", request=request)
        return json_response(200)

    client = make_client(handler)
    result = client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert len(calls) == 2
    assert result.model == EYES_LADDER[1]


def test_read_timeout_fails_over_and_is_categorized():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("read timed out", request=request)
        return json_response(200)

    client = make_client(handler)
    result = client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert len(calls) == 2
    assert result.model == EYES_LADDER[1]


def test_connect_error_fails_over_and_is_categorized():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return json_response(200)

    client = make_client(handler)
    result = client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert len(calls) == 2
    assert result.model == EYES_LADDER[1]


def test_all_transport_failure_kinds_produce_the_expected_attempt_categories():
    # One call per exception kind so the ladder exhausts; asserts the exact
    # category recorded for each, per FIX 2's fixed category set.
    exceptions = [
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectError("connection refused"),
    ]
    calls = []

    def handler(request):
        calls.append(request)
        raise exceptions[len(calls) - 1]

    client = make_client(handler)
    with pytest.raises(LadderExhaustedError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    categories = [a.category for a in exc_info.value.attempts]
    assert categories == ["timeout", "timeout", "transport_error"]


# --- Mutation proof that the timeout/transport-error branches matter -------
#
# See report: these branches were manually patched to `raise` and the new
# tests above were confirmed to fail, then restored. Documented here rather
# than automated, since automating "edit source, rerun, edit back" inside
# the suite itself would be its own source of flakiness.


# --- Structured attempts: mixed exhaustion (429, 500, budget cut) ----------


def test_mixed_exhaustion_produces_structured_attempts_with_categories(monkeypatch):
    monkeypatch.setitem(client_module.ROLE_TIMEOUTS, "eyes", 0.25)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return json_response(429, message="rate limited")
        if len(calls) == 2:
            time.sleep(0.3)  # eats the rest of the (patched) budget
            return json_response(500, message="upstream down")
        return json_response(200)  # should never be reached

    client = make_client(handler)
    with pytest.raises(LadderExhaustedError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    attempts = exc_info.value.attempts
    assert len(attempts) == len(EYES_LADDER)
    assert attempts[0].category == "rate_limited"
    assert attempts[0].status_code == 429
    assert attempts[1].category == "server_error"
    assert attempts[1].status_code == 500
    for untried in attempts[2:]:
        assert untried.category == "budget_exhausted"
        assert untried.status_code is None
    assert len(calls) == 2  # remaining rung(s) never attempted


# --- Unknown role: RegistryError must not leak past the documented contract


def test_unknown_role_raises_openrouter_error_not_registry_error():
    def handler(request):
        return json_response(200)

    client = make_client(handler)
    with pytest.raises(OpenRouterError) as exc_info:
        client.complete("nonsense", [{"role": "user", "content": "hi"}])

    assert not isinstance(exc_info.value, registry_module.RegistryError)
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, registry_module.RegistryError)


# --- Registry is loaded once, not once per request --------------------------


def test_registry_is_loaded_exactly_once_across_multiple_requests(monkeypatch):
    load_calls = []
    real_load_registry = registry_module.load_registry

    def counting_load_registry(*args, **kwargs):
        load_calls.append(1)
        return real_load_registry(*args, **kwargs)

    monkeypatch.setattr(client_module, "load_registry", counting_load_registry)

    def handler(request):
        return json_response(200)

    client = OpenRouterClient(api_key=SENTINEL_KEY, transport=httpx.MockTransport(handler))
    assert len(load_calls) == 1  # loaded at construction

    for _ in range(4):
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert len(load_calls) == 1  # never re-read from disk per request
