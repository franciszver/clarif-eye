"""Tests for clarif_eye.failure_messages (issue #18 / P6.2): category-specific
spoken failure messages.

Red-first artifact: every failure mode currently collapses into roughly the
same spoken text (see vision.py/synth.py/analysis.py before this issue), so
a user cannot tell "the service is busy, try again shortly" from "this is
misconfigured, tell the owner" - two states that need different actions.

These tests build REAL client.Attempt / client.LadderExhaustedError /
client.OpenRouterError objects - several of them via a real
OpenRouterClient.complete() call against httpx.MockTransport, exactly like
tests/test_client.py does - never a hand-built string standing in for one.
The mapping tests below (test_dispatch_*) additionally prove the mapping is
STRUCTURAL: it reads Attempt.category / OpenRouterError.status_code, never
the free-text detail/message a caller must not parse (see client.py's
Attempt docstring, "issue #18 ... without substring-matching upstream
prose").
"""

import httpx
import pytest

from clarif_eye.client import (
    Attempt,
    LadderExhaustedError,
    OpenRouterClient,
    OpenRouterError,
)
from clarif_eye.failure_messages import (
    BUSY_MESSAGE,
    CONFIG_ERROR_MESSAGE,
    GENERIC_FAILURE_MESSAGE,
    PAYLOAD_TOO_LARGE_MESSAGE,
    TIMED_OUT_MESSAGE,
    message_for_ladder_exhausted,
    message_for_terminal_error,
)
from clarif_eye.speech import to_spoken_text


def make_client(handler):
    return OpenRouterClient(api_key="sk-test-not-a-real-key", transport=httpx.MockTransport(handler))


def json_response(status_code, message="failure"):
    return httpx.Response(status_code, json={"error": {"message": message}})


ALL_MESSAGES = {
    "busy": BUSY_MESSAGE,
    "config": CONFIG_ERROR_MESSAGE,
    "payload_too_large": PAYLOAD_TOO_LARGE_MESSAGE,
    "timed_out": TIMED_OUT_MESSAGE,
    "generic": GENERIC_FAILURE_MESSAGE,
}


# --- Category: all rungs rate limited --> busy, not a fault -----------------


def test_all_rungs_rate_limited_yields_busy_message():
    def handler(request):
        return json_response(429, "rate limited")

    client = make_client(handler)
    with pytest.raises(LadderExhaustedError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    message = message_for_ladder_exhausted(exc_info.value)

    assert message == BUSY_MESSAGE
    assert "busy" in message.lower()
    # The expected free-tier state must not sound like a fault.
    assert "broken" not in message.lower()
    assert "error" not in message.lower()


# --- Category: timeouts / budget exhaustion --> distinct "took too long" ----


def test_all_timeouts_yield_timed_out_message():
    def handler(request):
        raise httpx.ReadTimeout("read timed out")

    client = make_client(handler)
    with pytest.raises(LadderExhaustedError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    message = message_for_ladder_exhausted(exc_info.value)

    assert message == TIMED_OUT_MESSAGE
    assert message != BUSY_MESSAGE


def test_budget_exhausted_alone_also_yields_timed_out_message(monkeypatch):
    import time

    from clarif_eye import client as client_module

    monkeypatch.setitem(client_module.ROLE_TIMEOUTS, "eyes", 0.0)

    def handler(request):  # pragma: no cover - budget is spent before any call
        return json_response(200)

    client = make_client(handler)
    with pytest.raises(LadderExhaustedError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    assert all(a.category == "budget_exhausted" for a in exc_info.value.attempts)
    assert message_for_ladder_exhausted(exc_info.value) == TIMED_OUT_MESSAGE


# --- Category: mixed failures --> falls back to the existing generic message


def test_mixed_categories_fall_back_to_generic_message():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return json_response(429, "rate limited")
        return json_response(500, "upstream down")

    client = make_client(handler)
    with pytest.raises(LadderExhaustedError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    message = message_for_ladder_exhausted(exc_info.value)

    assert message == GENERIC_FAILURE_MESSAGE
    assert message not in (BUSY_MESSAGE, TIMED_OUT_MESSAGE)


# --- Category: terminal 401/402/403 --> configuration problem, no retry -----


@pytest.mark.parametrize("status", [401, 402, 403])
def test_terminal_auth_and_credit_statuses_yield_config_message_never_retry(status):
    def handler(request):
        return json_response(status, "terminal failure")

    client = make_client(handler)
    with pytest.raises(OpenRouterError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    message = message_for_terminal_error(exc_info.value)

    assert message == CONFIG_ERROR_MESSAGE
    assert "try again" not in message.lower()
    assert "retry" not in message.lower()


# --- Category: 413 payload too large --> distinct, actionable message -------


def test_413_yields_distinct_payload_too_large_message():
    def handler(request):
        return json_response(413, "payload too large")

    client = make_client(handler)
    with pytest.raises(OpenRouterError) as exc_info:
        client.complete("eyes", [{"role": "user", "content": "hi"}])

    message = message_for_terminal_error(exc_info.value)

    assert message == PAYLOAD_TOO_LARGE_MESSAGE
    assert message != CONFIG_ERROR_MESSAGE
    assert "photo" in message.lower()


# --- Structural dispatch: category/status_code only, never string matching --


def test_dispatch_ignores_detail_text_and_uses_category_only():
    # An Attempt whose free-text detail talks about "rate limited" but whose
    # CATEGORY is server_error must not be treated as rate-limited - proves
    # the mapping reads .category, not .detail.
    attempts = (
        Attempt("model-a", "server_error", 500, "temporarily rate limited upstream"),
    )
    exc = LadderExhaustedError("eyes", attempts)

    assert message_for_ladder_exhausted(exc) == GENERIC_FAILURE_MESSAGE


def test_dispatch_recognizes_rate_limited_category_regardless_of_detail_text():
    attempts = (
        Attempt("model-a", "rate_limited", 429, "the server said everything is fine"),
    )
    exc = LadderExhaustedError("eyes", attempts)

    assert message_for_ladder_exhausted(exc) == BUSY_MESSAGE


def test_terminal_dispatch_uses_status_code_not_message_text():
    exc = OpenRouterError("payload too large, please shrink it", status_code=413)
    assert message_for_terminal_error(exc) == PAYLOAD_TOO_LARGE_MESSAGE

    exc2 = OpenRouterError("everything is fine, try again shortly", status_code=401)
    assert message_for_terminal_error(exc2) == CONFIG_ERROR_MESSAGE


# --- No unhandled exception on an empty attempts tuple ----------------------


def test_empty_attempts_tuple_falls_back_to_generic_message():
    exc = LadderExhaustedError("eyes", ())
    assert message_for_ladder_exhausted(exc) == GENERIC_FAILURE_MESSAGE


def test_missing_api_key_terminal_error_has_no_status_code_and_yields_config_message():
    exc = OpenRouterError("OPENROUTER_API_KEY is required and must not be blank")
    assert exc.status_code is None
    assert message_for_terminal_error(exc) == CONFIG_ERROR_MESSAGE


# --- TTS-safety and honesty across every message -----------------------------


@pytest.mark.parametrize("name,message", list(ALL_MESSAGES.items()))
def test_every_message_is_tts_safe(name, message):
    assert to_spoken_text(message) == message
    assert message.strip() != ""


def test_messages_are_distinct_from_each_other():
    values = list(ALL_MESSAGES.values())
    assert len(set(values)) == len(values)


def test_config_message_never_tells_user_to_retry():
    assert "try again" not in CONFIG_ERROR_MESSAGE.lower()
    assert "retry" not in CONFIG_ERROR_MESSAGE.lower()


def test_actionable_messages_do_tell_user_what_to_do():
    for message in (BUSY_MESSAGE, TIMED_OUT_MESSAGE, PAYLOAD_TOO_LARGE_MESSAGE, GENERIC_FAILURE_MESSAGE):
        assert "try again" in message.lower()
