"""Thin OpenRouter chat-completion client with ladder failover.

Sends a chat completion for a role ("eyes" or "brain") and walks that
role's model ladder (src/clarif_eye/registry.py) in order on failure.

Config is read from the environment only (OPENROUTER_API_KEY required;
OPENROUTER_APP_URL / OPENROUTER_APP_NAME optional). A missing or blank key
fails fast at construction time, not at request time.

Security: the API key must never appear in a log line, exception message,
repr(), or str() of anything this module creates. See tests/test_client.py
for the leak-prevention tests.
"""

import os
import time
from dataclasses import dataclass

import httpx

from clarif_eye.registry import load_registry

API_BASE_URL = "https://openrouter.ai/api/v1"

# Per-role latency budgets (seconds), from the product spec. This is a TOTAL
# deadline for the whole complete() call (a UX contract for a blind user
# waiting on spoken feedback), not a per-attempt timeout - each rung gets
# whatever time is left in the budget, never more.
ROLE_TIMEOUTS = {
    "eyes": 8.0,
    "brain": 25.0,
}
DEFAULT_TIMEOUT = 25.0

# Known model-availability error-message signatures, derived from live
# OpenRouter responses recorded in prd/openrouter-error-shapes.md. Matched
# case-insensitively against error.message only (not error.code, which is
# just the HTTP status and matches everything). Must be revisited if
# OpenRouter changes its error text.
_MODEL_NOT_FOUND_SIGNATURES = (
    "is not a valid model id",
    "no endpoints found",
    "no allowed providers",
    "model not found",
)

# HTTP statuses whose outcome cannot change by trying another model: same
# key, same credit, same payload every time. Advancing the ladder on these
# just burns round-trips and reports a "model" failure when the real cause
# is the key, credit, or the request itself.
_TERMINAL_STATUS_REASONS = {
    401: "authentication failed - check OPENROUTER_API_KEY",
    403: "authorization failed - check OPENROUTER_API_KEY",
    402: "out of credit on this OpenRouter account",
    413: "request payload too large",
}


class OpenRouterError(Exception):
    """Base error for OpenRouter client configuration and usage failures."""


class LadderExhaustedError(OpenRouterError):
    """Raised when every model in a role's ladder failed.

    Carries the role and, for each model tried, why it failed - enough
    detail for a caller to build a spoken error message (issue #18).
    """

    def __init__(self, role, attempts):
        self.role = role
        self.attempts = attempts  # tuple of (model_id, reason) pairs
        tried = "; ".join(f"{model}: {reason}" for model, reason in attempts)
        super().__init__(
            f"OpenRouter ladder exhausted for role {role!r} "
            f"({len(attempts)} model(s) tried): {tried}"
        )


@dataclass(frozen=True)
class CompletionResult:
    """A successful completion: the assistant text and which model served it."""

    content: str
    model: str


class OpenRouterClient:
    """Chat-completion client that fails over across a role's model ladder."""

    def __init__(self, *, api_key=None, app_url=None, app_name=None, transport=None):
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key or not api_key.strip():
            raise OpenRouterError("OPENROUTER_API_KEY is required and must not be blank")

        if app_url is None:
            app_url = os.environ.get("OPENROUTER_APP_URL")
        if app_name is None:
            app_name = os.environ.get("OPENROUTER_APP_NAME")

        headers = {"Authorization": f"Bearer {api_key}"}
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_name:
            headers["X-Title"] = app_name

        self._client = httpx.Client(base_url=API_BASE_URL, headers=headers, transport=transport)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return repr(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._client.close()

    def complete(self, role, messages, **params):
        """Run a chat completion for `role`, trying each ladder rung in order.

        `messages` and any extra `params` (e.g. temperature) are sent as-is
        in the OpenAI-compatible chat completions request body.

        The role's latency budget (ROLE_TIMEOUTS) is a TOTAL deadline for
        this call, not a per-attempt timeout: elapsed time is tracked with
        time.monotonic() and each attempt gets only what's left of the
        budget. Once the budget is exhausted, remaining rungs are skipped
        immediately.

        Returns a CompletionResult. Raises LadderExhaustedError if every
        rung fails (or the budget runs out), or OpenRouterError immediately
        for a terminal failure (auth/credit/payload) or a malformed request
        (a caller bug, not a ladder failure).
        """
        ladder = load_registry().ladder(role)
        budget = ROLE_TIMEOUTS.get(role, DEFAULT_TIMEOUT)
        deadline = time.monotonic() + budget

        attempts = []
        for index, model in enumerate(ladder):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                skipped = len(ladder) - index
                attempts.append(
                    (
                        model,
                        f"budget exhausted ({budget}s role budget used up); "
                        f"{skipped} rung(s) never tried",
                    )
                )
                break

            payload = {"model": model, "messages": messages, **params}
            try:
                response = self._client.post("/chat/completions", json=payload, timeout=remaining)
            except httpx.TimeoutException:
                attempts.append((model, "request timed out"))
                continue
            except httpx.HTTPError as exc:
                attempts.append((model, f"transport error: {type(exc).__name__}"))
                continue

            if response.status_code == 200:
                result = self._parse_success(response, model)
                if result is None:
                    attempts.append((model, "malformed or empty response body"))
                    continue
                return result

            reason = self._describe_failure(response)

            if response.status_code in _TERMINAL_STATUS_REASONS:
                raise OpenRouterError(
                    f"OpenRouter request failed for model {model!r} "
                    f"(HTTP {response.status_code}): {_TERMINAL_STATUS_REASONS[response.status_code]} "
                    f"({reason})"
                )

            if response.status_code == 429:
                attempts.append((model, f"rate limited (429): {reason}"))
                continue
            if response.status_code >= 500:
                attempts.append((model, f"server error ({response.status_code}): {reason}"))
                continue
            # OpenRouter has been observed to return 400 (not 404) for an
            # unknown model - see prd/openrouter-error-shapes.md. This 404
            # branch is likely dead against OpenRouter itself, but is kept
            # as defensive failover since other OpenAI-compatible proxies
            # do use 404 for the same condition.
            if response.status_code == 404:
                attempts.append((model, f"model not found (404): {reason}"))
                continue
            if response.status_code == 400:
                if self._is_model_not_found(response):
                    attempts.append((model, f"model not found (400): {reason}"))
                    continue
                raise OpenRouterError(
                    f"OpenRouter rejected the request as malformed (HTTP 400) "
                    f"for model {model!r}: {reason}"
                )
            attempts.append((model, f"unexpected status {response.status_code}: {reason}"))

        raise LadderExhaustedError(role, tuple(attempts))

    @staticmethod
    def _parse_success(response, model):
        """Defensively parse an HTTP 200 body into a CompletionResult.

        Returns None (a failed rung, not an exception) if the body isn't
        valid JSON, lacks the expected shape, or yields null/empty/
        whitespace-only content - never lets a raw IndexError/KeyError
        escape and never returns a CompletionResult with blank content.
        """
        try:
            data = response.json()
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        return CompletionResult(content=content, model=model)

    @staticmethod
    def _describe_failure(response):
        try:
            data = response.json()
        except ValueError:
            return response.text[:200]
        if isinstance(data, dict):
            message = data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else None
            if message:
                return str(message)
        return response.text[:200]

    @staticmethod
    def _is_model_not_found(response):
        """Distinguish a 400 for an unknown/invalid model from a caller bug.

        OpenRouter uses the same 400 status for both. We only advance the
        ladder when the error message matches a known model-availability
        signature (see _MODEL_NOT_FOUND_SIGNATURES); anything else is
        treated as a caller-side malformed request and must surface
        immediately instead of silently burning every rung.
        """
        try:
            data = response.json()
        except ValueError:
            return False
        error = data.get("error") if isinstance(data, dict) else None
        if not isinstance(error, dict):
            return False
        message = str(error.get("message", "")).lower()
        return any(signature in message for signature in _MODEL_NOT_FOUND_SIGNATURES)
