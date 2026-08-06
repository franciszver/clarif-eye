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

from clarif_eye.registry import RegistryError, load_registry

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


@dataclass(frozen=True)
class Attempt:
    """One ladder rung's outcome - machine-readable, for issue #18.

    `category` is drawn from a fixed, documented set so a caller can build a
    spoken message (e.g. "everything was rate-limited" vs "configuration is
    broken, do not retry") without substring-matching upstream prose:

    - "rate_limited"     - HTTP 429
    - "server_error"     - HTTP 5xx
    - "model_unavailable" - model not found / no endpoints (404, or 400
      matching a known model-availability signature)
    - "timeout"          - the attempt raised httpx.TimeoutException
    - "transport_error"  - any other httpx.HTTPError (connection failure etc.)
    - "bad_response"     - HTTP 200 with a malformed/empty body
    - "budget_exhausted" - the role's total time budget ran out before this
      rung could be tried
    - "unexpected_status" - any other HTTP status not covered above

    `detail` keeps the human/upstream text for logs and debugging.
    """

    model: str
    category: str
    status_code: int | None
    detail: str


class LadderExhaustedError(OpenRouterError):
    """Raised when every model in a role's ladder failed.

    Carries the role and, for each model tried (or never reached because the
    budget ran out), a structured Attempt - enough detail for a caller to
    build a spoken error message (issue #18) without parsing English prose.
    """

    def __init__(self, role, attempts):
        self.role = role
        self.attempts = attempts  # tuple of Attempt
        tried = "; ".join(
            f"{a.model} [{a.category}"
            + (f" HTTP {a.status_code}" if a.status_code is not None else "")
            + f"]: {a.detail}"
            for a in attempts
        )
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

    def __init__(
        self, *, api_key=None, app_url=None, app_name=None, transport=None, registry=None
    ):
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
        # Loaded once here (not per-request): a request-time re-read was
        # 4 disk reads + TOML parses for 4 requests, and ran outside the
        # role's time budget. `registry` is injectable for tests.
        self._registry = registry if registry is not None else load_registry()

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
        this call, not a per-attempt timeout: the clock starts at the very
        top of this method (before the registry lookup or anything else),
        elapsed time is tracked with time.monotonic(), and each attempt gets
        only what's left of the budget. Once the budget is exhausted,
        remaining rungs are recorded as skipped and never tried.

        Returns a CompletionResult. Raises LadderExhaustedError if every
        rung fails (or the budget runs out), or OpenRouterError immediately
        for a terminal failure (auth/credit/payload), an unknown role, or a
        malformed request (a caller bug, not a ladder failure).
        """
        budget = ROLE_TIMEOUTS.get(role, DEFAULT_TIMEOUT)
        deadline = time.monotonic() + budget

        try:
            ladder = self._registry.ladder(role)
        except RegistryError as e:
            raise OpenRouterError(f"invalid role {role!r}: {e}") from e

        attempts = []
        for index, model in enumerate(ladder):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                for untried_model in ladder[index:]:
                    attempts.append(
                        Attempt(
                            untried_model,
                            "budget_exhausted",
                            None,
                            f"role budget of {budget}s exhausted before this rung could be tried",
                        )
                    )
                break

            payload = {"model": model, "messages": messages, **params}
            try:
                response = self._client.post(
                    "/chat/completions", json=payload, timeout=self._build_timeout(remaining)
                )
            except httpx.TimeoutException:
                attempts.append(Attempt(model, "timeout", None, "request timed out"))
                continue
            except httpx.HTTPError as exc:
                attempts.append(
                    Attempt(
                        model, "transport_error", None, f"transport error: {type(exc).__name__}"
                    )
                )
                continue

            if response.status_code == 200:
                result = self._parse_success(response, model)
                if result is None:
                    attempts.append(
                        Attempt(model, "bad_response", 200, "malformed or empty response body")
                    )
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
                attempts.append(Attempt(model, "rate_limited", 429, reason))
                continue
            if response.status_code >= 500:
                attempts.append(
                    Attempt(model, "server_error", response.status_code, reason)
                )
                continue
            # OpenRouter has been observed to return 400 (not 404) for an
            # unknown model - see prd/openrouter-error-shapes.md. This 404
            # branch is likely dead against OpenRouter itself, but is kept
            # as defensive failover since other OpenAI-compatible proxies
            # do use 404 for the same condition.
            if response.status_code == 404:
                attempts.append(Attempt(model, "model_unavailable", 404, reason))
                continue
            if response.status_code == 400:
                if self._is_model_not_found(response):
                    attempts.append(Attempt(model, "model_unavailable", 400, reason))
                    continue
                raise OpenRouterError(
                    f"OpenRouter rejected the request as malformed (HTTP 400) "
                    f"for model {model!r}: {reason}"
                )
            attempts.append(
                Attempt(model, "unexpected_status", response.status_code, reason)
            )

        raise LadderExhaustedError(role, tuple(attempts))

    @staticmethod
    def _build_timeout(remaining):
        """Build a per-phase httpx.Timeout whose phases sum to <= `remaining`.

        A bare float (e.g. `timeout=remaining`) is NOT a total deadline in
        httpx: it expands to Timeout(connect=remaining, read=remaining,
        write=remaining, pool=remaining) - every phase gets its OWN
        full-length timer. Verified: httpx.Timeout(8.0) has connect=read=
        write=pool=8.0, so a single attempt's worst case (connect+write+
        read) could be ~3x the role's total budget. Do NOT "simplify" this
        back to a bare float - it silently breaks the total-budget contract
        this module documents and tests (test_total_budget_is_enforced...).

        `remaining` is always > 0 here (callers check remaining <= 0 before
        calling this). connect and pool share a cap (a stuck TCP handshake
        and a stuck pool-checkout are the same class of failure); write gets
        a smaller slice since our request bodies are small; read gets
        whatever is left, since waiting on the model's response is the
        expected dominant cost.
        """
        connect = min(remaining * 0.3, 5.0)
        write = remaining * 0.2
        read = max(remaining - connect - write, 0.001)
        pool = connect
        return httpx.Timeout(connect=connect, read=read, write=write, pool=pool)

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
