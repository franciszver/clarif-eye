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
from dataclasses import dataclass

import httpx

from clarif_eye.registry import load_registry

API_BASE_URL = "https://openrouter.ai/api/v1"

# Per-role latency budgets (seconds), from the product spec. Used as the
# request timeout so a stuck rung can't hang forever.
ROLE_TIMEOUTS = {
    "eyes": 8.0,
    "brain": 25.0,
}
DEFAULT_TIMEOUT = 25.0


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

    def close(self):
        self._client.close()

    def complete(self, role, messages, **params):
        """Run a chat completion for `role`, trying each ladder rung in order.

        `messages` and any extra `params` (e.g. temperature) are sent as-is
        in the OpenAI-compatible chat completions request body.

        Returns a CompletionResult. Raises LadderExhaustedError if every
        rung fails, or OpenRouterError immediately if the request itself is
        malformed (a caller bug, not a ladder failure).
        """
        ladder = load_registry().ladder(role)
        timeout = ROLE_TIMEOUTS.get(role, DEFAULT_TIMEOUT)

        attempts = []
        for model in ladder:
            payload = {"model": model, "messages": messages, **params}
            try:
                response = self._client.post("/chat/completions", json=payload, timeout=timeout)
            except httpx.TimeoutException:
                attempts.append((model, "request timed out"))
                continue
            except httpx.HTTPError as exc:
                attempts.append((model, f"transport error: {type(exc).__name__}"))
                continue

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return CompletionResult(content=content, model=model)

            reason = self._describe_failure(response)

            if response.status_code == 429:
                attempts.append((model, f"rate limited (429): {reason}"))
                continue
            if response.status_code >= 500:
                attempts.append((model, f"server error ({response.status_code}): {reason}"))
                continue
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
        ladder when the error clearly names the model as the problem;
        anything else is treated as a caller-side malformed request and
        must surface immediately instead of silently burning every rung.
        """
        try:
            data = response.json()
        except ValueError:
            return False
        error = data.get("error") if isinstance(data, dict) else None
        if not isinstance(error, dict):
            return False
        text = f"{error.get('message', '')} {error.get('code', '')}".lower()
        return "model" in text
