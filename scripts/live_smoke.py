"""MANUAL-ONLY live smoke test for the OpenRouter client.

This script makes ONE real network call to OpenRouter against the "eyes"
ladder, using OPENROUTER_API_KEY from the real environment. It is NOT part
of the pytest suite (tests must stay offline) and must not be run by an
automated agent - only by a human, or an orchestrator that has explicitly
decided to spend a real API call.

Usage:
    OPENROUTER_API_KEY=... python scripts/live_smoke.py
"""

from clarif_eye.client import LadderExhaustedError, OpenRouterClient, OpenRouterError


def main():
    try:
        client = OpenRouterClient()
        messages = [
            {
                "role": "user",
                "content": "Say a short one-sentence greeting to confirm this smoke test works.",
            }
        ]
        result = client.complete("eyes", messages)
    except LadderExhaustedError as exc:
        print(f"Ladder exhausted for role {exc.role!r}:")
        for model, reason in exc.attempts:
            print(f"  {model}: {reason}")
        raise SystemExit(1)
    except OpenRouterError as exc:
        print(f"Request failed: {exc}")
        raise SystemExit(1)
    else:
        print(f"Served by: {result.model}")
        print(f"Response: {result.content}")


if __name__ == "__main__":
    main()
