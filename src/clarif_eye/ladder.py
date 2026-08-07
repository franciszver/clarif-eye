"""One shared way to call a model ladder and degrade instead of raising.

THE DUPLICATION THIS CLOSES: synth.py, analysis.py and followup.py each had
a byte-identical copy of the same twenty lines - construct a client if none
was injected, call client.complete(role, messages), map
LadderExhaustedError / a terminal OpenRouterError / any other Exception onto
a spoken-ready message, and close the client in a `finally` if and only if
this code built it. Three copies of a block whose whole job is "never let an
exception escape into the graph" is three places for that contract to be
edited unevenly - one copy quietly losing its bare `except Exception`, say,
would reintroduce exactly the crash every one of those modules' docstrings
promises cannot happen.

WHAT STAYS WITH THE CALLER, ON PURPOSE: the prompt, the role, the
degradation WORDING, and what to do with a successful reply. This module
knows how to make the call safely; it does not know what any particular node
is for. That split is why the return shape is a plain
(result, failure_message) pair rather than something cleverer - the caller
still owns turning a failure message into its own module's state update (its
own `_degraded`, which sanitises for speech), so nothing about how a node
speaks moves in here.

WHY `client_factory` IS PASSED IN rather than imported here: each calling
module keeps its own `_default_client`, and the existing tests monkeypatch
that module attribute by name (see tests/test_synth.py,
tests/test_analysis.py). Passing the function in at the call site keeps that
seam working exactly as it did - the name is resolved from the caller's
module globals when the caller runs, so a monkeypatched factory is the one
that gets used.
"""

from clarif_eye.client import LadderExhaustedError, OpenRouterError
from clarif_eye.failure_messages import (
    message_for_ladder_exhausted,
    message_for_terminal_error,
)


def call_ladder(role, messages, client, client_factory, unexpected_error_message):
    """Call `role`'s ladder with `messages`, never raising.

    Returns a (result, failure_message) pair, exactly one of which is None:
      - (CompletionResult, None) when the call succeeded.
      - (None, "a plain-English message") for every failure mode, so the
        caller can hand it to its own `_degraded` and let the graph reach
        tts/END with something speakable.

    `client` is the injected client, or None to have one built from
    `client_factory` for the duration of this call and closed afterwards. A
    client this function built is closed in a `finally`; an INJECTED client
    is owned by the caller and is never closed here - the same ownership
    rule every call site documented before this helper existed.

    `unexpected_error_message` is the caller's own wording for the
    catch-all branch, which is the one place the three call sites genuinely
    differ ("The spoken description could not be prepared..." for a node
    writing a description, "The answer could not be prepared..." for the one
    answering a question). Passing it in keeps that difference visible at
    the call site instead of hiding a lookup table in here.

    The bare `except Exception` is deliberate and is the contract every
    calling module's docstring makes: an injected client can raise anything
    at all, and none of it may escape into the graph. It does NOT catch
    KeyboardInterrupt/SystemExit, which derive from BaseException.
    """
    owns_client = client is None
    if owns_client:
        try:
            client = client_factory()
        except OpenRouterError as exc:
            return None, message_for_terminal_error(exc)
    try:
        try:
            return client.complete(role, messages), None
        except LadderExhaustedError as exc:
            return None, message_for_ladder_exhausted(exc)
        except OpenRouterError as exc:
            return None, message_for_terminal_error(exc)
        except Exception:
            return None, unexpected_error_message
    finally:
        if owns_client:
            client.close()
