"""Category-to-spoken-message mapping for OpenRouter failures (issue #18 / P6.2).

Free-tier limits are 1,000 requests/day and 20/minute, and decision D10
means there is no paid fallback - exhaustion is a real, everyday user-facing
state, not a rare edge case. Before this module, vision.py, synth.py, and
analysis.py each collapsed every failure into roughly the same sentence, so
a user could not tell "the service is busy, try again shortly" from "this
is broken, tell the owner" - two states that call for different actions.

WHY HERE, NOT speech.py: speech.py holds shared TEXT-SANITISING MECHANICS
(to_spoken_text, strip_code_fence) - pure string transforms with no
knowledge of OpenRouterClient's failure model. This module is the opposite:
it knows about client.Attempt / client.LadderExhaustedError /
client.OpenRouterError and turns THEM into words, using speech.to_spoken_text
only as a final safety pass on the fixed message text below. Putting this in
speech.py would make a low-level string-sanitising module depend on the
OpenRouter client; putting it inline in vision.py (as before) meant synth.py
and analysis.py each had to keep their own copy in sync by hand. One module,
imported by all three nodes and by ui.py, removes both problems.

WHY NOT STRING MATCHING: Attempt.category and OpenRouterError.status_code
are the structured facts client.py already records (owner decision D15) so
this issue would not have to parse English prose. Every decision below reads
one of those two fields - never .detail or str(exc) - so a rewording of an
upstream detail string can never silently change which message a user
hears. See tests/test_failure_messages.py's test_dispatch_* cases, which
prove this by feeding in detail text that would mislead a substring match.

WHAT'S NOT HERE: retry/backoff logic. client.py already fails over across
the model ladder on 429/5xx/model-unavailable; adding a second retry layer
on top of that would just spend more of the user's wait budget for the same
rate limit, and free-tier limits are the thing being hit, not a transient
blip a delay would fix. This module only chooses WORDS for a failure that
already happened.

WHAT'S NOT DISTINGUISHED: the daily (1,000/day) vs per-minute (20/min) rate
limit. client.py's Attempt only records category="rate_limited" and the
HTTP 429 status; OpenRouter's error body carries no field this client reads
that says which limit was hit (see client._describe_failure, which reads
only error.message as free text, and the terminal-status table, which has
no 429 entry at all - 429 always fails over, never terminal). Inventing a
distinction the response does not carry would be a guess dressed up as
information, so every rate-limited exhaustion gets the same BUSY_MESSAGE
regardless of which limit was actually hit.
"""

from clarif_eye.speech import to_spoken_text

# --- The messages, one per category --------------------------------------
#
# Each is TTS-safe by construction (plain prose, no markup) and additionally
# run through to_spoken_text so a test can prove that with no special-casing
# (see tests/test_failure_messages.py::test_every_message_is_tts_safe).
# Short on purpose: every word here is spoken aloud before the user gets to
# try again, and length is an accessibility cost (docs/ACCESSIBILITY.md).

# All ladder rungs came back rate-limited: the expected free-tier state
# (D10, no paid fallback), not a fault - must not sound broken.
BUSY_MESSAGE = to_spoken_text(
    "The service is busy right now. Please try again in a few minutes."
)

# A terminal 401/402/403, a missing/blank API key, or any other
# configuration-shaped failure with no HTTP status to read. Retrying cannot
# help any of these, so this never tells the user to try again.
CONFIG_ERROR_MESSAGE = to_spoken_text(
    "This could not run because of a configuration problem with the "
    "service. Please tell whoever set this up."
)

# A terminal 413: the request itself, not the service, is the problem, and
# it is something the user can fix.
PAYLOAD_TOO_LARGE_MESSAGE = to_spoken_text(
    "The photo was too large to process. Please try again with a smaller "
    "or closer photo."
)

# Every ladder rung timed out, or the role's time budget ran out before a
# rung could even be tried. Distinct from BUSY_MESSAGE: this is a latency
# problem, not upstream capacity, though the user action is the same.
TIMED_OUT_MESSAGE = to_spoken_text(
    "This took too long to complete. Please try again."
)

# Anything else: mixed categories, server errors, model-unavailable, or an
# empty attempts tuple. This is the pre-existing catch-all wording (kept
# word-for-word so tests/test_vision.py's/test_synth.py's/
# test_analysis.py's existing "mentions='busy'" assertions - written before
# this module existed - still hold for this fallback case).
GENERIC_FAILURE_MESSAGE = to_spoken_text(
    "This could not run right now because every available model was busy "
    "or unavailable. Please try again in a moment."
)


def message_for_ladder_exhausted(exc):
    """Spoken message for a LadderExhaustedError, chosen from its Attempt
    categories alone (never from any attempt's .detail text).

    - every attempt category == "rate_limited" -> BUSY_MESSAGE.
    - every attempt category in {"timeout", "budget_exhausted"} -> TIMED_OUT_MESSAGE.
    - anything else (mixed categories, server errors, model-unavailable,
      or an empty attempts tuple) -> GENERIC_FAILURE_MESSAGE.
    """
    categories = {attempt.category for attempt in exc.attempts}
    if categories == {"rate_limited"}:
        return BUSY_MESSAGE
    if categories and categories <= {"timeout", "budget_exhausted"}:
        return TIMED_OUT_MESSAGE
    return GENERIC_FAILURE_MESSAGE


def message_for_terminal_error(exc):
    """Spoken message for an OpenRouterError raised OUTSIDE a ladder
    (client construction, or one of client.py's terminal statuses).

    Dispatches on exc.status_code - set by client.py for its terminal
    statuses (401/402/403/413) and for a malformed 400 request; None for a
    missing/blank API key or an unknown role. Never reads exc's message
    text.
    """
    if getattr(exc, "status_code", None) == 413:
        return PAYLOAD_TOO_LARGE_MESSAGE
    return CONFIG_ERROR_MESSAGE
