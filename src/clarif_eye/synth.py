"""Fast-synthesis node logic (issue #7 / P1.4): turn vision output into a spoken script.

Takes ocr_output + scene_context (already produced by vision.py) and asks
the `eyes` ladder (same role/budget as vision - see client.ROLE_TIMEOUTS)
to weave them into a short, linear, conversational script for
text-to-speech. Follows vision.py's contract exactly: this module must
NEVER let a raw exception escape into the graph, and every failure mode
(LadderExhaustedError, a terminal OpenRouterError, a non-string reply, an
empty reply) degrades into a plain-English, TTS-safe placeholder message
instead, so the graph can still reach tts/END.

THE OUTPUT IS SPOKEN ALOUD, so it is sanitised with _to_spoken_text before
being returned - never trust the model to actually follow the "no markup"
instruction in the prompt.

Two input edge cases the vision node's own degradation path produces
regularly (see vision.py's module docstring):
  - ocr_output == "" with a real scene_context (a photo with no text) -
    the model IS asked to describe the scene, just without any mention of
    text.
  - ocr_output == "" with scene_context holding one of vision.py's own
    degradation messages - the model is NOT called at all. Asking it to
    "describe" a failure message would either fabricate a description of a
    photo that was never actually seen, or just echo the failure message
    back dressed up as if it were a real scene description. Instead the
    already-written, already-TTS-safe degradation message is sanitised and
    passed straight through as final_output.
"""

from clarif_eye.client import LadderExhaustedError, OpenRouterClient, OpenRouterError
from clarif_eye.speech import to_spoken_text as _to_spoken_text
from clarif_eye.vision import is_degraded_scene

SYNTH_PROMPT = (
    "You are the final stage of an assistant that describes photos aloud "
    "for a visually impaired user. You will be given any text found in the "
    "photo and a description of the scene. Combine them into a short, "
    "natural, spoken description a person could listen to, in plain "
    "conversational sentences (for example: \"The image shows a tax "
    "document that reads...\"). "
    "Reply with PLAIN PROSE ONLY: no markdown, no headings, no bullet "
    "points or numbered lists, no tables, no pipe characters, no emoji, "
    "no code blocks or backticks, and no URLs. Just the words that should "
    "be read aloud, nothing else."
)

# Sanitisation (_to_spoken_text) lives in clarif_eye.speech and is imported
# above - shared with vision.py so issue #8's analysis node can reuse the
# same mechanics instead of copying them.

# Detecting "scene_context is actually one of vision.py's own degradation
# messages, not a real scene description" is vision.py's job (FIX 7): it
# owns those messages and exposes a structural predicate for them, so a
# rewording of the message text there can't silently break detection here.


def _default_client():
    """Factory for the real client. Called lazily (never at import time)."""
    return OpenRouterClient()


def _build_messages(ocr_output, scene_context):
    if ocr_output:
        body = f"Text found in the photo: {ocr_output}\n\nScene description: {scene_context}"
    else:
        body = f"No text was found in the photo.\n\nScene description: {scene_context}"
    return [{"role": "user", "content": [{"type": "text", "text": f"{SYNTH_PROMPT}\n\n{body}"}]}]


def _degraded(message):
    return {"final_output": _to_spoken_text(message)}


def run_fast_synth(ocr_output, scene_context, client=None):
    """Call the eyes ladder to turn (ocr_output, scene_context) into final_output.

    `client` is injectable (tests pass a fake); when omitted, a real
    OpenRouterClient is constructed lazily. A client built here (not
    injected) is closed in a `finally` before returning; an injected client
    is owned by the caller and is never closed here. See vision.run_vision
    for the identical pattern this mirrors.
    """
    ocr_output = ocr_output or ""
    scene_context = scene_context or ""

    if is_degraded_scene(scene_context):
        # Deliberate choice (FIX 8b): scene_context being a vision failure
        # message does not mean any real ocr_output that came with it is
        # worthless - run_fast_synth is public and issue #8 may compose
        # state differently than the current graph does (today vision.py's
        # own _degraded() always pairs a degradation message with
        # ocr_output == "", so this combination isn't reachable through the
        # graph yet, but this function must not silently drop real OCR text
        # if that ever changes). The description failed and is not safe to
        # feed to the model as if it were real, so the model is still not
        # called - but any OCR text is appended to what gets read aloud
        # instead of being discarded.
        if ocr_output.strip():
            return _degraded(
                f"{scene_context} However, this text was found in the photo: {ocr_output}"
            )
        return _degraded(scene_context)

    if not scene_context.strip():
        return _degraded("No description is available for this photo.")

    owns_client = client is None
    if owns_client:
        try:
            client = _default_client()
        except OpenRouterError:
            return _degraded(
                "The spoken description could not be prepared because of a "
                "configuration problem with the service. Please tell "
                "whoever set this up."
            )
    try:
        try:
            result = client.complete("eyes", _build_messages(ocr_output, scene_context))
        except LadderExhaustedError:
            return _degraded(
                "The spoken description could not be prepared right now: "
                "every available model was busy or unavailable. Please try "
                "again in a moment."
            )
        except OpenRouterError:
            return _degraded(
                "The spoken description could not be prepared because of a "
                "configuration problem with the service. Please tell "
                "whoever set this up."
            )
        except Exception:
            # Contract (module docstring): no raw exception may escape into
            # the graph. Catches everything else an injected client could
            # raise, without swallowing KeyboardInterrupt/SystemExit, which
            # derive from BaseException, not Exception.
            return _degraded(
                "The spoken description could not be prepared because of an "
                "unexpected internal error. Please try again, and tell "
                "whoever set this up if it keeps happening."
            )
    finally:
        if owns_client:
            client.close()

    reply = result.content
    if not isinstance(reply, str) or not reply.strip():
        return _degraded("The synthesis model returned an empty response.")

    spoken = _to_spoken_text(reply)
    if not spoken:
        return _degraded("The synthesis model returned an empty response.")

    return {"final_output": spoken}
