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

import re

from clarif_eye.client import LadderExhaustedError, OpenRouterClient, OpenRouterError
from clarif_eye.vision import _strip_code_fence

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

# vision.py's own degraded() messages all begin with one of these two
# phrases (see vision.py's four literal degradation strings). Matched by
# prefix, not full equality, so a minor rewording of the message text
# doesn't silently break this detection.
_VISION_DEGRADATION_PREFIXES = ("Vision could not run", "The vision model")

# --- Sanitisation -------------------------------------------------------

# Markdown bold/italic/underline markers: **x**, __x__, *x*, _x_. Just the
# marker characters are dropped; the wrapped words are spoken text, kept.
_BOLD_ITALIC_RE = re.compile(r"(\*\*|__|\*|_)")
# A line that starts (after whitespace) with a markdown heading marker.
_HEADING_RE = re.compile(r"^\s*#+\s*", re.MULTILINE)
# A line that starts (after whitespace) with a bullet marker.
_BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
# A line that starts (after whitespace) with a numbered-list marker.
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
# Table separator rows, e.g. "|------|-------|" or "---|---".
_TABLE_SEPARATOR_RE = re.compile(r"^\s*[-:|\s]*\|[-:|\s]*$", re.MULTILINE)
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "]"
)
# A punctuation character repeated 3+ times in a row (e.g. "!!!", "----",
# "..."), collapsed to a single occurrence rather than read aloud as noise.
_PUNCT_RUN_RE = re.compile(r"([^\w\s])\1{2,}")


def _to_spoken_text(text):
    """Strip/normalise markup so `text` is safe to hand to text-to-speech.

    Defensive, not cosmetic: models routinely ignore SYNTH_PROMPT's "plain
    prose only" instruction, so this runs on every reply regardless of what
    was asked for.
    """
    text = _strip_code_fence(text)
    text = text.replace("`", "")
    text = _TABLE_SEPARATOR_RE.sub(" ", text)
    text = _HEADING_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _NUMBERED_RE.sub("", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = text.replace("|", ", ")
    text = _URL_RE.sub("a web link", text)
    text = _EMOJI_RE.sub("", text)
    text = _PUNCT_RUN_RE.sub(r"\1", text)
    # Flatten to a single line of flowing prose: a list turned into
    # separate short lines above should read as one continuous script, not
    # be read aloud with unnatural pauses between fragments.
    lines = [line.strip() for line in text.split("\n")]
    text = " ".join(line for line in lines if line)
    text = re.sub(r"\s+", " ", text).strip()
    # Nothing but leftover punctuation/whitespace (e.g. a reply that was
    # pure markup noise) is not speakable content - treat it the same as
    # an empty reply rather than reading stray commas and pipes aloud.
    if not re.search(r"\w", text):
        return ""
    return text


def _looks_like_vision_degradation(scene_context):
    stripped = scene_context.strip()
    return any(stripped.startswith(prefix) for prefix in _VISION_DEGRADATION_PREFIXES)


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

    if _looks_like_vision_degradation(scene_context):
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
