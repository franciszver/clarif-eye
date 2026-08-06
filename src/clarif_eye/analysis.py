"""Deep-analysis node logic (issue #8 / P1.5): dense documents -> accurate spoken script.

This is the second stage of the RESEARCH path (research -> analysis), taken
for DENSE, INFORMATION-RICH photos - bills, receipts, medication labels,
forms, letters - where detail matters more than brevity. It is also the
FIRST use of the `brain` ladder (client.ROLE_TIMEOUTS["brain"], 45s
ceiling): vision.py and synth.py both call the `eyes` ladder; this module
must call "brain".

Inputs: ocr_output and scene_context (from vision.py, same as synth.py) plus
scraper_data (from research.py's web lookup, which may be "" - see the
module docstring in state.py: "" currently means both "research ran and
found nothing" and "not applicable". For THIS node, "" is simply treated as
"no external context available" and the model proceeds using ocr_output/
scene_context alone - no sentinel is invented, and an empty scrape must
never produce a hedged, contentless script).

THE CENTRAL RISK: hallucinated detail. A blind user cannot see the source
document and cannot check the script against it, so the model inventing a
plausible-sounding amount, date, or account number is worse than saying
less. SYNTH_PROMPT below instructs the model to reproduce numbers, dates,
amounts and identifiers EXACTLY as they appear, and to say only what it can
support from the given text and stop rather than fill gaps - this is a
prompting decision, not something this module can verify after the fact.

Follows synth.py's contract exactly otherwise: this module must NEVER let a
raw exception escape into the graph. Every failure mode (LadderExhaustedError,
a terminal OpenRouterError, an unexpected Exception, a non-string reply, an
empty/whitespace reply, a reply that sanitises to blank) degrades into a
plain-English, TTS-safe placeholder message instead, so the graph can still
reach tts/END. scene_context being one of vision.py's own degradation
messages is detected the same way synth.py does it - via vision.py's public
is_degraded_scene predicate, never by matching prose.

THE OUTPUT IS SPOKEN ALOUD, so it is always run through speech.to_spoken_text
before being returned - never trust the model to actually follow the
"no markup" instruction in the prompt.
"""

from clarif_eye.client import LadderExhaustedError, OpenRouterClient, OpenRouterError
from clarif_eye.speech import to_spoken_text as _to_spoken_text
from clarif_eye.vision import is_degraded_scene

ANALYSIS_PROMPT = (
    "You are the final stage of an assistant that describes photos aloud "
    "for a visually impaired user, working on a DENSE, information-rich "
    "document (a bill, receipt, medication label, form, or letter) where "
    "getting the details right matters more than being brief. You will be "
    "given any text found in the photo, a description of the scene, and "
    "(if available) extra context from a web lookup. "
    "Lead with what the document IS, then state the facts that matter most "
    "to someone who cannot see it. "
    "Reproduce every number, date, amount, and identifier EXACTLY as it "
    "appears in the given text - do not round, paraphrase, or reformat "
    "them. Do NOT invent or guess at any detail that is not present in the "
    "given text: if the text is thin or no web context is available, say "
    "what you can from what is given and then stop, rather than filling "
    "gaps with a plausible-sounding guess. "
    "Reply with PLAIN PROSE ONLY: no markdown, no headings, no bullet "
    "points or numbered lists, no tables, no pipe characters, no emoji, "
    "no code blocks or backticks, and no URLs. Just the words that should "
    "be read aloud, nothing else."
)


def _default_client():
    """Factory for the real client. Called lazily (never at import time)."""
    return OpenRouterClient()


def _build_messages(ocr_output, scene_context, scraper_data):
    if ocr_output:
        body = f"Text found in the photo: {ocr_output}\n\nScene description: {scene_context}"
    else:
        body = f"No text was found in the photo.\n\nScene description: {scene_context}"
    if scraper_data:
        body += f"\n\nAdditional context from a web lookup: {scraper_data}"
    return [{"role": "user", "content": [{"type": "text", "text": f"{ANALYSIS_PROMPT}\n\n{body}"}]}]


def _degraded(message):
    return {"final_output": _to_spoken_text(message)}


def run_analysis(ocr_output, scene_context, scraper_data, client=None):
    """Call the brain ladder to turn (ocr_output, scene_context, scraper_data) into final_output.

    `client` is injectable (tests pass a fake); when omitted, a real
    OpenRouterClient is constructed lazily. A client built here (not
    injected) is closed in a `finally` before returning; an injected client
    is owned by the caller and is never closed here. Mirrors
    synth.run_fast_synth's structure exactly, with role "brain" instead of
    "eyes" and scraper_data folded into the request.
    """
    ocr_output = ocr_output or ""
    scene_context = scene_context or ""
    scraper_data = scraper_data or ""

    if is_degraded_scene(scene_context):
        # Same deliberate choice as synth.py (FIX 8b): a vision failure
        # message is not safe to feed to the model as if it were a real
        # scene description, so the model is not called - but any real OCR
        # text that came with it is still appended to what gets read aloud
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
            result = client.complete("brain", _build_messages(ocr_output, scene_context, scraper_data))
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
        return _degraded("The analysis model returned an empty response.")

    spoken = _to_spoken_text(reply)
    if not spoken:
        return _degraded("The analysis model returned an empty response.")

    return {"final_output": spoken}
