"""Vision node logic (issue #5 / P1.2): OCR + scene description.

Builds an OpenAI-compatible multimodal request (image sent as a `data:`
URI) and calls OpenRouterClient.complete("eyes", messages). This module
must NEVER let a raw exception escape into the graph - the person on the
other end is relying on spoken feedback, not a stack trace. Every failure
mode (LadderExhaustedError, a terminal OpenRouterError, a malformed reply,
an empty reply) degrades into a plain-English placeholder message plus a
complexity_flag, so the graph can still route and reach tts/END. Issue #18
owns turning these placeholder messages into polished spoken text.

complexity_flag: computed by clarif_eye.router.classify_complexity (issue
#6 / P1.3) from ocr_output and scene_context. This module keeps ownership
of returning the key on every branch (including the degraded ones); it
just no longer computes the rule itself.
"""

from clarif_eye.client import LadderExhaustedError, OpenRouterClient, OpenRouterError
from clarif_eye.router import classify_complexity
from clarif_eye.speech import strip_code_fence

# Sentinel tokens (P1.8 / issue #29): a photographed document's own text can
# legitimately contain a line starting "OCR_TEXT:" or "SCENE:" (a
# screenplay, a shooting schedule, a meeting agenda), which collides with
# the legacy line-anchored markers below and forces a degrade even though
# the reply was genuinely readable. This exact token cannot plausibly occur
# in photographed text, so it is asked for and parsed first; the legacy
# markers remain a fallback for when a model drifts back to the familiar
# format (see _parse_legacy_reply and the recorded fixture).
OCR_SENTINEL = "<<<CLARIF_OCR>>>"
SCENE_SENTINEL = "<<<CLARIF_SCENE>>>"

# Legacy markers (P1.2 / issue #5). Chosen because they're trivial to
# generate reliably in a prompt and trivial to parse with a plain string
# search - no JSON-in-prose or code-fence fragility, and a model that
# garbles everything else usually still emits recognizable line-start
# markers. Kept as a fallback parse path - see _parse_legacy_reply.
OCR_MARKER = "OCR_TEXT:"
SCENE_MARKER = "SCENE:"

VISION_PROMPT = (
    "You are the vision stage of an assistant that describes photos aloud "
    "for a visually impaired user. Look at the attached image and reply "
    "with EXACTLY two sections, in this exact format and nothing else:\n\n"
    f"{OCR_SENTINEL}\n"
    "<any text visible in the image, or \"none\" if there is no text>\n"
    f"{SCENE_SENTINEL}\n"
    "<a concise description of the scene and layout>\n\n"
    "Do not add any other text before, between, or after these two lines. "
    f"{OCR_SENTINEL} and {SCENE_SENTINEL} must each appear alone on their "
    "own line, exactly once."
)

# The client only ever receives base64 image bytes (state["image_data"]),
# not the original file's mime type. JPEG is the deployed client's photo
# format; revisit this if a different capture format is ever supported.
_IMAGE_MIME_TYPE = "image/jpeg"


def _default_client():
    """Factory for the real client. Called lazily (never at import time)."""
    return OpenRouterClient()


def _build_messages(image_data):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{_IMAGE_MIME_TYPE};base64,{image_data}"},
                },
            ],
        }
    ]


def _parse_sentinel_reply(reply):
    """Parse a sentinel-delimited reply into (ocr_output, scene_context), or None.

    A line counts as a sentinel ONLY if it is EXACTLY the sentinel token
    (after stripping whitespace) - the sentinel cannot plausibly occur
    inside photographed text, so unlike the legacy markers there is no
    "mid-line occurrence" case to fold into body text. Content between/after
    the two sentinel lines is taken verbatim, including lines that happen to
    start with the legacy OCR_TEXT:/SCENE: markers - those are just body
    text here, not section headers. If either sentinel is missing or
    repeated, the reply is ambiguous and this returns None so the caller can
    fall back to the legacy parser rather than guessing.
    """
    lines = reply.split("\n")

    ocr_starts = [i for i, line in enumerate(lines) if line.strip() == OCR_SENTINEL]
    scene_starts = [i for i, line in enumerate(lines) if line.strip() == SCENE_SENTINEL]

    if len(ocr_starts) != 1 or len(scene_starts) != 1:
        return None

    ocr_index = ocr_starts[0]
    scene_index = scene_starts[0]

    def section_text(start_index, end_index):
        body_lines = lines[start_index + 1 : end_index]
        return strip_code_fence("\n".join(body_lines).strip())

    if ocr_index < scene_index:
        ocr_text = section_text(ocr_index, scene_index)
        scene_text = section_text(scene_index, len(lines))
    else:
        scene_text = section_text(scene_index, ocr_index)
        ocr_text = section_text(ocr_index, len(lines))

    if not scene_text:
        return None

    return ocr_text, scene_text


def _parse_legacy_reply(reply):
    """Parse the OLD OCR_TEXT:/SCENE: reply format into (ocr, scene), or None.

    Fallback path (P1.8 / issue #29): tried only when sentinel parsing
    fails, since a model that drifts back to this familiar format (or a
    recorded fixture predating the sentinel prompt) should still parse.

    None means the reply didn't follow the requested format and must be
    treated as a degraded response, not crash the node. An OCR section that
    parses to "" is valid (no visible text); the scene section is the only
    one required to be non-blank, since a legitimate reply always describes
    something.

    A line counts as a section header ONLY if it BEGINS (after stripping
    leading whitespace) with the marker - a marker string occurring mid-line
    (e.g. photographed text that happens to read "...SCENE: 4...") is just
    body text, not a new section, so it stays folded into whichever section
    it physically falls in. The requested reply format has exactly one
    header line per marker; if either marker's header line appears zero or
    more than once, the reply is ambiguous (a real header could be hiding
    among noise, or noise could look like a header) and the whole reply is
    treated as unparseable rather than guessing which occurrence is real.
    """
    lines = reply.split("\n")

    ocr_starts = []
    scene_starts = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(OCR_MARKER):
            ocr_starts.append(i)
        elif stripped.startswith(SCENE_MARKER):
            scene_starts.append(i)

    if len(ocr_starts) != 1 or len(scene_starts) != 1:
        return None

    ocr_index = ocr_starts[0]
    scene_index = scene_starts[0]

    def section_text(start_index, marker, end_index):
        first_line = lines[start_index].strip()[len(marker) :]
        body_lines = [first_line] + lines[start_index + 1 : end_index]
        return strip_code_fence("\n".join(body_lines).strip())

    if ocr_index < scene_index:
        ocr_text = section_text(ocr_index, OCR_MARKER, scene_index)
        scene_text = section_text(scene_index, SCENE_MARKER, len(lines))
    else:
        scene_text = section_text(scene_index, SCENE_MARKER, ocr_index)
        ocr_text = section_text(ocr_index, OCR_MARKER, len(lines))

    if not scene_text:
        return None

    return ocr_text, scene_text


def _parse_reply(reply):
    """Parse the model's reply into (ocr_output, scene_context), or None.

    Order (P1.8 / issue #29): sentinels first (collision-proof against a
    photographed document's own text), then the legacy OCR_TEXT:/SCENE:
    markers as a fallback, then degrade. None means neither format matched
    unambiguously and the caller must treat this as a degraded response,
    not crash the node.
    """
    parsed = _parse_sentinel_reply(reply)
    if parsed is not None:
        return parsed
    return _parse_legacy_reply(reply)


# This node's own degradation messages, as named constants rather than
# inline literals. Downstream consumers (synth.py) must be able to detect
# "vision failed and scene_context is actually an error message, not a
# scene description" WITHOUT matching against the English wording, so that
# rewording one of these messages for accessibility (issues #15/#18) can't
# silently break that detection - see is_degraded_scene below.
DEGRADED_CONFIG_ERROR = (
    "Vision could not run because of a configuration problem with the "
    "service. Please tell whoever set this up."
)
DEGRADED_LADDER_EXHAUSTED = (
    "Vision could not run right now: every available model was busy or "
    "unavailable. Please try again in a moment."
)
DEGRADED_UNEXPECTED_ERROR = (
    "Vision could not run because of an unexpected internal error. Please "
    "try again, and tell whoever set this up if it keeps happening."
)
DEGRADED_EMPTY_REPLY = "The vision model returned an empty response."
DEGRADED_UNPARSEABLE_REPLY = "The vision model's response could not be understood."


def is_degraded_scene(text):
    """True if `text` IS (exactly, modulo surrounding whitespace) one of
    this module's own degradation messages.

    Structural, not textual: this checks identity against the named
    constants above, not a hunt for prefixes/keywords in English prose. A
    caller (synth.py) uses this to tell "vision produced an error message"
    apart from "vision produced a real scene description" without having
    to know or guess at the wording.
    """
    stripped = (text or "").strip()
    return stripped in (
        DEGRADED_CONFIG_ERROR,
        DEGRADED_LADDER_EXHAUSTED,
        DEGRADED_UNEXPECTED_ERROR,
        DEGRADED_EMPTY_REPLY,
        DEGRADED_UNPARSEABLE_REPLY,
    )


def _degraded(message):
    return {
        "ocr_output": "",
        "scene_context": message,
        # A failed vision pass is never "complex" enough to justify the
        # slower research path - the fast path gets a spoken message to the
        # user sooner. This is a degradation-path decision, not part of the
        # complexity heuristic itself.
        "complexity_flag": False,
    }


def run_vision(image_data, client=None):
    """Call the eyes ladder for `image_data` and return a vision_node state update.

    `client` is injectable (tests pass a fake); when omitted, a real
    OpenRouterClient is constructed lazily, inside the same try/except that
    handles every other client failure, so a missing API key degrades the
    same way a ladder exhaustion would instead of raising. A client built
    here (not injected) is closed in a `finally` before returning, so its
    httpx connection pool doesn't leak; an injected client is owned by the
    caller and is never closed here.
    """
    owns_client = client is None
    if owns_client:
        try:
            client = _default_client()
        except OpenRouterError:
            return _degraded(DEGRADED_CONFIG_ERROR)
    try:
        try:
            result = client.complete("eyes", _build_messages(image_data))
        except LadderExhaustedError:
            return _degraded(DEGRADED_LADDER_EXHAUSTED)
        except OpenRouterError:
            return _degraded(DEGRADED_CONFIG_ERROR)
        except Exception:
            # Contract (module docstring): no raw exception may escape into
            # the graph. This catches everything else the injected client
            # could raise (ValueError, TimeoutError, KeyError, RuntimeError,
            # ...) without swallowing KeyboardInterrupt/SystemExit, which
            # derive from BaseException, not Exception.
            return _degraded(DEGRADED_UNEXPECTED_ERROR)
    finally:
        if owns_client:
            client.close()

    reply = result.content
    if not isinstance(reply, str) or not reply.strip():
        return _degraded(DEGRADED_EMPTY_REPLY)

    parsed = _parse_reply(reply)
    if parsed is None:
        return _degraded(DEGRADED_UNPARSEABLE_REPLY)

    ocr_output, scene_context = parsed
    return {
        "ocr_output": ocr_output,
        "scene_context": scene_context,
        "complexity_flag": classify_complexity(ocr_output, scene_context).complexity_flag,
    }
