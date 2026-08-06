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

# Chosen because it's trivial to generate reliably in a prompt and trivial
# to parse with a plain string search - no JSON-in-prose or code-fence
# fragility, and a model that garbles everything else usually still emits
# recognizable line-start markers.
OCR_MARKER = "OCR_TEXT:"
SCENE_MARKER = "SCENE:"

VISION_PROMPT = (
    "You are the vision stage of an assistant that describes photos aloud "
    "for a visually impaired user. Look at the attached image and reply "
    "with EXACTLY two sections, in this exact format and nothing else:\n\n"
    f"{OCR_MARKER} <any text visible in the image, or \"none\" if there is no text>\n"
    f"{SCENE_MARKER} <a concise description of the scene and layout>\n\n"
    "Do not add any other text before, between, or after these two lines."
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


def _strip_code_fence(text):
    """Strip a leading/trailing ``` fence line (and language tag) if present."""
    lines = text.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_reply(reply):
    """Parse the model's reply into (ocr_output, scene_context), or None.

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
        return _strip_code_fence("\n".join(body_lines).strip())

    if ocr_index < scene_index:
        ocr_text = section_text(ocr_index, OCR_MARKER, scene_index)
        scene_text = section_text(scene_index, SCENE_MARKER, len(lines))
    else:
        scene_text = section_text(scene_index, SCENE_MARKER, ocr_index)
        ocr_text = section_text(ocr_index, OCR_MARKER, len(lines))

    if not scene_text:
        return None

    return ocr_text, scene_text


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
            return _degraded(
                "Vision could not run because of a configuration problem with "
                "the service. Please tell whoever set this up."
            )
    try:
        try:
            result = client.complete("eyes", _build_messages(image_data))
        except LadderExhaustedError:
            return _degraded(
                "Vision could not run right now: every available model was busy "
                "or unavailable. Please try again in a moment."
            )
        except OpenRouterError:
            return _degraded(
                "Vision could not run because of a configuration problem with "
                "the service. Please tell whoever set this up."
            )
        except Exception:
            # Contract (module docstring): no raw exception may escape into
            # the graph. This catches everything else the injected client
            # could raise (ValueError, TimeoutError, KeyError, RuntimeError,
            # ...) without swallowing KeyboardInterrupt/SystemExit, which
            # derive from BaseException, not Exception.
            return _degraded(
                "Vision could not run because of an unexpected internal "
                "error. Please try again, and tell whoever set this up if "
                "it keeps happening."
            )
    finally:
        if owns_client:
            client.close()

    reply = result.content
    if not isinstance(reply, str) or not reply.strip():
        return _degraded("The vision model returned an empty response.")

    parsed = _parse_reply(reply)
    if parsed is None:
        return _degraded("The vision model's response could not be understood.")

    ocr_output, scene_context = parsed
    return {
        "ocr_output": ocr_output,
        "scene_context": scene_context,
        "complexity_flag": classify_complexity(ocr_output, scene_context).complexity_flag,
    }
