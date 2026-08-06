"""Vision node logic (issue #5 / P1.2): OCR + scene description.

Builds an OpenAI-compatible multimodal request (image sent as a `data:`
URI) and calls OpenRouterClient.complete("eyes", messages). This module
must NEVER let a raw exception escape into the graph - the person on the
other end is relying on spoken feedback, not a stack trace. Every failure
mode (LadderExhaustedError, a terminal OpenRouterError, a malformed reply,
an empty reply) degrades into a plain-English placeholder message plus a
complexity_flag, so the graph can still route and reach tts/END. Issue #18
owns turning these placeholder messages into polished spoken text.

complexity_flag placeholder: issue #6 (P1.3) owns the real complexity
heuristic and will replace the RULE below, not the node's ownership of the
key - vision_node must keep returning complexity_flag regardless of who
computes it.
"""

from clarif_eye.client import LadderExhaustedError, OpenRouterClient, OpenRouterError

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


def _parse_reply(reply):
    """Parse the model's reply into (ocr_output, scene_context), or None.

    None means the reply didn't follow the requested format and must be
    treated as a degraded response, not crash the node. An OCR section that
    parses to "" is valid (no visible text); the scene section is the only
    one required to be non-blank, since a legitimate reply always describes
    something.
    """
    if OCR_MARKER not in reply or SCENE_MARKER not in reply:
        return None

    ocr_index = reply.index(OCR_MARKER)
    scene_index = reply.index(SCENE_MARKER)
    if ocr_index == scene_index:
        return None

    if ocr_index < scene_index:
        ocr_text = reply[ocr_index + len(OCR_MARKER) : scene_index].strip()
        scene_text = reply[scene_index + len(SCENE_MARKER) :].strip()
    else:
        scene_text = reply[scene_index + len(SCENE_MARKER) : ocr_index].strip()
        ocr_text = reply[ocr_index + len(OCR_MARKER) :].strip()

    if not scene_text:
        return None

    return ocr_text, scene_text


def _degraded(message):
    return {
        "ocr_output": "",
        "scene_context": message,
        # Placeholder rule (issue #6 / P1.3 owns the real heuristic): a
        # failed vision pass is never "complex" enough to justify the
        # slower research path - the fast path gets a spoken message to the
        # user sooner.
        "complexity_flag": False,
    }


def run_vision(image_data, client=None):
    """Call the eyes ladder for `image_data` and return a vision_node state update.

    `client` is injectable (tests pass a fake); when omitted, a real
    OpenRouterClient is constructed lazily, inside the same try/except that
    handles every other client failure, so a missing API key degrades the
    same way a ladder exhaustion would instead of raising.
    """
    try:
        if client is None:
            client = _default_client()
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

    reply = result.content
    if not reply or not reply.strip():
        return _degraded("The vision model returned an empty response.")

    parsed = _parse_reply(reply)
    if parsed is None:
        return _degraded("The vision model's response could not be understood.")

    ocr_output, scene_context = parsed
    # Placeholder rule: issue #6 (P1.3) owns the real complexity heuristic
    # and will replace this rule, not the ownership of the key.
    complexity_flag = len(ocr_output) > 200
    return {
        "ocr_output": ocr_output,
        "scene_context": scene_context,
        "complexity_flag": complexity_flag,
    }
