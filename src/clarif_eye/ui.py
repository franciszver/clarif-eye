"""Gradio UI logic for Clarif-Eye (issue #13 / P4.1): wires the graph to a
human, one photo at a time.

Kept separate from app.py (the thin Spaces launcher) so this module is
TESTABLE without launching a server: tests/test_ui.py calls handle_submit
directly with fakes and never starts Gradio or touches the network.

SHARED CLIENT / PROVIDER CHAIN
-------------------------------
Nodes construct their own client when none is injected (see graph.py's
module docstrings), which is fine for tests but wasteful for a live app: a
fast-path request would otherwise open its own httpx connection pool per
node. build_resources() constructs ONE OpenRouterClient, ONE TTS provider
chain, and (best-effort) ONE research searcher/client at process startup;
handle_submit injects all of them via config["configurable"] on every
request, the same seam graph.py already documents for tests.

FAILURE BEHAVIOR - THIS MODULE MUST NEVER RAISE INTO GRADIO
--------------------------------------------------------------
A traceback in the UI is useless to a blind user. Every failure mode (no
image, an unreadable/corrupt image, a missing API key at startup, an
unexpected exception anywhere in the graph) degrades to a spoken-ready
message in the returned text, never an exception - same discipline every
node in this pipeline already follows.

THE THREE OUTCOMES, TOLD APART STRUCTURALLY
----------------------------------------------
  a. audio_file_path is truthy -> play it, show final_output as-is.
  b. audio_file_path == "" AND tts.is_chain_exhausted() is True -> no
     audio was produced despite a real attempt; announce that plainly and
     show final_output as text.
  c. The pipeline degraded upstream (vision/synth/analysis already wrote a
     human-readable message into final_output) -> that message IS the
     script. It needs no special-casing here: it flows through outcome (a)
     or (b) exactly like a non-degraded script would, so no second layer
     of "something went wrong" wrapping is ever added on top of it.
Never string-matched - (a)/(b) are told apart via audio_file_path
truthiness and the structural tts.is_chain_exhausted() predicate, the same
discipline vision.is_degraded_scene already established.
"""

import base64
import io
from dataclasses import dataclass

from clarif_eye.client import OpenRouterClient, OpenRouterError
from clarif_eye.graph import build_graph
from clarif_eye.state import make_initial_state
from clarif_eye.tts import DEFAULT_PROVIDER_CHAIN, is_chain_exhausted

# Spoken-ready messages, as named constants rather than inline literals -
# same reasoning as vision.py's DEGRADED_* constants: a caller (and tests)
# can rely on these without guessing at wording, and rewording one later
# can't silently break a test that was substring-matching prose instead.
NO_IMAGE_MESSAGE = (
    "No photo was provided. Please take or upload a photo to continue."
)
CONFIG_ERROR_MESSAGE = (
    "Clarif-Eye is not fully configured yet: the service is missing an "
    "API key. Please tell whoever set this up."
)
UNREADABLE_IMAGE_MESSAGE = (
    "The photo could not be read. Please try again with a different photo."
)
UNEXPECTED_ERROR_MESSAGE = (
    "Something went wrong while processing your photo. Please try again."
)
AUDIO_UNAVAILABLE_NOTE = (
    "Audio isn't available right now, so here is the description as text."
)


@dataclass
class AppResources:
    """Everything build_resources() constructs once at startup and
    handle_submit injects on every request. Plain dataclass, not a
    framework - one instance per process (CLAUDE.md Simplicity First)."""

    graph: object
    client: object
    client_error: str | None
    tts_providers: list
    searcher: object
    research_client: object


def build_resources():
    """Construct every injectable ONCE for the life of the process.

    Never raises: a missing OPENROUTER_API_KEY (the likely state of a
    fresh Hugging Face Space with no secret set yet) must not crash the
    app at import/startup time - it degrades to client=None plus a spoken
    message, checked by handle_submit before the graph is ever invoked.
    The research searcher/client are best-effort shared instances too (see
    module docstring); if either fails to construct, they're left None and
    research_node falls back to its own lazy per-call defaults.
    """
    try:
        client = OpenRouterClient()
        client_error = None
    except OpenRouterError:
        client = None
        client_error = CONFIG_ERROR_MESSAGE

    tts_providers = [factory() for factory in DEFAULT_PROVIDER_CHAIN]

    try:
        from ddgs import DDGS

        searcher = DDGS()
    except Exception:
        searcher = None

    try:
        import httpx

        research_client = httpx.Client()
    except Exception:
        research_client = None

    return AppResources(
        graph=build_graph(),
        client=client,
        client_error=client_error,
        tts_providers=tts_providers,
        searcher=searcher,
        research_client=research_client,
    )


def _encode_image(image):
    """Encode a PIL Image to base64 JPEG bytes for make_initial_state.

    Raises on anything unreadable (wrong object shape, a corrupt image
    PIL can't re-encode, ...) - handle_submit turns that into
    UNREADABLE_IMAGE_MESSAGE rather than letting it propagate.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def handle_submit(image, resources):
    """Run one photo through the graph; return (audio_path_or_None, text).

    NEVER raises (except KeyboardInterrupt/SystemExit) - every failure
    mode returns a spoken-ready message instead, per the module docstring.
    `resources` is an AppResources built once by build_resources() and
    passed through unchanged on every call, so the shared client/provider
    chain/searcher are injected identically on every request.
    """
    if image is None:
        return None, NO_IMAGE_MESSAGE

    if resources.client is None:
        return None, resources.client_error or CONFIG_ERROR_MESSAGE

    try:
        image_data = _encode_image(image)
    except Exception:
        return None, UNREADABLE_IMAGE_MESSAGE

    try:
        state = make_initial_state(image_data)
        config = {
            "configurable": {
                "client": resources.client,
                "tts_providers": resources.tts_providers,
                "searcher": resources.searcher,
                "research_client": resources.research_client,
            }
        }
        result = resources.graph.invoke(state, config=config)
    except Exception:
        return None, UNEXPECTED_ERROR_MESSAGE

    final_output = (result.get("final_output") or "").strip()
    audio_path = result.get("audio_file_path") or ""

    if audio_path:
        return audio_path, final_output

    if is_chain_exhausted():
        if final_output:
            return None, f"{final_output} {AUDIO_UNAVAILABLE_NOTE}"
        return None, AUDIO_UNAVAILABLE_NOTE

    return None, final_output or UNEXPECTED_ERROR_MESSAGE
