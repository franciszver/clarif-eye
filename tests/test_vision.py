"""Tests for the vision node (issue #5 / P1.2): OCR + scene description.

No network calls: every test substitutes a fake client object for
OpenRouterClient (not httpx.MockTransport - the client itself is the seam).
Covers the multimodal request shape, reply parsing, and every documented
degradation path (LadderExhaustedError, terminal OpenRouterError, malformed
reply, empty reply) - none of these may raise into the graph.
"""

import pytest

from clarif_eye.client import CompletionResult, LadderExhaustedError, OpenRouterError
from clarif_eye.graph import build_graph, vision_node
from clarif_eye.state import make_initial_state
from clarif_eye import vision
from clarif_eye.vision import _parse_reply, run_vision

LONG_TEXT = "x" * 250  # deliberately over the placeholder complexity threshold


def _assert_reasonable_message(message, *, mentions):
    """Shared shape check for degraded messages (FIX 5): not just non-blank.

    `mentions` is a lowercase substring the message must contain, tying the
    assertion to the actual condition being degraded so a reword to "." (or
    to an unrelated sentence) fails, while an honest rewording of the same
    condition still passes.
    """
    assert isinstance(message, str)
    words = message.split()
    assert len(words) >= 5, f"degraded message too short to be meaningful: {message!r}"
    assert mentions in message.lower(), f"expected {mentions!r} in message: {message!r}"


class FakeVisionClient:
    """Minimal stand-in for OpenRouterClient.complete - records calls, no network."""

    def __init__(self, content=None, exc=None, model="fake-eyes-model:free"):
        self.content = content
        self.exc = exc
        self.model = model
        self.calls = []
        self.closed = False

    def complete(self, role, messages, **params):
        self.calls.append({"role": role, "messages": messages, "params": params})
        if self.exc is not None:
            raise self.exc
        return CompletionResult(content=self.content, model=self.model)

    def close(self):
        self.closed = True


def well_formed_reply(ocr="a coffee cup label", scene="a kitchen counter"):
    return f"OCR_TEXT: {ocr}\nSCENE: {scene}"


# --- Request shape: multimodal, data: URI, targets the eyes role ----------


def test_request_targets_eyes_role_and_contains_data_uri_image():
    client = FakeVisionClient(content=well_formed_reply())

    run_vision("aGVsbG8=", client)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["role"] == "eyes"

    messages = call["messages"]
    assert isinstance(messages, list) and len(messages) == 1
    content = messages[0]["content"]
    assert isinstance(content, list)

    text_parts = [p for p in content if p.get("type") == "text"]
    image_parts = [p for p in content if p.get("type") == "image_url"]
    assert len(text_parts) == 1
    assert len(image_parts) == 1
    url = image_parts[0]["image_url"]["url"]
    assert url.startswith("data:image/")
    assert "aGVsbG8=" in url


# --- Happy path -------------------------------------------------------------


def test_happy_path_parses_ocr_and_scene():
    client = FakeVisionClient(content=well_formed_reply("Open 9-5", "a shop front"))

    result = run_vision("base64data", client)

    assert result["ocr_output"] == "Open 9-5"
    assert result["scene_context"] == "a shop front"
    assert isinstance(result["complexity_flag"], bool)


def test_happy_path_no_visible_text_is_not_treated_as_malformed():
    client = FakeVisionClient(content=well_formed_reply("none", "an empty hallway"))

    result = run_vision("base64data", client)

    assert result["scene_context"] == "an empty hallway"
    assert isinstance(result["complexity_flag"], bool)


# --- complexity_flag is always set, in every branch ------------------------


def test_complexity_flag_always_present_and_bool_on_happy_path():
    client = FakeVisionClient(content=well_formed_reply())
    result = run_vision("base64data", client)
    assert "complexity_flag" in result
    assert isinstance(result["complexity_flag"], bool)


def test_complexity_flag_always_present_and_bool_on_malformed_reply():
    client = FakeVisionClient(content="I cannot see the image clearly.")
    result = run_vision("base64data", client)
    assert "complexity_flag" in result
    assert isinstance(result["complexity_flag"], bool)


def test_complexity_flag_always_present_and_bool_on_empty_reply():
    client = FakeVisionClient(content="   ")
    result = run_vision("base64data", client)
    assert "complexity_flag" in result
    assert isinstance(result["complexity_flag"], bool)


def test_complexity_flag_always_present_and_bool_on_ladder_exhausted():
    client = FakeVisionClient(exc=LadderExhaustedError("eyes", ()))
    result = run_vision("base64data", client)
    assert "complexity_flag" in result
    assert isinstance(result["complexity_flag"], bool)


def test_complexity_flag_always_present_and_bool_on_openrouter_error():
    client = FakeVisionClient(exc=OpenRouterError("out of credit"))
    result = run_vision("base64data", client)
    assert "complexity_flag" in result
    assert isinstance(result["complexity_flag"], bool)


# --- Degradation: malformed reply -------------------------------------------


def test_malformed_reply_degrades_without_raising():
    client = FakeVisionClient(content="not the format I asked for at all")

    result = run_vision("base64data", client)

    assert result["ocr_output"] == ""
    _assert_reasonable_message(result["scene_context"], mentions="understood")


# --- Degradation: empty reply ------------------------------------------------


def test_empty_reply_degrades_without_raising():
    client = FakeVisionClient(content="   ")

    result = run_vision("base64data", client)

    assert result["ocr_output"] == ""
    _assert_reasonable_message(result["scene_context"], mentions="empty")


# --- Degradation: LadderExhaustedError --------------------------------------


def test_ladder_exhausted_degrades_without_raising():
    client = FakeVisionClient(exc=LadderExhaustedError("eyes", ()))

    result = run_vision("base64data", client)

    assert result["ocr_output"] == ""
    _assert_reasonable_message(result["scene_context"], mentions="busy")


# --- Degradation: terminal OpenRouterError ----------------------------------


def test_openrouter_error_degrades_without_raising():
    client = FakeVisionClient(exc=OpenRouterError("authentication failed"))

    result = run_vision("base64data", client)

    assert result["ocr_output"] == ""
    _assert_reasonable_message(result["scene_context"], mentions="configuration")


# --- Degradation: unexpected exception types (FIX 1) -------------------------


@pytest.mark.parametrize("exc", [ValueError("bad"), TimeoutError("timed out"), RuntimeError("oops")])
def test_unexpected_exception_types_degrade_without_raising(exc):
    client = FakeVisionClient(exc=exc)

    result = run_vision("base64data", client)

    assert result["ocr_output"] == ""
    _assert_reasonable_message(result["scene_context"], mentions="unexpected")
    assert isinstance(result["complexity_flag"], bool)


def test_key_error_degrades_without_raising():
    client = FakeVisionClient(exc=KeyError("missing"))

    result = run_vision("base64data", client)

    assert result["ocr_output"] == ""
    _assert_reasonable_message(result["scene_context"], mentions="unexpected")


def test_keyboard_interrupt_is_not_swallowed():
    client = FakeVisionClient(exc=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        run_vision("base64data", client)


# --- Degradation: non-string reply (FIX 2) ------------------------------------


@pytest.mark.parametrize("bad_reply", [{}, 5, None])
def test_non_string_reply_degrades_without_raising(bad_reply):
    client = FakeVisionClient(content=bad_reply)

    result = run_vision("base64data", client)

    assert result["ocr_output"] == ""
    _assert_reasonable_message(result["scene_context"], mentions="empty")
    assert isinstance(result["complexity_flag"], bool)


# --- Parser robustness (FIX 3) ------------------------------------------------


def test_parser_does_not_misfile_ocr_text_containing_scene_marker_lines():
    """The exact realistic agenda-photo reply from the review finding.

    Multiple lines start with SCENE: inside what is genuinely photographed
    text. The parser must not silently truncate the OCR text to one word
    and read the rest aloud as a scene description - it must either keep
    the OCR text whole or degrade the whole reply, never the old
    first-occurrence behaviour.
    """
    reply = (
        "OCR_TEXT: AGENDA\n"
        "SCENE: intro (5 min)\n"
        "SCENE: demo (10 min)\n"
        "SCENE: overall a whiteboard photo"
    )

    parsed = _parse_reply(reply)

    # Either the OCR text is kept whole (not truncated to "AGENDA"), or the
    # whole reply is treated as unparseable and degrades - never the old
    # buggy behaviour of a one-word OCR result plus the agenda items read
    # aloud as if they described the room.
    if parsed is not None:
        ocr_output, _scene_context = parsed
        assert ocr_output != "AGENDA"

    client = FakeVisionClient(content=reply)
    result = run_vision("base64data", client)
    assert result["ocr_output"] != "AGENDA"


def test_parser_treats_marker_mid_line_as_body_text_not_a_new_section():
    reply = "OCR_TEXT: the sign reads SCENE: closed\nSCENE: a shop front at night"

    parsed = _parse_reply(reply)

    assert parsed is not None
    ocr_output, scene_context = parsed
    assert ocr_output == "the sign reads SCENE: closed"
    assert scene_context == "a shop front at night"


def test_parser_degrades_when_a_marker_line_repeats_many_times():
    reply = "OCR_TEXT: menu\n" + "\n".join(f"SCENE: item {i}" for i in range(10))

    parsed = _parse_reply(reply)

    assert parsed is None


def test_parser_still_handles_preamble_before_markers():
    reply = "Sure, here is my analysis:\nOCR_TEXT: hello\nSCENE: a desk"

    parsed = _parse_reply(reply)

    assert parsed == ("hello", "a desk")


def test_parser_still_handles_reversed_marker_order():
    reply = "SCENE: a desk\nOCR_TEXT: hello"

    parsed = _parse_reply(reply)

    assert parsed == ("hello", "a desk")


def test_parser_still_returns_none_for_missing_marker():
    reply = "OCR_TEXT: hello, no scene marker here"

    parsed = _parse_reply(reply)

    assert parsed is None


def test_parser_strips_code_fence_artifacts():
    reply = "```\nOCR_TEXT: hello\nSCENE: a room\n```"

    parsed = _parse_reply(reply)

    assert parsed == ("hello", "a room")


# --- Client lifecycle (FIX 4) --------------------------------------------------


def test_self_constructed_client_is_closed(monkeypatch):
    fake = FakeVisionClient(content=well_formed_reply())
    monkeypatch.setattr(vision, "_default_client", lambda: fake)

    run_vision("base64data")

    assert fake.closed is True


def test_injected_client_is_not_closed():
    client = FakeVisionClient(content=well_formed_reply())

    run_vision("base64data", client)

    assert client.closed is False


def test_self_constructed_client_is_closed_even_on_degraded_path(monkeypatch):
    fake = FakeVisionClient(exc=LadderExhaustedError("eyes", ()))
    monkeypatch.setattr(vision, "_default_client", lambda: fake)

    run_vision("base64data")

    assert fake.closed is True


# --- vision_node: client injection, graph-facing wrapper --------------------


def test_vision_node_accepts_an_explicit_injected_client():
    client = FakeVisionClient(content=well_formed_reply())

    result = vision_node({"image_data": "base64data"}, client=client)

    assert result["ocr_output"] == "a coffee cup label"
    assert result["scene_context"] == "a kitchen counter"
    assert "complexity_flag" in result
    assert len(client.calls) == 1


def test_vision_node_accepts_client_injected_via_config_configurable():
    client = FakeVisionClient(content=well_formed_reply())

    result = vision_node(
        {"image_data": "base64data"}, config={"configurable": {"client": client}}
    )

    assert result["ocr_output"] == "a coffee cup label"
    assert len(client.calls) == 1


# --- Full compiled graph, real node, fake client -----------------------------


def test_full_compiled_graph_runs_end_to_end_with_fake_client_fast_path():
    client = FakeVisionClient(content=well_formed_reply("short text", "a room"))
    graph = build_graph()
    state = make_initial_state("base64data")

    result = graph.invoke(
        state, config={"configurable": {"trace": [], "client": client}}
    )

    assert result["ocr_output"] == "short text"
    assert result["scene_context"] == "a room"
    assert result["complexity_flag"] is False
    assert result["final_output"] != ""
    assert result["audio_file_path"] != ""


def test_full_compiled_graph_runs_end_to_end_with_fake_client_research_path():
    client = FakeVisionClient(content=well_formed_reply(LONG_TEXT, "a busy street"))
    graph = build_graph()
    state = make_initial_state("base64data")
    trace = []

    result = graph.invoke(state, config={"configurable": {"trace": trace, "client": client}})

    assert result["complexity_flag"] is True
    assert trace == ["vision", "research", "analysis", "tts"]
    assert result["final_output"] != ""
    assert result["audio_file_path"] != ""


def test_full_compiled_graph_degrades_gracefully_and_still_reaches_tts():
    client = FakeVisionClient(exc=LadderExhaustedError("eyes", ()))
    graph = build_graph()
    state = make_initial_state("base64data")
    trace = []

    result = graph.invoke(state, config={"configurable": {"trace": trace, "client": client}})

    assert trace[-1] == "tts"
    assert result["audio_file_path"] != ""
    assert isinstance(result["complexity_flag"], bool)
