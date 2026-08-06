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
from clarif_eye.vision import run_vision

LONG_TEXT = "x" * 250  # deliberately over the placeholder complexity threshold


class FakeVisionClient:
    """Minimal stand-in for OpenRouterClient.complete - records calls, no network."""

    def __init__(self, content=None, exc=None, model="fake-eyes-model:free"):
        self.content = content
        self.exc = exc
        self.model = model
        self.calls = []

    def complete(self, role, messages, **params):
        self.calls.append({"role": role, "messages": messages, "params": params})
        if self.exc is not None:
            raise self.exc
        return CompletionResult(content=self.content, model=self.model)


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
    assert isinstance(result["scene_context"], str)
    assert result["scene_context"].strip() != ""


# --- Degradation: empty reply ------------------------------------------------


def test_empty_reply_degrades_without_raising():
    client = FakeVisionClient(content="   ")

    result = run_vision("base64data", client)

    assert result["ocr_output"] == ""
    assert isinstance(result["scene_context"], str)
    assert result["scene_context"].strip() != ""


# --- Degradation: LadderExhaustedError --------------------------------------


def test_ladder_exhausted_degrades_without_raising():
    client = FakeVisionClient(exc=LadderExhaustedError("eyes", ()))

    result = run_vision("base64data", client)

    assert result["ocr_output"] == ""
    assert isinstance(result["scene_context"], str)
    assert result["scene_context"].strip() != ""


# --- Degradation: terminal OpenRouterError ----------------------------------


def test_openrouter_error_degrades_without_raising():
    client = FakeVisionClient(exc=OpenRouterError("authentication failed"))

    result = run_vision("base64data", client)

    assert result["ocr_output"] == ""
    assert isinstance(result["scene_context"], str)
    assert result["scene_context"].strip() != ""


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
