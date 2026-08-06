"""Tests for the fast-synthesis node (issue #7 / P1.4): OCR + scene -> spoken script.

No network calls: every test substitutes a fake client object for
OpenRouterClient (the client itself is the seam, same pattern as
test_vision.py). Covers TTS-safety of every produced output (happy path,
markup-laden replies, and every degradation path), the empty-OCR input
edge cases, and client lifecycle.
"""

import re

import pytest

from clarif_eye.client import CompletionResult, LadderExhaustedError, OpenRouterError
from clarif_eye.graph import build_graph, fast_synth_node
from clarif_eye.state import make_initial_state
from clarif_eye import synth
from clarif_eye.synth import _to_spoken_text, run_fast_synth

# --- Shared TTS-safety assertion, applied to every produced output ---------

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "]"
)


def assert_tts_safe(text):
    """A script that will be read aloud must contain none of these."""
    assert isinstance(text, str)
    assert text.strip() != ""
    assert "**" not in text, f"markdown bold survived: {text!r}"
    assert "__" not in text, f"markdown bold/underline survived: {text!r}"
    assert "`" not in text, f"backtick/code fence survived: {text!r}"
    assert not re.search(r"(^|\n)\s*#+\s", text), f"markdown heading survived: {text!r}"
    assert not re.search(r"(^|\n)\s*[-*•]\s", text), f"bullet point survived: {text!r}"
    assert not re.search(r"(^|\n)\s*\d+[.)]\s", text), f"numbered list survived: {text!r}"
    assert "|" not in text, f"pipe character (table) survived: {text!r}"
    assert not _EMOJI_PATTERN.search(text), f"emoji survived: {text!r}"
    assert "http://" not in text and "https://" not in text and "www." not in text, (
        f"raw URL survived: {text!r}"
    )
    assert not re.search(r"([^\w\s])\1{2,}", text), f"bare punctuation run survived: {text!r}"


# --- Fake client, same shape as test_vision.py's ----------------------------


class FakeSynthClient:
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


def _assert_reasonable_message(message, *, mentions):
    assert isinstance(message, str)
    words = message.split()
    assert len(words) >= 5, f"message too short to be meaningful: {message!r}"
    assert mentions in message.lower(), f"expected {mentions!r} in message: {message!r}"


# --- Request shape: text-only, targets eyes role ----------------------------


def test_request_targets_eyes_role_and_is_text_only():
    client = FakeSynthClient(content="The image shows a coffee cup label on a kitchen counter.")

    run_fast_synth("Open 9-5", "a kitchen counter", client)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["role"] == "eyes"
    messages = call["messages"]
    assert isinstance(messages, list) and len(messages) == 1
    content = messages[0]["content"]
    # Text-only: no image_url parts (unlike vision.py's multimodal request).
    if isinstance(content, list):
        assert all(p.get("type") == "text" for p in content)
    else:
        assert isinstance(content, str)


# --- Happy path ---------------------------------------------------------------


def test_happy_path_returns_spoken_final_output():
    client = FakeSynthClient(
        content="The image shows a coffee cup label that reads Open 9 to 5, sitting on a kitchen counter."
    )

    result = run_fast_synth("Open 9-5", "a kitchen counter", client)

    assert_tts_safe(result["final_output"])
    assert "final_output" in result


# --- Sanitisation: markdown, bullets, tables, emoji must be stripped --------


def test_markup_heavy_reply_is_sanitised_into_clean_spoken_prose():
    dirty_reply = (
        "# Description\n\n"
        "The image shows **a receipt** from `Joe's Diner` 🍔:\n\n"
        "- Burger: $8.00\n"
        "- Fries: $3.00\n\n"
        "| Item | Price |\n"
        "|------|-------|\n"
        "| Burger | $8.00 |\n\n"
        "1. Total due: $11.00\n"
        "2. Visit https://joesdiner.example.com for more!!!\n"
    )
    client = FakeSynthClient(content=dirty_reply)

    result = run_fast_synth("Joe's Diner receipt $11.00", "a paper receipt on a table", client)

    final_output = result["final_output"]
    assert_tts_safe(final_output)
    # The underlying words should still be present - sanitising strips
    # markup, not content.
    assert "Burger" in final_output
    assert "Total due" in final_output or "11.00" in final_output


def test_to_spoken_text_strips_markdown_bullets_tables_and_emoji_directly():
    dirty = "**Bold** text with a bullet:\n- item one\n- item two\n| a | b |\n🎉 done ```code```"

    cleaned = _to_spoken_text(dirty)

    assert_tts_safe(cleaned)


# --- Input edge case: empty OCR with a real scene ---------------------------


def test_empty_ocr_with_real_scene_calls_model_and_does_not_ask_about_absent_text():
    client = FakeSynthClient(content="The image shows an empty hallway with white walls.")

    result = run_fast_synth("", "an empty hallway", client)

    assert len(client.calls) == 1
    assert_tts_safe(result["final_output"])


# --- Input edge case: empty OCR with a vision degradation message ----------


@pytest.mark.parametrize(
    "degradation_message",
    [
        "Vision could not run right now: every available model was busy or "
        "unavailable. Please try again in a moment.",
        "Vision could not run because of a configuration problem with the "
        "service. Please tell whoever set this up.",
        "The vision model returned an empty response.",
        "The vision model's response could not be understood.",
    ],
)
def test_empty_ocr_with_vision_degradation_message_is_not_echoed_as_a_description(
    degradation_message,
):
    client = FakeSynthClient(content="This should never be used.")

    result = run_fast_synth("", degradation_message, client)

    # The node must not ask the model to "describe" a failure message as if
    # it were a photo - the model must not even be called.
    assert len(client.calls) == 0
    assert_tts_safe(result["final_output"])
    # The degradation message itself is what gets spoken - not fabricated
    # description text.
    assert degradation_message.split()[0] in result["final_output"]


# --- Degradation: LadderExhaustedError --------------------------------------


def test_ladder_exhausted_degrades_without_raising():
    client = FakeSynthClient(exc=LadderExhaustedError("eyes", ()))

    result = run_fast_synth("some text", "a scene", client)

    _assert_reasonable_message(result["final_output"], mentions="busy")
    assert_tts_safe(result["final_output"])


# --- Degradation: terminal OpenRouterError ----------------------------------


def test_openrouter_error_degrades_without_raising():
    client = FakeSynthClient(exc=OpenRouterError("authentication failed"))

    result = run_fast_synth("some text", "a scene", client)

    _assert_reasonable_message(result["final_output"], mentions="configuration")
    assert_tts_safe(result["final_output"])


# --- Degradation: unexpected exception types --------------------------------


@pytest.mark.parametrize("exc", [ValueError("bad"), TimeoutError("timed out"), RuntimeError("oops")])
def test_unexpected_exception_types_degrade_without_raising(exc):
    client = FakeSynthClient(exc=exc)

    result = run_fast_synth("some text", "a scene", client)

    _assert_reasonable_message(result["final_output"], mentions="unexpected")
    assert_tts_safe(result["final_output"])


def test_key_error_degrades_without_raising():
    client = FakeSynthClient(exc=KeyError("missing"))

    result = run_fast_synth("some text", "a scene", client)

    _assert_reasonable_message(result["final_output"], mentions="unexpected")
    assert_tts_safe(result["final_output"])


def test_keyboard_interrupt_is_not_swallowed():
    client = FakeSynthClient(exc=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        run_fast_synth("some text", "a scene", client)


# --- Degradation: non-string reply ------------------------------------------


@pytest.mark.parametrize("bad_reply", [{}, 5, None])
def test_non_string_reply_degrades_without_raising(bad_reply):
    client = FakeSynthClient(content=bad_reply)

    result = run_fast_synth("some text", "a scene", client)

    _assert_reasonable_message(result["final_output"], mentions="empty")
    assert_tts_safe(result["final_output"])


# --- Degradation: empty/whitespace-only reply -------------------------------


def test_empty_reply_degrades_without_raising():
    client = FakeSynthClient(content="   ")

    result = run_fast_synth("some text", "a scene", client)

    _assert_reasonable_message(result["final_output"], mentions="empty")
    assert_tts_safe(result["final_output"])


def test_reply_that_sanitises_to_blank_degrades_without_raising():
    # A reply that is entirely markup/punctuation noise sanitises to "".
    client = FakeSynthClient(content="### | | | ***")

    result = run_fast_synth("some text", "a scene", client)

    _assert_reasonable_message(result["final_output"], mentions="empty")
    assert_tts_safe(result["final_output"])


# --- Client lifecycle --------------------------------------------------------


def test_self_constructed_client_is_closed(monkeypatch):
    fake = FakeSynthClient(content="The image shows a room.")
    monkeypatch.setattr(synth, "_default_client", lambda: fake)

    run_fast_synth("some text", "a scene")

    assert fake.closed is True


def test_injected_client_is_not_closed():
    client = FakeSynthClient(content="The image shows a room.")

    run_fast_synth("some text", "a scene", client)

    assert client.closed is False


def test_self_constructed_client_is_closed_even_on_degraded_path(monkeypatch):
    fake = FakeSynthClient(exc=LadderExhaustedError("eyes", ()))
    monkeypatch.setattr(synth, "_default_client", lambda: fake)

    run_fast_synth("some text", "a scene")

    assert fake.closed is True


def test_no_client_constructed_when_scene_context_is_a_degradation_message(monkeypatch):
    """The pass-through-degradation-message path never touches the client at all."""

    def _boom():
        raise AssertionError("client should not be constructed on this path")

    monkeypatch.setattr(synth, "_default_client", _boom)

    result = run_fast_synth("", "The vision model returned an empty response.")

    assert_tts_safe(result["final_output"])


# --- fast_synth_node: client injection, graph-facing wrapper ---------------


def test_fast_synth_node_accepts_an_explicit_injected_client():
    client = FakeSynthClient(content="The image shows a room.")
    state = {"ocr_output": "some text", "scene_context": "a scene"}

    result = fast_synth_node(state, client=client)

    assert_tts_safe(result["final_output"])
    assert len(client.calls) == 1


def test_fast_synth_node_accepts_client_injected_via_config_configurable():
    client = FakeSynthClient(content="The image shows a room.")
    state = {"ocr_output": "some text", "scene_context": "a scene"}

    result = fast_synth_node(state, config={"configurable": {"client": client}})

    assert_tts_safe(result["final_output"])
    assert len(client.calls) == 1


# --- Full compiled graph, real node, fake client ----------------------------


def test_full_compiled_graph_runs_end_to_end_on_fast_path_with_fake_client():
    vision_reply = "OCR_TEXT: Open 9-5\nSCENE: a shop front"

    class FakeVisionThenSynthClient:
        def __init__(self):
            self.calls = []
            self.closed = False

        def complete(self, role, messages, **params):
            self.calls.append(role)
            if len(self.calls) == 1:
                return CompletionResult(content=vision_reply, model="fake-eyes")
            return CompletionResult(
                content="The image shows a sign reading Open 9 to 5 on a shop front.",
                model="fake-eyes",
            )

        def close(self):
            self.closed = True

    client = FakeVisionThenSynthClient()
    graph = build_graph()
    state = make_initial_state("base64data")

    result = graph.invoke(state, config={"configurable": {"trace": [], "client": client}})

    assert result["complexity_flag"] is False
    assert_tts_safe(result["final_output"])
    assert result["audio_file_path"] != ""
