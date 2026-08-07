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
from clarif_eye import synth, vision
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
    # FIX 5: reject the markup shapes review found assert_tts_safe missing.
    # Ordinary prose keeps balanced brackets/parens (e.g. "(open 9-5)"), so
    # these check balance/adjacency, not "no brackets/parens at all".
    assert text.count("[") == text.count("]"), f"unbalanced brackets survived: {text!r}"
    assert text.count("(") == text.count(")"), f"unbalanced parentheses survived: {text!r}"
    assert "](" not in text, f"raw markdown link fragment survived: {text!r}"
    assert not re.search(r"</?[a-zA-Z][^<>]*>", text), f"HTML/angle-bracket tag survived: {text!r}"
    assert ", ," not in text, f"comma run survived: {text!r}"
    assert not text.startswith(","), f"leading comma survived: {text!r}"
    assert not text.endswith(","), f"trailing comma survived: {text!r}"


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
        # Read live off vision.py's own constants (issue #18 / P6.2 changed
        # their wording) rather than a pinned copy, matching the module's
        # own "structural, not textual" detection - see
        # test_rewording_a_vision_degradation_message_does_not_break_detection
        # below for the same reasoning applied explicitly.
        vision.DEGRADED_LADDER_EXHAUSTED,
        vision.DEGRADED_CONFIG_ERROR,
        vision.DEGRADED_BUSY,
        vision.DEGRADED_PAYLOAD_TOO_LARGE,
        vision.DEGRADED_TIMED_OUT,
        vision.DEGRADED_EMPTY_REPLY,
        vision.DEGRADED_UNPARSEABLE_REPLY,
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


# --- Degradation: category-specific messages (issue #18 / P6.2) -------------


def test_payload_too_large_produces_a_message_distinct_from_config_error():
    client = FakeSynthClient(exc=OpenRouterError("too large", status_code=413))

    result = run_fast_synth("some text", "a scene", client)

    message = result["final_output"].lower()
    assert "photo" in message
    assert "configuration" not in message
    assert_tts_safe(result["final_output"])


def test_config_error_never_tells_the_user_to_retry():
    client = FakeSynthClient(exc=OpenRouterError("authentication failed", status_code=401))

    result = run_fast_synth("some text", "a scene", client)

    message = result["final_output"].lower()
    assert "try again" not in message
    assert "retry" not in message


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

# Minimal fake for tts_node's provider seam (clarif_eye.tts) so this
# end-to-end test - which is about fast_synth, not tts - never touches the
# network via a real EdgeTtsProvider. Writes a minimal valid-looking mp3
# (an ID3 tag) so run_tts's own "looks like audio" check passes.
class _FakeTtsProvider:
    def synthesize(self, text, out_path):
        with open(out_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 32)


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

    result = graph.invoke(
        state,
        config={"configurable": {"client": client, "tts_provider": _FakeTtsProvider()}},
    )

    assert result["complexity_flag"] is False
    assert_tts_safe(result["final_output"])
    assert result["audio_file_path"] != ""


# --- FIX 5: assert_tts_safe now rejects previously-passing broken output ---
#
# These pin the exact three defects the fresh review found: the OLD
# sanitiser produced these strings and the OLD assert_tts_safe passed them.
# The strings below are literal, hand-verified outputs of the pre-fix code
# (not run through to_spoken_text) - they exist purely to prove the
# strengthened assertion now catches this shape of breakage.


def test_assert_tts_safe_rejects_old_garbled_markdown_link_output():
    garbled = "Check [our menu](a web link for prices."
    with pytest.raises(AssertionError):
        assert_tts_safe(garbled)


def test_assert_tts_safe_rejects_old_table_comma_soup_output():
    garbled = ", Item , Price , , Burger , $8.00 ,"
    with pytest.raises(AssertionError):
        assert_tts_safe(garbled)


def test_assert_tts_safe_rejects_untouched_html():
    garbled = "The sign says <b>OPEN</b> and <i>24 hours</i>."
    with pytest.raises(AssertionError):
        assert_tts_safe(garbled)


# --- FIX 2/3/4 through the node: link/table/HTML replies sanitise cleanly --


def test_reply_containing_a_markdown_link_sanitises_cleanly():
    client = FakeSynthClient(content="Check [our menu](https://example.com/menu) for prices.")

    result = run_fast_synth("some text", "a scene", client)

    assert_tts_safe(result["final_output"])
    assert "our menu" in result["final_output"]


def test_reply_containing_a_table_sanitises_cleanly():
    client = FakeSynthClient(
        content="| Item | Price |\n|------|-------|\n| Burger | $8.00 |\n"
    )

    result = run_fast_synth("some text", "a scene", client)

    assert_tts_safe(result["final_output"])
    assert "Burger" in result["final_output"]


def test_reply_containing_html_sanitises_cleanly():
    client = FakeSynthClient(content="The sign says <b>OPEN</b> and <i>24 hours</i>.")

    result = run_fast_synth("some text", "a scene", client)

    assert_tts_safe(result["final_output"])
    assert "OPEN" in result["final_output"]


# --- FIX 7: vision failure detection is structural, not textual ------------


def test_rewording_a_vision_degradation_message_does_not_break_detection(monkeypatch):
    from clarif_eye import vision

    monkeypatch.setattr(
        vision,
        "DEGRADED_LADDER_EXHAUSTED",
        "Sorry, the picture-reading service is temporarily too busy to help.",
    )

    def _boom():
        raise AssertionError("model must not be called for a degraded scene")

    monkeypatch.setattr(synth, "_default_client", _boom)

    result = run_fast_synth("", vision.DEGRADED_LADDER_EXHAUSTED)

    assert vision.is_degraded_scene(vision.DEGRADED_LADDER_EXHAUSTED)
    assert_tts_safe(result["final_output"])
    assert "temporarily too busy" in result["final_output"]


# --- FIX 8(a): table separator row is pinned, not dead weight --------------


def test_table_separator_row_is_dropped_not_read_as_dashes():
    client = FakeSynthClient(content="| Item | Price |\n|------|-------|\n| Burger | $8.00 |\n")

    result = run_fast_synth("some text", "a scene", client)

    # Exact pin: the separator row must contribute nothing at all - not
    # dashes, not an extra empty sentence between the two real rows.
    assert result["final_output"] == "Item, Price. Burger, $8.00."


# --- FIX 8(b): real ocr_output with a degraded scene_context ---------------


def test_real_ocr_output_is_not_dropped_when_scene_context_is_degraded():
    """Documents the deliberate choice: OCR text is appended, not discarded.

    Today vision.py's own _degraded() always pairs a degradation message
    with ocr_output == "", so this combination isn't reachable through the
    compiled graph - but run_fast_synth is public, and issue #8 may compose
    state differently. The model is still not called (the scene
    description is unreliable), but real OCR text must survive into
    final_output rather than being silently discarded.
    """

    result = run_fast_synth(
        "WIFI_PASSWORD_2026",
        "The vision model's response could not be understood.",
    )

    assert_tts_safe(result["final_output"])
    assert "WIFI_PASSWORD_2026" in result["final_output"]
    assert "could not be understood" in result["final_output"]
