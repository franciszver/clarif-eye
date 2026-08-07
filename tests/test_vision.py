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
from clarif_eye.vision import _parse_reply, is_degraded_scene, run_vision

# 200 words, no data-density signals: trips only the router's long-document
# word-count fallback (see clarif_eye.router), not the digit/currency/keyword
# signals - i.e. exercises the "genuinely long document" branch of the
# complexity heuristic.
LONG_TEXT = " ".join(["x"] * 200)


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


# --- Degradation: category-specific messages (issue #18 / P6.2) -------------
#
# The exhaustion/terminal-error branches above already existed; these prove
# run_vision picks a DIFFERENT message per Attempt.category / status_code,
# not just "some reasonable-looking sentence" - a user must be able to tell
# a busy service from a broken configuration.


def test_all_rungs_rate_limited_produces_a_busy_not_broken_message():
    from clarif_eye.client import Attempt

    attempts = (
        Attempt("model-a", "rate_limited", 429, "rate limited"),
        Attempt("model-b", "rate_limited", 429, "rate limited"),
    )
    client = FakeVisionClient(exc=LadderExhaustedError("eyes", attempts))

    result = run_vision("base64data", client)

    message = result["scene_context"].lower()
    assert "busy" in message
    assert "configuration" not in message
    assert "broken" not in message


def test_payload_too_large_produces_a_message_distinct_from_config_error():
    client = FakeVisionClient(exc=OpenRouterError("too large", status_code=413))

    result = run_vision("base64data", client)

    message = result["scene_context"].lower()
    assert "photo" in message
    assert "configuration" not in message


def test_config_error_never_tells_the_user_to_retry():
    client = FakeVisionClient(exc=OpenRouterError("authentication failed", status_code=401))

    result = run_vision("base64data", client)

    message = result["scene_context"].lower()
    assert "try again" not in message
    assert "retry" not in message


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


# --- Sentinel-delimited format (P1.8 / issue #29) ---------------------------
#
# The legacy OCR_TEXT:/SCENE: markers are anchored to line starts, so a
# LEGITIMATE photographed document whose own text contains a line starting
# "SCENE:" or "OCR_TEXT:" (a screenplay, a shooting schedule, a meeting
# agenda) collided with the parser's own section markers and had to degrade
# rather than guess (P1.2). Sentinel tokens that cannot plausibly occur in
# photographed text remove that collision: the sentinels are tried first,
# the legacy markers are a fallback (models drift back to the old format),
# and repeated/missing sentinels still degrade rather than guess.

SENTINEL_OCR = "<<<CLARIF_OCR>>>"
SENTINEL_SCENE = "<<<CLARIF_SCENE>>>"


def sentinel_reply(ocr="a coffee cup label", scene="a kitchen counter"):
    return f"{SENTINEL_OCR}\n{ocr}\n{SENTINEL_SCENE}\n{scene}"


def test_sentinel_format_happy_path_parses_ocr_and_scene():
    parsed = _parse_reply(sentinel_reply("Open 9-5", "a shop front"))

    assert parsed == ("Open 9-5", "a shop front")


def test_sentinel_format_ocr_body_containing_legacy_marker_lines_is_returned_verbatim():
    """The issue's acceptance test: a screenplay/agenda photo, sentinel format.

    The OCR section's own text contains lines that start with SCENE: and
    OCR_TEXT: - exactly the shape that collided with the legacy line-anchor
    parser. Inside a sentinel-delimited section this text is just body
    content and must come back whole, not truncated and not degraded.
    """
    reply = (
        f"{SENTINEL_OCR}\n"
        "INT. COFFEE SHOP - DAY\n"
        "\n"
        "OCR_TEXT: the barista call sheet on the counter\n"
        "SCENE: 4 - JOE enters, orders a coffee\n"
        "SCENE: 5 - JOE sits at the table\n"
        f"{SENTINEL_SCENE}\n"
        "a printed screenplay page taped to a coffee shop wall"
    )

    parsed = _parse_reply(reply)

    assert parsed is not None
    ocr_output, scene_context = parsed
    assert "OCR_TEXT: the barista call sheet on the counter" in ocr_output
    assert "SCENE: 4 - JOE enters, orders a coffee" in ocr_output
    assert "SCENE: 5 - JOE sits at the table" in ocr_output
    assert scene_context == "a printed screenplay page taped to a coffee shop wall"

    client = FakeVisionClient(content=reply)
    result = run_vision("base64data", client)
    assert "SCENE: 4 - JOE enters, orders a coffee" in result["ocr_output"]
    assert result["scene_context"] == "a printed screenplay page taped to a coffee shop wall"


def test_sentinel_format_repeated_sentinel_degrades():
    reply = (
        f"{SENTINEL_OCR}\nmenu\n{SENTINEL_OCR}\nmenu2\n{SENTINEL_SCENE}\na room"
    )

    assert _parse_reply(reply) is None


def test_sentinel_format_empty_ocr_section_is_valid():
    reply = f"{SENTINEL_OCR}\n{SENTINEL_SCENE}\na room"

    assert _parse_reply(reply) == ("", "a room")


def test_sentinel_format_empty_scene_section_degrades():
    reply = f"{SENTINEL_OCR}\nsome text\n{SENTINEL_SCENE}\n"

    assert _parse_reply(reply) is None


def test_missing_one_sentinel_falls_back_to_legacy_when_unambiguous():
    """A partial/garbled sentinel reply with unambiguous legacy markers still parses."""
    reply = f"{SENTINEL_OCR}\nOCR_TEXT: hello\nSCENE: a desk"

    assert _parse_reply(reply) == ("hello", "a desk")


def test_neither_sentinel_nor_legacy_markers_degrades():
    reply = "not the format I asked for at all, no markers here"

    assert _parse_reply(reply) is None


# --- Inline sentinels (P1.10 / issue #38) ------------------------------------
#
# A real model reply put the sentinel and its content on the SAME line
# instead of the sentinel alone on its own line. The old parser required an
# exact line match, so this good reply (a transcribed TOXIC warning) silently
# degraded into "could not be understood" instead of reaching the user. The
# fix accepts a sentinel that STARTS a line (after optional leading
# whitespace), treating the remainder of that line as the first line of that
# section - but a sentinel occurring MID-LINE (not at line start) must still
# be ignored, since that could be photographed text.


def test_sentinel_format_inline_content_on_same_line_as_both_sentinels():
    reply = f"{SENTINEL_OCR} SOME TEXT\n{SENTINEL_SCENE} a label"

    parsed = _parse_reply(reply)

    assert parsed == ("SOME TEXT", "a label")


def test_sentinel_format_inline_toxic_warning_reaches_ocr_output():
    """The live bug report: an inline TOXIC warning must not be discarded."""
    reply = (
        f"{SENTINEL_OCR} WARNING: TOXIC. DO NOT INGEST. Contains methanol. "
        "Keep from children.\n"
        f"{SENTINEL_SCENE} A white rectangular safety label."
    )

    parsed = _parse_reply(reply)

    assert parsed is not None
    ocr_output, scene_context = parsed
    assert "WARNING: TOXIC" in ocr_output
    assert "DO NOT INGEST" in ocr_output
    assert "methanol" in ocr_output
    assert "Keep from children" in ocr_output
    assert scene_context == "A white rectangular safety label."

    client = FakeVisionClient(content=reply)
    result = run_vision("base64data", client)
    assert "WARNING: TOXIC" in result["ocr_output"]


def test_sentinel_format_mixed_one_inline_one_own_line():
    reply = f"{SENTINEL_OCR} inline text\n{SENTINEL_SCENE}\na room on its own line"

    parsed = _parse_reply(reply)

    assert parsed == ("inline text", "a room on its own line")


def test_sentinel_format_mid_line_sentinel_is_not_treated_as_a_delimiter():
    """A sentinel that does not start a line (e.g. photographed text quoting
    it) must never be treated as a section delimiter."""
    reply = (
        f"{SENTINEL_OCR}\n"
        f"the label reads {SENTINEL_SCENE} in bold letters\n"
        f"{SENTINEL_SCENE}\n"
        "a close-up of printed text"
    )

    parsed = _parse_reply(reply)

    assert parsed is not None
    ocr_output, scene_context = parsed
    assert f"the label reads {SENTINEL_SCENE} in bold letters" in ocr_output
    assert scene_context == "a close-up of printed text"


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

# Minimal fake for tts_node's provider seam (clarif_eye.tts) so these
# end-to-end tests - which are about vision, not tts - never touch the
# network via a real EdgeTtsProvider. Writes a minimal valid-looking mp3
# (an ID3 tag) so run_tts's own "looks like audio" check passes.
class _FakeTtsProvider:
    def synthesize(self, text, out_path):
        with open(out_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 32)


def _invoke_collecting_trace(graph, state, config):
    """Run the compiled graph via stream(..., stream_mode="updates") and
    return (final_state, visited_node_names_in_order) - replaces the old
    config["configurable"]["trace"] seam (issue #80 / P9.1, same helper
    shape as test_graph.py's `run()`): each stream chunk is keyed by the
    node that just completed, so collecting those keys in arrival order is
    a drop-in replacement for what graph._record used to append."""
    result = dict(state)
    trace = []
    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            result.update(update)
            trace.append(node_name)
    return result, trace


def test_full_compiled_graph_runs_end_to_end_with_fake_client_fast_path():
    client = FakeVisionClient(content=well_formed_reply("short text", "a room"))
    graph = build_graph()
    state = make_initial_state("base64data")

    result = graph.invoke(
        state,
        config={"configurable": {"client": client, "tts_provider": _FakeTtsProvider()}},
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

    result, trace = _invoke_collecting_trace(
        graph, state, config={"configurable": {"client": client, "tts_provider": _FakeTtsProvider()}}
    )

    assert result["complexity_flag"] is True
    assert trace == ["vision", "research", "analysis", "tts"]
    assert result["final_output"] != ""
    assert result["audio_file_path"] != ""


def test_full_compiled_graph_degrades_gracefully_and_still_reaches_tts():
    client = FakeVisionClient(exc=LadderExhaustedError("eyes", ()))
    graph = build_graph()
    state = make_initial_state("base64data")

    result, trace = _invoke_collecting_trace(
        graph, state, config={"configurable": {"client": client, "tts_provider": _FakeTtsProvider()}}
    )

    assert trace[-1] == "tts"
    assert result["audio_file_path"] != ""
    assert isinstance(result["complexity_flag"], bool)


# --- FIX 7: degradation detection is structural (named constants + a
# public predicate), not a textual guess at English prose --------------------


def test_is_degraded_scene_true_for_each_named_degradation_constant():
    for message in (
        vision.DEGRADED_CONFIG_ERROR,
        vision.DEGRADED_LADDER_EXHAUSTED,
        vision.DEGRADED_UNEXPECTED_ERROR,
        vision.DEGRADED_EMPTY_REPLY,
        vision.DEGRADED_UNPARSEABLE_REPLY,
    ):
        assert is_degraded_scene(message) is True


def test_is_degraded_scene_false_for_a_real_scene_description():
    assert is_degraded_scene("a kitchen counter with a coffee cup on it") is False


def test_rewording_a_degradation_constant_still_detects_via_the_constant(monkeypatch):
    """Detection must key off the constant, not a hardcoded copy of its text.

    Reword DEGRADED_UNPARSEABLE_REPLY (as issues #15/#18 might, for
    accessibility) and confirm is_degraded_scene still recognises the new
    wording - because it checks against the constant's current value, not
    a fixed prefix/keyword baked into is_degraded_scene itself.
    """
    monkeypatch.setattr(
        vision,
        "DEGRADED_UNPARSEABLE_REPLY",
        "I couldn't make sense of what the vision model sent back.",
    )

    assert is_degraded_scene("I couldn't make sense of what the vision model sent back.") is True
    # The stale, un-reworded text no longer matches - proving detection
    # really does key off the (now-updated) constant, not old wording.
    assert is_degraded_scene("The vision model's response could not be understood.") is False
