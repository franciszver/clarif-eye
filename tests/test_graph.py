"""Tests for the LangGraph state schema and compiling graph skeleton.

All nodes in this issue are stubs (no model calls, no network, no
filesystem). Tests assert on which nodes actually ran, not just on whether
output ended up truthy - a truthy check would pass even if the routing
were backwards.

Node visitation is observed via graph.stream(..., stream_mode="updates")
(issue #80 / P9.1), which yields one dict per COMPLETED node keyed by node
name - see the `run()` helper below - rather than a caller-supplied trace
list threaded through config["configurable"].
"""

import pytest

from clarif_eye.client import CompletionResult
from clarif_eye.graph import build_graph, dynamic_router, vision_node
from clarif_eye.state import ClarifEyeState, make_initial_state

from tests._stream_helpers import drain_stream_collecting_trace

# vision_node now calls the real "eyes" ladder (see tests/test_vision.py for
# the vision-specific behavior). The graph-shape tests below only care about
# routing and key presence, so they inject this no-network fake client
# rather than exercising vision parsing/degradation logic themselves.
# 200 words, no data-density signals: trips only the router's long-document
# word-count fallback (see clarif_eye.router), not the digit/currency/keyword
# signals.
LONG_OCR_TEXT = " ".join(["x"] * 200)


class FakeVisionClient:
    def __init__(self, content):
        self.content = content

    def complete(self, role, messages, **params):
        return CompletionResult(content=self.content, model="fake-eyes-model:free")


def _reply(ocr, scene):
    return f"OCR_TEXT: {ocr}\nSCENE: {scene}"


def run(graph, state, client=None, tts_provider=None):
    """Assemble the configurable dict this file's tests need, then drain
    the stream via the shared tests/_stream_helpers.py helper (issue #80 /
    P9.1) - `visited` replaces the old trace-list config seam."""
    configurable = {}
    if client is not None:
        configurable["client"] = client
    if tts_provider is not None:
        configurable["tts_provider"] = tts_provider
    return drain_stream_collecting_trace(graph, state, {"configurable": configurable})


# Minimal fake for tts_node's provider seam (clarif_eye.tts) so tests that
# assert on audio_file_path - without being about tts itself - never touch
# the network via a real EdgeTtsProvider. Writes a minimal valid-looking
# mp3 (an ID3 tag) so run_tts's own "looks like audio" check passes.
class _FakeTtsProvider:
    def synthesize(self, text, out_path):
        with open(out_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 32)


# --- State helper ------------------------------------------------------


def test_make_initial_state_has_every_key_with_correct_types():
    state = make_initial_state("base64imagedata")

    assert state["image_data"] == "base64imagedata"
    assert state["ocr_output"] == ""
    assert state["scene_context"] == ""
    assert state["complexity_flag"] is False
    # None, not "" - issue #81 / P9.2's explicit sentinel: None means
    # "research never ran yet" (this key), distinct from "" (research ran
    # and found nothing usable) - see state.py's ClarifEyeState.scraper_data
    # comment.
    assert state["scraper_data"] is None
    assert state["final_output"] == ""
    assert state["audio_file_path"] == ""
    assert state["messages"] == []
    # None, not {} (issue #83 / P9.4): None means "nothing is being held
    # back pending a question", and a photo run seeding it explicitly is
    # what CLEARS a hold left over from a run the user abandoned mid-
    # question - see state.py's ClarifEyeState.verification_hold.
    assert state["verification_hold"] is None
    # None, not "" (issue #82 / P9.3): None means "this is a photo run, not
    # a question", and a photo run seeding it explicitly is what RESETS a
    # question left over from the previous turn on a checkpointed thread -
    # see state.py's ClarifEyeState.question comment.
    assert state["question"] is None

    expected_keys = {
        "image_data",
        "ocr_output",
        "scene_context",
        "complexity_flag",
        "scraper_data",
        "final_output",
        "audio_file_path",
        "messages",
        "question",
        "verification_hold",
    }
    assert set(state.keys()) == expected_keys


def test_state_typeddict_has_exactly_the_expected_keys():
    assert set(ClarifEyeState.__annotations__.keys()) == {
        "image_data",
        "ocr_output",
        "scene_context",
        "complexity_flag",
        "scraper_data",
        "final_output",
        "audio_file_path",
        "messages",
        "question",
        "verification_hold",
    }


def test_make_initial_state_rejects_empty_image_data():
    with pytest.raises(ValueError):
        make_initial_state("")


def test_make_initial_state_rejects_none_image_data():
    with pytest.raises(ValueError):
        make_initial_state(None)


def test_make_initial_state_rejects_blank_image_data():
    with pytest.raises(ValueError):
        make_initial_state("   ")


# --- dynamic_router: guards its own input --------------------------------


def test_dynamic_router_routes_to_research_when_flag_true():
    assert dynamic_router({"complexity_flag": True}) == "research"


def test_dynamic_router_routes_to_fast_synth_when_flag_false():
    assert dynamic_router({"complexity_flag": False}) == "fast_synth"


def test_dynamic_router_raises_on_missing_complexity_flag():
    with pytest.raises(KeyError):
        dynamic_router({})


def test_dynamic_router_raises_on_non_bool_complexity_flag():
    with pytest.raises(TypeError):
        dynamic_router({"complexity_flag": "not-a-bool-but-a-string"})


# --- vision_node: owns complexity_flag ------------------------------------


def test_vision_node_returns_complexity_flag_key():
    # Direct call, no graph: proves the key is part of THIS node's return
    # value, not something it happens to inherit unchanged from the caller.
    client = FakeVisionClient(_reply("some text", "a room"))
    result = vision_node({"image_data": "base64imagedata"}, client=client)
    assert "complexity_flag" in result
    assert isinstance(result["complexity_flag"], bool)


def test_full_graph_routes_using_node_owned_complexity_flag_not_caller_value():
    # The fake reply's OCR text is long enough to trip the router's
    # complexity heuristic, computing complexity_flag=True - the OPPOSITE of
    # what we set below on the initial state. If a future vision node stops
    # returning complexity_flag, LangGraph's partial-update merge would
    # silently leave the caller's False in place and this test would fail.
    graph = build_graph()
    state = make_initial_state("base64imagedata payload")
    state["complexity_flag"] = False
    client = FakeVisionClient(_reply(LONG_OCR_TEXT, "a busy scene"))

    result, trace = run(graph, state, client=client)

    assert result["complexity_flag"] is True
    assert trace == ["entry", "vision", "research", "analysis", "tts"]
    assert "fast_synth" not in trace


# --- entry_node: the Command(goto) mechanism (issue #82 / P9.3) -----------
#
# This graph uses BOTH of LangGraph's routing mechanisms on purpose, each
# where it fits (see clarif_eye.graph's "TWO ROUTING MECHANISMS" docstring
# block). The conditional edge out of `vision` is already covered by the
# dynamic_router tests and the two path-trace tests above; these cover the
# other one. The assertion is on the Command OBJECT, not just on where the
# run ended up, so replacing the mechanism with a conditional edge that
# happens to route the same way would fail here rather than pass silently.


def test_entry_node_routes_with_a_command_not_a_returned_state_update():
    from langgraph.types import Command

    from clarif_eye.graph import entry_node

    photo_run = entry_node({"question": None})
    assert isinstance(photo_run, Command)
    assert photo_run.goto == "vision"

    question_run = entry_node({"question": "what is the expiry date?"})
    assert isinstance(question_run, Command)
    assert question_run.goto == "followup"


def test_entry_node_treats_a_blank_question_as_a_photo_run():
    # Truthiness, not `is not None`: a blank question is not a question, and
    # routing one to `followup` would ask a model to answer nothing.
    from clarif_eye.graph import entry_destination

    assert entry_destination({"question": "   "}) == "vision"
    # A never-checkpointed thread's state has no `question` key at all -
    # absent must behave like None, not raise.
    assert entry_destination({}) == "vision"


def test_entry_destination_raises_a_named_type_error_on_a_non_string_question():
    # Same discipline dynamic_router applies to complexity_flag: without an
    # explicit check this would be a bare AttributeError from .strip(),
    # raised deep inside a node with nothing naming the key or the value.
    from clarif_eye.graph import entry_destination

    with pytest.raises(TypeError) as excinfo:
        entry_destination({"question": 42})
    assert "question" in str(excinfo.value)

    with pytest.raises(TypeError):
        entry_destination({"question": ["what is the expiry date?"]})


# --- next_node_after: unknown stream keys must not crash narration --------
#
# clarif_eye.ui's narration iterates over whatever keys LangGraph's stream
# produces, and those are not all node names - LangGraph emits RESERVED keys
# too. "__interrupt__" is the concrete one arriving with issue #83
# (human-in-the-loop interrupts). A KeyError from this shared code would
# take down a run whose answer had already been computed, so an unknown name
# degrades to "no narration for this step" instead.


def test_next_node_after_returns_none_for_an_unknown_stream_key():
    from clarif_eye.graph import next_node_after

    assert next_node_after("__interrupt__", {}) is None
    assert next_node_after("not_a_real_node_name", {}) is None


def test_a_question_run_skips_vision_entirely():
    from clarif_eye.client import CompletionResult

    class _AnsweringClient:
        def complete(self, role, messages, **params):
            return CompletionResult(content="It says next April.", model="fake-brain-model:free")

    graph = build_graph()
    # A PARTIAL state, exactly what clarif_eye.ui._run_followup_events
    # passes for a follow-up: the question plus whatever the thread already
    # had. No image_data at all - if the run reached vision_node it would
    # raise KeyError, so this also proves vision is genuinely skipped.
    state = {
        "ocr_output": "best before next April",
        "scene_context": "a jar of jam",
        "question": "what is the expiry date?",
    }

    _, trace = run(graph, state, client=_AnsweringClient(), tts_provider=_FakeTtsProvider())

    assert trace == ["entry", "followup", "tts"]


# --- Graph: end-to-end -----------------------------------------------------


def test_compiled_graph_runs_end_to_end_and_returns_every_state_key():
    graph = build_graph()
    state = make_initial_state("base64imagedata")
    client = FakeVisionClient(_reply("some text", "a room"))

    result, _ = run(graph, state, client=client)

    for key in ClarifEyeState.__annotations__.keys():
        assert key in result


def test_fast_path_populates_every_key_it_touches_scraper_data_stays_empty():
    graph = build_graph()
    state = make_initial_state("base64imagedata")
    client = FakeVisionClient(_reply("some text", "a room"))

    result, _ = run(graph, state, client=client, tts_provider=_FakeTtsProvider())

    assert result["ocr_output"] != ""
    assert result["scene_context"] != ""
    assert result["final_output"] != ""
    assert result["audio_file_path"] != ""
    assert result["complexity_flag"] is False
    # Fast path never runs research_node, so scraper_data legitimately
    # stays at its make_initial_state default - present, but not
    # populated. That default is None (issue #81 / P9.2), not "": None
    # means "research never ran", distinct from research.run_research's
    # own "" ("ran, found nothing") - see state.py.
    assert result["scraper_data"] is None


# --- Fast path: complexity_flag False -----------------------------------


def test_fast_path_visits_vision_fast_synth_tts_only():
    # Short OCR text with no data-density signals keeps the router's
    # complexity heuristic under threshold, so complexity_flag=False and
    # the fast path is taken.
    graph = build_graph()
    state = make_initial_state("base64imagedata")
    client = FakeVisionClient(_reply("short text", "a room"))

    _, trace = run(graph, state, client=client)

    assert trace == ["entry", "vision", "fast_synth", "tts"]
    assert "research" not in trace
    assert "analysis" not in trace


# --- Research path: complexity_flag True ---------------------------------


def test_research_path_visits_vision_research_analysis_tts_only():
    # Long OCR text trips the router's complexity heuristic (long-document
    # word-count fallback), so complexity_flag=True and the research path
    # is taken.
    graph = build_graph()
    state = make_initial_state("base64imagedata")
    client = FakeVisionClient(_reply(LONG_OCR_TEXT, "a busy scene"))

    _, trace = run(graph, state, client=client)

    assert trace == ["entry", "vision", "research", "analysis", "tts"]
    assert "fast_synth" not in trace
