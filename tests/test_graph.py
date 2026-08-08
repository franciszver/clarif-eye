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
from langgraph.graph import END

from clarif_eye.graph import (
    _UNCONDITIONAL_SUCCESSOR,
    COMPOSE_NODE,
    DEEP_PATH_NODE,
    DESCRIBE_ONE_NODE,
    TTS_NODE,
    VERIFY_ANSWER_NODE,
    build_graph,
    dynamic_router,
    next_node_after,
    vision_node,
)
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

# The two whole-run traces, named once because five tests across three files
# assert on them (issue #110 / P10.2 moved every photo-describing node down
# into a child graph, and a literal in each place was five literals to keep
# in step).
#
# HOW TO READ THEM: `vision`, `fast_synth`, `deep_path`, `research` and
# `analysis` are CHILD graph nodes now - `vision`/`fast_synth`/`deep_path`
# belong to the per-photo graph (clarif_eye.graph.build_photo_graph),
# `research`/`analysis` to the deep path inside it
# (clarif_eye.deep_path) - and the trace helper streams with subgraphs=True
# so they stay visible with their namespaces dropped. `describe_one` is the
# parent's own node completing once ONE photo's whole pipeline is done, so a
# three-photo submission has three of them; `compose` is the join that runs
# once per turn, and `tts` speaks the joined script once.
FAST_TRACE = ["entry", "vision", "fast_synth", "describe_one", "compose", "tts"]
# The same run seen WITHOUT subgraphs=True: the parent's own nodes only, with
# a whole photo arriving as one `describe_one` completion. Every photo run
# looks like this from up here, fast path or deep, one photo or several -
# only the number of `describe_one` entries changes.
PARENT_TRACE = ["entry", "describe_one", "compose", "tts"]
DEEP_TRACE = [
    "entry",
    "vision",
    "research",
    "analysis",
    "deep_path",
    "describe_one",
    "compose",
    "tts",
]


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
    # `scraper_data`'s None/"" sentinel (issue #81 / P9.2's "research never
    # ran" vs "ran and found nothing") is asserted in
    # tests/test_research.py now, against the per-photo graph it moved into
    # with issue #110 / P10.2 - see the expected-key set below.
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
        "final_output",
        "audio_file_path",
        "messages",
        "question",
        "verification_hold",
        # issue #93 / P9.12 - see ClarifEyeState.output_degraded.
        "output_degraded",
        # issue #110 / P10.2 - the submission (one entry per photo) and the
        # accumulating per-photo results the fan-out joins on. See
        # ClarifEyeState.photos / .photo_results.
        "photos",
        "photo_results",
    }
    assert set(state.keys()) == expected_keys


def test_state_typeddict_has_exactly_the_expected_keys():
    assert set(ClarifEyeState.__annotations__.keys()) == {
        "image_data",
        "ocr_output",
        "scene_context",
        "final_output",
        "audio_file_path",
        "messages",
        "question",
        "verification_hold",
        # issue #93 / P9.12 - see ClarifEyeState.output_degraded.
        "output_degraded",
        # issue #110 / P10.2 - the submission (one entry per photo) and the
        # accumulating per-photo results the fan-out joins on. See
        # ClarifEyeState.photos / .photo_results.
        "photos",
        "photo_results",
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
    # "deep_path" since issue #84 / P9.5: research is the first node of the
    # deep path's own child graph now, and the parent sees the whole path as
    # one node. The routing DECISION is unchanged - only the name of what it
    # points at.
    assert dynamic_router({"complexity_flag": True}) == DEEP_PATH_NODE


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
    assert trace == DEEP_TRACE
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

    # A PHOTO RUN'S goto IS A LIST OF Send OBJECTS since issue #110 / P10.2 -
    # one per submitted photo, targeting the per-photo child graph's wrapper
    # node. A submission of one is a fan-out of one, which is the whole
    # claim: the single-photo case is the degenerate multi-photo case, not a
    # separate branch. The assertion is still on the Command OBJECT, so
    # replacing the mechanism with a conditional edge would fail here.
    from langgraph.types import Send

    photo_run = entry_node({"question": None, "image_data": "base64imagedata"})
    assert isinstance(photo_run, Command)
    assert [send.node for send in photo_run.goto] == [DESCRIBE_ONE_NODE]
    assert all(isinstance(send, Send) for send in photo_run.goto)
    assert photo_run.goto[0].arg["image_data"] == "base64imagedata"
    assert photo_run.goto[0].arg["index"] == 0

    three_photos = entry_node(
        {
            "question": None,
            "photos": [
                {"image_data": "one", "cached": None},
                {"image_data": "two", "cached": None},
                {"image_data": "three", "cached": None},
            ],
        }
    )
    assert [send.arg["index"] for send in three_photos.goto] == [0, 1, 2]
    assert [send.arg["image_data"] for send in three_photos.goto] == ["one", "two", "three"]

    question_run = entry_node({"question": "what is the expiry date?"})
    assert isinstance(question_run, Command)
    assert question_run.goto == "followup"


def test_entry_node_treats_a_blank_question_as_a_photo_run():
    # Truthiness, not `is not None`: a blank question is not a question, and
    # routing one to `followup` would ask a model to answer nothing.
    from clarif_eye.graph import entry_destination

    # DESCRIBE_ONE_NODE, not "vision", since issue #110 / P10.2: `vision` is
    # the first node of the per-photo child graph now, and the parent's
    # destination is the wrapper the fan-out targets.
    assert entry_destination({"question": "   "}) == DESCRIBE_ONE_NODE
    # A never-checkpointed thread's state has no `question` key at all -
    # absent must behave like None, not raise.
    assert entry_destination({}) == DESCRIBE_ONE_NODE


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


def test_fast_path_populates_every_key_it_touches():
    graph = build_graph()
    state = make_initial_state("base64imagedata")
    client = FakeVisionClient(_reply("some text", "a room"))

    result, _ = run(graph, state, client=client, tts_provider=_FakeTtsProvider())

    assert result["ocr_output"] != ""
    assert result["scene_context"] != ""
    assert result["final_output"] != ""
    assert result["audio_file_path"] != ""
    # complexity_flag and scraper_data left ClarifEyeState in issue #110 /
    # P10.2 - see that schema's own comment. They are PER-PHOTO values
    # (which path one photo takes; what the lookup found for one photo) and
    # a turn of several photos has no single value for either, so they live
    # in clarif_eye.graph.PhotoState and clarif_eye.deep_path.DeepPathState
    # where the nodes that produce them actually run.
    # `scraper_data` used to be asserted None here ("the fast path never
    # ran research"). The same claim is now made where the key lives - see
    # tests/test_research.py's never-ran/found-nothing test, which drives
    # the per-photo graph's fast path for it - and the route itself is
    # already pinned by the trace test below. Both keys still show up in
    # `result` here, and correctly so: this helper merges the SUBGRAPH
    # chunks too, so it sees the per-photo values as the child produces
    # them. What they are absent from is the parent's SCHEMA, which the
    # expected-key test above is what pins.


# --- Fast path: complexity_flag False -----------------------------------


def test_fast_path_visits_vision_fast_synth_tts_only():
    # Short OCR text with no data-density signals keeps the router's
    # complexity heuristic under threshold, so complexity_flag=False and
    # the fast path is taken.
    graph = build_graph()
    state = make_initial_state("base64imagedata")
    client = FakeVisionClient(_reply("short text", "a room"))

    _, trace = run(graph, state, client=client)

    assert trace == FAST_TRACE
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

    # research and analysis are the DEEP PATH child graph's nodes (issue #84
    # / P9.5), and vision/deep_path are the PER-PHOTO child graph's (issue
    # #110 / P10.2); the trace helper streams with subgraphs=True so they all
    # stay visible. "describe_one" is the parent's own node completing once
    # that photo's whole pipeline is done, and "compose" is the join.
    assert trace == DEEP_TRACE
    assert "fast_synth" not in trace


# --- TTS_NODE: the one node name that travels outside this module --------


def test_tts_node_constant_names_a_node_the_compiled_graph_actually_has():
    """TTS_NODE is passed to graph.update_state(as_node=...) from
    clarif_eye.ui, which validates it against the compiled node set and
    raises InvalidUpdateError if it misses - and ui's never-raise guard
    would turn that into a write that silently did nothing (issue #82's
    wrong-photo blocker, resurrected).

    Written to fail LOUDLY AND BY NAME on a half-finished rename, rather
    than leaving it to be inferred from three indirect assertions in
    test_followup.py and test_ask_before_speaking.py. Reads the COMPILED
    graph's own node registry, not build_graph's source.
    """
    assert TTS_NODE in build_graph().nodes

    # And it is genuinely the last node: nothing follows it, and every
    # other path's declared successor is this same constant, so a rename
    # cannot leave half the topology pointing at a stale string.
    assert next_node_after(TTS_NODE, {}) is None
    # `compose` is the only node that names tts on the photo path since issue
    # #110 / P10.2: `fast_synth` and `deep_path` are the last nodes of the
    # PER-PHOTO child graph now, and a child cannot name the parent's tts -
    # the same rule `verify_numbers` already followed. That is also what
    # keeps "Turning it into speech" announced once per TURN rather than
    # once per photo.
    assert _UNCONDITIONAL_SUCCESSOR[COMPOSE_NODE] == TTS_NODE
    assert next_node_after(COMPOSE_NODE, {}) == TTS_NODE
    assert _UNCONDITIONAL_SUCCESSOR["fast_synth"] == END
    assert _UNCONDITIONAL_SUCCESSOR[DEEP_PATH_NODE] == END
    assert next_node_after("fast_synth", {}) is None
    assert next_node_after(DEEP_PATH_NODE, {}) is None
    # And the fan-out's own edge: a branch finishing leads to the join, which
    # has no phrase - the per-photo progress count is announced by
    # clarif_eye.ui._narrate_stream counting completions instead.
    assert _UNCONDITIONAL_SUCCESSOR[DESCRIBE_ONE_NODE] == COMPOSE_NODE
    # `followup` left this table in issue #92 / P9.11: its edge is CONDITIONAL
    # now (followup_destination), because an answer holding an unverifiable
    # number goes to the parent's own asking node first. With nothing held it
    # still resolves to this same constant, which is what a clean follow-up
    # does - and that asking node's own successor is tts, unlike the child
    # graph's, so a rename still cannot leave half the topology stale.
    assert "followup" not in _UNCONDITIONAL_SUCCESSOR
    assert next_node_after("followup", {}) == TTS_NODE
    assert _UNCONDITIONAL_SUCCESSOR[VERIFY_ANSWER_NODE] == TTS_NODE
    assert next_node_after(VERIFY_ANSWER_NODE, {}) == TTS_NODE
    # verify_numbers points at END, not at tts, since issue #84 / P9.5: it is
    # the last node of the CHILD graph, which cannot name the parent's tts.
    # That is also what keeps "Turning it into speech" from being announced
    # twice on a resume - see clarif_eye.graph._UNCONDITIONAL_SUCCESSOR.
    assert _UNCONDITIONAL_SUCCESSOR["verify_numbers"] == END
    assert next_node_after("verify_numbers", {}) is None
