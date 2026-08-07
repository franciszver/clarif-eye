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
    """Run the compiled graph via stream(..., stream_mode="updates") and
    return (final_state, visited_node_names_in_order) - visited replaces
    the old trace-list config seam (issue #80 / P9.1): each stream chunk
    is keyed by the node that just completed, so collecting those keys in
    arrival order is a drop-in replacement for what _record used to do."""
    configurable = {}
    if client is not None:
        configurable["client"] = client
    if tts_provider is not None:
        configurable["tts_provider"] = tts_provider
    result = dict(state)
    visited = []
    for chunk in graph.stream(state, config={"configurable": configurable}, stream_mode="updates"):
        for node_name, update in chunk.items():
            result.update(update)
            visited.append(node_name)
    return result, visited


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
    assert state["scraper_data"] == ""
    assert state["final_output"] == ""
    assert state["audio_file_path"] == ""

    expected_keys = {
        "image_data",
        "ocr_output",
        "scene_context",
        "complexity_flag",
        "scraper_data",
        "final_output",
        "audio_file_path",
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
    assert trace == ["vision", "research", "analysis", "tts"]
    assert "fast_synth" not in trace


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
    # stays at its "" default - present, but not populated.
    assert result["scraper_data"] == ""


# --- Fast path: complexity_flag False -----------------------------------


def test_fast_path_visits_vision_fast_synth_tts_only():
    # Short OCR text with no data-density signals keeps the router's
    # complexity heuristic under threshold, so complexity_flag=False and
    # the fast path is taken.
    graph = build_graph()
    state = make_initial_state("base64imagedata")
    client = FakeVisionClient(_reply("short text", "a room"))

    _, trace = run(graph, state, client=client)

    assert trace == ["vision", "fast_synth", "tts"]
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

    assert trace == ["vision", "research", "analysis", "tts"]
    assert "fast_synth" not in trace
