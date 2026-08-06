"""Tests for the LangGraph state schema and compiling graph skeleton.

All nodes in this issue are stubs (no model calls, no network, no
filesystem). Tests assert on which nodes actually ran, not just on whether
output ended up truthy - a truthy check would pass even if the routing
were backwards.

Node visitation is tracked via a mutable list passed through the run
config's `configurable` dict (not through the state schema itself, which
must contain exactly the 7 architecture-doc keys - see state.py).
"""

import pytest

from clarif_eye.graph import build_graph, dynamic_router, vision_node
from clarif_eye.state import ClarifEyeState, make_initial_state


def run(graph, state):
    trace = []
    result = graph.invoke(state, config={"configurable": {"trace": trace}})
    return result, trace


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
    result = vision_node({"image_data": "base64imagedata"})
    assert "complexity_flag" in result
    assert isinstance(result["complexity_flag"], bool)


def test_full_graph_routes_using_node_owned_complexity_flag_not_caller_value():
    # image_data contains "complex", so vision_node's stub heuristic
    # computes complexity_flag=True - the OPPOSITE of what we set below on
    # the initial state. If a future vision node stops returning
    # complexity_flag, LangGraph's partial-update merge would silently
    # leave the caller's False in place and this test would fail.
    graph = build_graph()
    state = make_initial_state("a very complex base64imagedata payload")
    state["complexity_flag"] = False

    result, trace = run(graph, state)

    assert result["complexity_flag"] is True
    assert trace == ["vision", "research", "analysis", "tts"]
    assert "fast_synth" not in trace


# --- Graph: end-to-end -----------------------------------------------------


def test_compiled_graph_runs_end_to_end_and_returns_every_state_key():
    graph = build_graph()
    state = make_initial_state("base64imagedata")

    result, _ = run(graph, state)

    for key in ClarifEyeState.__annotations__.keys():
        assert key in result


def test_fast_path_populates_every_key_it_touches_scraper_data_stays_empty():
    graph = build_graph()
    state = make_initial_state("base64imagedata")

    result, _ = run(graph, state)

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
    # image_data has no "complex" substring, so vision_node's stub
    # heuristic computes complexity_flag=False and the fast path is taken.
    graph = build_graph()
    state = make_initial_state("base64imagedata")

    _, trace = run(graph, state)

    assert trace == ["vision", "fast_synth", "tts"]
    assert "research" not in trace
    assert "analysis" not in trace


# --- Research path: complexity_flag True ---------------------------------


def test_research_path_visits_vision_research_analysis_tts_only():
    # image_data contains "complex", so vision_node's stub heuristic
    # computes complexity_flag=True and the research path is taken.
    graph = build_graph()
    state = make_initial_state("a complex base64imagedata payload")

    _, trace = run(graph, state)

    assert trace == ["vision", "research", "analysis", "tts"]
    assert "fast_synth" not in trace
