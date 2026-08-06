"""Tests for the LangGraph state schema and compiling graph skeleton.

All nodes in this issue are stubs (no model calls, no network, no
filesystem). Tests assert on which nodes actually ran, not just on whether
output ended up truthy - a truthy check would pass even if the routing
were backwards.

Node visitation is tracked via a mutable list passed through the run
config's `configurable` dict (not through the state schema itself, which
must contain exactly the 7 architecture-doc keys - see state.py).
"""

from clarif_eye.graph import build_graph
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


# --- Graph: end-to-end, every key populated -----------------------------


def test_compiled_graph_runs_end_to_end_and_populates_every_state_key():
    graph = build_graph()
    state = make_initial_state("base64imagedata")
    state["complexity_flag"] = False

    result, _ = run(graph, state)

    for key in ClarifEyeState.__annotations__.keys():
        assert key in result
    assert result["ocr_output"] != ""
    assert result["scene_context"] != ""
    assert result["final_output"] != ""
    assert result["audio_file_path"] != ""
    assert isinstance(result["complexity_flag"], bool)


# --- Fast path: complexity_flag False -----------------------------------


def test_fast_path_visits_vision_fast_synth_tts_only():
    graph = build_graph()
    state = make_initial_state("base64imagedata")
    state["complexity_flag"] = False

    _, trace = run(graph, state)

    assert trace == ["vision", "fast_synth", "tts"]
    assert "research" not in trace
    assert "analysis" not in trace


# --- Research path: complexity_flag True ---------------------------------


def test_research_path_visits_vision_research_analysis_tts_only():
    graph = build_graph()
    state = make_initial_state("base64imagedata")
    state["complexity_flag"] = True

    _, trace = run(graph, state)

    assert trace == ["vision", "research", "analysis", "tts"]
    assert "fast_synth" not in trace
