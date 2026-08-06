"""Compiling LangGraph skeleton for Clarif-Eye.

Wires: vision -> dynamic_router -> fast_synth OR (research -> analysis) -> tts -> END

Every node here is a STUB: it returns a state update with placeholder
values and does not call a model, the network, or the filesystem. The
router stub decides purely from state["complexity_flag"] so both paths can
be driven deterministically in tests. Node functions are named and
importable individually so later issues (#5-#8) can replace them one at a
time without restructuring the graph.

Node visitation is recorded into config["configurable"]["trace"] (a list
supplied by the caller), not into the state schema itself - the state
schema is exactly the 7 architecture-doc keys, nothing more.
"""

from langgraph.graph import END, StateGraph

from clarif_eye.state import ClarifEyeState


def _record(config, node_name):
    trace = config.get("configurable", {}).get("trace") if config else None
    if trace is not None:
        trace.append(node_name)


def vision_node(state, config=None):
    """Stub for the vision node (issue #5): OCR + scene description.

    ALSO sets complexity_flag as part of its own state update: routing
    depends on this key, so it must be documented as this node's
    responsibility, not something left to the caller. If a future vision
    node forgets to return this key, LangGraph's partial-update merge means
    whatever the caller happened to set (or make_initial_state's hardcoded
    False) silently survives untouched - every request would take the fast
    path with no exception and no failing test. Issues #5/#6 own the real
    heuristic and MUST keep returning this key.

    The stub keeps the value simple, deterministic, and derived from the
    node's own stub OCR pass over the input (a trivial substring check)
    rather than from any complexity_flag the caller may have set - this
    also lets both routing branches stay exercisable through the compiled
    graph in tests, instead of being collapsed to a single hardcoded value.
    """
    _record(config, "vision")
    ocr_output = "stub ocr output"
    complexity_flag = "complex" in state.get("image_data", "").lower()
    return {
        "ocr_output": ocr_output,
        "scene_context": "stub scene context",
        "complexity_flag": complexity_flag,
    }


def fast_synth_node(state, config=None):
    """Stub for the fast-path synthesis node (issue #6)."""
    _record(config, "fast_synth")
    return {"final_output": "stub fast synthesis output"}


def research_node(state, config=None):
    """Stub for the research node (issue #7): web-lookup on the deep path."""
    _record(config, "research")
    return {"scraper_data": "stub scraper data"}


def analysis_node(state, config=None):
    """Stub for the deep-analysis node (issue #7/#8)."""
    _record(config, "analysis")
    return {"final_output": "stub analysis output"}


def tts_node(state, config=None):
    """Stub for the text-to-speech node (issue #8)."""
    _record(config, "tts")
    return {"audio_file_path": "stub/audio/path.mp3"}


def dynamic_router(state):
    """Route on state["complexity_flag"]: True -> research, False -> fast_synth.

    complexity_flag must be an actual bool. TypedDict gives no runtime
    protection, so `if state["complexity_flag"]` would otherwise route on
    truthiness - a confidence float or an error string used as a "no"
    sentinel would silently pick the wrong path, and the two paths differ
    by a whole model tier and ~17s of budget. Fail loudly instead of
    routing on a value that merely happens to be truthy or falsy.
    """
    if "complexity_flag" not in state:
        raise KeyError(
            "dynamic_router: state is missing required key 'complexity_flag'"
        )
    flag = state["complexity_flag"]
    if not isinstance(flag, bool):
        raise TypeError(
            "dynamic_router: state['complexity_flag'] must be bool, got "
            f"{type(flag).__name__} ({flag!r})"
        )
    return "research" if flag else "fast_synth"


def build_graph():
    """Build and compile the Clarif-Eye graph."""
    builder = StateGraph(ClarifEyeState)

    builder.add_node("vision", vision_node)
    builder.add_node("fast_synth", fast_synth_node)
    builder.add_node("research", research_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("tts", tts_node)

    builder.set_entry_point("vision")
    builder.add_conditional_edges(
        "vision",
        dynamic_router,
        {"fast_synth": "fast_synth", "research": "research"},
    )
    builder.add_edge("fast_synth", "tts")
    builder.add_edge("research", "analysis")
    builder.add_edge("analysis", "tts")
    builder.add_edge("tts", END)

    return builder.compile()
