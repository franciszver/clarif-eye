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
from clarif_eye.synth import run_fast_synth
from clarif_eye.vision import run_vision


def _record(config, node_name):
    trace = config.get("configurable", {}).get("trace") if config else None
    if trace is not None:
        trace.append(node_name)


def vision_node(state, config=None, client=None):
    """Vision node (issue #5/P1.2): calls the eyes ladder for OCR + scene description.

    ALSO sets complexity_flag as part of its own state update: routing
    depends on this key, so it must be documented as this node's
    responsibility, not something left to the caller. If a future vision
    node forgets to return this key, LangGraph's partial-update merge means
    whatever the caller happened to set (or make_initial_state's hardcoded
    False) silently survives untouched - every request would take the fast
    path with no exception and no failing test. Issue #6 (P1.3) owns the
    real complexity heuristic and MUST keep returning this key.

    The substantive logic (request building, parsing, degradation) lives in
    clarif_eye.vision.run_vision so this stays a thin adapter. `client` is
    injectable directly (for unit tests calling this function) or via
    config["configurable"]["client"] (for tests driving the compiled
    graph, the same pattern already used for `trace`); when neither is
    supplied, run_vision constructs a real OpenRouterClient lazily.
    """
    _record(config, "vision")
    if client is None:
        client = (config or {}).get("configurable", {}).get("client")
    return run_vision(state["image_data"], client)


def fast_synth_node(state, config=None, client=None):
    """Fast-path synthesis node (issue #7/P1.4): turns vision output into spoken text.

    The substantive logic (prompt building, sanitisation, degradation)
    lives in clarif_eye.synth.run_fast_synth so this stays a thin adapter,
    the same pattern vision_node already uses. `client` is injectable
    directly or via config["configurable"]["client"]; when neither is
    supplied, run_fast_synth constructs a real OpenRouterClient lazily.
    """
    _record(config, "fast_synth")
    if client is None:
        client = (config or {}).get("configurable", {}).get("client")
    return run_fast_synth(state["ocr_output"], state["scene_context"], client)


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
