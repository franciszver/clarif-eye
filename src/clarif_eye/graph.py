"""Compiling LangGraph skeleton for Clarif-Eye.

Wires: vision -> dynamic_router -> fast_synth OR (research -> analysis) -> tts -> END

Every node here is a STUB: it returns a state update with placeholder
values and does not call a model, the network, or the filesystem. The
router stub decides purely from state["complexity_flag"] so both paths can
be driven deterministically in tests. Node functions are named and
importable individually so later issues (#5-#8) can replace them one at a
time without restructuring the graph.

Node visitation is observable via the compiled graph's own
`graph.stream(state, config=config, stream_mode="updates")` (issue #80 /
P9.1) - one dict per COMPLETED node, keyed by node name - rather than a
caller-supplied trace list threaded through config["configurable"]. That
gives callers (and tests) real per-node progress instead of a debugging
side-channel, and needs nothing extra recorded by the nodes themselves.
"""

import time

from langgraph.graph import END, StateGraph

from clarif_eye.analysis import run_analysis
from clarif_eye.research import run_research
from clarif_eye.state import ClarifEyeState
from clarif_eye.synth import run_fast_synth
from clarif_eye.tts import run_tts
from clarif_eye.vision import run_vision

# Total-pipeline deadline (issue #17 / P6.1). D16 gave "eyes"/"brain" their
# own per-role ceilings inside client.complete (30s/45s), but nothing
# bounded the whole graph run: the ceilings cap each model call, not the
# sum of them plus research and tts, so the TAIL was unbounded. One live UI
# run was observed at 99.0s - an outlier, not typical (measured medians are
# ~21-31s; see the DEFAULT justification below) - but an unbounded tail is
# a structural gap regardless of how often it bites. This closes it.
#
# MECHANISM: config["configurable"]["deadline"], an ABSOLUTE
# time.monotonic() timestamp, set once by the caller before invoking the
# graph (see clarif_eye.ui.handle_submit) - not a new 8th state key (the
# schema stays at exactly the 7 architecture-doc keys, see state.py). Every
# node that makes a model or network call reads it via _deadline_exceeded
# below and, if it has already passed, skips that call and asks its
# module's run_* function to degrade from whatever state is already known
# instead (see vision.py/synth.py/analysis.py/research.py's
# `deadline_exceeded` parameter and *._degrade_from_known). tts_node is
# deliberately NEVER gated on the deadline: it produces the actual spoken
# deliverable and is the one node whose own latency (not blowing the
# budget further) matters more than respecting a budget that's already
# gone - skipping it would turn "degraded but spoken" into "silent",
# exactly the failure this issue exists to avoid. Absence of the
# "deadline" key means unbounded, i.e. today's behavior unchanged (every
# pre-existing caller/test that never sets one).
#
# DEFAULT justification: 60.0s. Measured research-path medians are ~21-31s
# depending on scraper configuration, with an observed maximum of 60.3s
# across runs (n=5 per config, scripts/benchmark_pipeline.py). The 99.0s
# from #17 was a single-run outlier from a live UI session, not typical
# behavior. The deadline's real structural purpose: per-role ceilings
# (eyes 30s + brain 45s) bound individual model calls, but nothing bounded
# a whole-pipeline run until this mechanism - the tail was theoretically
# unbounded. The 60s default knowingly fires on roughly 1 in 15 measured
# runs, gracefully degrading otherwise-fine-but-slow runs rather than
# allowing latency to exceed 75s+. This is deliberate: a user who cannot
# see a spinner is better served by degraded speech at 60s than by full
# quality at 75s or beyond. Provisional, like the scraper cap in analysis.py
# - retune via scripts/benchmark_pipeline.py.
DEFAULT_PIPELINE_BUDGET_SECONDS = 60.0


def _deadline_exceeded(config):
    """True if config["configurable"]["deadline"] (an absolute
    time.monotonic() timestamp) is set and has already passed.

    No "deadline" key at all means unbounded - always False - so every
    caller/test that never sets one keeps exactly today's behavior.
    """
    deadline = (config or {}).get("configurable", {}).get("deadline")
    if deadline is None:
        return False
    return time.monotonic() >= deadline


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
    graph); when neither is supplied, run_vision constructs a real
    OpenRouterClient lazily.

    Checks the total-pipeline deadline (see _deadline_exceeded above) and
    passes the result to run_vision, which skips the eyes-ladder call
    entirely if it has already passed - see this module's top-level
    "Total-pipeline deadline" docstring block.
    """
    if client is None:
        client = (config or {}).get("configurable", {}).get("client")
    return run_vision(state["image_data"], client, deadline_exceeded=_deadline_exceeded(config))


def fast_synth_node(state, config=None, client=None):
    """Fast-path synthesis node (issue #7/P1.4): turns vision output into spoken text.

    The substantive logic (prompt building, sanitisation, degradation)
    lives in clarif_eye.synth.run_fast_synth so this stays a thin adapter,
    the same pattern vision_node already uses. `client` is injectable
    directly or via config["configurable"]["client"]; when neither is
    supplied, run_fast_synth constructs a real OpenRouterClient lazily.

    Checks the total-pipeline deadline (see this module's top-level
    "Total-pipeline deadline" docstring block) and passes the result to
    run_fast_synth, which skips the eyes-ladder call and builds
    final_output straight from ocr_output/scene_context if it has already
    passed.
    """
    if client is None:
        client = (config or {}).get("configurable", {}).get("client")
    return run_fast_synth(
        state["ocr_output"], state["scene_context"], client, deadline_exceeded=_deadline_exceeded(config)
    )


def research_node(state, config=None, searcher=None, client=None):
    """Research node (issue #10/P2.1): web-lookup for the document's subject on the deep path.

    The substantive logic (query derivation, search, bounded fetch,
    extraction, degradation) lives in clarif_eye.research.run_research so
    this stays a thin adapter, the same pattern the other nodes use.
    `searcher`/`client` are injectable directly or via
    config["configurable"]["searcher"] / ["research_client"]; when neither
    is supplied, run_research constructs real defaults lazily. Deliberately
    a DIFFERENT configurable key than "client" (used by vision_node/
    analysis_node for their OpenRouterClient) - this client is an
    httpx.Client-like page fetcher, a different type serving a different
    purpose, and reusing the same key would silently hand the wrong kind of
    client to whichever node ran second.

    Checks the total-pipeline deadline (see this module's top-level
    "Total-pipeline deadline" docstring block) and passes the result to
    run_research, which skips the search+fetch entirely if it has already
    passed.
    """
    configurable = (config or {}).get("configurable", {})
    if searcher is None:
        searcher = configurable.get("searcher")
    if client is None:
        client = configurable.get("research_client")
    return run_research(
        state["ocr_output"],
        state["scene_context"],
        searcher=searcher,
        client=client,
        deadline_exceeded=_deadline_exceeded(config),
    )


def analysis_node(state, config=None, client=None):
    """Deep-analysis node (issue #8/P1.5): turns dense-document input into spoken text.

    The substantive logic (prompt building, sanitisation, degradation)
    lives in clarif_eye.analysis.run_analysis so this stays a thin adapter,
    the same pattern vision_node/fast_synth_node already use. `client` is
    injectable directly or via config["configurable"]["client"]; when
    neither is supplied, run_analysis constructs a real OpenRouterClient
    lazily. The scraped-context cap is injectable via
    config["configurable"]["scraper_data_cap"] (issue #17 / P6.1; see
    analysis._SCRAPER_DATA_CAP for why 4000 wasn't earning its latency).

    Checks the total-pipeline deadline (see this module's top-level
    "Total-pipeline deadline" docstring block) and passes the result to
    run_analysis, which skips the brain-ladder call and builds
    final_output straight from ocr_output/scene_context if it has already
    passed.
    """
    configurable = (config or {}).get("configurable", {})
    if client is None:
        client = configurable.get("client")
    return run_analysis(
        state["ocr_output"],
        state["scene_context"],
        state["scraper_data"],
        client,
        scraper_data_cap=configurable.get("scraper_data_cap"),
        deadline_exceeded=_deadline_exceeded(config),
    )


def tts_node(state, config=None, provider=None, providers=None):
    """TTS node (issue #11/P3.1, chain added by #12/P3.2): turns final_output
    into an audio file, trying a provider chain in order.

    The substantive logic (provider chain, file lifecycle, degradation)
    lives in clarif_eye.tts.run_tts so this stays a thin adapter, the same
    pattern the other nodes use. `provider` (single) / `providers` (chain)
    are injectable directly or via config["configurable"]["tts_provider"] /
    ["tts_providers"]; when none are supplied, run_tts falls back to its
    real DEFAULT_PROVIDER_CHAIN. The output directory is injectable only
    via config["configurable"]["tts_out_dir"] (used by tests driving the
    compiled graph, so audio never lands in a fixed path or the repo); when
    absent, run_tts falls back to its own bounded per-process temp
    directory. Deliberately a DIFFERENT configurable key than "client"
    (used by vision_node/fast_synth_node/analysis_node for their
    OpenRouterClient) - a TTS provider is a different type serving a
    different purpose, and reusing the same key would silently hand the
    wrong kind of object to whichever node ran second, the same reasoning
    research_node's "research_client" key documents.

    Deliberately NEVER gated on the total-pipeline deadline (issue #17 /
    P6.1, see this module's top-level "Total-pipeline deadline" docstring
    block): tts is what turns final_output into the actual spoken
    deliverable, so skipping it on a blown deadline would turn "degraded
    but spoken" into total silence - exactly the failure this pipeline
    exists to avoid.
    """
    configurable = (config or {}).get("configurable", {})
    if providers is None:
        providers = configurable.get("tts_providers")
    if provider is None:
        provider = configurable.get("tts_provider")
    out_dir = configurable.get("tts_out_dir")
    return run_tts(state["final_output"], provider=provider, providers=providers, out_dir=out_dir)


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
