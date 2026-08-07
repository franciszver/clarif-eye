"""Compiling LangGraph skeleton for Clarif-Eye.

Wires: entry -> vision -> dynamic_router -> fast_synth OR (research ->
analysis -> [verify_numbers]) -> tts -> END, with entry -> followup -> tts
as the second way in (issue #82 / P9.3 - a typed question about a photo
this thread already described). `verify_numbers` (issue #83 / P9.4) is in
brackets because a second conditional edge decides whether it runs at all:
it is entered ONLY when `analysis` could not trace a number in its draft
back to the photographed text, and it is the one node that can PAUSE the
run to ask the user about it - see its own section below for the narrow
rule that governs when.

TWO ROUTING MECHANISMS, EACH WHERE IT FITS (issue #82 / P9.3)
---------------------------------------------------------------
Both of LangGraph's routing mechanisms are used in this graph, deliberately,
and neither is a leftover:

  - add_conditional_edges + dynamic_router (out of `vision`): a STATIC
    branch on a flag some OTHER node already computed. vision_node writes
    complexity_flag; a separate, pure function reads it and names the next
    node. The decision and the state write live in different places, which
    is exactly the shape a conditional edge expresses - the routing function
    takes no part in producing state, so it can stay a tiny testable pure
    function (see dynamic_router's own guards below) with no client, no
    config, and no update to return. Issue #83 / P9.4 added a SECOND use of
    this same mechanism, out of `analysis` (analysis_destination), for the
    same reason and in the same shape: `analysis` writes verification_hold,
    a separate pure function reads it to decide whether the run stops to
    ask the user about a number it could not check.

  - Command(goto=...) (returned by `entry`): a node that decides its OWN
    successor. There is no upstream node to compute a flag for it - `entry`
    IS the first thing that runs, and what it decides from (state["question"]
    being None or a real question) is the run's own input. Expressed as a
    conditional edge instead, `entry` would have to exist purely to do
    nothing and hand off to a router function - a node whose only job is to
    be somewhere for an edge to leave from. Command lets the decision live
    in the node that owns it. It is also the mechanism that scales to the
    case where a routing node must WRITE state as part of deciding
    (Command(goto=..., update={...})), which a conditional edge cannot do at
    all.

Every node here is a thin adapter: it returns a state update with values
computed by its own module and does not itself call a model, the network, or
the filesystem. The router stub decides purely from state["complexity_flag"]
so both paths can be driven deterministically in tests. Node functions are
named and importable individually so later issues can replace them one at a
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
from langgraph.types import Command, interrupt

from clarif_eye.analysis import run_analysis
from clarif_eye.followup import run_followup
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


def entry_destination(state):
    """The node a run should START at: "followup" when this run carries a
    typed question, "vision" when it carries a photo.

    Pulled out of entry_node so next_node_after (this module's single source
    of truth for topology, used by clarif_eye.ui's narration) can name the
    same successor entry_node will actually go to, WITHOUT re-deriving the
    rule. The same reasoning next_node_after already applies to the vision
    branch by reusing dynamic_router itself rather than re-reading
    complexity_flag.

    Truthiness, not `is not None`, is deliberate and load-bearing: a blank
    or whitespace-only question is not a question, and routing one to
    `followup` would ask a model to answer nothing. clarif_eye.ui rejects a
    blank question before the graph is ever called (see NO_QUESTION_MESSAGE
    there), so this is the second line of defence, not the only one. Reads
    with .get() because a follow-up run's input is a PARTIAL state delta -
    on a thread that has never run, keys that no node has ever written are
    genuinely absent from `state`, verified empirically on langgraph 1.2.10.

    TYPE-CHECKED, the same discipline dynamic_router applies to
    complexity_flag and for the same reason: TypedDict gives no runtime
    protection, and the two destinations differ by a whole model call and a
    whole path. Without this, a non-string question (a number, a list, a
    Gradio component someone wired up wrong) would raise a bare
    AttributeError from .strip() deep inside a node, with nothing naming the
    key or the value. Fail loudly and say what was wrong instead.
    """
    question = state.get("question")
    if question is not None and not isinstance(question, str):
        raise TypeError(
            "entry_destination: state['question'] must be str or None, got "
            f"{type(question).__name__} ({question!r})"
        )
    return "followup" if (question or "").strip() else "vision"


def entry_node(state):
    """Entry node (issue #82 / P9.3): sends the run to `vision` or `followup`
    by returning Command(goto=...).

    Writes NO state of its own and makes no call of any kind - it completes
    instantly. See this module's top-level "TWO ROUTING MECHANISMS" block
    for why this decision is a Command returned by a node rather than a
    conditional edge like the one out of `vision`.

    EMPIRICALLY VERIFIED on langgraph 1.2.10 (issue #82, probed before
    being written, not assumed):
      - builder.set_entry_point("entry") works with a Command-returning
        node; the graph starts here and honours the goto.
      - add_node("entry", entry_node) needs NO `destinations=` argument for
        the goto to resolve. Declaring it changes nothing about execution
        (it is drawing/introspection metadata), so it is left off rather
        than added as a second place to keep in sync with the real targets.
      - A node that returns a bare Command with no `update` appears in
        stream(stream_mode="updates") as {"entry": None} - the chunk VALUE
        is None, not an empty dict. Every consumer of that stream in this
        repo must tolerate it; see clarif_eye.ui._run_pipeline_events and
        tests/_stream_helpers.py, both of which skip a None update.
    """
    return Command(goto=entry_destination(state))


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


# --- Asking before speaking an unverifiable number (issue #83 / P9.4) ------
#
# THE PRODUCT RULE, stated here because it is a product decision and not a
# framework one: this graph pauses ONLY when the number verification the
# deep-analysis path already performs FAILS. Never on general low
# confidence, never on the fast path, never on a follow-up answer. Every
# user of this app is visually impaired and cannot glance at the screen to
# see what is being asked; an unnecessary question costs them more than it
# would most audiences, so the gate below is deliberately narrow.
#
# The key LangGraph fact this design is built on, EMPIRICALLY VERIFIED on
# langgraph 1.2.10 before anything was written (probes, not assumption):
# resuming an interrupted run RE-EXECUTES THE WHOLE NODE the interrupt was
# raised in, from its first line. `interrupt()` then returns the resume
# value instead of pausing again. So anything expensive standing between
# the node's entry and its interrupt() call runs TWICE. That is why the
# question is asked from this small node and not from `analysis`, which has
# just spent a brain-ladder call: a resume from inside `analysis` would buy
# a second model call (money on a rate-limited free tier), a second ~20s
# wait for someone already waiting, and - worst - a second, possibly
# DIFFERENT draft, so the user would hear an answer to a question they were
# never actually asked. `analysis` writes its draft and the failing tokens
# into state["verification_hold"] instead, and this node does nothing but
# read that key, ask, and rewrite final_output.
#
# The chunk key LangGraph emits for a pause in stream(stream_mode="updates")
# - {"__interrupt__": (Interrupt(...),)}. Named here, next to the node that
# causes it, so clarif_eye.ui matches on a shared constant instead of
# repeating the magic string (next_node_after already returns None for it).
INTERRUPT_CHUNK_KEY = "__interrupt__"

# The two answers this app sends back via Command(resume=...). Constants,
# not inline literals, for the same reason vision.py's DEGRADED_* messages
# are: the UI's buttons and the node's branch must agree, and a typo in one
# of two string literals would silently mean "retake" forever.
RESUME_CONTINUE = "continue"
RESUME_RETAKE = "retake"

# Prepended to the drafted script when the user chooses to hear it anyway.
# PREPENDED, not appended: the caveat has to arrive BEFORE the number it is
# about, or the user hears an amount stated as fact and only afterwards
# learns it could not be checked. Plain spoken language, no hedging jargon.
UNVERIFIED_NUMBER_CAVEAT = (
    "A number in this description could not be checked against the photo, "
    "so please treat it with care."
)

# Spoken when the user chooses to take a new photo instead. This is a real
# spoken outcome, not silence - it goes through tts exactly like a
# description does, because a blind user who pressed a button and heard
# nothing back has no way to know whether it worked.
RETAKE_CONFIRMATION = (
    "All right, nothing was read out. Please take or upload a new photo, "
    'then activate "Describe this photo".'
)


def numbers_need_asking(state):
    """True when `analysis` held a drafted script back because a number in
    it could not be traced to the photographed text.

    THE GATE the product rule above lives in, and the SINGLE place it is
    expressed. Both the conditional edge out of `analysis`
    (analysis_destination, below) and verify_numbers_node's own guard read
    it, so "when does this app stop and ask?" can never be answered two
    different ways by two pieces of code.

    Reads with .get() for the same reason followup_node does: a run's input
    can be a partial state delta, so a key no node has written yet is
    genuinely ABSENT rather than None.
    """
    return bool(state.get("verification_hold"))


def analysis_destination(state):
    """The node that runs after `analysis`: "verify_numbers" when there is
    something to ask the user about, "tts" when there is not.

    A CONDITIONAL EDGE, not a straight line through the asking node, and
    that is a deliberate application of this module's own "TWO ROUTING
    MECHANISMS" rule (see the top of this file): this is a static branch on
    a flag some OTHER node already computed - `analysis` writes
    verification_hold, a separate pure function reads it - which is exactly
    the shape a conditional edge expresses, the same shape dynamic_router
    already has out of `vision`.

    IT IS ALSO WHAT KEEPS THE COMMON PATH FREE. Every completed node is a
    full checkpoint write, and each one stores the WHOLE state including
    image_data's base64 JPEG - measured at ~134KB per invoke with a 50KB
    photo (see clarif_eye.ui._trim_thread_to_latest_checkpoint's comment).
    Routing every deep-analysis run through an extra node just to have it
    look at an almost-always-empty dict would buy that write on every
    request, on a 512MB instance, for nothing. Branching here means the
    asking node is entered ONLY on the runs that actually ask.
    """
    return "verify_numbers" if numbers_need_asking(state) else "tts"


def verify_numbers_node(state):
    """Ask the user before speaking a number that could not be checked
    (issue #83 / P9.4) - or pass straight through when there is nothing to
    ask about.

    Reads ONE key, state["verification_hold"], which `analysis` wrote (see
    clarif_eye.analysis.run_analysis and state.py's own comment on that
    key), via the shared numbers_need_asking predicate above. In the graph
    this node is only ENTERED when that predicate is already true
    (analysis_destination routes past it otherwise), so the guard below is
    the second line of defence, not the gate: it keeps a direct call - a
    unit test, a future caller - from raising on a state that holds
    nothing, which this pipeline must never do.
    - RESUME_CONTINUE: the held draft IS spoken, with UNVERIFIED_NUMBER_CAVEAT
      in front of it. The user asked to hear it; hiding it now would be a
      second, quieter refusal.
    - ANYTHING ELSE (including RESUME_RETAKE): the retake confirmation is
      spoken and the draft is discarded. Defaulting the unrecognised case to
      "do not speak it" is deliberate - only an answer this app actually
      sent may be read as consent to speak an unverified number.
    Either way `verification_hold` is cleared, so the thread is left ready
    for the next photo with nothing pending.

    MAKES NO CALL OF ANY KIND - no model, no network, no filesystem - which
    is exactly what makes it safe to re-execute on resume. See this
    section's comment block above for the empirical basis.

    Takes no `config`: it has nothing to read from one, and the pipeline
    deadline deliberately does not apply here - a run that has stopped to
    ask a human a question is already outside any latency budget, and
    "degrade because the budget expired while the user was deciding" would
    throw away the answer they just gave.

    HONEST COST OF PAUSING, stated because it is not free: while a run
    waits here, its FULL checkpointed state stays live in this process's
    memory - including image_data's base64 JPEG, and the drafted script
    again in verification_hold - for as long as the user takes to answer.
    That is inherent (there is nothing to resume otherwise), and it is
    bounded by the same mechanisms every other thread is: MAX_LIVE_THREADS
    evicts the least-recently-touched thread, and a process restart clears
    everything. It does mean a paused thread holds more, for longer, than a
    completed one - which is trimmed to its newest checkpoint the moment it
    finishes.
    """
    if not numbers_need_asking(state):
        return {}
    hold = state["verification_hold"]

    answer = interrupt(
        {
            "reason": "unverified_numbers",
            "script": hold.get("script", ""),
            "numbers": list(hold.get("numbers") or []),
        }
    )

    if answer == RESUME_CONTINUE:
        return {
            "final_output": f"{UNVERIFIED_NUMBER_CAVEAT} {hold.get('script', '')}".strip(),
            "verification_hold": None,
        }
    return {"final_output": RETAKE_CONFIRMATION, "verification_hold": None}


def followup_node(state, config=None, client=None):
    """Follow-up node (issue #82 / P9.3): answers state["question"] from the
    ocr_output/scene_context this THREAD already has checkpointed.

    The substantive logic (prompt building, the no-photo-yet case,
    sanitisation, degradation) lives in clarif_eye.followup.run_followup so
    this stays a thin adapter, the same pattern every other node here uses.
    `client` is injectable directly or via config["configurable"]["client"] -
    the SAME key vision_node/analysis_node use, because it is the same kind
    of object (an OpenRouterClient); only role "brain" differs.

    Reads state with .get(): a follow-up run's input is a PARTIAL state
    delta (see clarif_eye.state.ClarifEyeState.question), so on a thread
    that has never described a photo, ocr_output/scene_context are genuinely
    ABSENT from `state` rather than empty strings - run_followup normalises
    both cases to the same no-photo-yet answer.

    Checks the total-pipeline deadline exactly like the other model-calling
    nodes (see this module's top-level "Total-pipeline deadline" block) and
    passes the result through, so a run that has already blown its budget
    reads back what is known instead of spending a brain call.
    """
    if client is None:
        client = (config or {}).get("configurable", {}).get("client")
    return run_followup(
        state.get("ocr_output"),
        state.get("scene_context"),
        state.get("question"),
        client,
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

    Does NOT append to `messages` (issue #81 / P9.2's reducer, see
    state.py) - an earlier version of this node did, but that put a
    conversation-boundary concern (recording one turn per completed run)
    inside a node whose job is turning text into audio. #82 (follow-ups)
    and #84 (subgraph extraction) would then each have had to duplicate or
    detour around that append. The turn is now recorded once, at the
    boundary, by clarif_eye.ui._run_pipeline_events via
    graph.update_state() after a run completes with a real thread_id - see
    that function's docstring.
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


def build_graph(checkpointer=None):
    """Build and compile the Clarif-Eye graph.

    `checkpointer` (issue #81 / P9.2) is OPTIONAL and defaults to None -
    every existing caller/test that calls build_graph() with no argument
    keeps compiling an uncheckpointed graph, exactly today's behavior, and
    can go on invoking it with no `thread_id` at all. When a checkpointer
    IS supplied (clarif_eye.ui.build_resources passes a fresh
    langgraph.checkpoint.memory.InMemorySaver - see that function's own
    comment for its honest limits), LangGraph then REQUIRES
    config["configurable"]["thread_id"] on every invoke()/stream() call
    against the compiled graph (verified empirically: omitting it raises
    ValueError) - clarif_eye.ui is responsible for minting one per browser
    session and passing it through.
    """
    builder = StateGraph(ClarifEyeState)

    builder.add_node("entry", entry_node)
    builder.add_node("vision", vision_node)
    builder.add_node("fast_synth", fast_synth_node)
    builder.add_node("research", research_node)
    builder.add_node("analysis", analysis_node)
    # issue #83 / P9.4: sits between analysis and tts so the question is
    # asked BEFORE anything is spoken, and so a resume never re-runs the
    # brain call analysis just made - see verify_numbers_node's docstring.
    builder.add_node("verify_numbers", verify_numbers_node)
    builder.add_node("followup", followup_node)
    builder.add_node("tts", tts_node)

    # entry has NO outgoing edge of any kind: it returns Command(goto=...)
    # and routes itself. See this module's "TWO ROUTING MECHANISMS" block
    # for why this one is a Command while vision's is a conditional edge,
    # and entry_node's docstring for the empirically-verified fact that no
    # `destinations=` declaration is needed on add_node for the goto to
    # resolve on langgraph 1.2.10.
    builder.set_entry_point("entry")
    # A STATIC branch on a flag vision_node already computed - the shape a
    # conditional edge fits (see "TWO ROUTING MECHANISMS" above).
    builder.add_conditional_edges(
        "vision",
        dynamic_router,
        {"fast_synth": "fast_synth", "research": "research"},
    )
    builder.add_edge("fast_synth", "tts")
    builder.add_edge("research", "analysis")
    # Only the deep-analysis path verifies numbers at all (see
    # clarif_eye.verification's module docstring), so only this path can
    # route through the asking node - and only on the runs that have
    # something to ask about. A conditional edge, for the reasons
    # analysis_destination's docstring gives (the same "branch on a flag
    # another node computed" shape as dynamic_router, and it keeps an extra
    # checkpoint write off every clean run). fast_synth and followup go
    # straight to tts, unchanged.
    builder.add_conditional_edges(
        "analysis",
        analysis_destination,
        {"verify_numbers": "verify_numbers", "tts": "tts"},
    )
    builder.add_edge("verify_numbers", "tts")
    # A follow-up answer is spoken exactly like a description is: same tts
    # node, same provider chain, same staged delivery in the UI.
    builder.add_edge("followup", "tts")
    builder.add_edge("tts", END)

    return builder.compile(checkpointer=checkpointer)


# Unconditional successor for every edge above EXCEPT the ones out of
# "entry" (self-routed by Command, resolved by entry_destination instead),
# "vision" and "analysis" (conditional edges, resolved by dynamic_router
# and analysis_destination instead) and "tts" (the last node - see
# next_node_after's END case below). Kept
# literally next to build_graph()'s add_edge calls, which is the ONLY
# reason this is safe to hand-maintain rather than deriving it from the
# StateGraph itself: whoever changes an edge above must update this table
# in the same diff, or next_node_after silently starts lying to callers
# like clarif_eye.ui's per-node progress narration (issue #80 / P9.1).
_UNCONDITIONAL_SUCCESSOR = {
    "fast_synth": "tts",
    "research": "analysis",
    "verify_numbers": "tts",
    "followup": "tts",
}


def next_node_after(node_name, state):
    """The node that runs immediately after `node_name`, given `state` as
    it stands once `node_name` has completed - or None if `node_name` is
    the last node (tts) and nothing follows.

    Single source of truth for this graph's topology, used by
    clarif_eye.ui's per-node progress narration (issue #80 / P9.1) so it
    never has to re-derive routing itself. Each branch reuses the SAME
    function the graph itself routes with, rather than re-deriving the rule
    with separate logic that could drift: the vision branch reuses
    dynamic_router (the function build_graph() wires as the conditional
    edge), and the entry branch reuses entry_destination (the function
    entry_node builds its Command(goto=...) from).

    RETURNS None FOR ANY NAME THIS GRAPH DOES NOT KNOW, rather than raising
    a KeyError. The caller is clarif_eye.ui's narration, which iterates over
    whatever keys LangGraph's stream produces - and those are not all node
    names. LangGraph emits RESERVED keys too: INTERRUPT_CHUNK_KEY
    ("__interrupt__") is the concrete one, emitted whenever verify_numbers
    pauses to ask the user about an unverifiable number (issue #83 / P9.4),
    and a KeyError raised from shared narration code would take down a run
    that was otherwise fine. Callers that need to ACT on a pause match that
    constant themselves before reaching here (see
    clarif_eye.ui._narrate_stream); this function only has to not crash on
    it. A typo'd or renamed node name lands here as well and
    degrades to "announce nothing for this step" - a quiet narration gap,
    which is the right failure for a progress announcement, rather than
    losing the user an answer that had already been computed.
    """
    if node_name == "entry":
        return entry_destination(state)
    if node_name == "vision":
        return dynamic_router(state)
    if node_name == "analysis":
        return analysis_destination(state)
    if node_name == "tts":
        return None
    return _UNCONDITIONAL_SUCCESSOR.get(node_name)
