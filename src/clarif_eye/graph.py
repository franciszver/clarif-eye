"""Compiling LangGraph skeleton for Clarif-Eye.

Wires: entry -> vision -> dynamic_router -> fast_synth OR deep_path -> tts
-> END, with entry -> followup -> [verify_answer] -> tts as the second way
in (issue #82 / P9.3 - a typed question about a photo this thread already
described). `verify_answer` is in brackets for the same reason
`verify_numbers` is below: a conditional edge decides whether it runs at
all, and it runs only when a number in the drafted ANSWER could not be
traced back to the photographed text (issue #92 / P9.11).

`deep_path` (issue #84 / P9.5) is ONE NODE HERE AND A WHOLE GRAPH INSIDE:
research -> analysis -> [verify_numbers], compiled separately with its own
schema and mounted through a mapping wrapper. That child lives in
clarif_eye.deep_path, which carries the reasoning for the split, the key
renaming, and the empirically-verified nesting behaviour (config flowing
in, child node events streaming out, an interrupt firing inside the child
and reaching the top-level caller). The node FUNCTIONS the child runs -
research_node, analysis_node, verify_numbers_node - are still defined here,
next to the parent's own, because they are the same thin adapters they
always were. `verify_numbers` is in brackets because a conditional edge
decides whether it runs at all: it is entered ONLY when `analysis` could
not trace a number in its draft back to the photographed text, and it is
the one node that can PAUSE the run to ask the user about it - see its own
section below for the narrow rule that governs when.

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
    ask the user about a number it could not check. THAT SECOND EDGE IS NO
    LONGER DECLARED IN THIS GRAPH - issue #84 / P9.5 moved it, together
    with the two nodes it joins, into the deep-path child graph (see
    clarif_eye.deep_path.build_deep_path_graph, which wires it). The
    mechanism and the reasoning are unchanged; only which graph declares
    the edge moved. analysis_destination itself still lives in this
    module, next to the node functions it routes between. Issue #92 /
    P9.11 added a THIRD use, out of `followup` (followup_destination), in
    exactly the same shape and for exactly the same reason - and that one
    IS declared in this graph, because both nodes it joins are this
    graph's own.

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
`graph.stream(state, config=config, stream_mode="updates", subgraphs=True)`
(issue #80 / P9.1, extended by #84 / P9.5) - one dict per COMPLETED node,
keyed by node name, paired with the checkpoint namespace it came from, so
the deep-path child's nodes are visible too - rather than a
caller-supplied trace list threaded through config["configurable"]. That
gives callers (and tests) real per-node progress instead of a debugging
side-channel, and needs nothing extra recorded by the nodes themselves.
"""

import time

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from clarif_eye.analysis import run_analysis
from clarif_eye.deep_path import build_deep_path_graph
from clarif_eye.followup import run_followup
from clarif_eye.preferences import verbosity_for_config
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


def fast_synth_node(state, config=None, client=None, store=None):
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

    `store` (issue #86 / P9.7): injectable directly (unit tests), or
    populated automatically by LangGraph when the compiled graph was built
    with build_graph(store=...) - EMPIRICALLY VERIFIED on langgraph 1.2.10,
    unlike `client`/`config`, a node parameter literally named `store`
    needs no config["configurable"] fallback at all; LangGraph injects it
    itself. Read into the cross-thread verbosity preference via
    clarif_eye.preferences.verbosity_for_config, which never raises and
    degrades to None (no preference on file) for a None store, a missing
    session_id, or anything else it does not recognise - see that
    function's own docstring.
    """
    if client is None:
        client = (config or {}).get("configurable", {}).get("client")
    verbosity = verbosity_for_config(store, config)
    return run_fast_synth(
        state["ocr_output"],
        state["scene_context"],
        client,
        deadline_exceeded=_deadline_exceeded(config),
        verbosity=verbosity,
    )


def research_node(state, config=None, searcher=None, client=None):
    """Research node (issue #10/P2.1): web-lookup for the document's subject on the deep path.

    A NODE OF THE CHILD GRAPH since issue #84 / P9.5, so it reads the DEEP
    PATH's own key names (`document_text`/`scene_description`, see
    clarif_eye.deep_path.DeepPathState) rather than the parent's
    `ocr_output`/`scene_context`. The function body is otherwise untouched:
    the values are the same values, mapped once by deep_path_node below.

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
        state["document_text"],
        state["scene_description"],
        searcher=searcher,
        client=client,
        deadline_exceeded=_deadline_exceeded(config),
    )


def analysis_node(state, config=None, client=None, store=None):
    """Deep-analysis node (issue #8/P1.5): turns dense-document input into spoken text.

    A NODE OF THE CHILD GRAPH since issue #84 / P9.5 - see research_node's
    docstring for the key renaming that applies here identically.

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

    `store` (issue #86 / P9.7): same injection mechanics as
    fast_synth_node's own `store` param (see its docstring) - EMPIRICALLY
    VERIFIED to reach THIS node too, even though it runs inside the
    deep-path CHILD graph (clarif_eye.deep_path), which is itself compiled
    with no store of its own: LangGraph's store propagates into a subgraph
    invoked manually from a node (make_deep_path_node's deep_path_node,
    child.invoke(...)) the same way config already does - see that
    function's own "NO config PASSED TO THE CHILD" docstring block for the
    config half of this fact; the store half was probed identically for
    this issue. Read via clarif_eye.preferences.verbosity_for_config.
    """
    configurable = (config or {}).get("configurable", {})
    if client is None:
        client = configurable.get("client")
    verbosity = verbosity_for_config(store, config)
    return run_analysis(
        state["document_text"],
        state["scene_description"],
        state["scraper_data"],
        client,
        scraper_data_cap=configurable.get("scraper_data_cap"),
        deadline_exceeded=_deadline_exceeded(config),
        verbosity=verbosity,
    )


# --- Asking before speaking an unverifiable number (issue #83 / P9.4) ------
#
# THE PRODUCT RULE, stated here because it is a product decision and not a
# framework one: this graph pauses ONLY when a NUMBER VERIFICATION FAILS.
# Never on general low confidence, never on the fast path. Every user of
# this app is visually impaired and cannot glance at the screen to see what
# is being asked; an unnecessary question costs them more than it would
# most audiences, so the gate below is deliberately narrow.
#
# WHICH PATHS VERIFY NUMBERS AT ALL grew by one in issue #92 / P9.11, and
# this comment used to say "never on a follow-up answer". It now says never
# on a follow-up answer whose numbers all trace back. The deep-analysis path
# checks the description it drafts; the FOLLOW-UP path checks the answer it
# drafts (clarif_eye.followup, which carries the reversal's full reasoning).
# The fast path still has no check and so can still never pause. The rule
# itself did not change - only how many drafts are held to it.
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

# The same choice, answered on the FOLLOW-UP path (issue #92 / P9.11) - and
# a DIFFERENT sentence, deliberately.
#
# THE BUTTONS AND THE PAYLOAD ARE UNCHANGED: two choices, the same labels,
# the same interrupt fields, so every part of clarif_eye.ui's pause flow
# works for either path with no idea which one asked. What differs is what
# is TRUE afterwards. On the photo path the drafted DESCRIPTION could not be
# verified, so there is nothing more to do with that photo and "take a new
# one" is the honest next step. On a follow-up the photo is fine - the user
# photographed it correctly and it is still on the thread - and it is the
# ANSWER that could not be checked. Sending them off to re-photograph
# something that was never the problem would be work for no reason, so this
# names the two things that will actually help: ask again, or photograph the
# part the question is about if it was not in frame.
#
# The retake button's own label ("I'll retake the photo") still reads as
# "never mind" here, which is why no third button was added - see
# clarif_eye.ui.RESUME_RETAKE_LABEL.
ANSWER_RETAKE_CONFIRMATION = (
    "All right, that answer was not read out. The photo you already sent is "
    "still here, so you can ask your question again. If the part you are "
    "asking about was not in the photo, please take a new photo of it."
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
    something to ask the user about, the END OF THE CHILD GRAPH when there
    is not.

    IT RETURNED "tts" UNTIL ISSUE #84 / P9.5, when `analysis` moved into the
    deep-path child graph (clarif_eye.deep_path). `tts` is the parent's node
    and the child cannot name it; the parent's own edge out of "deep_path"
    leads there instead, so a clean run still ends in speech and the user
    hears exactly the same thing. next_node_after translates this END back
    to None for the narration, which is what keeps "Turning it into speech"
    from being announced twice - once by the child finishing and again by
    the parent's deep_path chunk arriving.

    A CONDITIONAL EDGE, not a straight line through the asking node, and
    that is a deliberate application of this module's own "TWO ROUTING
    MECHANISMS" rule (see the top of this file): this is a static branch on
    a flag some OTHER node already computed - `analysis` writes
    verification_hold, a separate pure function reads it - which is exactly
    the shape a conditional edge expresses, the same shape dynamic_router
    already has out of `vision`.

    IT IS ALSO WHAT KEEPS THE COMMON PATH FREE, though the saving is
    SMALLER THAN THIS COMMENT USED TO CLAIM and the correction is worth
    stating rather than quietly editing out. It cited ~134KB per invoke,
    the measured cost of a checkpoint carrying image_data's base64 JPEG
    (see clarif_eye.ui._trim_thread_to_latest_checkpoint). That figure
    belongs to the PARENT graph. This edge is inside the deep-path child
    now (issue #84 / P9.5), whose schema has no image_data in it at all -
    so what routing every run through the asking node would actually cost
    is one more checkpoint write of the child's small text-only state, on
    every deep-path request, on a 512MB instance. Still not worth paying to
    have a node look at an almost-always-empty dict, and the branch also
    keeps the node that can PAUSE a run off every path that has nothing to
    ask about - which is the reason that does not depend on a number.
    """
    return "verify_numbers" if numbers_need_asking(state) else END


def followup_destination(state):
    """The node that runs after `followup` (issue #92 / P9.11):
    VERIFY_ANSWER_NODE when the answer holds a number that could not be
    traced to the photographed text, `tts` when it does not.

    THE FOLLOW-UP PATH'S HALF of the same gate analysis_destination is the
    deep path's half of - same predicate (numbers_need_asking), same shape,
    same reason it is a conditional edge rather than a straight line through
    the asking node (see this module's "TWO ROUTING MECHANISMS" block and
    analysis_destination's own docstring): `followup` writes
    verification_hold, a separate pure function reads it.

    IT GOES TO `tts`, NOT END, and that is the one structural difference from
    analysis_destination: this edge is declared in the PARENT graph, which
    owns `tts` and can name it. The deep path's equivalent runs in a child
    that cannot.

    Reads with .get() for the same reason numbers_need_asking does: a
    follow-up run's input is a partial state delta, so a key no node has
    written yet is genuinely ABSENT rather than None.
    """
    return VERIFY_ANSWER_NODE if numbers_need_asking(state) else TTS_NODE


def _ask_about_held_numbers(state, reason, retake_confirmation):
    """Ask the user before speaking a number that could not be checked
    (issue #83 / P9.4) - or pass straight through when there is nothing to
    ask about.

    THE WHOLE BODY OF BOTH ASKING NODES, in one place (issue #92 / P9.11).
    verify_numbers_node (the deep path's) and verify_answer_node (the
    follow-up path's) differ by exactly two values - the `reason` they stamp
    on the interrupt payload, and the sentence they speak when the user
    declines - and by nothing else at all. Two copies of an interrupt-raising
    node would be two places for the payload shape to drift, and the payload
    shape is what clarif_eye.ui's entire pause flow is built on.

    WHY TWO REGISTERED NODES RATHER THAN ONE SHARED REGISTRATION: the two
    graphs' node names must stay DISJOINT. clarif_eye.graph's
    _UNCONDITIONAL_SUCCESSOR and clarif_eye.ui's _NODE_PHRASE are flat dicts
    keyed by BARE node names that answer for the parent AND the deep-path
    child (see tests/test_deep_path_subgraph.py's disjointness test), so a
    parent node also called "verify_numbers" would make one graph's entry
    answer for the other graph's node - silently, since a dict lookup cannot
    notice. Registering the parent's under its own name (VERIFY_ANSWER_NODE)
    keeps that property while the behaviour stays shared here.

    Reads ONE key, state["verification_hold"], which `analysis` or
    `followup` wrote (see clarif_eye.analysis.run_analysis,
    clarif_eye.followup.run_followup, and state.py's own comment on that
    key), via the shared numbers_need_asking predicate above. In both graphs
    these nodes are only ENTERED when that predicate is already true
    (analysis_destination / followup_destination route past them otherwise),
    so the guard below is the second line of defence, not the gate: it keeps
    a direct call - a unit test, a future caller - from raising on a state
    that holds nothing, which this pipeline must never do.
    - RESUME_CONTINUE: the held draft IS spoken, with UNVERIFIED_NUMBER_CAVEAT
      in front of it. The user asked to hear it; hiding it now would be a
      second, quieter refusal. THE SAME CAVEAT on both paths, on purpose:
      what it warns about ("a number in this could not be checked against the
      photo") is identical whichever draft is being spoken.
    - ANYTHING ELSE (including RESUME_RETAKE): `retake_confirmation` is
      spoken and the draft is discarded. Defaulting the unrecognised case to
      "do not speak it" is deliberate - only an answer this app actually
      sent may be read as consent to speak an unverified number.
    Either way `verification_hold` is cleared, so the thread is left ready
    for the next photo or question with nothing pending.

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

    # THE PAYLOAD SHAPE IS FIXED - three fields, always the same three - so
    # clarif_eye.ui._interrupt_question, the resume buttons, the refusals and
    # the staging all work identically whichever node raised it. `reason` is
    # the ONE field that says which flow asked; it is a value, not an extra
    # key, so the shape does not widen. Nothing in the UI branches on it
    # today (the spoken question is built from `script` and `numbers`); it is
    # carried because a structural signal beats re-deriving the flow later
    # from the wording of a sentence.
    answer = interrupt(
        {
            "reason": reason,
            "script": hold.get("script", ""),
            "numbers": list(hold.get("numbers") or []),
        }
    )

    if answer == RESUME_CONTINUE:
        # output_degraded=False (issue #93 / P9.12): the caveated script IS
        # this photo's answer - the user asked to hear it and it is what
        # they heard - so the thread should remember it. This OVERWRITES the
        # True `analysis` set alongside its refusal (see that module's
        # _degraded), which is exactly the point of writing the flag on
        # every path: the last node to speak owns it.
        return {
            "final_output": f"{UNVERIFIED_NUMBER_CAVEAT} {hold.get('script', '')}".strip(),
            "verification_hold": None,
            "output_degraded": False,
        }
    # A retake confirmation is not an answer about the photo - it is the
    # user declining one. clarif_eye.ui._run_resume_events already declines
    # to record anything but a RESUME_CONTINUE, so this flag changes no
    # behaviour today; it is set because leaving the key saying "this is a
    # real description" while final_output says "please take another photo"
    # would be a lie waiting for the next reader of this state.
    return {
        "final_output": retake_confirmation,
        "verification_hold": None,
        "output_degraded": True,
    }


def verify_numbers_node(state):
    """The deep path's asking node - a node of the CHILD graph (see
    clarif_eye.deep_path.build_deep_path_graph, which wires it). Asks about a
    number in a drafted DESCRIPTION; declining means the photo itself is the
    thing to retake. All behaviour is in _ask_about_held_numbers above."""
    return _ask_about_held_numbers(state, "unverified_numbers", RETAKE_CONFIRMATION)


def verify_answer_node(state):
    """The follow-up path's asking node (issue #92 / P9.11) - a node of the
    PARENT graph, registered as VERIFY_ANSWER_NODE. Asks about a number in a
    drafted ANSWER; declining leaves the photo where it is, so it speaks
    ANSWER_RETAKE_CONFIRMATION instead. All behaviour is in
    _ask_about_held_numbers above."""
    return _ask_about_held_numbers(state, "unverified_answer", ANSWER_RETAKE_CONFIRMATION)


# The name the deep path's child graph is mounted under in the parent. A
# CONSTANT for the same reason TTS_NODE is: the string does not stay inside
# this module. clarif_eye.ui maps it to a spoken phrase (_NODE_PHRASE), and
# LangGraph builds every one of the child's checkpoint namespaces out of it
# ("deep_path:<task id>" - see clarif_eye.ui._trim_thread_to_latest_checkpoint,
# which has to recognise those namespaces to bound them).
DEEP_PATH_NODE = "deep_path"

# The name the follow-up path's asking node is registered under (issue #92 /
# P9.11). A CONSTANT for the same reasons TTS_NODE and DEEP_PATH_NODE are:
# the string leaves this module (clarif_eye.ui's _NODE_PHRASE, and the
# conditional-edge mapping below), and it must never collide with the child
# graph's "verify_numbers" - see _ask_about_held_numbers's docstring for what
# a collision would silently break, and tests/test_deep_path_subgraph.py's
# disjointness test, which reads both compiled graphs' node sets.
#
# NAMED FOR WHAT IT ASKS ABOUT, not for what it does: this node verifies an
# ANSWER to a typed question, the child's verifies the NUMBERS in a drafted
# description. Both raise the same interrupt from the same function body.
VERIFY_ANSWER_NODE = "verify_answer"


def make_deep_path_node(child):
    """Build the parent's `deep_path` node: the MAPPING WRAPPER that lets a
    child graph with its own schema run inside this one (issue #84 / P9.5).

    THIS FUNCTION IS THE OWN-SCHEMA MODE. A child whose keys are a subset of
    the parent's, with the same names, needs nothing at all - `add_node(name,
    child)` maps the channels itself (verified on langgraph 1.2.10). This
    child deliberately renames its two inputs, so something has to translate,
    and this is it: parent state in the parent's words on the way down,
    child result in the parent's words on the way back up. See
    clarif_eye.deep_path's module docstring for why the rename is there.

    A CLOSURE OVER AN ALREADY-COMPILED CHILD, not a lazy build inside the
    node: compiling on every request would be work done per photo for a
    result that never differs, and a module-level singleton would compile at
    import time, before this module has finished defining the node functions
    the child graph is built from.

    NO `config` PARAMETER, AND NO CONFIG PASSED TO THE CHILD. LangGraph
    propagates the running config into a subgraph invoked from a node on its
    own - EMPIRICALLY VERIFIED on 1.2.10: the child's nodes see the parent's
    config["configurable"] (client, searcher, research_client,
    scraper_data_cap, deadline, thread_id) untouched, which is what keeps the
    whole-pipeline deadline working across the boundary. Passing the parent's
    config down BY HAND would also hand over the parent's checkpoint
    namespace, which is exactly what must NOT happen: the child gets its own.

    RE-EXECUTED ON RESUME, and cheap by design. When the child pauses to ask
    about an unverifiable number, LangGraph re-runs this whole node from its
    first line once the answer arrives - so it does nothing but read two keys
    and call the child, which itself resumes at its paused node rather than
    from the start. The brain model call is not paid twice; see
    clarif_eye.deep_path's docstring for the probes behind that.
    """

    def deep_path_node(state):
        result = child.invoke(
            {
                "document_text": state["ocr_output"],
                "scene_description": state["scene_context"],
            }
        )
        # MAPPED BACK OUT, all four of them, so the parent's checkpoint says
        # what it said before this extraction. They are the child's OUTPUTS
        # being translated, not shared channels: the child owns producing
        # them, and the parent never writes into them.
        #
        # HOW MUCH EACH ONE ACTUALLY DOES, measured by deleting it rather
        # than asserted from intent - because they are not equally
        # load-bearing and pretending otherwise would mislead the next
        # editor:
        #   - final_output: the deliverable. Everything depends on it.
        #   - output_degraded (issue #93 / P9.12): NOT observable through any
        #     real flow today, exactly like verification_hold below, and
        #     measured the same way rather than assumed - deleting this line
        #     leaves the whole suite green. The reason is worth knowing: the
        #     consumer of this flag (clarif_eye.ui._record_turn) reads it off
        #     the run's STREAMED updates, and clarif_eye.ui._narrate_stream
        #     streams with subgraphs=True, so `analysis`'s own chunk carries
        #     the flag to the boundary whether or not this wrapper maps it.
        #     It is kept because the parent's CHECKPOINT would otherwise say
        #     output_degraded=False on a thread whose last spoken output was
        #     a failure message - a stored lie waiting for the first
        #     consumer to read this key from state rather than from a live
        #     run. What IS load-bearing and IS pinned is the CHILD-side
        #     declaration (clarif_eye.deep_path.DeepPathState): undeclare it
        #     and the bracket read below raises KeyError, which
        #     tests/test_degraded_turns.py's deep-path test catches.
        #   - scraper_data: genuinely observable. Without it the parent stays
        #     at make_initial_state's None ("research never ran") after a run
        #     in which research demonstrably did. Pinned by
        #     tests/test_deep_path_subgraph.py's write-back test, which reds
        #     when this line is removed.
        #   - verification_hold: NOT observable through any real flow, and
        #     removing it leaves the whole suite green. Every photo run seeds
        #     the parent's copy None (make_initial_state), the child clears
        #     the hold before this wrapper returns, and on a PAUSE this
        #     wrapper never returns at all - so the parent's value is None
        #     with or without this line. It is kept for symmetry (all three
        #     of the child's outputs cross the boundary the same way, so a
        #     future change to when the hold is cleared cannot silently strand
        #     the parent) and NOT because anything today would notice. THE
        #     ONE PLACE THE PARENT'S STATE GENUINELY CHANGED with this
        #     extraction is a PAUSED thread: `analysis` used to write the
        #     hold straight into the parent, so get_state() showed it while
        #     the question was pending; now it lives in the child's
        #     checkpoint and the parent shows None. Nothing in this app reads
        #     it there (clarif_eye.ui only ever writes it back to None), so
        #     no behaviour changed - but it is a real difference and is
        #     recorded here rather than left to be rediscovered.
        #
        # BRACKET ACCESS, NOT .get(), ON PURPOSE - LOUD BEATS SILENT. All four
        # keys are written by the child on every path it can finish through
        # (`research` writes scraper_data in every branch including its
        # degradations; `analysis` writes verification_hold AND
        # output_degraded in every return, None/True-or-False as the case
        # may be; `verify_numbers` rewrites all three of those), so
        # a missing one means the child's contract changed and this wrapper
        # was not updated with it. A KeyError says that immediately; a .get()
        # would quietly write None over a real value and nobody would find out
        # until a user was told the wrong thing (or, for output_degraded,
        # until a failure message was replayed as a description). It cannot
        # reach Gradio as a traceback either - clarif_eye.ui._run_pipeline_events
        # catches it and speaks a message, per this pipeline's never-raise
        # contract.
        #
        # THE OTHER HALF OF THIS BOUNDARY CANNOT BE CHECKED FROM HERE, and
        # that is why a test pins it instead. LangGraph SILENTLY DROPS an
        # update key that the target schema does not declare - verified: a
        # node returning {"not_in_schema": ...} raises nothing, and the key
        # does not even appear in the stream chunk. So renaming any of these
        # three in ClarifEyeState would make this write-back quietly stop
        # updating that channel, with no error anywhere. See
        # tests/test_deep_path_subgraph.py's boundary-overlap test, which
        # names this function as the dependent.
        return {
            "final_output": result["final_output"],
            "scraper_data": result["scraper_data"],
            "verification_hold": result["verification_hold"],
            "output_degraded": result["output_degraded"],
        }

    return deep_path_node


def followup_node(state, config=None, client=None, store=None):
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

    `store` (issue #86 / P9.7): same injection mechanics as
    fast_synth_node's own `store` param (see its docstring). Only a genuine
    QUESTION ever reaches this node - a preference-SETTING command is
    recognised and answered by clarif_eye.ui before the graph is invoked at
    all (see clarif_eye.preferences.detect_preference_command) - so
    verbosity here only ever changes HOW an answer is phrased, never
    whether this node runs.
    """
    if client is None:
        client = (config or {}).get("configurable", {}).get("client")
    verbosity = verbosity_for_config(store, config)
    return run_followup(
        state.get("ocr_output"),
        state.get("scene_context"),
        state.get("question"),
        client,
        deadline_exceeded=_deadline_exceeded(config),
        verbosity=verbosity,
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
    """Route on state["complexity_flag"]: True -> deep_path, False -> fast_synth.

    IT NAMED "research" UNTIL ISSUE #84 / P9.5. research is now the first
    node of the deep path's own child graph (clarif_eye.deep_path), which the
    parent sees as the single node DEEP_PATH_NODE - so that is what this
    router names. Nothing about the decision itself changed.

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
    return DEEP_PATH_NODE if flag else "fast_synth"


def build_graph(checkpointer=None, store=None):
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

    `store` (issue #86 / P9.7) is OPTIONAL and defaults to None, the SAME
    "optional at compile" shape as `checkpointer` above - every existing
    caller/test keeps compiling a store-less graph unaffected. When a store
    IS supplied (clarif_eye.ui.build_resources passes a fresh
    langgraph.store.memory.InMemoryStore - see that function's own comment
    for its honest limits, the same in-process-only ones the checkpointer
    already has), fast_synth_node/analysis_node/followup_node - the nodes
    that build a spoken prompt - receive it automatically as their own
    `store` parameter (EMPIRICALLY VERIFIED on langgraph 1.2.10, see those
    nodes' docstrings) and read a cross-thread verbosity preference from it
    via clarif_eye.preferences. UNLIKE `checkpointer`, LangGraph does NOT
    require a `thread_id` (or anything else) just because a store is
    configured - a store-less run and a run with no session_id both simply
    see no preference (verbosity_for_config degrades to None), never an
    error.
    """
    builder = StateGraph(ClarifEyeState)

    builder.add_node("entry", entry_node)
    builder.add_node("vision", vision_node)
    builder.add_node("fast_synth", fast_synth_node)
    # issue #84 / P9.5: research -> analysis -> [verify_numbers] are no
    # longer this graph's own nodes. They are a compiled CHILD GRAPH with its
    # own schema (clarif_eye.deep_path), mounted here as one node through the
    # mapping wrapper make_deep_path_node builds. A fresh child per compiled
    # parent, so two graphs built in one process (the app's, and a test's)
    # never share one.
    builder.add_node(DEEP_PATH_NODE, make_deep_path_node(build_deep_path_graph()))
    builder.add_node("followup", followup_node)
    # issue #92 / P9.11: the follow-up path's own asking node. Registered
    # under a name the deep-path child does not use - see VERIFY_ANSWER_NODE.
    builder.add_node(VERIFY_ANSWER_NODE, verify_answer_node)
    builder.add_node(TTS_NODE, tts_node)

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
        {"fast_synth": "fast_synth", DEEP_PATH_NODE: DEEP_PATH_NODE},
    )
    builder.add_edge("fast_synth", TTS_NODE)
    # The whole deep path is one edge from here now. Its internal wiring -
    # research -> analysis, and the conditional edge into the asking node -
    # moved with it into clarif_eye.deep_path.build_deep_path_graph, which
    # keeps its own comments for why each edge is the shape it is.
    builder.add_edge(DEEP_PATH_NODE, TTS_NODE)
    # A follow-up answer is spoken exactly like a description is: same tts
    # node, same provider chain, same staged delivery in the UI - unless a
    # number in it could not be traced back to the photographed text, in
    # which case the run stops to ask first (issue #92 / P9.11). A STATIC
    # branch on a flag `followup` already computed, so a conditional edge,
    # exactly like the one out of `vision` and the one the deep-path child
    # declares out of `analysis`.
    builder.add_conditional_edges(
        "followup",
        followup_destination,
        {VERIFY_ANSWER_NODE: VERIFY_ANSWER_NODE, TTS_NODE: TTS_NODE},
    )
    builder.add_edge(VERIFY_ANSWER_NODE, TTS_NODE)
    builder.add_edge(TTS_NODE, END)

    return builder.compile(checkpointer=checkpointer, store=store)


# The name the final node is registered under. A CONSTANT because that
# string is not confined to this module's own wiring: clarif_eye.ui passes
# it to graph.update_state(as_node=...) when it resolves a paused thread
# (see _update_thread_state / the cache-hit branch's RESOLVE-THEN-WRITE
# comment), and LangGraph validates that name against the compiled node set
# - InvalidUpdateError("Node <name> does not exist"), verified. Renaming
# the node without updating that far-away call site would turn the whole
# cache-hit thread write into a no-op. With the constant a rename is ONE
# edit; tests/test_graph.py additionally pins TTS_NODE against the compiled
# graph's own node set, so a half-done rename fails immediately and by
# name rather than through three indirect assertions elsewhere.
TTS_NODE = "tts"

# Unconditional successor for every edge above EXCEPT the ones out of
# "entry" (self-routed by Command, resolved by entry_destination instead),
# "vision", "analysis" and "followup" (conditional edges, resolved by
# dynamic_router, analysis_destination and followup_destination instead)
# and "tts" (the last node - see
# next_node_after's END case below). Kept
# literally next to build_graph()'s add_edge calls, which is the ONLY
# reason this is safe to hand-maintain rather than deriving it from the
# StateGraph itself: whoever changes an edge above must update this table
# in the same diff, or next_node_after silently starts lying to callers
# like clarif_eye.ui's per-node progress narration (issue #80 / P9.1).
_UNCONDITIONAL_SUCCESSOR = {
    "fast_synth": TTS_NODE,
    DEEP_PATH_NODE: TTS_NODE,
    # TTS_NODE, not END (issue #92 / P9.11), unlike the child's
    # `verify_numbers` below: this asking node is the PARENT's, and the
    # parent's own edge out of it leads to speech. So the narration announces
    # "turning it into speech" exactly once when a resumed follow-up
    # continues, the same as any other node that precedes tts.
    #
    # `followup` itself is NO LONGER IN THIS TABLE: its edge is conditional
    # now (followup_destination), so next_node_after resolves it by calling
    # that function rather than by looking it up here.
    VERIFY_ANSWER_NODE: TTS_NODE,
    # BOTH GRAPHS' TOPOLOGY, IN ONE TABLE (issue #84 / P9.5). The two names
    # below belong to the CHILD graph (clarif_eye.deep_path), not this one.
    # They are here because the narration that reads this table now sees
    # child node events too - stream(subgraphs=True) surfaces them at the top
    # level (see clarif_eye.ui._narrate_stream) - and a node name is
    # unambiguous across the two graphs, so one table is one place to keep
    # correct rather than two that could disagree.
    #
    # THAT LAST CLAIM IS A REAL PRECONDITION, NOT AN OBSERVATION: this table
    # and clarif_eye.ui's _NODE_PHRASE are both flat dicts keyed by BARE node
    # names, so they are only correct while the parent's node names and the
    # child's never collide. If they ever did, one graph's entry would answer
    # for the other's node and the narration would announce the wrong step -
    # silently, since a dict lookup cannot notice. Pinned by
    # tests/test_deep_path_subgraph.py's disjointness test, which reads both
    # COMPILED graphs' node sets rather than trusting these literals.
    "research": "analysis",
    # END, not TTS_NODE, and that is what stops "Turning it into speech"
    # being announced twice: `verify_numbers` is the last node of the CHILD,
    # so nothing follows it there. The parent announces tts when its own
    # `deep_path` chunk arrives a moment later.
    "verify_numbers": END,
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
        return _successor_or_none(analysis_destination(state))
    if node_name == "followup":
        return _successor_or_none(followup_destination(state))
    if node_name == TTS_NODE:
        return None
    return _successor_or_none(_UNCONDITIONAL_SUCCESSOR.get(node_name))


def _successor_or_none(successor):
    """Translate LangGraph's END sentinel to None (issue #84 / P9.5).

    Two of the names next_node_after resolves belong to the deep-path CHILD
    graph and end AT ITS OWN END rather than at another node. Callers of
    next_node_after already treat None as "there is nothing to announce for
    this step", so folding END into None here means the narration needs no
    idea that a second graph exists - and nothing has to know that END is
    spelled "__end__".
    """
    return None if successor == END else successor
