"""The deep path as a compiled child graph, with its OWN schema (issue #84 /
P9.5).

WHAT MOVED AND WHY
--------------------
`research -> analysis -> [verify_numbers]` used to be three nodes sitting
directly in clarif_eye.graph's parent graph, sharing ClarifEyeState with
`vision`, `fast_synth`, `followup` and `tts`. They are now a graph of their
own, mounted in the parent as a single node called "deep_path" (see
clarif_eye.graph.build_graph and deep_path_node).

The parent is UNCHANGED where it matters: same entry, same vision, same
fast-path, same tts, same spoken result, and - the red line of this issue -
the same sequence of announcements a blind user hears (see
tests/test_deep_path_subgraph.py's two byte-identity guards).

OWN SCHEMA, NOT SHARED STATE - AND WHY THE INPUT KEYS ARE RENAMED
-------------------------------------------------------------------
LangGraph has two ways to mount a child graph, and they are chosen by the
child's SCHEMA, not by a flag:

  - SHARED STATE: the child's schema keys are a subset of the parent's, with
    the SAME names. `add_node("deep_path", child)` then works bare -
    LangGraph maps the channels for you. EMPIRICALLY VERIFIED on langgraph
    1.2.10: a child declaring {ocr_output, scene_context, final_output}
    mounts and runs inside a parent that has those keys, with no wrapper
    written at all. This is the mode the parent graph already IS internally,
    which is exactly why demonstrating it again here would demonstrate
    nothing.

  - OWN SCHEMA + A MAPPING WRAPPER: the child declares its own vocabulary,
    and a plain function node translates parent state into child input and
    child output back into parent state. This is the mode that proves
    encapsulation, and it is the one this module is in.

The two INPUT keys are deliberately renamed - `ocr_output` becomes
`document_text`, `scene_context` becomes `scene_description` - and that
rename is what makes the wrapper genuinely load-bearing rather than
decorative. VERIFIED, not assumed: mounted bare inside a StateGraph
(ClarifEyeState), this child raises KeyError('document_text') the instant
its first node runs, because nothing in the parent ever wrote that key. See
tests/test_deep_path_subgraph.py's
test_the_child_cannot_be_mounted_without_the_mapping_wrapper, which pins it
by mounting the child bare and expecting the failure.

The three remaining keys keep the names they already had
(`scraper_data`, `verification_hold`, `final_output`). That is a judgement
call, made for two concrete reasons and not for lack of nerve:
  - clarif_eye.research.run_research and clarif_eye.analysis.run_analysis
    already RETURN dicts keyed exactly that way, so keeping the names means
    those two modules need no adapter and no edit at all - the extraction
    touches wiring, not the code that does the work.
  - clarif_eye.graph.next_node_after (the single source of truth this app's
    narration reads) resolves the branch out of `analysis` by reading
    `verification_hold`. One name for that key across both graphs keeps
    that one function honest for parent AND child node names.
Encapsulation is not weakened by it: the child's schema is its own
TypedDict, it cannot be mounted shared-state (proven above), and it does not
declare - so cannot see, checkpoint, or leak - `image_data`,
`complexity_flag`, `audio_file_path`, `question` or `messages`.

THE PRIVACY WIN, STATED PLAINLY: the child's checkpoints hold no photo.
Every checkpoint LangGraph writes stores the WHOLE state of the graph
writing it, and the parent's state includes `image_data`, a base64 JPEG of
something a blind user cannot see (see clarif_eye.ui's ThreadRegistry
comment for the measured footprint). The deep path never needed the photo -
it works from text vision already read - and now it structurally cannot
receive one.

WHY `verify_numbers` IS INSIDE THE CHILD AND NOT LEFT BEHIND
--------------------------------------------------------------
Because the asking node is the deep path's own business, and because this
issue is about proving that a pause fires from INSIDE a child and still
reaches the top-level caller. EMPIRICALLY VERIFIED on langgraph 1.2.10
before this was designed (probes, not assumption):
  - `interrupt()` raised inside a child invoked from a parent node surfaces
    on the PARENT's stream. With `subgraphs=True` it arrives twice: once
    namespaced to the child, then again at the parent level - see
    clarif_eye.ui._narrate_stream, which acts on the parent-level copy only
    so the question is spoken once.
  - `graph.get_state(parent_config).interrupts` reports the pending
    question, and `.next` names ("deep_path",) - the node the child is
    mounted at, not the child's own paused node. clarif_eye.ui's
    _has_pending_interrupt reads `.interrupts`, so it keeps working
    unchanged.
  - `graph.stream(Command(resume=answer), parent_config)` reaches back INTO
    the child. The wrapper node re-executes from its first line (that is
    what LangGraph does with any interrupted node), but the child resumes at
    its own paused node: `research` and `analysis` do NOT re-run, so the
    brain model call is not paid twice. That is the same guarantee issue
    #83 was built on, and tests/test_deep_path_subgraph.py's resume test
    counts the calls to keep it honest.

NO CHECKPOINTER OF ITS OWN, ON PURPOSE
----------------------------------------
build_deep_path_graph() compiles WITHOUT a checkpointer. Mounted in a
checkpointed parent, the child inherits the parent's through the config
LangGraph propagates into it, and its checkpoints land under their own
namespace (`deep_path:<task id>`) on the same thread - so a pause inside it
is resumable exactly like one in the parent. Invoked STANDALONE (see
clarif_eye.ui.describe_document_text), it has no checkpointer at all, which
is the right answer there: that caller has no UI to ask a question through
and nothing to resume with. VERIFIED: an `interrupt()` in an uncheckpointed
graph does NOT raise - invoke() returns the state it had reached plus an
"__interrupt__" key - so that caller simply reads the safe script
`analysis` already wrote and never sees an exception.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph


class DeepPathState(TypedDict):
    """The deep path's own vocabulary - see this module's docstring for why
    the two inputs are renamed and the other three are not."""

    # INPUT: the text vision read out of the photo. Called `document_text`
    # here, not `ocr_output`, because this graph does not care that the text
    # came from a camera - clarif_eye.ui.describe_document_text hands it
    # typed text with no photo anywhere in sight.
    document_text: str
    # INPUT: what the photo showed, in words. Called `scene_description` for
    # the same reason.
    scene_description: str
    # INTERNAL: what the web lookup found. None means research never ran,
    # "" means it ran and found nothing - the same two-value distinction
    # clarif_eye.state.ClarifEyeState.scraper_data documents.
    scraper_data: str | None
    # INTERNAL: the drafted script held back because a number in it could
    # not be traced to the document text, plus which tokens failed. See
    # clarif_eye.state.ClarifEyeState.verification_hold - same shape, same
    # meaning, now owned by the graph that actually produces and consumes it.
    verification_hold: dict | None
    # OUTPUT: the spoken script. The one key the wrapper maps back out.
    final_output: str


def build_deep_path_graph():
    """Compile the deep path: research -> analysis -> [verify_numbers].

    Node functions live in clarif_eye.graph alongside the parent's, and are
    imported here rather than the other way round: they are the same thin
    adapters they always were (config seams, deadline check, delegate to
    run_*), and moving them would have made this a rename diff on top of a
    structural one. This module owns the SHAPE; clarif_eye.graph owns the
    node bodies.

    No checkpointer - see this module's docstring for why that is right for
    both callers.
    """
    from clarif_eye.graph import (
        analysis_destination,
        analysis_node,
        research_node,
        verify_numbers_node,
    )

    builder = StateGraph(DeepPathState)
    builder.add_node("research", research_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("verify_numbers", verify_numbers_node)

    builder.set_entry_point("research")
    builder.add_edge("research", "analysis")
    # The same conditional edge, on the same flag, for the same reason it
    # was a conditional edge in the parent (see analysis_destination's
    # docstring): only the runs with something to ask about pay for the
    # asking node's checkpoint. What changed is where it goes when there is
    # nothing to ask: END OF THE CHILD, not straight to `tts` - `tts` is the
    # parent's node and this graph cannot name it. The parent's own edge out
    # of "deep_path" leads to tts, so the run still ends in speech.
    builder.add_conditional_edges(
        "analysis",
        analysis_destination,
        {"verify_numbers": "verify_numbers", END: END},
    )
    builder.add_edge("verify_numbers", END)

    return builder.compile()
