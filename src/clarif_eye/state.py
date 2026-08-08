"""Graph state schema for Clarif-Eye.

Exactly the keys from the architecture doc's pipeline
(vision -> router -> fast synth or research + analysis -> tts), plus
`messages` (issue #81 / P9.2 - see below). Every key is present from the
start via make_initial_state; nodes fill them in, they never invent new
keys later.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

# --- messages: the app's first LangGraph reducer (issue #81 / P9.2) --------
#
# Every other key above is a PLAIN value: LangGraph's default merge behavior
# REPLACES it wholesale with whatever the most recent node update (or a
# fresh make_initial_state() passed into graph.invoke/stream) supplied.
# `messages` is different on purpose - it must ACCUMULATE across runs on
# the same checkpointed thread (clarif_eye.graph.build_graph's
# `checkpointer` param / clarif_eye.ui's per-session thread_id), so a
# second run's initial state does not erase the first run's entry. That is
# exactly what a reducer is for: `add_messages` (langgraph.graph.message)
# APPENDS incoming messages to the existing list instead of replacing it,
# and coerces plain {"role": ..., "content": ...} dicts into the message
# objects LangGraph checkpoints - see clarif_eye.graph.tts_node, the one
# node that appends to this key.
#
# EMPIRICALLY VERIFIED (not assumed - see issue #81's report): invoking a
# checkpointed graph twice on the SAME thread_id with `messages=[]` in both
# initial states does NOT wipe the first run's entry - add_messages merges
# `[]` as "nothing new to append", never as "replace with []". A DIFFERENT
# thread_id sees none of another thread's messages. Removing this
# Annotated(...) reducer (making `messages` a plain list) turns `messages`
# back into a normally-replaced key, so the second run's `messages=[]`
# WOULD wipe the first run's entry - tests/test_checkpointing.py's
# accumulation assertion is written to catch exactly that regression.
#
# BOUNDED (issue #81 / P9.2 "bound memory"): an unbounded thread's message
# list would grow forever across enough runs on one thread_id, and every
# checkpoint snapshot stores the FULL state - including image_data's base64
# JPEG bytes (clarif_eye.ui build_resources's InMemorySaver comment) - on a
# 512MB free instance. `_keep_last_n_messages` wraps add_messages so a
# thread's message list never exceeds MAX_MESSAGES_PER_THREAD entries after
# a merge, capping THIS key's own contribution to that footprint
# independently of how many runs a thread accumulates. 40 is chosen as
# "generous for a demo session (20 back-and-forth turns' worth) without
# letting an unusually long-lived thread grow its checkpoint without bound"
# - no measurement backs the exact number, the same honestly-provisional
# spirit as analysis.py's _SCRAPER_DATA_CAP.
MAX_MESSAGES_PER_THREAD = 40


def _keep_last_n_messages(left, right):
    """add_messages, then truncate the MERGED list to the last
    MAX_MESSAGES_PER_THREAD entries.

    Truncating AFTER merging (not capping `right` alone) is what actually
    bounds a thread's checkpointed state long-term - capping only the
    incoming update would still let `left` (the accumulated history so
    far) grow forever across enough runs.
    """
    merged = add_messages(left, right)
    return merged[-MAX_MESSAGES_PER_THREAD:]


class ClarifEyeState(TypedDict):
    image_data: str
    ocr_output: str
    scene_context: str
    complexity_flag: bool
    # str | None (issue #81 / P9.2 - see this key's history below):
    #   None -> research never ran (fast path).
    #   ""   -> research ran and found nothing usable.
    # Previously both cases collapsed to "" with no way to tell them apart
    # (see git history / issue #10's module docstring in research.py for
    # why that was a deliberate, reasoned choice at the time - analysis.py
    # never needed to distinguish them, and still doesn't; see
    # analysis.run_analysis, which treats both None and "" as falsy and
    # proceeds identically either way). This is the explicit sentinel this
    # issue's schema-change moment introduces: make_initial_state seeds
    # None (never ran); research.run_research still returns "" in every one
    # of its own degrade-to-nothing branches (ran, found nothing) - that
    # module's own behavior is UNCHANGED, only the value it starts from
    # before it ever runs is now distinguishable from the value it produces
    # when it runs and comes up empty.
    scraper_data: str | None
    final_output: str
    audio_file_path: str
    # str | None (issue #82 / P9.3): the typed follow-up question for THIS
    # run, or None when the run is a photo run. This is the key
    # clarif_eye.graph.entry_node reads to decide - via Command(goto=...) -
    # whether the run goes to `vision` (a photo) or straight to `followup`
    # (a question answered from what this thread already has stored).
    #
    # A PLAIN (non-reducer) KEY, ON PURPOSE, and that is what makes the
    # reset below work. EMPIRICALLY RE-VERIFIED for this issue on langgraph
    # 1.2.10 (the same fact issue #81 established for the other scalar
    # keys): on a checkpointed thread, keys present in the input REPLACE
    # the checkpointed value, and keys ABSENT from the input are preserved
    # untouched. Both halves are load-bearing here:
    #   - a follow-up passes ONLY {"question": q} (see
    #     clarif_eye.ui._run_followup_events), so the thread's stored
    #     ocr_output/scene_context survive to be answered from. Passing a
    #     full make_initial_state() there would wipe them and there would
    #     be nothing left to answer from.
    #   - a photo run passes make_initial_state(), which seeds
    #     question=None (below), so a question left over from the previous
    #     turn is RESET rather than surviving to divert the new photo run
    #     into `followup` - a stale question must never eat a fresh photo.
    question: str | None
    # dict | None (issue #83 / P9.4): the drafted script the deep-analysis
    # path is HOLDING BACK because a number in it could not be traced to
    # the photographed text, plus which tokens failed:
    #     {"script": "<the drafted spoken script>", "numbers": ["999.99"]}
    # None means "nothing is being held" - a verified reply, or any path
    # that never verifies numbers at all (fast_synth, followup).
    #
    # WHY THIS TRAVELS IN STATE RATHER THAN BEING RECOMPUTED where it is
    # used: `verify_numbers` (clarif_eye.graph) is the node that asks the
    # user about it, and LangGraph re-executes the WHOLE interrupted node
    # when the run is resumed (verified empirically on langgraph 1.2.10 -
    # see verify_numbers_node's docstring). Anything expensive ahead of the
    # interrupt would therefore run TWICE. Keeping the brain model's draft
    # here means the asking node does no work of its own beyond reading
    # this key, so a resume costs nothing but the speech at the end.
    #
    # A PLAIN (non-reducer) KEY, like every other scalar here, and analysis
    # writes it on EVERY return path (None when there is nothing to hold).
    # That matters on a checkpointed thread: a hold left over from an
    # earlier photo would otherwise survive into the next run and stop it
    # to ask about a number nobody just heard.
    verification_hold: dict | None
    # bool (issue #93 / P9.12): True means final_output is a DEGRADATION
    # message - "every model was busy", "the photo could not be read", "there
    # is no photo to answer questions about yet" - rather than an answer
    # about what was photographed. False means it is the real thing.
    #
    # WHY THIS HAD TO TRAVEL IN STATE, rather than being worked out at the
    # conversation boundary where it is used (clarif_eye.ui._record_turn):
    # only the node that degraded knows. Every degradation message here is
    # SPOKEN - it goes through tts exactly like a description does, because a
    # blind user must hear why they got nothing - so by the time the run
    # reaches the boundary it has real audio and non-empty text, and looks
    # byte-for-byte like a success. clarif_eye.ui's outcome status
    # (STATUS_DEGRADED) describes how the RUN ended, not whether the ANSWER
    # is honest, and reports STATUS_SUCCESS_AUDIO for both. The only other
    # way to tell them apart from outside would be to match the failure
    # wording, which this codebase does not do (D15) and which would break
    # silently the first time a message was reworded.
    #
    # WRITTEN BY EVERY NODE THAT WRITES final_output, on every return path:
    # clarif_eye.synth, clarif_eye.analysis and clarif_eye.followup each set
    # it from their single `_degraded()` helper (True) and from their one
    # success return (False), and clarif_eye.graph.verify_numbers_node sets
    # it for the two answers it can be resumed with. A plain, non-reducer
    # key like every other scalar here, so a True left over from an earlier
    # failed run is REPLACED by the next run rather than surviving - and
    # make_initial_state seeds False for the same reason it seeds
    # question/verification_hold explicitly.
    #
    # DEFAULTS TO "not degraded" WHEN ABSENT at the reading end, deliberately:
    # a follow-up run's input is only {"question": ...} (see
    # clarif_eye.ui._run_followup_events), so nothing seeds this key on that
    # path and the answering node is what puts it there. Absent therefore
    # means "no node claimed a degradation", which is the honest reading and
    # keeps a future node that forgets to set it recording turns exactly as
    # it does today rather than silently losing them.
    output_degraded: bool
    # See "messages: the app's first LangGraph reducer" above.
    messages: Annotated[list, _keep_last_n_messages]


def make_initial_state(image_data):
    """Build the initial state from `image_data`, every other key at its empty default."""
    if image_data is None or not str(image_data).strip():
        raise ValueError(
            "make_initial_state: image_data is required and must be a "
            "non-empty, non-blank string"
        )
    return ClarifEyeState(
        image_data=image_data,
        ocr_output="",
        scene_context="",
        complexity_flag=False,
        # None = "research never ran yet" - see ClarifEyeState.scraper_data.
        scraper_data=None,
        final_output="",
        audio_file_path="",
        # None = "this is a photo run, not a question" - and, on a thread
        # that already answered a question, this explicitly RESETS the
        # stored question so the new photo run is not diverted into the
        # followup node. See ClarifEyeState.question.
        question=None,
        # None = "nothing is being held back pending a question" - and, on
        # a thread whose previous run was abandoned mid-question, this
        # explicitly CLEARS that hold so the new photo run is not stopped
        # to ask about the old photo's number. See
        # ClarifEyeState.verification_hold.
        verification_hold=None,
        # False = "nothing has degraded yet" - and, on a checkpointed thread
        # whose previous run degraded, this explicitly RESETS that flag so
        # the new run is judged on its own outcome. See
        # ClarifEyeState.output_degraded.
        output_degraded=False,
        # [] is safe to pass on every run, including a second run on an
        # already-checkpointed thread: add_messages merges an empty `right`
        # as "nothing new to append", never as "replace with []" - verified
        # empirically, see the reducer comment above. This is NOT a reset.
        messages=[],
    )
