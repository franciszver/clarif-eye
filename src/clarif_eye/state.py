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
        # [] is safe to pass on every run, including a second run on an
        # already-checkpointed thread: add_messages merges an empty `right`
        # as "nothing new to append", never as "replace with []" - verified
        # empirically, see the reducer comment above. This is NOT a reset.
        messages=[],
    )
