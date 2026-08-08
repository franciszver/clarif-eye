"""Tests for checkpointed threads and the first reducer (issue #81 / P9.2).

RED FIRST (original commit): at the time this file was first committed,
clarif_eye.graph.build_graph took no `checkpointer` argument and
ClarifEyeState had no `messages` key, so every test here failed (most with
a TypeError from build_graph(checkpointer=...) or a KeyError from
state["messages"]).

Fakes follow the same pattern tests/test_graph.py already uses
(FakeVisionClient / _FakeTtsProvider) - no model or network call, so the
graph can be driven deterministically and cheaply.

BOUNDARY RECORDING (issue #81 / P9.2, simplify-gate follow-up): tts_node
itself does not append to `messages` - clarif_eye.ui._run_pipeline_events
records one turn per completed run via graph.update_state() at the
conversation boundary (see that function's docstring). `_invoke` below
mirrors that exact call so these tests keep exercising the real
accumulate/isolate contract without importing all of ui.py's guards/cache/
narration - the full UI-level seam (build_interface -> handle_submit_staged
-> _run_pipeline_events with a real ThreadRegistry) is covered separately
in tests/test_ui.py.
"""

from clarif_eye.client import CompletionResult
from clarif_eye.graph import build_graph
from clarif_eye.state import make_initial_state
from clarif_eye.ui import _trim_thread_to_latest_checkpoint


class FakeVisionClient:
    """Same shape as tests/test_graph.py's FakeVisionClient - a fixed reply
    regardless of image, short enough to keep the router on the fast path
    (vision -> fast_synth -> tts) so this file never has to fake a search
    backend too."""

    def __init__(self, ocr, scene):
        self.ocr = ocr
        self.scene = scene

    def complete(self, role, messages, **params):
        return CompletionResult(
            content=f"OCR_TEXT: {self.ocr}\nSCENE: {self.scene}", model="fake-eyes-model:free"
        )


class _FakeTtsProvider:
    """Writes a minimal valid-looking mp3 so run_tts's own "looks like
    audio" check passes without touching the network - same double
    tests/test_graph.py uses for tts_node's provider seam."""

    def synthesize(self, text, out_path):
        with open(out_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 32)


def _invoke(graph, image_data, thread_id, ocr="short text", scene="a room", trim=False):
    """Run one turn on `thread_id` and record it, mirroring exactly what
    clarif_eye.ui._run_pipeline_events does at the conversation boundary
    (graph.update_state after a completed run, `as_node` omitted - verified
    empirically to resolve unambiguously to "tts", the only node every path
    ends on). `trim=True` also calls the trim helper afterward, the same
    order _run_pipeline_events uses."""
    state = make_initial_state(image_data)
    config = {
        "configurable": {
            "thread_id": thread_id,
            "client": FakeVisionClient(ocr, scene),
            "tts_provider": _FakeTtsProvider(),
        }
    }
    result = graph.invoke(state, config=config)
    final_output = (result.get("final_output") or "").strip()
    if final_output:
        graph.update_state(config, {"messages": [{"role": "assistant", "content": final_output}]})
    if trim:
        _trim_thread_to_latest_checkpoint(graph.checkpointer, thread_id)
    return result


def test_get_state_after_two_runs_on_same_thread_has_second_runs_scalar_keys_and_accumulated_messages():
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "thread-a"}}

    _invoke(graph, "image-one-payload", "thread-a", ocr="first photo text", scene="first scene")
    _invoke(graph, "image-two-payload", "thread-a", ocr="second photo text", scene="second scene")

    state = graph.get_state(config)
    values = state.values

    # Second run's scalar (non-reducer) keys win - LangGraph replaces them,
    # it does not merge them.
    assert values["image_data"] == "image-two-payload"
    assert "second photo text" in values["ocr_output"]

    # messages ACCUMULATED: two entries, in order, the first run's entry
    # still present - not replaced by the second run's.
    messages = values["messages"]
    assert len(messages) == 2
    assert "first photo text" in messages[0].content
    assert "second photo text" in messages[1].content


def test_different_thread_id_sees_no_bleed_through():
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(checkpointer=InMemorySaver())

    _invoke(graph, "image-thread-a", "thread-a", ocr="thread a text", scene="scene a")
    _invoke(graph, "image-thread-b", "thread-b", ocr="thread b text", scene="scene b")

    state_b = graph.get_state({"configurable": {"thread_id": "thread-b"}})
    messages_b = state_b.values["messages"]

    assert len(messages_b) == 1
    assert "thread b text" in messages_b[0].content
    assert "thread a text" not in messages_b[0].content


# --- Internal-shape pin for _trim_thread_to_latest_checkpoint --------------
#
# _trim_thread_to_latest_checkpoint (clarif_eye.ui) reaches into
# InMemorySaver's undocumented-as-public internals (`storage`, `writes`,
# `blobs`) because InMemorySaver has no supported trim API. That is a real
# coupling to a specific langgraph version's internals - this test pins the
# assumption so a langgraph upgrade that changes the shape fails LOUDLY
# here, instead of the trim function silently no-op'ing (it wraps its body
# in try/except Exception, on purpose - see that function's docstring -
# specifically so a housekeeping failure never takes down a real photo's
# outcome; that same broad except would otherwise hide a shape change
# forever).
def test_in_memory_saver_internal_shape_matches_trim_helpers_assumptions():
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    assert hasattr(saver, "storage")
    assert hasattr(saver, "writes")
    assert hasattr(saver, "blobs")
    assert hasattr(saver, "serde")
    assert hasattr(saver.serde, "loads_typed")

    graph = build_graph(checkpointer=saver)
    _invoke(graph, "shape-probe-image", "shape-probe-thread")

    # storage: thread_id -> checkpoint_ns -> checkpoint_id -> (ckpt_bytes, meta_bytes, parent_id)
    ns_checkpoints = saver.storage["shape-probe-thread"][""]
    assert ns_checkpoints, "expected at least one stored checkpoint after one invoke"
    latest_id = max(ns_checkpoints.keys())
    entry = ns_checkpoints[latest_id]
    assert len(entry) == 3
    checkpoint_bytes, metadata_bytes, _parent = entry
    checkpoint = saver.serde.loads_typed(checkpoint_bytes)
    assert "channel_versions" in checkpoint

    # blobs: (thread_id, checkpoint_ns, channel, version) -> (type, bytes)
    matching_blob_keys = [k for k in saver.blobs if k[0] == "shape-probe-thread" and k[1] == ""]
    assert matching_blob_keys
    assert all(len(k) == 4 for k in matching_blob_keys)

    # writes: (thread_id, checkpoint_ns, checkpoint_id) -> {...}
    assert all(len(k) == 3 for k in saver.writes)


# --- Cross-namespace ordering pin for _drop_dead_subgraph_namespaces -------
#
# _drop_dead_subgraph_namespaces (clarif_eye.ui) decides which subgraph
# namespace is the LIVE one by comparing namespaces on their newest
# checkpoint id and keeping the greatest - i.e. it assumes checkpoint ids are
# time-ordered GLOBALLY, across namespaces, not merely within one. That is a
# strictly stronger assumption than the per-namespace max() the trim above
# already makes (and than InMemorySaver.get_tuple makes itself), and it is
# load-bearing in the worst possible way: if a langgraph upgrade made ids
# random rather than time-ordered (v4 rather than v6 UUIDs, say), this
# function would delete the WRONG namespace - possibly the one holding a
# PAUSED run's checkpoint - and the user's unanswered safety question would
# become unresumable. Silently: the deletion cannot fail, it would simply
# take the wrong thing.
def test_checkpoint_ids_are_time_ordered_across_namespaces():
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = build_graph(checkpointer=saver)
    thread_id = "ns-ordering-thread"

    # Two DEEP-PATH runs on ONE thread: each mounts the child under its own
    # `deep_path:<task id>` namespace, so this produces exactly the situation
    # the function has to judge - two child namespaces, one older. Long OCR
    # text is what trips the router onto the deep path at all (the same
    # long-document fixture the rest of the suite uses).
    long_ocr = " ".join(["alpha"] * 200)

    def _child_namespaces():
        return {ns for ns in saver.storage[thread_id] if ns}

    _invoke(graph, "ns-ordering-image-one", thread_id, ocr=long_ocr)
    after_first = _child_namespaces()
    _invoke(graph, "ns-ordering-image-two", thread_id, ocr=long_ocr)
    namespaces = _child_namespaces()

    assert len(namespaces) == 2, f"expected two child namespaces, got {namespaces}"

    # RECENCY, ESTABLISHED WITHOUT LOOKING AT AN ID: the newer namespace is
    # simply the one that was not there after the first run. Comparing that
    # against what max() picks is what makes this a proof of the ordering
    # assumption rather than a restatement of it.
    by_recency = (namespaces - after_first).pop()
    by_id = max(namespaces, key=lambda ns: max(saver.storage[thread_id][ns]))

    assert by_id == by_recency, (
        "checkpoint ids are no longer time-ordered across namespaces - "
        "clarif_eye.ui._drop_dead_subgraph_namespaces would delete the wrong "
        "namespace, possibly one holding a paused run."
    )


# --- Measured defect: unbounded per-thread checkpoint growth --------------
#
# MEASURED (issue #81's simplify-gate report): InMemorySaver.put() writes
# new storage/writes/blobs entries on every node completion and never
# prunes - ~134KB/invoke with a 50KB image, 8MB after 10 invokes with a
# 400KB image, all on ONE thread. This test proves _trim_thread_to_latest_checkpoint
# actually closes that: run several invokes on ONE thread, trimming after
# each (the same order _run_pipeline_events uses), and assert the
# checkpoint count for that thread stays flat at 1 rather than growing with
# the invoke count.
#
# PROVEN TO FAIL WITHOUT THE FIX: manually verified by disabling the
# `_trim_thread_to_latest_checkpoint` call inside `_invoke` (short-circuit
# `if trim:` to never fire) and re-running this test - it goes red,
# reporting 30 stored checkpoints (6 per invoke x 5 invokes) instead of 1.
# Restored immediately after. Not committed as a permanently-skipped
# variant since that would just be dead code here.
def test_trimming_keeps_one_threads_checkpoint_count_flat_across_repeated_invokes():
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = build_graph(checkpointer=saver)
    thread_id = "growth-probe-thread"

    for i in range(5):
        _invoke(graph, f"image-payload-{i}", thread_id, ocr=f"text {i}", scene=f"scene {i}", trim=True)

    ns_checkpoints = saver.storage[thread_id][""]
    assert len(ns_checkpoints) == 1, (
        f"expected exactly 1 stored checkpoint after trimming, found {len(ns_checkpoints)} - "
        "checkpoint history is growing unbounded"
    )

    # The trim must not have broken anything: get_state still shows every
    # accumulated message, and the graph still works for a further invoke.
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    assert len(state.values["messages"]) == 5

    _invoke(graph, "image-payload-final", thread_id, ocr="final text", scene="final scene", trim=True)
    state = graph.get_state(config)
    assert len(state.values["messages"]) == 6
    assert len(saver.storage[thread_id][""]) == 1
