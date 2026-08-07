"""Tests for checkpointed threads and the first reducer (issue #81 / P9.2).

RED FIRST: at the time this file is committed, clarif_eye.graph.build_graph
takes no `checkpointer` argument and ClarifEyeState has no `messages` key,
so every test below fails (most with a TypeError from build_graph(checkpointer=...)
or a KeyError from state["messages"]).

Fakes follow the same pattern tests/test_graph.py already uses
(FakeVisionClient / _FakeTtsProvider) - no model or network call, so the
graph can be driven deterministically and cheaply.
"""

from clarif_eye.client import CompletionResult
from clarif_eye.graph import build_graph
from clarif_eye.state import make_initial_state


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


def _invoke(graph, image_data, thread_id, ocr="short text", scene="a room"):
    state = make_initial_state(image_data)
    config = {
        "configurable": {
            "thread_id": thread_id,
            "client": FakeVisionClient(ocr, scene),
            "tts_provider": _FakeTtsProvider(),
        }
    }
    return graph.invoke(state, config=config)


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
