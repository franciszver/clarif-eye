"""A degraded run's wording must never enter conversation memory (issue #93 / P9.12).

RED FIRST (original commit): three of this file's tests failed before
clarif_eye.state grew the `output_degraded` key and
clarif_eye.ui._record_turn learned to read it. The failures were not
theoretical - they were the exact shapes the issue was filed for:

  - a follow-up on a thread that has never described a photo answers with
    clarif_eye.followup.NO_PHOTO_YET_MESSAGE, which TTS then speaks
    perfectly happily. The run therefore looks like a complete success at
    the recording boundary (real audio, non-empty text) and both the typed
    question AND "there is no photo to answer questions about yet" were
    written into the thread's history.
  - a photo run whose vision call fails degrades to a spoken failure
    message ("...could not be described..."), which is likewise spoken and
    likewise recorded - so the thread's history claimed that message was
    this photo's description.

WHY THE STATUS AT THE BOUNDARY WAS NOT ENOUGH, stated here because it is
the design question this issue turned on: clarif_eye.ui._outcome_for's
STATUS_DEGRADED describes the RUN's ending (did TTS produce audio, was the
provider chain exhausted), not the ANSWER's honesty. Both scenarios above
end on STATUS_SUCCESS_AUDIO. The signal has to come from the node that
degraded, which is why it travels in state - see clarif_eye.state's
`output_degraded` comment.

NO PROSE MATCHING ANYWHERE (D15): nothing here asserts that a particular
failure sentence is absent from history. The tests assert on the number of
recorded messages, which is what the guard actually controls.

The deep-path test below came with the FIX rather than with that red
commit, and it is honest to say why: mutation testing showed the
parent/child half of the wiring was unpinned, so a test was added to pin
it. Its own docstring records what the mutation actually proved.

Same no-network discipline as tests/test_ui.py: real compiled graph, real
checkpointer, real ThreadRegistry, fake client, fake TTS provider.
"""

from langgraph.checkpoint.memory import InMemorySaver

from clarif_eye import tts as tts_module
from clarif_eye.client import CompletionResult
from clarif_eye.followup import NO_PHOTO_YET_MESSAGE
from clarif_eye.graph import build_graph
from clarif_eye.ui import (
    AppResources,
    ThreadRegistry,
    handle_ask_staged,
    handle_submit_staged,
)
from clarif_eye.vision import is_degraded_scene

from tests.test_ask_before_speaking import BILL_OCR, BILL_SCENE
from tests.test_ui import BrokenImage, FakeImage

QUESTION = "what is the expiry date?"
GOOD_OCR = "best before next April"
GOOD_SCENE = "a jar of jam on a kitchen counter"


class _WorkingClient:
    """Answers every call the fast path makes, so a photo run produces a
    genuine description and a follow-up a genuine answer."""

    def complete(self, role, messages, **params):
        has_image = any(
            isinstance(message.get("content"), list)
            and any(part.get("type") == "image_url" for part in message["content"])
            for message in messages
        )
        if has_image:
            return CompletionResult(
                content=f"OCR_TEXT: {GOOD_OCR}\nSCENE: {GOOD_SCENE}",
                model="fake-eyes-model:free",
            )
        return CompletionResult(content="A jar of jam on a counter.", model="fake-brain-model:free")

    def close(self):
        pass


class _FailingClient:
    """Every model call blows up, so `vision` degrades through its own
    never-raise path and `fast_synth` passes that degradation message
    through as final_output - a run that is spoken aloud and looks like a
    success from the outside. NOT a TTS failure: the speech below still
    works, which is precisely what makes this run indistinguishable from a
    real answer at the recording boundary."""

    def complete(self, role, messages, **params):
        raise RuntimeError("every model is unavailable")

    def close(self):
        pass


class _EmptyBrainClient:
    """Vision reads a DENSE DOCUMENT (so clarif_eye.router sends the run
    down the deep path and `analysis` is what writes final_output), and the
    brain model then returns an empty reply - one of clarif_eye.analysis's
    own degradation branches.

    THIS IS THE ONLY WAY TO REACH THE PARENT/CHILD BOUNDARY with a degraded
    answer: `analysis` runs inside the deep-path CHILD graph, so its flag
    has to be declared by clarif_eye.deep_path.DeepPathState AND mapped
    back out by clarif_eye.graph.make_deep_path_node's wrapper. LangGraph
    drops an undeclared update key without raising, so nothing but a test
    that runs the whole graph can catch that half of the wiring.
    """

    def complete(self, role, messages, **params):
        has_image = any(
            isinstance(message.get("content"), list)
            and any(part.get("type") == "image_url" for part in message["content"])
            for message in messages
        )
        if has_image:
            return CompletionResult(
                content=f"OCR_TEXT: {BILL_OCR}\nSCENE: {BILL_SCENE}",
                model="fake-eyes-model:free",
            )
        return CompletionResult(content="   ", model="fake-brain-model:free")

    def close(self):
        pass


class _EmptySearcher:
    """No search results, so `research` degrades to "found nothing" without
    opening a socket. The web lookup is not what this file is about - it is
    just on the road to `analysis`."""

    def text(self, query, **kwargs):
        return []


class _FakeTtsProvider:
    def synthesize(self, text, out_path):
        with open(out_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 32)


def _resources(client):
    checkpointer = InMemorySaver()
    return AppResources(
        graph=build_graph(checkpointer=checkpointer),
        client=client,
        client_error=None,
        tts_providers=[_FakeTtsProvider()],
        searcher=_EmptySearcher(),
        research_client=None,
        thread_registry=ThreadRegistry(checkpointer),
    )


def setup_function(_fn):
    tts_module._last_result_set(None)


def _messages(resources, thread_id):
    snapshot = resources.graph.get_state({"configurable": {"thread_id": thread_id}})
    return snapshot.values.get("messages") or []


# --- The degraded runs -----------------------------------------------------


def test_a_follow_up_with_no_photo_yet_records_no_turn_at_all():
    """THE ORIGINAL PROBE. A question typed before any photo is answered
    with NO_PHOTO_YET_MESSAGE - a real, spoken, honest answer to the user,
    and a lie if a later turn reads it back as this thread's answer.

    NEITHER SIDE of the turn is recorded, not just the assistant's. The
    user really did type the question, but recording it alone would leave a
    question standing in history with no answer under it, and the next real
    answer sitting directly beneath it - see clarif_eye.ui._record_turn for
    why "a failed turn leaves no trace" beats "record half of it".
    """
    resources = _resources(_WorkingClient())
    thread_id = "no-photo-yet"

    updates = list(handle_ask_staged(QUESTION, resources, thread_id=thread_id))

    # The scenario is real: the user WAS answered, and it WAS spoken aloud.
    final_status, final_audio, final_text = updates[-1]
    assert NO_PHOTO_YET_MESSAGE in final_text
    assert final_audio, "the explanation is spoken - this is not a TTS failure"

    assert _messages(resources, thread_id) == []


def test_a_photo_run_that_degraded_inside_the_pipeline_records_no_turn():
    """The deeper hole: the run COMPLETED and was spoken, so the boundary
    sees audio plus non-empty text - exactly what a real description looks
    like. Only the node knows it degraded."""
    resources = _resources(_FailingClient())
    thread_id = "degraded-photo"

    updates = list(handle_submit_staged(FakeImage(), resources, thread_id=thread_id))

    final_status, final_audio, final_text = updates[-1]
    assert final_audio, "the failure message is spoken - this is not a TTS failure"
    assert final_text
    # Structural, not prose: what was spoken is vision's own degradation
    # message, recognised by the predicate vision.py already owns.
    assert is_degraded_scene(final_text)

    assert _messages(resources, thread_id) == []


def test_a_deep_path_run_that_degraded_inside_the_child_graph_records_no_turn():
    """THE PARENT/CHILD BOUNDARY, pinned by behaviour rather than by schema.

    `analysis` degrades inside the deep-path child graph, so its flag has to
    cross back into the parent's state before the recording boundary can see
    it. MEASURED BY MUTATION, not assumed, and the answer was not the
    obvious one: removing clarif_eye.deep_path.DeepPathState's declaration
    of the key reds THIS test (the wrapper's bracket read raises KeyError,
    which the never-raise boundary turns into a spoken failure with no
    audio) - while removing the wrapper's write-back LINE changes nothing,
    because clarif_eye.ui._narrate_stream streams with subgraphs=True and
    `analysis`'s own chunk already carries the flag to the boundary. See
    that write-back's comment, which now says so.
    """
    resources = _resources(_EmptyBrainClient())
    thread_id = "degraded-deep-path"

    updates = list(handle_submit_staged(FakeImage(), resources, thread_id=thread_id))

    final_status, final_audio, final_text = updates[-1]
    assert final_audio, "the failure message is spoken - this is not a TTS failure"
    assert final_text
    assert _messages(resources, thread_id) == []
    assert resources.image_cache._entries == {}


def test_a_degraded_photo_run_is_not_cached_so_no_hit_can_record_it_later():
    """The cache-hit branch records the CACHED text as this thread's turn.
    If a degraded description could be cached, that branch would put it
    into a second visitor's history with no node left to ask - so the
    invariant is kept where the entry is written. This is also what
    ImageResultCache's docstring has always claimed ("only successful
    results are ever stored here")."""
    resources = _resources(_FailingClient())

    list(handle_submit_staged(FakeImage(content=b"degraded-photo"), resources, thread_id="visitor-one"))

    assert resources.image_cache._entries == {}


def test_an_early_failure_photo_run_records_nothing():
    """Pins what was already true: a run that fails before the graph can
    produce anything writes no turn. Named here so the new guard cannot be
    written in a way that accidentally starts recording these."""
    resources = _resources(_WorkingClient())
    thread_id = "unreadable-photo"

    updates = list(handle_submit_staged(BrokenImage(), resources, thread_id=thread_id))

    assert updates[-1][1] is None, "an early failure speaks nothing"
    assert _messages(resources, thread_id) == []


# --- The real answers, unchanged ------------------------------------------


def test_a_real_description_and_a_real_answer_are_still_recorded():
    """The guard must not be over-broad. A working photo run records its
    description; a working follow-up records both sides of its turn."""
    resources = _resources(_WorkingClient())
    thread_id = "healthy-thread"

    list(handle_submit_staged(FakeImage(), resources, thread_id=thread_id))
    assert len(_messages(resources, thread_id)) == 1

    list(handle_ask_staged(QUESTION, resources, thread_id=thread_id))
    messages = _messages(resources, thread_id)
    assert len(messages) == 3
    assert QUESTION in messages[1].content
