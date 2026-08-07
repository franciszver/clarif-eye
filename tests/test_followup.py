"""Tests for follow-up questions on a checkpointed thread (issue #82 / P9.3).

RED FIRST (original commit): at the time this file was first committed,
none of the symbols it imports existed - clarif_eye.followup had no module
at all, clarif_eye.ui had no handle_ask_staged/STATUS_ASKING, and
ClarifEyeState had no `question` key - so every test here failed on the
import line.

WHAT A FOLLOW-UP MUST COST: exactly ONE model call, against the `brain`
ladder, built from the ocr_output/scene_context the THREAD already has
checkpointed. No vision call, no re-upload, no second photo. That is not
something a "the answer looks plausible" assertion can prove, so the fake
client here RECORDS every call it receives and the tests assert on the
recording:
  - a VISION call is identified STRUCTURALLY, by the request carrying an
    `{"type": "image_url", ...}` content part (exactly what
    clarif_eye.vision._build_messages puts there) - never by matching the
    role name, because clarif_eye.synth also calls the "eyes" role and does
    NOT send an image, so a role-name count would conflate the two and
    silently pass even if a follow-up re-ran vision.
  - the answer is proven to derive from STORED state by asserting on the
    prompt the brain model actually RECEIVED: it must contain the ocr text
    the earlier photo run stored on this thread, plus the typed question.

Same no-network discipline as tests/test_ui.py and tests/test_graph.py:
real compiled graph, real checkpointer, real ThreadRegistry, fake client and
fake TTS provider. Nothing here launches Gradio or opens a socket.
"""

from clarif_eye import tts as tts_module
from clarif_eye.client import CompletionResult
from clarif_eye.followup import NO_PHOTO_YET_MESSAGE
from clarif_eye.graph import build_graph
from clarif_eye.ui import (
    AppResources,
    STATUS_ASKING,
    STATUS_NODE_ANSWERING,
    STATUS_NODE_TTS,
    ThreadRegistry,
    handle_ask_staged,
    handle_submit_staged,
)

# Reused, not re-declared: tests/test_ui.py already owns the stand-in for a
# PIL Image (a `content=` per "photo" so two photos are told apart by
# CONTENT, which is what issue #75's cache keys on). A second copy here
# would be one more place to update if _encode_image's expectations ever
# change.
from tests.test_ui import FakeImage

# No digits and no document keywords, so clarif_eye.router keeps the photo
# run on the FAST path (vision -> fast_synth -> tts) and this file never has
# to fake a search backend. The exact wording matters: the follow-up test
# asserts this string reaches the brain model's prompt from CHECKPOINTED
# state, not from anything the follow-up call itself passed in.
STORED_OCR_TEXT = "best before next April"
STORED_SCENE_TEXT = "a jar of jam on a kitchen counter"

QUESTION = "what is the expiry date?"
CANNED_ANSWER = "The label says best before next April."


class RecordingClient:
    """Records every complete() call, so a test can assert on what the
    models were actually ASKED, not just on what came back.

    `calls` entries are (role, has_image, prompt_text) triples. `has_image`
    is computed structurally from the request body (an "image_url" content
    part), which is the only honest way to tell a real vision call apart
    from clarif_eye.synth's image-free call on the SAME "eyes" role.

    `ocr_texts`, when given, makes each successive VISION call report a
    different photo - which is what lets a test tell "the thread remembers
    the photo the user is actually looking at" apart from "the thread
    remembers whichever photo happened to run last".
    """

    def __init__(self, ocr_texts=None):
        self.calls = []
        self.ocr_texts = list(ocr_texts) if ocr_texts else None

    def _next_ocr(self):
        if self.ocr_texts is None:
            return STORED_OCR_TEXT
        return self.ocr_texts[len(self.vision_calls()) - 1]

    @staticmethod
    def _flatten(messages):
        parts = []
        has_image = False
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
                continue
            for part in content:
                if part.get("type") == "image_url":
                    has_image = True
                elif part.get("type") == "text":
                    parts.append(part.get("text", ""))
        return has_image, "\n".join(parts)

    def complete(self, role, messages, **params):
        has_image, prompt = self._flatten(messages)
        self.calls.append((role, has_image, prompt))
        if has_image:
            return CompletionResult(
                content=f"OCR_TEXT: {self._next_ocr()}\nSCENE: {STORED_SCENE_TEXT}",
                model="fake-eyes-model:free",
            )
        if role == "brain":
            return CompletionResult(content=CANNED_ANSWER, model="fake-brain-model:free")
        # clarif_eye.synth's fast-path call: same "eyes" role, no image.
        return CompletionResult(content="A jar of jam on a counter.", model="fake-eyes-model:free")

    def close(self):
        pass

    def vision_calls(self):
        return [call for call in self.calls if call[1]]

    def brain_calls(self):
        return [call for call in self.calls if call[0] == "brain"]


class _FakeTtsProvider:
    """Writes a minimal valid-looking mp3 so run_tts's own "looks like
    audio" check passes without touching the network - the same double
    tests/test_graph.py and tests/test_ui.py already use."""

    def synthesize(self, text, out_path):
        with open(out_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 32)


def _resources(client):
    """A REAL checkpointed graph plus a REAL ThreadRegistry - the pairing
    invariant clarif_eye.ui.AppResources documents, and the only
    combination for which thread_configurable actually threads a thread_id
    through to the graph."""
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    return AppResources(
        graph=build_graph(checkpointer=checkpointer),
        client=client,
        client_error=None,
        tts_providers=[_FakeTtsProvider()],
        searcher=None,
        research_client=None,
        thread_registry=ThreadRegistry(checkpointer),
    )


def setup_function(_fn):
    # tts.is_chain_exhausted() reads module-level state left by the last
    # real run_tts() call - reset it so tests don't leak into each other.
    tts_module._last_result_set(None)


def test_follow_up_answers_from_stored_state_with_no_second_vision_call():
    client = RecordingClient()
    resources = _resources(client)
    thread_id = "follow-up-session"

    list(handle_submit_staged(FakeImage(), resources, thread_id=thread_id))
    updates = list(handle_ask_staged(QUESTION, resources, thread_id=thread_id))

    # (b) THE COST ASSERTION: exactly one image-carrying request across BOTH
    # runs - the photo run's. If the follow-up re-ran vision (or the entry
    # node routed it through vision), this count would be 2.
    assert len(client.vision_calls()) == 1, (
        f"expected exactly one vision call across both runs, got {client.vision_calls()}"
    )

    # (a) THE DERIVATION ASSERTION: the follow-up's single brain call was
    # given the ocr text the PHOTO run stored on this thread, plus the typed
    # question. Asserting on the prompt the model RECEIVED is what proves
    # the answer came from checkpointed state rather than from anything the
    # follow-up call itself supplied.
    brain_calls = client.brain_calls()
    assert len(brain_calls) == 1, f"expected exactly one brain call, got {brain_calls}"
    _role, has_image, prompt = brain_calls[0]
    assert has_image is False
    assert STORED_OCR_TEXT in prompt
    assert STORED_SCENE_TEXT in prompt
    assert QUESTION in prompt

    # (c) THE STAGED CONTRACT: same shape as a photo run - an opening
    # "received, working on it" yield, one narration yield per node that has
    # a successor, then status+text with NO audio, then (after the existing
    # delay) the same status+text WITH audio.
    assert all(len(update) == 3 for update in updates)
    assert updates[0][0] == STATUS_ASKING
    assert updates[0][1] is None
    narration = [status for status, _audio, _text in updates]
    assert STATUS_NODE_ANSWERING in narration
    assert STATUS_NODE_TTS in narration
    assert updates[-2][1] is None
    assert updates[-1][1], "the final yield must carry the audio path"
    assert updates[-2][0] == updates[-1][0]
    assert updates[-2][2] == updates[-1][2]
    assert CANNED_ANSWER in updates[-1][2]


def test_a_new_photo_after_a_question_is_not_diverted_into_the_followup_node():
    """A stale question must never eat the next photo.

    `question` is a PLAIN (non-reducer) state key, so it survives on the
    thread's checkpoint after a follow-up. If a photo run's input did not
    seed question=None, clarif_eye.graph.entry_destination would still see
    the previous turn's question and send the new photo straight to
    `followup` - the user would upload a photo and be told about the old
    one. make_initial_state seeding question=None is what prevents that;
    this proves the seeding actually reaches the graph.
    """
    client = RecordingClient()
    resources = _resources(client)
    thread_id = "stale-question-session"

    list(handle_submit_staged(FakeImage(content=b"first-photo"), resources, thread_id=thread_id))
    list(handle_ask_staged(QUESTION, resources, thread_id=thread_id))
    list(handle_submit_staged(FakeImage(content=b"second-photo"), resources, thread_id=thread_id))

    # Two photos means two vision calls. One would mean the second photo was
    # routed to `followup` and never looked at.
    assert len(client.vision_calls()) == 2

    state = resources.graph.get_state({"configurable": {"thread_id": thread_id}})
    assert state.values["question"] is None


def test_a_follow_ups_partial_input_preserves_the_threads_other_stored_keys():
    """The mechanism the whole feature rests on, asserted directly.

    A follow-up passes ONLY {"question": q} to the graph. Every key it does
    NOT mention must survive from the photo run's checkpoint - if LangGraph
    replaced unmentioned scalar keys with empty defaults instead, there
    would be nothing to answer from. Verified empirically before this
    feature was written; pinned here so a langgraph upgrade that changes it
    fails loudly rather than silently turning every follow-up into the
    no-photo-yet message.
    """
    client = RecordingClient()
    resources = _resources(client)
    thread_id = "partial-input-session"
    config = {"configurable": {"thread_id": thread_id}}

    list(handle_submit_staged(FakeImage(), resources, thread_id=thread_id))
    stored_image_data = resources.graph.get_state(config).values["image_data"]
    assert stored_image_data

    list(handle_ask_staged(QUESTION, resources, thread_id=thread_id))

    values = resources.graph.get_state(config).values
    assert STORED_OCR_TEXT in values["ocr_output"]
    assert STORED_SCENE_TEXT in values["scene_context"]
    # image_data too: untouched by a follow-up, which never re-encodes or
    # re-sends the photo.
    assert values["image_data"] == stored_image_data


def test_a_follow_up_records_both_the_question_and_the_answer_as_turns():
    client = RecordingClient()
    resources = _resources(client)
    thread_id = "turn-recording-session"

    list(handle_submit_staged(FakeImage(), resources, thread_id=thread_id))
    list(handle_ask_staged(QUESTION, resources, thread_id=thread_id))

    messages = resources.graph.get_state({"configurable": {"thread_id": thread_id}}).values["messages"]
    # The photo run records one turn (the description); the follow-up
    # records two (the typed question AND the answer) - see
    # clarif_eye.ui._run_followup_events for why the user side is recorded
    # here but not on a photo run.
    assert len(messages) == 3
    assert QUESTION in messages[1].content
    assert CANNED_ANSWER in messages[2].content


def test_a_follow_up_never_reads_or_writes_the_image_cache():
    """The image cache is keyed on photo CONTENT and holds a whole photo's
    result. Two different questions about one photo have different answers,
    so a hit there would be a wrong answer read aloud with confidence."""
    client = RecordingClient()
    resources = _resources(client)
    thread_id = "cache-untouched-session"

    list(handle_submit_staged(FakeImage(), resources, thread_id=thread_id))
    entries_after_photo = dict(resources.image_cache._entries)

    list(handle_ask_staged(QUESTION, resources, thread_id=thread_id))
    list(handle_ask_staged("and what is it made of?", resources, thread_id=thread_id))

    assert dict(resources.image_cache._entries) == entries_after_photo
    # Both questions really did reach the model - neither was served from
    # anything - so the assertion above is about an untouched cache, not
    # about two runs that never happened.
    assert len(client.brain_calls()) == 2


def test_a_blank_question_is_refused_without_running_the_graph():
    """entry_destination routes on the question being non-blank, so a blank
    one would fall through to `vision` and re-run the WHOLE photo pipeline -
    a second vision call the user never asked for. clarif_eye.ui rejects it
    first."""
    from clarif_eye.ui import NO_QUESTION_MESSAGE

    client = RecordingClient()
    resources = _resources(client)
    thread_id = "blank-question-session"

    list(handle_submit_staged(FakeImage(), resources, thread_id=thread_id))
    calls_after_photo = len(client.calls)

    updates = list(handle_ask_staged("   ", resources, thread_id=thread_id))

    assert len(client.calls) == calls_after_photo
    assert updates[-1][2] == NO_QUESTION_MESSAGE


# --- Image-cache hits must keep the thread in step with what was spoken ---
#
# The image cache (issue #75) short-circuits the graph entirely on a repeat
# photo: nothing runs, so nothing writes ocr_output/scene_context onto the
# thread. Before this was fixed, that left the checkpoint describing a
# DIFFERENT photo than the one the user had just been told about - and a
# follow-up answers from the checkpoint. Two ways that goes wrong, both
# reproduced below as they were proven in review.


def test_a_follow_up_after_a_cache_hit_answers_about_the_photo_just_described():
    """Scenario (a): one thread, photo A, then photo B, then photo A again.

    The third submit is a cache hit, so the graph never runs and the
    checkpoint still holds photo B's OCR. The user has just heard photo A
    described. Asking a question then answered about B - the wrong document,
    with no signal to a blind user that anything was amiss.
    """
    client = RecordingClient(ocr_texts=["alpha label text", "bravo label text"])
    resources = _resources(client)
    thread_id = "cache-hit-same-thread"

    list(handle_submit_staged(FakeImage(content=b"photo-a"), resources, thread_id=thread_id))
    list(handle_submit_staged(FakeImage(content=b"photo-b"), resources, thread_id=thread_id))
    # Photo A again: a cache hit, so no third vision call.
    list(handle_submit_staged(FakeImage(content=b"photo-a"), resources, thread_id=thread_id))
    assert len(client.vision_calls()) == 2, "third submit should have been a cache hit"

    list(handle_ask_staged(QUESTION, resources, thread_id=thread_id))

    _role, _has_image, prompt = client.brain_calls()[-1]
    assert "alpha label text" in prompt, (
        "the follow-up must answer about the photo the user was just told "
        "about (A), not whichever photo last ran through the graph (B)"
    )
    assert "bravo label text" not in prompt


def test_a_follow_up_after_a_cache_hit_on_a_fresh_thread_still_has_a_photo():
    """Scenario (b): the cache is process-wide, threads are per-session.

    A second visitor submits a photo the first visitor already sent. They
    get the cached description read aloud - but their OWN thread never ran
    the graph, so it holds nothing. Asking a question then said "no photo
    yet" seconds after a description had been spoken.
    """
    client = RecordingClient(ocr_texts=["alpha label text"])
    resources = _resources(client)

    list(handle_submit_staged(FakeImage(content=b"shared-photo"), resources, thread_id="visitor-one"))
    list(handle_submit_staged(FakeImage(content=b"shared-photo"), resources, thread_id="visitor-two"))
    assert len(client.vision_calls()) == 1, "visitor two should have been a cache hit"

    updates = list(handle_ask_staged(QUESTION, resources, thread_id="visitor-two"))

    final_text = updates[-1][2]
    assert NO_PHOTO_YET_MESSAGE not in final_text, (
        "a visitor who was just read a description must not be told there is "
        "no photo yet"
    )
    assert CANNED_ANSWER in final_text
    _role, _has_image, prompt = client.brain_calls()[-1]
    assert "alpha label text" in prompt


def test_follow_up_before_any_photo_speaks_an_explanation_and_calls_no_model():
    client = RecordingClient()
    resources = _resources(client)

    updates = list(handle_ask_staged(QUESTION, resources, thread_id="thread-with-no-photo-yet"))

    # Not a crash, not a traceback: a spoken-ready explanation, produced
    # through the SAME staged path (so it reaches tts and is actually read
    # aloud), and costing no model call at all.
    assert client.calls == [], f"a question with no photo must cost no model call, got {client.calls}"
    final_status, final_audio, final_text = updates[-1]
    assert NO_PHOTO_YET_MESSAGE in final_text
    assert final_audio, "the explanation must be spoken, not text-only"
    assert updates[0][0] == STATUS_ASKING
