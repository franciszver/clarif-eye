"""Tests for verifying numbers in follow-up ANSWERS (issue #92 / P9.11).

RED FIRST: at the time this file was first committed, clarif_eye.graph had
no `verify_answer` node, no `followup_destination` and no
ANSWER_RETAKE_CONFIRMATION, and clarif_eye.followup never called
clarif_eye.verification at all - so every test here failed on the import
line or on the first "did it pause?" assertion.

WHAT THIS ISSUE SETTLES. A follow-up is the highest-stakes place this app
reads a number aloud: the user asked for the expiry date, the total, the
dosage. Until now only the deep-analysis path checked drafted numbers
against the photographed text; clarif_eye.followup documented why it
skipped the check (a correctly REFORMATTED date would be rejected by a
loose token-equality check, turning a right answer into a refusal) and
deferred the decision to this issue, to be settled once #83's ask-first
mechanism existed. It does now, so the trade-off has changed: a false
positive costs ONE clarifying question instead of a wrong refusal. That is
the whole argument, and
`test_a_correctly_reformatted_number_becomes_a_question_never_a_refusal`
below is the test that pins the half of it the issue calls out by name.

THE PRODUCT RULE IS UNCHANGED (issue #83, restated for this path): the run
pauses ONLY on FAILED number verification. Never on general low confidence,
never on the fast path, never on a follow-up whose numbers all trace back.
`test_a_clean_follow_up_answer_never_pauses` is written to go RED the
moment that gate stops gating.

Same no-network discipline as tests/test_followup.py and
tests/test_ask_before_speaking.py: real compiled graph, real checkpointer,
real ThreadRegistry, fake client, fake TTS provider. Nothing here launches
Gradio or opens a socket.
"""

from langgraph.checkpoint.memory import InMemorySaver

from clarif_eye import tts as tts_module
from clarif_eye.client import CompletionResult
from clarif_eye.followup import run_followup
from clarif_eye.graph import (
    ANSWER_RETAKE_CONFIRMATION,
    INTERRUPT_CHUNK_KEY,
    RESUME_CONTINUE,
    RESUME_RETAKE,
    RETAKE_CONFIRMATION,
    UNVERIFIED_NUMBER_CAVEAT,
    VERIFY_ANSWER_NODE,
    build_graph,
    followup_destination,
    next_node_after,
)
from clarif_eye.ui import (
    ASK_BUTTON_ELEM_ID,
    AppResources,
    NOTHING_TO_RESUME_MESSAGE,
    QUESTION_PENDING_MESSAGE,
    RESUME_CONTINUE_LABEL,
    RESUME_RETAKE_LABEL,
    STATUS_ASKING,
    ThreadRegistry,
    _PauseSignal,
    build_interface,
    handle_ask_staged,
    handle_resume_staged,
    handle_submit_staged,
)

from tests.test_ui import FakeImage

# The stored photo. No currency, few digits and no document keywords, so
# clarif_eye.router keeps the PHOTO run on the fast path (vision ->
# fast_synth -> tts) - this file is about the FOLLOW-UP's numbers, and a
# deep-path photo run would drag the deep path's own asking node into every
# assertion about which node paused.
STORED_OCR_TEXT = "STRAWBERRY JAM best before 19 April 2027"
STORED_SCENE_TEXT = "a jar of jam on a kitchen counter"

# The same photo, with the date printed the way a real label often prints
# it. Used only by the reformatted-number test below.
SHORT_DATE_OCR_TEXT = "STRAWBERRY JAM best before 19/04/27"

QUESTION = "how much does the jar hold?"

# 500 appears NOWHERE in the stored photo text - this is the fabricated
# amount a follow-up could previously read aloud with no check at all.
INVENTED_ANSWER = "The jar holds 500 grams."
# Every number here traces back to STORED_OCR_TEXT, so this one must sail
# straight through to speech with no question asked.
HONEST_ANSWER = "The label says best before 19 April 2027."
# THE FALSE POSITIVE THE ISSUE IS ABOUT: read against SHORT_DATE_OCR_TEXT
# ("19/04/27") this is a CORRECT answer that reformats a two-digit year into
# a four-digit one. Token equality cannot see that "2027" and "27" are the
# same year, so the check fails - and the whole point of #92 is that the
# outcome of that failure is a QUESTION, not a refusal.
REFORMATTED_ANSWER = "It is best before 19 April 2027."


class RecordingClient:
    """Records every complete() call and answers follow-ups with a canned
    reply.

    A VISION call is identified STRUCTURALLY, by an {"type": "image_url"}
    content part in the request (exactly what clarif_eye.vision._build_messages
    puts there), never by role name - clarif_eye.synth calls the same "eyes"
    role without an image.
    """

    def __init__(self, brain_reply, ocr_text=STORED_OCR_TEXT):
        self.brain_reply = brain_reply
        self.ocr_text = ocr_text
        self.calls = []

    @staticmethod
    def _flatten(messages):
        has_image = False
        parts = []
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
                content=f"OCR_TEXT: {self.ocr_text}\nSCENE: {STORED_SCENE_TEXT}",
                model="fake-eyes-model:free",
            )
        if role == "brain":
            return CompletionResult(content=self.brain_reply, model="fake-brain-model:free")
        # clarif_eye.synth's fast-path call: same "eyes" role, no image.
        return CompletionResult(content="A jar of jam on a counter.", model="fake-eyes-model:free")

    def close(self):
        pass

    def vision_calls(self):
        return [call for call in self.calls if call[1]]

    def brain_calls(self):
        return [call for call in self.calls if call[0] == "brain" and not call[1]]


class _FakeTtsProvider:
    """Writes a minimal valid-looking mp3 so run_tts's own "looks like
    audio" check passes without touching the network - the same double
    tests/test_followup.py and tests/test_ask_before_speaking.py use."""

    def synthesize(self, text, out_path):
        with open(out_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 32)


def _resources(client):
    """A REAL checkpointed graph plus a REAL ThreadRegistry - the pairing
    invariant clarif_eye.ui.AppResources documents."""
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


def _staged(updates):
    return [(status, audio, text) for status, audio, text in updates]


def _state_of(resources, thread_id):
    return resources.graph.get_state({"configurable": {"thread_id": thread_id}})


def _photo_then_question(resources, thread_id, question=QUESTION):
    """The two runs every test here starts from: describe a photo, then ask
    a question about it on the same thread."""
    list(handle_submit_staged(FakeImage(), resources, thread_id=thread_id))
    signal = _PauseSignal()
    updates = _staged(
        handle_ask_staged(question, resources, thread_id=thread_id, pause_signal=signal)
    )
    return updates, signal


# --- The node module: the check, and what it hands forward ----------------


def test_run_followup_holds_an_unverifiable_answer_for_the_asker():
    """run_followup must hand the questioned answer FORWARD in state, the
    same way clarif_eye.analysis.run_analysis already does - not speak it,
    and not throw it away either.

    THE HAYSTACK IS WHAT THE MODEL WAS SHOWN: the stored ocr_output and
    scene_context, and nothing else. `scraper_data` is passed as "" because
    the follow-up prompt does not include a web scrape (see
    clarif_eye.followup._build_messages), and the user's QUESTION is
    deliberately not part of it either - see this file's ruling test below.
    """
    client = RecordingClient(INVENTED_ANSWER)
    result = run_followup(STORED_OCR_TEXT, STORED_SCENE_TEXT, QUESTION, client)

    assert "500" not in result["final_output"]
    hold = result["verification_hold"]
    assert hold["script"] == INVENTED_ANSWER
    assert hold["numbers"] == ["500"]


def test_a_verified_follow_up_answer_holds_nothing():
    client = RecordingClient(HONEST_ANSWER)
    result = run_followup(STORED_OCR_TEXT, STORED_SCENE_TEXT, QUESTION, client)

    assert result["verification_hold"] is None
    assert result["output_degraded"] is False
    assert HONEST_ANSWER in result["final_output"]


def test_a_number_the_user_typed_is_not_treated_as_verified():
    """THE RULING (issue #92): the user's own question is NOT part of the
    haystack.

    "is it 200 mg?" is a legitimate thing to ask, and "yes, 200 mg" is a
    legitimate answer - but only if the photo actually says 200. Letting a
    user-typed number verify itself would let a wrong guess launder itself
    into a confident-sounding confirmation, read aloud to someone who cannot
    check it. The haystack stays "what the camera saw", which is the same
    rule clarif_eye.analysis applies. The cost is one question, which is
    exactly the cost this issue decided was acceptable.
    """
    client = RecordingClient("Yes, it is 200 milligrams.")
    result = run_followup(STORED_OCR_TEXT, STORED_SCENE_TEXT, "is it 200 mg?", client)

    assert result["verification_hold"] is not None
    assert result["verification_hold"]["numbers"] == ["200"]


# --- The graph: topology, the pause, and the two ways out ------------------


def test_the_parent_registers_the_asking_node_under_its_own_name():
    """The parent's asking node may not be called `verify_numbers`: that is
    the deep-path CHILD's node name, and clarif_eye.graph's
    _UNCONDITIONAL_SUCCESSOR and clarif_eye.ui's _NODE_PHRASE are flat dicts
    keyed by BARE node names across both graphs (see
    tests/test_deep_path_subgraph.py's disjointness test)."""
    graph = build_graph()

    assert VERIFY_ANSWER_NODE in graph.nodes
    assert "verify_numbers" not in graph.nodes


def test_followup_destination_is_the_gate_and_next_node_after_agrees():
    """The narration must not announce speech for a run that is about to
    stop and ask - clarif_eye.ui reads next_node_after, so it has to resolve
    the same branch the graph itself takes."""
    held = {"verification_hold": {"script": INVENTED_ANSWER, "numbers": ["500"]}}
    clean = {"verification_hold": None}

    assert followup_destination(held) == VERIFY_ANSWER_NODE
    assert followup_destination(clean) == "tts"
    assert next_node_after("followup", held) == VERIFY_ANSWER_NODE
    assert next_node_after("followup", clean) == "tts"
    assert next_node_after(VERIFY_ANSWER_NODE, held) == "tts"


def test_an_unverifiable_follow_up_answer_pauses_and_carries_the_questioned_text():
    resources = _resources(RecordingClient(INVENTED_ANSWER))
    updates, signal = _photo_then_question(resources, "followup-pause")

    assert signal.paused is True
    status, audio, text = updates[-1]
    # No audio: the question is spoken by the screen reader through the
    # aria-live status region, not by TTS.
    assert audio is None
    assert status == text, "the question must also be readable in the result box"
    # It must say what was drafted and which number could not be checked.
    for expected in (INVENTED_ANSWER, "500"):
        assert expected in status, f"{expected!r} missing from the spoken question"
    assert RESUME_CONTINUE_LABEL in status
    # The narration ran normally up to the pause.
    assert updates[0][0] == STATUS_ASKING

    snapshot = _state_of(resources, "followup-pause")
    assert snapshot.next == (VERIFY_ANSWER_NODE,)
    assert snapshot.interrupts, "graph state shows no pending interrupt"

    payload = snapshot.interrupts[0].value
    # THE SAME PAYLOAD SHAPE the deep path uses, so every part of the UI
    # flow (staging, buttons, resume, refusals) works unchanged. Only
    # `reason` distinguishes the two flows, structurally.
    assert set(payload) == {"reason", "script", "numbers"}
    assert payload["reason"] == "unverified_answer"
    assert payload["script"] == INVENTED_ANSWER
    assert payload["numbers"] == ["500"]

    # A paused run records nothing: only the PHOTO run's own description is
    # in history, and neither side of the unanswered question is.
    recorded = [message.content for message in snapshot.values["messages"]]
    assert len(recorded) == 1
    assert QUESTION not in recorded[0]
    assert INVENTED_ANSWER not in recorded[0]


def test_a_clean_follow_up_answer_never_pauses():
    """THE PRODUCT RULE: an answer whose numbers all trace back must be
    spoken with no question asked.

    MUTATION TARGET: removing the gate (making
    clarif_eye.graph.numbers_need_asking return True unconditionally, or
    routing every follow-up through the asking node) must turn this RED.
    """
    resources = _resources(RecordingClient(HONEST_ANSWER))
    updates, signal = _photo_then_question(
        resources, "followup-clean", "what is the expiry date?"
    )

    assert signal.paused is False
    _status, audio, text = updates[-1]
    assert audio, "a verified answer must be spoken as audio"
    assert HONEST_ANSWER in text

    snapshot = _state_of(resources, "followup-clean")
    assert not snapshot.interrupts
    assert snapshot.next == ()
    assert snapshot.values["verification_hold"] is None
    # The photo's description, then the question and its answer.
    assert len(snapshot.values["messages"]) == 3


def test_a_correctly_reformatted_number_becomes_a_question_never_a_refusal():
    """THE ISSUE'S SECOND DONE-WHEN, and the reason #82 deferred this check.

    The photo says "19/04/27"; the answer says "19 April 2027". That answer
    is CORRECT, and the loose token-equality check cannot see it - "2027"
    does not appear in the photo text. Before the ask-first mechanism
    existed, wiring the check in would have turned this right answer into a
    refusal. Now it costs one question, and the user can hear the answer.
    """
    resources = _resources(
        RecordingClient(REFORMATTED_ANSWER, ocr_text=SHORT_DATE_OCR_TEXT)
    )
    updates, signal = _photo_then_question(
        resources, "followup-reformatted", "what is the expiry date?"
    )

    assert signal.paused is True
    _status, _audio, text = updates[-1]
    # A QUESTION, carrying the answer the user can then choose to hear -
    # not a flat "this could not be verified" refusal with the answer
    # thrown away.
    assert REFORMATTED_ANSWER in text
    assert "2027" in text
    assert "not safe to read aloud" not in text
    assert RESUME_CONTINUE_LABEL in text and RESUME_RETAKE_LABEL in text

    resumed = _staged(
        handle_resume_staged(RESUME_CONTINUE, resources, thread_id="followup-reformatted")
    )
    _status, audio, spoken = resumed[-1]
    assert audio, "the user asked to hear it, so it must be spoken"
    assert REFORMATTED_ANSWER in spoken


def test_resume_continue_speaks_the_answer_with_the_caveat_and_records_it():
    resources = _resources(RecordingClient(INVENTED_ANSWER))
    _photo_then_question(resources, "followup-continue")

    updates = _staged(
        handle_resume_staged(RESUME_CONTINUE, resources, thread_id="followup-continue")
    )

    _status, audio, text = updates[-1]
    assert audio, "the caveated answer must be spoken as audio"
    assert INVENTED_ANSWER in text
    # THE SAME CAVEAT the photo path uses - the user hears the warning
    # BEFORE the number it is about, whichever flow asked.
    assert UNVERIFIED_NUMBER_CAVEAT in text
    # The staged audio-delay contract: the same status/text without audio,
    # then again with it.
    assert updates[-2] == (_status, None, text)

    snapshot = _state_of(resources, "followup-continue")
    assert snapshot.values["verification_hold"] is None
    assert not snapshot.interrupts
    # output_degraded=False on a continued answer (issue #93 / P9.12), so
    # the turn IS recorded: the photo's description, then this answer.
    recorded = [message.content for message in snapshot.values["messages"]]
    assert recorded[-1] == text


def test_resume_retake_speaks_honest_wording_for_a_follow_up_and_records_nothing():
    """THE RESUME-WORDING DECISION (issue #92): same two buttons, same
    payload shape, DIFFERENT spoken confirmation.

    On the photo path, "take a new photo" is the honest next step - the
    drafted DESCRIPTION was unverifiable, so there is nothing else to do
    with that photo. On a follow-up the photo is fine; it is the ANSWER that
    could not be checked, and telling the user to re-photograph something
    they photographed correctly would send them off to redo work for no
    reason. So the retake button (whose label still reads as "never mind"
    for this flow) speaks ANSWER_RETAKE_CONFIRMATION instead.
    """
    resources = _resources(RecordingClient(INVENTED_ANSWER))
    _photo_then_question(resources, "followup-retake")

    updates = _staged(
        handle_resume_staged(RESUME_RETAKE, resources, thread_id="followup-retake")
    )

    _status, audio, text = updates[-1]
    assert audio, "the confirmation must be spoken, not silence"
    assert text == ANSWER_RETAKE_CONFIRMATION
    assert text != RETAKE_CONFIRMATION, (
        "a follow-up retake must not tell the user to re-photograph a photo "
        "that was never the problem"
    )
    assert "500" not in text

    snapshot = _state_of(resources, "followup-retake")
    assert snapshot.next == ()
    assert not snapshot.interrupts
    assert snapshot.values["verification_hold"] is None
    # A declined answer is not an answer: only the photo's own description
    # is in history.
    assert len(snapshot.values["messages"]) == 1

    # THE READY ASSERTION: the thread still holds the photo, so the user can
    # simply ask again - which is what the honest wording tells them to do.
    resources.client.brain_reply = HONEST_ANSWER
    again = _staged(
        handle_ask_staged(
            "what is the expiry date?", resources, thread_id="followup-retake"
        )
    )
    assert HONEST_ANSWER in again[-1][2]


def test_an_answer_that_is_not_continue_is_treated_as_a_retake():
    """Anything this app did not send must never be read as consent to
    speak an unverified number - on this path either."""
    resources = _resources(RecordingClient(INVENTED_ANSWER))
    _photo_then_question(resources, "followup-garbled")

    updates = _staged(
        handle_resume_staged("something else entirely", resources, thread_id="followup-garbled")
    )

    assert updates[-1][2] == ANSWER_RETAKE_CONFIRMATION


def test_resuming_a_follow_up_pause_does_not_call_the_brain_model_again():
    """LangGraph re-executes the whole interrupted NODE on resume, so the
    asking node must sit AFTER the model call, not inside `followup`."""
    resources = _resources(RecordingClient(INVENTED_ANSWER))
    _photo_then_question(resources, "followup-cost")
    assert len(resources.client.brain_calls()) == 1

    list(handle_resume_staged(RESUME_CONTINUE, resources, thread_id="followup-cost"))

    assert len(resources.client.brain_calls()) == 1, (
        "resuming re-ran the brain model - the interrupt must sit AFTER the "
        f"model call: {resources.client.brain_calls()}"
    )


# --- The pause-lifecycle matrix, driven against a FOLLOW-UP pause ----------
#
# A pending question is a SAFETY question whichever flow raised it, so every
# collision the photo path already has a decided answer for must behave
# identically here (see tests/test_ask_before_speaking.py's own matrix).


def test_a_follow_up_typed_while_a_follow_up_question_is_pending_is_refused():
    resources = _resources(RecordingClient(INVENTED_ANSWER))
    _photo_then_question(resources, "followup-ask-while-paused")
    calls_before = len(resources.client.calls)

    updates = _staged(
        handle_ask_staged(
            "and what colour is it?", resources, thread_id="followup-ask-while-paused"
        )
    )

    _status, audio, text = updates[-1]
    assert audio is None
    assert text == QUESTION_PENDING_MESSAGE
    assert RESUME_CONTINUE_LABEL in text and RESUME_RETAKE_LABEL in text
    # It cost nothing: no graph run, so no model call.
    assert len(resources.client.calls) == calls_before

    # THE PAUSE SURVIVED and is still answerable, which is the point of
    # refusing rather than letting the new question supersede it.
    snapshot = _state_of(resources, "followup-ask-while-paused")
    assert snapshot.next == (VERIFY_ANSWER_NODE,)
    assert snapshot.interrupts

    resumed = _staged(
        handle_resume_staged(
            RESUME_CONTINUE, resources, thread_id="followup-ask-while-paused"
        )
    )
    assert UNVERIFIED_NUMBER_CAVEAT in resumed[-1][2]


def test_a_cached_photo_submitted_while_a_follow_up_question_is_pending_resolves_it():
    """The IMPLICIT RETAKE, against a follow-up pause: submitting another
    photo means the user moved on, and the cache-hit branch must resolve the
    pending question rather than leaving a zombie thread (interrupts
    cleared, .next still naming the paused node, buttons wired to nothing).
    """
    resources = _resources(RecordingClient(INVENTED_ANSWER))
    photo = FakeImage(content=b"jam-photo-bytes")

    # 1. Describe the photo (this also puts it in the image cache) and ask a
    #    question whose answer cannot be verified, so the thread pauses.
    list(handle_submit_staged(photo, resources, thread_id="followup-hit-while-paused"))
    list(handle_ask_staged(QUESTION, resources, thread_id="followup-hit-while-paused"))
    assert _state_of(resources, "followup-hit-while-paused").interrupts, "setup: should have paused"
    vision_calls_before = len(resources.client.vision_calls())

    # 2. The user gives up on the question and submits the same photo again.
    signal = _PauseSignal()
    updates = _staged(
        handle_submit_staged(
            photo, resources, thread_id="followup-hit-while-paused", pause_signal=signal
        )
    )

    # Served from the cache - no second vision call.
    assert len(resources.client.vision_calls()) == vision_calls_before
    assert updates[-1][1], "the cached audio must still be served"
    assert signal.paused is False, "the buttons must end hidden"

    # NO ZOMBIE: nothing pending, nothing half-resumable.
    snapshot = _state_of(resources, "followup-hit-while-paused")
    assert snapshot.next == ()
    assert not snapshot.interrupts
    assert snapshot.values["verification_hold"] is None
    assert snapshot.values["question"] is None

    # A late click on a button that is now hidden is answered in words.
    late = _staged(
        handle_resume_staged(
            RESUME_CONTINUE, resources, thread_id="followup-hit-while-paused"
        )
    )
    assert late[-1][2] == NOTHING_TO_RESUME_MESSAGE


def test_a_fresh_photo_submitted_while_a_follow_up_question_is_pending_leaves_nothing_pending():
    """The non-cached sibling: LangGraph supersedes the pending task itself
    here, but both branches must end in the same place and that has to be
    pinned rather than assumed."""
    resources = _resources(RecordingClient(INVENTED_ANSWER))

    list(handle_submit_staged(FakeImage(content=b"first-jam"), resources, thread_id="followup-fresh"))
    list(handle_ask_staged(QUESTION, resources, thread_id="followup-fresh"))
    assert _state_of(resources, "followup-fresh").interrupts, "setup: should have paused"

    signal = _PauseSignal()
    updates = _staged(
        handle_submit_staged(
            FakeImage(content=b"second-jam"),
            resources,
            thread_id="followup-fresh",
            pause_signal=signal,
        )
    )

    assert signal.paused is False
    assert updates[-1][1], "the new photo must be described and spoken"
    snapshot = _state_of(resources, "followup-fresh")
    assert snapshot.next == ()
    assert not snapshot.interrupts
    assert snapshot.values["verification_hold"] is None


def test_the_ask_handler_reveals_the_answer_buttons_but_never_hides_them():
    """The Gradio wiring, read off the BUILT Blocks rather than the source.

    Two halves, and the second is the one an obvious implementation gets
    wrong. A follow-up that pauses must REVEAL the two answer buttons, or a
    user who cannot see the screen is asked a question with nothing on the
    page to answer it with. But a follow-up typed while a question is
    already pending is REFUSED - and that yield must leave the buttons
    alone, because hiding them would take away the only way to answer the
    question the refusal has just told the user to answer.
    """
    resources = _resources(RecordingClient(INVENTED_ANSWER))
    demo = build_interface(resources)
    try:
        question_id = next(
            block_id
            for block_id, block in demo.blocks.items()
            if getattr(block, "elem_id", None) == ASK_BUTTON_ELEM_ID
        )
        ask_handler = next(
            fn.fn
            for fn in demo.fns.values()
            if (question_id, "click") in (fn.targets or []) and fn.fn is not None
        )

        list(handle_submit_staged(FakeImage(), resources, thread_id="ui-wiring"))

        paused = list(ask_handler(QUESTION, "ui-wiring", "ui-wiring-session"))
        _status, _audio, _text, continue_update, retake_update = paused[-1]
        assert dict(continue_update).get("visible") is True
        assert dict(retake_update).get("visible") is True

        refused = list(ask_handler("what colour is it?", "ui-wiring", "ui-wiring-session"))
        _status, _audio, text, continue_update, retake_update = refused[-1]
        assert text == QUESTION_PENDING_MESSAGE, "setup: the refusal branch"
        # An empty update carries no `visible` key at all, which is what
        # "leave them exactly as they are" looks like on the wire.
        assert "visible" not in dict(continue_update)
        assert "visible" not in dict(retake_update)
    finally:
        demo.close()


def test_the_asking_node_never_runs_when_no_photo_has_been_described():
    """A degraded follow-up (no photo yet) writes no hold, so it cannot
    divert into the asking node - the interrupt discipline holds for every
    degradation branch, not only the success one."""
    resources = _resources(RecordingClient(INVENTED_ANSWER))

    signal = _PauseSignal()
    updates = _staged(
        handle_ask_staged(QUESTION, resources, thread_id="followup-no-photo", pause_signal=signal)
    )

    assert signal.paused is False
    assert updates[-1][1], "the explanation must be spoken"
    assert INTERRUPT_CHUNK_KEY not in updates[-1][2]
    assert _state_of(resources, "followup-no-photo").values.get("verification_hold") is None
