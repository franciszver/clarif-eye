"""Tests for asking before speaking an unverifiable number (issue #83 / P9.4).

RED FIRST (original commit): at the time this file was first committed,
none of the things it reaches for existed - clarif_eye.graph had no
`verify_numbers` node, no INTERRUPT_CHUNK_KEY and no resume answers,
ClarifEyeState had no `verification_hold` key, clarif_eye.verification had
no `unverified_numbers`, and clarif_eye.ui had no handle_resume_staged - so
every test here failed on the import line.

THE PRODUCT RULE THESE TESTS PIN (from the owner, stated in issue #83): the
run pauses ONLY when the number-verification the deep-analysis path already
performs FAILS. Never on general low confidence, never on the fast path,
never on a follow-up answer. An unnecessary question costs this audience
more than most, so `test_verified_numbers_never_interrupt` below is written
to go RED the moment the verification-failure gate stops gating.

WHAT MUST NOT COST TWICE: resuming re-executes the interrupted NODE from
its start (verified empirically on langgraph 1.2.10 before this was
designed - see clarif_eye.graph.verify_numbers_node's docstring). If the
interrupt sat inside `analysis`, the brain model call ahead of it would run
a second time on every resume: real money on a rate-limited free tier, ~20
extra seconds for a blind user already waiting, and a second, possibly
DIFFERENT draft answering a question the user was asked about the first
one. `test_resume_does_not_call_the_brain_model_a_second_time` is the
assertion that pins that; it is written to go red if the interrupt is ever
moved back in front of a model call.

Same no-network discipline as tests/test_ui.py, tests/test_graph.py and
tests/test_followup.py: real compiled graph, real checkpointer, real
ThreadRegistry, fake client, fake searcher, fake TTS provider. Nothing here
launches Gradio or opens a socket.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from clarif_eye import tts as tts_module
from clarif_eye.analysis import run_analysis
from clarif_eye.client import CompletionResult
from clarif_eye.graph import (
    INTERRUPT_CHUNK_KEY,
    RESUME_CONTINUE,
    RESUME_RETAKE,
    RETAKE_CONFIRMATION,
    UNVERIFIED_NUMBER_CAVEAT,
    build_graph,
)
from clarif_eye.state import make_initial_state
from clarif_eye.ui import (
    AUDIO_PLAY_DELAY_MS,
    AppResources,
    NOTHING_TO_RESUME_MESSAGE,
    QUESTION_PENDING_MESSAGE,
    RESUME_CONTINUE_BUTTON_ELEM_ID,
    RESUME_CONTINUE_LABEL,
    RESUME_RETAKE_BUTTON_ELEM_ID,
    RESUME_RETAKE_LABEL,
    STATUS_RESUMING,
    STATUS_WORKING,
    ThreadRegistry,
    _PauseSignal,
    _trim_thread_to_latest_checkpoint,
    build_interface,
    handle_ask_staged,
    handle_resume_staged,
    handle_submit_staged,
)
from clarif_eye.verification import unverified_numbers

from tests._stream_helpers import drain_stream_collecting_trace
from tests.test_ui import FakeImage

# A dense document: currency, a long identifier, a date, and document
# keywords, so clarif_eye.router scores it COMPLEX and the run takes the
# research -> analysis path. The analysis path is the only one that verifies
# numbers at all (see clarif_eye.verification's module docstring), so it is
# the only one that can pause.
BILL_OCR = (
    "CITY OF RIVERTON WATER UTILITY STATEMENT Account Number: 4471-2205-88 "
    "AMOUNT DUE $104.95 PAYMENT DUE BY: 22 JULY 2026"
)
BILL_SCENE = "a water utility statement"

# $999.99 appears NOWHERE in the inputs - this is the fabricated amount the
# whole feature exists to refuse to speak unasked.
INVENTED_DRAFT = "This is a water utility bill. The amount due is $999.99."
# Every number here traces back to BILL_OCR, so this one must sail straight
# through to speech with no question asked.
HONEST_DRAFT = "This is a water utility bill. The amount due is $104.95, due by 22 JULY 2026."


class RecordingClient:
    """Fake OpenRouter client that records every call it receives.

    A VISION call is identified STRUCTURALLY, by an {"type": "image_url"}
    content part in the request (exactly what clarif_eye.vision._build_messages
    puts there), never by role name - clarif_eye.synth calls the same "eyes"
    role without an image. `brain_calls` is what the no-double-model-call
    assertion counts.
    """

    def __init__(self, brain_reply):
        self.brain_reply = brain_reply
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
                content=f"OCR_TEXT: {BILL_OCR}\nSCENE: {BILL_SCENE}",
                model="fake-eyes-model:free",
            )
        if role == "brain":
            return CompletionResult(content=self.brain_reply, model="fake-brain-model:free")
        return CompletionResult(content="A bill on a table.", model="fake-eyes-model:free")

    def close(self):
        pass

    def brain_calls(self):
        return [call for call in self.calls if call[0] == "brain" and not call[1]]


class FakeSearcher:
    """Returns no results, so research_node degrades to "found nothing"
    without ever touching the network. The research step is not what these
    tests are about - it is just on the road to `analysis`."""

    def text(self, query, **kwargs):
        return []


class _FakeTtsProvider:
    """Writes a minimal valid-looking mp3 so run_tts's own "looks like
    audio" check passes without touching the network - the same double
    tests/test_graph.py, test_ui.py and test_followup.py already use."""

    def synthesize(self, text, out_path):
        with open(out_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 32)


def setup_function(_fn):
    # tts.is_chain_exhausted() reads module-level state left by the last
    # real run_tts() call - reset it so tests don't leak into each other.
    tts_module._last_result_set(None)


def _graph_and_config(thread_id="interrupt-thread"):
    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    return graph, checkpointer, {"configurable": {"thread_id": thread_id}}


def _run_config(client, thread_id="interrupt-thread"):
    return {
        "configurable": {
            "client": client,
            "searcher": FakeSearcher(),
            "tts_provider": _FakeTtsProvider(),
            "thread_id": thread_id,
        }
    }


def _resources(client, thread_id_unused=None):
    """A REAL checkpointed graph plus a REAL ThreadRegistry - the pairing
    invariant clarif_eye.ui.AppResources documents. `searcher` is a fake so
    the research path never opens a socket."""
    checkpointer = InMemorySaver()
    return AppResources(
        graph=build_graph(checkpointer=checkpointer),
        client=client,
        client_error=None,
        tts_providers=[_FakeTtsProvider()],
        searcher=FakeSearcher(),
        research_client=None,
        thread_registry=ThreadRegistry(checkpointer),
    )


# --- The verification primitive: WHICH numbers failed, structurally --------


def test_unverified_numbers_names_the_tokens_that_do_not_trace_back():
    """The payload must carry the failing numbers STRUCTURALLY - a list of
    tokens - so nothing downstream has to parse them back out of prose."""
    failing = unverified_numbers(INVENTED_DRAFT, BILL_OCR, BILL_SCENE, "")

    # The ORIGINAL spoken form, currency symbol and all - this token is
    # read aloud back to the user, so it should sound like what they were
    # about to be told, not like a normalised comparison key.
    assert failing == ["$999.99"]
    assert unverified_numbers(HONEST_DRAFT, BILL_OCR, BILL_SCENE, "") == []


def test_analysis_holds_the_draft_and_the_failing_numbers_for_the_asker():
    """run_analysis must hand the questioned text FORWARD in state, not
    throw it away.

    The `final_output` it writes is UNCHANGED (still the safe "could not be
    verified" script, so a graph without the asking node degrades exactly as
    it did before this issue) - what is new is `verification_hold`, which
    carries the drafted script and the failing tokens so a LATER node can
    ask about them without re-running the brain model.
    """
    client = RecordingClient(INVENTED_DRAFT)
    result = run_analysis(BILL_OCR, BILL_SCENE, "", client)

    assert "999.99" not in result["final_output"]
    assert "could not be verified" in result["final_output"]

    hold = result["verification_hold"]
    assert hold["script"] == INVENTED_DRAFT
    assert hold["numbers"] == ["$999.99"]


def test_a_verified_analysis_reply_holds_nothing():
    client = RecordingClient(HONEST_DRAFT)
    result = run_analysis(BILL_OCR, BILL_SCENE, "", client)

    assert result["verification_hold"] is None
    assert "$104.95" in result["final_output"]


# --- The graph: pause, and the two ways out -------------------------------


def test_unverifiable_number_pauses_the_run_and_carries_the_questioned_text():
    graph, _checkpointer, config = _graph_and_config()
    client = RecordingClient(INVENTED_DRAFT)
    config["configurable"].update(_run_config(client)["configurable"])

    chunks = list(
        graph.stream(make_initial_state("base64photo"), config=config, stream_mode="updates")
    )

    keys = [key for chunk in chunks for key in chunk]
    assert INTERRUPT_CHUNK_KEY in keys, f"run did not pause: {keys}"
    assert "tts" not in keys, "a paused run must not reach speech"

    snapshot = graph.get_state(config)
    assert snapshot.next == ("verify_numbers",)
    assert snapshot.interrupts, "graph state shows no pending interrupt"

    payload = snapshot.interrupts[0].value
    # STRUCTURAL, not prose: the questioned script and the failing tokens
    # travel as their own fields so the UI never parses a sentence.
    assert payload["reason"] == "unverified_numbers"
    assert payload["script"] == INVENTED_DRAFT
    assert payload["numbers"] == ["$999.99"]


def test_resume_continue_speaks_the_drafted_script_with_a_spoken_caveat():
    graph, _checkpointer, config = _graph_and_config()
    client = RecordingClient(INVENTED_DRAFT)
    config["configurable"].update(_run_config(client)["configurable"])
    list(graph.stream(make_initial_state("base64photo"), config=config, stream_mode="updates"))

    chunks = list(graph.stream(Command(resume=RESUME_CONTINUE), config=config, stream_mode="updates"))

    keys = [key for chunk in chunks for key in chunk]
    assert "tts" in keys, f"resume did not reach speech: {keys}"

    snapshot = graph.get_state(config)
    final_output = snapshot.values["final_output"]
    # The user asked to hear it, so the drafted script IS spoken - with an
    # honest caveat attached, never silently.
    assert INVENTED_DRAFT in final_output
    assert UNVERIFIED_NUMBER_CAVEAT in final_output
    assert snapshot.values["audio_file_path"]
    # The run is over and the thread holds nothing pending.
    assert snapshot.next == ()
    assert not snapshot.interrupts
    assert snapshot.values["verification_hold"] is None


def test_resume_retake_ends_clean_and_leaves_the_thread_ready_for_a_new_photo():
    graph, _checkpointer, config = _graph_and_config()
    client = RecordingClient(INVENTED_DRAFT)
    config["configurable"].update(_run_config(client)["configurable"])
    list(graph.stream(make_initial_state("base64photo"), config=config, stream_mode="updates"))

    list(graph.stream(Command(resume=RESUME_RETAKE), config=config, stream_mode="updates"))

    snapshot = graph.get_state(config)
    # Spoken confirmation, not silence - and the fabricated amount is gone.
    assert snapshot.values["final_output"] == RETAKE_CONFIRMATION
    assert "999.99" not in snapshot.values["final_output"]
    assert snapshot.values["audio_file_path"]
    # Nothing half-written: no pending task, no held draft, ready state.
    assert snapshot.next == ()
    assert not snapshot.interrupts
    assert snapshot.values["verification_hold"] is None

    # THE READY ASSERTION: a fresh photo run on this SAME thread works and
    # is not diverted by anything the abandoned run left behind.
    honest_client = RecordingClient(HONEST_DRAFT)
    fresh_config = _run_config(honest_client)
    result, trace = drain_stream_collecting_trace(
        graph, make_initial_state("second-photo"), fresh_config
    )
    # Straight past the asking node: this photo's numbers all check out,
    # so the conditional edge out of `analysis` never enters it.
    assert trace == ["entry", "vision", "research", "analysis", "tts"]
    assert "$104.95" in result["final_output"]


def test_verified_numbers_never_interrupt():
    """THE PRODUCT RULE (issue #83): a clean run must never ask anything.

    MUTATION TARGET: removing the verification-failure gate (making
    clarif_eye.graph.numbers_need_asking return True unconditionally, so
    every run routes into the asking node and stops) must turn this test
    RED.
    """
    graph, _checkpointer, config = _graph_and_config("clean-thread")
    client = RecordingClient(HONEST_DRAFT)
    config["configurable"].update(_run_config(client, "clean-thread")["configurable"])

    chunks = list(
        graph.stream(make_initial_state("base64photo"), config=config, stream_mode="updates")
    )

    keys = [key for chunk in chunks for key in chunk]
    assert INTERRUPT_CHUNK_KEY not in keys, f"a clean run must never pause: {keys}"
    # The asking node is not merely silent on a clean run - it is never
    # ENTERED, so a clean run costs no extra checkpoint for it either.
    assert keys == ["entry", "vision", "research", "analysis", "tts"]

    snapshot = graph.get_state(config)
    assert not snapshot.interrupts
    assert snapshot.values["audio_file_path"]
    assert "$104.95" in snapshot.values["final_output"]


def test_fast_path_never_reaches_the_asking_node():
    """The fast path has no number verification at all (see
    clarif_eye.verification's docstring), so it must not acquire an asking
    node by accident either."""
    graph, _checkpointer, config = _graph_and_config("fast-thread")
    client = RecordingClient(HONEST_DRAFT)
    config["configurable"].update(_run_config(client, "fast-thread")["configurable"])
    # A short, number-free reply keeps the router on the fast path.
    client.complete = lambda role, messages, **params: CompletionResult(
        content="OCR_TEXT: hello\nSCENE: a wall", model="fake"
    ) if any(
        part.get("type") == "image_url"
        for message in messages
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
    ) else CompletionResult(content="A plain wall.", model="fake")

    chunks = list(
        graph.stream(make_initial_state("base64photo"), config=config, stream_mode="updates")
    )

    keys = [key for chunk in chunks for key in chunk]
    assert keys == ["entry", "vision", "fast_synth", "tts"]
    assert INTERRUPT_CHUNK_KEY not in keys


def test_resume_does_not_call_the_brain_model_a_second_time():
    """MUTATION TARGET: moving the interrupt back in front of the brain
    call (e.g. raising it inside `analysis`) must turn this test RED, because
    LangGraph re-executes the whole interrupted node on resume."""
    graph, _checkpointer, config = _graph_and_config()
    client = RecordingClient(INVENTED_DRAFT)
    config["configurable"].update(_run_config(client)["configurable"])

    list(graph.stream(make_initial_state("base64photo"), config=config, stream_mode="updates"))
    assert len(client.brain_calls()) == 1, "the draft should cost exactly one brain call"

    list(graph.stream(Command(resume=RESUME_CONTINUE), config=config, stream_mode="updates"))

    assert len(client.brain_calls()) == 1, (
        "resuming re-ran the brain model - the interrupt must sit AFTER the "
        f"model call, not before it: {client.brain_calls()}"
    )


def test_resume_still_works_after_the_thread_is_trimmed():
    """An interrupted run's resume depends on its checkpoint surviving, and
    _trim_thread_to_latest_checkpoint is reachable while a thread is paused
    (a cache hit, or any other bookkeeping write, can trim mid-pause).

    EMPIRICALLY VERIFIED on langgraph 1.2.10 before this was relied on: the
    pending interrupt's write lives under the thread's NEWEST checkpoint,
    which is exactly the one the trim keeps. Trimmed repeatedly here, not
    once, because nothing bounds how many trims can land between pause and
    resume.
    """
    graph, checkpointer, config = _graph_and_config()
    client = RecordingClient(INVENTED_DRAFT)
    config["configurable"].update(_run_config(client)["configurable"])
    list(graph.stream(make_initial_state("base64photo"), config=config, stream_mode="updates"))

    for _ in range(3):
        _trim_thread_to_latest_checkpoint(checkpointer, "interrupt-thread")

    assert graph.get_state(config).interrupts, "the trim destroyed the pending interrupt"

    list(graph.stream(Command(resume=RESUME_CONTINUE), config=config, stream_mode="updates"))

    snapshot = graph.get_state(config)
    assert INVENTED_DRAFT in snapshot.values["final_output"]
    assert snapshot.values["audio_file_path"]


# --- The UI: staged, spoken, and never raising ----------------------------


def _staged(updates):
    return [(status, audio, text) for status, audio, text in updates]


def test_paused_run_stages_the_spoken_question_through_the_status_path():
    resources = _resources(RecordingClient(INVENTED_DRAFT))
    signal = _PauseSignal()

    updates = _staged(
        handle_submit_staged(FakeImage(), resources, thread_id="ui-pause", pause_signal=signal)
    )

    assert signal.paused is True
    status, audio, text = updates[-1]
    # No audio: the question is spoken by the screen reader through the
    # aria-live status region, not by TTS.
    assert audio is None
    # It must say what was read, that a number could not be checked, and
    # the choice.
    for expected in (INVENTED_DRAFT, "999.99"):
        assert expected in status, f"{expected!r} missing from the spoken question"
    assert "Continue anyway" in status
    assert status == text, "the question must also be readable in the result box"
    # A paused run must NOT fall into the empty-final_output error branch.
    assert "Something went wrong" not in status
    # The narration ran normally up to the pause - the question is not the
    # first thing the user hears after submitting.
    assert STATUS_WORKING == updates[0][0]


def test_paused_run_records_no_turn_and_caches_nothing():
    resources = _resources(RecordingClient(INVENTED_DRAFT))

    list(handle_submit_staged(FakeImage(), resources, thread_id="ui-pause"))

    snapshot = resources.graph.get_state({"configurable": {"thread_id": "ui-pause"}})
    assert snapshot.values.get("messages") == [], "an unresolved run must not record a turn"
    # Nothing spoken, nothing cached - the next submit must ask again.
    assert resources.image_cache._entries == {}


def test_resume_continue_stages_a_spoken_outcome_and_records_the_turn():
    resources = _resources(RecordingClient(INVENTED_DRAFT))
    list(handle_submit_staged(FakeImage(), resources, thread_id="ui-continue"))

    updates = _staged(handle_resume_staged(RESUME_CONTINUE, resources, thread_id="ui-continue"))

    assert updates[0] == (STATUS_RESUMING, None, "")
    final_status, final_audio, final_text = updates[-1]
    assert final_audio, "the caveated script must be spoken as audio"
    assert INVENTED_DRAFT in final_text
    assert UNVERIFIED_NUMBER_CAVEAT in final_text
    # The staged audio-delay contract: the same status/text is yielded once
    # WITHOUT the audio path, then again WITH it.
    assert updates[-2] == (final_status, None, final_text)

    snapshot = resources.graph.get_state({"configurable": {"thread_id": "ui-continue"}})
    recorded = [message.content for message in snapshot.values["messages"]]
    assert recorded == [final_text]


def test_resume_retake_speaks_a_confirmation_and_records_no_turn():
    resources = _resources(RecordingClient(INVENTED_DRAFT))
    list(handle_submit_staged(FakeImage(), resources, thread_id="ui-retake"))

    updates = _staged(handle_resume_staged(RESUME_RETAKE, resources, thread_id="ui-retake"))

    _status, audio, text = updates[-1]
    assert audio, "the retake confirmation must be spoken"
    assert text == RETAKE_CONFIRMATION
    assert "999.99" not in text

    snapshot = resources.graph.get_state({"configurable": {"thread_id": "ui-retake"}})
    # A retake is NOT an answer about the photo - recording it as one would
    # put a non-answer into the history a follow-up reads back.
    assert snapshot.values.get("messages") == []


def test_resume_when_nothing_is_paused_speaks_an_explanation():
    resources = _resources(RecordingClient(HONEST_DRAFT))
    # A completed, never-paused run.
    list(handle_submit_staged(FakeImage(), resources, thread_id="ui-nopause"))

    updates = _staged(handle_resume_staged(RESUME_CONTINUE, resources, thread_id="ui-nopause"))

    _status, audio, text = updates[-1]
    assert audio is None
    assert text == NOTHING_TO_RESUME_MESSAGE


def test_resume_on_a_thread_this_process_never_had_degrades_spokenly():
    """The pause survives only as long as the process (InMemorySaver). If
    the instance restarted between pause and resume, the checkpoint is gone
    - detected structurally (no pending interrupt), never by an exception."""
    resources = _resources(RecordingClient(HONEST_DRAFT))

    updates = _staged(handle_resume_staged(RESUME_CONTINUE, resources, thread_id="never-existed"))

    _status, audio, text = updates[-1]
    assert audio is None
    assert text == NOTHING_TO_RESUME_MESSAGE


def test_resume_with_no_thread_at_all_never_raises():
    resources = _resources(RecordingClient(HONEST_DRAFT))

    updates = _staged(handle_resume_staged(RESUME_CONTINUE, resources))

    assert updates[-1][2] == NOTHING_TO_RESUME_MESSAGE


def test_resume_answer_that_is_not_continue_is_treated_as_a_retake():
    """Anything this app did not send must never be read as consent to
    speak an unverified number."""
    resources = _resources(RecordingClient(INVENTED_DRAFT))
    list(handle_submit_staged(FakeImage(), resources, thread_id="ui-garbled"))

    updates = _staged(handle_resume_staged("something else entirely", resources, thread_id="ui-garbled"))

    assert updates[-1][2] == RETAKE_CONFIRMATION


def test_resume_buttons_are_labelled_hidden_and_carry_elem_ids():
    resources = _resources(RecordingClient(HONEST_DRAFT))
    demo = build_interface(resources)
    try:
        buttons = {
            component.elem_id: component
            for component in demo.blocks.values()
            if getattr(component, "elem_id", None)
            in (RESUME_CONTINUE_BUTTON_ELEM_ID, RESUME_RETAKE_BUTTON_ELEM_ID)
        }
        assert set(buttons) == {RESUME_CONTINUE_BUTTON_ELEM_ID, RESUME_RETAKE_BUTTON_ELEM_ID}
        for button in buttons.values():
            # A real label a screen reader can announce, and out of the way
            # until a run is actually paused.
            assert button.value
            assert button.visible is False
    finally:
        demo.close()


def test_the_audio_delay_contract_is_the_shared_one():
    """Guard against a second copy of the 1.8s gap appearing for resumes."""
    assert AUDIO_PLAY_DELAY_MS == 1800


# --- Pause-lifecycle collisions (independent-gate findings) ----------------
#
# A pending question is a SAFETY question: the app is holding back a number
# it could not check. Every other thing the user can do while it is pending
# has to have a decided answer, because the failure mode of getting one
# wrong is either speaking an unverified number to someone who declined it,
# or silently throwing the question away and leaving two buttons on screen
# wired to nothing. The three collisions are: typing a follow-up,
# submitting a photo that is already cached, and submitting a fresh photo.


class ScriptedClient:
    """Like RecordingClient, but each successive VISION call reports a
    DIFFERENT photo and each successive BRAIN call returns a different
    draft - which is what lets one test hold two photos, one clean and one
    with a fabricated number, on a single thread.
    """

    def __init__(self, vision_replies, brain_replies):
        self.vision_replies = list(vision_replies)
        self.brain_replies = list(brain_replies)
        self.calls = []

    def complete(self, role, messages, **params):
        has_image, prompt = RecordingClient._flatten(messages)
        self.calls.append((role, has_image, prompt))
        if has_image:
            ocr, scene = self.vision_replies[len(self.vision_calls()) - 1]
            return CompletionResult(
                content=f"OCR_TEXT: {ocr}\nSCENE: {scene}", model="fake-eyes-model:free"
            )
        if role == "brain":
            return CompletionResult(
                content=self.brain_replies[len(self.brain_calls()) - 1],
                model="fake-brain-model:free",
            )
        return CompletionResult(content="A plain wall.", model="fake-eyes-model:free")

    def close(self):
        pass

    def vision_calls(self):
        return [call for call in self.calls if call[1]]

    def brain_calls(self):
        return [call for call in self.calls if call[0] == "brain" and not call[1]]


# Short, number-free, no document keywords: keeps clarif_eye.router on the
# FAST path, so this photo never reaches the numbers check and caches
# cleanly with real audio.
PLAIN_OCR = "hello"
PLAIN_SCENE = "a plain wall"


def _state_of(resources, thread_id):
    return resources.graph.get_state({"configurable": {"thread_id": thread_id}})


def test_a_follow_up_typed_while_a_question_is_pending_is_refused():
    """A pending safety question must survive a typed follow-up.

    Before this was fixed the follow-up SUPERSEDED the pending task
    (probe-proven: get_state().next went back to ()), silently destroying
    the question while the two answer buttons stayed on screen wired to a
    resume that would then find nothing to resume. Refusing is the decided
    rule: the user is told what is pending and what their two choices are,
    and nothing is thrown away.
    """
    resources = _resources(RecordingClient(INVENTED_DRAFT))
    list(handle_submit_staged(FakeImage(), resources, thread_id="ask-while-paused"))
    calls_before = len(resources.client.calls)

    updates = _staged(
        handle_ask_staged("what is the total?", resources, thread_id="ask-while-paused")
    )

    _status, audio, text = updates[-1]
    assert audio is None
    assert text == QUESTION_PENDING_MESSAGE
    # It must name BOTH ways out, or the user is told they are stuck
    # without being told how to get unstuck.
    assert RESUME_CONTINUE_LABEL in text
    assert RESUME_RETAKE_LABEL in text

    # THE PAUSE SURVIVED: still pending, still resumable.
    snapshot = _state_of(resources, "ask-while-paused")
    assert snapshot.next == ("verify_numbers",)
    assert snapshot.interrupts

    # And it cost nothing: no graph run, so no model call.
    assert len(resources.client.calls) == calls_before

    # The user can still answer afterwards, which is the point of refusing.
    resumed = _staged(
        handle_resume_staged(RESUME_CONTINUE, resources, thread_id="ask-while-paused")
    )
    assert UNVERIFIED_NUMBER_CAVEAT in resumed[-1][2]


def test_a_cached_photo_submitted_while_paused_resolves_the_pause_first():
    """Submitting another photo means the user moved on - an IMPLICIT
    RETAKE - and the cache-hit branch must resolve the pending question
    before it writes anything.

    Before this was fixed, the hit's graph.update_state() ran straight over
    a pending interrupt and left a ZOMBIE: get_state().interrupts empty
    (the pending write cleared by the state write) but .next still naming
    the paused node, so the thread was neither running nor resumable and
    the buttons pointed at nothing.
    """
    client = ScriptedClient(
        vision_replies=[(PLAIN_OCR, PLAIN_SCENE), (BILL_OCR, BILL_SCENE)],
        brain_replies=[INVENTED_DRAFT],
    )
    resources = _resources(client)
    plain_photo = FakeImage(content=b"plain-photo-bytes")

    # 1. A clean photo on another thread, so it lands in the image cache
    #    with real audio (only audio-bearing results are ever cached).
    list(handle_submit_staged(plain_photo, resources, thread_id="cache-owner"))
    # 2. A dense photo with a fabricated number pauses THIS thread.
    list(
        handle_submit_staged(
            FakeImage(content=b"bill-photo-bytes"), resources, thread_id="hit-while-paused"
        )
    )
    assert _state_of(resources, "hit-while-paused").interrupts, "setup: the run should have paused"

    # 3. The user gives up on the question and submits the cached photo.
    signal = _PauseSignal()
    updates = _staged(
        handle_submit_staged(
            plain_photo, resources, thread_id="hit-while-paused", pause_signal=signal
        )
    )

    # Served normally, from the cache - no third vision call.
    assert len(client.vision_calls()) == 2
    _status, audio, _text = updates[-1]
    assert audio, "the cached audio must still be served"
    assert signal.paused is False, "the buttons must end hidden"

    # NO ZOMBIE: nothing pending, nothing half-resumable.
    snapshot = _state_of(resources, "hit-while-paused")
    assert snapshot.next == ()
    assert not snapshot.interrupts

    # The thread describes the photo the user was just told about, and
    # carries no leftover draft from the abandoned question.
    assert snapshot.values["ocr_output"] == PLAIN_OCR
    assert snapshot.values["scene_context"] == PLAIN_SCENE
    assert snapshot.values["verification_hold"] is None

    # A late click on a button that is now hidden is answered in words.
    late = _staged(handle_resume_staged(RESUME_CONTINUE, resources, thread_id="hit-while-paused"))
    assert late[-1][2] == NOTHING_TO_RESUME_MESSAGE


def test_a_fresh_photo_submitted_while_paused_leaves_no_pending_question():
    """The non-cached sibling of the test above: LangGraph supersedes the
    pending task on its own here, but the two branches must end in the same
    place, and that has to be pinned rather than assumed."""
    client = ScriptedClient(
        vision_replies=[(BILL_OCR, BILL_SCENE), (PLAIN_OCR, PLAIN_SCENE)],
        brain_replies=[INVENTED_DRAFT],
    )
    resources = _resources(client)

    list(
        handle_submit_staged(
            FakeImage(content=b"bill-photo"), resources, thread_id="fresh-while-paused"
        )
    )
    assert _state_of(resources, "fresh-while-paused").interrupts, "setup: should have paused"

    signal = _PauseSignal()
    updates = _staged(
        handle_submit_staged(
            FakeImage(content=b"other-photo"),
            resources,
            thread_id="fresh-while-paused",
            pause_signal=signal,
        )
    )

    assert signal.paused is False
    assert updates[-1][1], "the new photo must be described and spoken"
    snapshot = _state_of(resources, "fresh-while-paused")
    assert snapshot.next == ()
    assert not snapshot.interrupts
    assert snapshot.values["verification_hold"] is None


def _resume_binding(demo, button_elem_id):
    """The (handler, answer_value) pair Gradio will actually run when
    `button_elem_id` is activated.

    Walks the built Blocks' own dependency table - demo.fns, whose targets
    name the component id that triggers each handler - so this reads what
    the app WILL DO, not what build_interface's source looks like.
    """
    button_id = next(
        block_id
        for block_id, block in demo.blocks.items()
        if getattr(block, "elem_id", None) == button_elem_id
    )
    for fn in demo.fns.values():
        if (button_id, "click") in (fn.targets or []) and fn.fn is not None:
            return fn.fn, fn.inputs[0].value
    raise AssertionError(f"no click handler bound to {button_elem_id}")


def test_each_resume_button_is_wired_to_the_answer_its_label_promises():
    """MUTATION TARGET: swapping the two answers in build_interface must
    turn this test RED.

    Nothing else in this suite pinned the wiring, and an inverted one is
    the worst bug this feature can have: the button a user activates to
    DECLINE an unverified number would speak it to them instead. Asserted
    by driving the handler Gradio itself would call, through the binding
    Gradio itself would use - not by reading the constants back out of the
    module, which would pass happily under a swap.
    """
    resources = _resources(RecordingClient(INVENTED_DRAFT))
    demo = build_interface(resources)
    try:
        for elem_id, thread_id, expecting_continue in (
            (RESUME_CONTINUE_BUTTON_ELEM_ID, "wired-continue", True),
            (RESUME_RETAKE_BUTTON_ELEM_ID, "wired-retake", False),
        ):
            list(handle_submit_staged(FakeImage(), resources, thread_id=thread_id))
            assert _state_of(resources, thread_id).interrupts, "setup: should have paused"

            handler, answer = _resume_binding(demo, elem_id)
            spoken = list(handler(answer, thread_id))[-1][2]

            if expecting_continue:
                assert UNVERIFIED_NUMBER_CAVEAT in spoken and INVENTED_DRAFT in spoken, (
                    f"the continue button did not speak the description: {spoken!r}"
                )
            else:
                assert spoken == RETAKE_CONFIRMATION, (
                    "the retake button spoke something other than the retake "
                    "confirmation - an inverted wiring would read an unverified "
                    f"number to a user who declined it: {spoken!r}"
                )
    finally:
        demo.close()
