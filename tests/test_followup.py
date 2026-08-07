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

# No digits and no document keywords, so clarif_eye.router keeps the photo
# run on the FAST path (vision -> fast_synth -> tts) and this file never has
# to fake a search backend. The exact wording matters: the follow-up test
# asserts this string reaches the brain model's prompt from CHECKPOINTED
# state, not from anything the follow-up call itself passed in.
STORED_OCR_TEXT = "best before next April"
STORED_SCENE_TEXT = "a jar of jam on a kitchen counter"

QUESTION = "what is the expiry date?"
CANNED_ANSWER = "The label says best before next April."


class FakeImage:
    """Stand-in for a PIL Image good enough for base64 encoding - same
    shape tests/test_ui.py's FakeImage uses."""

    mode = "RGB"

    def __init__(self, content=b"\xff\xd8\xff\xe0fakejpegbytes"):
        self.content = content

    def convert(self, mode):
        return self

    def save(self, buf, format=None):
        buf.write(self.content)


class RecordingClient:
    """Records every complete() call, so a test can assert on what the
    models were actually ASKED, not just on what came back.

    `calls` entries are (role, has_image, prompt_text) triples. `has_image`
    is computed structurally from the request body (an "image_url" content
    part), which is the only honest way to tell a real vision call apart
    from clarif_eye.synth's image-free call on the SAME "eyes" role.
    """

    def __init__(self):
        self.calls = []

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
                content=f"OCR_TEXT: {STORED_OCR_TEXT}\nSCENE: {STORED_SCENE_TEXT}",
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
