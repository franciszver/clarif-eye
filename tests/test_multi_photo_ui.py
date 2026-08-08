"""Submitting several photos at once, from the app's side (issue #110 /
P10.2).

tests/test_send_fanout.py pins the GRAPH's half of this feature - the Send
fan-out, the join, the submission ordering. This file pins the APP's half:
what a screen-reader user is told, which photos cost model calls, what is
remembered afterwards, and what is written back to the image cache.

THE SINGLE-PHOTO PATH IS THE PRIMARY EXPERIENCE and is deliberately not
re-tested here - tests/test_ui.py and tests/test_accessibility.py already
own it, unchanged, and that is the contract this feature had to fit inside
rather than around.

Real compiled graph, real checkpointer, real ThreadRegistry, fake client,
fake TTS provider - the same no-network discipline as
tests/test_ask_before_speaking.py. Nothing here launches Gradio.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from clarif_eye import tts as tts_module
from clarif_eye.client import CompletionResult, LadderExhaustedError
from clarif_eye.graph import build_graph
from clarif_eye.ui import (
    AppResources,
    STATUS_DEGRADED,
    STATUS_WORKING,
    ThreadRegistry,
    handle_submit_staged,
    working_status_for,
)

from tests.test_ui import FakeImage

# Each "photo" is told apart by the bytes it encodes to, which is also what
# the image cache keys on - so these doubles exercise the same distinction
# the real thing makes.
APPLES = FakeImage(content=b"\xff\xd8\xff\xe0apples")
BREAD = FakeImage(content=b"\xff\xd8\xff\xe0bread")
CHEESE = FakeImage(content=b"\xff\xd8\xff\xe0cheese")

SUBJECTS = {
    "apples": "a fruit bowl",
    "bread": "a bakery shelf",
    "cheese": "a deli counter",
}


class SubjectClient:
    """Answers about whichever photo the request is actually about, and
    counts the calls so a cache hit's saved model work is measurable.

    The eyes call carries the base64 photo, whose decoded bytes end in the
    subject word; the writing call carries the OCR text this branch just
    read. Anything else is a crossed wire and fails loudly rather than
    quietly describing the wrong photo.
    """

    def __init__(self, failing_subject=None):
        self.failing_subject = failing_subject
        self.calls = []

    def complete(self, role, messages, **params):
        import base64

        blob = str(messages)
        for subject, scene in SUBJECTS.items():
            try:
                decoded = base64.b64decode(_image_payload(messages) or "").decode(
                    "ascii", "ignore"
                )
            except Exception:
                decoded = ""
            if decoded.endswith(subject):
                self.calls.append(("eyes", subject))
                if subject == self.failing_subject:
                    raise LadderExhaustedError("every model was busy", attempts=())
                return CompletionResult(
                    content=f"OCR_TEXT: {subject}\nSCENE: {scene}",
                    model="fake-eyes-model:free",
                )
        for subject in SUBJECTS:
            if subject in blob:
                self.calls.append(("write", subject))
                return CompletionResult(
                    content=f"This is {subject}.", model="fake-brain-model:free"
                )
        raise AssertionError(f"fake client got a request about no known photo: {blob[:200]}")

    def subjects_read(self):
        return [subject for role, subject in self.calls if role == "eyes"]

    def close(self):
        pass


def _image_payload(messages):
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                return url.split(",", 1)[-1]
    return None


class FakeTtsProvider:
    def __init__(self):
        self.spoken = []

    def synthesize(self, text, out_path):
        self.spoken.append(text)
        with open(out_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 32)


def setup_function(_fn):
    tts_module._last_result_set(None)


def _resources(client, provider=None):
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    return AppResources(
        graph=build_graph(checkpointer=checkpointer, store=store),
        client=client,
        client_error=None,
        tts_providers=[provider or FakeTtsProvider()],
        searcher=None,
        research_client=None,
        thread_registry=ThreadRegistry(checkpointer),
        store=store,
    )


def _stage(resources, image, extra_images=None, thread_id="multi-thread"):
    return list(
        handle_submit_staged(
            image, resources, extra_images=extra_images, thread_id=thread_id
        )
    )


# --- What the user is told ------------------------------------------------


def test_the_opening_announcement_says_how_many_photos():
    # One photo says exactly what it always said - the primary path's
    # wording is not allowed to drift because a second one exists.
    assert working_status_for(1) == STATUS_WORKING
    assert "3" in working_status_for(3) or "three" in working_status_for(3).lower()


def test_three_photos_are_described_in_one_turn_and_spoken_once():
    provider = FakeTtsProvider()
    client = SubjectClient()
    resources = _resources(client, provider)

    yields = _stage(resources, APPLES, extra_images=[BREAD, CHEESE])

    opening = yields[0][0]
    assert opening == working_status_for(3)

    status, audio, text = yields[-1]
    assert audio, "a multi-photo turn must still produce spoken audio"
    assert text.index("apples") < text.index("bread") < text.index("cheese")
    # ONE voice for the turn: tts was handed the combined script once.
    assert len(provider.spoken) == 1
    assert "apples" in provider.spoken[0] and "cheese" in provider.spoken[0]
    assert status != STATUS_DEGRADED


def test_progress_is_announced_per_photo_only_when_there_is_more_than_one():
    resources = _resources(SubjectClient())

    statuses = [status for status, _audio, _text in _stage(resources, APPLES, [BREAD, CHEESE])]
    counted = [s for s in statuses if "of 3 photos described" in s]
    assert counted == [
        "1 of 3 photos described.",
        "2 of 3 photos described.",
        "3 of 3 photos described.",
    ]

    # A single photo hears nothing of the sort - there is no count worth
    # announcing, and the primary path's spoken sequence is unchanged.
    solo = [status for status, _a, _t in _stage(resources, CHEESE, thread_id="solo-thread")]
    assert not [s for s in solo if "photos described" in s]


# --- The per-photo cache --------------------------------------------------


def test_a_cached_photo_inside_a_fan_out_costs_no_model_call_but_is_still_spoken():
    client = SubjectClient()
    resources = _resources(client)

    # Describe one photo on its own first, so it lands in the image cache.
    _stage(resources, BREAD, thread_id="warm-up")
    assert client.subjects_read() == ["bread"]

    _status, _audio, text = _stage(resources, APPLES, [BREAD, CHEESE])[-1]

    # The cached photo was NOT read again, but it is still in the script.
    assert sorted(client.subjects_read()) == ["apples", "bread", "cheese"]
    assert "bread" in text
    assert text.index("apples") < text.index("bread") < text.index("cheese")


def test_a_multi_photo_turn_is_never_cached_as_one_result():
    client = SubjectClient()
    resources = _resources(client)

    _stage(resources, APPLES, [BREAD])
    before = client.subjects_read()
    # The SAME submission again must not be served from a combined entry -
    # there is no per-photo audio in a multi-photo turn, so nothing about it
    # is admissible to a per-photo cache.
    _stage(resources, APPLES, [BREAD], thread_id="second-thread")

    assert client.subjects_read() != before, (
        "a multi-photo turn was replayed from the cache as if it were one photo"
    )


# --- One photo failing ----------------------------------------------------


def test_one_unreadable_photo_degrades_the_whole_turn_and_says_which_one():
    resources = _resources(SubjectClient(failing_subject="bread"))

    status, audio, text = _stage(resources, APPLES, [BREAD, CHEESE])[-1]

    assert "The second photo could not be described." in text
    # The photos that DID work are still described. Throwing away two good
    # descriptions because a third photo failed would be worse than useless.
    assert "apples" in text and "cheese" in text
    # NOT STATUS_DEGRADED, and that is issue #93 / P9.12's established
    # semantics rather than a gap: the run COMPLETED and produced real spoken
    # audio, so the status describes how the run ended, not whether the
    # answer is honest. What marks the turn as degraded travels in state and
    # is proved by the memory test below.
    assert audio and status != STATUS_DEGRADED


def test_a_degraded_multi_photo_turn_is_not_remembered():
    resources = _resources(SubjectClient(failing_subject="bread"))

    _stage(resources, APPLES, [BREAD, CHEESE], thread_id="degraded-thread")

    snapshot = resources.graph.get_state(
        {"configurable": {"thread_id": "degraded-thread"}}
    )
    assert snapshot.values.get("messages") == [], (
        "a turn one photo failed in was recorded as a real description"
    )
