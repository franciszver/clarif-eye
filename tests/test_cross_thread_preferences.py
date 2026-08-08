"""Tests for the cross-thread spoken-verbosity preference (issue #86 / P9.7).

RED FIRST: at the time this file was first committed, clarif_eye.preferences
did not exist, clarif_eye.graph.build_graph took no `store` parameter, and
clarif_eye.ui.AppResources had no `store` field - so every test here failed
on the import line or on a TypeError from an unexpected keyword argument.

WHAT THESE TESTS PROVE, matching the issue's own "Red-first" section
verbatim:
  1. A preference set (via the follow-up box) on one thread applies to a
     LATER run on a DIFFERENT thread of the SAME session - proving the
     Store, not the checkpointer, is what carries it.
  2. The store NEVER holds anything but a preference-shaped value - walked
     directly via InMemoryStore.search(()), the same content-free privacy
     guarantee the issue is built around.
  3. A store-less graph (build_graph() with no `store` argument - the
     existing default every other test file in this repo already relies on)
     keeps working exactly as before: store is optional at compile, the
     same shape `checkpointer` already has.

Same no-network discipline as tests/test_followup.py: a real compiled
graph, a real checkpointer, a real InMemoryStore, fake client and fake TTS
provider. Nothing here launches Gradio or opens a socket.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from clarif_eye import preferences
from clarif_eye import tts as tts_module
from clarif_eye.graph import build_graph
from clarif_eye.prompting import SHORT_VERBOSITY_INSTRUCTION
from clarif_eye.ui import (
    AppResources,
    STATUS_SUCCESS_TEXT_ONLY,
    ThreadRegistry,
    handle_ask_staged,
    handle_submit_staged,
)

from tests.test_followup import RecordingClient, _FailingTtsProvider, _FakeTtsProvider
# Reused, not re-declared - the same PIL-image stand-in tests/test_ui.py and
# tests/test_followup.py already use.
from tests.test_ui import FakeImage

# No digits/document keywords, so the router stays on the FAST path
# (vision -> fast_synth -> tts) for every photo submitted here - this file
# never has to fake a search backend either.
SESSION_ID = "session-under-test"


def _resources(client):
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    return AppResources(
        graph=build_graph(checkpointer=checkpointer, store=store),
        client=client,
        client_error=None,
        tts_providers=[_FakeTtsProvider()],
        searcher=None,
        research_client=None,
        thread_registry=ThreadRegistry(checkpointer),
        store=store,
    )


def setup_function(_fn):
    tts_module._last_result_set(None)


def test_preference_set_on_thread_a_applies_to_a_later_run_on_thread_b():
    """The issue's own red-first test (1), verbatim: preference set under
    session S on thread A applies on thread B under the same store."""
    client = RecordingClient()
    resources = _resources(client)

    list(handle_submit_staged(FakeImage(content=b"photo-a"), resources, thread_id="thread-a", session_id=SESSION_ID))
    calls_before_command = len(client.calls)

    list(
        handle_ask_staged(
            "shorter descriptions please", resources, thread_id="thread-a", session_id=SESSION_ID
        )
    )
    # Setting the preference costs NO model call at all (issue #86's own rule).
    assert len(client.calls) == calls_before_command

    list(handle_submit_staged(FakeImage(content=b"photo-b"), resources, thread_id="thread-b", session_id=SESSION_ID))

    # fast_synth's call is the "eyes" role with no image (has_image is False).
    eyes_calls = [c for c in client.calls if c[0] == "eyes" and c[1] is False]
    assert eyes_calls, f"expected a fast_synth call, got {client.calls}"
    _role, _has_image, prompt = eyes_calls[-1]
    assert SHORT_VERBOSITY_INSTRUCTION in prompt, (
        "thread B's synthesis prompt must carry the brevity instruction "
        "even though the preference was set on thread A"
    )


def test_a_preference_set_on_one_session_does_not_leak_into_another_session():
    client = RecordingClient()
    resources = _resources(client)

    list(handle_submit_staged(FakeImage(content=b"photo-a"), resources, thread_id="thread-a", session_id="session-one"))
    list(handle_ask_staged("shorter descriptions please", resources, thread_id="thread-a", session_id="session-one"))

    list(handle_submit_staged(FakeImage(content=b"photo-c"), resources, thread_id="thread-c", session_id="session-two"))

    eyes_calls = [c for c in client.calls if c[0] == "eyes" and c[1] is False]
    _role, _has_image, prompt = eyes_calls[-1]
    assert SHORT_VERBOSITY_INSTRUCTION not in prompt


def test_a_recognised_preference_command_speaks_a_confirmation_and_never_runs_the_graph():
    client = RecordingClient()
    resources = _resources(client)

    list(handle_submit_staged(FakeImage(), resources, thread_id="thread-a", session_id=SESSION_ID))
    calls_after_photo = len(client.calls)

    updates = list(
        handle_ask_staged("shorter descriptions please", resources, thread_id="thread-a", session_id=SESSION_ID)
    )

    assert len(client.calls) == calls_after_photo, "a recognised preference command must cost no model call"
    final_status, final_audio, final_text = updates[-1]
    assert final_audio, "the confirmation must be spoken, not text-only"
    assert "shorter" in final_text.lower()


def test_preference_confirmation_announces_text_only_status_when_chain_exhausted():
    # Issue #88 / P9.9 coverage: _handle_preference_command shares
    # _outcome_for too (see that function's own docstring), so a REAL chain
    # exhaustion on THIS confirmation must still announce
    # STATUS_SUCCESS_TEXT_ONLY, not just on the photo path - named here so
    # the sharing can't silently unshare later.
    client = RecordingClient()
    resources = _resources(client)

    list(handle_submit_staged(FakeImage(), resources, thread_id="thread-a", session_id=SESSION_ID))
    # Only the PREFERENCE COMMAND needs every provider to fail - the photo
    # run above already spoke successfully with the fake provider.
    resources.tts_providers = [_FailingTtsProvider()]

    updates = list(
        handle_ask_staged("shorter descriptions please", resources, thread_id="thread-a", session_id=SESSION_ID)
    )

    final_status, final_audio, final_text = updates[-1]
    assert final_status == STATUS_SUCCESS_TEXT_ONLY
    assert final_audio is None
    assert "shorter" in final_text.lower()


def test_the_store_holds_only_preference_shaped_values():
    """The issue's own red-first test (2): a full photo + follow-up +
    preference flow through the UI layer, then every value the store holds
    is walked and checked against the one allowed shape."""
    client = RecordingClient()
    resources = _resources(client)

    list(handle_submit_staged(FakeImage(), resources, thread_id="thread-a", session_id=SESSION_ID))
    list(handle_ask_staged("what is the expiry date?", resources, thread_id="thread-a", session_id=SESSION_ID))
    list(handle_ask_staged("shorter descriptions please", resources, thread_id="thread-a", session_id=SESSION_ID))
    list(handle_submit_staged(FakeImage(content=b"photo-b"), resources, thread_id="thread-b", session_id=SESSION_ID))

    items = resources.store.search(())
    assert items, "expected at least one stored preference"
    for item in items:
        assert isinstance(item.value, dict)
        assert set(item.value.keys()) == {"verbosity"}
        assert item.value["verbosity"] in ("short", "detailed")


def test_build_graph_with_no_store_still_works():
    """Issue #86's third red-first requirement: store optional at compile,
    like checkpointer - the existing store-less build_graph() default
    (every other test file in this repo) must keep working unchanged."""
    from clarif_eye.state import make_initial_state

    client = RecordingClient()
    graph = build_graph()

    result = graph.invoke(
        make_initial_state("aW1hZ2UtZGF0YQ=="),
        config={"configurable": {"client": client, "tts_providers": [_FakeTtsProvider()]}},
    )
    assert result["final_output"]


def test_detect_preference_command_recognises_the_documented_phrases():
    assert preferences.detect_preference_command("shorter") == preferences.VERBOSITY_SHORT
    assert preferences.detect_preference_command("shorter descriptions") == preferences.VERBOSITY_SHORT
    assert preferences.detect_preference_command("Shorter descriptions, please!") == preferences.VERBOSITY_SHORT
    assert preferences.detect_preference_command("be brief") == preferences.VERBOSITY_SHORT
    assert preferences.detect_preference_command("longer") == preferences.VERBOSITY_DETAILED
    assert preferences.detect_preference_command("more detail") == preferences.VERBOSITY_DETAILED
    assert preferences.detect_preference_command("what is the expiry date?") is None
    assert preferences.detect_preference_command(None) is None


def test_get_verbosity_never_raises_on_a_none_store_or_missing_session():
    assert preferences.get_verbosity(None, "some-session") is None
    assert preferences.get_verbosity(InMemoryStore(), None) is None
    assert preferences.verbosity_for_config(None, {"configurable": {"session_id": "x"}}) is None
    assert preferences.verbosity_for_config(InMemoryStore(), None) is None


def test_set_verbosity_is_a_no_op_with_no_store_or_no_session_id():
    # Must not raise.
    preferences.set_verbosity(None, "session", preferences.VERBOSITY_SHORT)
    preferences.set_verbosity(InMemoryStore(), None, preferences.VERBOSITY_SHORT)
