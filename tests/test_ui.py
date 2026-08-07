"""Tests for the Gradio UI handler (issue #13 / P4.1).

These tests call clarif_eye.ui.handle_submit directly with fakes - they
NEVER launch a Gradio server and NEVER touch the network. The handler must
never raise into Gradio (a traceback is useless to a blind user), so every
test here asserts on the RETURNED (audio_path, text) tuple, never on an
exception escaping.

The three outcomes the UI must distinguish are told apart STRUCTURALLY
(via audio_file_path truthiness and tts.is_chain_exhausted()), never by
string-matching the script text - the same discipline vision.py/tts.py
already use (is_degraded_scene / is_chain_exhausted).
"""

from clarif_eye import tts as tts_module
from clarif_eye.client import Attempt, LadderExhaustedError, OpenRouterError
from clarif_eye.failure_messages import BUSY_MESSAGE
from clarif_eye.failure_messages import CONFIG_ERROR_MESSAGE as MAPPED_CONFIG_ERROR_MESSAGE
from clarif_eye.ui import (
    AppResources,
    AUDIO_UNAVAILABLE_NOTE,
    CONFIG_ERROR_MESSAGE,
    NO_IMAGE_MESSAGE,
    UNEXPECTED_ERROR_MESSAGE,
    UNREADABLE_IMAGE_MESSAGE,
    build_resources,
    handle_submit,
    handle_submit_staged,
)


class FakeImage:
    """Stand-in for a PIL Image good enough for base64 encoding."""

    def __init__(self, mode="RGB"):
        self.mode = mode

    def convert(self, mode):
        return self

    def save(self, buf, format=None):
        buf.write(b"\xff\xd8\xff\xe0fakejpegbytes")


class ContentImage:
    """Stand-in PIL Image whose encoded bytes are caller-controlled, so
    cache tests (issue #75) can tell two "photos" apart by content rather
    than by identity/path - the same distinction the cache itself must
    make.
    """

    mode = "RGB"

    def __init__(self, content):
        self.content = content

    def convert(self, mode):
        return self

    def save(self, buf, format=None):
        buf.write(self.content)


class BrokenImage:
    """Stand-in for a corrupt/unreadable image - .save() blows up."""

    mode = "RGB"

    def convert(self, mode):
        return self

    def save(self, buf, format=None):
        raise OSError("cannot identify image data")


class FakeGraph:
    """Records the config it was invoked with and returns a canned result."""

    def __init__(self, result=None, exc=None):
        self.result = result or {}
        self.exc = exc
        self.invocations = []

    def invoke(self, state, config=None):
        self.invocations.append({"state": state, "config": config})
        if self.exc is not None:
            raise self.exc
        return self.result


class SequencedGraph:
    """Like FakeGraph, but returns a DIFFERENT final_output on each
    invocation - lets cache tests (issue #75) assert that two different
    images produce genuinely different results, not just different call
    counts."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.invocations = []

    def invoke(self, state, config=None):
        idx = len(self.invocations)
        self.invocations.append({"state": state, "config": config})
        return {"final_output": self.outputs[idx], "audio_file_path": ""}


def _resources(graph, client="fake-client"):
    return AppResources(
        graph=graph,
        client=client,
        client_error=None,
        tts_providers=["fake-provider-chain"],
        searcher=None,
        research_client=None,
    )


def setup_function(_fn):
    # tts.is_chain_exhausted() reads module-level state left by the last
    # real run_tts() call - reset it so tests don't leak into each other.
    tts_module._last_result_set(None)


# --- Outcome (a): success with audio -----------------------------------


def test_success_with_audio_plays_audio_and_shows_script():
    tts_module._last_result_set(
        tts_module.TtsResult("/tmp/out.mp3", (tts_module.ProviderAttempt("Edge", "success", ""),), "Edge")
    )
    graph = FakeGraph(result={"final_output": "A cat sits on a mat.", "audio_file_path": "/tmp/out.mp3"})
    resources = _resources(graph)

    audio, text = handle_submit(FakeImage(), resources)

    assert audio == "/tmp/out.mp3"
    assert text == "A cat sits on a mat."


# --- Outcome (b): script produced, no audio, chain exhausted ------------


def test_text_only_when_tts_chain_exhausted_announces_it():
    tts_module._last_result_set(
        tts_module.TtsResult(
            "",
            (
                tts_module.ProviderAttempt("Edge", "error", "boom"),
                tts_module.ProviderAttempt("Gtts", "error", "boom"),
            ),
            None,
        )
    )
    graph = FakeGraph(result={"final_output": "A cat sits on a mat.", "audio_file_path": ""})
    resources = _resources(graph)

    audio, text = handle_submit(FakeImage(), resources)

    assert audio is None
    assert "A cat sits on a mat." in text
    assert AUDIO_UNAVAILABLE_NOTE in text


# --- Outcome (c): pipeline degraded upstream, message IS the script -----


def test_upstream_degraded_message_is_spoken_and_shown_without_extra_layer():
    degraded_message = "Vision could not run right now: every available model was busy or unavailable."
    tts_module._last_result_set(
        tts_module.TtsResult(
            "/tmp/degraded.mp3", (tts_module.ProviderAttempt("Edge", "success", ""),), "Edge"
        )
    )
    graph = FakeGraph(result={"final_output": degraded_message, "audio_file_path": "/tmp/degraded.mp3"})
    resources = _resources(graph)

    audio, text = handle_submit(FakeImage(), resources)

    assert audio == "/tmp/degraded.mp3"
    assert text == degraded_message


# --- No image submitted --------------------------------------------------


def test_no_image_submitted_returns_helpful_message_without_invoking_graph():
    graph = FakeGraph(result={"final_output": "should not be reached", "audio_file_path": ""})
    resources = _resources(graph)

    audio, text = handle_submit(None, resources)

    assert audio is None
    assert text == NO_IMAGE_MESSAGE
    assert graph.invocations == []


# --- Missing API key at startup ------------------------------------------


def test_missing_api_key_returns_clear_message_without_invoking_graph():
    graph = FakeGraph(result={"final_output": "should not be reached", "audio_file_path": ""})
    resources = AppResources(
        graph=graph,
        client=None,
        client_error=CONFIG_ERROR_MESSAGE,
        tts_providers=["fake-provider-chain"],
        searcher=None,
        research_client=None,
    )

    audio, text = handle_submit(FakeImage(), resources)

    assert audio is None
    assert text == CONFIG_ERROR_MESSAGE
    assert graph.invocations == []


def test_build_resources_never_raises_when_api_key_missing(monkeypatch):
    def _raise(*args, **kwargs):
        raise OpenRouterError("OPENROUTER_API_KEY is required and must not be blank")

    monkeypatch.setattr("clarif_eye.ui.OpenRouterClient", _raise)

    resources = build_resources()

    assert resources.client is None
    assert resources.client_error == CONFIG_ERROR_MESSAGE
    assert resources.graph is not None


# --- Corrupt / unreadable image ------------------------------------------


def test_corrupt_image_returns_helpful_message_without_invoking_graph():
    graph = FakeGraph(result={"final_output": "should not be reached", "audio_file_path": ""})
    resources = _resources(graph)

    audio, text = handle_submit(BrokenImage(), resources)

    assert audio is None
    assert text == UNREADABLE_IMAGE_MESSAGE
    assert graph.invocations == []


# --- Unexpected exception inside the graph --------------------------------


def test_unexpected_exception_in_graph_is_handled_not_raised():
    graph = FakeGraph(exc=RuntimeError("boom"))
    resources = _resources(graph)

    audio, text = handle_submit(FakeImage(), resources)

    assert audio is None
    assert text == UNEXPECTED_ERROR_MESSAGE


# --- A LadderExhaustedError/OpenRouterError escaping the graph itself -------
# (issue #18 / P6.2 scope item 4): every node already degrades these
# internally, but if the whole pipeline fails before a node can degrade
# (e.g. a client-construction failure not caught by a node), the UI must
# apply the same category mapping, not collapse it into the generic
# UNEXPECTED_ERROR_MESSAGE.


def test_ladder_exhausted_escaping_the_graph_produces_the_busy_message():
    attempts = (
        Attempt("model-a", "rate_limited", 429, "rate limited"),
        Attempt("model-b", "rate_limited", 429, "rate limited"),
    )
    graph = FakeGraph(exc=LadderExhaustedError("eyes", attempts))
    resources = _resources(graph)

    audio, text = handle_submit(FakeImage(), resources)

    assert audio is None
    assert text == BUSY_MESSAGE


def test_terminal_config_error_escaping_the_graph_never_says_try_again():
    graph = FakeGraph(exc=OpenRouterError("authentication failed", status_code=401))
    resources = _resources(graph)

    audio, text = handle_submit(FakeImage(), resources)

    assert audio is None
    assert text == MAPPED_CONFIG_ERROR_MESSAGE
    assert "try again" not in text.lower()


# --- Shared client: constructed once, injected on every invocation -------


def test_handle_submit_shares_one_client_across_invocations():
    graph = FakeGraph(result={"final_output": "ok", "audio_file_path": "/tmp/ok.mp3"})
    tts_module._last_result_set(
        tts_module.TtsResult("/tmp/ok.mp3", (tts_module.ProviderAttempt("Edge", "success", ""),), "Edge")
    )
    shared_client = object()
    resources = _resources(graph, client=shared_client)

    for _ in range(3):
        handle_submit(FakeImage(), resources)

    assert len(graph.invocations) == 3
    clients_used = [inv["config"]["configurable"]["client"] for inv in graph.invocations]
    assert clients_used == [shared_client, shared_client, shared_client]
    assert all(c is shared_client for c in clients_used)


def test_build_resources_constructs_client_only_once(monkeypatch):
    calls = []

    class SpyClient:
        def __init__(self, *args, **kwargs):
            calls.append(1)

    monkeypatch.setattr("clarif_eye.ui.OpenRouterClient", SpyClient)

    resources = build_resources()
    for _ in range(3):
        handle_submit(FakeImage(), resources)

    assert len(calls) == 1


# --- Image content cache (issue #75): identical photos cost one model call -
#
# The same photo submitted twice must cost one model call, not two.
# Keyed on the DECODED image content (the bytes handle_submit actually
# sends), never on filename/path - the same photo uploaded twice arrives
# at a different temp path each time.


def test_identical_image_content_reuses_cached_result_no_second_call():
    tts_module._last_result_set(
        tts_module.TtsResult("/tmp/out.mp3", (tts_module.ProviderAttempt("Edge", "success", ""),), "Edge")
    )
    graph = FakeGraph(result={"final_output": "A cat sits on a mat.", "audio_file_path": "/tmp/out.mp3"})
    resources = _resources(graph)

    first = handle_submit(ContentImage(b"photo-one-bytes"), resources)
    second = handle_submit(ContentImage(b"photo-one-bytes"), resources)

    assert len(graph.invocations) == 1
    assert first == second == ("/tmp/out.mp3", "A cat sits on a mat.")


def test_cache_hit_yields_the_same_staged_sequence_as_a_miss():
    tts_module._last_result_set(
        tts_module.TtsResult("/tmp/out.mp3", (tts_module.ProviderAttempt("Edge", "success", ""),), "Edge")
    )
    graph = FakeGraph(result={"final_output": "A cat sits on a mat.", "audio_file_path": "/tmp/out.mp3"})
    resources = _resources(graph)

    miss_updates = list(handle_submit_staged(ContentImage(b"photo-two-bytes"), resources))
    hit_updates = list(handle_submit_staged(ContentImage(b"photo-two-bytes"), resources))

    assert len(graph.invocations) == 1  # confirms the second run was a hit
    assert len(miss_updates) == len(hit_updates) == 3
    # Same shape on both: working status first, then status+text with no
    # audio, then (only after the existing delay) status+text+audio - a
    # hit must not skip or reorder the audio-delay stage.
    for miss, hit in zip(miss_updates, hit_updates):
        assert len(miss) == len(hit) == 3
    assert miss_updates[0][1] is None
    assert hit_updates[0][1] is None
    assert miss_updates[1][1] is None
    assert hit_updates[1][1] is None
    assert miss_updates[2][1] == hit_updates[2][1] == "/tmp/out.mp3"
    assert miss_updates[2][2] == hit_updates[2][2] == "A cat sits on a mat."


def test_different_images_produce_two_calls_and_two_different_results():
    graph = SequencedGraph(["first description", "second description"])
    resources = _resources(graph)

    _, text_a = handle_submit(ContentImage(b"photo-a-bytes"), resources)
    _, text_b = handle_submit(ContentImage(b"photo-b-bytes"), resources)

    assert len(graph.invocations) == 2
    assert text_a == "first description"
    assert text_b == "second description"
    assert text_a != text_b


def test_failed_result_is_not_cached_a_retry_makes_a_second_call():
    graph = FakeGraph(exc=RuntimeError("boom"))
    resources = _resources(graph)

    handle_submit(ContentImage(b"photo-fail-bytes"), resources)
    handle_submit(ContentImage(b"photo-fail-bytes"), resources)

    assert len(graph.invocations) == 2
