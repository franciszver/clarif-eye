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

import os

import pytest

from clarif_eye import tts as tts_module
from clarif_eye.client import Attempt, CompletionResult, LadderExhaustedError, OpenRouterError
from clarif_eye.failure_messages import BUSY_MESSAGE
from clarif_eye.failure_messages import CONFIG_ERROR_MESSAGE as MAPPED_CONFIG_ERROR_MESSAGE
from clarif_eye.graph import build_graph
from clarif_eye.ui import (
    AppResources,
    AUDIO_UNAVAILABLE_NOTE,
    CONFIG_ERROR_MESSAGE,
    IMAGE_CACHE_MAX_ENTRIES,
    NO_IMAGE_MESSAGE,
    STATUS_NODE_RESEARCH,
    STATUS_NODE_TTS,
    STATUS_NODE_WRITING,
    STATUS_WORKING,
    UNEXPECTED_ERROR_MESSAGE,
    UNREADABLE_IMAGE_MESSAGE,
    build_resources,
    handle_submit,
    handle_submit_staged,
)


class FakeImage:
    """Stand-in for a PIL Image good enough for base64 encoding.

    `content` defaults to fixed bytes for tests that don't care what's in
    the "photo"; cache tests (issue #75) pass a distinct `content` per
    "photo" so they can tell two images apart by content rather than by
    identity/path - the same distinction the cache itself must make.
    """

    def __init__(self, mode="RGB", content=b"\xff\xd8\xff\xe0fakejpegbytes"):
        self.mode = mode
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
    """Records the config it was invoked with and returns a canned result.

    stream() (issue #80 / P9.1) yields exactly one chunk, keyed "tts" -
    _run_pipeline_events always streams now (no invoke()/stream() fork),
    and a real graph's LAST chunk is always tts's, so this is the minimal
    shape that satisfies it: invoke()'s own logic still runs (recording
    the invocation, raising self.exc), and graph.next_node_after("tts", ...)
    is None (nothing follows tts), so this maps to no narration phrase and
    every existing staged-contract test keeps its exact yield sequence.
    """

    def __init__(self, result=None, exc=None):
        self.result = result or {}
        self.exc = exc
        self.invocations = []

    def invoke(self, state, config=None):
        self.invocations.append({"state": state, "config": config})
        if self.exc is not None:
            raise self.exc
        return self.result

    def stream(self, state, config=None, stream_mode="updates"):
        yield {"tts": self.invoke(state, config=config)}


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

    def stream(self, state, config=None, stream_mode="updates"):
        yield {"tts": self.invoke(state, config=config)}


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

    # Distinct content per call (issue #75's cache would otherwise turn
    # repeats of the SAME image into hits) - this test is about the
    # client being shared, not about caching.
    for i in range(3):
        handle_submit(FakeImage(content=f"photo-{i}".encode()), resources)

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


def test_image_cache_size_never_exceeds_the_tts_retention_count():
    # This bound only rules out slots that are CERTAIN to be dead: a cache
    # entry beyond the retained-file count can never be served, because
    # _prune_old_files (mtime-based) will already have deleted its mp3
    # before it could ever be a hit. It does NOT guarantee a live entry
    # keeps its file even within the bound - cache eviction is LRU by
    # ACCESS time while file pruning is by WRITE time, and a cache hit
    # never updates mtime, so a repeatedly-hit entry can stay "live" here
    # while its file still ages out of the MAX_KEPT_FILES most-recently-
    # written set. The stale-file guard in handle_submit (treating a
    # missing file as a miss, never a lying hit) is what actually makes
    # that safe, not this bound.
    assert IMAGE_CACHE_MAX_ENTRIES <= tts_module.MAX_KEPT_FILES, (
        f"IMAGE_CACHE_MAX_ENTRIES ({IMAGE_CACHE_MAX_ENTRIES}) exceeds "
        f"tts.MAX_KEPT_FILES ({tts_module.MAX_KEPT_FILES}): a cache entry "
        "beyond the retained-file count can never be served, because "
        "_prune_old_files will already have deleted its mp3. This bound "
        "does not, by itself, guarantee a live entry keeps its file even "
        "within it (eviction is by access time, pruning is by write "
        "time) - the stale-file guard in handle_submit is what makes "
        "that case safe. Lower IMAGE_CACHE_MAX_ENTRIES to at most "
        "MAX_KEPT_FILES."
    )


def test_identical_image_content_reuses_cached_result_no_second_call(tmp_path):
    audio_path = tmp_path / "out.mp3"
    audio_path.write_bytes(b"fake-mp3-bytes")  # must exist: a cache hit checks this
    tts_module._last_result_set(
        tts_module.TtsResult(str(audio_path), (tts_module.ProviderAttempt("Edge", "success", ""),), "Edge")
    )
    graph = FakeGraph(result={"final_output": "A cat sits on a mat.", "audio_file_path": str(audio_path)})
    resources = _resources(graph)

    first = handle_submit(FakeImage(content=b"photo-one-bytes"), resources)
    second = handle_submit(FakeImage(content=b"photo-one-bytes"), resources)

    assert len(graph.invocations) == 1
    assert first == second == (str(audio_path), "A cat sits on a mat.")


def test_cache_hit_yields_the_same_staged_sequence_as_a_miss(tmp_path):
    audio_path = tmp_path / "out.mp3"
    audio_path.write_bytes(b"fake-mp3-bytes")  # must exist: a cache hit checks this
    tts_module._last_result_set(
        tts_module.TtsResult(str(audio_path), (tts_module.ProviderAttempt("Edge", "success", ""),), "Edge")
    )
    graph = FakeGraph(result={"final_output": "A cat sits on a mat.", "audio_file_path": str(audio_path)})
    resources = _resources(graph)

    miss_updates = list(handle_submit_staged(FakeImage(content=b"photo-two-bytes"), resources))
    hit_updates = list(handle_submit_staged(FakeImage(content=b"photo-two-bytes"), resources))

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
    assert miss_updates[2][1] == hit_updates[2][1] == str(audio_path)
    assert miss_updates[2][2] == hit_updates[2][2] == "A cat sits on a mat."


def test_different_images_produce_two_calls_and_two_different_results():
    graph = SequencedGraph(["first description", "second description"])
    resources = _resources(graph)

    _, text_a = handle_submit(FakeImage(content=b"photo-a-bytes"), resources)
    _, text_b = handle_submit(FakeImage(content=b"photo-b-bytes"), resources)

    assert len(graph.invocations) == 2
    assert text_a == "first description"
    assert text_b == "second description"
    assert text_a != text_b


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("boom"),
        LadderExhaustedError(
            "eyes",
            (
                Attempt("model-a", "rate_limited", 429, "rate limited"),
                Attempt("model-b", "rate_limited", 429, "rate limited"),
            ),
        ),
        OpenRouterError("authentication failed", status_code=401),
    ],
    ids=["generic-exception", "ladder-exhausted", "openrouter-error"],
)
def test_failed_result_is_not_cached_a_retry_makes_a_second_call(exc):
    graph = FakeGraph(exc=exc)
    resources = _resources(graph)

    handle_submit(FakeImage(content=b"photo-fail-bytes"), resources)
    handle_submit(FakeImage(content=b"photo-fail-bytes"), resources)

    assert len(graph.invocations) == 2


class FileWritingGraph:
    """FakeGraph that actually writes bytes to `audio_path` on each
    invoke() call, standing in for the real pipeline actually producing a
    fresh mp3 - lets a test tell "the cache lied about a path" apart from
    "the pipeline never ran again" (issue #75 follow-up)."""

    def __init__(self, audio_path, final_output):
        self.audio_path = audio_path
        self.final_output = final_output
        self.invocations = []

    def invoke(self, state, config=None):
        self.invocations.append({"state": state, "config": config})
        self.audio_path.write_bytes(b"fake-mp3-bytes")
        return {"final_output": self.final_output, "audio_file_path": str(self.audio_path)}

    def stream(self, state, config=None, stream_mode="updates"):
        yield {"tts": self.invoke(state, config=config)}


class SequencedTtsGraph:
    """Like SequencedGraph, but also plants tts_module's last-result state
    on each invoke() call - simulating a real run_tts() call happening
    inside the graph - so is_chain_exhausted() reflects THIS call's TTS
    outcome, not whatever setup_function/a previous call left behind.
    `outcomes` is a list of (final_output, audio_path, TtsResult) tuples,
    one per successive invoke() call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.invocations = []

    def invoke(self, state, config=None):
        idx = len(self.invocations)
        self.invocations.append({"state": state, "config": config})
        final_output, audio_path, tts_result = self.outcomes[idx]
        tts_module._last_result_set(tts_result)
        return {"final_output": final_output, "audio_file_path": audio_path}

    def stream(self, state, config=None, stream_mode="updates"):
        yield {"tts": self.invoke(state, config=config)}


def test_chain_exhausted_result_is_not_cached_a_retry_reruns_and_recovers(tmp_path):
    # Issue found in review: outcome (b) (description succeeded, every TTS
    # provider failed for THIS call) used to be cached anyway, because
    # handle_submit cached whenever the pipeline call didn't raise. That
    # permanently muted a photo: TTS recovering later never mattered
    # because the cache kept replaying the old text-only result. A
    # text-only outcome is a degraded result and must be retried next
    # time, per this module's own cache docstring ("a quota/API failure
    # must never be replayed to the next visitor as if it were that
    # photo's own answer").
    audio_path = tmp_path / "out.mp3"
    audio_path.write_bytes(b"fake-mp3-bytes")
    failed_attempts = (
        tts_module.ProviderAttempt("Edge", "error", "boom"),
        tts_module.ProviderAttempt("Gtts", "error", "boom"),
    )
    graph = SequencedTtsGraph(
        [
            ("A cat sits on a mat.", "", tts_module.TtsResult("", failed_attempts, None)),
            (
                "A cat sits on a mat.",
                str(audio_path),
                tts_module.TtsResult(
                    str(audio_path), (tts_module.ProviderAttempt("Edge", "success", ""),), "Edge"
                ),
            ),
        ]
    )
    resources = _resources(graph)

    first_audio, first_text = handle_submit(FakeImage(content=b"photo-tts-recovers"), resources)
    second_audio, second_text = handle_submit(FakeImage(content=b"photo-tts-recovers"), resources)

    assert first_audio is None
    assert AUDIO_UNAVAILABLE_NOTE in first_text
    assert len(graph.invocations) == 2  # second call re-ran the pipeline, not a cache hit
    assert second_audio == str(audio_path)
    assert second_text == "A cat sits on a mat."


def test_cache_hit_with_a_deleted_audio_file_reruns_the_pipeline(tmp_path):
    # The cache and the audio file normally share a process lifetime, but
    # a temp cleaner, disk pressure, or a future change to file pruning
    # (see tts.py's _prune_old_files - it DOES delete old mp3s) could
    # leave a cached path pointing at nothing. Returning that path as a
    # "hit" would have status_for_result announce "Description ready."
    # while the audio element plays silence - the worst failure mode for
    # a user who cannot see the screen. A miss is recoverable; a stale
    # path is not.
    audio_path = tmp_path / "out.mp3"
    tts_module._last_result_set(
        tts_module.TtsResult(str(audio_path), (tts_module.ProviderAttempt("Edge", "success", ""),), "Edge")
    )
    graph = FileWritingGraph(audio_path, "A cat sits on a mat.")
    resources = _resources(graph)

    first_audio, _ = handle_submit(FakeImage(content=b"photo-vanishing-file"), resources)
    assert first_audio == str(audio_path)
    assert len(graph.invocations) == 1
    assert audio_path.exists()

    audio_path.unlink()  # simulate the file disappearing between requests

    second_audio, _ = handle_submit(FakeImage(content=b"photo-vanishing-file"), resources)

    assert len(graph.invocations) == 2  # ran again - not a lying cache hit
    assert second_audio == str(audio_path)
    assert os.path.exists(second_audio)


# --- Per-node stream progress (issue #80 / P9.1) ---------------------------
#
# handle_submit_staged used to fake progress with a single "working" status
# because graph.invoke() is one opaque blocking call with no intermediate
# hook. graph.stream(..., stream_mode="updates") yields one dict per
# COMPLETED node (keyed by node name), so real per-node narration is now
# possible - these tests drive a REAL compiled graph (not FakeGraph, which
# only implements .invoke() and is intentionally left alone so every
# existing FakeGraph-based test above keeps passing unchanged) and assert
# the live-region status sequence carries one narration phrase per node
# that actually ran, in execution order, for both router paths.


class _RoutingVisionClient:
    """Real vision-node reply shape (see tests/test_graph.py's
    FakeVisionClient) whose OCR text length drives the router - long text
    with no data-density signals trips only the router's word-count
    fallback (same LONG_OCR_TEXT trick test_graph.py/test_pipeline_deadline.py
    use), giving deterministic control over which path runs."""

    def __init__(self, ocr, scene):
        self.ocr = ocr
        self.scene = scene

    def complete(self, role, messages, **params):
        if role == "eyes":
            return CompletionResult(
                content=f"OCR_TEXT: {self.ocr}\nSCENE: {self.scene}", model="fake-eyes-model:free"
            )
        return CompletionResult(content="A full analysis of the document.", model="fake-brain-model:free")

    def close(self):
        pass


class _EmptySearcher:
    """No results -> research_node's fetch step never runs -> no network."""

    def text(self, query, **kwargs):
        return []


class _FakeTtsProvider:
    """Same minimal double test_graph.py/test_vision.py use for tts_node's
    provider seam - writes a minimal valid-looking mp3 (an ID3 tag) so
    run_tts's own "looks like audio" check passes, without touching the
    network via a real EdgeTtsProvider."""

    def synthesize(self, text, out_path):
        with open(out_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 32)


SHORT_OCR_TEXT = "short text"
LONG_OCR_TEXT = " ".join(["x"] * 200)


def _node_statuses(updates):
    """The subsequence of yielded status strings that are node-narration
    phrases, in the order they were yielded - excludes STATUS_WORKING (the
    unconditional first yield) and the two final-result yields, which use
    status_for_result's vocabulary, not the node-phrase vocabulary."""
    node_phrases = {STATUS_NODE_RESEARCH, STATUS_NODE_WRITING, STATUS_NODE_TTS}
    return [status for status, _audio, _text in updates if status in node_phrases]


def test_staged_submit_narrates_each_node_transition_in_order_fast_path():
    client = _RoutingVisionClient(SHORT_OCR_TEXT, "a room")
    resources = AppResources(
        graph=build_graph(),
        client=client,
        client_error=None,
        tts_providers=[_FakeTtsProvider()],
        searcher=_EmptySearcher(),
        research_client=None,
    )

    updates = list(handle_submit_staged(FakeImage(), resources))

    assert updates[0][0] == STATUS_WORKING
    # Fast path executes vision -> fast_synth -> tts. vision itself gets no
    # dedicated phrase (nothing precedes it in the stream to trigger one -
    # STATUS_WORKING already covers it, see clarif_eye.ui's STATUS_NODE_*
    # comment); a phrase is announced for whatever node comes next each
    # time one completes: fast_synth's phrase when vision finishes, tts's
    # phrase when fast_synth finishes.
    assert _node_statuses(updates) == [STATUS_NODE_WRITING, STATUS_NODE_TTS]


def test_staged_submit_narrates_each_node_transition_in_order_deep_path():
    client = _RoutingVisionClient(LONG_OCR_TEXT, "a busy scene")
    resources = AppResources(
        graph=build_graph(),
        client=client,
        client_error=None,
        tts_providers=[_FakeTtsProvider()],
        searcher=_EmptySearcher(),
        research_client=None,
    )

    updates = list(handle_submit_staged(FakeImage(), resources))

    assert updates[0][0] == STATUS_WORKING
    # Deep path executes vision -> research -> analysis -> tts; same
    # successor-announcement pattern as the fast path above.
    assert _node_statuses(updates) == [
        STATUS_NODE_RESEARCH,
        STATUS_NODE_WRITING,
        STATUS_NODE_TTS,
    ]


# --- Checkpointed threads, driven through the real UI seam (issue #81 / -----
# P9.2, simplify-gate follow-up)
#
# tests/test_checkpointing.py exercises the reducer/checkpointer mechanism
# directly against a compiled graph (graph.invoke + graph.update_state).
# Nothing there drives the ACTUAL seam a real browser session uses:
# build_interface -> handle_submit_staged -> _run_pipeline_events, with a
# real ThreadRegistry doing the touching and a real thread_id flowing all
# the way through config["configurable"]. This test closes that gap - same
# real-compiled-graph pattern the two tests above already use (not
# FakeGraph, which only implements .invoke() and has no checkpointer to
# exercise).
def test_handle_submit_staged_accumulates_messages_on_one_thread_and_isolates_another():
    from langgraph.checkpoint.memory import InMemorySaver

    from clarif_eye.ui import ThreadRegistry

    checkpointer = InMemorySaver()
    thread_registry = ThreadRegistry(checkpointer)
    resources = AppResources(
        graph=build_graph(checkpointer=checkpointer),
        client=_RoutingVisionClient(SHORT_OCR_TEXT, "a room"),
        client_error=None,
        tts_providers=[_FakeTtsProvider()],
        searcher=_EmptySearcher(),
        research_client=None,
        thread_registry=thread_registry,
    )

    thread_id = "session-under-test"
    # Distinct image content per call (issue #75's cache keys on DECODED
    # image bytes) - two calls with the SAME content would cache-hit on the
    # second and never reach the graph a second time, which would prove
    # nothing about accumulation.
    list(handle_submit_staged(FakeImage(content=b"turn-one-bytes"), resources, thread_id=thread_id))
    list(handle_submit_staged(FakeImage(content=b"turn-two-bytes"), resources, thread_id=thread_id))

    state = resources.graph.get_state({"configurable": {"thread_id": thread_id}})
    assert len(state.values["messages"]) == 2

    # The registry actually saw the thread (issue #81's ThreadRegistry /
    # thread_configurable chokepoint), not just the checkpointer.
    assert thread_id in thread_registry._entries

    # A different thread_id sees no bleed-through, driven through the same
    # real seam.
    other_thread_id = "another-session"
    list(handle_submit_staged(FakeImage(content=b"other-session-bytes"), resources, thread_id=other_thread_id))
    other_state = resources.graph.get_state({"configurable": {"thread_id": other_thread_id}})
    assert len(other_state.values["messages"]) == 1
