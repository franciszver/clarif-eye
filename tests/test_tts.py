"""Tests for the TTS node (issue #11 / P3.1): final_output -> an audio file.

No network calls: every test substitutes a fake provider object for
EdgeTtsProvider (the provider itself is the seam, same pattern
test_synth.py/test_vision.py use for their fake OpenRouterClient). Covers
the provider seam's TtsError contract, bounded/unique output files, and
every failure path leaving audio_file_path == "" so the graph can still
reach END without audio.
"""

import re

import pytest

from clarif_eye.graph import build_graph, tts_node
from clarif_eye.state import make_initial_state
from clarif_eye.tts import (
    DEFAULT_PROVIDER_CHAIN,
    MAX_KEPT_FILES,
    OUTCOME_ERROR,
    OUTCOME_INVALID_AUDIO,
    OUTCOME_SUCCESS,
    EdgeTtsProvider,
    GttsProvider,
    TtsError,
    get_last_tts_result,
    is_chain_exhausted,
    run_tts,
)

# Minimal valid-looking mp3 payloads for the "looks like audio" check:
# an ID3v2 tag header, and a raw MPEG frame sync (0xFF 0xFB...).
_ID3_AUDIO_BYTES = b"ID3" + b"\x03\x00\x00\x00\x00\x00\x21" + b"\x00" * 64
_FRAME_SYNC_AUDIO_BYTES = b"\xff\xfb\x90\x00" + b"\x00" * 64


# --- Fake provider, same shape as test_synth.py's FakeSynthClient ----------


class FakeTtsProvider:
    def __init__(self, content=_ID3_AUDIO_BYTES, exc=None, write_file=True, name=None):
        self.content = content
        self.exc = exc
        self.write_file = write_file
        self.name = name
        self.calls = []

    def synthesize(self, text, out_path):
        self.calls.append({"text": text, "out_path": out_path})
        if self.exc is not None:
            raise self.exc
        if self.write_file:
            with open(out_path, "wb") as f:
                f.write(self.content)


# --- Provider seam: TtsError is the typed failure contract ------------------


def test_tts_error_is_an_exception_subclass():
    assert issubclass(TtsError, Exception)


# --- Happy path ---------------------------------------------------------------


def test_happy_path_writes_a_non_empty_file_and_returns_its_path(tmp_path):
    provider = FakeTtsProvider(content=_ID3_AUDIO_BYTES)

    result = run_tts("The image shows a coffee cup.", provider=provider, out_dir=tmp_path)

    assert "audio_file_path" in result
    path = result["audio_file_path"]
    assert path != ""
    assert tmp_path.joinpath(*[]) or True  # sanity: tmp_path usable
    from pathlib import Path

    p = Path(path)
    assert p.exists()
    assert p.stat().st_size > 0
    assert p.suffix == ".mp3"
    assert len(provider.calls) == 1
    assert provider.calls[0]["text"] == "The image shows a coffee cup."


def test_happy_path_with_frame_sync_audio_bytes_succeeds(tmp_path):
    provider = FakeTtsProvider(content=_FRAME_SYNC_AUDIO_BYTES)

    result = run_tts("Some spoken text.", provider=provider, out_dir=tmp_path)

    assert result["audio_file_path"] != ""


# --- Unique filenames across calls -------------------------------------------


def test_unique_filenames_across_multiple_calls(tmp_path):
    provider = FakeTtsProvider(content=_ID3_AUDIO_BYTES)

    paths = {
        run_tts("Some text.", provider=provider, out_dir=tmp_path)["audio_file_path"]
        for _ in range(5)
    }

    assert len(paths) == 5
    assert all(p != "" for p in paths)


# --- Bounded file accumulation ------------------------------------------------


def test_file_accumulation_is_bounded_to_max_kept_files(tmp_path):
    provider = FakeTtsProvider(content=_ID3_AUDIO_BYTES)

    for _ in range(MAX_KEPT_FILES + 15):
        result = run_tts("Some text.", provider=provider, out_dir=tmp_path)
        assert result["audio_file_path"] != ""

    mp3_files = list(tmp_path.glob("*.mp3"))
    assert len(mp3_files) == MAX_KEPT_FILES


def test_most_recent_file_survives_pruning(tmp_path):
    provider = FakeTtsProvider(content=_ID3_AUDIO_BYTES)

    last_path = None
    for _ in range(MAX_KEPT_FILES + 5):
        last_path = run_tts("Some text.", provider=provider, out_dir=tmp_path)["audio_file_path"]

    from pathlib import Path

    assert Path(last_path).exists()


# --- Failure: provider raises TtsError ---------------------------------------


def test_provider_raising_tts_error_degrades_to_empty_path(tmp_path):
    provider = FakeTtsProvider(exc=TtsError("synthesis backend unavailable"))

    result = run_tts("Some text.", provider=provider, out_dir=tmp_path)

    assert result == {"audio_file_path": ""}


# --- Failure: provider produces no file --------------------------------------


def test_provider_producing_no_file_degrades_to_empty_path(tmp_path):
    provider = FakeTtsProvider(write_file=False)

    result = run_tts("Some text.", provider=provider, out_dir=tmp_path)

    assert result == {"audio_file_path": ""}


# --- Failure: provider produces an EMPTY file ---------------------------------


def test_provider_producing_an_empty_file_degrades_to_empty_path(tmp_path):
    provider = FakeTtsProvider(content=b"")

    result = run_tts("Some text.", provider=provider, out_dir=tmp_path)

    assert result == {"audio_file_path": ""}


def test_provider_producing_a_non_audio_file_degrades_to_empty_path(tmp_path):
    provider = FakeTtsProvider(content=b"not really audio, just some bytes")

    result = run_tts("Some text.", provider=provider, out_dir=tmp_path)

    assert result == {"audio_file_path": ""}


# --- Failure: blank/empty final_output ----------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_final_output_degrades_without_calling_provider(tmp_path, blank):
    provider = FakeTtsProvider(content=_ID3_AUDIO_BYTES)

    result = run_tts(blank, provider=provider, out_dir=tmp_path)

    assert result == {"audio_file_path": ""}
    assert len(provider.calls) == 0


# --- Failure: unexpected exception types --------------------------------------


@pytest.mark.parametrize("exc", [ValueError("bad"), TimeoutError("timed out"), RuntimeError("oops")])
def test_unexpected_exception_types_degrade_without_raising(tmp_path, exc):
    provider = FakeTtsProvider(exc=exc)

    result = run_tts("Some text.", provider=provider, out_dir=tmp_path)

    assert result == {"audio_file_path": ""}


def test_keyboard_interrupt_is_not_swallowed(tmp_path):
    provider = FakeTtsProvider(exc=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        run_tts("Some text.", provider=provider, out_dir=tmp_path)


def test_system_exit_is_not_swallowed(tmp_path):
    provider = FakeTtsProvider(exc=SystemExit())

    with pytest.raises(SystemExit):
        run_tts("Some text.", provider=provider, out_dir=tmp_path)


# --- tts_node: provider injection, graph-facing wrapper ----------------------


def test_tts_node_accepts_an_explicit_injected_provider(tmp_path):
    provider = FakeTtsProvider(content=_ID3_AUDIO_BYTES)
    state = {"final_output": "Some spoken text."}

    result = tts_node(state, provider=provider, config={"configurable": {"tts_out_dir": tmp_path}})

    assert result["audio_file_path"] != ""
    assert len(provider.calls) == 1


def test_tts_node_accepts_provider_injected_via_config_configurable(tmp_path):
    provider = FakeTtsProvider(content=_ID3_AUDIO_BYTES)
    state = {"final_output": "Some spoken text."}

    result = tts_node(
        state,
        config={"configurable": {"tts_provider": provider, "tts_out_dir": tmp_path}},
    )

    assert result["audio_file_path"] != ""
    assert len(provider.calls) == 1


def test_tts_node_degrades_cleanly_when_provider_raises(tmp_path):
    provider = FakeTtsProvider(exc=TtsError("boom"))
    state = {"final_output": "Some spoken text."}

    result = tts_node(state, provider=provider, config={"configurable": {"tts_out_dir": tmp_path}})

    # issue #81 / P9.2: tts_node also appends one messages entry per run
    # (see clarif_eye.graph.tts_node's docstring) regardless of whether the
    # provider itself succeeded or degraded to "" - the run still produced
    # a final_output that was shown/spoken to the user.
    assert result == {
        "audio_file_path": "",
        "messages": [{"role": "assistant", "content": "Some spoken text."}],
    }


# --- Full compiled graph, real nodes, fake client + fake provider ----------


def test_full_compiled_graph_runs_end_to_end_on_fast_path_with_fake_provider(tmp_path):
    from clarif_eye.client import CompletionResult

    vision_reply = "OCR_TEXT: Open 9-5\nSCENE: a shop front"

    class FakeVisionThenSynthClient:
        def __init__(self):
            self.calls = []

        def complete(self, role, messages, **params):
            self.calls.append(role)
            if len(self.calls) == 1:
                return CompletionResult(content=vision_reply, model="fake-eyes")
            return CompletionResult(
                content="The image shows a sign reading Open 9 to 5 on a shop front.",
                model="fake-eyes",
            )

        def close(self):
            pass

    client = FakeVisionThenSynthClient()
    provider = FakeTtsProvider(content=_ID3_AUDIO_BYTES)
    graph = build_graph()
    state = make_initial_state("base64data")

    result = graph.invoke(
        state,
        config={
            "configurable": {
                "client": client,
                "tts_provider": provider,
                "tts_out_dir": tmp_path,
            }
        },
    )

    assert result["audio_file_path"] != ""
    from pathlib import Path

    assert Path(result["audio_file_path"]).exists()


def test_full_compiled_graph_runs_end_to_end_on_research_path_with_fake_provider(tmp_path):
    from clarif_eye.client import CompletionResult

    LONG_OCR_TEXT = " ".join(["x"] * 200)
    vision_reply = f"OCR_TEXT: {LONG_OCR_TEXT}\nSCENE: a busy scene"

    class FakeVisionThenAnalysisClient:
        def __init__(self):
            self.calls = []

        def complete(self, role, messages, **params):
            self.calls.append(role)
            if len(self.calls) == 1:
                return CompletionResult(content=vision_reply, model="fake-eyes")
            return CompletionResult(
                content="A dense document describing many things.",
                model="fake-brain",
            )

        def close(self):
            pass

    class FakeSearcher:
        def text(self, query, max_results=1):
            return []

    client = FakeVisionThenAnalysisClient()
    provider = FakeTtsProvider(content=_ID3_AUDIO_BYTES)
    graph = build_graph()
    state = make_initial_state("base64data")

    result = graph.invoke(
        state,
        config={
            "configurable": {
                "client": client,
                "searcher": FakeSearcher(),
                "tts_provider": provider,
                "tts_out_dir": tmp_path,
            }
        },
    )

    assert result["complexity_flag"] is True
    assert result["audio_file_path"] != ""


# --- Sanity: output filenames don't collide with mp3-glob TTS-safety re -----
_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}\.mp3$")


def test_output_filename_is_a_uuid_hex_mp3(tmp_path):
    provider = FakeTtsProvider(content=_ID3_AUDIO_BYTES)

    result = run_tts("Some text.", provider=provider, out_dir=tmp_path)

    from pathlib import Path

    assert _UUID_HEX_RE.match(Path(result["audio_file_path"]).name)


# --- Provider chain (issue #12 / P3.2): try in order, first success wins ----


def test_default_provider_chain_has_two_independent_providers():
    assert DEFAULT_PROVIDER_CHAIN == (EdgeTtsProvider, GttsProvider)


def test_chain_tries_next_provider_when_first_fails(tmp_path):
    first = FakeTtsProvider(exc=TtsError("first provider down"), name="first")
    second = FakeTtsProvider(content=_ID3_AUDIO_BYTES, name="second")

    result = run_tts("Some text.", providers=[first, second], out_dir=tmp_path)

    assert result["audio_file_path"] != ""
    assert len(first.calls) == 1
    assert len(second.calls) == 1

    last = get_last_tts_result()
    assert last.provider == "second"
    assert last.audio_file_path == result["audio_file_path"]
    assert [a.provider for a in last.attempts] == ["first", "second"]
    assert last.attempts[0].outcome == OUTCOME_ERROR
    assert last.attempts[1].outcome == OUTCOME_SUCCESS


def test_chain_stops_at_first_success_second_provider_never_called(tmp_path):
    first = FakeTtsProvider(content=_ID3_AUDIO_BYTES, name="first")
    second = FakeTtsProvider(content=_ID3_AUDIO_BYTES, name="second")

    run_tts("Some text.", providers=[first, second], out_dir=tmp_path)

    assert len(first.calls) == 1
    assert len(second.calls) == 0


def test_chain_all_providers_fail_returns_empty_path_and_is_observably_exhausted(tmp_path):
    first = FakeTtsProvider(exc=TtsError("first down"), name="first")
    second = FakeTtsProvider(exc=TtsError("second down"), name="second")

    result = run_tts("Some text.", providers=[first, second], out_dir=tmp_path)

    assert result == {"audio_file_path": ""}
    last = get_last_tts_result()
    assert is_chain_exhausted(last) is True
    assert is_chain_exhausted() is True
    assert [a.outcome for a in last.attempts] == [OUTCOME_ERROR, OUTCOME_ERROR]


def test_chain_continues_past_a_zero_byte_file_from_first_provider(tmp_path):
    first = FakeTtsProvider(content=b"", name="first")
    second = FakeTtsProvider(content=_ID3_AUDIO_BYTES, name="second")

    result = run_tts("Some text.", providers=[first, second], out_dir=tmp_path)

    assert result["audio_file_path"] != ""
    last = get_last_tts_result()
    assert last.attempts[0].outcome == OUTCOME_INVALID_AUDIO
    assert last.provider == "second"


def test_blank_final_output_is_not_reported_as_chain_exhausted(tmp_path):
    provider = FakeTtsProvider(content=_ID3_AUDIO_BYTES)

    result = run_tts("", providers=[provider], out_dir=tmp_path)

    assert result == {"audio_file_path": ""}
    assert is_chain_exhausted() is False
    assert get_last_tts_result().attempts == ()


def test_single_provider_injection_old_api_still_supported(tmp_path):
    provider = FakeTtsProvider(content=_ID3_AUDIO_BYTES, name="only")

    result = run_tts("Some text.", provider=provider, out_dir=tmp_path)

    assert result["audio_file_path"] != ""
    last = get_last_tts_result()
    assert last.provider == "only"
    assert len(last.attempts) == 1


# --- tts_node: chain injection via config -------------------------------------


def test_tts_node_accepts_providers_chain_injected_via_config(tmp_path):
    first = FakeTtsProvider(exc=TtsError("boom"), name="first")
    second = FakeTtsProvider(content=_ID3_AUDIO_BYTES, name="second")
    state = {"final_output": "Some spoken text."}

    result = tts_node(
        state,
        config={"configurable": {"tts_providers": [first, second], "tts_out_dir": tmp_path}},
    )

    assert result["audio_file_path"] != ""
    assert len(first.calls) == 1
    assert len(second.calls) == 1


# --- Full compiled graph: failing-then-succeeding chain, and full exhaustion -


def test_full_compiled_graph_falls_through_provider_chain(tmp_path):
    from clarif_eye.client import CompletionResult

    vision_reply = "OCR_TEXT: Open 9-5\nSCENE: a shop front"

    class FakeVisionThenSynthClient:
        def __init__(self):
            self.calls = []

        def complete(self, role, messages, **params):
            self.calls.append(role)
            if len(self.calls) == 1:
                return CompletionResult(content=vision_reply, model="fake-eyes")
            return CompletionResult(
                content="The image shows a sign reading Open 9 to 5 on a shop front.",
                model="fake-eyes",
            )

        def close(self):
            pass

    client = FakeVisionThenSynthClient()
    first = FakeTtsProvider(exc=TtsError("first provider down"), name="first")
    second = FakeTtsProvider(content=_ID3_AUDIO_BYTES, name="second")
    graph = build_graph()
    state = make_initial_state("base64data")

    result = graph.invoke(
        state,
        config={
            "configurable": {
                "client": client,
                "tts_providers": [first, second],
                "tts_out_dir": tmp_path,
            }
        },
    )

    assert result["audio_file_path"] != ""
    assert len(first.calls) == 1
    assert len(second.calls) == 1


def test_full_compiled_graph_reaches_end_with_text_only_when_chain_exhausted(tmp_path):
    from clarif_eye.client import CompletionResult

    vision_reply = "OCR_TEXT: Open 9-5\nSCENE: a shop front"

    class FakeVisionThenSynthClient:
        def __init__(self):
            self.calls = []

        def complete(self, role, messages, **params):
            self.calls.append(role)
            if len(self.calls) == 1:
                return CompletionResult(content=vision_reply, model="fake-eyes")
            return CompletionResult(
                content="The image shows a sign reading Open 9 to 5 on a shop front.",
                model="fake-eyes",
            )

        def close(self):
            pass

    client = FakeVisionThenSynthClient()
    first = FakeTtsProvider(exc=TtsError("first down"), name="first")
    second = FakeTtsProvider(exc=TtsError("second down"), name="second")
    graph = build_graph()
    state = make_initial_state("base64data")

    result = graph.invoke(
        state,
        config={
            "configurable": {
                "client": client,
                "tts_providers": [first, second],
                "tts_out_dir": tmp_path,
            }
        },
    )

    assert result["audio_file_path"] == ""
    assert result["final_output"] != ""
    assert is_chain_exhausted() is True
