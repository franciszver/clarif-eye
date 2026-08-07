"""Tests for the total-pipeline deadline (issue P6.1 / #17).

D16 gave each role (eyes/brain) a per-ATTEMPT-spanning budget inside
client.complete, but nothing bounded the whole graph run - a live
measurement found a research-path request taking 99.0s end to end even
though the per-role ceilings alone (30 + 45) should have been the worst
case. This file pins the fix: a deadline threaded through
config["configurable"]["deadline"] (an absolute time.monotonic()
timestamp), checked at the top of every node that makes a model or
network call. When it has already passed, that node skips its own
expensive work and returns a state update built from whatever is already
known, instead of trying (and probably failing/timing out) anyway.

No deadline key at all means unbounded, exactly today's behavior - every
existing test in the suite that never sets one is the regression guard
for that (CHECK G in the P6.1 issue).

Also covers: the scraped-context cap (analysis._SCRAPER_DATA_CAP) is
configurable via config["configurable"]["scraper_data_cap"] instead of a
hardcoded 4000.
"""

import time

from clarif_eye import analysis, research, synth, vision
from clarif_eye.client import CompletionResult
from clarif_eye.graph import (
    analysis_node,
    build_graph,
    fast_synth_node,
    research_node,
    vision_node,
)
from clarif_eye.state import make_initial_state

from tests._stream_helpers import drain_stream_collecting_trace


def _reply(ocr, scene):
    return f"OCR_TEXT: {ocr}\nSCENE: {scene}"


# 200 words, no data-density signals - trips only the router's
# long-document word-count fallback, same fixture test_graph.py uses to
# force the research path deterministically.
LONG_OCR_TEXT = " ".join(["x"] * 200)


class ExplosiveClient:
    """A client seam that RECORDS whether it was ever called, rather than
    raising - deliberately not exception-based, since every node here
    already catches a raising client and degrades anyway (that generic
    exception-handling path would otherwise mask a missing deadline
    check: the test would still see a non-empty final_output and pass for
    the wrong reason). Recording `.called_roles` lets a test assert the
    call never happened at all, which only a real skip can satisfy."""

    def __init__(self):
        self.called_roles = []

    def complete(self, role, messages, **params):
        self.called_roles.append(role)
        return CompletionResult(content="a model reply that must never be used", model="fake:free")

    def close(self):
        pass


class ExplosiveSearcher:
    """Same reasoning as ExplosiveClient: records rather than raises, so a
    missing skip is caught by a call-count assertion, not masked by
    exception handling elsewhere in the node."""

    def __init__(self):
        self.called = False

    def text(self, query, **kwargs):
        self.called = True
        return []


class FakeEyesClient:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def complete(self, role, messages, **params):
        self.calls.append(role)
        return CompletionResult(content=self.content, model="fake-eyes-model:free")

    def close(self):
        pass


class SlowSearcher:
    """Returns no results, but takes real wall-clock time first - used to
    let a short deadline expire BETWEEN two node checks without any real
    network call."""

    def __init__(self, delay):
        self.delay = delay

    def text(self, query, **kwargs):
        time.sleep(self.delay)
        return []


class _FakeTtsProvider:
    def synthesize(self, text, out_path):
        with open(out_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 32)


# --- Unit level: each node's run_* function skips its own call ------------


def test_run_vision_skips_model_call_when_deadline_exceeded():
    client = ExplosiveClient()
    result = vision.run_vision("imgdata", client=client, deadline_exceeded=True)

    assert client.called_roles == []
    assert result["scene_context"] == vision.DEGRADED_DEADLINE_EXCEEDED
    assert result["ocr_output"] == ""
    assert result["complexity_flag"] is False


def test_vision_deadline_message_is_recognised_as_degraded():
    assert vision.is_degraded_scene(vision.DEGRADED_DEADLINE_EXCEEDED) is True


def test_run_research_skips_search_when_deadline_exceeded():
    searcher = ExplosiveSearcher()
    result = research.run_research(
        "some query text", "a scene", searcher=searcher, client=None, deadline_exceeded=True
    )

    assert searcher.called is False
    assert result == {"scraper_data": ""}


def test_run_fast_synth_builds_from_known_state_when_deadline_exceeded():
    client = ExplosiveClient()
    result = synth.run_fast_synth(
        "account 12345 balance $9.00", "a bill", client=client, deadline_exceeded=True
    )

    assert client.called_roles == []
    final_output = result["final_output"]
    assert final_output != ""
    assert "12345" in final_output or "9.00" in final_output


def test_run_analysis_builds_from_known_state_when_deadline_exceeded():
    client = ExplosiveClient()
    result = analysis.run_analysis(
        "account 12345 balance $9.00",
        "a bill",
        "some scraped web context",
        client=client,
        deadline_exceeded=True,
    )

    assert client.called_roles == []
    final_output = result["final_output"]
    assert final_output != ""
    assert "12345" in final_output or "9.00" in final_output


# --- CHECK B: deadline already blown before the graph starts --------------


def test_graph_with_deadline_already_blown_still_produces_usable_output():
    graph = build_graph()
    state = make_initial_state("imgdata")
    client = ExplosiveClient()
    searcher = ExplosiveSearcher()
    config = {
        "configurable": {
            "client": client,
            "searcher": searcher,
            "tts_provider": _FakeTtsProvider(),
            "deadline": time.monotonic() - 1.0,
        }
    }

    result, trace = drain_stream_collecting_trace(graph, state, config)

    assert client.called_roles == []
    assert searcher.called is False
    assert result["final_output"] != ""
    assert result["final_output"] == vision.DEGRADED_DEADLINE_EXCEEDED
    assert result["audio_file_path"] != ""
    assert "vision" in trace
    assert "fast_synth" in trace
    assert "tts" in trace
    # The fast path is what a skipped vision routes to, so research/analysis
    # never even run - nothing left to skip there.
    assert "research" not in trace
    assert "analysis" not in trace


# --- CHECK C: deadline blown midway, after vision, before analysis --------


def test_deadline_blown_midway_produces_output_from_known_state():
    graph = build_graph()
    state = make_initial_state("imgdata")
    eyes_client = FakeEyesClient(_reply(LONG_OCR_TEXT, "a dense long document"))
    config = {
        "configurable": {
            "client": eyes_client,
            "searcher": SlowSearcher(delay=0.05),
            "tts_provider": _FakeTtsProvider(),
            # Long enough for vision's near-instant fake call, short enough
            # that SlowSearcher's sleep pushes the clock past it before
            # analysis_node checks again.
            "deadline": time.monotonic() + 0.02,
        }
    }

    result, trace = drain_stream_collecting_trace(graph, state, config)

    assert "vision" in trace
    assert "research" in trace
    assert "analysis" in trace
    assert eyes_client.calls == ["eyes"]  # brain was never called
    final_output = result["final_output"]
    assert final_output != ""
    assert "not be prepared" not in final_output  # not a generic error message
    assert LONG_OCR_TEXT.split()[0] in final_output or "dense long document" in final_output


# --- CHECK D: a generous deadline runs the full-quality path unskipped ----


def test_generous_deadline_runs_full_quality_path_unskipped():
    graph = build_graph()
    state = make_initial_state("imgdata")

    class RoleAwareClient:
        def __init__(self):
            self.calls = []

        def complete(self, role, messages, **params):
            self.calls.append(role)
            if role == "eyes":
                return CompletionResult(
                    content=_reply(LONG_OCR_TEXT, "a dense long document"), model="fake-eyes:free"
                )
            return CompletionResult(content="Full quality analysis of the document.", model="fake-brain:free")

        def close(self):
            pass

    client = RoleAwareClient()
    config = {
        "configurable": {
            "client": client,
            "searcher": SlowSearcher(delay=0.0),
            "tts_provider": _FakeTtsProvider(),
            "deadline": time.monotonic() + 1000.0,
        }
    }

    result, trace = drain_stream_collecting_trace(graph, state, config)

    assert client.calls == ["eyes", "brain"]
    assert "Full quality analysis" in result["final_output"]
    assert "vision" in trace and "research" in trace and "analysis" in trace and "tts" in trace


# --- CHECK G: no deadline key at all behaves exactly like before ----------


def test_no_deadline_key_present_calls_model_normally():
    config = {"configurable": {}}
    state = make_initial_state("imgdata")
    state["image_data"] = "imgdata"

    client = FakeEyesClient(_reply("some text", "a scene"))
    result = vision_node(state, config=config, client=client)

    assert client.calls == ["eyes"]
    assert result["scene_context"] == "a scene"


# --- Node-level regression guard for CHECK F (mutation test target) -------
# Each of these directly proves a specific node CHECKS the deadline itself,
# so removing that check from any one node is independently caught.


def test_vision_node_checks_deadline_itself():
    client = ExplosiveClient()
    config = {"configurable": {"deadline": time.monotonic() - 1.0, "client": client}}
    state = make_initial_state("imgdata")

    result = vision_node(state, config=config)

    assert client.called_roles == []
    assert result["scene_context"] == vision.DEGRADED_DEADLINE_EXCEEDED


def test_research_node_checks_deadline_itself():
    searcher = ExplosiveSearcher()
    config = {
        "configurable": {
            "deadline": time.monotonic() - 1.0,
            "searcher": searcher,
            "research_client": None,
        }
    }
    state = make_initial_state("imgdata")
    state["ocr_output"] = "some query text"
    state["scene_context"] = "a scene"

    result = research_node(state, config=config)

    assert searcher.called is False
    assert result == {"scraper_data": ""}


def test_fast_synth_node_checks_deadline_itself():
    client = ExplosiveClient()
    config = {"configurable": {"deadline": time.monotonic() - 1.0, "client": client}}
    state = make_initial_state("imgdata")
    state["ocr_output"] = "account 12345"
    state["scene_context"] = "a bill"

    result = fast_synth_node(state, config=config)

    assert client.called_roles == []
    assert result["final_output"] != ""


def test_analysis_node_checks_deadline_itself():
    client = ExplosiveClient()
    config = {"configurable": {"deadline": time.monotonic() - 1.0, "client": client}}
    state = make_initial_state("imgdata")
    state["ocr_output"] = "account 12345"
    state["scene_context"] = "a bill"
    state["scraper_data"] = "some scrape"

    result = analysis_node(state, config=config)

    assert client.called_roles == []
    assert result["final_output"] != ""


# --- CHECK E: the scraped-context cap is config-driven ---------------------


def test_scraper_data_cap_is_configurable_via_build_messages():
    scrape = "word " * 2000  # 10000 chars, well over any cap below

    small = analysis._build_messages("ocr", "scene", scrape, cap=200)
    large = analysis._build_messages("ocr", "scene", scrape, cap=8000)

    small_text = small[0]["content"][0]["text"]
    large_text = large[0]["content"][0]["text"]

    assert len(small_text) < len(large_text)


def test_run_analysis_honours_scraper_data_cap_param():
    scrape = "word " * 2000
    small_client = analysis  # placeholder, replaced below

    class RecordingClient:
        def __init__(self):
            self.messages = None

        def complete(self, role, messages, **params):
            self.messages = messages
            return CompletionResult(content="A short verified reply.", model="fake-brain:free")

        def close(self):
            pass

    small_client = RecordingClient()
    analysis.run_analysis("ocr text", "a scene", scrape, client=small_client, scraper_data_cap=200)

    large_client = RecordingClient()
    analysis.run_analysis("ocr text", "a scene", scrape, client=large_client, scraper_data_cap=8000)

    small_len = len(small_client.messages[0]["content"][0]["text"])
    large_len = len(large_client.messages[0]["content"][0]["text"])
    assert small_len < large_len


def test_analysis_node_reads_scraper_data_cap_from_config():
    scrape = "word " * 2000

    class RecordingClient:
        def __init__(self):
            self.messages = None

        def complete(self, role, messages, **params):
            self.messages = messages
            return CompletionResult(content="A short verified reply.", model="fake-brain:free")

        def close(self):
            pass

    client = RecordingClient()
    config = {"configurable": {"client": client, "scraper_data_cap": 300}}
    state = make_initial_state("imgdata")
    state["ocr_output"] = "ocr text"
    state["scene_context"] = "a scene"
    state["scraper_data"] = scrape

    analysis_node(state, config=config)

    text = client.messages[0]["content"][0]["text"]
    # Cap is 300 (plus the fixed prompt/preamble text), far below the
    # ~10000-char raw scrape, so the truncation marker must appear.
    assert "[context truncated]" in text
