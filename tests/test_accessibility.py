"""Accessibility tests for the Gradio UI (issue #15 / P5.1).

Owner decision D13: this phase is satisfied by AUTOMATED checks only - no
manual screen-reader pass happens as part of this work, so every claim in
this file (and in clarif_eye.ui) is "machine-verified", never "screen-reader
tested". These tests build the real gr.Blocks object via
clarif_eye.ui.build_interface() and walk its component tree; they NEVER
call .launch() and NEVER touch the network, matching tests/test_ui.py's
existing discipline for clarif_eye.ui.

Every user of this product is visually impaired (see app.py/ui.py module
docstrings) and the pipeline can take ~15-30s (measured in issue #17) with
nothing visible to look at - that silence is the accessibility problem this
issue exists to close, via an ARIA live-region status announcement and a
guaranteed-populated text fallback.
"""

import gradio as gr

from clarif_eye import tts as tts_module
from clarif_eye.ui import (
    AppResources,
    AUDIO_UNAVAILABLE_NOTE,
    ARIA_LIVE_HEAD,
    FOCUS_RESULT_JS,
    RESULT_ELEM_ID,
    STATUS_DEGRADED,
    STATUS_ELEM_CLASSES,
    STATUS_ELEM_ID,
    STATUS_IDLE,
    STATUS_SUCCESS_AUDIO,
    STATUS_SUCCESS_TEXT_ONLY,
    STATUS_WORKING,
    build_interface,
    build_resources,
    handle_submit_staged,
    status_for_result,
)


class FakeImage:
    """Stand-in for a PIL Image good enough for base64 encoding."""

    def __init__(self, mode="RGB"):
        self.mode = mode

    def convert(self, mode):
        return self

    def save(self, buf, format=None):
        buf.write(b"\xff\xd8\xff\xe0fakejpegbytes")


class FakeGraph:
    """Records the config it was invoked with and returns a canned result."""

    def __init__(self, result=None):
        self.result = result or {}
        self.invocations = []

    def invoke(self, state, config=None):
        self.invocations.append({"state": state, "config": config})
        return self.result


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
    tts_module._last_result_set(None)


# --- status_for_result: the structural 3-way split ------------------------
#
# Mirrors clarif_eye.ui's existing THREE OUTCOMES docstring exactly: told
# apart by audio_path truthiness and the chain_exhausted bool - never by
# matching a message string.


def test_status_success_with_audio():
    assert status_for_result("/tmp/out.mp3", chain_exhausted=False) == STATUS_SUCCESS_AUDIO


def test_status_success_text_only_when_chain_exhausted():
    # Structurally derived from the chain_exhausted bool (issue #15 scope
    # item 6), not by scanning any text for a substring.
    status = status_for_result(None, chain_exhausted=True)
    assert status == STATUS_SUCCESS_TEXT_ONLY
    assert AUDIO_UNAVAILABLE_NOTE in status


def test_status_degraded_upstream_when_neither_audio_nor_exhausted():
    assert status_for_result(None, chain_exhausted=False) == STATUS_DEGRADED


def test_status_idle_and_working_are_distinct_nonempty_strings():
    assert STATUS_IDLE
    assert STATUS_WORKING
    assert STATUS_IDLE != STATUS_WORKING
    # Owner-provided honesty requirement: no invented percentages, just an
    # honest ceiling drawn from issue #17's measured numbers.
    assert "30 seconds" in STATUS_WORKING


# --- handle_submit_staged: two-stage yield (received+working, then result) -


def test_staged_submit_yields_working_then_success_with_audio():
    tts_module._last_result_set(
        tts_module.TtsResult("/tmp/out.mp3", (tts_module.ProviderAttempt("Edge", "success", ""),), "Edge")
    )
    graph = FakeGraph(result={"final_output": "A cat sits on a mat.", "audio_file_path": "/tmp/out.mp3"})
    resources = _resources(graph)

    updates = list(handle_submit_staged(FakeImage(), resources))

    assert len(updates) == 2
    first_status, first_audio, first_text = updates[0]
    assert first_status == STATUS_WORKING
    assert first_audio is None

    final_status, final_audio, final_text = updates[1]
    assert final_status == STATUS_SUCCESS_AUDIO
    assert final_audio == "/tmp/out.mp3"
    assert final_text == "A cat sits on a mat."


def test_staged_submit_announces_audio_unavailable_when_chain_exhausted():
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

    *_, (final_status, final_audio, final_text) = handle_submit_staged(FakeImage(), resources)

    assert final_audio is None
    assert AUDIO_UNAVAILABLE_NOTE in final_status
    assert "A cat sits on a mat." in final_text


def test_staged_submit_degraded_when_no_image_supplied():
    graph = FakeGraph(result={"final_output": "should not be reached"})
    resources = _resources(graph)

    *_, (final_status, final_audio, final_text) = handle_submit_staged(None, resources)

    assert final_status == STATUS_DEGRADED
    assert final_audio is None
    assert graph.invocations == []


# --- build_interface: builds real Blocks, never launches ------------------


def test_build_interface_returns_blocks_without_launching():
    resources = _resources(FakeGraph())
    demo = build_interface(resources)
    try:
        assert isinstance(demo, gr.Blocks)
        # No server has been started: launch()'s local_url is only set
        # once a server actually starts listening.
        assert demo.local_url is None
        assert demo.server is None
    finally:
        demo.close()


def _components(demo):
    return list(demo.blocks.values())


def test_every_interactive_component_has_a_nonempty_accessible_name():
    resources = _resources(FakeGraph())
    demo = build_interface(resources)
    try:
        interactive_types = (gr.Image, gr.Button, gr.Audio, gr.Textbox)
        found = []
        for component in _components(demo):
            if not isinstance(component, interactive_types):
                continue
            name = getattr(component, "label", None) or (
                component.value if isinstance(component, gr.Button) else None
            )
            found.append((type(component).__name__, name))
            assert name, f"{type(component).__name__} has no accessible name (label)"
        # Sanity: we actually walked the controls we expect to exist.
        assert len(found) >= 4
    finally:
        demo.close()


def test_live_region_has_expected_elem_id_and_classes():
    resources = _resources(FakeGraph())
    demo = build_interface(resources)
    try:
        status_components = [c for c in _components(demo) if getattr(c, "elem_id", None) == STATUS_ELEM_ID]
        assert len(status_components) == 1
        status_component = status_components[0]
        assert list(status_component.elem_classes) == list(STATUS_ELEM_CLASSES)
        assert status_component.value == STATUS_IDLE
        # The live region itself must never be reachable via tab focus in a
        # way that steals focus from the user's flow - it is read-only.
        assert status_component.interactive is False
    finally:
        demo.close()


def test_result_textbox_has_focus_target_elem_id():
    resources = _resources(FakeGraph())
    demo = build_interface(resources)
    try:
        result_components = [c for c in _components(demo) if getattr(c, "elem_id", None) == RESULT_ELEM_ID]
        assert len(result_components) == 1
        assert result_components[0].label
    finally:
        demo.close()


def test_focus_and_aria_live_js_target_the_right_elements():
    # Structural wiring checks, not a browser test (no server/network here,
    # see module docstring) - these assert the injected JS targets the
    # SAME ids the components declare, so the two can't silently drift.
    assert RESULT_ELEM_ID in FOCUS_RESULT_JS
    assert STATUS_ELEM_ID in ARIA_LIVE_HEAD
    assert "aria-live" in ARIA_LIVE_HEAD
