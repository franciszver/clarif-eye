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
        assert demo.is_running is False
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


def test_aria_live_shim_uses_mutation_observer_not_load_only():
    # Regression test for a real-browser bug found via Chrome DevTools:
    # #status-live-region existed but carried NONE of aria-live/aria-atomic/
    # role. The old shim ran on window "load", which fires BEFORE Gradio's
    # SPA renders its components, so getElementById returned null and the
    # shim was a silent no-op - a screen reader announced nothing during
    # the ~15-30s wait.
    #
    # This asserts the shim observes the DOM for the element to appear
    # (and survives Gradio re-rendering the status control on every run)
    # instead of assuming it exists at a fixed moment - written to FAIL
    # against the old, load-only shim.
    #
    # STATIC/SOURCE-LEVEL CHECK ONLY (same discipline as the rest of this
    # file): this verifies the shim's structure, not that the attributes
    # are actually applied in a rendered browser. That was verified
    # manually with Chrome DevTools for this fix; there is no automated
    # browser test in this suite covering it (Owner decision D13 accepted
    # automated-only coverage for P5.1 - this does not extend that to
    # "browser-verified").
    assert "MutationObserver" in ARIA_LIVE_HEAD
    assert "observe(" in ARIA_LIVE_HEAD
    assert "childList" in ARIA_LIVE_HEAD and "subtree" in ARIA_LIVE_HEAD
    # The bug, precisely: attributes applied ONLY from inside a window
    # "load" handler. A correct shim must not depend solely on that event
    # firing after the element already exists in the DOM.
    assert 'addEventListener("load"' not in ARIA_LIVE_HEAD


def test_description_output_is_made_keyboard_focusable():
    # Regression test for a real-browser bug found via Chrome DevTools
    # (issue #15 keyboard-accessibility follow-up): Gradio renders
    # output-only Textboxes as `disabled`. A disabled form control cannot
    # receive focus and is removed from the tab order entirely, so the
    # description text - the accessible fallback when audio is
    # unavailable - could not be reached by keyboard at all, and
    # FOCUS_RESULT_JS's el.focus() call silently did nothing (focus
    # stayed on the submit button after a run).
    #
    # STATIC/SOURCE-LEVEL CHECK ONLY (same discipline as the rest of this
    # file, see test_aria_live_shim_uses_mutation_observer_not_load_only):
    # this cannot see the rendered DOM or prove the description is
    # actually reachable by pressing Tab in a real browser - only that the
    # mechanism which fixes it is present in the shim source. Keyboard
    # reachability of the description output was verified manually in
    # Chrome DevTools for this fix; there is no automated browser test in
    # this suite covering it (Owner decision D13 accepted automated-only
    # coverage for P5.1 - this does not extend that claim to "browser-
    # verified").
    #
    # Written to FAIL against the pre-fix shim, which only ever touched
    # #status-live-region and never looked at #description-output.
    assert RESULT_ELEM_ID in ARIA_LIVE_HEAD
    assert "disabled" in ARIA_LIVE_HEAD
    assert "readOnly" in ARIA_LIVE_HEAD or "readonly" in ARIA_LIVE_HEAD
    assert "tabindex" in ARIA_LIVE_HEAD.lower()

    # The SAME apply()/MutationObserver pair that tags the live region must
    # do this, not a second observer, so the fix survives Gradio
    # re-rendering the output component's DOM node on every run just like
    # the aria-live tagging already must.
    assert ARIA_LIVE_HEAD.count("MutationObserver") == 1


def test_focus_result_js_is_defensive_against_a_non_focusable_element():
    # FOCUS_RESULT_JS must never throw even if, for any reason, the
    # element it targets can't be focused - a thrown error client-side
    # would be silent to the user but is still a correctness bug.
    assert "try" in FOCUS_RESULT_JS and "catch" in FOCUS_RESULT_JS


# --- P5.2 / issue #16: audit-script surface closable offline ---------------
#
# Everything below is checkable WITHOUT a browser (same "STATIC/SOURCE-LEVEL
# CHECK ONLY" discipline as the rest of this file). Anything that requires a
# rendered DOM or an assistive technology is out of scope here - that is the
# orchestrator's live audit (scripts/audit_accessibility.py's checklist) and
# the human screen-reader pass recorded in prd/DECISIONS.md (D19).

import re


def test_no_positive_tabindex_anywhere_in_injected_markup():
    # A positive tabindex (>0) hijacks the natural tab order, which is a
    # WCAG 2.4.3 violation - the only tabindex this codebase should ever
    # inject is "0" (re-enters the natural tab order without reordering
    # it). Written to FAIL if any positive tabindex sneaks into either
    # piece of injected JS/HTML.
    injected = ARIA_LIVE_HEAD + FOCUS_RESULT_JS
    for value in re.findall(r'tabindex["\'\s:=,]+["\']?(\d+)', injected, re.IGNORECASE):
        assert int(value) == 0, f"positive tabindex found: {value}"


def test_audit_script_exists_and_defines_the_required_checks():
    from scripts import audit_accessibility as audit

    # At least the categories issue #16 explicitly requires: accessible
    # names, live-region wiring, focusable/tab-order description output,
    # no positive tabindex, image alt text, and colour never being the
    # sole carrier of information.
    check_ids = {check["id"] for check in audit.CHECKS}
    required = {
        "accessible-names",
        "live-region",
        "description-focusable",
        "no-positive-tabindex",
        "image-alt-text",
        "color-not-sole-carrier",
    }
    assert required <= check_ids


def test_audit_checklist_stays_in_sync_with_ui_elem_ids():
    # The checklist must reference the ACTUAL elem_ids the UI uses (import
    # identity, not a hand-copied string) so the two can never silently
    # drift apart - written to FAIL if the checklist is hand-authored
    # against stale/copied ids instead of the real constants.
    from scripts import audit_accessibility as audit

    assert audit.STATUS_ELEM_ID is STATUS_ELEM_ID
    assert audit.RESULT_ELEM_ID is RESULT_ELEM_ID

    checklist_text = audit.render_checklist("http://example.invalid")
    assert STATUS_ELEM_ID in checklist_text
    assert RESULT_ELEM_ID in checklist_text


def test_audit_js_payload_checks_for_positive_tabindex_and_missing_names():
    from scripts import audit_accessibility as audit

    payload = audit.js_payload()
    assert "aria-live" in payload
    assert "tabindex" in payload.lower()
    # The payload must inspect accessible names (aria-label / innerText /
    # label text), not just presence of elements.
    assert "aria-label" in payload or "getAccessibleName" in payload


# --- P5.3 / issue #47: audio autoplay talks over the screen reader --------
#
# Real screen-reader use (owner, Narrator on Windows - see
# docs/ACCESSIBILITY.md's "Human screen-reader verified" section) found the
# synthesized audio starting at the same moment the completion status was
# still being spoken, so neither was intelligible. Two changes close this:
# (1) the with-audio announcement is made SHORT, since the audio itself is
# the completion signal and a long announcement is redundant *and* actively
# collides with it; (2) playback is deliberately sequenced to start after a
# delay instead of relying on brevity alone. Both are checked here
# STRUCTURALLY, same discipline as the rest of this file - no string-
# matching the description text, no browser/screen-reader claims.
#
# RED-FIRST: written to fail against the pre-fix code, where
# STATUS_SUCCESS_AUDIO was the long "Description ready. Audio is playing;
# the text is below too." and gr.Audio had no elem_id / used autoplay=True
# with no deferred-play shim at all.


def test_with_audio_status_is_short_while_without_audio_stays_full():
    # The with-audio announcement must be short enough to be "over in about
    # a second" even if it still overlaps slightly with audio starting.
    # The without-audio statuses have no audio to collide with and must
    # stay full, since they are the ONLY signal the user gets.
    assert len(STATUS_SUCCESS_AUDIO) <= 40
    assert len(STATUS_SUCCESS_TEXT_ONLY) > len(STATUS_SUCCESS_AUDIO)
    assert len(STATUS_DEGRADED) > len(STATUS_SUCCESS_AUDIO)


def test_with_audio_status_no_longer_asserts_audio_is_playing_as_fact():
    # "Audio is playing" may be FALSE if the browser blocks programmatic
    # playback (see the sequencing mechanism below) - the wording must be
    # true either way, so it must not claim playback as an accomplished
    # fact.
    assert "is playing" not in STATUS_SUCCESS_AUDIO
    assert "playing" not in STATUS_SUCCESS_AUDIO.lower()


def test_audio_component_does_not_autoplay():
    # Sequencing option (a): autoplay is turned OFF on the component itself
    # so audio never starts the instant a src is set - a JS shim (checked
    # below) starts it deliberately, after a delay, instead.
    resources = _resources(FakeGraph())
    demo = build_interface(resources)
    try:
        audio_components = [c for c in _components(demo) if isinstance(c, gr.Audio)]
        assert len(audio_components) == 1
        assert audio_components[0].autoplay is False
    finally:
        demo.close()


def test_audio_component_has_elem_id_the_sequencing_shim_can_target():
    from clarif_eye.ui import AUDIO_ELEM_ID

    resources = _resources(FakeGraph())
    demo = build_interface(resources)
    try:
        audio_components = [c for c in _components(demo) if getattr(c, "elem_id", None) == AUDIO_ELEM_ID]
        assert len(audio_components) == 1
        assert isinstance(audio_components[0], gr.Audio)
    finally:
        demo.close()


def test_sequencing_shim_defers_audio_playback_after_a_delay():
    # STATIC/SOURCE-LEVEL CHECK ONLY (same discipline as the rest of this
    # file): asserts the deferred-play mechanism is present in the injected
    # markup, not that two voices actually stop colliding in a real
    # screen reader - only a human screen-reader pass by the owner can
    # confirm that (see docs/ACCESSIBILITY.md's Known defects entry for
    # #47, which is updated to say "awaiting confirmation", never
    # "confirmed fixed").
    from clarif_eye.ui import AUDIO_ELEM_ID, AUDIO_PLAY_DELAY_MS

    assert AUDIO_ELEM_ID in ARIA_LIVE_HEAD
    assert "setTimeout" in ARIA_LIVE_HEAD
    assert str(AUDIO_PLAY_DELAY_MS) in ARIA_LIVE_HEAD
    assert ".play()" in ARIA_LIVE_HEAD
    # Structural check (audio.src truthiness), never a text/string match
    # against the status wording - same discipline status_for_result uses.
    assert "audioEl.src" in ARIA_LIVE_HEAD
    # Must not leave the user stuck in silence if the browser blocks
    # programmatic playback: the rejected play() promise is swallowed, not
    # left to throw, and the <audio> control itself stays present/visible
    # (autoplay=False never hides it) so it remains reachable to play
    # manually.
    assert ".catch(" in ARIA_LIVE_HEAD
