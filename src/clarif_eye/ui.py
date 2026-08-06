"""Gradio UI logic for Clarif-Eye (issue #13 / P4.1): wires the graph to a
human, one photo at a time.

Kept separate from app.py (the thin Spaces launcher) so this module is
TESTABLE without launching a server: tests/test_ui.py calls handle_submit
directly with fakes and never starts Gradio or touches the network.

SHARED CLIENT / PROVIDER CHAIN
-------------------------------
Nodes construct their own client when none is injected (see graph.py's
module docstrings), which is fine for tests but wasteful for a live app: a
fast-path request would otherwise open its own httpx connection pool per
node. build_resources() constructs ONE OpenRouterClient, ONE TTS provider
chain, and (best-effort) ONE research searcher/client at process startup;
handle_submit injects all of them via config["configurable"] on every
request, the same seam graph.py already documents for tests.

FAILURE BEHAVIOR - THIS MODULE MUST NEVER RAISE INTO GRADIO
--------------------------------------------------------------
A traceback in the UI is useless to a blind user. Every failure mode (no
image, an unreadable/corrupt image, a missing API key at startup, an
unexpected exception anywhere in the graph) degrades to a spoken-ready
message in the returned text, never an exception - same discipline every
node in this pipeline already follows.

THE THREE OUTCOMES, TOLD APART STRUCTURALLY
----------------------------------------------
  a. audio_file_path is truthy -> play it, show final_output as-is.
  b. audio_file_path == "" AND tts.is_chain_exhausted() is True -> no
     audio was produced despite a real attempt; announce that plainly and
     show final_output as text.
  c. The pipeline degraded upstream (vision/synth/analysis already wrote a
     human-readable message into final_output) -> that message IS the
     script. It needs no special-casing here: it flows through outcome (a)
     or (b) exactly like a non-degraded script would, so no second layer
     of "something went wrong" wrapping is ever added on top of it.
Never string-matched - (a)/(b) are told apart via audio_file_path
truthiness and the structural tts.is_chain_exhausted() predicate, the same
discipline vision.is_degraded_scene already established.
"""

import base64
import io
import time
from dataclasses import dataclass

import gradio as gr

from clarif_eye.client import OpenRouterClient, OpenRouterError
from clarif_eye.graph import DEFAULT_PIPELINE_BUDGET_SECONDS, build_graph
from clarif_eye.state import make_initial_state
from clarif_eye.tts import DEFAULT_PROVIDER_CHAIN, is_chain_exhausted

# Spoken-ready messages, as named constants rather than inline literals -
# same reasoning as vision.py's DEGRADED_* constants: a caller (and tests)
# can rely on these without guessing at wording, and rewording one later
# can't silently break a test that was substring-matching prose instead.
NO_IMAGE_MESSAGE = (
    "No photo was provided. Please take or upload a photo to continue."
)
CONFIG_ERROR_MESSAGE = (
    "Clarif-Eye is not fully configured yet: the service is missing an "
    "API key. Please tell whoever set this up."
)
UNREADABLE_IMAGE_MESSAGE = (
    "The photo could not be read. Please try again with a different photo."
)
UNEXPECTED_ERROR_MESSAGE = (
    "Something went wrong while processing your photo. Please try again."
)
AUDIO_UNAVAILABLE_NOTE = (
    "Audio isn't available right now, so here is the description as text."
)

# --- Accessibility (issue #15 / P5.1) ---------------------------------------
#
# THE PROBLEM: every user of this product is visually impaired (see this
# module's top-level docstring / app.py). They cannot see Gradio's spinner
# during the ~15-30s the pipeline typically takes (measured in issue #17;
# observed max 60.3s against a 60s pipeline deadline). A screen reader only
# hears something if it is told to, via an ARIA live region - hence
# STATUS_* below, a short spoken-style status distinct from the description
# text itself.
#
# Owner decision D13: this phase is verified by AUTOMATED checks only (see
# tests/test_accessibility.py) - no manual screen-reader pass happens, so
# nothing here or in that test file may claim "screen-reader tested"; the
# accurate claim is "machine-verified".
STATUS_ELEM_ID = "status-live-region"
STATUS_ELEM_CLASSES = ["live-status"]
RESULT_ELEM_ID = "description-output"
AUDIO_ELEM_ID = "audio-output"
IMAGE_INPUT_ELEM_ID = "photo-input"

# Accessible name given to the user's own uploaded/captured photo preview
# (issue #48 / P5.4 - see ARIA_LIVE_HEAD's image-labelling comment below).
UPLOADED_PHOTO_ALT = "The photo you submitted"

# How long the deferred-play shim (see ARIA_LIVE_HEAD below) waits after
# the audio element gets a src before it calls .play() - long enough for
# a screen reader to finish speaking the (now short) STATUS_SUCCESS_AUDIO
# announcement before the audio starts (issue #47 / P5.3).
AUDIO_PLAY_DELAY_MS = 1800

STATUS_IDLE = 'Ready. Choose or take a photo, then activate "Describe this photo".'
STATUS_WORKING = (
    "Photo received. Describing it now; this can take up to about 30 seconds."
)
# SHORT ON PURPOSE (issue #47 / P5.3): when audio will play, the audio
# itself is the completion signal - a long spoken status is redundant with
# it and, worse, collides with it (two voices talking over each other; see
# docs/ACCESSIBILITY.md's Known defects entry for #47). Kept to "over in
# about a second" so even a slight overlap with the deferred audio start
# (see AUDIO_PLAY_DELAY_MS / ARIA_LIVE_HEAD) is easy to hear past. Does NOT
# assert "Audio is playing" as a fact, since the browser may block
# programmatic playback (see ARIA_LIVE_HEAD) and that would be a lie.
STATUS_SUCCESS_AUDIO = "Description ready."
# FULL ON PURPOSE: no audio plays in this outcome, so there is nothing for
# this announcement to collide with, and it is the user's ONLY signal - it
# must say the text is a fallback, not just "ready".
STATUS_SUCCESS_TEXT_ONLY = f"Description ready as text. {AUDIO_UNAVAILABLE_NOTE}"
STATUS_DEGRADED = "Finished, but with a limited result. See the text below for details."

# Gradio has no native aria-live prop (as of 6.22.0), so a minimal,
# commented JS shim marks the status control's wrapper as a polite live
# region by hand. Passed to demo.launch(head=...) - NOT the Blocks
# constructor (deprecated there since Gradio 6.0) - so it only takes effect
# when app.py actually launches a real server, never during
# build_interface() itself (see that function's docstring: it never
# launches).
#
# BUG FOUND VIA REAL-BROWSER CHECK (Chrome DevTools): the original shim ran
# on window "load", which fires BEFORE Gradio's client-side SPA renders its
# components. getElementById returned null and the shim silently did
# nothing - #status-live-region existed in the final DOM but carried none
# of aria-live/aria-atomic/role, so a screen reader announced no progress
# at all during the ~15-30s wait.
#
# FIX: apply immediately in case the element is already there (harmless
# no-op otherwise), and also watch the DOM with a MutationObserver so the
# attributes get applied as soon as Gradio actually renders the element -
# whenever that happens to be. The observer is never disconnected because
# Gradio re-renders this output component's DOM node on every run (it's a
# graph output that updates every submit); apply() re-tags it each time
# that happens. The `aria-live` check inside apply() is a cheap guard so a
# already-tagged element is skipped on the (many) unrelated mutations that
# fire elsewhere on the page, instead of re-setting three attributes on
# every single DOM change.
#
# AUDIO SEQUENCING FIX (issue #47 / P5.3, reported from real screen-reader
# use by the owner): the synthesized audio used to start via
# gr.Audio(autoplay=True) at the exact moment the completion status was
# still being announced, so a screen reader and the spoken audio talked
# over each other and neither was intelligible. Two changes, together:
# (1) STATUS_SUCCESS_AUDIO above is now short, so even a slight overlap is
# brief; (2) autoplay is turned OFF on the component (see build_interface)
# and this SAME apply()/MutationObserver pair instead starts playback
# deliberately, after AUDIO_PLAY_DELAY_MS, once it sees the audio element
# has a src - option (a) from the three sequencing options considered,
# chosen because it is the only one that also solves "what if playback
# can't be started programmatically": the delayed .play() call's rejected
# promise is swallowed (.catch), and because autoplay is off (not merely
# short), the <audio> control was never hidden or auto-triggered in the
# first place, so it remains visible/reachable for the user to press
# manually either way. The check is STRUCTURAL - whether the <audio>
# element has a src at all - never a match against the status text, same
# discipline status_for_result already uses. The `deferredPlaySrc` guard
# compares against the LAST src this shim scheduled, not just "have we
# ever scheduled anything", so a new audio src on a later run is still
# picked up even though Gradio reuses the same DOM node.
#
# HONESTY: whether the two voices actually stop colliding for a real
# screen-reader user can only be confirmed by a human screen-reader pass -
# see docs/ACCESSIBILITY.md's Known defects entry for #47, which is not
# marked "confirmed fixed" until the owner re-tests.
#
# IMAGE LABELLING FIX (issue #48 / P5.4, reported from real screen-reader
# use by the owner, Narrator on Windows): every image on the page - Gradio's
# own chrome (footer "Built with Gradio" logo, the "Use via API" button's
# logo, button glyphs for upload/camera/fullscreen/remove/download/share/
# volume/playback controls) AND the user's own uploaded photo preview - was
# announcing as a bare, unlabelled "graphic": noise for the chrome, and
# useless for the photo, which is the one image that actually matters to a
# blind user. Every image must end up either MEANINGFUL (a real accessible
# name) or DECORATIVE (`alt=""` + `aria-hidden="true"`, and removed from the
# tab order if it was somehow focusable) - never silently unlabelled.
#
# IDENTIFYING THE UPLOADED PHOTO, STRUCTURALLY: the ONLY image treated as
# meaningful is an <img> element that lives inside the photo-input
# component's own container (#{IMAGE_INPUT_ELEM_ID}, set on the gr.Image in
# build_interface) - that is where Gradio renders the uploaded/captured
# photo preview once one exists. The upload/camera icon glyphs in that SAME
# container are <svg> elements on buttons that already carry their own
# accessible name (the button's aria-label, per the real accessibility-tree
# dump this issue was filed from), never <img> tags, so this can't
# mislabel them. This is a structural check (tag + container), never a
# string/URL match against image src, same discipline the rest of this
# module uses elsewhere.
#
# Everything else - any other img/svg/[role="img"] anywhere on the page -
# is Gradio chrome with no information for the user and is marked
# decorative so a screen reader skips it entirely.
#
# HONESTY: same as the rest of this shim, this is machine-verified only -
# see docs/ACCESSIBILITY.md's Known defects entry for #48, not marked
# "confirmed fixed" until a human screen-reader pass confirms it.
#
# KEYBOARD-REACHABILITY FIX (found via real-browser Chrome DevTools check,
# same follow-up to issue #15): Gradio 6.22 renders an output-only Textbox
# with a `disabled` <textarea>. A disabled form control cannot receive
# focus and is skipped by keyboard Tab navigation entirely - so
# #description-output, the accessible fallback shown whenever audio isn't
# available, was NOT in the tab order at all, and FOCUS_RESULT_JS's
# el.focus() call below silently did nothing. The SAME apply()/
# MutationObserver pair (reused, not duplicated, so it survives Gradio
# re-rendering this output on every run exactly like the aria-live tagging
# must) also strips `disabled` off that textarea and swaps in `readOnly` +
# `tabindex="0"` instead: readonly text inputs remain non-editable but,
# unlike disabled ones, ARE focusable and part of the tab order.
ARIA_LIVE_HEAD = f"""
<script>
// Aria-live shim for the status control (issue #15 / P5.1).
// Gradio doesn't expose an aria-live prop, so mark the element by hand:
// "polite" means a screen reader announces the change without
// interrupting whatever the user is doing, and never steals focus.
(function () {{
  function apply() {{
    const el = document.getElementById("{STATUS_ELEM_ID}");
    // Guard: skip already-tagged elements so we're not re-setting
    // attributes on every unrelated mutation elsewhere on the page.
    if (el && el.getAttribute("aria-live") !== "polite") {{
      el.setAttribute("aria-live", "polite");
      el.setAttribute("aria-atomic", "true");
      el.setAttribute("role", "status");
    }}
    // Keyboard-reachability fix: swap disabled -> readOnly + tabindex on
    // the description output so it stays non-editable but re-enters the
    // tab order. Guard: only touch it while still disabled, so this isn't
    // re-run on every unrelated mutation either.
    const resultEl = document.querySelector("#{RESULT_ELEM_ID} textarea");
    if (resultEl && resultEl.disabled) {{
      resultEl.disabled = false;
      resultEl.readOnly = true;
      resultEl.setAttribute("tabindex", "0");
    }}
    // Audio sequencing fix (issue #47 / P5.3): the <audio> element has
    // autoplay=False (see build_interface), so playback never starts on
    // its own - start it deliberately, after a delay, once this run
    // actually produced audio (structural: a src is present at all).
    const audioEl = document.querySelector("#{AUDIO_ELEM_ID} audio");
    if (audioEl && audioEl.src && audioEl.dataset.deferredPlaySrc !== audioEl.src) {{
      audioEl.dataset.deferredPlaySrc = audioEl.src;
      setTimeout(() => {{
        // If the browser blocks programmatic playback (autoplay policies
        // vary), swallow the rejection: the control stays visible and
        // reachable so the user can press play manually instead of the
        // page throwing a silent, uncaught error.
        audioEl.play().catch(() => {{}});
      }}, {AUDIO_PLAY_DELAY_MS});
    }}
    // Image labelling fix (issue #48 / P5.4) - see the comment above this
    // function for the full reasoning. Guard: `a11yImgDone` marks an image
    // as already classified so it isn't re-processed on every unrelated
    // mutation, same pattern as the guards above.
    document.querySelectorAll('img, svg, [role="img"]').forEach((img) => {{
      if (img.dataset.a11yImgDone) {{
        return;
      }}
      img.dataset.a11yImgDone = "1";
      const isUploadedPhoto =
        img.tagName === "IMG" && img.closest("#{IMAGE_INPUT_ELEM_ID}");
      if (isUploadedPhoto) {{
        // MEANINGFUL: the user's own submitted photo, not chrome.
        img.setAttribute("alt", "{UPLOADED_PHOTO_ALT}");
        img.setAttribute("aria-label", "{UPLOADED_PHOTO_ALT}");
        img.removeAttribute("aria-hidden");
      }} else {{
        // DECORATIVE: Gradio chrome (footer logo, API logo, button
        // glyphs) - carries no information, so a screen reader should
        // skip it entirely rather than announce a bare "graphic".
        //
        // CORRECTION from real screen-reader testing: the reported
        // behaviour was REPEATED "graphic, graphic, graphic" while
        // navigating, so these nodes must become genuinely ABSENT from
        // the accessibility tree, not merely unnamed. `alt=""` only
        // applies to <img> - it is MEANINGLESS on inline <svg>, and
        // Gradio's chrome is largely inline SVG - so alt="" alone would
        // leave the svg glyphs announcing exactly as before. Handle each
        // element type explicitly:
        const tag = img.tagName.toLowerCase();
        if (tag === "img") {{
          img.setAttribute("alt", "");
        }}
        if (tag === "svg") {{
          // aria-hidden alone can still leave inline SVG reachable/
          // focusable in some engines, so also explicitly mark it
          // non-focusable - this is the load-bearing pair for SVG, not
          // alt (which does nothing here).
          img.setAttribute("focusable", "false");
        }}
        // Covers [role="img"] on any element too: aria-hidden is what
        // actually removes it from the accessibility tree - simply
        // dropping the role would not be sufficient.
        img.setAttribute("aria-hidden", "true");
        // Only ever drive tabindex to -1 (out of the tab order); never
        // introduce a positive one.
        if (img.hasAttribute("tabindex")) {{
          img.setAttribute("tabindex", "-1");
        }}
      }}
    }});
  }}
  apply(); // covers the element already being present
  // Covers the element appearing later (Gradio's SPA render) or being
  // replaced later (Gradio re-rendering this output on each run).
  new MutationObserver(apply).observe(document.documentElement, {{
    childList: true,
    subtree: true,
  }});
}})();
</script>
"""

# Client-side focus management (issue #15 / P5.1 scope item 4): once the
# result is ready, move focus to the description text so a screen-reader
# user is told the answer without hunting for it, instead of leaving focus
# wherever it was (typically the submit button). Wired via .then(js=...)
# on the submit click AFTER the handler's fn resolves, so it only ever
# runs once a result exists - never while the user is still interacting
# with the image input.
FOCUS_RESULT_JS = f"""
() => {{
  // Defensive: if the element isn't there, or isn't focusable for any
  // reason (e.g. the keyboard-reachability shim above hasn't run yet, or
  // Gradio changes how it renders this output), this must not throw -
  // an uncaught client-side error would be silent to the user but is
  // still a correctness bug.
  try {{
    const el = document.querySelector('#{RESULT_ELEM_ID} textarea');
    if (el && typeof el.focus === 'function') {{ el.focus(); }}
  }} catch (e) {{}}
}}
"""


def status_for_result(audio_path, chain_exhausted):
    """Derive the live-region status text for a finished run.

    STRUCTURAL, same three-way split handle_submit already uses (see this
    module's top-level "THE THREE OUTCOMES" docstring): audio_path
    truthiness and the tts.is_chain_exhausted() bool, never a string match
    against the description text. `chain_exhausted` is passed in rather
    than read here so callers control exactly when it's sampled (it must
    be read right after the graph call that produced audio_path, before
    any other run_tts() call could overwrite the module-level result).
    """
    if audio_path:
        return STATUS_SUCCESS_AUDIO
    if chain_exhausted:
        return STATUS_SUCCESS_TEXT_ONLY
    return STATUS_DEGRADED


@dataclass
class AppResources:
    """Everything build_resources() constructs once at startup and
    handle_submit injects on every request. Plain dataclass, not a
    framework - one instance per process (CLAUDE.md Simplicity First)."""

    graph: object
    client: object
    client_error: str | None
    tts_providers: list
    searcher: object
    research_client: object


def build_resources():
    """Construct every injectable ONCE for the life of the process.

    Never raises: a missing OPENROUTER_API_KEY (the likely state of a
    fresh Hugging Face Space with no secret set yet) must not crash the
    app at import/startup time - it degrades to client=None plus a spoken
    message, checked by handle_submit before the graph is ever invoked.
    The research searcher/client are best-effort shared instances too (see
    module docstring); if either fails to construct, they're left None and
    research_node falls back to its own lazy per-call defaults.
    """
    try:
        client = OpenRouterClient()
        client_error = None
    except OpenRouterError:
        client = None
        client_error = CONFIG_ERROR_MESSAGE

    tts_providers = [factory() for factory in DEFAULT_PROVIDER_CHAIN]

    try:
        from ddgs import DDGS

        searcher = DDGS()
    except Exception:
        searcher = None

    try:
        import httpx

        research_client = httpx.Client()
    except Exception:
        research_client = None

    return AppResources(
        graph=build_graph(),
        client=client,
        client_error=client_error,
        tts_providers=tts_providers,
        searcher=searcher,
        research_client=research_client,
    )


def _encode_image(image):
    """Encode a PIL Image to base64 JPEG bytes for make_initial_state.

    Raises on anything unreadable (wrong object shape, a corrupt image
    PIL can't re-encode, ...) - handle_submit turns that into
    UNREADABLE_IMAGE_MESSAGE rather than letting it propagate.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def handle_submit(image, resources, pipeline_budget_seconds=DEFAULT_PIPELINE_BUDGET_SECONDS):
    """Run one photo through the graph; return (audio_path_or_None, text).

    NEVER raises (except KeyboardInterrupt/SystemExit) - every failure
    mode returns a spoken-ready message instead, per the module docstring.
    `resources` is an AppResources built once by build_resources() and
    passed through unchanged on every call, so the shared client/provider
    chain/searcher are injected identically on every request.

    `pipeline_budget_seconds` (issue #17 / P6.1) sets the total-pipeline
    deadline (see clarif_eye.graph's module docstring "Total-pipeline
    deadline"): an absolute time.monotonic() timestamp computed fresh for
    THIS request, `time.monotonic() + pipeline_budget_seconds`, and passed
    through config["configurable"]["deadline"] - never a shared/reused
    value, since each request needs its own clock start. Defaults to
    graph.DEFAULT_PIPELINE_BUDGET_SECONDS but is overridable per call
    (e.g. by scripts/benchmark_pipeline.py sweeping it).
    """
    if image is None:
        return None, NO_IMAGE_MESSAGE

    if resources.client is None:
        return None, resources.client_error or CONFIG_ERROR_MESSAGE

    try:
        image_data = _encode_image(image)
    except Exception:
        return None, UNREADABLE_IMAGE_MESSAGE

    try:
        state = make_initial_state(image_data)
        config = {
            "configurable": {
                "client": resources.client,
                "tts_providers": resources.tts_providers,
                "searcher": resources.searcher,
                "research_client": resources.research_client,
                "deadline": time.monotonic() + pipeline_budget_seconds,
            }
        }
        result = resources.graph.invoke(state, config=config)
    except Exception:
        return None, UNEXPECTED_ERROR_MESSAGE

    final_output = (result.get("final_output") or "").strip()
    audio_path = result.get("audio_file_path") or ""

    if audio_path:
        return audio_path, final_output

    if is_chain_exhausted():
        if final_output:
            return None, f"{final_output} {AUDIO_UNAVAILABLE_NOTE}"
        return None, AUDIO_UNAVAILABLE_NOTE

    return None, final_output or UNEXPECTED_ERROR_MESSAGE


def handle_submit_staged(image, resources, pipeline_budget_seconds=DEFAULT_PIPELINE_BUDGET_SECONDS):
    """Generator version of handle_submit that also drives the live-region
    status text (issue #15 / P5.1 scope item 3).

    Yields (status_text, audio_path_or_None, description_text) tuples;
    Gradio streams each yield straight to the UI as it's produced, which is
    what lets the live region announce progress at all.

    STAGING: graph.invoke() is one synchronous, blocking call with no
    intermediate progress hook (see graph.py), so there is no real
    per-node progress to report without a larger restructure of the graph
    itself - and inventing fake percentages was explicitly ruled out.
    Instead this yields twice: once immediately with an honest "received
    and working, up to about 30 seconds" message (submission-received and
    still-working collapsed into one announcement, since nothing
    observable happens between them), then once more when the blocking
    call returns with the final status/audio/text. A screen reader hears
    both, in order, via aria-live="polite" on the status control.
    """
    yield STATUS_WORKING, None, ""
    audio_path, text = handle_submit(image, resources, pipeline_budget_seconds)
    status = status_for_result(audio_path, is_chain_exhausted())
    yield status, audio_path, text


def build_interface(resources):
    """Build the Clarif-Eye gr.Blocks UI, wired to `resources`.

    NEVER launches a server or touches the network - this only constructs
    the Blocks object (see tests/test_accessibility.py, which builds it
    and walks the component tree, and this module's top-level docstring's
    "TESTABLE without launching a server" discipline, the same reasoning
    build_resources()/handle_submit already follow). app.py is the only
    caller that calls .launch() on the result, and only inside its
    `if __name__ == "__main__":` guard.

    ACCESSIBILITY (issue #15 / P5.1): every interactive control below
    carries a real label (its accessible name); the status control is a
    read-only (non-focusable, so it never steals focus) live region wired
    to aria-live via ARIA_LIVE_HEAD (passed to .launch(head=...) by
    app.py, since Gradio 6.0 moved `head` off the Blocks constructor);
    and FOCUS_RESULT_JS moves focus to the description text once (and
    only once) a result is ready, via .then(js=...) chained after the
    submit click's fn resolves.
    """

    def _submit(image):
        yield from handle_submit_staged(image, resources)

    with gr.Blocks(title="Clarif-Eye") as demo:
        gr.Markdown(
            "# Clarif-Eye\n"
            "Clarif-Eye describes a photo aloud for visually impaired users. "
            "Take or upload a photo below. This can take up to about 30 "
            "seconds, especially for photos with dense text."
        )
        image_input = gr.Image(
            label="Photo to describe",
            sources=["upload", "webcam"],
            type="pil",
            # issue #48 / P5.4: lets ARIA_LIVE_HEAD's image-labelling shim
            # find the uploaded-photo preview structurally (it's the <img>
            # inside this container), instead of any icon glyph elsewhere
            # on the page.
            elem_id=IMAGE_INPUT_ELEM_ID,
        )
        submit_button = gr.Button("Describe this photo", variant="primary")
        status_output = gr.Textbox(
            value=STATUS_IDLE,
            label="Status",
            interactive=False,
            elem_id=STATUS_ELEM_ID,
            elem_classes=STATUS_ELEM_CLASSES,
        )
        # autoplay=False (issue #47 / P5.3): ARIA_LIVE_HEAD's shim starts
        # playback deliberately, after a delay, instead of the browser
        # firing it the instant a src is set - see ARIA_LIVE_HEAD's "AUDIO
        # SEQUENCING FIX" comment for why.
        audio_output = gr.Audio(label="Spoken description", autoplay=False, elem_id=AUDIO_ELEM_ID)
        text_output = gr.Textbox(label="Description (text)", lines=6, elem_id=RESULT_ELEM_ID)

        submit_event = submit_button.click(
            fn=_submit,
            inputs=image_input,
            outputs=[status_output, audio_output, text_output],
        )
        # Runs client-side only after the handler above has produced its
        # final yield - see FOCUS_RESULT_JS's docstring for why that
        # timing matters (never steals focus mid-interaction).
        submit_event.then(fn=None, inputs=None, outputs=None, js=FOCUS_RESULT_JS)

    return demo
