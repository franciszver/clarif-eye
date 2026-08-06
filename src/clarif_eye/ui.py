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

# --- "How this works" section (issue #49 / P4.3) ---------------------------
#
# Owner request: a section near the bottom explaining the pipeline, the data
# flow, and the LangGraph implementation, since this is a demo application.
# Content below is checked against the source it describes, not against the
# issue's own wording or the (older, no longer accurate) architecture doc:
#   - graph.py: build_graph() registers exactly 5 nodes (vision, fast_synth,
#     research, analysis, tts). "router" is NOT a 6th node - dynamic_router
#     is the function evaluated by a conditional edge out of "vision"; it is
#     plain Python (state["complexity_flag"] in/out, no client, no network -
#     see router.py's module docstring "computed locally with no model
#     call, per the architecture doc's requirement that routing be pure
#     Python").
#   - state.py: ClarifEyeState is a 7-key TypedDict.
#   - registry.py / config/models.toml: the "eyes" and "brain" roles each
#     hold an ORDERED ladder of free-only (":free", policy D10) model IDs,
#     tried in turn on failure.
#   - pyproject.toml's [project].dependencies lists langgraph, not
#     langchain - this app does not use langchain, so the text below never
#     claims it does.
#   - tts.py: DEFAULT_PROVIDER_CHAIN is (EdgeTtsProvider, GttsProvider); if
#     every provider fails, audio_file_path == "" and the UI falls back to
#     text (see this module's "THE THREE OUTCOMES" docstring above).
#   - analysis.py: on the deep-analysis path only, _numbers_verified checks
#     every number-like token in the drafted script against the
#     photographed text (+ scene description + any web lookup) before it is
#     spoken; a token that doesn't trace back degrades to a safe
#     "could not be verified" message instead of risking a wrong number.
#     fast_synth.py has no equivalent check - the text below says "on the
#     deep-analysis path", not "always", so it stays true to that asymmetry.
#   - graph.py: DEFAULT_PIPELINE_BUDGET_SECONDS = 60.0, a total-pipeline
#     deadline after which nodes degrade rather than block further (tts is
#     deliberately exempt - see graph.py - so a blown deadline still ends in
#     speech, not silence).
#
# ACCESSIBILITY (issue #49, learning from #48's mistake): real Markdown
# heading syntax ("## "/"### "), which gr.Markdown renders as genuine
# <h2>/<h3> elements a screen reader can navigate by heading - not
# visually-styled prose. The flow is a plain numbered list, never a
# diagram/image: a diagram that announces as a bare "graphic" (#48) is
# worse than no diagram, and a text list carries the same information with
# no such risk, so no image is used here at all (see
# test_how_it_works_introduces_no_unlabelled_image). Always visible, no
# collapsible toggle: simpler, and it avoids needing to get
# aria-expanded/keyboard-toggle wiring right for a chunk of content that
# costs a screen-reader/keyboard user nothing extra to skip past by
# navigating to the next heading. Placed after the result textbox in
# build_interface() below, so it never delays someone using the tool and
# never sits between the live region and the result it announces.
HOW_IT_WORKS_ELEM_ID = "how-it-works"
HOW_IT_WORKS_MARKDOWN = """## How this works

Clarif-Eye is a demo application. This section explains, honestly, what the
code actually does with your photo.

### The flow, step by step

1. You take or upload a photo.
2. A vision-language model reads any text in the photo and separately
   describes the scene (what it is, its layout).
3. A router decides, from that text alone, whether a quick description is
   enough or the document is dense enough to need a closer look (an
   itemised bill, a prescription label, a form with numbers on it). This
   decision is plain Python - no model or network call - scoring things
   like digit density, currency amounts, and document keywords.
4. If a quick description is enough, the photographed text and scene
   description are turned directly into the spoken script.
5. If a closer look is needed, the app first does a web search related to
   what was photographed, then a stronger text-reasoning model writes the
   script from the photographed text, the scene description, and whatever
   the search turned up.
6. On that closer-look path, before anything is spoken, every number in the
   drafted script is checked against the photographed text. If a number
   doesn't trace back to what the camera actually saw, the app reports that
   the result could not be verified rather than risk reading a wrong amount
   or date aloud.
7. The final script is converted to speech.

### Inside the LangGraph pipeline

This pipeline is built with [LangGraph](https://github.com/langchain-ai/langgraph)'s
`StateGraph` (this app depends on `langgraph`, not `langchain` - there is no
LangChain in this codebase). The graph state is a 7-key `TypedDict`:
`image_data`, `ocr_output`, `scene_context`, `complexity_flag`,
`scraper_data`, `final_output`, `audio_file_path`.

Five nodes are registered: `vision`, `fast_synth`, `research`, `analysis`,
and `tts`. Routing between them is a conditional edge out of `vision`,
evaluated against `complexity_flag`: `False` goes to `fast_synth` then
straight to `tts`; `True` goes to `research`, then `analysis`, then `tts`.
That routing decision is evaluated locally in Python, with no model call -
it is a deliberate design point, not an implementation shortcut: the router
only ever needs to read text density and keywords, so it would be wasteful
and slower to spend a model call deciding whether to spend a bigger one.

### The two model roles

Every text/vision model call goes through one of two roles, each an ordered
ladder of free models tried in turn if an earlier one fails or times out:

- **eyes** (reads the photo): `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`,
  then `google/gemma-4-26b-a4b-it:free`.
- **brain** (writes the closer-look description): `nvidia/nemotron-3-ultra-550b-a55b:free`,
  then `nvidia/nemotron-3-super-120b-a12b:free`.

### Honest operational notes

- Every model in both ladders is a free-tier model. Free models can be
  slower or less available than paid ones; the ladder exists so one model
  being down or rate-limited doesn't take the app down with it.
- Measured runs on the closer-look (deep-analysis) path have taken roughly
  21 to 31 seconds end to end; the app tells you up front to expect up to
  about 30 seconds, especially for photos with dense text.
- The whole pipeline has a 60-second total budget. If it's about to be
  exceeded, the app degrades to a simpler, faster answer built from
  whatever was already read, rather than failing outright - except turning
  the script into speech, which always still happens, so a blown budget
  never means silence.
- Speech synthesis tries two independent providers in order (Microsoft
  Edge's TTS service, then Google Translate's TTS service). If both fail,
  the description is still shown, and read by your own screen reader, as
  text.
- On the closer-look (deep-analysis) path, numbers spoken aloud are checked
  against the photographed text before being read, as described above; this
  check does not currently run on the quick-description path.
"""

# How long the deferred-play shim (see ARIA_LIVE_HEAD below) waits after
# the audio element gets a src before it calls .play() - long enough for
# a screen reader to finish speaking the (now short) STATUS_SUCCESS_AUDIO
# announcement before the audio starts (issue #47 / P5.3).
AUDIO_PLAY_DELAY_MS = 1800

# How long the SAME shim delays a USER-INITIATED play (issue #52 / P5.5) -
# a different collision from AUDIO_PLAY_DELAY_MS above:
#   - AUDIO_PLAY_DELAY_MS (#47) is this APP's OWN status announcement vs
#     AUTOMATIC playback. The app controls both sides, so it can afford a
#     longer, deliberately-chosen gap (1800ms) tuned against the wording of
#     STATUS_SUCCESS_AUDIO.
#   - USER_PLAY_DELAY_MS (#52) is the SCREEN READER'S OWN announcement of
#     the Play control being activated (its name, and/or the state change
#     to "Pause") vs playback the user just started by pressing it. This
#     app neither controls nor can suppress that announcement - and per
#     the issue, must NOT try to detect whether a screen reader is even
#     running (no reliable heuristic exists, and a wrong guess is worse
#     than a small fixed delay for everyone). A short, fixed delay is
#     tolerable for every user: sighted users experience it as an
#     imperceptibly late button response (well under the ~1s where a UI
#     starts to feel unresponsive), while it reliably closes the window
#     the collision is reported in. Chosen mid-range within the ~0.8-1.2s
#     the issue calls for.
USER_PLAY_DELAY_MS = 1000

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
# deliberately, after AUDIO_PLAY_DELAY_MS.
#
# REGRESSION, then FIX (found by the owner in a REAL BROWSER, not by this
# test suite - see the module-level tests' "Regression: audio never played
# at all" section in tests/test_accessibility.py): the #47 implementation
# originally scheduled playback by checking whether the <audio> element
# "has a src at all" from inside apply(). That check is never satisfied in
# practice: Gradio's Svelte player assigns `audio.src` as a JS PROPERTY,
# which produces no DOM mutation, so the childList/subtree
# MutationObserver has no reliable reason to re-run apply() at the moment
# a src actually appears. Audio never played, at all, despite the full
# suite passing. The fix (see the `loadeddata` listener inside apply()
# below) instead attaches one real media-load event listener to the audio
# element the first time it's seen, and schedules playback from THAT
# event firing - not from apply() happening to observe src. The delayed
# .play() call's rejected promise is still swallowed (.catch), and
# autoplay is still off (not merely short), so the <audio> control was
# never hidden or auto-triggered in the first place and remains
# visible/reachable for the user to press manually either way. The
# `deferredPlaySrc` guard, now checked inside the event listener's
# callback rather than inside apply() itself, still compares against the
# LAST src this shim scheduled, not just "have we ever scheduled
# anything", so a new audio src on a later submission is still picked up
# even though Gradio reuses the same DOM node.
#
# HONESTY: whether audio actually plays - and whether the two voices
# actually stop colliding - for a real screen-reader user can only be
# confirmed in a real browser by a human; pytest never renders a DOM or
# loads real media. See docs/ACCESSIBILITY.md's Known defects entry for
# #47, which is not marked "confirmed fixed" until the owner re-tests.
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
# USER-GESTURE PLAY COLLISION FIX (issue #52 / P5.5, reported from real
# screen-reader use by the owner, Narrator on Windows): pressing the audio
# widget's own Play control (to replay, or because the browser blocked the
# automatic attempt above) makes the screen reader announce the control's
# activation/state change at the exact same instant this shim used to call
# .play() - so the opening seconds of the description, often the most
# important part, were lost under that announcement. Distinct from #47:
# #47 was this app's OWN status text colliding with AUTOMATIC playback;
# this is the SCREEN READER'S OWN announcement (which this app cannot
# detect or suppress - see USER_PLAY_DELAY_MS's comment on why detecting a
# screen reader isn't attempted) colliding with a USER GESTURE.
#
# FIX: audioEl.play is wrapped ONCE (guarded by a11yPlayDelayWrapped, same
# guard discipline as the rest of apply()) so that ANY call to it - in
# particular the one Gradio's own Play button makes internally when
# clicked - goes through a setTimeout of USER_PLAY_DELAY_MS before the
# real (native) play() actually runs. This is NOT "start playback then
# pause/resume it" (a stutter, explicitly ruled out): the real play() call
# is never made at all until the delay elapses, so nothing is heard early.
# audioEl.pause is left completely untouched, so pausing stays immediate.
#
# NO DOUBLE-DELAY: the deferred-AUTOPLAY block below (issue #47) must not
# route through this same wrapper, or its own automatic .play() call would
# pick up a SECOND, redundant delay on top of AUDIO_PLAY_DELAY_MS (~2.8s of
# dead air instead of the intended ~1.8s). It instead calls the captured
# native play function directly. The wrapper separately tracks, via the
# a11yAutoplayPending flag, whether that automatic attempt is still
# in-flight for the current src; if a user gesture arrives during that
# window, the wrapper skips adding ITS OWN delay on top (the pending
# autoplay timer already guarantees a gap before sound), rather than
# stacking a second wait after the first.
#
# HONESTY: same as the rest of this shim, this is machine-verified only -
# see docs/ACCESSIBILITY.md's Known defects entry for #52, not marked
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
    // its own - start it deliberately, after a delay, once a real media
    // load event says a source is actually there (see the
    // a11yMediaListenerAttached block below for why this is no longer a
    // check of "does audioEl.src look truthy right now").
    const audioEl = document.querySelector("#{AUDIO_ELEM_ID} audio");
    // User-gesture play delay (issue #52 / P5.5) - see the module comment
    // above this shim for the full reasoning. Wrap .play() ONCE per
    // element (guard, same pattern as the other checks in this function)
    // so a click on this control's own Play button - which internally
    // calls this same audioEl.play() - is delayed just like every other
    // caller of it, reusing the SAME observer/apply() pair as the rest of
    // this function rather than a second one.
    if (audioEl && !audioEl.dataset.a11yPlayDelayWrapped) {{
      audioEl.dataset.a11yPlayDelayWrapped = "1";
      const nativePlay = audioEl.play.bind(audioEl);
      audioEl._a11yNativePlay = nativePlay;
      audioEl.play = function () {{
        // No double-delay: if the automatic-autoplay attempt for this src
        // is still pending, that timer already guarantees a gap before
        // sound - don't stack a second USER_PLAY_DELAY_MS on top of it.
        if (audioEl.dataset.a11yAutoplayPending === "1") {{
          return nativePlay();
        }}
        return new Promise((resolve, reject) => {{
          setTimeout(() => {{
            nativePlay().then(resolve, reject);
          }}, {USER_PLAY_DELAY_MS});
        }});
      }};
    }}
    if (audioEl && !audioEl.dataset.a11yMediaListenerAttached) {{
      audioEl.dataset.a11yMediaListenerAttached = "1";
      // REGRESSION FIX (found by the owner in a real browser, NOT by this
      // test suite - the suite passed 484 tests while audio was completely
      // broken on main): the block used to be gated on `audioEl.src` being
      // truthy at the moment apply() ran. Gradio's Svelte player assigns
      // `audio.src` as a JS PROPERTY, and a property assignment produces NO
      // DOM mutation - so the childList/subtree observer above (the one
      // constructed at the bottom of this IIFE) never had a reliable
      // reason to re-run apply() at the moment a src
      // actually became available, and the gate was, in practice, never
      // satisfied. Audio never played at all.
      //
      // FIX: attach ONE real media-load event listener to the element
      // instead of relying on apply() happening to observe src. Chose
      // "loadeddata" (fires once the browser actually has decoded data for
      // the current playback position - a genuine "a source is here and
      // usable" signal) over "canplay" (fires later, requires enough
      // buffered to estimate uninterrupted playback - an unnecessary bar
      // for a short spoken description) and over "durationchange" (fires
      // on metadata alone, before there is necessarily any audio DATA to
      // play - could schedule playback before the browser is actually
      // ready to render sound).
      //
      // PRELOAD: "loadeddata" only fires once the browser actually FETCHES
      // data. Left at the browser/Gradio default, `preload` can be "none"
      // or "metadata", either of which can leave the element never
      // fetching audio data until playback is attempted - which would make
      // this whole fix a silent no-op. Force eager loading so the event is
      // guaranteed to fire once Gradio assigns a real src.
      audioEl.preload = "auto";
      // Guarded by a11yMediaListenerAttached above (same guard discipline
      // as every other check in apply()) so a DOM node Gradio reuses across
      // submissions never accumulates a second listener - one loadeddata
      // event would otherwise schedule N overlapping play attempts.
      //
      // RE-ARMS PER RESULT: this listener is attached ONCE, but it fires
      // on EVERY load (Gradio reuses the same <audio> node across
      // submissions, and "loadeddata" fires again each time a new src is
      // loaded into it) - so a second, third, ... submission's audio is
      // scheduled too, not just the first. The per-src dedupe check below
      // runs INSIDE the callback (not as a gate on attaching the listener)
      // precisely so it re-evaluates against the CURRENT src on every load
      // instead of only ever firing once.
      audioEl.addEventListener("loadeddata", () => {{
        if (audioEl.src && audioEl.dataset.deferredPlaySrc !== audioEl.src) {{
          audioEl.dataset.deferredPlaySrc = audioEl.src;
          audioEl.dataset.a11yAutoplayPending = "1";
          setTimeout(() => {{
            audioEl.dataset.a11yAutoplayPending = "0";
            // Calls the captured NATIVE play directly (not the wrapped
            // audioEl.play above) so this automatic attempt is never
            // double-delayed by the user-gesture wrapper.
            //
            // If the browser blocks programmatic playback (autoplay
            // policies vary), swallow the rejection: the control stays
            // visible and reachable so the user can press play manually
            // instead of the page throwing a silent, uncaught error, and
            // this code never claims audio is playing when it isn't (see
            // STATUS_SUCCESS_AUDIO's wording, which makes the same
            // promise).
            (audioEl._a11yNativePlay || audioEl.play)().catch(() => {{}});
          }}, {AUDIO_PLAY_DELAY_MS});
        }}
      }});
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

        # issue #49 / P4.3: placed AFTER the result area (never before it),
        # so it never delays someone who came to use the tool and never
        # sits between the live region and the result it announces. See
        # HOW_IT_WORKS_MARKDOWN's module-level docstring for content
        # sourcing and the accessibility reasoning behind a text-only
        # section with no image/diagram.
        gr.Markdown(HOW_IT_WORKS_MARKDOWN, elem_id=HOW_IT_WORKS_ELEM_ID)

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
