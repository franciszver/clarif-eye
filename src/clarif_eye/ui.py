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
import hashlib
import io
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import gradio as gr
from langgraph.checkpoint.memory import InMemorySaver

from clarif_eye.client import LadderExhaustedError, OpenRouterClient, OpenRouterError
from clarif_eye.failure_messages import (
    message_for_ladder_exhausted,
    message_for_terminal_error,
)
from clarif_eye.graph import DEFAULT_PIPELINE_BUDGET_SECONDS, build_graph, next_node_after
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
#   - state.py: ClarifEyeState is an 8-key TypedDict (issue #81 / P9.2 added
#     `messages`, the app's first LangGraph reducer).
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
# visually-styled prose. The flow is a plain numbered list, which stays
# text-only in THIS string (see test_how_it_works_introduces_no_unlabelled_image
# - no img/svg markup is ever added here). Always visible, no collapsible
# toggle: simpler, and it avoids needing to get aria-expanded/keyboard-toggle
# wiring right for a chunk of content that costs a screen-reader/keyboard
# user nothing extra to skip past by navigating to the next heading. Placed
# after the result textbox in build_interface() below, so it never delays
# someone using the tool and never sits between the live region and the
# result it announces.
#
# THE DIAGRAM (issue #56 / P4.4): the owner later asked for a graphic. P4.3
# deliberately shipped text-only because #48 had JUST been fixed and an
# unlabelled diagram would have announced as a bare "graphic" - worse than
# no diagram. Now that a labelling mechanism exists, PIPELINE_DIAGRAM_HTML
# below adds an inline SVG as a SEPARATE gr.HTML component, alongside (never
# instead of) the ordered list above - the list stays the accessible source
# of truth; the diagram is for sighted readers, with its own name and text
# description for anyone who lands on it. See PIPELINE_DIAGRAM_HTML's
# docstring for the exemption mechanism that keeps ARIA_LIVE_HEAD's #48 pass
# from silencing it.
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
LangChain in this codebase). The graph state is an 8-key `TypedDict`:
`image_data`, `ocr_output`, `scene_context`, `complexity_flag`,
`scraper_data`, `final_output`, `audio_file_path`, `messages`.

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

# --- Pipeline diagram (issue #56 / P4.4) ------------------------------------
#
# Adds a sighted-friendly diagram of the pipeline next to the ordered list
# above, WITHOUT replacing it - the list stays the accessible source of
# truth for a screen-reader user; the diagram is additional, not a
# substitute (a listener gets the list; a sighted reader gets both).
#
# CONTENT, checked against graph.py's build_graph() as built, not against
# memory or the older architecture doc: vision runs first; dynamic_router
# is a locally-evaluated conditional edge on complexity_flag (no model or
# network call, plain Python) that sends the run to fast_synth, or to
# research then analysis; every path ends at tts. "Router" names that
# conditional edge for a reader of the diagram - it is not a claim that a
# sixth node named "router" is registered in the graph.
#
# WHY INLINE SVG, PLAIN CURRENTCOLOR: no new dependency, no build step, no
# raster asset to keep in sync with a theme. currentColor on every stroke
# and text fill means the diagram inherits whatever text color the page
# already uses, so it reads correctly in both light and dark themes without
# any separate dark-mode markup.
#
# THE #48 TRAP: ARIA_LIVE_HEAD's image-labelling pass marks every
# img/svg/[role="img"] it finds as aria-hidden="true" UNLESS it is
# structurally exempted - exactly like #photo-input's uploaded-photo <img>
# already is. Without an equivalent exemption here, this new diagram would
# be silenced by our own accessibility code. DIAGRAM_ELEM_ID is the
# structural anchor: ARIA_LIVE_HEAD checks whether an element lives inside
# `#{DIAGRAM_ELEM_ID}` via `.closest()`, the same mechanism (id/container,
# never a string/content match) IMAGE_INPUT_ELEM_ID already uses. See
# ARIA_LIVE_HEAD's own comment for the combined check.
#
# NAME VS. DESCRIPTION: aria-label gives the SVG a short accessible name
# (what it is); aria-describedby points at DIAGRAM_DESC_ELEM_ID, a visible
# paragraph carrying the flow in full sentences, so a screen-reader user who
# lands on the diagram gets the content, not just a name.
#
# KEYBOARD (issue #56 scope item 6): the SVG carries no tabindex attribute
# at all, so it is not a tab stop. It is a non-interactive image - nothing
# happens when it receives focus, and adding tabindex="0" would present it
# as a stopping point that implies interactivity it doesn't have. Its
# content still reaches a screen-reader user in the normal document reading
# order via its name/description, exactly like the uploaded-photo <img>
# neither needs nor gets a tabindex either.
DIAGRAM_ELEM_ID = "how-it-works-diagram"
DIAGRAM_DESC_ELEM_ID = "how-it-works-diagram-desc"

PIPELINE_DIAGRAM_LABEL = "Diagram of the Clarif-Eye pipeline, from photo to spoken description"

PIPELINE_DIAGRAM_DESCRIPTION = (
    "A photo goes to the vision step, which reads any text and describes the "
    "scene. A router then checks how complex the result is, using plain "
    "Python with no model call. If the result is simple, fast synthesis "
    "writes the script right away. If the result is complex, the app runs a "
    "web search first, then a stronger writing model: a research step "
    "followed by an analysis step. Both paths end at text to speech, which "
    "turns the final script into spoken audio."
)

PIPELINE_DIAGRAM_SVG = f"""<svg viewBox="0 0 800 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{PIPELINE_DIAGRAM_LABEL}" aria-describedby="{DIAGRAM_DESC_ELEM_ID}" style="max-width:100%;height:auto;color:inherit;">
<title>{PIPELINE_DIAGRAM_LABEL}</title>
<defs>
<marker id="how-it-works-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="currentColor" />
</marker>
</defs>
<g fill="none" stroke="currentColor" stroke-width="2">
<rect x="10" y="115" width="110" height="50" rx="6" />
<rect x="170" y="115" width="140" height="50" rx="6" />
<rect x="380" y="30" width="150" height="50" rx="6" />
<rect x="380" y="215" width="120" height="50" rx="6" />
<rect x="530" y="215" width="110" height="50" rx="6" />
<rect x="660" y="115" width="120" height="50" rx="6" />
<line x1="120" y1="140" x2="164" y2="140" marker-end="url(#how-it-works-arrow)" />
<line x1="310" y1="122" x2="378" y2="58" marker-end="url(#how-it-works-arrow)" />
<line x1="310" y1="158" x2="378" y2="235" marker-end="url(#how-it-works-arrow)" />
<line x1="500" y1="240" x2="524" y2="240" marker-end="url(#how-it-works-arrow)" />
<line x1="524" y1="58" x2="654" y2="128" marker-end="url(#how-it-works-arrow)" />
<line x1="636" y1="234" x2="656" y2="155" marker-end="url(#how-it-works-arrow)" />
</g>
<g fill="currentColor" stroke="none" font-family="sans-serif" font-size="14" text-anchor="middle">
<text x="65" y="145">Vision</text>
<text x="240" y="135">Router</text>
<text x="240" y="152" font-size="11">checks complexity</text>
<text x="455" y="60">Fast synthesis</text>
<text x="440" y="245">Research</text>
<text x="585" y="245">Analysis</text>
<text x="720" y="145">Text to speech</text>
<text x="335" y="82" font-size="11">simple</text>
<text x="335" y="205" font-size="11">complex</text>
</g>
</svg>"""

PIPELINE_DIAGRAM_HTML = f"""<div>
{PIPELINE_DIAGRAM_SVG}
<p id="{DIAGRAM_DESC_ELEM_ID}">{PIPELINE_DIAGRAM_DESCRIPTION}</p>
</div>"""

# How long handle_submit_staged (below) waits, after yielding the
# completion status + text WITHOUT the audio path, before yielding again
# WITH the audio path - long enough for a screen reader to finish speaking
# the (now short) STATUS_SUCCESS_AUDIO announcement before Gradio mounts
# the autoplaying player (issue #47 / P5.3).
#
# HISTORY: the original #47 fix tried to create this same gap with
# client-side JS instead - autoplay=False on the gr.Audio component, plus
# a shim that waited for a "loadeddata" event on the <audio> element
# before calling .play() itself. That was a dead end: a real-browser check
# (orchestrator) showed that with autoplay=False, Gradio never assigns the
# element a src at all - preload="auto" ran, but src stayed absent,
# readyState was 0, and loadeddata/play/error never fired, so audio never
# played at all despite the full automated suite passing throughout. The
# fix restores autoplay=True (see build_interface) - the only thing that
# makes Gradio assign a source and play it - and creates the gap with
# ordinary Python instead: handle_submit_staged yields the status and text
# first, sleeps AUDIO_PLAY_DELAY_MS, then yields the audio path in a
# second update.
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

# --- Per-node stream progress (issue #80 / P9.1) ----------------------------
#
# graph.invoke() used to be one opaque blocking call with no way to report
# real progress, so STATUS_WORKING above was the only thing a screen reader
# ever heard until the whole pipeline finished. graph.stream(...,
# stream_mode="updates") (verified API on langgraph 1.2.10) yields one dict
# per node the MOMENT it completes, keyed by node name - see
# _run_pipeline_events below for how that's turned into these phrases.
#
# TIMING HONESTY: stream_mode="updates" tells you when a node COMPLETED,
# never when one STARTED - there is no "X has begun" event to hook a
# narration onto. What IS true and knowable without fabricating an observed
# start: the instant one node's completion chunk arrives, its SUCCESSOR
# (clarif_eye.graph.next_node_after) is exactly what begins next - so each
# phrase below is announced for whatever node comes after the one that just
# finished, never for the node currently running. This has one consequence
# worth stating explicitly: vision - the entry node, nothing precedes it in
# the stream - never gets a dedicated phrase of its own, because nothing
# ever completes to trigger one. STATUS_WORKING above already tells the
# user their photo was received and is being read, so nothing is lost by
# not restating it a moment later. tts - the last node - is exactly the
# opposite case: it DOES get announced (as the successor of fast_synth/
# analysis), but next_node_after("tts", ...) returns None, so nothing is
# ever announced AFTER it; the pipeline's actual completion is reported
# separately, by status_for_result once the whole run is done.
STATUS_NODE_RESEARCH = "Looking it up."
# Shared by fast_synth and analysis - both are "the model is writing the
# spoken script now", the honest description of either node from the
# outside, and the issue's own phrase mapping treats them as one narration
# step rather than inventing a distinction a screen-reader user gains
# nothing from.
STATUS_NODE_WRITING = "Writing the description."
STATUS_NODE_TTS = "Turning it into speech."

# node name -> spoken phrase, for whichever node clarif_eye.graph.
# next_node_after names as coming next. This is the ONLY topology
# knowledge ui.py keeps - which node a name maps to in words; WHICH node
# runs next (the graph's edges/routing) lives entirely in next_node_after,
# the single source of truth build_graph() itself uses. No "vision" key:
# vision is never anyone's successor (it's the entry node), so a phrase
# for it would be unreachable dead code, not just unused.
_NODE_PHRASE = {
    "research": STATUS_NODE_RESEARCH,
    "fast_synth": STATUS_NODE_WRITING,
    "analysis": STATUS_NODE_WRITING,
    "tts": STATUS_NODE_TTS,
}

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
# AUDIO SEQUENCING (issue #47 / P5.3, reported from real screen-reader use
# by the owner): the synthesized audio used to start via
# gr.Audio(autoplay=True) at the exact moment the completion status was
# still being announced, so a screen reader and the spoken audio talked
# over each other and neither was intelligible. STATUS_SUCCESS_AUDIO above
# is short, so even a slight overlap is brief, and the gap itself is no
# longer created here in JS: two attempts at a client-side shim (waiting
# for a src to appear, then waiting for a `loadeddata` event) both turned
# out to depend on Gradio actually assigning the <audio> element a source,
# which never happens with autoplay=False. Gradio's `autoplay` prop is
# what causes it to assign a source and play it at all - see
# AUDIO_PLAY_DELAY_MS's comment above for the full history. autoplay=True
# again (see build_interface) and the gap comes from handle_submit_staged
# yielding the status/text before the audio path, sleeping in between.
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
# NO OVERLAP WITH THE AUTOPLAY GAP: the AUDIO_PLAY_DELAY_MS gap (#47) is no
# longer created by this shim at all - it's now created in Python, before
# Gradio ever mounts the player (see handle_submit_staged). Native browser
# autoplay does not go through this wrapped audioEl.play(), so there is
# nothing here to double-delay; this wrapper only ever affects a JS-
# initiated call to .play(), i.e. a user pressing the widget's own Play
# button.
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
#
# PIPELINE DIAGRAM EXEMPTION (issue #56 / P4.4): the image-labelling pass
# below would otherwise mark the new pipeline diagram (see
# PIPELINE_DIAGRAM_SVG's module comment) as decorative and hide it, exactly
# the trap the issue warns about. The SAME apply()/img-classification loop
# now also checks `img.closest("#{DIAGRAM_ELEM_ID}")` - structural,
# by container, the same mechanism `isUploadedPhoto` already uses for
# #photo-input - and leaves anything inside that container alone rather
# than marking it aria-hidden. The diagram's own markup already carries
# role="img", aria-label, and aria-describedby, so nothing further needs
# setting here; this only needs to NOT undo them.
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
    // User-gesture play delay (issue #52 / P5.5) - see the module comment
    // above this shim for the full reasoning. Wrap .play() ONCE per
    // element (guard, same pattern as the other checks in this function)
    // so a click on this control's own Play button - which internally
    // calls this same audioEl.play() - is delayed just like every other
    // caller of it, reusing the SAME observer/apply() pair as the rest of
    // this function rather than a second one. Native browser autoplay
    // (issue #47's gap is now created in Python before Gradio mounts the
    // player - see AUDIO_PLAY_DELAY_MS's comment) does not go through this
    // wrapped method, so it is never double-delayed by it.
    const audioEl = document.querySelector("#{AUDIO_ELEM_ID} audio");
    if (audioEl && !audioEl.dataset.a11yPlayDelayWrapped) {{
      audioEl.dataset.a11yPlayDelayWrapped = "1";
      const nativePlay = audioEl.play.bind(audioEl);
      audioEl.play = function () {{
        return new Promise((resolve, reject) => {{
          setTimeout(() => {{
            nativePlay().then(resolve, reject);
          }}, {USER_PLAY_DELAY_MS});
        }});
      }};
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
      // STRUCTURAL exemption for the "How this works" pipeline diagram
      // (issue #56 / P4.4): true only for an element living inside the
      // diagram's OWN container (#{DIAGRAM_ELEM_ID}), the same closest()-
      // by-container check isUploadedPhoto already uses for #photo-input -
      // never a string/class-name match against the SVG's label or
      // content. This is what stops this very pass from silencing the
      // diagram it would otherwise treat as decorative chrome.
      const isDiagram = img.closest("#{DIAGRAM_ELEM_ID}");
      if (isUploadedPhoto || isDiagram) {{
        // MEANINGFUL: the user's own submitted photo, or the pipeline
        // diagram - both already carry their own real accessible name
        // (alt/aria-label set here for the photo; role="img"+aria-label+
        // aria-describedby set directly in the diagram's own markup), so
        // leave them alone rather than silencing them as chrome.
        if (isUploadedPhoto) {{
          img.setAttribute("alt", "{UPLOADED_PHOTO_ALT}");
          img.setAttribute("aria-label", "{UPLOADED_PHOTO_ALT}");
        }}
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


# --- Image content cache (issue #75) ----------------------------------------
#
# The same photo submitted twice used to cost two model calls against a
# 1,000/day allowance. IMAGE_CACHE_MAX_ENTRIES bounds the cache small and
# in-memory only - it is NEVER written to disk: these are photographs of
# things a blind user cannot see, so persisting them is not acceptable.
#
# Kept <= tts.MAX_KEPT_FILES on purpose: this bound only avoids slots that
# are CERTAIN to be dead. Cache eviction is LRU by ACCESS time (get()
# calls move_to_end()); _prune_old_files deletes by mtime (write time),
# which a cache hit never updates. Because the two structures evict on
# different keys, satisfying this bound does NOT guarantee a live cache
# entry keeps its file - a repeatedly-hit entry stays "live" here while
# its mp3 can still age out of the MAX_KEPT_FILES most-recently-WRITTEN
# set and get pruned. What actually makes this safe is the stale-file
# guard in handle_submit (a missing file is treated as a miss, not a
# lying hit). Sizing this past MAX_KEPT_FILES would add slots that can
# NEVER be served even under the most generous access pattern, which is
# the one case this bound does rule out. Not imported from tts.py to
# avoid coupling the UI to the TTS layer for one integer - keep this
# value <= that one.
IMAGE_CACHE_MAX_ENTRIES = 20


class ImageResultCache:
    """Tiny in-memory LRU cache from image-content hash -> (audio_path,
    text) result, bounded to IMAGE_CACHE_MAX_ENTRIES entries.

    Only successful results are ever stored here (see handle_submit) - a
    quota/API failure must never be replayed to the next visitor as if it
    were that photo's own answer.

    Guarded by a single lock: Gradio serves requests from a thread pool,
    and put()'s assign / move_to_end / conditional popitem sequence is not
    atomic, so without one the size bound could be transiently exceeded
    under concurrent access. One plain (non-reentrant) lock around each
    method's compound operation - nothing fancier.
    """

    def __init__(self, max_entries=IMAGE_CACHE_MAX_ENTRIES):
        self._max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            return self._entries[key]

    def put(self, key, value):
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            if len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def discard(self, key):
        with self._lock:
            self._entries.pop(key, None)


# --- Thread registry (issue #81 / P9.2) -------------------------------------
#
# A LangGraph checkpointer keeps every checkpointed thread's state in memory
# for as long as the process (and, for InMemorySaver specifically, the
# checkpointer instance) lives - see build_resources()'s InMemorySaver
# comment for the honest limits of that. Nothing here ever ends a thread on
# its own: a browser tab can be left open, or a visitor can simply not come
# back, so without a bound the set of live threads (and, worse, each one's
# full checkpointed state - see below) would grow for as long as the
# process stays up.
#
# FOOTPRINT REALITY: every checkpoint stores the FULL graph state, including
# image_data - a base64-encoded JPEG, easily tens of KB per photo - not just
# the small scalar keys. On a free/512MB instance, a handful of large photos
# multiplied across many live threads is a real memory hazard, not a
# theoretical one. That is why THREADS themselves are capped here, not just
# the messages list within one thread (see state.py's MAX_MESSAGES_PER_THREAD
# for the complementary per-thread bound).
#
# CAP CHOICE: MAX_LIVE_THREADS=20, matching IMAGE_CACHE_MAX_ENTRIES above -
# "a double-digit number of concurrent demo sessions" is a reasonable size
# for a small free-tier demo app, and reusing the same order of magnitude as
# the existing image cache bound keeps the two caps easy to reason about
# together. Like every other cap in this codebase (IMAGE_CACHE_MAX_ENTRIES,
# analysis._SCRAPER_DATA_CAP), this is a deliberate, documented guess, not a
# measured optimum - retune it if real usage says otherwise.
MAX_LIVE_THREADS = 20


class ThreadRegistry:
    """Tracks minted thread_ids LRU-style (by last-touched time, not
    creation time), bounded to `max_threads`. When a touch would push the
    registry over that bound, the LEAST-recently-touched thread_id is
    evicted and its checkpointed state is deleted from `checkpointer` via
    delete_thread - verified present and working on
    langgraph.checkpoint.memory.InMemorySaver in langgraph 1.2.10 (see
    build_resources()'s comment) - so eviction here actually frees the
    checkpointer's memory, not just this registry's own bookkeeping.

    Guarded by a single lock, same discipline as ImageResultCache above:
    Gradio serves requests from a thread pool, so the touch/evict sequence
    below is not safe to leave unguarded under concurrent access.
    """

    def __init__(self, checkpointer, max_threads=MAX_LIVE_THREADS):
        self._checkpointer = checkpointer
        self._max_threads = max_threads
        self._ids = OrderedDict()
        self._lock = threading.Lock()

    def touch(self, thread_id):
        with self._lock:
            self._ids[thread_id] = None
            self._ids.move_to_end(thread_id)
            if len(self._ids) > self._max_threads:
                oldest, _ = self._ids.popitem(last=False)
                if self._checkpointer is not None:
                    self._checkpointer.delete_thread(oldest)


def _image_content_key(image_data):
    """Hash of the DECODED image bytes (the base64 JPEG _encode_image
    produces), never a filename/upload path - the same photo uploaded
    twice arrives at a different temp path each time, but re-encodes to
    the same bytes."""
    return hashlib.sha256(image_data.encode("ascii")).hexdigest()


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
    image_cache: object = field(default_factory=ImageResultCache)
    # None by default (issue #81 / P9.2): every existing test that builds
    # an AppResources directly (test_ui.py's FakeGraph-based tests) never
    # sets these, so they keep running an uncheckpointed graph with no
    # thread_id, exactly today's behavior - see _run_pipeline_events, which
    # only touches thread_registry / adds thread_id to config when a
    # caller actually passes one in. build_resources() (the live app) sets
    # both.
    thread_registry: object = None


def build_resources():
    """Construct every injectable ONCE for the life of the process.

    Never raises: a missing OPENROUTER_API_KEY (the likely state of a
    fresh Hugging Face Space with no secret set yet) must not crash the
    app at import/startup time - it degrades to client=None plus a spoken
    message, checked by handle_submit before the graph is ever invoked.
    The research searcher/client are best-effort shared instances too (see
    module docstring); if either fails to construct, they're left None and
    research_node falls back to its own lazy per-call defaults.

    CHECKPOINTING (issue #81 / P9.2): the live app compiles WITH a fresh
    langgraph.checkpoint.memory.InMemorySaver, one per process (the same
    "construct once at startup" discipline this function already follows
    for the client/tts/searcher). HONEST LIMITS, stated here because this
    is where a reader would look for them, not left to be discovered in
    production:
      - InMemorySaver keeps every checkpoint in this PROCESS's own memory -
        nothing is written to disk or any external store. A process
        restart (a redeploy, a crash, Hugging Face Spaces restarting the
        container) loses every thread's history. This is fine for this
        issue's scope (state surviving a run within one session) and is
        never described to the user as more durable than that - no
        user-facing text in this app claims otherwise.
      - A free Hugging Face Space sleeps after ~15 minutes of no traffic;
        waking it starts a NEW process, so the same loss happens on every
        cold start a visitor's return happens to trigger.
      - See ThreadRegistry / MAX_LIVE_THREADS above for how the number of
        live threads (and, in turn, the checkpointer's own memory use) is
        bounded while the process IS running.
    Every existing caller of build_graph() with no arguments, and every
    test in this repo that builds an AppResources directly instead of via
    this function, is unaffected: build_graph()'s `checkpointer` param
    defaults to None, so an uncheckpointed graph needs no thread_id at all
    - only THIS function's live graph does.
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

    checkpointer = InMemorySaver()

    return AppResources(
        graph=build_graph(checkpointer=checkpointer),
        client=client,
        client_error=client_error,
        tts_providers=tts_providers,
        searcher=searcher,
        research_client=research_client,
        thread_registry=ThreadRegistry(checkpointer),
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


def _run_pipeline_events(image, resources, pipeline_budget_seconds, thread_id=None):
    """Generator: the ONE place that knows how to run a photo through the
    pipeline - guards, the image cache, graph execution, and the
    exception/outcome mapping.

    Yields ("status", phrase) once per completed graph node, followed by
    exactly one final ("outcome", (audio_path_or_None, text)) item.
    handle_submit drains this and returns only that last item, so its
    return-tuple contract is unchanged; handle_submit_staged consumes it
    directly so each ("status", ...) item can become its own live yield
    WHILE the graph is still running - the whole point of streaming. Every
    graph injected here - the real one from build_graph(), and every test
    double that stands in for it - is expected to implement .stream(...,
    stream_mode="updates"); there is no invoke()-only fallback, so an
    incompatible double fails loudly (AttributeError) instead of silently
    losing all narration.

    NEVER raises (except KeyboardInterrupt/SystemExit) - every failure mode
    yields a spoken-ready message instead, per the module docstring. The
    try/except below wraps the ENTIRE graph-stream consumption (issue #80 /
    P9.1): with graph.stream() an exception can surface mid-iteration, not
    just from a single invoke() call, so the same discipline has to cover
    the whole loop, not just its start.

    `thread_id` (issue #81 / P9.2) is OPTIONAL and defaults to None -
    exactly today's behavior for every caller that doesn't pass one (the
    entire existing test fleet, which drives an uncheckpointed
    resources.graph). build_interface (below) mints one gr.State per
    browser session and threads it through here; when it's not None, this
    function (a) registers it with resources.thread_registry, if one is
    configured, so a bounded live-thread count is maintained (see
    ThreadRegistry above), and (b) adds it to
    config["configurable"]["thread_id"], which is what a checkpointed
    graph requires to persist/restore state across calls (verified
    empirically - see build_graph's docstring).
    """
    if image is None:
        yield "outcome", (None, NO_IMAGE_MESSAGE)
        return

    if resources.client is None:
        yield "outcome", (None, resources.client_error or CONFIG_ERROR_MESSAGE)
        return

    try:
        image_data = _encode_image(image)
    except Exception:
        yield "outcome", (None, UNREADABLE_IMAGE_MESSAGE)
        return

    # Issue #75: key on a hash of the DECODED image content (never the
    # upload path/filename - the same photo uploaded twice arrives at a
    # different temp path each time), so a repeat photo costs no quota. A
    # hit yields no "status" events at all - see this module's top-level
    # "hits bypass the graph" docstring note - so handle_submit_staged
    # stages a hit exactly like it did before streaming existed.
    cache_key = _image_content_key(image_data)
    cached = resources.image_cache.get(cache_key)
    if cached is not None:
        cached_audio_path, _ = cached
        # A cached audio path could have been deleted since it was stored
        # (tts.py's _prune_old_files DOES delete old mp3s once more than
        # MAX_KEPT_FILES exist; a temp cleaner or disk pressure could too).
        # Returning it anyway would have the UI announce "ready" over
        # silence - the worst failure mode for a user who cannot see the
        # screen. Treat a missing file as a miss: drop the stale entry and
        # fall through to run the pipeline for real.
        if not cached_audio_path or os.path.exists(cached_audio_path):
            yield "outcome", cached
            return
        resources.image_cache.discard(cache_key)

    try:
        state = make_initial_state(image_data)
        configurable = {
            "client": resources.client,
            "tts_providers": resources.tts_providers,
            "searcher": resources.searcher,
            "research_client": resources.research_client,
            "deadline": time.monotonic() + pipeline_budget_seconds,
        }
        if thread_id is not None:
            if resources.thread_registry is not None:
                resources.thread_registry.touch(thread_id)
            configurable["thread_id"] = thread_id
        config = {"configurable": configurable}
        graph = resources.graph
        result = dict(state)
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            for node_name, update in chunk.items():
                result.update(update)
                # next_node_after is the single source of truth for this
                # graph's topology (clarif_eye.graph, right next to
                # build_graph()'s edges) - this module only supplies the
                # WORDING for whatever node it names.
                next_node = next_node_after(node_name, result)
                if next_node is not None:
                    yield "status", _NODE_PHRASE[next_node]
    except LadderExhaustedError as exc:
        # Every node already catches and degrades this internally (see
        # vision.py/synth.py/analysis.py); this branch only matters if the
        # whole pipeline fails before a node can degrade (issue #18 / P6.2
        # scope item 4, e.g. a client-construction failure a node did not
        # catch). Uses the SAME category mapping the nodes use, rather than
        # collapsing into the generic UNEXPECTED_ERROR_MESSAGE below. Not
        # cached (issue #75): a quota/API failure must never be replayed
        # to the next visitor as if it were that photo's own answer.
        yield "outcome", (None, message_for_ladder_exhausted(exc))
        return
    except OpenRouterError as exc:
        yield "outcome", (None, message_for_terminal_error(exc))
        return
    except Exception:
        yield "outcome", (None, UNEXPECTED_ERROR_MESSAGE)
        return

    final_output = (result.get("final_output") or "").strip()
    audio_path = result.get("audio_file_path") or ""

    if audio_path:
        outcome = (audio_path, final_output)
    elif is_chain_exhausted():
        if final_output:
            outcome = (None, f"{final_output} {AUDIO_UNAVAILABLE_NOTE}")
        else:
            outcome = (None, AUDIO_UNAVAILABLE_NOTE)
    else:
        outcome = (None, final_output or UNEXPECTED_ERROR_MESSAGE)

    # The pipeline ran to completion (no exception above), but only a
    # real audio outcome is cached. Outcome (b) - description succeeded,
    # every TTS provider failed for THIS call - is a degraded result, not
    # a success: caching it would permanently mute a photo across a
    # transient TTS outage, replaying text-with-no-audio to every later
    # visitor long after TTS recovered. That violates this cache's own
    # docstring ("a quota/API failure must never be replayed to the next
    # visitor as if it were that photo's own answer"). Quota is cheaper
    # than permanently serving silence to a blind user, so a text-only
    # outcome is left to retry next time.
    if audio_path:
        resources.image_cache.put(cache_key, outcome)
    yield "outcome", outcome


def handle_submit(image, resources, pipeline_budget_seconds=DEFAULT_PIPELINE_BUDGET_SECONDS, thread_id=None):
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

    `thread_id` (issue #81 / P9.2) is OPTIONAL and defaults to None, same
    as _run_pipeline_events - see that function's docstring for what
    passing one actually does.

    All the actual work lives in _run_pipeline_events (issue #80 / P9.1) so
    handle_submit_staged can share it and turn its "status" events into
    live per-node progress yields; this just drains the generator and
    returns its final ("outcome", ...) item, ignoring any "status" events -
    same return-tuple contract as before streaming existed.
    """
    outcome = (None, UNEXPECTED_ERROR_MESSAGE)
    for kind, payload in _run_pipeline_events(image, resources, pipeline_budget_seconds, thread_id=thread_id):
        if kind == "outcome":
            outcome = payload
    return outcome


def handle_submit_staged(
    image, resources, pipeline_budget_seconds=DEFAULT_PIPELINE_BUDGET_SECONDS, thread_id=None
):
    """Generator version of handle_submit that also drives the live-region
    status text (issue #15 / P5.1 scope item 3).

    Yields (status_text, audio_path_or_None, description_text) tuples;
    Gradio streams each yield straight to the UI as it's produced, which is
    what lets the live region announce progress at all.

    STAGING (issue #80 / P9.1 - real per-node progress): the first yield
    is always the honest "received and working, up to about 30 seconds"
    message (submission-received and still-working collapsed into one
    announcement, since nothing has run yet - this doubles as vision's own
    "now reading the photo" announcement, since nothing ever completes to
    trigger a dedicated one for the entry node - see STATUS_NODE_* above).
    Then, for every graph node that completes and HAS a successor,
    _run_pipeline_events yields its own "status" event (see STATUS_NODE_*
    / _NODE_PHRASE above and clarif_eye.graph.next_node_after for the
    phrase mapping and the timing-honesty reasoning), which is forwarded
    here as its own live yield - so a screen reader hears "Looking it up"
    or "Writing the description", then "Turning it into speech", as each
    stage actually finishes, instead of one silent wait. A cache hit, an
    early failure (no image, missing client, unreadable image), or a test
    double whose graph reports its whole run as one opaque completion
    (see FakeGraph's own docstring in tests/test_ui.py) produces no
    "status" events at all, so those cases still yield only the three
    tuples this function always ended with. Once the pipeline's final
    outcome arrives, the last two yields are unchanged from before
    streaming existed: the final status AND description text but NO audio
    yet, so a screen-reader user can read the answer immediately; and,
    only when audio was actually produced, once more after a short delay
    with the SAME status/text plus the audio path, so Gradio only mounts
    the (autoplaying) player once the completion status has had time to be
    spoken - see AUDIO_PLAY_DELAY_MS's comment for why this replaced an
    earlier, broken JS-only attempt at the same gap. A screen reader hears
    each yield in order via aria-live="polite" on the status control.

    `thread_id` (issue #81 / P9.2) is OPTIONAL and defaults to None, same
    as _run_pipeline_events - build_interface passes each browser session's
    own minted thread_id (a gr.State) through here.
    """
    yield STATUS_WORKING, None, ""
    audio_path, text = None, ""
    for kind, payload in _run_pipeline_events(image, resources, pipeline_budget_seconds, thread_id=thread_id):
        if kind == "status":
            yield payload, None, ""
        else:
            audio_path, text = payload
    status = status_for_result(audio_path, is_chain_exhausted())
    if not audio_path:
        yield status, audio_path, text
        return
    # Status + text land immediately (screen reader can read the answer
    # right away); the audio path is withheld for one more beat so
    # Gradio's autoplaying player doesn't mount over the still-being-
    # spoken completion announcement.
    yield status, None, text
    time.sleep(AUDIO_PLAY_DELAY_MS / 1000)
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

    PER-SESSION THREAD_ID (issue #81 / P9.2): thread_state below is a
    gr.State whose value is a callable, `lambda: str(uuid.uuid4())` -
    Gradio calls it once per browser session (see gr.State's own
    docstring: "If a callable is provided, the function will be called
    whenever the app loads to set the initial value of the state"), which
    is exactly "one thread_id minted per session" with no extra wiring.
    It never appears in the UI itself (no label, not one of the
    click's declared `outputs`) - it exists purely to be threaded through
    to handle_submit_staged so a checkpointed resources.graph can persist
    state across that ONE visitor's runs.
    """

    def _submit(image, thread_id):
        yield from handle_submit_staged(image, resources, thread_id=thread_id)

    with gr.Blocks(title="Clarif-Eye") as demo:
        thread_state = gr.State(value=lambda: str(uuid.uuid4()))
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
            # issue #59 / P4.5: Gradio mirrors the webcam by default
            # (WebcamOptions.mirror=True), which suits selfies. This app
            # has no selfie case - users photograph bills, labels, and
            # signs so the vision model can read the text, and a blind
            # user is not looking at the preview to notice a mirrored
            # frame. Mirroring reversed that text before the model ever
            # saw it. Do not restore the default thinking it "looks more
            # natural" - it makes captured text unreadable.
            webcam_options=gr.WebcamOptions(mirror=False),
        )
        submit_button = gr.Button("Describe this photo", variant="primary")
        status_output = gr.Textbox(
            value=STATUS_IDLE,
            label="Status",
            interactive=False,
            elem_id=STATUS_ELEM_ID,
            elem_classes=STATUS_ELEM_CLASSES,
        )
        # autoplay=True (issue #47 / P5.3): Gradio only assigns the <audio>
        # element a source at all when autoplay is on - with it off, no
        # src is ever set and playback can never start by any means (see
        # AUDIO_PLAY_DELAY_MS's comment for how that was diagnosed). The
        # gap between the completion announcement and audio starting is
        # created instead by handle_submit_staged withholding the audio
        # path for one extra yield, not by JS here.
        audio_output = gr.Audio(label="Spoken description", autoplay=True, elem_id=AUDIO_ELEM_ID)
        text_output = gr.Textbox(label="Description (text)", lines=6, elem_id=RESULT_ELEM_ID)

        # issue #49 / P4.3: placed AFTER the result area (never before it),
        # so it never delays someone who came to use the tool and never
        # sits between the live region and the result it announces. See
        # HOW_IT_WORKS_MARKDOWN's module-level docstring for content
        # sourcing; the ordered list here stays text-only and is the
        # accessible source of truth for a screen-reader user.
        gr.Markdown(HOW_IT_WORKS_MARKDOWN, elem_id=HOW_IT_WORKS_ELEM_ID)
        # issue #56 / P4.4: a sighted-friendly diagram alongside (never
        # instead of) the list above. gr.HTML, not gr.Markdown, because the
        # diagram is raw inline SVG with its own role/aria-label/
        # aria-describedby wiring - see PIPELINE_DIAGRAM_SVG's module
        # comment for the exemption that keeps ARIA_LIVE_HEAD's #48 pass
        # from silencing it.
        gr.HTML(PIPELINE_DIAGRAM_HTML, elem_id=DIAGRAM_ELEM_ID)

        submit_event = submit_button.click(
            fn=_submit,
            inputs=[image_input, thread_state],
            outputs=[status_output, audio_output, text_output],
        )
        # Runs client-side only after the handler above has produced its
        # final yield - see FOCUS_RESULT_JS's docstring for why that
        # timing matters (never steals focus mid-interaction).
        submit_event.then(fn=None, inputs=None, outputs=None, js=FOCUS_RESULT_JS)

    return demo
