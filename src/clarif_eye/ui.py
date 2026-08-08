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
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from clarif_eye.client import LadderExhaustedError, OpenRouterClient, OpenRouterError
from clarif_eye.failure_messages import (
    message_for_ladder_exhausted,
    message_for_terminal_error,
)
from clarif_eye.deep_path import build_deep_path_graph
from clarif_eye.graph import (
    DEEP_PATH_NODE,
    DEFAULT_PIPELINE_BUDGET_SECONDS,
    INTERRUPT_CHUNK_KEY,
    INTERRUPT_REASON_ANSWER,
    RESUME_CONTINUE,
    RESUME_RETAKE,
    TTS_NODE,
    build_graph,
    next_node_after,
)
from clarif_eye.preferences import detect_preference_command, set_verbosity, VERBOSITY_SHORT
from clarif_eye.speech import to_spoken_text as _to_spoken_text
from clarif_eye.state import make_initial_state
from clarif_eye.tts import DEFAULT_PROVIDER_CHAIN, is_chain_exhausted, run_tts

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
# Issue #82 / P9.3: the ask button was activated with an empty question box.
# Guarded here rather than left to the graph, for a reason that is not just
# tidiness: clarif_eye.graph.entry_destination routes on the question being
# non-blank, so a blank one would fall through to `vision` and re-run the
# WHOLE photo pipeline - a second vision call the user never asked for.
NO_QUESTION_MESSAGE = (
    "No question was typed. Please type a question about the photo, then "
    "activate the ask button."
)
# Issue #86 / P9.7: spoken back when the follow-up box was recognised as a
# preference COMMAND, not a question (see clarif_eye.preferences.
# detect_preference_command). Named constants, same reasoning as every
# other spoken message here: what is actually said must be readable from
# the code, not guessed at from behaviour, and a later reword can't
# silently break a test that was matching prose. Says WHAT changed and that
# it is FOR THIS SESSION - a user who cannot see any settings screen has no
# other way to learn what "shorter descriptions please" actually did, or
# how long it lasts.
PREFERENCE_CONFIRMATION_SHORT = (
    "Understood. Descriptions will be shorter for the rest of this "
    "session."
)
PREFERENCE_CONFIRMATION_DETAILED = (
    "Understood. Descriptions will include more detail for the rest of "
    "this session."
)
# Issue #84 / P9.5: the text-only API route was called with nothing to
# describe. Worded for an API caller (there is no button and no photo in
# this route), which is why it does not reuse NO_QUESTION_MESSAGE.
NO_DOCUMENT_TEXT_MESSAGE = (
    "No document text was provided. Send the text of the document to "
    "describe."
)
# The two answers to "a number could not be checked" (issue #83 / P9.4).
# These are BUTTON LABELS, but they are declared here with the other spoken
# constants and not beside their elem_ids, because they are quoted verbatim
# inside two spoken messages below (the question itself, and the refusal
# when a follow-up is typed while it is pending). A label and the sentence
# telling a blind user which label to activate must never be able to drift
# apart.
RESUME_CONTINUE_LABEL = "Continue anyway"
RESUME_RETAKE_LABEL = "I'll retake the photo"

# Issue #83 / P9.4: a resume button was activated when no run is waiting on
# an answer. Two different situations produce this and BOTH are honestly
# covered by one message, so no guessing is needed to tell them apart:
#   - a stray activation (the buttons were left visible after some other
#     run finished, or a keyboard user tabbed onto one out of habit);
#   - the pause is genuinely gone because THIS PROCESS restarted between
#     the question and the answer. The pause lives in the checkpointer,
#     which is an InMemorySaver (see build_resources) - it survives exactly
#     as long as the process does, and Render's free tier spins the service
#     down after ~15 minutes of no traffic. Nothing here pretends otherwise.
# Detected STRUCTURALLY (graph.get_state(...).interrupts is empty), never by
# catching an exception out of a resume attempt - see _run_resume_events.
NOTHING_TO_RESUME_MESSAGE = (
    "There is nothing waiting for an answer right now. If you were asked "
    "about a number that could not be checked, please submit the photo "
    "again."
)
# Issue #83 / P9.4: a follow-up question was typed while the app is still
# waiting to be told whether to speak a number it could not check.
#
# THE DECIDED RULE IS TO REFUSE, and to refuse LOUDLY. The alternative -
# letting the follow-up run - is what the code used to do by accident, and
# it silently DESTROYED the pending question (LangGraph supersedes a
# pending task when a new input arrives on the thread), leaving two answer
# buttons on screen wired to a resume that would then find nothing to
# resume. A safety question the user never answered must never be thrown
# away by a side action.
#
# It restates the situation and NAMES BOTH BUTTONS, because a user who
# cannot see the screen has no other way to find out what is blocking them
# or how to get past it - "you can't do that right now" on its own would
# be a dead end.
QUESTION_PENDING_MESSAGE = (
    "There is still a question waiting for your answer: a number in the "
    "description could not be checked against the photo. Please activate "
    f'"{RESUME_CONTINUE_LABEL}" to hear the description anyway, or '
    f'"{RESUME_RETAKE_LABEL}" to take a new photo. Then you can ask your '
    "question."
)
# The same refusal when what is pending is a question about a follow-up
# ANSWER (issue #92 / P9.11 deep-review MAJOR). Which one is spoken is
# decided STRUCTURALLY, from the pending interrupt payload's own `reason` -
# see _question_pending_message. The message above would tell this user a
# DESCRIPTION is waiting when an answer to their own question is, and would
# offer "take a new photo" as the way past it when the photo was never the
# problem: they can simply ask again.
ANSWER_QUESTION_PENDING_MESSAGE = (
    "There is still a question waiting for your answer: a number in the "
    "answer to your last question could not be checked against the photo. "
    f'Please activate "{RESUME_CONTINUE_LABEL}" to hear that answer anyway, '
    f'or "{RESUME_RETAKE_LABEL}" to leave it unread. Then you can ask your '
    "new question."
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
# Issue #82 / P9.3: the follow-up question box and its own submit button.
# Both carry elem_ids for the same reason every other control here does -
# so the accessibility audit script and its tests can find them
# structurally, by id, rather than by guessing at a label string.
QUESTION_INPUT_ELEM_ID = "question-input"
ASK_BUTTON_ELEM_ID = "ask-button"
# Issue #83 / P9.4: the two answers to "a number could not be checked".
# Ordinary gr.Buttons, so they are ordinary tab stops with real labels the
# moment they become visible - no custom keyboard wiring, nothing to get
# wrong. They carry elem_ids for the same reason every other control here
# does: so the accessibility tests find them structurally, by id.
RESUME_CONTINUE_BUTTON_ELEM_ID = "resume-continue-button"
RESUME_RETAKE_BUTTON_ELEM_ID = "resume-retake-button"

# --- Two tabs: product and explanation (issue #87 / P9.8) ------------------
#
# Owner request, directly: "two tabs on top, one tab for the product, and
# one tab on the explanation". Everything used to sit on one long page;
# splitting it keeps the product view uncluttered and gives the
# demo/explainer content (Phase 9's LangGraph write-up + diagram) room to
# grow without pushing the photo/description controls further down the
# page every time a sentence is added below them.
#
# LABELS ARE PLAIN LANGUAGE ON PURPOSE, not "Product"/"Explanation" (jargon
# that means nothing to the person using this): "Describe a photo" says
# what the tab DOES, "How it works" says what the tab explains. Gradio
# renders each gr.Tab's `label` as the accessible name of its tab button
# (verified by building a throwaway Tabs/Tab interface and inspecting the
# component tree - see tests/test_accessibility.py's P9.8 section for how),
# so these strings are both the visible tab text and the name a screen
# reader announces - no separate aria-label needed.
PRODUCT_TAB_ELEM_ID = "product-tab"
PRODUCT_TAB_LABEL = "Describe a photo"
EXPLANATION_TAB_ELEM_ID = "explanation-tab"
EXPLANATION_TAB_LABEL = "How it works"

# Accessible name given to the user's own uploaded/captured photo preview
# (issue #48 / P5.4 - see ARIA_LIVE_HEAD's image-labelling comment below).
UPLOADED_PHOTO_ALT = "The photo you submitted"

# --- "How this works" section (issue #49 / P4.3) ---------------------------
#
# Owner request: a section explaining the pipeline, the data flow, and the
# LangGraph implementation, since this is a demo application. UPDATED for
# issue #87 / P9.8: this content now lives on its own explanation tab
# (EXPLANATION_TAB_LABEL) rather than at the bottom of a single long page -
# see build_interface() below. Content below is checked against the source
# it describes, not against the
# issue's own wording or the (older, no longer accurate) architecture doc:
#   - graph.py: build_graph() registers exactly 7 nodes (entry, vision,
#     fast_synth, deep_path, followup, verify_answer, tts) - `verify_answer`
#     added by issue #92 / P9.11, the follow-up path's own asking node.
#     `deep_path` is a whole COMPILED
#     CHILD GRAPH mounted as one node (issue #84 / P9.5, see
#     clarif_eye.deep_path): research, analysis and verify_numbers are its
#     nodes now, not the parent's. "router" is NOT one of
#     them - dynamic_router is the function evaluated by a conditional edge
#     out of "vision"; it is plain Python (state["complexity_flag"] in/out,
#     no client, no network - see router.py's module docstring "computed
#     locally with no model call, per the architecture doc's requirement
#     that routing be pure Python"). "entry" IS a real registered node, but
#     it routes differently again: it returns Command(goto=...) and picks
#     its own successor rather than having an edge evaluated for it (issue
#     #82 / P9.3 - see graph.py's "TWO ROUTING MECHANISMS" block for why
#     both mechanisms are in this graph deliberately).
#   - state.py: ClarifEyeState is an 11-key TypedDict (issue #81 / P9.2 added
#     `messages`, the app's first LangGraph reducer; #82 / P9.3 added
#     `question`; #83 / P9.4 added `verification_hold`; #93 / P9.12 added
#     `output_degraded`). Counted off the TypedDict itself, not carried
#     forward from the last time this comment was edited - it had drifted
#     twice.
#   - registry.py / config/models.toml: the "eyes" and "brain" roles each
#     hold an ORDERED ladder of free-only (":free", policy D10) model IDs,
#     tried in turn on failure.
#   - pyproject.toml's [project].dependencies lists langgraph, not
#     langchain - this app does not use langchain, so the text below never
#     claims it does.
#   - tts.py: DEFAULT_PROVIDER_CHAIN is (EdgeTtsProvider, GttsProvider); if
#     every provider fails, audio_file_path == "" and the UI falls back to
#     text (see this module's "THE THREE OUTCOMES" docstring above).
#   - verification.py (imported by analysis.py and, since issue #92 / P9.11,
#     by followup.py, both as _unverified_numbers): on the deep-analysis
#     path and on the follow-up path, every number-like token in the drafted
#     script is checked against the photographed text (+ scene description,
#     + any web lookup on the deep path - the follow-up prompt has no web
#     lookup in it, so neither does its haystack) before it is spoken. A
#     token that doesn't trace back stops the run and asks the user (issue
#     #83 / P9.4 - see graph.verify_numbers_node and graph.verify_answer_node),
#     rather than silently degrading to a "could not be verified" message.
#     fast_synth.py has no equivalent check, so the text below says which
#     paths have one rather than claiming "always".
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
# user nothing extra to skip past by navigating to the next heading. UPDATED
# for issue #87 / P9.8: this content lives on its own explanation tab now,
# not stacked after the result textbox on the same page - a stronger version
# of the same goal the old placement served. It cannot merely be scrolled
# past while using the tool, it is on a DIFFERENT TAB entirely, so it can
# never delay someone submitting a photo and can never sit between the live
# region and the result it announces, since the product tab's layout is
# unaffected by anything here.
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
   doesn't trace back to what the camera actually saw, the app stops and
   asks you: it reads out the description it wrote, tells you which number
   it could not check, and offers two buttons - hear it anyway, or take a
   new photo. Nothing is read aloud as fact until you choose. That question
   is only ever asked about a number that failed this check.
7. The final script is converted to speech.
8. You can then type a question about that same photo. The app answers it
   from the text and scene description it already read, so it does not look
   at the photo again and you do not have to take another one. That answer
   is spoken the same way the description is.
9. An answer to your question goes through the same number check, and the
   same two buttons, for the same reason: a question about an expiry date, a
   total or a dose is exactly where a made-up number would do the most harm.
   The check is a strict one, so it sometimes stops on a number the app read
   correctly and simply wrote out differently. That is why it asks instead of
   refusing: you still get to hear the answer.

### Inside the LangGraph pipeline

This pipeline is built with [LangGraph](https://github.com/langchain-ai/langgraph)'s
`StateGraph` (this app depends on `langgraph`, not `langchain` - there is no
LangChain in this codebase). The graph state is an 11-key `TypedDict`:
`image_data`, `ocr_output`, `scene_context`, `complexity_flag`,
`scraper_data`, `final_output`, `audio_file_path`, `messages`, `question`,
`verification_hold`, `output_degraded`.

Seven nodes are registered: `entry`, `vision`, `fast_synth`, `deep_path`,
`followup`, `verify_answer`, and `tts`. Every run starts at
`entry`, which does
no work of its own: it looks at whether this run carries a photo or a typed
question and sends the run to `vision` or to `followup` accordingly, by
returning a `Command` naming the next node.

On the photo route, the step after `vision` is chosen by a conditional edge
evaluated against `complexity_flag`: `False` goes to `fast_synth` then
straight to `tts`; `True` goes to `deep_path`, then `tts`.

`deep_path` is not an ordinary step. It is a second, separately compiled
graph - `research`, then `analysis`, then optionally `verify_numbers` -
mounted inside the first one as a single node. It keeps its own state, with
its own names for things, and a small piece of code translates between the
two at the boundary. Two things follow from that, and both are deliberate:
the photo itself never enters it (it works from the text already read, so
the closer-look steps have no copy of your picture at all), and the same
graph can be used on its own by something that has no photo to begin with -
there is an API endpoint that describes a document from its text alone.

Two steps can PAUSE the whole run: `verify_numbers` inside `deep_path`, and
`verify_answer` on the question route. They are the same code, wired into
two graphs under two names, and each is guarded by a conditional edge that
decides whether it runs at all.

Taking the first one:
When `analysis` writes a
number it cannot trace back to the photographed text, it records that fact
in the state, and the edge out of `analysis` sends the run to
`verify_numbers` instead of straight to the end of that inner graph. That
node raises a LangGraph
interrupt carrying the drafted script and the numbers that failed, and the
run stops there - before speech - until you answer. The question is asked
from inside the inner graph and travels out to the app around it, and your
answer travels back in to the exact step that asked. On every run where the
numbers check out, the step is skipped entirely. Your answer resumes the same conversation thread exactly where it
stopped. It sits after `analysis` rather than inside it on purpose:
resuming re-runs the paused step from its start, so putting the question
after the writing model means answering it never spends a second model
call. A paused run lives in this server's memory only, so if the service
restarts while it is waiting, the question is gone and you are asked to
submit the photo again.

`verify_answer` is the same story on the question route: `followup` checks
the answer it drafted against the photographed text, records any number it
cannot trace back, and the edge out of `followup` sends the run to
`verify_answer` rather than straight to `tts`. You get the same two buttons
and the same choice. Only the wording after "never mind" differs, because
there your photo was never the problem - you can simply ask again.
That routing decision is evaluated locally in Python, with no model call -
it is a deliberate design point, not an implementation shortcut: the router
only ever needs to read text density and keywords, so it would be wasteful
and slower to spend a model call deciding whether to spend a bigger one.

The two routing mechanisms are both used on purpose. A conditional edge
suits the choice after `vision`, where one node works out a flag and a
separate plain function reads it. A `Command` suits `entry`, which has no
earlier node to work anything out for it and so decides its own next step
from the run's own input.

Each browser session gets its own conversation thread, kept in this
server's memory for as long as the process runs, which is what lets a typed
question be answered from the photo you already sent. Nothing is written to
disk, and a restart clears every thread.

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
  check does not currently run on the quick-description path. The app only
  ever stops to ask you something when that specific check fails - never
  because it is generally unsure.
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

# SCOPE, STATED IN THE DESCRIPTION ITSELF (issue #82 / P9.3): the drawing
# covers the PHOTO route only. The follow-up route (entry -> followup ->
# text to speech) is described in words in the ordered list above instead
# of being drawn, and the description below says so rather than letting a
# reader assume the picture is the whole graph. Redrawing the boxes to fit
# a second branch would buy a sighted reader little - the follow-up route
# is three steps in a straight line - and the list, not the diagram, is
# this section's accessible source of truth either way.
PIPELINE_DIAGRAM_DESCRIPTION = (
    "This diagram covers what happens to a photo. Answering a typed "
    "follow-up question is a shorter route, described in the numbered list "
    "above. "
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
# The follow-up equivalent of STATUS_WORKING (issue #82 / P9.3): the opening
# announcement for a typed question. Shorter about the wait than
# STATUS_WORKING is, because it is honest to be - a follow-up costs ONE
# model call over already-stored text, with no vision call and no web
# lookup, so the "up to about 30 seconds" warning a photo needs would
# overstate it.
STATUS_ASKING = "Question received. Working out the answer now."
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
# Follow-up questions only (issue #82 / P9.3): the `followup` node reads the
# stored description of the photo and works out an answer to what was typed.
# Deliberately NOT reusing STATUS_NODE_WRITING ("Writing the description."),
# which would be a lie here - no description is being written, a question is
# being answered, and the user just typed that question so they know which
# of the two they asked for.
STATUS_NODE_ANSWERING = "Working out the answer."
# The opening announcement for a resume (issue #83 / P9.4) - the equivalent
# of STATUS_WORKING/STATUS_ASKING for the third way a run can start. Says
# nothing about how long it will take because, unlike either of those, the
# work left is only the speech at the end: the model call already happened
# before the question was asked.
STATUS_RESUMING = "Thank you. Finishing up now."

# node name -> spoken phrase, for whichever node clarif_eye.graph.
# next_node_after names as coming next. This is the ONLY topology
# knowledge ui.py keeps - which node a name maps to in words; WHICH node
# runs next (the graph's edges/routing) lives entirely in next_node_after,
# the single source of truth build_graph() itself uses.
#
# THREE NODES DELIBERATELY HAVE NO PHRASE, and a missing key means
# "announce nothing" (see _run_pipeline_events, which looks this up with
# .get() and skips a None):
#   - "vision": announced only as entry's successor, which happens the
#     instant the run starts - STATUS_WORKING has just said "Photo received.
#     Describing it now" and a second announcement a few milliseconds later
#     would be the same sentence twice, back to back, in a screen reader.
#   - "entry": never anyone's successor (it is the first node), so nothing
#     ever completes to trigger a phrase for it - the same reason vision had
#     no phrase before issue #82 added a node in front of it. It also does
#     no work worth narrating: it returns a Command(goto=...) and completes
#     instantly, so "announcing" it would be narrating nothing.
#   - "verify_numbers" (issue #83 / P9.4): entered only on a run that is
#     about to stop and ask the user a question, and the question itself
#     arrives as the very next announcement. A phrase here would be a
#     sentence spoken over the top of the one that actually matters. On
#     every other deep-analysis run the graph routes straight past this
#     node (see clarif_eye.graph.analysis_destination), so nothing is
#     announced for it there either - `analysis` completing still leads to
#     "Turning it into speech", exactly as it did before this issue.
#   - "research" (issue #84 / P9.5): it HAS an announcement, but not under
#     its own name any more. It is the deep path's first node, and since the
#     deep path became a child graph the router names DEEP_PATH_NODE, not
#     "research" - so the "Looking it up." entry moved to that key. Nothing
#     ever names "research" as a successor now, and a phrase here would be
#     dead wording nobody could hear.
# The TIMING-HONESTY comment above still holds exactly as written: every
# phrase here is announced for the node that is ABOUT to run, at the moment
# its predecessor's completion chunk arrives.
_NODE_PHRASE = {
    # The whole deep path (issue #84 / P9.5), announced when `vision`
    # completes and the router names it. The phrase is the LOOKUP one
    # because the lookup is what the deep path does first - `research` is
    # the child graph's entry node - and this announcement lands at the
    # exact moment "Looking it up." used to, when `research` was a node of
    # the parent graph and the router named it directly. What the user hears
    # is unchanged; only which name the graph reports has moved.
    DEEP_PATH_NODE: STATUS_NODE_RESEARCH,
    "fast_synth": STATUS_NODE_WRITING,
    # A CHILD graph's node (clarif_eye.deep_path), reached through
    # stream(subgraphs=True) - see _narrate_stream. Announced when the
    # child's `research` node completes, exactly as before.
    "analysis": STATUS_NODE_WRITING,
    "followup": STATUS_NODE_ANSWERING,
    TTS_NODE: STATUS_NODE_TTS,
}


# --- The spoken question (issue #83 / P9.4) --------------------------------
#
# THE PAYLOAD IS STRUCTURAL, THE QUESTION IS PROSE, and the two are kept
# apart on purpose. clarif_eye.graph.verify_numbers_node interrupts with
# {"reason", "script", "numbers"} - fields, not a sentence - so this is the
# ONLY place that has to decide how any of it sounds, and nothing
# downstream ever parses a number back out of English.
#
# WHAT IT MUST SAY, in this order, because a listener has no screen to
# glance back at: what was read, that a number in it could not be checked,
# which number, and what the two choices are (named with the EXACT button
# labels, so "activate Continue anyway" points at something findable).
INTERRUPT_QUESTION_TEMPLATE = (
    "Here is the description that was written: {script} "
    "One number in it could not be checked against the photo: {numbers}. "
    'Activate "{continue_label}" to hear the description anyway, or '
    '"{retake_label}" to take a new photo instead.'
)
# The SAME question about a follow-up ANSWER (issue #92 / P9.11 deep-review
# MAJOR). Two differences from the template above, both of them defects
# being fixed rather than polish:
#   - what was written is "the answer", not "the description". The user
#     typed a question a moment ago and is being read the reply to it.
#   - the second choice says WHAT IT ACTUALLY DOES. The photo wording sent
#     the user off to "take a new photo instead" - and then the retake
#     button spoke ANSWER_RETAKE_CONFIRMATION, which tells them the photo
#     they already sent is still there. A spoken flow that contradicts the
#     instruction it just gave is worse than a clumsy one, because there is
#     no screen to glance at and check.
# The button LABEL is still quoted verbatim, so "activate I'll retake the
# photo" points at something findable in the tab order - only the
# description of its effect is honest to this path.
INTERRUPT_ANSWER_QUESTION_TEMPLATE = (
    "Here is the answer that was written: {script} "
    "One number in it could not be checked against the photo: {numbers}. "
    'Activate "{continue_label}" to hear the answer anyway, or '
    '"{retake_label}" to leave it unread and ask again.'
)
# Used when the payload is not the shape this module expects. Should be
# unreachable - the only things that raise this interrupt are
# verify_numbers_node and verify_answer_node, right next door - but this
# module's contract is "never raise into Gradio", and a KeyError while
# building a sentence would cost the user the whole run rather than one
# detail of it. One per path, for the same reason the templates are: a
# fallback that named the wrong thing would be a quieter version of the very
# defect above.
INTERRUPT_QUESTION_FALLBACK = (
    "A number in the description could not be checked against the photo. "
    f'Activate "{RESUME_CONTINUE_LABEL}" to hear the description anyway, or '
    f'"{RESUME_RETAKE_LABEL}" to take a new photo instead.'
)
INTERRUPT_ANSWER_QUESTION_FALLBACK = (
    "A number in the answer could not be checked against the photo. "
    f'Activate "{RESUME_CONTINUE_LABEL}" to hear the answer anyway, or '
    f'"{RESUME_RETAKE_LABEL}" to leave it unread and ask again.'
)


def _asked_about_an_answer(payload):
    """True when the pending question is about a follow-up ANSWER rather
    than a drafted description.

    STRUCTURAL (D15): reads the `reason` field the asking node stamped on
    its own payload (clarif_eye.graph.INTERRUPT_REASON_ANSWER), never the
    wording of anything. Defaults to False - the photo path - for a payload
    of any shape this module does not recognise, which is the older and
    more common flow and therefore the safer thing to be wrong about.
    """
    try:
        return payload.get("reason") == INTERRUPT_REASON_ANSWER
    except Exception:
        return False


def _interrupt_question(payload):
    """Turn an asking node's structural interrupt payload into the sentence
    a screen reader reads out. Never raises - see the two fallbacks above."""
    about_answer = _asked_about_an_answer(payload)
    fallback = (
        INTERRUPT_ANSWER_QUESTION_FALLBACK if about_answer else INTERRUPT_QUESTION_FALLBACK
    )
    try:
        script = (payload["script"] or "").strip()
        numbers = ", ".join(str(number) for number in payload["numbers"])
        if not script or not numbers:
            return fallback
        template = (
            INTERRUPT_ANSWER_QUESTION_TEMPLATE if about_answer else INTERRUPT_QUESTION_TEMPLATE
        )
        return template.format(
            script=script,
            numbers=numbers,
            continue_label=RESUME_CONTINUE_LABEL,
            retake_label=RESUME_RETAKE_LABEL,
        )
    except Exception:
        return fallback


def _question_pending_message(payload):
    """The refusal spoken when the user types while a safety question is
    still waiting - worded for the question that is ACTUALLY pending.

    Same structural signal as _interrupt_question above, read off the same
    payload, so the refusal can never describe a different pending question
    than the one the user was actually asked.
    """
    return (
        ANSWER_QUESTION_PENDING_MESSAGE
        if _asked_about_an_answer(payload)
        else QUESTION_PENDING_MESSAGE
    )

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


class _BoundedLRU:
    """Shared bounded-LRU machinery: OrderedDict + lock + move_to_end +
    conditional popitem, bounded to `max_entries`. ImageResultCache
    (issue #75) and ThreadRegistry (issue #81 / P9.2) both need exactly
    this - the same lock/eviction sequence duplicated in two classes would
    be one more place to get subtly wrong (e.g. only one of them staying
    thread-safe after a future edit). `on_evict(key, value)`, if given, is
    called with the evicted (key, value) pair OUTSIDE the lock (never
    while holding it, so an evict callback that itself takes time - e.g.
    ThreadRegistry's checkpointer.delete_thread - can't create lock
    contention or reentrancy issues).

    Guarded by a single (plain, non-reentrant) lock: Gradio serves
    requests from a thread pool, and the assign / move_to_end /
    conditional popitem sequence below is not atomic, so without one the
    size bound could be transiently exceeded under concurrent access.
    """

    def __init__(self, max_entries, on_evict=None):
        self._max_entries = max_entries
        self._on_evict = on_evict
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def _get(self, key):
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            return self._entries[key]

    def _touch(self, key, value=None):
        evicted = None
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            if len(self._entries) > self._max_entries:
                evicted = self._entries.popitem(last=False)
        if evicted is not None and self._on_evict is not None:
            self._on_evict(*evicted)

    def _discard(self, key):
        with self._lock:
            self._entries.pop(key, None)


class ImageResultCache(_BoundedLRU):
    """Tiny in-memory LRU cache from image-content hash -> (audio_path,
    text, ocr_output, scene_context), bounded to IMAGE_CACHE_MAX_ENTRIES
    entries.

    Only successful results are ever stored here (see handle_submit) - a
    quota/API failure must never be replayed to the next visitor as if it
    were that photo's own answer.

    WHY ocr_output/scene_context ARE STORED TOO (deep-review BLOCKER, issue
    #82 / P9.3): a cache hit short-circuits the graph entirely, so nothing
    writes those keys onto the caller's THREAD - and follow-up questions are
    answered from the thread. That desynchronised what the user had just
    been told from what the thread remembered, two ways:
      a. one thread, photo A then photo B then photo A again: the third
         submit hit the cache, the checkpoint still held B, and a follow-up
         answered about B while the user was holding A.
      b. two visitors, one process-wide cache: visitor two submitted a photo
         visitor one had already sent, heard the cached description, and -
         their own thread having never run the graph - was told there was no
         photo yet when they asked about it.
    Neither is visible to a blind user, which is what made them blockers
    rather than rough edges. Carrying the two keys here lets
    _run_pipeline_events write them onto the thread on a hit (see its
    cache-hit branch), so the thread always describes the photo the user was
    last told about.

    NO NEW PRIVACY EXPOSURE: ocr_output/scene_context are text DERIVED from
    the photo, the same class of thing as the `text` already cached here,
    held in this process's memory only and never written to disk. The base64
    photo itself is deliberately NOT stored.

    NO MIGRATION CONCERN: this cache has no persisted format - it lives and
    dies with the process - so widening the entry only needs every put/get
    site in this module updated, which this change does.
    """

    def __init__(self, max_entries=IMAGE_CACHE_MAX_ENTRIES):
        super().__init__(max_entries)

    def get(self, key):
        return self._get(key)

    def put(self, key, value):
        self._touch(key, value)

    def discard(self, key):
        self._discard(key)


# How much document text the text-only API route will ever send to the model.
#
# WHY A CAP AT ALL: everywhere else in this app, the text handed to the deep
# path is whatever a vision pass read off ONE photograph - bounded by
# physics. That route takes text straight from a caller, so nothing bounds
# it. Probed before this constant existed: a 2,361,688-character body went to
# the brain model in full. That is a quota hazard on a rate-limited free tier
# (see describe_document_text on the shared allowance), and a correctness one
# too - a model that truncates rather than errors would silently answer about
# the first fraction of the document while sounding exactly as confident, the
# "partial reproduction dressed as success" failure analysis.py's own cap
# comment describes.
#
# THE NUMBER: 20,000 characters, roughly 3,000 words. Chosen to sit
# comfortably ABOVE what this route's own purpose can produce and well below
# where prompt size becomes the problem: a densely printed A4 page holds
# ~3,000-5,000 characters, so this is several pages' worth of an unusually
# text-heavy document. Like every other cap in this codebase
# (analysis._SCRAPER_DATA_CAP, IMAGE_CACHE_MAX_ENTRIES, MAX_LIVE_THREADS), it
# is a documented, deliberate guess rather than a measured optimum.
DOCUMENT_TEXT_CAP = 20_000

# How many distinct documents the text route remembers at once. Matched to
# IMAGE_CACHE_MAX_ENTRIES for the same reason that one is 20 - a double-digit
# number of live entries suits a small free-tier demo - and because two
# caches of wildly different sizes would be harder to reason about together
# than two of the same order.
DOCUMENT_CACHE_MAX_ENTRIES = 20


def _cap_document_text(document_text, cap=DOCUMENT_TEXT_CAP):
    """Truncate `document_text` to `cap` characters at a word boundary,
    leaving a visible marker.

    THE SAME SHAPE AS analysis._cap_scraper_data, deliberately, and for the
    same reason that one gives: a silent cut mid-word - or, worse,
    mid-number - reads as real evidence, so the truncation is made visible in
    the prompt body instead. Not imported from that module: it is a private
    helper there, it is worded for scraped web context ("[context
    truncated]"), and this text is the document itself rather than supporting
    material - the two say different things to the model and should be able
    to keep doing so.
    """
    if len(document_text) <= cap:
        return document_text
    truncated = document_text[:cap].rsplit(" ", 1)[0]
    return f"{truncated} [document truncated]"


class DocumentTextCache(_BoundedLRU):
    """Tiny in-memory LRU cache from document-text hash -> the spoken-ready
    description, bounded to DOCUMENT_CACHE_MAX_ENTRIES entries.

    THE SAME RULES AS ImageResultCache, because it exists for the same reason
    and spends the same allowance: only real, successful outcomes are ever
    stored (see describe_document_text), a failure is never replayed to the
    next caller as if it were that document's answer, and nothing is ever
    written to disk. It is SIMPLER than that one because this route produces
    less: there is no audio path to go stale, and no thread whose stored
    ocr/scene could drift out of step with what the caller was told, so the
    entry is just the text.
    """

    def __init__(self, max_entries=DOCUMENT_CACHE_MAX_ENTRIES):
        super().__init__(max_entries)

    def get(self, key):
        return self._get(key)

    def put(self, key, value):
        self._touch(key, value)


def _document_text_key(document_text):
    """Hash of the CAPPED text, so two bodies that differ only past the cap -
    i.e. two requests this route would answer identically - share one entry
    instead of each paying for the same model call."""
    return hashlib.sha256(document_text.encode("utf-8")).hexdigest()


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
# for the complementary per-thread bound) - AND why a single thread's OWN
# checkpoint history is separately trimmed after every completed run (see
# _trim_thread_to_latest_checkpoint below): capping thread COUNT alone does
# nothing for a single long-lived thread that keeps getting invoked, and
# capping messages alone does nothing about the checkpointer's own
# unpruned per-invoke history (measured: ~134KB/invoke with a 50KB image,
# 8MB after 10 invokes with a 400KB image, all on ONE thread, before this
# fix - see _trim_thread_to_latest_checkpoint's own comment).
#
# CAP CHOICE: MAX_LIVE_THREADS=20, matching IMAGE_CACHE_MAX_ENTRIES above -
# "a double-digit number of concurrent demo sessions" is a reasonable size
# for a small free-tier demo app, and reusing the same order of magnitude as
# the existing image cache bound keeps the two caps easy to reason about
# together. Like every other cap in this codebase (IMAGE_CACHE_MAX_ENTRIES,
# analysis._SCRAPER_DATA_CAP), this is a deliberate, documented guess, not a
# measured optimum - retune it if real usage says otherwise.
MAX_LIVE_THREADS = 20


class ThreadRegistry(_BoundedLRU):
    """Tracks minted thread_ids LRU-style (by last-touched time, not
    creation time), bounded to `max_threads`. When a touch would push the
    registry over that bound, the LEAST-recently-touched thread_id is
    evicted and its checkpointed state is deleted from `checkpointer` via
    delete_thread - verified present and working on
    langgraph.checkpoint.memory.InMemorySaver in langgraph 1.2.10 (see
    build_resources()'s comment) - so eviction here actually frees the
    checkpointer's memory, not just this registry's own bookkeeping.

    EVICTING A THREAD THAT IS MID-RUN: touch() (and therefore eviction) is
    called BEFORE a run starts (see thread_configurable), so the thread
    being evicted is always some OTHER, older thread - never the one this
    request is about. But that older thread's checkpointed history can
    still be deleted while its OWN run is still in flight (e.g. a slow
    research-path request still executing when 20 newer sessions arrive
    and push it out). This does NOT crash that in-flight run: LangGraph's
    InMemorySaver.put()/put_writes() re-create a thread's dict entries on
    demand (defaultdict), so a mid-run write after delete_thread just
    starts a fresh history rather than raising. The real cost is
    correctness, not a crash: that run's PRIOR turns (before this
    request) are gone from get_state() once it finishes, since
    delete_thread doesn't distinguish "prior history" from "in-flight
    checkpoint". Requires MAX_LIVE_THREADS=20 (or more) OTHER concurrent
    distinct sessions to trigger - not the common case for a free-tier
    demo app, and accepted as such rather than solved with, e.g., a
    "don't evict a thread with a run in flight" tracking mechanism this
    app's scale doesn't warrant.
    """

    def __init__(self, checkpointer, max_threads=MAX_LIVE_THREADS):
        super().__init__(max_threads, on_evict=self._evict)
        self._checkpointer = checkpointer

    def _evict(self, thread_id, _value):
        if self._checkpointer is not None:
            self._checkpointer.delete_thread(thread_id)

    def touch(self, thread_id):
        self._touch(thread_id)


# --- Per-thread checkpoint trimming (issue #81 / P9.2, measured defect) -----
#
# MEASURED: InMemorySaver.put() (langgraph/checkpoint/memory/__init__.py)
# writes a NEW checkpoint entry into self.storage and new blob entries into
# self.blobs on every single node completion, for every invoke - and NEVER
# prunes the old ones. Confirmed empirically (see this issue's own report):
# ~134KB/invoke with a 50KB image, 8MB after 10 invokes with a 400KB image,
# all on ONE thread. Neither MAX_MESSAGES_PER_THREAD (state.py, bounds only
# the `messages` key) nor MAX_LIVE_THREADS above (bounds only the number of
# DIFFERENT threads) touches this - a single thread invoked repeatedly grows
# without either of those caps ever engaging.
#
# THE FIX: after every COMPLETED run on a thread, trim that thread's stored
# history down to just its newest checkpoint (per checkpoint_ns) - the only
# one get_state()/a future invoke can ever need. InMemorySaver has NO public
# API for this (delete_thread removes a thread entirely, not selectively) -
# this function deliberately reaches into InMemorySaver's two internal dicts
# (`storage`, `writes`, `blobs`; see langgraph.checkpoint.memory.InMemorySaver
# for their documented shapes) because there is no supported alternative.
# That is a real coupling to memory-checkpointer internals, so
# test_checkpointing.py pins the assumption: a test asserts
# InMemorySaver instances expose exactly these three attributes in exactly
# this shape, and FAILS LOUDLY (not silently no-ops) if a langgraph upgrade
# changes it, rather than this function silently stopping to trim anything.
#
# EMPIRICALLY VERIFIED (see this issue's report): after trimming,
# checkpoint/write/blob counts for the trimmed thread stay FLAT across
# repeated invokes (not growing), graph.get_state() still returns the
# accumulated `messages` list and the latest run's scalar keys, and a
# further invoke on the SAME thread still works and keeps accumulating.
#
# AUDITED FOR PAUSED RUNS (issue #83 / P9.4 - this block previously said
# a future interrupt "must re-examine every call site here"; this is that
# re-examination, and the answer is measured, not assumed).
#
# EMPIRICALLY VERIFIED on langgraph 1.2.10: trimming a thread that is
# PAUSED on an interrupt does NOT break the resume. The pending interrupt's
# write is stored against the thread's NEWEST checkpoint - which is exactly
# the one this function keeps - and only the writes belonging to the OLDER,
# deleted checkpoints are dropped. After trimming (repeatedly, not once),
# get_state().interrupts still reports the pending question and
# graph.stream(Command(resume=...)) still completes the run. See
# tests/test_ask_before_speaking.py's
# test_resume_still_works_after_the_thread_is_trimmed, which is written to
# fail if a future langgraph version changes that.
#
# THE ONE THING THAT IS NOT SAFE DURING A PAUSE is a BARE
# graph.update_state(). Probe-confirmed: a state write while a task is
# pending clears the pending interrupt's entry from get_state().interrupts
# but leaves .next still naming the paused node - a ZOMBIE thread, neither
# running nor resumable, with the UI's two answer buttons pointing at
# nothing. This is not hypothetical; the cache-hit branch used to do
# exactly that.
#
# THE RULE, therefore, for every state write on a thread that MIGHT be
# paused: RESOLVE-THEN-WRITE. Decide what the user's action means as an
# answer to the pending question, resolve the pause to that answer, and
# only then write. Submitting another photo means an implicit RETAKE (the
# user moved on), and passing as_node=TTS_NODE to _update_thread_state
# resolves and writes in one go with no node executed - see the cache-hit
# branch in _run_pipeline_events for the full reasoning and the rejected
# alternative. The other two writes need no resolve: _record_turn only ever
# runs after a run has COMPLETED (a paused run returns before reaching it -
# see _run_pipeline_events' `paused` branch), and a follow-up on a paused
# thread is refused before it can write anything at all.
#
# THE CALL SITES. Most are reached via _update_thread_state, but NOT all -
# and NOT all are post-run either (this list has been rewritten twice as
# those two claims stopped being true: by issue #82 / P9.3, when it still
# said "only after a run reaches its final outcome", and by issue #93 /
# P9.12, when trimming stopped implying writing):
#   - _run_pipeline_events, after a photo run reaches its final outcome.
#     NOT reached when that run paused to ask a question. Trims WITHOUT
#     writing when that run degraded - see the next bullet.
#   - _record_turn's DEGRADED branch (issue #93 / P9.12), called DIRECTLY:
#     a degraded run records no turn, but it still created checkpoints, so
#     this thread still wants bounding. Reached from the photo and
#     follow-up call sites above, which is why both of them can now trim
#     with no state write behind it.
#   - _run_followup_events, after a question run reaches its final outcome.
#     Same degraded caveat as the photo path.
#   - _run_pipeline_events' CACHE-HIT branch, which runs at the START of a
#     request and trims before yielding. NOT a post-run call, and the one
#     place a thread can genuinely still be PAUSED when a write lands -
#     which is why that call passes as_node=TTS_NODE to resolve the pause
#     first (see RESOLVE-THEN-WRITE above). Trimming after that resolve is
#     safe like the others: there is no longer a pending checkpoint to
#     preserve.
#   - _run_resume_events (issue #83 / P9.4), called DIRECTLY (not via
#     _update_thread_state) on the retake path, which completes a run
#     without recording a turn and so has no state write to trim behind -
#     the same shape _record_turn's degraded branch above now has.
#   - AND, since issue #84 / P9.5, one MORE thing has to be bounded here:
#     dead SUBGRAPH NAMESPACES. The deep path is a child graph now
#     (clarif_eye.deep_path) and LangGraph checkpoints it under its own
#     namespace on the same thread - "deep_path:<task id>", with a FRESH
#     task id per run (measured: three deep runs on one thread left three
#     namespaces). Keeping the newest checkpoint PER NAMESPACE, which is all
#     this function used to do, therefore stopped bounding anything: a
#     thread that runs the deep path repeatedly accumulates one dead
#     namespace per run, which is the same unbounded growth this function
#     was written to close, reopened one level down. _drop_dead_subgraph_
#     namespaces below deletes them.
def _drop_dead_subgraph_namespaces(checkpointer, thread_storage, thread_id):
    """Delete every non-root checkpoint namespace on this thread except the
    most recent one (issue #84 / P9.5).

    THE ROOT NAMESPACE ("") IS NEVER TOUCHED here - it is the parent graph's
    own history, and its trimming is the caller's job.

    WHY THE MOST RECENT ONE IS KEPT rather than all of them dropped: a run
    that is PAUSED inside the child still needs its child checkpoint to
    resume from, and this function is reachable while a thread is paused
    (see the RESOLVE-THEN-WRITE block above - the cache-hit branch trims
    mid-pause). A pause always lives in the newest run, so keeping exactly
    one namespace keeps every resumable pause resumable while still bounding
    the total at one dead namespace instead of one per run.

    ORDERING: namespaces are compared by their newest checkpoint id, which
    LangGraph generates as a time-ordered UUID (v6) - the SAME assumption
    the per-namespace trim above already makes when it calls max() on
    checkpoint ids, and the same one InMemorySaver.get_tuple makes itself,
    now applied across namespaces rather than within one. Verified: ids from
    two different namespaces on one thread sort in creation order.
    """
    child_namespaces = [ns for ns in thread_storage if ns]
    if len(child_namespaces) < 2:
        return
    newest = max(child_namespaces, key=lambda ns: max(thread_storage[ns].keys()))
    for namespace in child_namespaces:
        if namespace == newest:
            continue
        del thread_storage[namespace]
        for key in [k for k in checkpointer.writes if k[0] == thread_id and k[1] == namespace]:
            del checkpointer.writes[key]
        for key in [k for k in checkpointer.blobs if k[0] == thread_id and k[1] == namespace]:
            del checkpointer.blobs[key]


def _trim_thread_to_latest_checkpoint(checkpointer, thread_id):
    """Delete every checkpoint/write/blob for `thread_id` except what its
    newest checkpoint (per checkpoint_ns) still references, then drop every
    dead subgraph namespace (see _drop_dead_subgraph_namespaces).

    No-op if `checkpointer` is None (an uncheckpointed graph - see
    build_graph's `checkpointer` param) or if `thread_id` has no stored
    checkpoints yet (nothing to trim). Never raises: this is memory
    housekeeping, not something a photo's own outcome should ever depend
    on - a trim that failed for any reason should leave the (larger, but
    still correct) untrimmed history in place rather than take down a run.
    """
    if checkpointer is None:
        return
    try:
        thread_storage = checkpointer.storage.get(thread_id)
        if not thread_storage:
            return
        for checkpoint_ns, ns_checkpoints in thread_storage.items():
            if not ns_checkpoints:
                continue
            # Checkpoint IDs sort lexicographically by creation order - the
            # SAME assumption InMemorySaver.get_tuple() itself relies on
            # (`checkpoint_id = max(checkpoints.keys())` for "give me the
            # latest checkpoint when none is specified"), not a new one
            # introduced here.
            latest_id = max(ns_checkpoints.keys())
            checkpoint_bytes, metadata_bytes, _old_parent = ns_checkpoints[latest_id]
            checkpoint = checkpointer.serde.loads_typed(checkpoint_bytes)
            live_versions = checkpoint.get("channel_versions", {})

            for old_id in [cid for cid in ns_checkpoints if cid != latest_id]:
                del ns_checkpoints[old_id]
                checkpointer.writes.pop((thread_id, checkpoint_ns, old_id), None)
            # The retained checkpoint's parent no longer exists - clear the
            # link rather than leave it dangling at a checkpoint_id that
            # was just deleted.
            ns_checkpoints[latest_id] = (checkpoint_bytes, metadata_bytes, None)

            for blob_key in [
                k for k in checkpointer.blobs if k[0] == thread_id and k[1] == checkpoint_ns
            ]:
                _t, _ns, channel, version = blob_key
                if live_versions.get(channel) != version:
                    del checkpointer.blobs[blob_key]

        _drop_dead_subgraph_namespaces(checkpointer, thread_storage, thread_id)
    except Exception:
        # Housekeeping only - see docstring. Never swallows
        # KeyboardInterrupt/SystemExit (BaseException, not Exception), the
        # same contract every node in this pipeline already follows.
        pass


def thread_configurable(resources, thread_id, session_id=None):
    """The ONE chokepoint for every thread-scoped call into
    resources.graph (issue #81 / P9.2). Touches resources.thread_registry
    (if configured) so the live-thread LRU bound (MAX_LIVE_THREADS above)
    stays accurate, and returns the config["configurable"] entries a
    thread-scoped call needs - {"thread_id": thread_id} and, since issue
    #86 / P9.7, {"session_id": session_id}.

    Returns {} when `thread_id` is None, so callers can unconditionally
    `configurable.update(thread_configurable(resources, thread_id))`
    without an extra branch, and the uncheckpointed/no-thread_id case
    (every existing test/caller) is unaffected.

    ALSO returns {} whenever `resources.thread_registry` is None (deep-
    review BLOCKER fix, issue #81 / P9.2) - see AppResources's own
    docstring for the pairing invariant this enforces: thread_registry is
    only ever set by build_resources() alongside a REAL checkpointed
    graph (see that function's CHECKPOINTING comment), so "no registry"
    is the reliable signal that resources.graph has no checkpointer, or
    is a test double with no update_state at all. Passing thread_id
    through to graph.stream() in that situation is exactly what produced
    the bug this fixes: LangGraph raises ValueError("No checkpointer
    set") the instant config["configurable"]["thread_id"] is present on
    an uncheckpointed graph, an exception this function's own caller
    (_run_pipeline_events) is bound by the module docstring to never let
    escape. A caller that WANTS thread-scoped behavior must therefore
    pair a checkpointed graph with a thread_registry - build_resources()
    already does this correctly; nothing else in this codebase should
    construct one without the other.

    `session_id` (issue #86 / P9.7) is added to the returned dict
    INDEPENDENTLY of the thread_id/thread_registry guard above - it has no
    pairing invariant with `graph` to enforce, because a Store is not a
    checkpointer: an uncheckpointed/no-registry graph can still be compiled
    WITH a store (see clarif_eye.graph.build_graph's `store` param), and a
    node reading a preference with no store configured simply sees None
    (clarif_eye.preferences.verbosity_for_config never raises). Threaded
    through THIS chokepoint - the SAME ONE thread_id already uses - rather
    than a second one, for the reason this function's own docstring already
    states about future call sites: one gate that everything goes through
    cannot silently stop covering a call site the way two could.

    FUTURE CALL SITES MUST GO THROUGH THIS, not resources.thread_registry
    directly: #82 (follow-ups) and #83 (interrupts) will add more places
    that invoke or otherwise touch a thread-scoped graph call, and a
    registry that only some of them remember to touch would silently stop
    bounding the live-thread count for the call sites that forgot.
    """
    configurable = {}
    if thread_id is not None and resources.thread_registry is not None:
        resources.thread_registry.touch(thread_id)
        configurable["thread_id"] = thread_id
    if session_id is not None:
        configurable["session_id"] = session_id
    return configurable


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
    # The text-only API route's own cache (issue #84 / P9.5). A SEPARATE
    # cache, not a second use of image_cache: that one is keyed on image
    # content and its entries carry an audio path plus the ocr/scene a thread
    # needs, none of which exist here. Defaulted like image_cache so every
    # test that builds an AppResources directly keeps working untouched.
    document_cache: object = field(default_factory=DocumentTextCache)
    # None by default (issue #81 / P9.2): every existing test that builds
    # an AppResources directly (test_ui.py's FakeGraph-based tests) never
    # sets these, so they keep running an uncheckpointed graph with no
    # thread_id, exactly today's behavior - see _run_pipeline_events, which
    # only touches thread_registry / adds thread_id to config when a
    # caller actually passes one in. build_resources() (the live app) sets
    # both.
    #
    # PAIRING INVARIANT (deep-review BLOCKER fix, issue #81 / P9.2):
    # thread_registry must be non-None IF AND ONLY IF `graph` is a real
    # checkpointed graph (compiled via build_graph(checkpointer=...)) that
    # actually supports thread-scoped calls (graph.update_state, and
    # thread_id in config["configurable"] without raising). thread_
    # configurable() (below) uses "thread_registry is not None" as the
    # SOLE signal that it's safe to thread thread_id through to the graph
    # at all - constructing an AppResources with a thread_registry but an
    # uncheckpointed graph (or vice versa) would silently defeat that
    # guard and reopen the exact bug it exists to prevent. build_resources()
    # is the only place in this codebase that should ever set both
    # together; every test that wants a thread-scoped graph must do the
    # same (see tests/test_ui.py's checkpointed-thread tests).
    thread_registry: object = None
    # None by default (issue #86 / P9.7), same "every existing test unaffected"
    # reasoning as thread_registry above: an AppResources built directly with
    # no `store` keeps handle_ask_staged's preference-command branch a no-op
    # write (clarif_eye.preferences.set_verbosity degrades to nothing on a
    # None store) and every node's read degrades to "no preference on file" -
    # exactly today's behavior. build_resources() (the live app) sets a real
    # langgraph.store.memory.InMemoryStore, one per process, alongside the
    # checkpointer - see that function's own comment for the honest,
    # in-process-only limits it shares with the checkpointer. UNLIKE
    # thread_registry there is no pairing invariant with `graph` to enforce:
    # a store is independent of whether `graph` is checkpointed at all (see
    # clarif_eye.graph.build_graph's `store` param docstring), so this can be
    # set (or not) without reference to thread_registry/checkpointer.
    store: object = None


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
    production - this app deploys on Render (see render.yaml), not
    Hugging Face Spaces:
      - InMemorySaver keeps every checkpoint in this PROCESS's own memory -
        nothing is written to disk or any external store. A process
        restart (a redeploy, a crash, Render restarting the container)
        loses every thread's history. This is fine for this issue's scope
        (state surviving a run within one session) and is never described
        to the user as more durable than that - no user-facing text in
        this app claims otherwise.
      - Render's free tier spins the service down after ~15 minutes of no
        traffic (see README.md / docs/ACCESSIBILITY.md's "cold-start
        loading page" note); waking it back up starts a NEW process, so
        the same loss happens on every cold start a visitor's return
        happens to trigger.
      - See ThreadRegistry / MAX_LIVE_THREADS and
        _trim_thread_to_latest_checkpoint above for how both the number of
        live threads AND each thread's own unpruned checkpoint history
        (measured defect - see that function's comment) are bounded while
        the process IS running.
    Every existing caller of build_graph() with no arguments, and every
    test in this repo that builds an AppResources directly instead of via
    this function, is unaffected: build_graph()'s `checkpointer` param
    defaults to None, so an uncheckpointed graph needs no thread_id at all
    - only THIS function's live graph does.

    CROSS-THREAD PREFERENCE STORE (issue #86 / P9.7): also compiles WITH a
    fresh langgraph.store.memory.InMemoryStore, one per process, the same
    "construct once at startup" discipline as the checkpointer above.
    SAME HONEST LIMITS AS THE CHECKPOINTER, stated here for the same
    reason: everything lives in THIS PROCESS's memory only, nothing is
    written to disk, and a restart (redeploy, crash, a free-tier cold
    start) loses every session's preference exactly as it already loses
    every thread's history. UNLIKE the checkpointer, this store is NEVER
    bounded by ThreadRegistry/MAX_LIVE_THREADS - it holds at most one tiny
    dict per session_id ({"verbosity": "short"|"detailed"}, see
    clarif_eye.preferences), so an unbounded number of sessions would cost
    kilobytes, not the base64 photo bytes a checkpointer thread can carry.
    No eviction mechanism exists for it today; that is an accepted,
    honestly-stated gap for a demo app's scale, not an oversight.
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
    store = InMemoryStore()

    return AppResources(
        graph=build_graph(checkpointer=checkpointer, store=store),
        client=client,
        client_error=client_error,
        tts_providers=tts_providers,
        searcher=searcher,
        research_client=research_client,
        thread_registry=ThreadRegistry(checkpointer),
        store=store,
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


def _degraded_outcome(text):
    """The "outcome" payload for a guard/early-failure exit that never
    attempted TTS (no image, no client, an exception escaping the graph,
    ...) - always STATUS_DEGRADED, never is_chain_exhausted()-derived (see
    _outcome_for's ISSUE #88 / P9.9 note for why). One helper instead of
    the same (None, text, STATUS_DEGRADED) literal repeated at every such
    call site, so the invariant lives in one place.
    """
    return (None, text, STATUS_DEGRADED)


def _outcome_for(final_output, audio_path):
    """Map a completed run's (final_output, audio_path) to the
    (audio_path_or_None, text, status) tuple every "outcome" event carries.

    THE THREE OUTCOMES, told apart STRUCTURALLY - see this module's
    top-level docstring. Shared by _run_pipeline_events, _run_followup_events
    and _run_resume_events (issue #82 / P9.3, issue #83 / P9.4) so a spoken
    ANSWER degrades through exactly the same three branches a spoken
    DESCRIPTION does; two copies would be two places for the
    audio-unavailable wording and the is_chain_exhausted() ordering to drift
    apart.

    ISSUE #88 / P9.9: `status` is computed HERE, at the one point where
    is_chain_exhausted() is guaranteed to describe THIS run - immediately
    after the TTS attempt that produced `audio_path` - and carried out in
    the outcome tuple from then on. Callers (_stage_events) use this status
    as-is and never call is_chain_exhausted() themselves, which is what
    used to let a later, unrelated run's stale module-global TTS state leak
    into an early-failure outcome that never attempted TTS at all (that
    None/2-tuple case never reaches this function - see the "outcome"
    literals in _run_pipeline_events/_run_followup_events/_run_resume_events,
    which carry STATUS_DEGRADED directly instead).
    """
    if audio_path:
        return (audio_path, final_output, status_for_result(audio_path, False))
    chain_exhausted = is_chain_exhausted()
    if chain_exhausted:
        text = f"{final_output} {AUDIO_UNAVAILABLE_NOTE}" if final_output else AUDIO_UNAVAILABLE_NOTE
        return (None, text, status_for_result(None, True))
    return (None, final_output or UNEXPECTED_ERROR_MESSAGE, status_for_result(None, False))


def _update_thread_state(graph, config, thread_id, update, as_node=None):
    """Write `update` onto `thread_id`'s checkpoint, then bound that
    thread's checkpoint history.

    `as_node` (issue #83 / P9.4) names the node LangGraph should record the
    write as having come from. Omitted, LangGraph INFERS it from the last
    node that updated the state - which is what every caller relied on
    before this parameter existed, and which is unambiguous on a thread
    whose last run completed (tts is the only node every path ends on).
    It stops being unambiguous, and stops being safe, on a thread that is
    PAUSED on an interrupt - see the caller in _run_pipeline_events'
    cache-hit branch for the zombie that produced, and RESOLVE-THEN-WRITE
    below for the rule.

    NEVER RAISES (deep-review BLOCKER fix, issue #81 / P9.2): this is
    bookkeeping - state for a LATER run - not part of THIS run's
    deliverable, which was already computed before the caller got here. A
    failure here (a stale/evicted thread_id whose checkpoint ThreadRegistry
    deleted mid-run - see that class's docstring for why that is tolerated -
    or any other unforeseen edge in update_state/trim) must never cost the
    user the answer that already exists. thread_configurable() is the FIRST
    line of defence (it declines to thread a thread_id through to an
    uncheckpointed graph at all); this is the second, catching whatever that
    guard doesn't.

    Works on a thread with NO prior checkpoint (verified empirically on
    langgraph 1.2.10 for issue #82's cache-hit fix): update_state creates
    the thread's first checkpoint and needs no `as_node` to do it, which is
    exactly the case a second visitor's cache hit produces. Verified again
    for this issue with an explicit as_node="tts" on a thread that has
    never run: it creates the first checkpoint just the same.

    THE ONE THING THE NEVER-RAISE GUARD BELOW CAN HIDE is a WRONG `as_node`.
    LangGraph validates the name against the compiled node set and raises
    langgraph.errors.InvalidUpdateError("Node <name> does not exist"), which
    the blanket except turns into "this write silently did nothing" - and
    for the cache-hit caller that write is what stops a follow-up answering
    about a different photo than the user was just told about (issue #82's
    blocker). Two things keep that honest, and no third mechanism is added
    here because neither production behaviour nor the never-raise contract
    should change for a programming error:
      - the name is not a literal at either end - it is
        clarif_eye.graph.TTS_NODE, so a node rename is one edit;
      - a wrong name is LOUD IN THE SUITE, verified by mutation, not
        assumed: pointing as_node at a node that does not exist turns THREE
        tests red - test_ask_before_speaking.py's paused-cache-hit test and
        both of test_followup.py's cache-hit-then-follow-up tests, which are
        the very tests #82's blocker was closed with. tests/test_graph.py
        additionally pins TTS_NODE against the compiled graph's node set, so
        a half-finished rename fails by name rather than through those three
        indirect assertions.
    """
    try:
        if as_node is None:
            graph.update_state(config, update)
        else:
            graph.update_state(config, update, as_node=as_node)
        _trim_thread_to_latest_checkpoint(getattr(graph, "checkpointer", None), thread_id)
    except Exception:
        pass


def _record_turn(graph, config, thread_id, messages, degraded):
    """Record `messages` against `thread_id` at the conversation boundary -
    unless the run DEGRADED, in which case nothing is recorded.

    Called only after a run has fully COMPLETED and produced something worth
    remembering - see _run_pipeline_events's "CONVERSATION-BOUNDARY
    RECORDING" docstring section for the full reasoning, and
    clarif_eye.graph.tts_node's docstring for why this lives at the boundary
    rather than inside a node.

    `degraded` IS THE ANSWER TO ISSUE #93 / P9.12, and it comes from state:
    clarif_eye.state.ClarifEyeState.output_degraded, written by whichever
    node produced final_output. Read that key's comment for why the signal
    could not be derived here. In short: every failure in this pipeline is
    SPOKEN, so a degraded run arrives at this boundary with real audio and
    real text and is indistinguishable from a success by anything except
    the wording - and matching the wording is exactly what this codebase
    does not do (D15).

    SKIPPED, NOT MARKED - the decision, and why, since the alternative was
    real. Marking (an additional_kwargs flag on the recorded message, or a
    parallel list) would keep the history complete and let a reader decide
    what to do with a failed turn. It was rejected because it is FAIL-OPEN:
    every future consumer - #92's verification rework, the ask-before-
    speaking flows, anything that builds a model prompt out of `messages` -
    would have to remember to filter, and the one that forgets reads "every
    model was busy" back to a blind user as if it were what their photo
    said. Skipping is FAIL-SAFE: a message that was never written cannot be
    replayed by a consumer that has not been written yet. It also matches
    every precedent already in this module - a paused run records nothing, a
    retake records nothing, a failed run is not cached (issue #75's "a
    failure must never be replayed as that photo's answer"). What is lost is
    real and worth naming: the thread forgets that a degraded turn happened
    at all. That is acceptable because history exists to GROUND later
    answers, and a failure grounds nothing - the user heard it live, which
    is where it mattered.

    BOTH SIDES OF A FOLLOW-UP GO OR STAY TOGETHER, which is why this takes
    the whole `messages` list rather than filtering it. Recording the user's
    question alone would leave a question in history with no answer beneath
    it and the NEXT run's real answer sitting directly under it - a reader
    pairing them off would attach the wrong answer to it. The user did
    genuinely ask, but a turn that produced no answer is better left out
    whole than left half-written.

    THE PRODUCT RULING THIS FORCES, stated so nobody has to rediscover it as
    a bug: a follow-up asked after a photo run that DEGRADED degrades in
    turn - it does not quietly answer about an earlier, good photo on the
    same thread. That is deliberate. The thread's stored ocr/scene describe
    the LATEST photo, and answering about an older one would mean
    confidently describing something other than what the user is holding,
    to someone who cannot see the difference.

    THE TRIM STILL RUNS on a skipped turn: a degraded run still created
    checkpoints, and bounding this thread's checkpoint history is
    housekeeping that has nothing to do with what was said (see
    _trim_thread_to_latest_checkpoint's measured defect). Same call
    _run_resume_events's retake branch already makes for the same reason.

    THE RESIDUAL GAP, NAMED HONESTLY: this flag says "the node that wrote
    final_output degraded". It does NOT catch a run that answered well from
    partly-degraded inputs - e.g. research came back empty and `analysis`
    still wrote a good description from ocr + scene alone. That is recorded,
    and deliberately so: it IS a real description of the photo. The line
    drawn here is "was the spoken text an answer, or an explanation of why
    there is none", not "was every step of the pipeline healthy".
    """
    if degraded:
        _trim_thread_to_latest_checkpoint(getattr(graph, "checkpointer", None), thread_id)
        return
    _update_thread_state(graph, config, thread_id, {"messages": messages})


def _narrate_stream(graph, state, config, result):
    """Drive one graph run to completion, merging each node's update into
    `result` (in place, so the caller sees the finished state) and yielding
    ("status", phrase) for each completed node that HAS a successor with a
    phrase.

    Shared by _run_pipeline_events (photo runs) and _run_followup_events
    (issue #82 / P9.3): both need byte-identical narration behaviour, and
    duplicating this loop would be one more place for the two to drift apart
    - the follow-up path would silently stop narrating, or stop tolerating a
    None update, the moment one copy was edited and the other wasn't.

    A None `update` is SKIPPED, not merged: a node returning a bare
    Command(goto=...) with no state update - clarif_eye.graph.entry_node
    does exactly this - streams as {"entry": None}, and dict.update(None)
    raises TypeError. Verified empirically on langgraph 1.2.10; see
    entry_node's docstring.

    A node with no entry in _NODE_PHRASE announces NOTHING (see that dict's
    comment for which two nodes those are and why) - looked up with .get()
    so adding a node to the graph can never turn into a KeyError crash mid
    run, which for this app would mean losing an answer that was already
    half-computed.

    NESTED STREAMING (issue #84 / P9.5): the stream is opened with
    `subgraphs=True`, so each item is a (namespace, chunk) PAIR rather than
    a bare chunk. That is what makes the deep path's own nodes - `research`
    and `analysis`, now inside a child graph (clarif_eye.deep_path) -
    visible from up here at all. Without it, the whole deep path arrives as
    one opaque `deep_path` completion and a blind user hears nothing between
    "photo received" and "turning it into speech": a ~30-second silence
    where three announcements used to be.

    The namespace is () for the parent's own nodes and
    ("deep_path:<task id>",) for the child's. Node updates are narrated the
    SAME WAY whichever it is - a node name is unambiguous across the two
    graphs, and clarif_eye.graph.next_node_after knows both graphs'
    topology, which is what keeps the spoken sequence byte-identical to what
    it was before the extraction.

    THE THIRD EVENT KIND (issue #83 / P9.4): a chunk keyed
    INTERRUPT_CHUNK_KEY means the run PAUSED to ask the user something -
    clarif_eye.graph.verify_numbers_node is the only thing in this graph
    that does it. Its value is a TUPLE of Interrupt objects, not a state
    update, so it must be intercepted BEFORE the result.update() above (a
    dict.update() with a tuple raises TypeError - this is not a
    hypothetical: this loop would have crashed on the first real pause).
    It yields ("interrupt", spoken_question) and the stream ends there;
    LangGraph produces no further chunks until the run is resumed. The
    caller is responsible for not treating a paused run as a finished one -
    see _run_pipeline_events.

    `state` is whatever graph.stream accepts as input: a full initial state,
    a partial delta (a follow-up), or a langgraph Command(resume=...) (a
    resumed run). This function does not care which - it only drives and
    narrates.
    """
    for namespace, chunk in graph.stream(
        state, config=config, stream_mode="updates", subgraphs=True
    ):
        for node_name, update in chunk.items():
            if node_name == INTERRUPT_CHUNK_KEY:
                # A pause raised INSIDE the child arrives TWICE - once
                # namespaced to the child, then again at the parent level
                # with the identical payload (verified empirically on
                # langgraph 1.2.10). Skipping the namespaced copy makes this
                # generator emit exactly ONE ("interrupt", ...) event, and
                # makes this branch behave identically whether the pause came
                # from a child node or a parent one.
                #
                # WHAT THIS IS AND IS NOT RESPONSIBLE FOR, stated because an
                # earlier version of this comment took credit it had not
                # earned: it is NOT what stops the user hearing the question
                # twice. _stage_events collects the question into a local and
                # yields it once AFTER the loop, so the spoken output would
                # be single even without this guard. What this guard actually
                # protects is every OTHER consumer of this generator - it is
                # shared by the photo, follow-up and resume paths, and
                # _run_pipeline_events counts an interrupt event to decide a
                # run paused. A duplicate is not visibly wrong today, which is
                # exactly why it needs a test rather than an assumption: see
                # tests/test_deep_path_subgraph.py's one-event test.
                if namespace:
                    continue
                interrupts = update or ()
                payload = interrupts[0].value if interrupts else {}
                yield "interrupt", _interrupt_question(payload)
                continue
            if update is not None:
                result.update(update)
            # next_node_after is the single source of truth for this
            # graph's topology (clarif_eye.graph, right next to
            # build_graph()'s edges) - this module only supplies the
            # WORDING for whatever node it names.
            next_node = next_node_after(node_name, result)
            if next_node is None:
                continue
            phrase = _NODE_PHRASE.get(next_node)
            if phrase is not None:
                yield "status", phrase


def _run_pipeline_events(image, resources, pipeline_budget_seconds, thread_id=None, session_id=None):
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
    stream_mode="updates", subgraphs=True); there is no invoke()-only
    fallback, so an incompatible double fails loudly (AttributeError/
    TypeError) instead of silently losing all narration.

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
    browser session and threads it through here; when it's not None,
    thread_configurable(resources, thread_id) is the ONE chokepoint used
    to (a) register it with resources.thread_registry, if one is
    configured, so a bounded live-thread count is maintained (see
    ThreadRegistry above), and (b) add it to config["configurable"], which
    is what a checkpointed graph requires to persist/restore state across
    calls (verified empirically - see build_graph's docstring).

    CONVERSATION-BOUNDARY RECORDING (issue #81 / P9.2): tts_node itself no
    longer appends to `messages` (an earlier version did - see
    clarif_eye.graph.tts_node's docstring for why that moved). Once a run
    completes with a real thread_id AND a non-empty final_output, this
    function records the turn exactly ONCE via
    graph.update_state(config, {"messages": [...]}) - verified empirically
    to route through state.py's reducer (accumulates, and coerces the
    plain dict into a real message object) without needing `as_node`
    (LangGraph infers it from the last node that updated the state, which
    is unambiguous here - tts is the only node every path ends on).
    Immediately after, _trim_thread_to_latest_checkpoint bounds that
    thread's own checkpoint history (see that function's comment for the
    measured defect this closes). Neither happens on a cache hit (nothing
    ran) or an early/exception failure (no final_output was produced) -
    same "no bleed of a bad run into cached/replayed state" discipline the
    image cache above already follows. NOR ON A RUN THAT DEGRADED INSIDE
    THE PIPELINE (issue #93 / P9.12): that run COMPLETED and produced real
    spoken audio, but its text explains why there is no description rather
    than being one, so _record_turn skips it (the trim still runs) and the
    image cache is not written either - see _record_turn's docstring for the
    skip-not-mark decision, and clarif_eye.state.ClarifEyeState.output_degraded
    for where the signal comes from.

    Wrapped in try/except - see the call site's own comment for why a
    recording failure must never cost the user the answer that was already
    computed.

    ACCUMULATION IS BEST-EFFORT UNDER CONCURRENT SUBMITS ON ONE
    thread_id: two overlapping requests on the SAME thread_id (e.g. a
    double-submit race from one browser tab) each read, then separately
    write, this thread's state with no lock across the two - LangGraph's
    update_state and this function's own trim call are each individually
    consistent, but nothing coordinates the two full boundary-recording
    sequences against each other. The result is ordinary last-writer-wins,
    not a guaranteed 2-entries-for-2-submits outcome - deliberately NOT
    locked, since a global lock across a request as slow as this pipeline
    (up to DEFAULT_PIPELINE_BUDGET_SECONDS) would serialize every
    concurrent visitor, not just the rare double-submit case. Acceptable
    for a demo app; the single-threaded-per-tab tests in this file cannot
    exercise the race itself.
    """
    if image is None:
        yield "outcome", _degraded_outcome(NO_IMAGE_MESSAGE)
        return

    if resources.client is None:
        yield "outcome", _degraded_outcome(resources.client_error or CONFIG_ERROR_MESSAGE)
        return

    try:
        image_data = _encode_image(image)
    except Exception:
        yield "outcome", _degraded_outcome(UNREADABLE_IMAGE_MESSAGE)
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
        cached_audio_path, cached_text, cached_ocr, cached_scene = cached
        # A cached audio path could have been deleted since it was stored
        # (tts.py's _prune_old_files DOES delete old mp3s once more than
        # MAX_KEPT_FILES exist; a temp cleaner or disk pressure could too).
        # Returning it anyway would have the UI announce "ready" over
        # silence - the worst failure mode for a user who cannot see the
        # screen. Treat a missing file as a miss: drop the stale entry and
        # fall through to run the pipeline for real.
        if not cached_audio_path or os.path.exists(cached_audio_path):
            # THE THREAD MUST DESCRIBE THE PHOTO THE USER WAS JUST TOLD
            # ABOUT (deep-review BLOCKER, issue #82 / P9.3 - see
            # ImageResultCache's docstring for the two proven scenarios).
            # The graph did not run, so nothing else will write these keys,
            # and a follow-up answers from the thread. `question` is reset
            # to None for exactly the reason make_initial_state resets it on
            # a real photo run: a question left over from the previous turn
            # must not divert the next one.
            #
            # image_data IS BLANKED, DELIBERATELY: this cache stores text
            # derived from the photo, never the base64 photo itself (see
            # ImageResultCache's privacy note), so there is no real
            # image_data to write here. Leaving the key alone would strand
            # a DIFFERENT photo's base64 in the checkpoint right next to
            # this photo's ocr/scene - probe-confirmed, and invisible to
            # everyone until some future consumer reads the stored image
            # and quietly gets the wrong one (#83's ask-first flows are the
            # obvious candidate). "" is the loud answer: not available.
            #
            # THE DESCRIPTION IS RECORDED AS A TURN in the same write, and
            # it needs no degradation check of its own (issue #93 / P9.12):
            # only real, successful, audio-bearing, NON-DEGRADED outcomes are
            # ever cached (see ImageResultCache and the put site below -
            # failures, text-only degradations and, since #93, any run whose
            # node degraded are deliberately not), so cached_text is exactly
            # what the user just heard AND a genuine description. That is the
            # invariant this branch depends on: a hit has no state left to
            # read a flag from, only this entry. A thread carrying
            # ocr/scene with an empty history would be inconsistent for the
            # consumers reading that history (#93, #83, #84). One combined
            # update, not two: `messages` goes through state.py's reducer
            # and appends, while the rest replace, so a single call is both
            # atomic and one checkpoint write instead of two.
            # RESOLVE-THEN-WRITE (issue #83 / P9.4) - THE rule for any state
            # write on a thread that might be paused, and the fix for a real
            # zombie this branch used to create.
            #
            # A plain graph.update_state() on a thread paused at
            # `verify_numbers` clears the pending interrupt's own write but
            # leaves .next still naming the paused node - probe-confirmed.
            # The thread was then neither running nor resumable: nothing
            # would ever answer the question, and the two answer buttons
            # were still on screen pointing at it.
            #
            # Submitting another photo IS an answer, though - an IMPLICIT
            # RETAKE. The user moved on, which is exactly what the retake
            # button means. So the pause is resolved as a retake rather than
            # being asked about again, and `verification_hold` is cleared
            # below with the same write, for the same reason `question` is:
            # a draft held back for a photo the user has left behind must
            # never divert the next run.
            #
            # RESOLVED WITH as_node=TTS_NODE, NOT by streaming
            # Command(resume=RESUME_RETAKE). Both end identically
            # (.next == (), no pending interrupt - both probed), but the
            # resume path RUNS the graph's remaining nodes, which means
            # verify_numbers plus a full tts synthesis of a retake
            # confirmation nobody will hear - a real network round trip and
            # a stray mp3, spent to reach a state the user is about to
            # leave anyway. as_node=TTS_NODE reaches the same end state with
            # NO node executed at all, and folds into the single write this
            # branch was already making. It is also what this call was
            # already doing implicitly: with no as_node, LangGraph infers
            # the last node to update the state, which on a completed
            # thread is tts. Naming it makes that true on a paused thread
            # too, instead of quietly not being.
            #
            # THE NAME COMES FROM clarif_eye.graph.TTS_NODE, never a literal
            # here: LangGraph validates as_node against the compiled node
            # set and raises InvalidUpdateError("Node <name> does not
            # exist") if it misses, which _update_thread_state's never-raise
            # guard would turn into "this whole write silently did nothing"
            # - i.e. the #82 wrong-photo blocker, back. See that function's
            # docstring for the three tests that catch it anyway.
            if thread_id is not None:
                configurable = dict(thread_configurable(resources, thread_id))
                if configurable:
                    update = {
                        "image_data": "",
                        "ocr_output": cached_ocr,
                        "scene_context": cached_scene,
                        "question": None,
                        "verification_hold": None,
                    }
                    if cached_text:
                        update["messages"] = [{"role": "assistant", "content": cached_text}]
                    _update_thread_state(
                        resources.graph,
                        {"configurable": configurable},
                        thread_id,
                        update,
                        as_node=TTS_NODE,
                    )
            yield "outcome", (cached_audio_path or None, cached_text, STATUS_SUCCESS_AUDIO)
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
        configurable.update(thread_configurable(resources, thread_id, session_id=session_id))
        config = {"configurable": configurable}
        graph = resources.graph
        result = dict(state)
        # DRAINED, not `yield from` (issue #83 / P9.4): this function has to
        # SEE an ("interrupt", ...) event go past, not just forward it. A
        # paused run has produced no outcome yet - final_output currently
        # holds analysis's safe "could not be verified" script and there is
        # no audio - so everything below (recording the turn, caching the
        # result, mapping an outcome) would be recording a non-answer as
        # this photo's answer. It gets none of that; the run resumes, or it
        # doesn't.
        paused = False
        for kind, payload in _narrate_stream(graph, state, config, result):
            if kind == "interrupt":
                paused = True
            yield kind, payload
        if paused:
            return
    except LadderExhaustedError as exc:
        # Every node already catches and degrades this internally (see
        # vision.py/synth.py/analysis.py); this branch only matters if the
        # whole pipeline fails before a node can degrade (issue #18 / P6.2
        # scope item 4, e.g. a client-construction failure a node did not
        # catch). Uses the SAME category mapping the nodes use, rather than
        # collapsing into the generic UNEXPECTED_ERROR_MESSAGE below. Not
        # cached (issue #75): a quota/API failure must never be replayed
        # to the next visitor as if it were that photo's own answer.
        yield "outcome", _degraded_outcome(message_for_ladder_exhausted(exc))
        return
    except OpenRouterError as exc:
        yield "outcome", _degraded_outcome(message_for_terminal_error(exc))
        return
    except Exception:
        yield "outcome", _degraded_outcome(UNEXPECTED_ERROR_MESSAGE)
        return

    final_output = (result.get("final_output") or "").strip()
    audio_path = result.get("audio_file_path") or ""

    # Conversation-boundary recording (issue #81 / P9.2) - see this
    # function's own docstring "CONVERSATION-BOUNDARY RECORDING". Only for
    # a real thread_id (an uncheckpointed resources.graph has no
    # update_state-worthy thread to record against) and only when the run
    # actually produced something worth remembering.
    #
    # WRAPPED IN try/except (deep-review BLOCKER fix, issue #81 / P9.2):
    # this block is bookkeeping - accumulating the conversation history for
    # a LATER run - not part of THIS run's actual deliverable. `final_output`/
    # `audio_path` above were already computed by the time execution reaches
    # here; a failure recording the turn (a stale/evicted thread_id whose
    # checkpoint was deleted mid-run by ThreadRegistry - see that class's
    # docstring for why that's tolerated - or any other unforeseen edge in
    # update_state/trim) must never cost the user the answer that already
    # exists. thread_configurable() above is the FIRST line of defense
    # (skips thread_id entirely for an uncheckpointed graph/registry-less
    # AppResources, the exact case that used to raise ValueError/
    # AttributeError straight through this generator); this try/except is
    # the second, catching whatever thread_configurable's guard doesn't -
    # any future graph/thread_registry combination this module hasn't
    # anticipated degrades to "the turn wasn't recorded" instead of "the
    # user got no answer at all".
    # `degraded` (issue #93 / P9.12) comes from the RUN's own merged state,
    # written by whichever node produced final_output - see
    # clarif_eye.state.ClarifEyeState.output_degraded and _record_turn.
    # .get() with a falsy default, not bracket access: a run that somehow
    # reached here without any node claiming a degradation records exactly
    # as it did before this flag existed.
    degraded = bool(result.get("output_degraded"))
    if thread_id is not None and final_output:
        _record_turn(
            graph, config, thread_id, [{"role": "assistant", "content": final_output}], degraded
        )

    outcome = _outcome_for(final_output, audio_path)

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
    #
    # ocr_output/scene_context are stored ALONGSIDE the spoken result so a
    # later hit can put them back on the hitting caller's thread - see
    # ImageResultCache's docstring for the two ways a hit used to leave the
    # thread describing a different photo than the user had just heard about.
    #
    # NOR IS A DEGRADED ONE (issue #93 / P9.12) - and this closes a real
    # hole, not a hypothetical one. Every failure inside the pipeline is
    # SPOKEN, so a photo whose vision call failed still reaches here with a
    # working audio path and was, until now, cached: "the photo could not be
    # read" was stored as that image's result and replayed to the next
    # visitor, which is precisely what this cache's own docstring says never
    # happens ("only successful results are ever stored here"). It also fed
    # back into conversation memory - the cache-hit branch above records the
    # cached text as the hitting thread's turn, and that branch has no node
    # left to ask, only this entry. Keeping the degraded result out of the
    # cache is what keeps that branch honest without smuggling the flag
    # through the cache entry. The cost is a repeat of a failed photo paying
    # its quota again, which is the right trade: the failure may well have
    # been transient, and replaying it forever cannot be.
    if audio_path and not degraded:
        resources.image_cache.put(
            cache_key,
            (audio_path, outcome[1], result.get("ocr_output") or "", result.get("scene_context") or ""),
        )
    yield "outcome", outcome


def _handle_preference_command(verbosity, resources, session_id):
    """Store `verbosity` for `session_id` and speak a confirmation - NO
    graph run, NO model call (issue #86 / P9.7's own rule).

    THE STAGED CONTRACT, EVEN THOUGH THE GRAPH NEVER RUNS: this yields
    exactly the same ("outcome", (audio_path_or_None, text, status)) shape
    every other branch of _run_followup_events/_run_pipeline_events yields
    (see _outcome_for's issue #88 / P9.9 note for `status`), so
    _stage_events (the caller two levels up) stages it identically to a
    real answer - the confirmation is genuinely SPOKEN (through run_tts,
    called directly here since there is no graph run to produce it), not
    merely written to the text box. clarif_eye.tts.run_tts already never
    raises (see its own docstring) and always returns {"audio_file_path":
    "" } on any failure, so this function inherits that same guarantee with
    no try/except of its own needed. The (audio_path, text, status) MAPPING
    itself is _outcome_for, the SAME helper _run_pipeline_events/_run_followup_events
    use for every other outcome - reused rather than hand-rolled, so a
    confirmation spoken while the TTS provider chain happens to be
    exhausted gets the same AUDIO_UNAVAILABLE_NOTE fallback wording every
    other spoken response already gets, instead of a second, silently
    different degrade path.

    set_verbosity (clarif_eye.preferences) is itself never-raising and a
    silent no-op when `resources.store` is None or `session_id` is falsy -
    see that function's docstring - so a store-less AppResources (every
    existing test that builds one directly) or a caller with no session_id
    still gets the SAME spoken confirmation. That is a deliberate honesty
    trade-off: the confirmation describes intent ("I will keep descriptions
    shorter"), and intent is genuinely what was just recognised and acted
    on, even on the rare path where there is nowhere durable to record it.
    """
    set_verbosity(resources.store, session_id, verbosity)
    confirmation = (
        PREFERENCE_CONFIRMATION_SHORT
        if verbosity == VERBOSITY_SHORT
        else PREFERENCE_CONFIRMATION_DETAILED
    )
    spoken = _to_spoken_text(confirmation)
    audio_path = run_tts(spoken, providers=resources.tts_providers).get("audio_file_path") or ""
    yield "outcome", _outcome_for(spoken, audio_path)


def _run_followup_events(question, resources, pipeline_budget_seconds, thread_id=None, session_id=None):
    """Generator: the follow-up sibling of _run_pipeline_events (issue #82 /
    P9.3). Answers a typed question about the photo this thread already
    described, and yields the SAME ("status", phrase) / ("outcome", (audio,
    text)) event shape, so handle_ask_staged can stage an answer exactly the
    way handle_submit_staged stages a description.

    A SIBLING, NOT A BRANCH INSIDE _run_pipeline_events: every guard at the
    top of that function is about a PHOTO (no image, an unreadable image,
    the image cache) and none of them apply here. Threading a "is this a
    question?" flag through them would mean four dead branches per call and
    a reader having to hold both flows in their head at once. What the two
    genuinely share is factored out instead and called from both:
    _narrate_stream, _record_turn, _outcome_for.

    ONE MODEL CALL, NO VISION CALL. The input passed to the graph is ONLY
    the delta {"question": question} - never a full make_initial_state().
    That is not a micro-optimisation, it is the whole mechanism: on a
    checkpointed thread, keys present in the input REPLACE the checkpointed
    value and keys absent are preserved (verified empirically on langgraph
    1.2.10; see clarif_eye.state.ClarifEyeState.question). Passing a full
    initial state here would overwrite ocr_output/scene_context with empty
    strings and there would be nothing left to answer from - the run would
    correctly report that no photo has been described yet, on a thread that
    just described one. clarif_eye.graph.entry_node then reads `question`
    and returns Command(goto="followup"), so `vision` never runs at all.

    THE IMAGE CACHE IS NEVER TOUCHED (neither read nor written): it is keyed
    on image CONTENT and holds a whole photo's (audio, description) result.
    Two different questions about the same photo have different answers, and
    the same question on two different threads is about two different
    photos, so the key that cache uses is meaningless here - a hit would be
    a wrong answer read aloud with confidence.

    NEVER RAISES (except KeyboardInterrupt/SystemExit) - same contract as
    every other entry point in this module.

    NO USABLE THREAD IS NOT AN ERROR: with thread_id=None, or an
    AppResources whose thread_registry is None (an uncheckpointed graph -
    see thread_configurable's pairing invariant), the graph runs with no
    stored state, `followup` finds no ocr/scene, and
    clarif_eye.followup.NO_PHOTO_YET_MESSAGE is what gets spoken. That is
    the correct answer for that situation, reached through the ordinary
    path with no special case here.

    A PENDING QUESTION BLOCKS THIS ENTIRELY (issue #83 / P9.4, and this is
    a real bug fixed, not a precaution). Running a follow-up on a thread
    that is paused waiting for an answer SUPERSEDES the pending task -
    probe-confirmed on langgraph 1.2.10: get_state().next goes back to ()
    and the interrupt is gone. So the question the app asked about an
    unverified number would be silently destroyed by the user typing
    something unrelated, while both answer buttons stayed on screen wired
    to a resume that would then find nothing to resume. The rule is to
    REFUSE and say so: see QUESTION_PENDING_MESSAGE. Checked BEFORE the
    graph is touched, so the refusal costs no model call and writes no
    state - the pause is left exactly as it was and can still be answered.

    THE HOLD IS NOT CLEARED HERE, and does not need to be - but the reason
    CHANGED COMPLETELY in issue #92 / P9.11 and this paragraph used to state
    the old one as if it still held. It said `followup` never routes through
    an asking node and that a stale verification_hold therefore could not
    divert an answer. Both clauses are now false: there IS a conditional edge
    out of `followup` (clarif_eye.graph.followup_destination), and a stale
    hold on the thread is exactly what would divert a perfectly good answer
    into a question about a number nobody just heard.

    THE REAL INVARIANT, and the only thing keeping that safe:
    clarif_eye.followup.run_followup writes `verification_hold` on EVERY
    return path - None on all of them except the one that genuinely holds
    something, which it writes fresh. So whatever the thread was carrying
    before this run is REPLACED by this run's own verdict before the edge is
    ever evaluated (it is a plain, non-reducer state key - see
    clarif_eye.state.ClarifEyeState.verification_hold). Nothing here has to
    clear anything, because the answering node cannot leave last time's
    verdict standing. The pending-question guard above is a separate
    protection for a separate problem: it stops a follow-up running at all
    while a question is unanswered.

    A PREFERENCE-SETTING COMMAND (issue #86 / P9.7) IS CHECKED AFTER THE
    BLANK/NON-STRING GUARDS AND AFTER THE PENDING-INTERRUPT CHECK BELOW -
    see _handle_preference_command. "shorter descriptions please" and its
    tiny closed-vocabulary siblings (clarif_eye.preferences.
    detect_preference_command) never reach `followup`/the brain model at
    all: they cost no model call, write only to the Store (never to
    resources.graph's checkpointer), and the run returns immediately with a
    spoken confirmation. Checked AFTER `resources.client is None` above on
    purpose, not before - a client-not-configured process is broken beyond
    what a stored preference could meaningfully affect, so that guard's
    existing CONFIG_ERROR_MESSAGE answer is left as the honest one even for
    a typed preference command.

    A PENDING QUESTION OUTRANKS A PREFERENCE COMMAND TOO (deep-review MAJOR
    fix, issue #86 / P9.7) - THIS WAS A REAL BUG, not a precaution, the same
    class as the follow-up-vs-pause bug this function's own docstring
    already describes above. An earlier version of this function checked
    detect_preference_command BEFORE _has_pending_interrupt, so
    "shorter descriptions please" typed while the app was waiting on the
    unverified-number safety question was silently accepted and confirmed -
    the pause itself stayed intact (nothing here touches the graph on that
    path), but the user heard only the verbosity confirmation, never
    QUESTION_PENDING_MESSAGE, with no idea a safety question was still
    waiting for them. THE DECIDED RULE, matching the ordinary-follow-up
    collision exactly: while a question is pending, the ONLY accepted
    actions are the two choice buttons or a new photo. A preference command
    is now detected AFTER the pending-interrupt check in the try block
    below, so it gets the SAME QUESTION_PENDING_MESSAGE refusal an ordinary
    follow-up gets, and nothing is ever written to the Store while a pause
    is waiting.
    """
    if resources.client is None:
        yield "outcome", _degraded_outcome(resources.client_error or CONFIG_ERROR_MESSAGE)
        return

    # See NO_QUESTION_MESSAGE: a blank question must never reach the graph,
    # because entry_destination would route it to `vision` and re-run the
    # whole photo pipeline.
    #
    # THE isinstance CHECK IS NOT DECORATION (deep-review MINOR, issue #82 /
    # P9.3): this line runs BEFORE the try below, so a non-string question
    # would have raised AttributeError from .strip() straight into Gradio.
    # Nothing reaches this with a non-string today - Gradio's Textbox
    # preprocesses to str - but this module's contract is "never raise",
    # not "never raise while Gradio behaves as expected", and
    # _run_followup_events is a public-ish seam that scripts and tests call
    # directly. Anything that is not a usable string is treated as no
    # question at all, which is the honest reading of it.
    # clarif_eye.graph.entry_destination's TypeError stays what it already
    # was: the second layer, for a question that reaches the graph by some
    # other route.
    if not isinstance(question, str):
        yield "outcome", _degraded_outcome(NO_QUESTION_MESSAGE)
        return
    question = question.strip()
    if not question:
        yield "outcome", _degraded_outcome(NO_QUESTION_MESSAGE)
        return

    try:
        configurable = {
            "client": resources.client,
            "tts_providers": resources.tts_providers,
            "deadline": time.monotonic() + pipeline_budget_seconds,
        }
        configurable.update(thread_configurable(resources, thread_id, session_id=session_id))
        config = {"configurable": configurable}
        graph = resources.graph
        # A pending safety question outranks a follow-up - see this
        # function's docstring for what running one anyway used to destroy.
        # Checked here, inside the try and after thread_configurable, so it
        # reads the SAME config the run would have used and so a broken
        # get_state can never raise into Gradio.
        #
        # CHECKED BEFORE detect_preference_command BELOW, ON PURPOSE (deep-
        # review MAJOR fix, issue #86 / P9.7 - see this function's own
        # docstring "A PENDING QUESTION OUTRANKS A PREFERENCE COMMAND TOO"):
        # a preference-setting command typed while a run is paused must get
        # the SAME refusal an ordinary follow-up gets, not be silently
        # accepted while the safety question goes unanswered and unheard.
        # The refusal is WORDED FOR WHATEVER IS ACTUALLY PENDING (issue #92 /
        # P9.11 deep-review MAJOR): the same read that detects the pause
        # yields the payload, and _question_pending_message picks the noun
        # from its `reason` field. Telling a user that a DESCRIPTION is
        # waiting when the question they were asked was about the answer to
        # their own last question - and offering "take a new photo" as the
        # way past it - would be a third piece of photo wording on a path
        # where the photo was never the problem.
        pending = _pending_interrupt_payload(_thread_snapshot(graph, config))
        if pending is not None:
            yield "outcome", _degraded_outcome(_question_pending_message(pending))
            return

        verbosity_command = detect_preference_command(question)
        if verbosity_command is not None:
            yield from _handle_preference_command(verbosity_command, resources, session_id)
            return

        # The DELTA, and nothing else - see this function's docstring.
        state = {"question": question}
        result = dict(state)
        # DRAINED, not `yield from` (issue #92 / P9.11): this function has to
        # SEE an ("interrupt", ...) event go past, exactly like
        # _run_pipeline_events does and for the same reason. A follow-up run
        # can now pause - `followup` holds an answer back when a number in it
        # could not be traced to the photographed text, and `verify_answer`
        # stops to ask about it. A paused run has produced no outcome yet:
        # final_output currently holds followup's safe "could not be checked"
        # refusal and there is no audio, so everything below (recording the
        # turn, mapping an outcome) would speak a non-answer AND record it as
        # this question's answer. It gets none of that; the run resumes, or it
        # doesn't.
        paused = False
        for kind, payload in _narrate_stream(graph, state, config, result):
            if kind == "interrupt":
                paused = True
            yield kind, payload
        if paused:
            return
    except LadderExhaustedError as exc:
        yield "outcome", _degraded_outcome(message_for_ladder_exhausted(exc))
        return
    except OpenRouterError as exc:
        yield "outcome", _degraded_outcome(message_for_terminal_error(exc))
        return
    except Exception:
        yield "outcome", _degraded_outcome(UNEXPECTED_ERROR_MESSAGE)
        return

    final_output = (result.get("final_output") or "").strip()
    audio_path = result.get("audio_file_path") or ""

    # BOTH SIDES OF THE TURN are recorded, unlike a photo run (which records
    # only the assistant's description - the "user" side of that turn is a
    # photograph, not text, and the base64 JPEG is already in the checkpoint
    # under image_data). Here the user's side IS text and is the thing that
    # makes the assistant's answer make sense when read back.
    #
    # AND BOTH SIDES GO OR STAY TOGETHER when the answer degraded (issue #93
    # / P9.12): the pair is handed to _record_turn whole and it decides - see
    # its docstring for why a question recorded without its answer is worse
    # than no record at all. This is THE path the issue was filed from: a
    # question typed before any photo is answered with
    # clarif_eye.followup.NO_PHOTO_YET_MESSAGE, which is spoken perfectly
    # normally and used to be written into history as this thread's answer.
    degraded = bool(result.get("output_degraded"))
    if thread_id is not None and final_output:
        _record_turn(
            graph,
            config,
            thread_id,
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": final_output},
            ],
            degraded,
        )

    yield "outcome", _outcome_for(final_output, audio_path)


def handle_ask_staged(
    question, resources, pipeline_budget_seconds=DEFAULT_PIPELINE_BUDGET_SECONDS, thread_id=None,
    session_id=None, pause_signal=None,
):
    """The follow-up sibling of handle_submit_staged (issue #82 / P9.3):
    answers a typed question and yields the SAME staged (status_text,
    audio_path_or_None, description_text) contract, including the
    AUDIO_PLAY_DELAY_MS gap before the audio path appears.

    Only the opening announcement differs (STATUS_ASKING rather than
    STATUS_WORKING - see that constant for why the wait warning is not
    repeated), because everything after it - per-node narration, the
    status/text-then-audio split, the delay - must feel identical to a
    photo run. A user who has just heard a description read aloud should not
    have to learn a second interaction rhythm to ask about it.

    `session_id` (issue #86 / P9.7) is OPTIONAL and defaults to None, same
    shape as `thread_id` - see clarif_eye.ui.thread_configurable and
    clarif_eye.preferences for what passing one actually does: it is both
    the namespace a recognised preference COMMAND is written under, and the
    namespace later runs (on this thread or another) read a stored
    preference back from.

    `pause_signal` (issue #92 / P9.11) is OPTIONAL and defaults to None,
    same shape and same purpose as handle_submit_staged's own - see
    _PauseSignal. It exists here now because a follow-up CAN pause: an
    answer whose numbers do not trace back to the photographed text stops to
    ask the user, exactly as a drafted description already could. Without
    it, build_interface would have no way to reveal the two answer buttons
    for a question raised on this path, and a blind user would be asked
    something with nothing on screen to answer it with.
    """
    yield from _stage_events(
        STATUS_ASKING,
        _run_followup_events(
            question, resources, pipeline_budget_seconds, thread_id=thread_id, session_id=session_id
        ),
        pause_signal=pause_signal,
    )


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
    same return-tuple contract as before streaming existed. An "outcome"
    payload is (audio_path_or_None, text, status) as of issue #88 / P9.9
    (see _stage_events); only the first two elements are this function's
    own (audio, text) return contract, so `status` is dropped here rather
    than widening what every existing caller of handle_submit unpacks.
    """
    outcome = (None, UNEXPECTED_ERROR_MESSAGE)
    for kind, payload in _run_pipeline_events(image, resources, pipeline_budget_seconds, thread_id=thread_id):
        if kind == "outcome":
            outcome = payload[:2]
        elif kind == "interrupt":
            # A paused run (issue #83 / P9.4) never produces an "outcome"
            # event, so without this branch this function would return the
            # generic UNEXPECTED_ERROR_MESSAGE default and tell the user
            # something went wrong when nothing did. The question IS the
            # result for this call; the caller resumes via
            # handle_resume_staged. No audio - the question is spoken by
            # the screen reader through the live region, not by TTS.
            outcome = (None, payload)
    return outcome


@dataclass
class _PauseSignal:
    """One mutable bit shared between a staged run and its Gradio wrapper:
    did this run PAUSE to ask the user a question? (issue #83 / P9.4)

    WHY A SIDE-CHANNEL AND NOT A FOURTH ELEMENT in the staged tuple: the
    (status, audio, text) triple _stage_events yields is a contract every
    existing caller and test in this repo unpacks by shape. Widening it
    would rewrite all of them for one boolean that only ONE consumer -
    build_interface, deciding whether to show the two resume buttons -
    actually needs. It is set the moment the interrupt event arrives, which
    is strictly before the final yield, so a wrapper reading it per-yield
    always sees the right value on the yield that matters.

    Optional everywhere: pass nothing and the staged behaviour is exactly
    what it was.
    """

    paused: bool = False


def _stage_events(opening_status, events, pause_signal=None):
    """THE staged yield contract, in one place.

    Turns an ("status"/"outcome", payload) event stream - from
    _run_pipeline_events for a photo, _run_followup_events for a question -
    into the (status_text, audio_path_or_None, description_text) tuples
    Gradio streams straight to the UI.

    ISSUE #88 / P9.9: an "outcome" payload is now a 3-tuple, (audio_path,
    text, status) - `status` is read from it verbatim, never recomputed
    here via status_for_result(audio_path, is_chain_exhausted()). That
    used to re-read tts.is_chain_exhausted()'s MODULE-GLOBAL state at the
    end of every run, including a run that failed before TTS was ever
    attempted (e.g. a LadderExhaustedError while reading the photo) - so
    an early failure could announce a PREVIOUS request's "chain exhausted,
    here is the text" status over its own failure message. Every "outcome"
    site now carries the correct status for what actually happened in
    THAT run: _outcome_for computes it at the one point
    is_chain_exhausted() is guaranteed to describe THIS run's own TTS
    attempt (see that function), and every early-failure/guard "outcome"
    literal in _run_pipeline_events/_run_followup_events/_run_resume_events
    carries STATUS_DEGRADED directly, since none of them ever reach TTS.

    ONE COPY, ON PURPOSE: handle_submit_staged and handle_ask_staged
    differed only in their opening announcement and which event generator
    they drained, but each carried its own copy of this ending. The
    AUDIO_PLAY_DELAY_MS gap is an accessibility contract measured against
    the wording of STATUS_SUCCESS_AUDIO (see that constant's comment, and
    AUDIO_PLAY_DELAY_MS's history of a broken JS-only attempt at the same
    gap) - two copies of it is two chances for a future edit to fix the
    timing for descriptions and silently leave answers talking over
    themselves.

    The sequence, unchanged from before this was extracted:
      1. `opening_status` immediately, with no audio and no text - the
         "received, working on it" announcement, before anything has run.
      2. one yield per ("status", phrase) event, as each node completes.
      3. the final status AND text but NO audio, so a screen-reader user
         can read the answer straight away.
      4. only when audio was actually produced, the SAME status and text
         once more after AUDIO_PLAY_DELAY_MS, now carrying the audio path -
         so Gradio only mounts the autoplaying player once the completion
         status has had time to be spoken.

    A PAUSED RUN (issue #83 / P9.4) replaces steps 3 and 4 with a single
    yield carrying the spoken question, and NEVER reaches status_for_result
    or _outcome_for. That is structural, not a nicety: a paused run has no
    audio and no finished answer, so _outcome_for would read its empty
    audio path, consult is_chain_exhausted() - which reports on whatever
    the LAST completed run did, not on this one, since no tts call happened
    here at all - and quite possibly announce UNEXPECTED_ERROR_MESSAGE over
    a run that is working exactly as designed. The question goes into BOTH
    the status (which is the aria-live region, so a screen reader speaks it
    without being asked) and the description text (so it can be re-read,
    and so FOCUS_RESULT_JS lands the user on it).
    """
    yield opening_status, None, ""
    audio_path, text, status = None, "", STATUS_DEGRADED
    question = None
    for kind, payload in events:
        if kind == "status":
            yield payload, None, ""
        elif kind == "interrupt":
            question = payload
            if pause_signal is not None:
                pause_signal.paused = True
        else:
            audio_path, text, status = payload
    if question is not None:
        yield question, None, question
        return
    if not audio_path:
        yield status, audio_path, text
        return
    yield status, None, text
    time.sleep(AUDIO_PLAY_DELAY_MS / 1000)
    yield status, audio_path, text


def _run_resume_events(answer, resources, pipeline_budget_seconds, thread_id=None):
    """Generator: the resume sibling of _run_pipeline_events (issue #83 /
    P9.4). Answers the question a paused run asked, and yields the same
    ("status", phrase) / ("outcome", (audio, text)) event shape, so a
    resumed run is staged exactly like a description or an answer.

    NEVER RAISES (except KeyboardInterrupt/SystemExit) - same contract as
    every other entry point in this module.

    THE "NOTHING IS PAUSED" CASE IS DETECTED STRUCTURALLY, BEFORE RESUMING,
    and that ordering is the whole guard (D15 - no exception-driven control
    flow, and no prose matching). Verified empirically on langgraph 1.2.10:
    graph.get_state(config).interrupts is a non-empty tuple exactly while a
    run is waiting on an answer, and empty otherwise - on a thread that
    completed, on a thread that never paused, and on a thread this process
    has never seen (a restart between question and answer, since the pause
    lives in an in-process InMemorySaver - see build_resources). Resuming
    anyway is not harmless: on a thread with no stored state at all,
    graph.stream(Command(resume=...)) raises KeyError from inside LangGraph
    as the first node reads a key that was never written. So the check
    comes first, and the answer is a spoken explanation - see
    NOTHING_TO_RESUME_MESSAGE, which covers both situations honestly.

    RECORDING: only a RESUME_CONTINUE that produced something spoken records
    a turn. The caveated script IS this photo's honest answer, so the thread
    should remember it. A retake is NOT an answer about the photo - it is the
    user declining one - so recording it would put "please take a new photo"
    into the history a later follow-up reads back as if it described the
    document. Issue #93 / P9.12 has since settled the wider question this
    branch was an early instance of: no degraded outcome enters conversation
    memory on ANY path, decided from state rather than per call site - see
    _record_turn and clarif_eye.state.ClarifEyeState.output_degraded. The
    answer-specific check below stays as well, because "the user declined"
    is a fact only this function knows.

    THE IMAGE CACHE IS NEVER WRITTEN HERE, even on a fully-spoken continue.
    Caching is keyed on image CONTENT and this function never sees the
    photo - the key would have to be smuggled across the pause through
    session state. It is also not obviously desirable: replaying a cached
    caveated script would answer the question ("continue anyway?") on
    behalf of the NEXT person to submit that photo, who was never asked.
    An interrupted photo simply costs its quota again, and asks again.
    """
    if resources.client is None:
        yield "outcome", _degraded_outcome(resources.client_error or CONFIG_ERROR_MESSAGE)
        return

    try:
        configurable = {
            "client": resources.client,
            "tts_providers": resources.tts_providers,
            "deadline": time.monotonic() + pipeline_budget_seconds,
        }
        configurable.update(thread_configurable(resources, thread_id))
        config = {"configurable": configurable}
        graph = resources.graph
        # No thread_id survived thread_configurable's pairing guard (no
        # checkpointer, no registry, or no thread at all) - then nothing
        # can be paused, because a pause only exists in a checkpoint.
        #
        # READ ONCE, USED TWICE (issue #92 / P9.11): the same snapshot that
        # proves something is pending also carries WHICH FLOW asked (the
        # payload's `reason`) and, for a follow-up, the question that was
        # typed before the pause. Both are needed below to record the turn
        # honestly, and both must be read BEFORE the resume - the hold is
        # cleared by the asking node as it completes.
        snapshot = _thread_snapshot(graph, config)
        pending = _pending_interrupt_payload(snapshot)
        if "thread_id" not in configurable or pending is None:
            yield "outcome", _degraded_outcome(NOTHING_TO_RESUME_MESSAGE)
            return
        # `question` survives the pause untouched - it is a plain,
        # non-reducer state key and the paused run never overwrote it (see
        # clarif_eye.state.ClarifEyeState.question). None on a photo pause,
        # which make_initial_state seeds explicitly.
        paused_question = (getattr(snapshot, "values", None) or {}).get("question")
        result = {}
        yield from _narrate_stream(graph, Command(resume=answer), config, result)
    except LadderExhaustedError as exc:
        yield "outcome", _degraded_outcome(message_for_ladder_exhausted(exc))
        return
    except OpenRouterError as exc:
        yield "outcome", _degraded_outcome(message_for_terminal_error(exc))
        return
    except Exception:
        yield "outcome", _degraded_outcome(UNEXPECTED_ERROR_MESSAGE)
        return

    final_output = (result.get("final_output") or "").strip()
    audio_path = result.get("audio_file_path") or ""

    if answer == RESUME_CONTINUE and final_output:
        # The caveated script IS this photo's answer, so the asking node
        # writes output_degraded=False on the continue branch and this
        # records (issue #93 / P9.12). The flag is passed through rather than
        # hard-coded False so that if a future resume path can degrade, it
        # degrades here too.
        #
        # BOTH SIDES OF A RESUMED FOLLOW-UP (issue #92 / P9.11 deep-review
        # MEDIUM), and this fixes a real orphan rather than tidying one up.
        # A photo run records the assistant's side alone - the user's side of
        # that turn is a photograph, not text. A follow-up's user side IS
        # text, and _run_followup_events already records the pair for the
        # unpaused case precisely because an answer with no question above it
        # is misread: stored alone under the photo's description, "The jar
        # holds 500 grams." reads back as a SECOND description of the photo,
        # which is the same failure class #93 exists to prevent. Recording
        # both keeps the paused path's history identical in shape to the
        # unpaused path's.
        #
        # WHICH FLOW IS DECIDED STRUCTURALLY, from the pending payload's own
        # `reason` - not from "is there a question in state?", which would
        # also be true on a PHOTO pause taken on a thread that answered a
        # question earlier in the session and would attach that stale
        # question to a description. The question text still has to be
        # non-empty for there to be anything to record.
        messages = [{"role": "assistant", "content": final_output}]
        if _asked_about_an_answer(pending) and (paused_question or "").strip():
            messages.insert(0, {"role": "user", "content": paused_question})
        _record_turn(
            graph,
            config,
            thread_id,
            messages,
            bool(result.get("output_degraded")),
        )
    else:
        # No turn to record, but the run DID complete, so this thread's
        # checkpoint history still wants bounding - see
        # _trim_thread_to_latest_checkpoint's measured defect. _record_turn
        # trims as a side effect of writing; this is the same housekeeping
        # without the write.
        _trim_thread_to_latest_checkpoint(getattr(graph, "checkpointer", None), thread_id)

    yield "outcome", _outcome_for(final_output, audio_path)


def _thread_snapshot(graph, config):
    """graph.get_state(config), or None if this graph/thread cannot report
    one. Never raises.

    Split out (issue #92 / P9.11) because two callers now need MORE than
    "is something paused": _run_followup_events needs the pending payload to
    word its refusal, and _run_resume_events needs both that payload and the
    thread's stored `question` to record the resumed turn as a pair. One
    read, several answers, and no second source of truth alongside the
    checkpointer.
    """
    try:
        return graph.get_state(config)
    except Exception:
        return None


def _pending_interrupt_payload(snapshot):
    """The pending question's structural payload dict, or None when nothing
    is paused.

    STRUCTURAL (D15): reads the `interrupts` tuple LangGraph itself
    populates from the pending task's interrupt writes - never a flag this
    module maintains alongside it, which would be a second source of truth
    able to disagree with the checkpointer.

    Returns {} rather than None for a pending interrupt whose value is not a
    dict, so callers can tell "nothing is paused" (None) apart from "paused,
    but the payload is not a shape this module knows" ({}) - the latter
    still has to be refused and still has to be resumable, it just gets the
    default wording.
    """
    interrupts = getattr(snapshot, "interrupts", None) or ()
    if not interrupts:
        return None
    value = interrupts[0].value
    return value if isinstance(value, dict) else {}


def _has_pending_interrupt(graph, config):
    """True if `config`'s thread is paused waiting on an answer.

    Never raises: a graph with no checkpointer, a test double with no
    get_state, or any other shape this module has not anticipated is
    reported as "nothing is paused", which is both true (nothing that can
    be resumed exists) and the safe answer - the caller then speaks
    NOTHING_TO_RESUME_MESSAGE instead of losing the click to a traceback.
    """
    return _pending_interrupt_payload(_thread_snapshot(graph, config)) is not None


def handle_resume_staged(
    answer, resources, pipeline_budget_seconds=DEFAULT_PIPELINE_BUDGET_SECONDS, thread_id=None
):
    """The resume sibling of handle_submit_staged / handle_ask_staged
    (issue #83 / P9.4): answers the question a paused run asked and yields
    the SAME staged (status_text, audio_path_or_None, description_text)
    contract, including the AUDIO_PLAY_DELAY_MS gap.

    `answer` is RESUME_CONTINUE or RESUME_RETAKE (clarif_eye.graph). It is
    passed through to the graph untouched and interpreted there, not here -
    verify_numbers_node treats anything that is not RESUME_CONTINUE as a
    retake, so a garbled value can never become consent to speak an
    unverified number.
    """
    yield from _stage_events(
        STATUS_RESUMING,
        _run_resume_events(answer, resources, pipeline_budget_seconds, thread_id=thread_id),
    )


def handle_submit_staged(
    image, resources, pipeline_budget_seconds=DEFAULT_PIPELINE_BUDGET_SECONDS, thread_id=None,
    pause_signal=None, session_id=None,
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

    `pause_signal` (issue #83 / P9.4) is OPTIONAL and defaults to None - see
    _PauseSignal for why the "did this run pause?" bit travels beside the
    yields rather than inside them.

    `session_id` (issue #86 / P9.7) is OPTIONAL and defaults to None, same
    shape as `thread_id` - build_interface passes each browser session's own
    minted session_id (a SECOND gr.State, alongside thread_state) through
    here, so fast_synth_node/analysis_node can read a verbosity preference
    set on THIS OR ANY OTHER thread of the same session - see
    clarif_eye.preferences's module docstring for the honest scope of that
    ("crosses threads" in the sense the store's namespacing genuinely
    supports, not in the sense that a live browser session normally has more
    than one thread to demonstrate it with).
    """
    yield from _stage_events(
        STATUS_WORKING,
        _run_pipeline_events(image, resources, pipeline_budget_seconds, thread_id=thread_id, session_id=session_id),
        pause_signal=pause_signal,
    )


# --- The second consumer of the deep path (issue #84 / P9.5) ---------------
#
# WHY THIS EXISTS AT ALL: extracting the deep path into a child graph only
# earns its keep if something other than the photo pipeline can actually use
# it. This is that something - text in, spoken-ready description out, no
# photo and no vision call anywhere in it. It is also the honest proof that
# the child's schema is its own: this caller has never seen a camera, and it
# still speaks the child's vocabulary without a translation layer of its own.
#
# AN API ROUTE, NOT A CONTROL ON THE PAGE. The main UI is built around one
# task - point a camera at something and hear what it says - and every
# control added to it is one more thing a screen-reader user has to tab past
# to get to that task. A second text box for a different job would cost them
# on every visit to pay for a capability aimed at programs, not people. So it
# is registered with gr.api (see build_interface), which adds a callable
# endpoint and NO components at all.
#
# TEXT OUT, NOT AUDIO, and that is a deliberate narrowing rather than an
# oversight. This app speaks, so returning text needs justifying: an API
# caller has no browser to autoplay into, no aria-live region, and no way to
# use the staged (status, audio, text) contract the UI is built on - it would
# get back a path to an mp3 in this server's temp directory, which is
# useless to it and which tts.py's own pruning may delete out from under it.
# The description IS the deliverable here; anything that wants it spoken can
# speak it.
DESCRIBE_TEXT_API_NAME = "describe_document_text"

# What the deep path is told about "the scene" when there is no photo. The
# child graph needs a scene description (clarif_eye.analysis refuses to call
# the model on a blank one), and the truthful answer for this route is that
# there is no scene - so it says exactly that rather than inventing one.
TEXT_ONLY_SCENE = "a document supplied as text, with no photo"


def describe_document_text(
    document_text, resources, pipeline_budget_seconds=DEFAULT_PIPELINE_BUDGET_SECONDS
):
    """Describe a document from its TEXT alone, by invoking the deep-path
    child graph directly (issue #84 / P9.5). Returns the spoken-ready
    description as a string.

    NEVER RAISES (except KeyboardInterrupt/SystemExit) - same contract as
    every other entry point in this module. An API route that returns a 500
    is worse than one that returns a sentence saying what went wrong.

    THE CHILD IS COMPILED WITHOUT A CHECKPOINTER, and that decides what
    happens when the deep path wants to ask a question. This route has no UI:
    nobody is there to be asked about a number that could not be checked, and
    there is no thread to resume on. VERIFIED empirically on langgraph
    1.2.10: `interrupt()` in an uncheckpointed graph does NOT raise - invoke()
    returns the state the run had reached, plus an "__interrupt__" key nobody
    here has to look at. `analysis` has already written the safe "could not be
    verified" script into final_output by then (see
    clarif_eye.analysis.run_analysis, which has always done that so a graph
    without the asking node degrades cleanly), so this route returns that:
    the honest, verified-safe answer, with the unverifiable number left out.
    Structural, not exception-driven - nothing here catches a pause to detect
    it.

    A FRESH CHILD PER CALL. Compiling is pure Python graph construction with
    no network and no model, this route is not a hot path, and a
    process-lifetime singleton would buy a lifecycle question (who builds it,
    when, and with what) for nothing measurable.

    IT SPENDS THE SAME DAILY ALLOWANCE THE UI DOES. There is one shared
    OpenRouter account behind this app and one free-tier quota (see
    clarif_eye.client), so every call here is a model call the photo pipeline
    cannot make later that day - and every uncached call is an outbound web
    search and page fetch too. That is why this route is CAPPED
    (DOCUMENT_TEXT_CAP) and CACHED (DocumentTextCache) rather than left open.

    WHAT THE CACHE DOES AND DOES NOT DEFEND AGAINST, stated plainly because
    "it's cached" invites more confidence than it has earned:
      - It stops SEQUENTIAL repeats. The same document asked for again, after
        the first answer exists, costs nothing.
      - It does NOT stop CONCURRENT ones. There is no single-flight: twelve
        identical requests arriving together all miss, all run, and all spend
        a model call, because the entry is only written once a run finishes.
        Deliberate, and the same shape ImageResultCache has always had -
        in-flight de-duplication means a lock or a futures map held across a
        request that can take a minute, which is real machinery to get right
        (and to not deadlock) for a demo app whose realistic worst case is a
        handful of visitors.
      - It has NO TTL, also like ImageResultCache: the first successful answer
        for a given document is pinned until it is evicted by LRU pressure or
        the process restarts. Fine here - the same text has the same
        description - but it does mean a better model shipping tomorrow does
        not re-answer today's cached documents.

    NO IMAGE CACHE (it is keyed on image content and there is no image), and
    NO THREAD (nothing to remember between calls - a follow-up question is a
    UI feature and belongs to a session, which an API caller does not have).
    """
    if not isinstance(document_text, str) or not document_text.strip():
        return NO_DOCUMENT_TEXT_MESSAGE
    if resources.client is None:
        return resources.client_error or CONFIG_ERROR_MESSAGE

    # CAPPED BEFORE ANYTHING ELSE, including before the cache key is taken,
    # so two bodies differing only past the cap share one entry - see
    # _document_text_key.
    capped = _cap_document_text(document_text.strip())
    cache_key = _document_text_key(capped)
    cached = resources.document_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        config = {
            "configurable": {
                "client": resources.client,
                "searcher": resources.searcher,
                "research_client": resources.research_client,
                "deadline": time.monotonic() + pipeline_budget_seconds,
            }
        }
        result = build_deep_path_graph().invoke(
            {
                "document_text": capped,
                "scene_description": TEXT_ONLY_SCENE,
            },
            config=config,
        )
        # READ INSIDE THE PROTECTED REGION, deliberately. These are
        # bracket-accesses, so a child that stopped writing any of these keys
        # would raise KeyError - and out here, past the excepts, that KeyError
        # would escape a gr.api route as a 500. Loud is right (see
        # clarif_eye.graph.make_deep_path_node on why the wrapper reads the
        # same keys the same way), but loud in the test suite, not at an API
        # caller: inside the try it degrades to UNEXPECTED_ERROR_MESSAGE like
        # every other unforeseen failure in this module.
        spoken = (result["final_output"] or "").strip()
        held_back = result["verification_hold"] is not None
        degraded = bool(result["output_degraded"])
    except LadderExhaustedError as exc:
        return message_for_ladder_exhausted(exc)
    except OpenRouterError as exc:
        return message_for_terminal_error(exc)
    except Exception:
        return UNEXPECTED_ERROR_MESSAGE

    if not spoken:
        return UNEXPECTED_ERROR_MESSAGE

    # CACHED ONLY WHEN THE RUN GENUINELY PRODUCED THIS DOCUMENT'S ANSWER, the
    # same discipline the image cache follows:
    #   - `analysis` did not degrade, so a busy-ladder message, an empty
    #     model reply, or a deadline-degraded read-back of the input is left
    #     to be retried rather than served to the next caller as this
    #     document's answer;
    #   - nothing was held back for being unverifiable. That outcome is this
    #     route's honest degradation (there is nobody here to ask - see
    #     above), not the description of the document, so a later caller gets
    #     a fresh attempt instead of a replayed refusal.
    #
    # `degraded` COMES FROM THE NODE, NOT FROM THE CLIENT SEAM (issue #93 /
    # P9.12), and this route used to do it the other way round: it wrapped
    # the client and watched whether any completion came back with usable
    # content. That wrapper was removed with this change because the flag
    # replaces it exactly - but its lesson is worth keeping, since it was
    # driven twice and failed both times at the same crack. Watching the
    # seam answers "did the model reply?", which is not the question. A
    # reply of "   " is non-empty at the transport and empty as an answer;
    # a reply that is an empty fenced code block is non-empty even AFTER
    # that check and still sanitises to nothing in
    # clarif_eye.speech.to_spoken_text, so run_analysis degrades and the
    # watcher, having seen a perfectly good completion, admitted the failure
    # sentence to this cache and served it to every later caller. Only the
    # node that produced final_output knows whether it is an answer, so only
    # the node's own flag can be trusted here - see
    # clarif_eye.state.ClarifEyeState.output_degraded.
    if not degraded and not held_back:
        resources.document_cache.put(cache_key, spoken)
    return spoken


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

    # The two resume buttons (issue #83 / P9.4) are hidden until a run
    # actually pauses, so nothing appears in the tab order offering a choice
    # about a question nobody was asked. _resume always hides them again,
    # whichever way the answer went.
    #
    # _ask REVEALS THEM TOO, as of issue #92 / P9.11, and it did not before.
    # A follow-up ANSWER is now held to the same number check a drafted
    # description is, so a typed question can raise the same pause - and a
    # question asked with no visible way to answer it would be worse than
    # not asking, for a user who cannot see that there is nothing there.
    # Both handlers use the same _PauseSignal mechanism and the same
    # gr.update, so the two paths cannot drift into revealing the buttons
    # differently.
    #
    # _ask ONLY EVER REVEALS THEM, NEVER HIDES THEM, and that asymmetry with
    # _submit is load-bearing rather than an oversight. A follow-up typed
    # while a question is ALREADY pending is REFUSED before the graph is
    # touched (see QUESTION_PENDING_MESSAGE and _run_followup_events): no
    # interrupt event is produced, so pause_signal stays False - and hiding
    # the buttons on that yield would take away the only way to answer a
    # question that is still very much pending, immediately after telling the
    # user to activate one of them. An empty gr.update() changes no property
    # at all (verified: it serialises to {"__type__": "update"} with no
    # `visible` key), which is exactly "leave them as they are". _submit can
    # safely hide, because submitting a photo RESOLVES any pending question
    # (an implicit retake - see _run_pipeline_events' cache-hit branch).
    def _submit(image, thread_id, session_id):
        pause_signal = _PauseSignal()
        for status, audio, text in handle_submit_staged(
            image, resources, thread_id=thread_id, pause_signal=pause_signal, session_id=session_id
        ):
            visible = gr.update(visible=pause_signal.paused)
            yield status, audio, text, visible, visible

    def _resume(answer, thread_id):
        hidden = gr.update(visible=False)
        for status, audio, text in handle_resume_staged(answer, resources, thread_id=thread_id):
            yield status, audio, text, hidden, hidden

    def _ask(question, thread_id, session_id):
        pause_signal = _PauseSignal()
        for status, audio, text in handle_ask_staged(
            question, resources, thread_id=thread_id, session_id=session_id,
            pause_signal=pause_signal,
        ):
            reveal = gr.update(visible=True) if pause_signal.paused else gr.update()
            yield status, audio, text, reveal, reveal

    # Issue #84 / P9.5. Fully type-hinted because gr.api derives the
    # endpoint's typing from the signature rather than from components -
    # there are no components here, which is the point (see
    # DESCRIBE_TEXT_API_NAME's comment for why this is not a box on the
    # page).
    def _describe_document_text(document_text: str) -> str:
        """Describe a document from its text alone, with no photo."""
        return describe_document_text(document_text, resources)

    with gr.Blocks(title="Clarif-Eye") as demo:
        thread_state = gr.State(value=lambda: str(uuid.uuid4()))
        # Issue #86 / P9.7: a SECOND per-session id, minted the same way
        # thread_state is (one gr.State callable, called once per browser
        # session - see thread_state's own comment below) but namespacing
        # the cross-thread PREFERENCE store instead of the checkpointer.
        # Kept deliberately separate from thread_state rather than reusing
        # it: today the two are equal for the lifetime of one session (one
        # session, one thread - see clarif_eye.preferences's module
        # docstring for the honest scope this implies), but they answer
        # different questions - "which checkpoint" vs. "which preference" -
        # and collapsing them would make a future change to either one
        # (e.g. re-minting thread_id without starting a new session) alter
        # the other by accident.
        session_id_state = gr.State(value=lambda: str(uuid.uuid4()))
        gr.Markdown(
            "# Clarif-Eye\n"
            "Clarif-Eye describes a photo aloud for visually impaired users."
        )
        # Issue #87 / P9.8: two tabs, product and explanation (owner
        # request, verbatim - see PRODUCT_TAB_LABEL/EXPLANATION_TAB_LABEL's
        # module comment). gr.Tabs renders a keyboard-reachable tablist of
        # buttons, each named by its gr.Tab's `label` - both tabs carry an
        # elem_id too, for the same reason every other control here does:
        # so tests and the accessibility audit can find them structurally.
        with gr.Tabs():
            with gr.Tab(PRODUCT_TAB_LABEL, elem_id=PRODUCT_TAB_ELEM_ID):
                gr.Markdown(
                    "Take or upload a photo below. This can take up to "
                    "about 30 seconds, especially for photos with dense "
                    "text."
                )
                image_input = gr.Image(
                    label="Photo to describe",
                    sources=["upload", "webcam"],
                    type="pil",
                    # issue #48 / P5.4: lets ARIA_LIVE_HEAD's image-labelling
                    # shim find the uploaded-photo preview structurally (it's
                    # the <img> inside this container), instead of any icon
                    # glyph elsewhere on the page.
                    elem_id=IMAGE_INPUT_ELEM_ID,
                    # issue #59 / P4.5: Gradio mirrors the webcam by default
                    # (WebcamOptions.mirror=True), which suits selfies. This
                    # app has no selfie case - users photograph bills,
                    # labels, and signs so the vision model can read the
                    # text, and a blind user is not looking at the preview
                    # to notice a mirrored frame. Mirroring reversed that
                    # text before the model ever saw it. Do not restore the
                    # default thinking it "looks more natural" - it makes
                    # captured text unreadable.
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
                # autoplay=True (issue #47 / P5.3): Gradio only assigns the
                # <audio> element a source at all when autoplay is on - with
                # it off, no src is ever set and playback can never start by
                # any means (see AUDIO_PLAY_DELAY_MS's comment for how that
                # was diagnosed). The gap between the completion announcement
                # and audio starting is created instead by
                # handle_submit_staged withholding the audio path for one
                # extra yield, not by JS here.
                audio_output = gr.Audio(
                    label="Spoken description", autoplay=True, elem_id=AUDIO_ELEM_ID
                )
                text_output = gr.Textbox(
                    label="Description (text)", lines=6, elem_id=RESULT_ELEM_ID
                )

                # --- Answering the unverifiable-number question (issue #83
                # / P9.4)
                #
                # PLACED IMMEDIATELY AFTER the result text, which is where
                # the question itself appears and where FOCUS_RESULT_JS has
                # just put focus - so the two answers are the very next
                # thing a keyboard or screen-reader user reaches by tabbing
                # forward from the question they were just read. Ordinary
                # gr.Buttons: real labels, real tab stops, Enter and Space
                # activate them, nothing custom to get wrong.
                #
                # visible=False initially, and every handler that touches
                # them sets visibility explicitly (see _submit/_resume
                # above) - so they exist in the tab order only while there
                # is genuinely a question waiting.
                with gr.Row():
                    resume_continue_button = gr.Button(
                        RESUME_CONTINUE_LABEL,
                        elem_id=RESUME_CONTINUE_BUTTON_ELEM_ID,
                        visible=False,
                    )
                    resume_retake_button = gr.Button(
                        RESUME_RETAKE_LABEL,
                        elem_id=RESUME_RETAKE_BUTTON_ELEM_ID,
                        visible=False,
                    )

                # --- Follow-up question (issue #82 / P9.3) ---------------
                #
                # PLACED AFTER the description output: asking about the
                # photo is a FOLLOW-ON action, so it belongs after the thing
                # it follows on from. Putting it above the result would sit
                # an input the user cannot use yet between the live region
                # and the answer it announces.
                #
                # IT MUST NOT STEAL FOCUS, and nothing here makes it: it is
                # an ordinary Textbox with `autofocus` left at its default
                # (off), and FOCUS_RESULT_JS keeps targeting
                # `#{RESULT_ELEM_ID} textarea` and nothing else, so after
                # either run focus still lands on the answer rather than on
                # this box.
                #
                # KEYBOARD-REACHABLE, BY DEFAULT AND BY A SECOND ROUTE: this
                # Textbox is interactive (not the disabled-textarea case
                # ARIA_LIVE_HEAD has to repair for the read-only description
                # output), so it is an ordinary tab stop with a real label.
                # `.submit` fires on Enter inside the box, so a keyboard
                # user never has to tab onward to the button - both events
                # are wired to the SAME handler below.
                question_input = gr.Textbox(
                    label="Question about the photo",
                    placeholder="For example: what is the expiry date?",
                    lines=1,
                    elem_id=QUESTION_INPUT_ELEM_ID,
                )
                ask_button = gr.Button("Ask about this photo", elem_id=ASK_BUTTON_ELEM_ID)

            # --- Explanation tab (issue #87 / P9.8) -----------------------
            #
            # HOW_IT_WORKS_MARKDOWN and PIPELINE_DIAGRAM_HTML moved here,
            # off the product tab, so the product flow stays uncluttered and
            # this content has room to grow. See HOW_IT_WORKS_MARKDOWN's and
            # PIPELINE_DIAGRAM_SVG's own module comments for content
            # sourcing and the #48 image-labelling exemption
            # (DIAGRAM_ELEM_ID) - both survive unchanged, just relocated.
            with gr.Tab(EXPLANATION_TAB_LABEL, elem_id=EXPLANATION_TAB_ELEM_ID):
                gr.Markdown(HOW_IT_WORKS_MARKDOWN, elem_id=HOW_IT_WORKS_ELEM_ID)
                # issue #56 / P4.4: a sighted-friendly diagram alongside
                # (never instead of) the list above. gr.HTML, not
                # gr.Markdown, because the diagram is raw inline SVG with
                # its own role/aria-label/aria-describedby wiring - see
                # PIPELINE_DIAGRAM_SVG's module comment for the exemption
                # that keeps ARIA_LIVE_HEAD's #48 pass from silencing it.
                gr.HTML(PIPELINE_DIAGRAM_HTML, elem_id=DIAGRAM_ELEM_ID)

        # The text-only consumer (issue #84 / P9.5): a callable endpoint with
        # no components, so it changes nothing about what is on the page, in
        # the tab order, or in the reading order.
        gr.api(_describe_document_text, api_name=DESCRIBE_TEXT_API_NAME)

        result_outputs = [status_output, audio_output, text_output]
        # The two resume buttons are outputs of all three handlers (issue
        # #83 / P9.4, extended by #92 / P9.11 to the ask handler, which can
        # now raise the same pause) - see _submit/_resume/_ask above for what
        # each one is allowed to do with them.
        result_and_controls = result_outputs + [resume_continue_button, resume_retake_button]

        submit_event = submit_button.click(
            fn=_submit,
            inputs=[image_input, thread_state, session_id_state],
            outputs=result_and_controls,
        )
        # Runs client-side only after the handler above has produced its
        # final yield - see FOCUS_RESULT_JS's docstring for why that
        # timing matters (never steals focus mid-interaction). A PAUSED run
        # gets this too, and should: the last yield put the question in the
        # description box, so focus lands on the question rather than being
        # left on the submit button the user has finished with.
        submit_event.then(fn=None, inputs=None, outputs=None, js=FOCUS_RESULT_JS)

        # Each resume button sends its OWN fixed answer, as a gr.State
        # constant rather than anything read back out of the page, so what
        # the graph receives cannot depend on client-side state at all.
        for resume_button, resume_answer in (
            (resume_continue_button, RESUME_CONTINUE),
            (resume_retake_button, RESUME_RETAKE),
        ):
            resume_event = resume_button.click(
                fn=_resume,
                inputs=[gr.State(resume_answer), thread_state],
                outputs=result_and_controls,
            )
            resume_event.then(fn=None, inputs=None, outputs=None, js=FOCUS_RESULT_JS)

        # An answer replaces the SAME status/audio/description outputs a
        # description uses, so a screen-reader user hears it announced by
        # the live region they already know and finds it where the last
        # answer was - not in a second result area they have to go looking
        # for. Both the button click and Enter-in-the-box run the same
        # handler, and both get the same focus-the-answer follow-up.
        for ask_event in (
            ask_button.click(
                fn=_ask,
                inputs=[question_input, thread_state, session_id_state],
                outputs=result_and_controls,
            ),
            question_input.submit(
                fn=_ask,
                inputs=[question_input, thread_state, session_id_state],
                outputs=result_and_controls,
            ),
        ):
            ask_event.then(fn=None, inputs=None, outputs=None, js=FOCUS_RESULT_JS)

    return demo
