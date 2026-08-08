"""Cross-thread spoken-verbosity preference via LangGraph's Store (issue
#86 / P9.7).

WHAT THIS DEMONSTRATES: `checkpointer` (issue #81 / P9.2) scopes state to
ONE thread; `InMemoryStore` (compile(store=...)) scopes it to whatever
namespace the caller chooses - which can span every thread that shares that
namespace. The bounded, honest use this module exists for: a user says
"shorter descriptions please" in the follow-up box, and later photos on
OTHER threads of the SAME BROWSER SESSION get a briefer script.

HONEST SCOPE, STATED HERE BECAUSE IT IS EASY TO OVERSTATE: a browser session
today mints exactly ONE thread_id (see clarif_eye.ui.build_interface's
thread_state), so in the live app "crosses threads" is not something a real
visitor normally sees happen - one session, one thread, no observable
difference from a preference stored on the thread itself. The capability is
still real (a second gr.State, session_id, is minted per session alongside
thread_id and threaded through the same clarif_eye.ui.thread_configurable
chokepoint - see that function - so a session that DID end up with more
than one thread, e.g. a future multi-tab or re-minted-thread flow, would
share the preference across them), and it is what the red-first tests in
tests/test_cross_thread_preferences.py prove directly: two different
thread_ids, one session_id, one store. Like the checkpointer's own honest
limits (clarif_eye.ui.build_resources), InMemoryStore keeps everything in
THIS PROCESS's memory only - a restart loses every session's preference,
exactly as a restart already loses every thread's checkpointed history.

PRIVACY, NON-NEGOTIABLE (the issue's own words): these are photographs of
medication labels and bills. The store may hold ONLY preference-shaped
values - never image content, OCR text, scene descriptions, scraped web
text, or the questions/answers a user typed - because cross-thread
persistence of any of that would outlive the single-run lifetime a user
implicitly expects from this app. Enforced by construction (this module is
the ONLY code that ever calls store.put with a real value, and the value it
writes is always exactly {"verbosity": "short" | "detailed"}) and by test
(tests/test_cross_thread_preferences.py walks every item search(()) returns
and asserts on its shape; a node that mutated to write ocr_output into the
store instead was proven, by hand, to turn that test red).

SETTING vs. APPLYING, the two halves this module owns:
  - detect_preference_command (SETTING): a CLOSED, TINY vocabulary match on
    the exact text the user typed in the follow-up box - "shorter",
    "shorter descriptions", "be brief" for VERBOSITY_SHORT; "longer",
    "more detail", "more details" for VERBOSITY_DETAILED. THIS IS NOT THE
    D15-STYLE PROHIBITION ("no structured decisions parsed from unstructured
    model/error prose") - that rule is about not inferring structure from
    text a MODEL or a FAILURE produced, where the shape can drift under the
    author's nose. Here the "unstructured" text is the opposite of that: it
    is the literal, deliberate command the USER just typed into a box whose
    whole purpose is to accept typed commands about the photo. The user's
    words ARE the data being matched, not a proxy inferred from something
    else. clarif_eye.ui checks this BEFORE the question ever reaches the
    graph (see _run_followup_events) - a recognised command costs no model
    call at all, matching the issue's "do NOT run the graph/model" rule.
  - verbosity_for_config / get_verbosity (APPLYING): read back by
    clarif_eye.graph's fast_synth_node/analysis_node/followup_node - the
    three nodes issue #86 names as the shared prompting seam - and folded
    into the prompt via clarif_eye.prompting.verbosity_instruction. A store
    read NEVER breaks a run: every function here that touches `store`
    degrades to None (i.e. "no preference on file, describe normally") on
    anything it does not expect - a None store (an uncompiled/store-less
    graph, or a direct unit-test call), a missing session_id, a namespace
    with nothing in it, or a stored value that does not match the one
    allowed shape.
"""

# The two recognised preference values (issue #86). Only these two strings
# are ever written into the store's "verbosity" value - see this module's
# top-level PRIVACY note and the shape check in get_verbosity below, which
# is the same test tests/test_cross_thread_preferences.py's privacy test
# re-derives independently rather than trusting this module's own claim.
VERBOSITY_SHORT = "short"
VERBOSITY_DETAILED = "detailed"

# Store namespace: (NAMESPACE_ROOT, session_id). A tuple, not a single
# string, because that is the shape langgraph.store.memory.InMemoryStore's
# put/get/search all take (verified empirically - see this issue's own
# probe). Keyed by SESSION, never by thread_id - the entire point of this
# module is that the same preference is visible from a different thread_id
# under the same session_id.
NAMESPACE_ROOT = "preferences"
PREFERENCE_KEY = "verbosity"

# THE TINY, CLOSED VOCABULARY (issue #86: "keep the set tiny and
# documented"). Every phrase here is normalised the same way the user's
# typed text is before matching (see _normalise) - lowercased, trailing
# punctuation and a trailing "please" stripped - so "Shorter descriptions,
# please!" and "shorter descriptions" match the same entry. Anything NOT in
# either set is not a command at all and flows to the followup node
# unchanged (issue #86's own wording), including a genuine QUESTION that
# happens to contain one of these words in passing - this is intentionally
# a small, literal set rather than any kind of fuzzy/semantic match, so a
# false positive here would require typing close to these exact phrases.
_SHORT_PHRASES = frozenset({"shorter", "shorter descriptions", "be brief", "briefer"})
_DETAILED_PHRASES = frozenset({"longer", "more detail", "more details", "longer descriptions"})


def _normalise(text):
    """Lowercase, strip surrounding whitespace/punctuation, and drop one
    trailing "please" - the minimum normalisation that lets "Shorter
    descriptions, please!" match the same entry as "shorter descriptions"
    without turning this into a fuzzy matcher. Not reused outside this
    module: it is shaped around _SHORT_PHRASES/_DETAILED_PHRASES only.
    """
    normalised = text.strip().lower().strip(".,!? ")
    if normalised.endswith(" please"):
        normalised = normalised[: -len(" please")].strip(".,!? ")
    return normalised


def detect_preference_command(question):
    """VERBOSITY_SHORT, VERBOSITY_DETAILED, or None if `question` does not
    match the tiny closed vocabulary above. Never raises: a non-string
    `question` (already rejected earlier by clarif_eye.ui before this could
    be reached, but this function's own contract is never-raise regardless)
    is treated as "not a command" rather than let a .strip() crash out.
    """
    if not isinstance(question, str):
        return None
    normalised = _normalise(question)
    if normalised in _SHORT_PHRASES:
        return VERBOSITY_SHORT
    if normalised in _DETAILED_PHRASES:
        return VERBOSITY_DETAILED
    return None


def set_verbosity(store, session_id, verbosity):
    """Write {"verbosity": verbosity} into (NAMESPACE_ROOT, session_id).

    NO-OP, silently, when `store` is None (a store-less graph/AppResources -
    issue #86's own "store optional at compile like checkpointer" rule) or
    `session_id` is falsy (nothing to namespace by - this must never fall
    back to a shared/global key, which would leak one visitor's preference
    to every other). NEVER RAISES beyond that: a real InMemoryStore.put()
    should not fail, but this is a preference, not the deliverable - the
    same never-raise discipline every node in this pipeline already follows
    for its own writes.
    """
    if store is None or not session_id:
        return
    try:
        store.put((NAMESPACE_ROOT, session_id), PREFERENCE_KEY, {PREFERENCE_KEY: verbosity})
    except Exception:
        pass


def get_verbosity(store, session_id):
    """VERBOSITY_SHORT, VERBOSITY_DETAILED, or None (absent, unavailable, or
    not shaped like a preference this module could have written).

    THE SHAPE CHECK IS DEFENSIVE, NOT DECORATIVE: this module is the only
    writer, but a read must still validate rather than trust, the same
    "loud beats silent, but here silence beats a crash" trade every
    never-raise degrade in this codebase makes - a corrupted or
    unexpectedly-shaped entry degrades to "no preference" (describe
    normally) rather than propagate a bad value into a prompt.
    """
    if store is None or not session_id:
        return None
    try:
        item = store.get((NAMESPACE_ROOT, session_id), PREFERENCE_KEY)
    except Exception:
        return None
    if item is None:
        return None
    value = item.value
    if not isinstance(value, dict) or set(value.keys()) != {PREFERENCE_KEY}:
        return None
    verbosity = value.get(PREFERENCE_KEY)
    if verbosity not in (VERBOSITY_SHORT, VERBOSITY_DETAILED):
        return None
    return verbosity


def verbosity_for_config(store, config):
    """Convenience for clarif_eye.graph's nodes: pull session_id out of
    config["configurable"] (written by clarif_eye.ui.thread_configurable)
    and read the preference for it. NEVER RAISES - a malformed/absent
    `config` (every existing test/caller that calls a node function
    directly, with config=None or no session_id at all) degrades to None,
    same as get_verbosity's own "no preference on file" default.
    """
    try:
        session_id = (config or {}).get("configurable", {}).get("session_id")
    except Exception:
        return None
    return get_verbosity(store, session_id)
